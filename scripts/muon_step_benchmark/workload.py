# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Extract VeOmni's actual parameter inventory and replay local optimizer shards."""

import hashlib
import importlib
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import hyper_parallel
import torch
import torch.distributed as dist
import yaml
from hyper_parallel import DeviceMesh
from hyper_parallel import DTensor as HyperDTensor
from hyper_parallel import Replicate as HyperReplicate
from hyper_parallel import Shard as HyperShard
from hyper_parallel.core.optimizer import get_hyper_optimizer
from torch import nn
from torch.distributed.tensor import DTensor as TorchDTensor
from torch.distributed.tensor import Replicate as TorchReplicate
from torch.distributed.tensor import Shard as TorchShard

from scripts.muon_step_benchmark.support import local_shard


def git_metadata(root: Path) -> dict[str, Any]:
    """Record both revision and worktree diff so uncommitted benchmark code is identifiable."""

    def _git(*arguments):
        return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()

    return {
        "root": str(root),
        "commit": _git("rev-parse", "HEAD"),
        "status": _git("status", "--short"),
        "diff": _git("diff", "HEAD"),
    }


def load_veomni(root: Path) -> dict[str, Any]:
    """Load the optional installed VeOmni checkout only when an experiment requests it."""
    # VeOmni is an optional external dependency, absent from ordinary HyperParallel UT environments.
    package = importlib.import_module("veomni")
    actual = Path(package.__file__).resolve()
    if not actual.is_relative_to(root.resolve()):
        raise ValueError(f"VeOmni imports from {actual}, expected editable checkout {root}")
    return {
        "arguments": importlib.import_module("veomni.arguments"),
        "models": importlib.import_module("veomni.models"),
        "optimizer": importlib.import_module("veomni.optim.optimizer"),
        "muon": importlib.import_module("veomni.optim.muon"),
        "parallel": importlib.import_module("veomni.distributed.parallel_state"),
        "parallelize": importlib.import_module("veomni.distributed.torch_parallelize"),
        "adapter": importlib.import_module("veomni.distributed.hyper_fsdp2"),
    }


class ReplayModel(nn.Module):
    """Preserve original Muon parameter names without allocating unrelated training state."""

    def __init__(self, named: list[tuple[str, nn.Parameter]]) -> None:
        """Register local shards while retaining their original FQNs for VeOmni grouping."""
        super().__init__()
        self.weights = nn.ParameterList([param for _, param in named])
        self.original_names = [name for name, _ in named]

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield source names in the original optimizer's ordering."""
        del recurse, remove_duplicate
        for name, param in zip(self.original_names, self.weights):
            yield (f"{prefix}.{name}" if prefix else name), param


def global_stride(param: torch.Tensor) -> list[int]:
    """Read logical metadata without dispatching an unsupported Hyper stride op."""
    if isinstance(param, HyperDTensor):
        stride = param.layout.tensor_stride
        if stride is not None:
            return list(stride)
        if not param.to_local().is_contiguous():
            raise ValueError("Hyper DTensor has no logical stride metadata and a noncontiguous local shard")
        return list(torch.empty(tuple(param.shape), device="meta").stride())
    return list(param.stride())


def _dtensor(local, shape, stride, mesh, placements, backend):
    cls = HyperDTensor if backend == "hyper" else TorchDTensor
    return cls.from_local(local, mesh, placements, run_check=False, shape=tuple(shape), stride=tuple(stride))


def _source_configs(args) -> tuple[dict, Any]:
    name = "qwen3_muon_2k_16.yaml" if args.scenario == "fsdp" else "qwen3_muon_128k_hsdp_16.yaml"
    path = args.veomni_root / "configs" / "hyperparallel" / name
    raw = path.read_text()
    config = yaml.safe_load(raw)
    accelerator = config["model"]["accelerator"]
    expected = (1, 16) if args.scenario == "fsdp" else (2, 8)
    actual = (accelerator["dp_replicate_size"], accelerator["dp_shard_size"])
    if actual != expected or any(
        accelerator.get(key, 1) != 1 for key in ("ep_size", "tp_size", "pp_size", "cp_size", "ulysses_size")
    ):
        raise ValueError(f"Unsupported VeOmni topology: {actual}; expected {expected} with other axes disabled")
    fsdp = accelerator["fsdp_config"]
    if fsdp["fsdp_mode"] != "fsdp2" or fsdp.get("offload", False):
        raise ValueError("This extraction supports FSDP2, no-offload scenarios")
    return {
        "yaml_path": str(path),
        "yaml": raw,
        "config": config,
        "effective_fsdp_backend": args.fsdp_backend,
        "forced_use_hyper_optimizer": True,
    }, config["model"]["optimizer"]


def _make_synthetic(args, api, mesh, device):
    del device
    source, options = _source_configs(args)
    if options["muon_ns_implementation"] != "std" or options.get("muon_expert_zero_comm", False):
        raise ValueError("The source benchmark requires std Newton-Schulz and expert_zero_comm=false")
    options = {**options, "use_hyper_optimizer": True}
    opt_config = api["arguments"].OptimizerConfig(**options)
    parallel = api["parallel"].get_parallel_state()
    mixed = source["config"]["model"]["accelerator"]["fsdp_config"]["mixed_precision"]
    dtype = "float32" if mixed["enable"] else "bfloat16"
    ops = api["arguments"].OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation="eager",
        rms_norm_implementation="eager",
        rotary_pos_emb_implementation="eager",
        cross_entropy_loss_implementation="eager",
        swiglu_mlp_implementation="eager",
        load_balancing_loss_implementation="eager",
    )
    with api["parallel"].use_parallel_state(parallel):
        model = api["models"].build_foundation_model(
            config_path=str(args.model_config),
            weights_path=None,
            init_device="meta",
            torch_dtype=dtype,
            ops_implementation=ops,
        )
    if model.config.model_type not in ("qwen3", "qwen3_moe"):
        raise ValueError(f"Expected Qwen3 or Qwen3 MoE, got {model.config.model_type}")
    # Run the real VeOmni wrap on meta tensors. Only weight loading is replaced;
    # no full checkpoint, unrelated AdamW state or activation is allocated.
    fsdp = source["config"]["model"]["accelerator"]["fsdp_config"]
    mixed_config = api["arguments"].MixedPrecisionConfig(**mixed)
    with (
        api["parallel"].use_parallel_state(parallel),
        patch.object(api["parallelize"], "_materialize_and_load_weights", return_value=None),
    ):
        model = api["parallelize"].build_parallelize_model(
            model,
            weights_path=None,
            mixed_precision=mixed_config,
            enable_gradient_checkpointing=False,
            enable_reshard_after_forward=fsdp["reshard_after_forward"],
            fsdp_backend=args.fsdp_backend,
            init_device="meta",
            enable_reshard_after_backward=fsdp["reshard_after_backward"],
            enable_forward_prefetch=fsdp["forward_prefetch"],
            enable_fsdp_cpu_offload=False,
        )
    params, _, names, _ = api["muon"].split_muon_adamw_params(
        model, no_decay_modules=opt_config.no_decay_modules, no_decay_params=opt_config.no_decay_params
    )
    inventory = []
    for name, param in zip(names, params):
        if not isinstance(param, (HyperDTensor, TorchDTensor)):
            raise ValueError(f"VeOmni did not shard {name}")
        placements = param.placements
        shards = [place for place in placements if place.is_shard()]
        if len(shards) != 1 or shards[0].dim != 0:
            raise ValueError(f"Unsupported placement for {name}: {placements}; use a dedicated replay")
        if param.device_mesh.mesh.tolist() != mesh.mesh.tolist():
            raise ValueError(f"Unexpected rank mapping for {name}")
        inventory.append(
            {
                "name": name,
                "shape": list(param.shape),
                "stride": global_stride(param),
                "dtype": str(param.dtype).removeprefix("torch."),
                "placements": [str(place) for place in placements],
            }
        )
    source.update(
        {
            "model_config": model.config.to_dict(),
            "model_class": type(model).__module__,
            "parameter_dtype": dtype,
            "extraction": "real VeOmni FSDP2 wrap on meta; weight loading bypassed",
            "reduce_dtype": mixed.get("reduce_dtype"),
            "original_use_hyper_optimizer": source["config"]["model"]["optimizer"]["use_hyper_optimizer"],
            "forced_use_hyper_optimizer": True,
            "parallel_mesh": parallel.fsdp_mesh.mesh.tolist(),
            "config_resolved": asdict(opt_config),
        }
    )
    del model, params
    return inventory, opt_config, source


def _materialize(inventory, mesh, device, seed, backend, captured=None):
    replicate, shard = (HyperReplicate, HyperShard) if backend == "hyper" else (TorchReplicate, TorchShard)
    placements = (replicate(), shard(0)) if mesh.ndim == 2 else (shard(0),)
    named = []
    shard_rank = mesh.get_coordinate()[-1]
    for index, entry in enumerate(inventory):
        shape, stride = entry["shape"], entry["stride"]
        shard_shape, _ = local_shard(shape, mesh.shape[-1], shard_rank)
        dtype = getattr(torch, entry["dtype"])
        current_placements = list(placements)
        if backend == "hyper" and shape[0] % mesh.shape[-1]:
            current_placements[-1] = HyperShard(0, uneven_shard=True)
        if captured is None:
            # Replica peers share a seed. Different shards may differ; data is fixed across every ablation.
            generator = torch.Generator(device=device).manual_seed(seed + 1009 * index + shard_rank)
            local = torch.empty(shard_shape, dtype=dtype, device=device).normal_(0, 0.02, generator=generator)
            grad = torch.empty_like(local).normal_(0, 0.01, generator=generator)
        else:
            local = captured[index]["parameter"].to(device)
            grad = captured[index]["gradient"]
            grad = grad.to(device) if grad is not None else None
            if local.dtype != dtype or (grad is not None and (grad.shape != local.shape or grad.dtype != dtype)):
                raise ValueError(f"Captured dtype/gradient mismatch for {entry['name']}")
            if list(local.shape) != shard_shape:
                raise ValueError(f"Captured local shape differs for {entry['name']}: {local.shape} vs {shard_shape}")
        tensor = _dtensor(local, shape, stride, mesh, current_placements, backend)
        param = nn.Parameter(tensor)
        if backend == "hyper":
            # Parameter's detach can drop logical metadata. Preserve it as HP's
            # own fully_shard parameter constructor does, including uneven shapes.
            param._layout = tensor.layout
            param._placements = tuple(tensor.placements)
            param._global_shape = tuple(shape)
        if tuple(param.shape) != tuple(shape):
            raise ValueError(f"Parameter construction lost the global shape for {entry['name']}")
        if grad is not None:
            param.grad = _dtensor(grad, shape, stride, mesh, current_placements, backend)
        named.append((entry["name"], param))
    model = ReplayModel(named)
    model._veomni_fsdp_backend = backend
    return model


def _build_optimizer(model, args, api, opt_config):
    requested = {
        "momentum_mode": args.momentum,
        "hsdp_replicate_dedup": bool(args.dedup),
        "fsdp_overlap": bool(args.overlap),
        "fsdp_overlap_memory_budget_bytes": args.budget_mib * 1024**2,
    }
    if api is None:
        chain = get_hyper_optimizer(
            model,
            list(model.parameters()),
            [],
            muon_kwargs={**requested, "ns_steps": 2, "hsdp_replica_count": 2 if args.scenario == "hsdp" else None},
        )
    else:
        chain = api["optimizer"].build_optimizer(
            model,
            lr=opt_config.lr,
            betas=tuple(opt_config.betas),
            weight_decay=opt_config.weight_decay,
            fused=args.device != "cpu",
            optimizer_type="muon",
            no_decay_modules=opt_config.no_decay_modules,
            no_decay_params=opt_config.no_decay_params,
            optimizer_config=opt_config,
            use_hyper_optimizer=True,
            muon_kwargs=requested,
        )
    leaf = chain.optimizers_dict["muon"]
    effective = {
        "momentum_mode": leaf._update_muon_momentum_mode,  # pylint: disable=protected-access
        "hsdp_replicate_dedup": leaf.hsdp_replicate_dedup,
        "fsdp_overlap": leaf.fsdp_overlap,
        "fsdp_overlap_memory_budget_bytes": leaf.fsdp_overlap_memory_budget_bytes,
    }
    if effective != requested:
        raise ValueError(f"Feature injection was not effective: {effective} vs {requested}")
    if len(leaf.param_groups) != 1:
        raise ValueError("Expected VeOmni's single Muon parameter group")
    if [id(param) for param in model.parameters()] != [id(param) for param in leaf.param_groups[0]["params"]]:
        raise ValueError("VeOmni changed the replay parameter inventory or ordering")
    return chain, leaf, effective


def build_workload(args: Any, mesh: Any, device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    """Construct actual VeOmni Muon shards, or a clearly labelled tiny control workload."""
    api, captured, opt_config = None, None, None
    tiny_control = args.smoke and args.model_config is None and args.snapshot_root is None
    if args.fsdp_backend == "hyper":
        adapter = importlib.import_module("veomni.distributed.hyper_fsdp2") if not tiny_control else None
        if adapter is not None:
            adapter.get_hyper_fsdp2_api()
            mesh = adapter.to_hyper_device_mesh(mesh, "fsdp")
        else:
            groups = mesh.get_all_groups()
            mesh = DeviceMesh.from_group(
                groups[0] if mesh.ndim == 1 else groups,
                args.device,
                mesh=mesh.mesh.tolist(),
                mesh_dim_names=mesh.mesh_dim_names,
            )
    if tiny_control:
        inventory = [
            {"name": f"matrix_{index}", "shape": [32, 64], "stride": [64, 1], "dtype": "float32"}
            for index in range(24)
        ]
        source = {"kind": "tiny_smoke_not_training"}
    else:
        api = load_veomni(args.veomni_root)
        api["parallel"]._init_parallel_state(  # pylint: disable=protected-access
            dp_size=dist.get_world_size(),
            dp_replicate_size=2 if args.scenario == "hsdp" else 1,
            dp_shard_size=mesh.shape[-1],
            device_type=args.device,
            name="muon_benchmark",
        )
        if args.snapshot_root:
            directory = args.snapshot_root / args.scenario
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest["format_version"] != 1 or manifest["state"] != "empty_first_step":
                raise ValueError("Unsupported snapshot format or optimizer state")
            if manifest["world_size"] != dist.get_world_size() or manifest["mesh"] != mesh.mesh.tolist():
                raise ValueError("Snapshot world size or rank-to-mesh mapping differs from this replay")
            expected_placements = ["R", "S(0)"] if mesh.ndim == 2 else ["S(0)"]
            if manifest["placements"] != expected_placements or manifest["fsdp_backend"] != args.fsdp_backend:
                raise ValueError("Snapshot requires the supplied replicated / Shard(0) layout")
            payload = torch.load(directory / f"rank{dist.get_rank():05d}.pt", map_location="cpu", weights_only=True)
            inventory, captured = manifest["inventory"], payload["parameters"]
            if len(inventory) != len(captured):
                raise ValueError("Capture tensor count differs from manifest")
            source, options = _source_configs(args)
            opt_config = api["arguments"].OptimizerConfig(**{**options, "use_hyper_optimizer": True})
            source.update({"kind": "captured_first_step", "capture": manifest, "snapshot_dir": str(directory)})
            expected = {
                "ns_steps": opt_config.muon_ns_steps,
                "momentum": opt_config.muon_momentum,
                "nesterov": opt_config.muon_nesterov,
                "ns_epsilon": opt_config.muon_eps,
                "ns_variant": "custom",
                "weight_decay": opt_config.muon_weight_decay,
                "lr": opt_config.muon_lr if opt_config.muon_lr is not None else opt_config.lr,
                "ns_coefficients": [opt_config.muon_ns_coefficients] * opt_config.muon_ns_steps,
            }
            expected["momentum"] = [opt_config.muon_momentum] * 2
            for key, value in expected.items():
                if manifest["optimizer_group"].get(key) != value:
                    raise ValueError(f"Snapshot optimizer {key} differs from source YAML")
        else:
            inventory, opt_config, source = _make_synthetic(args, api, mesh, device)
            source["kind"] = "veomni_meta_wrapped_inventory_fixed_synthetic_gradients"
        source["veomni"] = git_metadata(args.veomni_root)
    model = _materialize(inventory, mesh, device, args.seed, args.fsdp_backend, captured)
    chain, leaf, effective = _build_optimizer(model, args, api, opt_config)
    source.update(
        {
            "inventory": inventory,
            "inventory_sha256": hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
            "effective": effective,
            "optimizer_groups": [
                {key: value for key, value in group.items() if key != "params"} for group in leaf.param_groups
            ],
            "dtensor_type": type(leaf.param_groups[0]["params"][0]).__module__,
            "smoke": args.smoke,
            "world_size": dist.get_world_size(),
            "global_muon_numel": sum(math.prod(entry["shape"]) for entry in inventory),
            "hyper_parallel": git_metadata(Path(hyper_parallel.__file__).resolve().parent.parent),
            "hyper_parallel_import": hyper_parallel.__file__,
            "scope": "leaf Muon.step only; fixed post-reduction/post-clip gradients; no AdamW, fwd, bwd or clipping",
        }
    )
    return chain, leaf, source
