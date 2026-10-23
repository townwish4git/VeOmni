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
"""Capture post-reduction/post-clip Muon inputs from a real VeOmni first training step."""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any

import hyper_parallel.core.optimizer as optimizer_api
import torch
import torch.distributed as dist
from hyper_parallel import DTensor as HyperDTensor
from torch.distributed.tensor import DTensor as TorchDTensor

from scripts.muon_step_benchmark.workload import git_metadata, global_stride


def capture_first_step(optimizer: Any, directory: Path) -> None:
    """Save exact local weights and gradients before the first Muon update, on every rank.

    This captures only first-step inputs: optimizer momentum must still be empty.
    Call through the pre-step hook after VeOmni's backward reduction and clipping.
    """
    if optimizer.state:
        raise ValueError("Only first-step capture with empty optimizer state is supported")
    if len(optimizer.param_groups) != 1:
        raise ValueError("Expected VeOmni's single Muon parameter group")
    params = optimizer.param_groups[0]["params"]
    if not params or not all(isinstance(param, (TorchDTensor, HyperDTensor)) for param in params):
        raise ValueError("Capture requires the FSDP2 DTensor backend")
    mesh = params[0].device_mesh
    shard_placements = [place for place in params[0].placements if place.is_shard()]
    if len(shard_placements) != 1 or shard_placements[0].dim != 0:
        raise ValueError("The supplied scenarios require one Shard(0) dimension")
    directory.mkdir(parents=True, exist_ok=True)
    payload, inventory = [], []
    for param in params:
        name = getattr(param, "model_name", None)
        if name is None or param.device_mesh != mesh or param.placements != params[0].placements:
            raise ValueError("Capture requires named parameters with one common FSDP2 layout")
        inventory.append(
            {
                "name": name,
                "shape": list(param.shape),
                "stride": global_stride(param),
                "dtype": str(param.dtype).removeprefix("torch."),
            }
        )
        payload.append(
            {
                "parameter": param.to_local().detach().to(device="cpu", copy=True),
                "gradient": None if param.grad is None else param.grad.to_local().detach().to(device="cpu", copy=True),
            }
        )
    path = directory / f"rank{dist.get_rank():05d}.pt"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite capture {path}")
    torch.save({"parameters": payload}, path)
    dist.barrier()
    if dist.get_rank() == 0:
        group = {key: value for key, value in optimizer.param_groups[0].items() if key != "params"}
        manifest = {
            "format_version": 1,
            "fsdp_backend": "hyper" if isinstance(params[0], HyperDTensor) else "torch",
            "world_size": dist.get_world_size(),
            "mesh": mesh.mesh.tolist(),
            "mesh_dim_names": mesh.mesh_dim_names,
            "placements": [str(place) for place in params[0].placements],
            "inventory": inventory,
            "optimizer_group": group,
            "state": "empty_first_step",
            "hyper_parallel": git_metadata(Path(optimizer_api.__file__).resolve().parents[3]),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def install_capture(directory: Path) -> None:
    """Attach a one-shot pre-step hook at the same factory boundary VeOmni uses."""
    original = optimizer_api.get_hyper_optimizer

    def _factory(*args, **kwargs):
        optimizer = original(*args, **kwargs)
        muon = optimizer.optimizers_dict.get("muon")
        if muon is None:
            raise ValueError("The real training task did not construct HyperParallel Muon")

        def _before_step(leaf, _args, _kwargs):
            capture_first_step(leaf, directory)
            hook.remove()

        hook = muon.register_step_pre_hook(_before_step)
        return optimizer

    optimizer_api.get_hyper_optimizer = _factory


def main() -> None:
    """Run an existing VeOmni task with capture enabled and its normal CLI unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Capture directory for one topology, e.g. snapshots/fsdp"
    )
    parser.add_argument("--task", type=Path, required=True, help="Existing VeOmni training task Python file")
    parser.add_argument("task_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    install_capture(args.output)
    forwarded = args.task_args[1:] if args.task_args[:1] == ["--"] else args.task_args
    sys.argv = [str(args.task), *forwarded]
    runpy.run_path(str(args.task), run_name="__main__")
    if not (args.output / f"rank{int(os.environ.get('RANK', '0')):05d}.pt").exists():
        raise RuntimeError("No capture produced: ensure use_hyper_optimizer=true and complete the first training step")


if __name__ == "__main__":
    main()
