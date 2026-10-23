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
"""Torchrun worker: stable warmup, unprofiled wall timing, then a separate trace window."""

import argparse
import importlib
import json
import logging
import os
import platform
import socket
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import hyper_parallel
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from scripts.muon_step_benchmark.run import parser as suite_parser
from scripts.muon_step_benchmark.support import stable_window, timing_stats
from scripts.muon_step_benchmark.workload import build_workload


LOGGER = logging.getLogger(__name__)


def _arguments():
    parser = argparse.ArgumentParser(parents=[suite_parser()], add_help=False)
    parser.add_argument("--case", required=True)
    parser.add_argument("--momentum", choices=("legacy", "vanilla"), required=True)
    parser.add_argument("--dedup", type=int, choices=(0, 1), required=True)
    parser.add_argument("--overlap", type=int, choices=(0, 1), required=True)
    parser.add_argument("--trial", type=int, required=True)
    return parser.parse_args()


class DeviceRuntime:
    """Use native Torch device synchronization and the appropriate profiler."""

    def __init__(self, device_type: str) -> None:
        """Initialize the local accelerator before creating HCCL/NCCL process groups."""
        self.kind = device_type
        self.npu_module = importlib.import_module("torch_npu") if device_type == "npu" else None
        self.backend = getattr(torch, device_type) if device_type != "cpu" else None
        if self.backend is not None:
            self.backend.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.device(device_type, int(os.environ["LOCAL_RANK"])) if self.backend else torch.device("cpu")

    def synchronize(self) -> None:
        """Include completion of every optimizer stream in end-to-end step latency."""
        if self.backend is not None:
            self.backend.synchronize()

    def memory(self) -> dict[str, int]:
        """Return peak allocations and reservations for this rank's measurement window."""
        if self.backend is None:
            return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
        return {
            "peak_allocated_bytes": self.backend.max_memory_allocated(),
            "peak_reserved_bytes": self.backend.max_memory_reserved(),
        }


def _timed_step(optimizer, runtime):
    start = time.perf_counter_ns()
    optimizer.step()
    runtime.synchronize()
    return (time.perf_counter_ns() - start) / 1e6


def _warmup(optimizer, runtime, args):
    samples, stable = [], 0
    for step in range(args.warmup_max):
        elapsed = _timed_step(optimizer, runtime)
        sample = torch.tensor(elapsed, dtype=torch.float32, device=runtime.device)
        # Warmup decisions must be identical on every rank; this reduction is outside the timed step.
        dist.all_reduce(sample, op=dist.ReduceOp.MAX)
        samples.append(sample.item())
        if (step + 1) % args.stability_window:
            continue
        ready = step + 1 >= args.warmup_min and stable_window(
            samples, args.stability_window, args.stability_cv, args.stability_drift
        )
        stable = stable + 1 if ready else 0
        if dist.get_rank() == 0:
            LOGGER.info(
                "Warmup %d: mean %.3f ms, stable windows %d/%d",
                step + 1,
                timing_stats(samples[-args.stability_window :])["mean_ms"],
                stable,
                args.stable_windows,
            )
        if stable >= args.stable_windows:
            break
    result = {
        "steps": len(samples),
        "stable": stable >= args.stable_windows,
        "slowest_rank_samples_ms": samples,
        "window": args.stability_window,
        "required_stable_windows": args.stable_windows,
        "max_cv": args.stability_cv,
        "max_median_drift": args.stability_drift,
    }
    if dist.get_rank() == 0:
        (args.output / "warmup.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["stable"]:
        raise RuntimeError("Warmup did not stabilize; see warmup.json. No performance result will be reported.")
    return result


def _measure(optimizer, runtime, args):
    runtime.synchronize()
    dist.barrier()
    if runtime.backend is not None:
        runtime.backend.reset_peak_memory_stats()
    start = time.perf_counter_ns()
    samples = []
    for _ in range(args.measure_steps):
        # Align step starts. The barrier is outside each optimizer latency sample.
        dist.barrier()
        runtime.synchronize()
        samples.append(_timed_step(optimizer, runtime))
    window_ms = (time.perf_counter_ns() - start) / 1e6
    return {
        "rank": dist.get_rank(),
        "host": socket.gethostname(),
        "samples_ms": samples,
        "window_ms": window_ms,
        "stats": timing_stats(samples),
        **runtime.memory(),
    }


def _profile(optimizer, runtime, args):
    if not args.profile_steps:
        return {"enabled": False}
    ranks = (
        list(range(dist.get_world_size()))
        if args.profile_ranks == "all"
        else [int(value) for value in args.profile_ranks.split(",")]
    )
    if any(rank < 0 or rank >= dist.get_world_size() for rank in ranks):
        raise ValueError(f"Invalid profile ranks: {ranks}")
    directory = args.output / "profiles" / f"rank{dist.get_rank():05d}"
    selected = dist.get_rank() in ranks
    context = nullcontext()
    if selected:
        directory.mkdir(parents=True, exist_ok=True)
        module = runtime.npu_module.profiler if runtime.npu_module else torch.profiler
        activities = [module.ProfilerActivity.CPU]
        options = {}
        if runtime.kind == "npu":
            activities.append(module.ProfilerActivity.NPU)
            _handler = module.tensorboard_trace_handler(str(directory))
            options["experimental_config"] = module._ExperimentalConfig(  # pylint: disable=protected-access
                profiler_level=module.ProfilerLevel.Level1, data_simplification=False
            )
        else:
            if runtime.kind == "cuda":
                activities.append(module.ProfilerActivity.CUDA)

            def _handler(profiler):
                profiler.export_chrome_trace(str(directory / "trace.json"))

        context = module.profile(
            activities=activities,
            schedule=module.schedule(wait=0, warmup=args.profile_warmup, active=args.profile_steps, repeat=1),
            on_trace_ready=_handler,
            record_shapes=args.record_shapes,
            profile_memory=args.profile_memory,
            with_stack=args.with_stack,
            **options,
        )
    runtime.synchronize()
    dist.barrier()
    with context as profiler:
        for _ in range(args.profile_warmup + args.profile_steps):
            with torch.profiler.record_function("veomni_benchmark.muon_step"):
                optimizer.step()
                runtime.synchronize()
            if selected:
                profiler.step()
    runtime.synchronize()
    dist.barrier()
    return {
        "enabled": True,
        "selected": selected,
        "directory": str(directory) if selected else None,
        "active_steps": args.profile_steps,
        "warmup_steps": args.profile_warmup,
    }


def _preflight(optimizer, runtime):
    # Count actual bucket execution before timing; the measured path has no instrumentation monkeypatch.
    with patch.object(optimizer, "_enqueue_pipeline_ns", wraps=optimizer._enqueue_pipeline_ns) as enqueue:
        optimizer.step()
        runtime.synchronize()
        count = enqueue.call_count
    assignments = [assignment for group in optimizer.shard_assignments_by_group.values() for assignment in group]
    plans = []
    for assignment in assignments:
        if not optimizer._can_overlap_shard(assignment):
            continue
        size = dist.get_world_size(assignment.shard_pgs[0])
        batches = optimizer._shard_pipeline_batches(assignment.owned_params, size, use_overlap_budget=True)
        plans.append(
            {
                "bucket_count": len(batches),
                "bucket_param_counts": [len(batch) for batch in batches],
                "prepared_bytes": optimizer._pipeline_prepared_bytes(assignment.owned_params),
                "peak_bucket_bytes": max(optimizer._pipeline_bucket_bytes(batch, size) for batch in batches),
            }
        )
    return {
        "pipeline_buckets": count,
        "pipeline_plans": plans,
        "eligible_assignments": sum(optimizer._can_overlap_shard(item) for item in assignments),
        "shard_assignments": len(assignments),
        "preflight_steps": 1,
    }


def _run(args, runtime):
    if args.hyper_root is not None:
        actual = Path(hyper_parallel.__file__).resolve()
        if not actual.is_relative_to(args.hyper_root.resolve()):
            raise ValueError(f"HyperParallel imports from {actual}, expected editable checkout {args.hyper_root}")
    replicate = 2 if args.scenario == "hsdp" else 1
    world = dist.get_world_size()
    if world % replicate or (not args.smoke and world != 16):
        raise ValueError(f"Invalid world size {world} for {args.scenario}")
    shape = (replicate, world // replicate) if replicate > 1 else (world,)
    names = ("dp_replicate", "dp_shard_sp") if replicate > 1 else ("dp_shard_sp",)
    mesh = init_device_mesh(args.device, shape, mesh_dim_names=names)
    chain, optimizer, source = build_workload(args, mesh, runtime.device)
    del chain
    if dist.get_rank() == 0:
        (args.output / "workload.json").write_text(json.dumps(source, indent=2, default=str) + "\n")
    preflight = _preflight(optimizer, runtime)
    warmup = _warmup(optimizer, runtime, args)
    rank = {**_measure(optimizer, runtime, args), **preflight}
    # Timing artifacts are durable before the profiler is created or writes any trace.
    (args.output / f"timing-rank{dist.get_rank():05d}.json").write_text(json.dumps(rank, indent=2) + "\n")
    rank["profile"] = _profile(optimizer, runtime, args)
    ranks = [None] * world
    dist.all_gather_object(ranks, rank)
    if dist.get_rank() == 0:
        critical = [max(values) for values in zip(*(item["samples_ms"] for item in ranks))]
        pipeline = sum(item["pipeline_buckets"] for item in ranks)
        effective = "executed" if pipeline else ("fell_back" if args.overlap else "disabled")
        if args.overlap and not pipeline:
            LOGGER.warning("Overlap requested but no pipeline buckets ran; inspect preflight diagnostics")
        if args.scenario == "hsdp" and args.dedup and args.overlap and not pipeline:
            effective = "expected_fallback_replicate_dedup"
        result = {
            "case": args.case,
            "scenario": args.scenario,
            "trial": args.trial,
            "smoke": args.smoke,
            "dedup": bool(args.dedup),
            "overlap": bool(args.overlap),
            "momentum": args.momentum,
            "overlap_status": effective,
            "world_size": world,
            "mesh": mesh.mesh.tolist(),
            "source": source["kind"],
            "inventory_sha256": source["inventory_sha256"],
            "warmup": warmup,
            "timing": {
                "slowest_rank_per_step": timing_stats(critical),
                "slowest_rank_samples_ms": critical,
                "max_rank_window_ms": max(item["window_ms"] for item in ranks),
            },
            "ranks": ranks,
            "environment": {
                "torch": torch.__version__,
                "python": platform.python_version(),
                "device": args.device,
                "fsdp_backend": args.fsdp_backend,
                "device_name": runtime.backend.get_device_name() if runtime.backend else "cpu",
                "hostname": socket.gethostname(),
                "device_visibility": {
                    key: os.environ.get(key)
                    for key in ("ASCEND_RT_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HCCL_IF_BASE_PORT")
                },
                "torch_npu": getattr(runtime.npu_module, "__version__", None),
            },
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        LOGGER.info(
            "%s: %.3f ms, overlap=%s", args.case, result["timing"]["slowest_rank_per_step"]["mean_ms"], effective
        )


def main() -> None:
    """Initialize one isolated experiment; fail without publishing partial performance claims."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _arguments()
    runtime = DeviceRuntime(args.device)
    backend = {"npu": "hccl", "cuda": "nccl", "cpu": "gloo"}[args.device]
    dist.init_process_group(backend, timeout=timedelta(seconds=args.timeout_seconds))
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        _run(args, runtime)
    except Exception as error:
        (args.output / f"failure-rank{dist.get_rank():05d}.json").write_text(
            json.dumps({"error": repr(error), "case": args.case}, indent=2) + "\n"
        )
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
