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
"""Summarize completed Muon experiments; never launch training processes."""

import argparse
import csv
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def summarize(directory: Path) -> None:
    """Write per-repeat CSV with same-repeat baseline speedups and effective pipeline counts."""
    results = [json.loads(path.read_text()) for path in sorted(directory.glob("r*/result.json"))]
    baselines = {
        (row["scenario"], row["trial"]): row
        for row in results
        if row["dedup"] and not row["overlap"] and row["momentum"] == "legacy"
    }
    rows = []
    for result in results:
        baseline = baselines.get((result["scenario"], result["trial"]))
        timing = result["timing"]["slowest_rank_per_step"]
        baseline_mean = baseline["timing"]["slowest_rank_per_step"]["mean_ms"] if baseline else None
        rows.append(
            {
                "case": result["case"],
                "trial": result["trial"],
                "scenario": result["scenario"],
                "dedup": result["dedup"],
                "overlap_requested": result["overlap"],
                "momentum": result["momentum"],
                "overlap_status": result.get("overlap_status", "unknown"),
                **timing,
                "speedup_vs_baseline": baseline_mean / timing["mean_ms"] if baseline else "",
                "peak_allocated_bytes": max(rank["peak_allocated_bytes"] for rank in result["ranks"]),
                "pipeline_buckets_preflight": sum(rank["pipeline_buckets"] for rank in result["ranks"]),
                "source": result["source"],
                "fsdp_backend": result["environment"]["fsdp_backend"],
                "smoke": result.get("smoke", False),
            }
        )
    if not rows:
        raise ValueError(f"No completed results under {directory}")
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    by_factors = {
        (item["scenario"], item["trial"], item["dedup"], item["overlap"], item["momentum"]): item for item in results
    }
    effects = []
    for before in results:
        for factor, old, new in (("dedup", True, False), ("overlap", False, True), ("momentum", "legacy", "vanilla")):
            if before[factor] != old or (factor == "dedup" and before["scenario"] == "fsdp"):
                continue
            changed = {**before, factor: new}
            after = by_factors.get(
                tuple(changed[key] for key in ("scenario", "trial", "dedup", "overlap", "momentum"))
            )
            if after is None:
                continue
            before_ms = before["timing"]["slowest_rank_per_step"]["mean_ms"]
            after_ms = after["timing"]["slowest_rank_per_step"]["mean_ms"]
            effects.append(
                {
                    "scenario": before["scenario"],
                    "trial": before["trial"],
                    "factor": factor,
                    "before": before["case"],
                    "after": after["case"],
                    "before_mean_ms": before_ms,
                    "after_mean_ms": after_ms,
                    "speedup": before_ms / after_ms,
                    "latency_reduction_pct": 100 * (1 - after_ms / before_ms),
                    "before_overlap_status": before.get("overlap_status"),
                    "after_overlap_status": after.get("overlap_status"),
                }
            )
    if effects:
        with (directory / "ablations.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(effects[0]))
            writer.writeheader()
            writer.writerows(effects)
    LOGGER.info("Saved %d completed experiment rows to %s", len(rows), directory / "summary.csv")


def environment_snapshot(args: argparse.Namespace, directory: Path) -> None:
    """Record analysis context without importing any accelerator framework."""
    packages = {}
    for name in ("torch", "torch-npu", "transformers", "numpy", "veomni", "hyper-parallel", "triton"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    keys = (
        "ASCEND_RT_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "ASCEND_HOME_PATH",
        "ASCEND_OPP_PATH",
        "HCCL_IF_BASE_PORT",
        "HCCL_CONNECT_TIMEOUT",
        "HCCL_EXEC_TIMEOUT",
        "HCCL_SOCKET_IFNAME",
        "HCCL_OP_EXPANSION_MODE",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "NCCL_ALGO",
        "PYTORCH_NPU_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
        "CUDA_LAUNCH_BLOCKING",
        "ASCEND_LAUNCH_BLOCKING",
    )
    result = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "packages": packages,
        "environment": {key: os.environ.get(key) for key in keys},
    }
    if args.device != "cpu":
        command = ["npu-smi", "info"] if args.device == "npu" else ["nvidia-smi"]
        try:
            probe = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
            result["device_probe"] = {
                "command": command,
                "returncode": probe.returncode,
                "stdout": probe.stdout,
                "stderr": probe.stderr,
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            result["device_probe"] = {"command": command, "error": str(error)}
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")) / "version.cfg"
    if args.device == "npu" and cann.is_file():
        result["cann_version_cfg"] = cann.read_text()
    (directory / f"environment-node{args.node_rank}.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    """Build analysis tables from durable worker results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    summarize(args.directory)


if __name__ == "__main__":
    main()
