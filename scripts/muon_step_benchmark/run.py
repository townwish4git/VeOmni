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
"""Launch isolated VeOmni Muon ablations without importing training frameworks."""

import argparse
import csv
import json
import logging
import math
import os
import platform
import random
import shlex
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from scripts.muon_step_benchmark.support import Experiment, experiments


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = "scripts.muon_step_benchmark"


def parser() -> argparse.ArgumentParser:
    """Expose one suite command for local torchrun, per-node launch, or SSH fan-out."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--veomni-root", type=Path, default=REPO_ROOT)
    result.add_argument("--model-config", type=Path, help="Local HF model directory or config.json; no weights read")
    result.add_argument("--snapshot-root", type=Path, help="Root containing fsdp/ and hsdp/ first-step captures")
    result.add_argument("--output", type=Path, default=REPO_ROOT.parent / "muon_step_results")
    result.add_argument("--run-id")
    result.add_argument("--scenario", choices=("all", "fsdp", "hsdp"), default="all")
    result.add_argument("--cases", nargs="+", help="Subset of case names printed by --dry-run")
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--nproc-per-node", type=int, default=8)
    result.add_argument("--nnodes", type=int, default=1)
    result.add_argument("--node-rank", type=int, default=0)
    result.add_argument("--master-addr", default="127.0.0.1")
    result.add_argument("--master-port", type=int, default=29540)
    result.add_argument(
        "--hccl-base-port",
        type=int,
        default=40000,
        help="Reserve 256 HCCL ports per case, independently of the rendezvous port",
    )
    result.add_argument("--hosts", help="SSH coordinator: comma-separated node hostnames, in rank order")
    result.add_argument("--python", default=sys.executable, help="Interpreter on every worker node")
    result.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Same VeOmni checkout path on each node")
    result.add_argument("--fsdp-backend", choices=("hyper", "torch"), default="hyper")
    result.add_argument("--hyper-root", type=Path, required=False, help="Expected editable HyperParallel checkout")
    result.add_argument("--device", choices=("npu", "cuda", "cpu"), default="npu")
    result.add_argument("--warmup-min", type=int, default=20)
    result.add_argument("--warmup-max", type=int, default=200)
    result.add_argument("--stability-window", type=int, default=10)
    result.add_argument("--stable-windows", type=int, default=3)
    result.add_argument("--stability-cv", type=float, default=0.05)
    result.add_argument("--stability-drift", type=float, default=0.03)
    result.add_argument("--measure-steps", type=int, default=50)
    result.add_argument("--profile-steps", type=int, default=5)
    result.add_argument("--profile-warmup", type=int, default=2)
    result.add_argument("--profile-ranks", default="all", help="all or comma-separated global ranks")
    result.add_argument("--profile-memory", action="store_true")
    result.add_argument("--record-shapes", action="store_true")
    result.add_argument("--with-stack", action="store_true")
    result.add_argument(
        "--budget-mib", type=int, default=512, help="Identical temporary budget for serial and overlap"
    )
    result.add_argument("--timeout-seconds", type=int, default=1800)
    result.add_argument(
        "--smoke",
        action="store_true",
        help="Small-world check; uses 24 matrices, or real VeOmni wrap when model-config is supplied",
    )
    result.add_argument(
        "--dry-run", action="store_true", help="Print the experiment matrix and exact torchrun commands"
    )
    result.add_argument("--summarize-only", action="store_true")
    return result


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid windows and topologies before distributed workers start."""
    positive = (
        "repeats",
        "nproc_per_node",
        "nnodes",
        "warmup_min",
        "warmup_max",
        "stable_windows",
        "measure_steps",
        "budget_mib",
        "timeout_seconds",
    )
    if any(getattr(args, name) < 1 for name in positive):
        raise ValueError(f"These options must be positive: {positive}")
    if not math.isfinite(args.stability_cv) or not math.isfinite(args.stability_drift):
        raise ValueError("Stability thresholds must be finite")
    if args.stability_window < 2 or args.warmup_max < max(
        args.warmup_min, (args.stable_windows + 1) * args.stability_window
    ):
        raise ValueError("Warmup max must cover warmup min and two stability windows; window must be >= 2")
    if args.model_config and args.snapshot_root:
        raise ValueError("Choose model-config or snapshot-root, not both")
    if args.profile_steps < 0 or args.profile_warmup < 0 or args.stability_cv <= 0 or args.stability_drift <= 0:
        raise ValueError("Invalid profile length or stability threshold")
    if args.profile_ranks != "all":
        try:
            ranks = [int(value) for value in args.profile_ranks.split(",")]
        except ValueError as error:
            raise ValueError("profile-ranks must be all or comma-separated global rank integers") from error
        if any(rank < 0 or rank >= args.nnodes * args.nproc_per_node for rank in ranks):
            raise ValueError("profile-ranks contains a rank outside this job")
    if not 0 <= args.node_rank < args.nnodes:
        raise ValueError("node-rank must be in [0, nnodes)")
    if not args.smoke and args.nnodes * args.nproc_per_node != 16:
        raise ValueError("The supplied training scenarios require exactly 16 ranks; use --smoke for a tiny check")
    if args.scenario in ("all", "hsdp") and args.nnodes * args.nproc_per_node % 2:
        raise ValueError("HSDP requires an even world size")
    if args.device == "cpu" and not args.smoke:
        raise ValueError("CPU is supported only for --smoke; accelerator timing requires NPU or CUDA")
    if not args.smoke and (args.veomni_root is None or (args.model_config is None and args.snapshot_root is None)):
        raise ValueError("Provide --veomni-root and either --model-config or --snapshot-root")
    if not args.dry_run and args.nnodes > 1 and not args.run_id:
        raise ValueError("Multi-node runs require the same explicit --run-id on every node")


def job_command(args: argparse.Namespace, case: Experiment, trial: int, index: int) -> list[str]:
    """Build a fresh distributed process group for each experiment and repeat."""
    command = [
        args.python,
        "-m",
        "torch.distributed.run",
        f"--nnodes={args.nnodes}",
        f"--nproc-per-node={args.nproc_per_node}",
        f"--node-rank={args.node_rank}",
        f"--master-addr={args.master_addr}",
        f"--master-port={args.master_port + index}",
        "--max-restarts=0",
        "--log-dir",
        str(args.output / args.run_id / f"r{trial:02d}_{case.name}" / "torchrun"),
        "--tee=3",
        "-m",
        f"{MODULE}._worker",
        "--case",
        case.name,
        "--scenario",
        case.scenario,
        "--momentum",
        case.momentum,
        "--dedup",
        str(int(case.dedup)),
        "--overlap",
        str(int(case.overlap)),
        "--trial",
        str(trial),
        "--output",
        str(args.output / args.run_id / f"r{trial:02d}_{case.name}"),
    ]
    for name in (
        "veomni_root",
        "hyper_root",
        "model_config",
        "snapshot_root",
        "device",
        "fsdp_backend",
        "seed",
        "warmup_min",
        "warmup_max",
        "stability_window",
        "stable_windows",
        "stability_cv",
        "stability_drift",
        "measure_steps",
        "profile_steps",
        "profile_warmup",
        "profile_ranks",
        "budget_mib",
        "timeout_seconds",
    ):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    for name in ("smoke", "profile_memory", "record_shapes", "with_stack"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    return command


def make_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Randomize case order deterministically within each repeat, identically on every node."""
    cases = experiments(args.scenario)
    if args.cases:
        unknown = set(args.cases) - {case.name for case in cases}
        if unknown:
            raise ValueError(f"Unknown case names: {sorted(unknown)}")
        cases = [case for case in cases if case.name in args.cases]
    plan = []
    for trial in range(args.repeats):
        ordered = list(cases)
        random.Random(args.seed + trial).shuffle(ordered)
        for case in ordered:
            plan.append(
                {
                    "trial": trial,
                    "experiment": case.to_dict(),
                    "command": job_command(args, case, trial, len(plan)),
                    "environment": {"HCCL_IF_BASE_PORT": str(args.hccl_base_port + 256 * len(plan))}
                    if args.device == "npu"
                    else {},
                }
            )
    if args.master_port < 1024 or args.master_port + len(plan) > 65535:
        raise ValueError("Experiment rendezvous ports must stay in [1024, 65535]")
    if args.device == "npu" and (args.hccl_base_port < 1024 or args.hccl_base_port + 256 * len(plan) > 65535):
        raise ValueError("HCCL base port plus 256 ports per experiment must stay in [1024, 65535]")
    return plan


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


def launch_hosts(args: argparse.Namespace) -> None:
    """Fan out the same suite over SSH; each remote launcher owns its local torchrun children."""
    hosts = args.hosts.split(",")
    if len(hosts) < 2 or any(not host.strip() for host in hosts):
        raise ValueError("--hosts requires at least two nonempty SSH hostnames")
    forwarded, original = [], list(sys.argv[1:])
    removed = {"--hosts", "--nnodes", "--node-rank", "--run-id", "--master-addr"}
    while original:
        token = original.pop(0)
        if token in removed:
            original.pop(0)
        elif token.split("=", 1)[0] not in removed:
            forwarded.append(token)
    children = []
    try:
        for rank, host in enumerate(hosts):
            command = [
                args.python,
                "-m",
                f"{MODULE}.run",
                *forwarded,
                "--nnodes",
                str(len(hosts)),
                "--node-rank",
                str(rank),
                "--master-addr",
                args.master_addr,
                "--run-id",
                args.run_id,
            ]
            remote = f"cd {shlex.quote(str(args.repo_root))} && exec {shlex.join(command)}"
            children.append(subprocess.Popen(["ssh", "-o", "BatchMode=yes", host.strip(), remote]))
        while any(child.poll() is None for child in children):
            failed = [child.returncode for child in children if child.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"Remote suite failed with exit codes {failed}; collect partial artifacts")
            time.sleep(0.5)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        # Profiles live on the rank's node unless the experiment directory is shared.
        directory = args.output / args.run_id
        directory.mkdir(parents=True, exist_ok=True)
        retrieval_failures = []
        for host in hosts:
            remote_path = f"{host.strip()}:{shlex.quote(str(directory))}/"
            result = subprocess.run(
                ["rsync", "-a", "--ignore-existing", remote_path, str(directory) + "/"], check=False
            )
            if result.returncode:
                retrieval_failures.append(host)
                LOGGER.error("Artifact retrieval failed for %s; data remains at %s", host, directory)
    if list(directory.glob("r*/result.json")):
        summarize(directory)
    if retrieval_failures:
        raise RuntimeError(f"Artifact retrieval incomplete from {retrieval_failures}")


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


def execute_job(command: list[str], root: Path, overrides: dict[str, str]) -> None:
    """Let torchrun terminate its workers when the suite is interrupted."""
    with subprocess.Popen(command, cwd=root, env={**os.environ, **overrides}) as child:
        try:
            code = child.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()


def _stop_suite(signum, _frame):
    raise KeyboardInterrupt(f"Suite received signal {signum}")


def main() -> None:
    """Execute the requested factorial suite and preserve a reviewable launch manifest."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGTERM, _stop_suite)
    signal.signal(signal.SIGHUP, _stop_suite)
    args = parser().parse_args()
    for name in ("veomni_root", "repo_root", "hyper_root", "output", "model_config", "snapshot_root"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.expanduser().resolve())
    if args.hyper_root is not None and not args.dry_run and not args.summarize_only and not args.hosts:
        # Check inside the worker too; the launcher deliberately never imports Torch.
        if not (args.hyper_root / "hyper_parallel").is_dir():
            raise ValueError(f"Invalid HyperParallel checkout: {args.hyper_root}")
    if args.summarize_only:
        if not args.run_id:
            raise ValueError("--summarize-only requires --run-id")
        summarize(args.output / args.run_id)
        return
    if args.hosts:
        args.nnodes = len(args.hosts.split(","))
        if args.master_addr == "127.0.0.1":
            raise ValueError("SSH fan-out requires --master-addr reachable from every node")
    validate_args(args)
    args.run_id = args.run_id or ("preview" if args.dry_run else datetime.now().strftime("%Y%m%d-%H%M%S"))
    plan = make_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.hosts:
        launch_hosts(args)
        return
    directory = args.output / args.run_id
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"plan-node{args.node_rank}.json").open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
    environment_snapshot(args, directory)
    for job in plan:
        LOGGER.info("Launching repeat %d: %s", job["trial"], job["experiment"]["name"])
        execute_job(job["command"], args.repo_root, job["environment"])
    if args.node_rank == 0:
        summarize(directory)


if __name__ == "__main__":
    main()
