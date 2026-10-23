# VeOmni dummy training / HyperParallel Muon ST

The primary entry is the repository-root **`script.sh`**. It starts exactly
one experiment on the current node and directly executes torchrun:

```text
script.sh --hsdp_dedup=0 --vanilla_momentum=1
  └─ exec torchrun ... tasks/train_muon_step.py <default-training.yaml>
```

## Two nodes, eight devices each: one experiment

Copy `VeOmni/script.sh` beside `VeOmni/` and `hyper-parallel/`, or run it inside
the VeOmni checkout. Set these variables on **each node**:

```bash
export MASTER_ADDR=10.0.0.10        # first node communication IP
export MODEL_CONFIG=/models/Qwen3-30B-A3B
export NODE_RANK=0                  # set 1 on the second node
bash script.sh --hsdp_dedup=0 --vanilla_momentum=1
```

Defaults are `NNODES=2`, `NPROC_PER_NODE=8`, `MASTER_PORT=29540` and
`SCENARIO=hsdp`, `DEVICE=cuda`. The first node enters torchrun and waits for the other node.
Run the same flags on both nodes; only NODE_RANK differs. There is no remote
launch or automatic artifact copying. Uppercase NODE_RANK/MODEL_CONFIG are
preferred; the lowercase `node_rank` / `model_config` aliases also work.

For two nodes using only the back four GPUs on each eight-GPU host, set
`NPROC_PER_NODE=4`. Both launchers set `CUDA_VISIBLE_DEVICES=4,5,6,7` for this
CUDA configuration, including replacing a preexisting visibility list. Total
world size becomes 8; HSDP uses replicate 2 × shard 4, FSDP uses shard 8.
The 8-rank YAMLs are selected automatically. At 8 workers per node the existing
CUDA visibility is preserved, and the total world size remains 16. Python and
PYTHONPATH are unaffected by GPU selection.

Feature arguments accept either `--name=value` or `--name value`, and override
the corresponding environment defaults:

| Argument | Environment default | Effect |
| --- | --- | --- |
| `--hsdp_dedup=0` | `HSDP_DEDUP=1` | Disable HSDP replicate dedup |
| `--vanilla_momentum=1` | `VANILLA_MOMENTUM=0` | Select vanilla instead of legacy momentum |
| `--fsdp_overlap=1` | `FSDP_OVERLAP=0` | Request Muon FSDP overlap |
| `--scenario=fsdp` | `SCENARIO=hsdp` | Select the FSDP 2K YAML instead of HSDP 128K YAML |

At 16 total ranks the default YAML is `configs/hyperparallel/qwen3_muon_128k_hsdp_16.yaml`;
FSDP selects `qwen3_muon_2k_16.yaml`. At 8 total ranks these become
`qwen3_muon_128k_hsdp_8.yaml` (2×4) and `qwen3_muon_2k_8.yaml`, respectively. `CONFIG_FILE` explicitly overrides that
path. The script uses `torchrun` from your existing PATH, or an explicitly
configured TORCHRUN. It preserves PYTHON, PYTHON_EXEC, PYTHONPATH and the working
directory exactly as supplied by the caller. Prepare your environment to import
the intended checkouts before launching; there is no interpreter selection,
source-path injection, project installation or Python launch planner.

### Results selected by the flags

For the example above, the default output is:

```text
<parent-of-VeOmni>/muon_step_results/
  hsdp_dedup0_overlap0_vanilla/
    run0/
      launch-node0.sh / launch-node1.sh
      workload.json
      environment-node0.json / environment-node1.json
      warmup.json
      timing-rank00000.json ...
      result.json
      torchrun/node0/ ...
      torchrun/node1/ ...
      profiles/rank00000/ ...
```

`OUTPUT_DIR` overrides the **base** directory; the flag-derived case name and
`RUN_ID` are appended. `RUN_ID=run0` is deterministic on both nodes by default.
For a repeated experiment set the same new ID on both nodes, for example
`RUN_ID=repeat2 bash script.sh --hsdp_dedup=0 --vanilla_momentum=1`.
Existing per-node launch markers prevent accidental overwriting. Profiles and
worker logs stay on their respective nodes unless output storage is shared;
rank0 writes combined timing statistics in result.json.

Other environment overrides are `DEVICE`, `FSDP_BACKEND`, `SEED`, `BUDGET_MIB`,
`WARMUP_MIN`, `WARMUP_MAX`, `STABILITY_WINDOW`, `STABLE_WINDOWS`, `STABILITY_CV`,
`STABILITY_DRIFT`, `MEASURE_STEPS`, `PROFILE_STEPS`, `PROFILE_WARMUP`,
`PROFILE_RANKS`, `TIMEOUT_SECONDS` and `TRIAL`. `SNAPSHOT_ROOT` selects captured
inputs instead of MODEL_CONFIG. Additional training-entry flags, such as
`--profile-memory --record-shapes --with-stack`, are forwarded directly.
`--dry-run` prints the final result directory and exactly one torchrun command.
The entry supports CUDA training and CPU infrastructure smoke only; it has no
Ascend backend/profiler dependency and performs no device-backend imports.

The remainder describes the training entry and the optional **matrix** helper
`scripts/run_muon_step_st.sh`, which enumerates combinations. Use `script.sh`
for the single-case, manually launched interface above.

## Training replacement

`tasks/train_muon_step.py` follows `tasks/train_text.py`'s model/optimizer build
boundaries through VeOmni's `build_foundation_model`, `build_parallelize_model`
and `build_optimizer`. It reads the original FSDP/HSDP training YAML.

The default path retains the actual wrapped Qwen3/Qwen3-MoE module tree. It builds
on meta, bypasses checkpoint loading, classifies parameters using VeOmni's own
Muon/AdamW split, and materializes Muon local shards at their original model FQNs.
Non-Muon parameters remain on meta, and AdamW state is never allocated.
Hyper DTensor is the default; `FSDP_BACKEND=torch` selects Torch DTensor while
retaining Hyper Muon. The supported layout is Shard(0), optionally with an HSDP
replicate axis; EP/TP/PP/CP/SP and CPU offload are disabled as in the source YAMLs.

Each `MuonStepTrainer.step()` runs:

```text
forward()             dummy: returns None; no model/FSDP forward hooks
backward(loss)        dummy: keeps preinitialized reduced gradients
clip_grad_norm()      dummy: gradients represent post-clip inputs
muon.step()           REAL: momentum, NS, collectives and parameter update
adamw_step()          dummy
scheduler_step()      dummy: constant source LR
zero_grad()           dummy: preserves gradients for the next iteration
```

Dataset/tokenizer loading, checkpoint I/O and activation allocation are omitted.
Initial Muon weights and gradients are deterministic nonzero normal values;
replicate peers receive identical local shards. Weights and momentum evolve,
while gradients stay fixed across iterations. All model Muon tensors, including
3D experts, have gradients in this mode. Optional first-step snapshot replay
preserves real input values and missing-gradient masks instead.

This preserves Muon shapes, grouping, layout, dtypes, settings and computation;
it does not reproduce training gradient distributions, activation memory pressure
or convergence. Timings cover the dummy training iteration and all-stream device
completion, with Muon as the only real training stage. They are optimizer latency
results, not full-training speedups. The 2K/128K sequence lengths affect real
forward/backward but do not change this dummy iteration's work.

## GPU environment and source-only setup

Use an existing GPU VeOmni training environment with its dependencies,
CUDA-enabled Torch/NCCL, Python 3.11/3.12 and transformers 5.16.1.
The two project checkouts themselves do **not** need pip installation.

```text
experiment/
  hyper-parallel/
  VeOmni/                # this branch
  run_muon_step_st.sh     # copy VeOmni/scripts/run_muon_step_st.sh here
```

The Bash entries use the caller's existing Python environment and source
import configuration. They do not set or modify PYTHON_EXEC or PYTHONPATH,
do not locate an interpreter, and keep the original working directory so
relative entries in PATH/PYTHONPATH retain their meaning. Training is launched
through `${TORCHRUN:-torchrun}`; its interpreter and worker selection follow
your environment. Workers still check that imported packages come from the
requested source roots.

## One machine, 16 devices

From `experiment/`:

```bash
cp VeOmni/scripts/run_muon_step_st.sh .
NNODES=1 NPROC_PER_NODE=16 MODEL_CONFIG=/models/Qwen3-30B-A3B bash run_muon_step_st.sh
```

`MODEL_CONFIG` is the **same actual model's local HF config directory** as real
training. The YAMLs contain placeholder model paths, so it must be supplied.
No weights, tokenizer or dataset are read. Paths to a local config.json are also
accepted by the VeOmni loader.

```bash
# Print all direct torchrun commands without starting Python or training jobs.
NNODES=1 NPROC_PER_NODE=16 MODEL_CONFIG=/models/Qwen3-30B-A3B bash run_muon_step_st.sh --dry-run

# Select one topology; CUDA/NCCL is the default.
SCENARIO=hsdp NNODES=1 NPROC_PER_NODE=16 MODEL_CONFIG=/models/Qwen3-30B-A3B bash run_muon_step_st.sh
```

A single case can also be launched directly, without the suite Bash script:

```bash
# Use your environment already configured for both source checkouts.
torchrun --nproc-per-node=16 VeOmni/tasks/train_muon_step.py \
  VeOmni/configs/hyperparallel/qwen3_muon_128k_hsdp_16.yaml \
  --model-config /models/Qwen3-30B-A3B --output ./one-case \
  --case hsdp_dedup0_overlap1_vanilla --scenario hsdp \
  --dedup 0 --overlap 1 --momentum vanilla
```

## Matrix suite: one coordinator launches all hosts

Configure the environment **only on the coordinator** and run the matrix entry
once. Both hosts need passwordless SSH, the same checkout/model/interpreter
paths, and rsync for result collection. MASTER_ADDR must be the first host's
communication IP reachable from every node.

```bash
export HOSTS=gpu-node0,gpu-node1
export PYTHON=/path/to/training-env/bin/python
# Keep your existing PYTHONPATH configured for both source checkouts.
export MASTER_ADDR=10.0.0.10
export MODEL_CONFIG=/models/Qwen3-30B-A3B
export NPROC_PER_NODE=4
bash VeOmni/scripts/run_muon_step_st.sh
```

This launches both hosts through SSH. Each remote Bash worker executes:

```bash
"$PYTHON" -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=4 --node-rank=<0-or-1> \
  --master-addr="$MASTER_ADDR" --master-port=<case-port> \
  ... tasks/train_muon_step.py configs/hyperparallel/qwen3_muon_128k_hsdp_8.yaml ...
```

NNODES is inferred from HOSTS; NODE_RANK is assigned by host-list order. RUN_ID
is generated once on the coordinator and shared by all hosts. PYTHON is required
for coordinator mode. PYTHONPATH, existing PYTHON_EXEC, PATH, LD_LIBRARY_PATH,
CUDA/NCCL settings and experiment overrides are forwarded as configured; the
script adds no Python source paths and rewrites no interpreter path. The remote
working directory matches the coordinator so relative environment paths retain
their meaning. Existing PYTHON_EXEC is honored by PyTorch's worker launcher.

Every host independently runs the same deterministic case order and rendezvous
ports. `NPROC_PER_NODE=4` selects GPUs 4–7 on both hosts and uses HSDP 2×4 (8
ranks total). `NPROC_PER_NODE=8` uses HSDP 2×8 (16 ranks total).
Coordinator logs are saved as `coordinator/node<N>.log`; exact SSH commands are
saved in `dispatch.sh`. After all hosts complete, rsync collects rank artifacts
and traces, and the configured PYTHON writes the summary/ablation tables. If a
node fails, remaining SSH jobs are terminated and partial artifacts retrieved.

`--dry-run` with HOSTS prints SSH dispatch commands without contacting hosts.
Without HOSTS, the matrix entry still supports local/manual launch, using
`${PYTHON:-python3} -m torch.distributed.run`. Rebuild summaries with
`RUN_ID=<id> bash VeOmni/scripts/run_muon_step_st.sh --summarize-only`.

Rendezvous ports start at MASTER_PORT=29540, one per case. Independent suites
need different ranges. GPU communication uses NCCL.

## Factorial experiments

| Scenario | Dedup | Overlap | Momentum | Cases |
| --- | --- | --- | --- | --- |
| FSDP 8 / 16 | on (no replicate axis) | off/on | legacy/vanilla | 4 |
| HSDP 2×4 / 2×8 | on/off | off/on | legacy/vanilla | 8 |

Default 3 repeats produce 36 independent torchrun jobs. Baseline within each
topology/repeat is dedup on + overlap off + legacy. `summary.csv` reports matched
baseline speedups; `ablations.csv` compares one factor at a time with the other
factors held fixed (16 matched pairs per complete repeat). Vanilla changes the
momentum formula; these runs do not establish numerical equivalence to legacy.

The source YAMLs default to Torch FSDP, and the FSDP YAML disables Hyper Muon.
The ST intentionally selects `fsdp_backend=hyper` and `use_hyper_optimizer=true`,
records source and effective settings, and preserves NS coefficients/steps,
Nesterov, RMS scaling, LR, weight decay and parameter grouping through VeOmni.
Eager ops are bound during meta construction only; model forward never executes.

Current HP overlap requires native callbacks and a single shard assignment
without replicate dedup. HSDP dedup-on + overlap-on are expected fallback controls.
A single bucket or insufficient budget can also cause fallback. Preflight records
actual pipeline bucket execution, eligibility and memory plans, so requested
flags are never mistaken for actual overlap execution. Timed code has no patches.
`BUDGET_MIB=512` applies the same explicit budget to serial and overlap paths;
this is an optimizer estimate, not a hard HBM limit. Inspect oversized fallback
before interpreting a budget sweep.

## Warmup, timing and profiling windows

Defaults:

1. One preflight Muon step initializes state and records actual pipeline execution.
2. Warmup: minimum 20, maximum 200 steps. Two adjacent 10-step windows must each
   have CV ≤ 5% and median drift ≤ 3%, for 3 consecutive checks (earliest stop:
   step 40). Slowest-rank timings drive identical decisions on every rank.
   Failure to stabilize saves warmup evidence and stops the case.
3. Measure the next 50 dummy training iterations **without profiling**. Each begins
   after an untimed world barrier/device sync and ends after the Muon update and
   all-stream device completion. Preserve each rank's samples and per-step rank
   maxima; report mean/median/p90/p95/p99/min/max/std/CV and peak memory.
4. Flush timing artifacts, then start a **separate** profiler window: 2 warmup +
   5 active iterations. Profile all ranks by default; `--profile-ranks 0,8`
   selects ranks while every rank still runs the same iterations. CUDA exports
   Chrome JSON through `torch.profiler` (CPU + CUDA activities).

Step synchronization measures isolated iteration latency, including Muon's internal
overlap. `max_rank_window_ms` includes alignment overhead and is not throughput.
Profiler overhead does not enter latency statistics. `PROFILE_STEPS=0` explicitly
skips profiling. Common environment overrides include `REPEATS`, `WARMUP_MIN`,
`WARMUP_MAX`, `STABILITY_WINDOW`, `STABLE_WINDOWS`, `STABILITY_CV`, `STABILITY_DRIFT`,
`MEASURE_STEPS`, `PROFILE_WARMUP`, `PROFILE_STEPS`, `PROFILE_RANKS`, `BUDGET_MIB` and
`OUTPUT_DIR`. Equivalent CLI flags are accepted; `--cases` selects names shown by
the dry run. Additional trace flags: `--record-shapes --profile-memory --with-stack`.

## Analyze node0.log before the suite finishes

The standalone log analyzer needs only Python's standard library. It reads a
snapshot of the current log, prints a Chinese Markdown report, and saves CSVs:

```bash
bash VeOmni/scripts/analyze_muon_step_log.sh \
  /path/to/muon_step_results/<RUN_ID>/coordinator/node0.log
```

By default it analyzes only the first repeat (`--trial 0`). Reports are saved
beside the log in `node0-analysis-r00/report.md`, `summary.csv`, and `ablations.csv`.
Repeat the command to refresh the report while training continues. The analyzer
never launches training, reads result.json or traces, modifies the input log,
or changes Python paths. It uses the caller's PYTHON or otherwise python3.

Options:

```bash
# HSDP only; customize where the analysis files are written.
bash VeOmni/scripts/analyze_muon_step_log.sh /path/to/node0.log \
  --scenario hsdp --output /path/to/analysis-first-round
# Second repeat (indices start at zero).
bash VeOmni/scripts/analyze_muon_step_log.sh /path/to/node0.log --trial 1
# If the suite ran selected cases, specify their names for accurate completion counts.
bash VeOmni/scripts/analyze_muon_step_log.sh /path/to/node0.log \
  --cases fsdp_dedup1_overlap0_legacy fsdp_dedup1_overlap1_legacy
```

The parser recognizes final rank0 result lines such as
`hsdp_dedup0_overlap1_vanilla: 12.345 ms, overlap=executed`, excluding warmup
means. It associates them with the preceding torchrun command's `--case` and
`--trial`; commandless logs use each case's occurrence count and emit a warning
because truncated or concatenated logs can make that inference ambiguous.
ANSI colors and torchrun prefixes are supported. Duplicate case/trial results
keep the last logged value with a warning. A last line without a newline is
skipped because it may still be in progress. Other formats are diagnosed rather
than silently interpreted as benchmark results.

The report shows per-scenario rankings, baseline speedups and latency reductions,
and matched single-factor comparisons, including both cases' actual overlap
status in CSV. Baselines are dedup on + overlap off + legacy within the selected
repeat and scenario. Missing results remain listed; no requirement to finish
all cases or all repeats. Negative reductions indicate regressions. Overlap
fallback is flagged so the switch's effect is not attributed to an executed
pipeline. Case names provide the pairing: logs cannot verify workload inventories,
model configurations, topology or source versions. Only logged means are used
(to their printed precision); P95, CV and memory cannot be recovered from these
lines. One repeat provides preliminary observations, not repeatability evidence.

## Artifacts

```text
muon_step_results/<RUN_ID>/
  plan-node<N>.sh                   # exact direct torchrun invocations
  summary.csv / summary.json        # each repeat, baseline speedup, actual overlap
  ablations.csv                    # matched single-factor comparisons
  r00_<case>/
    environment-node<N>.json        # versions, selected runtime settings, device probe
    workload.json                   # actual inventory, source YAML, settings, git revisions/diffs
    warmup.json
    timing-rank00000.json           # durable before profiler creation
    result.json                     # combined ranks, statistics, peaks, environment
    failure-rank00000.json          # on failure
    torchrun/                       # per-worker stdout/stderr
    profiles/rank00000/trace.json    # per-rank Chrome traces
```

No outliers are trimmed. Keep repeats separate when comparing results. A profiler
failure leaves timing files but does not publish a completed result.json.

## Optional actual first-step capture

The capture wrapper is independent of the default dummy training entry. It can
run the original training task once and capture exact post-reduction/post-clip
Muon weights/gradients with empty momentum state:

```bash
cd VeOmni
# Use your existing environment configured to import both source roots.
torchrun --nproc-per-node=16 -m scripts.muon_step_benchmark.capture \
  --output /experiment/snapshots/fsdp --task tasks/train_text.py -- \
  configs/hyperparallel/qwen3_muon_2k_16.yaml \
  --model.model_path /models/Qwen3-30B-A3B --data.train_path /data/train \
  --model.accelerator.fsdp_config.fsdp_backend hyper \
  --model.optimizer.use_hyper_optimizer true --train.max_steps 1
```

Repeat for the HSDP YAML with snapshots/hsdp. Real capture needs the actual
training environment and sufficient memory (the 128K forward/backward may OOM).
Store every rank's CPU tensor file plus rank0's manifest on shared storage or
combine node files before replay. Existing captures are never overwritten.
Replay with `SNAPSHOT_ROOT=/experiment/snapshots bash run_muon_step_st.sh`.
Replay checks layout, topology, shapes, dtypes and optimizer settings; subsequent
gradients remain fixed, so this is still a dummy training experiment.

## Infrastructure smoke

```bash
# Small matrices only; CPU checks cannot exercise accelerator overlap.
NNODES=1 DEVICE=cpu NPROC_PER_NODE=2 REPEATS=1 \
bash run_muon_step_st.sh --smoke --scenario fsdp \
  --warmup-min 4 --warmup-max 60 --stability-window 2 --stable-windows 1 \
  --stability-cv 0.8 --stability-drift 0.8 --measure-steps 2 --profile-steps 1

# Real VeOmni tiny model tree, 4 accelerator ranks, all ablations.
NNODES=1 NPROC_PER_NODE=4 REPEATS=1 BUDGET_MIB=1 \
MODEL_CONFIG=VeOmni/scripts/muon_step_benchmark/tiny_qwen3_moe \
bash run_muon_step_st.sh --smoke \
  --warmup-min 8 --warmup-max 100 --stability-window 4 --stable-windows 1 \
  --stability-cv 0.8 --stability-drift 0.8 --measure-steps 3 --profile-steps 2
```

CPU supports the matrix control; real VeOmni FSDP wrapping requires CUDA.
Smoke results are explicitly marked and are not Qwen performance evidence.
See [validation record](VALIDATION.md) for executed checks and remaining coverage.
