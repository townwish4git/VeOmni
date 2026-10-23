# VeOmni dummy training / HyperParallel Muon ST

Launch path:

```text
run_muon_step_st.sh
  └─ torchrun ... tasks/train_muon_step.py <original-training.yaml>
       └─ MuonStepTrainer: real VeOmni construction + dummy training stages
```

Bash owns the experiment matrix, repeats, case ordering and distributed launch.
There is no Python experiment launcher. After training jobs finish, Bash invokes
`report.py` to summarize their existing result files; that module launches no jobs.

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

## Source-only setup

Use an existing VeOmni training environment with its dependencies, matching
Torch/torch_npu/CANN or CUDA, Python 3.11/3.12 and transformers 5.16.1.
The two project checkouts themselves do **not** need pip installation.

```text
experiment/
  hyper-parallel/
  VeOmni/                # this branch
  run_muon_step_st.sh     # copy VeOmni/scripts/run_muon_step_st.sh here
```

The Bash entry automatically exports both source roots in `PYTHONPATH` and sets
`PYTHON_EXEC` so torchrun workers use `PYTHON`. Worker import-path checks ensure
that those source directories supply the packages. `PYTHON` defaults to python3;
`TORCHRUN` uses its sibling executable if present, otherwise torchrun from PATH.
Activate the training environment first, or provide both executable paths.

## One machine, 16 devices

From `experiment/`:

```bash
cp VeOmni/scripts/run_muon_step_st.sh .
MODEL_CONFIG=/models/Qwen3-30B-A3B bash run_muon_step_st.sh
```

`MODEL_CONFIG` is the **same actual model's local HF config directory** as real
training. The YAMLs contain placeholder model paths, so it must be supplied.
No weights, tokenizer or dataset are read. Paths to a local config.json are also
accepted by the VeOmni loader.

```bash
# Print all direct torchrun commands without starting Python or training jobs.
MODEL_CONFIG=/models/Qwen3-30B-A3B bash run_muon_step_st.sh --dry-run

# Select one topology; use DEVICE=cuda for NVIDIA.
SCENARIO=hsdp MODEL_CONFIG=/models/Qwen3-30B-A3B bash run_muon_step_st.sh
```

A single case can also be launched directly, without the suite Bash script:

```bash
export PYTHONPATH="$PWD/hyper-parallel:$PWD/VeOmni${PYTHONPATH:+:$PYTHONPATH}"
torchrun --nproc-per-node=16 VeOmni/tasks/train_muon_step.py \
  VeOmni/configs/hyperparallel/qwen3_muon_128k_hsdp_16.yaml \
  --model-config /models/Qwen3-30B-A3B --output ./one-case \
  --case hsdp_dedup0_overlap1_vanilla --scenario hsdp \
  --dedup 0 --overlap 1 --momentum vanilla
```

## Two machines, 8 devices each

Manual per-node launch directly enters torchrun from Bash. Run on each node
with `NODE_RANK=0` / `1` and identical flags/RUN_ID:

```bash
MODEL_CONFIG=/models/Qwen3-30B-A3B NNODES=2 NPROC_PER_NODE=8 \
NODE_RANK=0 MASTER_ADDR=10.0.0.10 RUN_ID=muon-ablation-001 \
bash run_muon_step_st.sh
```

For one-command SSH launch from the coordinator:

```bash
MODEL_CONFIG=/models/Qwen3-30B-A3B HOSTS=node0,node1 \
NPROC_PER_NODE=8 MASTER_ADDR=10.0.0.10 \
PYTHON=/path/to/env/bin/python TORCHRUN=/path/to/env/bin/torchrun \
bash run_muon_step_st.sh
```

`HOSTS` contains SSH names/IPs/aliases, first node rank 0. Both nodes need the
same absolute paths and dependencies; the coordinator needs passwordless SSH
and rsync. Each SSH child runs the **Bash script**, which sets source paths
locally and directly calls torchrun. Thus source-only imports also work remotely.
The SSH shells must supply their normal CANN/CUDA runtime environment.

After jobs finish, profiles are copied back with rsync and results summarized.
In manual mode rank0 summarizes combined timing results, while profiles stay on
each rank's node unless output storage is shared. Rebuild summaries after copying:

```bash
RUN_ID=muon-ablation-001 bash run_muon_step_st.sh --summarize-only
```

Rendezvous ports start at `MASTER_PORT=29540`, one per case. NPU cases use distinct
`HCCL_IF_BASE_PORT` ranges starting at `HCCL_BASE_PORT=40000`, 256 ports per case.
Both nodes use identical bases; independent suites need different ranges.
Port ranges are validated against 65535. Case order is deterministically shuffled
per repeat by Bash; both nodes obtain identical order and ports.

## Factorial experiments

| Scenario | Dedup | Overlap | Momentum | Cases |
| --- | --- | --- | --- | --- |
| FSDP 16 | on (no replicate axis) | off/on | legacy/vanilla | 4 |
| HSDP 2×8 | on/off | off/on | legacy/vanilla | 8 |

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
   Chrome JSON; NPU exports Level1 native traces and operator data.

Step synchronization measures isolated iteration latency, including Muon's internal
overlap. `max_rank_window_ms` includes alignment overhead and is not throughput.
Profiler overhead does not enter latency statistics. `PROFILE_STEPS=0` explicitly
skips profiling. Common environment overrides include `REPEATS`, `WARMUP_MIN`,
`WARMUP_MAX`, `STABILITY_WINDOW`, `STABLE_WINDOWS`, `STABILITY_CV`, `STABILITY_DRIFT`,
`MEASURE_STEPS`, `PROFILE_WARMUP`, `PROFILE_STEPS`, `PROFILE_RANKS`, `BUDGET_MIB` and
`OUTPUT_DIR`. Equivalent CLI flags are accepted; `--cases` selects names shown by
the dry run. Additional trace flags: `--record-shapes --profile-memory --with-stack`.

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
    profiles/rank00000/              # per-rank traces
```

No outliers are trimmed. Keep repeats separate when comparing results. A profiler
failure leaves timing files but does not publish a completed result.json.

## Optional actual first-step capture

The capture wrapper is independent of the default dummy training entry. It can
run the original training task once and capture exact post-reduction/post-clip
Muon weights/gradients with empty momentum state:

```bash
cd VeOmni
# PYTHONPATH must include both source roots, as set by the suite Bash entry.
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
DEVICE=cpu NPROC_PER_NODE=2 REPEATS=1 \
bash run_muon_step_st.sh --smoke --scenario fsdp \
  --warmup-min 4 --warmup-max 60 --stability-window 2 --stable-windows 1 \
  --stability-cv 0.8 --stability-drift 0.8 --measure-steps 2 --profile-steps 1

# Real VeOmni tiny model tree, 4 accelerator ranks, all ablations.
NPROC_PER_NODE=4 REPEATS=1 BUDGET_MIB=1 \
MODEL_CONFIG=VeOmni/scripts/muon_step_benchmark/tiny_qwen3_moe \
bash run_muon_step_st.sh --smoke \
  --warmup-min 8 --warmup-max 100 --stability-window 4 --stable-windows 1 \
  --stability-cv 0.8 --stability-drift 0.8 --measure-steps 3 --profile-steps 2
```

CPU supports the matrix control; real VeOmni FSDP wrapping requires NPU or CUDA.
Smoke results are explicitly marked and are not Qwen performance evidence.
See [validation record](VALIDATION.md) for executed checks and remaining coverage.
