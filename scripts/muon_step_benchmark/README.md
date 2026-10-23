# VeOmni / HyperParallel Muon step ST

This is an optimizer latency and profiling experiment, derived from
`configs/hyperparallel/qwen3_muon_2k_16.yaml` (FSDP 16) and
`configs/hyperparallel/qwen3_muon_128k_hsdp_16.yaml` (HSDP replicate 2 × shard 8).
It measures the **leaf HyperParallel `Muon.step()`**, including host dispatch,
communication, momentum, Newton–Schulz, parameter updates and device completion.
It excludes AdamW, gradient clipping, scheduler, data loading, forward and backward.
The result is optimizer-step speedup; it is not full-training speedup or a numerical/convergence test.

## Directory and environment

Use the same accelerator environment as real VeOmni training (Python 3.11/3.12,
VeOmni dependencies, transformers 5.16.1, matching Torch/torch_npu/CANN or CUDA).
The benchmark does not install or replace your accelerator stack.

```text
experiment/
  hyper-parallel/       # current Muon implementation, all three features present
  VeOmni/               # this branch
  run_muon_step_st.sh    # copy VeOmni/scripts/run_muon_step_st.sh here
  muon_step_results/     # created automatically
```

In the active environment on **each node**, install both checkouts as editables:

```bash
python -m pip install --no-deps -e ./hyper-parallel -e ./VeOmni
python -c 'import hyper_parallel, veomni; print(hyper_parallel.__file__, veomni.__file__)'
```

`MODEL_CONFIG` must point to the **same Qwen3 model's local HF directory** used
for training (a directory containing `config.json`; a config file is also accepted
by the VeOmni loader). The supplied YAMLs contain placeholder checkpoint paths,
so the harness cannot infer which model size you trained. No checkpoint weights,
tokenizer or dataset are needed in the default extraction mode.

## One machine with 16 devices

From `experiment/`:

```bash
MODEL_CONFIG=/models/Qwen3-30B-A3B bash run_muon_step_st.sh
```

Preview the full matrix and commands without importing Torch or launching jobs:

```bash
MODEL_CONFIG=/models/Qwen3-30B-A3B bash run_muon_step_st.sh --dry-run
```

The default is NPU, Hyper FSDP backend, 3 repeats, warmup 20–200 steps,
50 unprofiled measurement steps, then 2 profiler warmup + 5 active steps,
all ranks profiled. Cases execute in a deterministic randomized order per repeat.
Each case/repeat starts a fresh torchrun job and fresh optimizer state.
Select CUDA using `DEVICE=cuda`. Select one topology using `SCENARIO=fsdp` or `hsdp`.
Use `--cases ...` with names printed by the dry run to select individual cases.

## Two machines with 8 devices each

For SSH one-command launch, both machines need the same absolute repository,
model/config, environment/interpreter and output paths. The coordinator needs
passwordless SSH plus `rsync`, and `MASTER_ADDR` must be reachable by both nodes.
SSH shells must have the normal CANN/CUDA runtime environment configured.

```bash
MODEL_CONFIG=/models/Qwen3-30B-A3B \
NPROC_PER_NODE=8 HOSTS=node0,node1 MASTER_ADDR=10.0.0.10 \
PYTHON=/path/to/training/env/bin/python \
bash run_muon_step_st.sh
```

The script SSH-launches both nodes, waits for completion, retrieves each node's
rank artifacts/profiles with rsync, and writes the combined summary locally.
Retrieval errors are printed with the remote path; raw data remains on its node.
Do not launch a second copy of the SSH coordinator on node1.

Alternatively run the following **on both nodes**, with `NODE_RANK=0` / `1`:

```bash
MODEL_CONFIG=/models/Qwen3-30B-A3B NNODES=2 NPROC_PER_NODE=8 \
NODE_RANK=0 MASTER_ADDR=10.0.0.10 RUN_ID=muon-ablation-001 \
bash run_muon_step_st.sh
```

Use identical CLI flags and RUN_ID on both nodes. In manual mode rank0 writes
combined timing results; profiles remain on each node unless output is shared.
Every case uses a distinct rendezvous port starting at `MASTER_PORT` (default
29540). On NPU the launcher also assigns a distinct `HCCL_IF_BASE_PORT`, reserving
256 ports per case starting at `HCCL_BASE_PORT` (default 40000), recorded in the
plan and worker environment. Both nodes must use the same bases. Choose different
bases when running another suite concurrently; rendezvous ports alone do not
isolate HCCL. The launcher rejects port ranges extending beyond 65535.

## What is extracted

The harness builds the requested model through VeOmni's registry on `meta`,
runs the actual VeOmni FSDP2 wrap, and calls its Muon/AdamW classifier **after
wrapping**. Only the materialize/load routine is bypassed during this pass.
It extracts Muon FQNs/order, global shape/stride, FP32 optimizer dtype and shard
placements, then allocates only those local parameter and gradient shards.
Reconstruction uses Hyper DTensor by default, including the 2D HSDP mesh;
`FSDP_BACKEND=torch` explicitly selects Torch DTensor while retaining HP Muon.
EP/TP/PP/CP/SP and CPU offload must be disabled as in the supplied configurations.

The two source YAMLs default to `fsdp_backend: torch`, and the FSDP YAML also
has `use_hyper_optimizer: false`. This experiment intentionally overrides those
selections to `hyper` / `true`, records both source YAML and effective selections,
and preserves NS coefficients (custom, 5 iterations), momentum coefficient,
Nesterov, RMS scaling, LR and weight decay through the VeOmni optimizer factory.
Attention/MLP kernels use eager only during meta inventory construction; no
forward kernel executes in the measured workload.

Default weights and gradients are deterministic nonzero normal values, seeded
by original parameter order and shard coordinate. Replicate peers receive the
same shard. Inputs and initial states match across ablations; each step reuses
the fixed post-reduction/post-clip gradient, evolving weights and momentum.
All model Muon parameters, including MoE experts, have a gradient in this mode.
This preserves compute/communication shape and volume, not training gradient
values, sparsity, evolving gradient distributions or activation memory pressure.
Sequence length (2K vs 128K) affects full training but not this isolated step.

## Factorial ablations

| Scenario | Dedup | Overlap | Momentum | Cases |
| --- | --- | --- | --- | --- |
| FSDP 16 | on (no replicate axis) | off/on | legacy/vanilla | 4 |
| HSDP 2×8 | on/off | off/on | legacy/vanilla | 8 |

Baseline within each topology/repeat is dedup on + overlap off + legacy.
This permits single-factor comparisons with other factors held fixed, plus
all combinations and interactions. Vanilla changes the momentum formula;
performance comparison does not establish numerical equivalence to legacy.
`ablations.csv` reports every available matched before/after pair with other
factors fixed, including dedup on→off, overlap off→on and legacy→vanilla.
Speedup is before/after; latency reduction is (1 − after/before) × 100%.

Current HP overlap requires a single shard axis with no replicate work assignment
and native NS/momentum callbacks. HSDP dedup on + overlap on therefore falls back;
those two cases remain as negative controls. HSDP dedup off exposes the FSDP
pipeline path. A preflight counts actual pipeline bucket execution and records
eligibility and bucket/memory plans; timing has no instrumentation patches. `overlap_status` distinguishes
execution, disabled, expected dedup fallback and unexpected fallback.

`BUDGET_MIB=512` sets the same explicit temporary budget for **serial and overlap**
for a fair bucket comparison. This is an optimizer estimate, not a hard HBM cap.
Large expert matrices may exceed the budget and use the serial oversized path.
Inspect preflight bucket counts and workload inventory; repeat with a larger
identical budget across every case if you want a budget sweep.

## Windows and metrics

1. Untimed preflight initializes optimizer state and checks pipeline execution.
2. Warmup continues until two adjacent 10-step windows each have CV ≤ 5% and
   median drift ≤ 3%, repeated for 3 consecutive checks (minimum 40 steps by
   default). Slowest-rank timings drive the same stop decision on every rank.
   If 200 steps do not stabilize, the case fails and writes `warmup.json`;
   it publishes no successful performance result. Relax explicitly with
   `--stability-cv`, `--stability-drift`, `--stable-windows` if needed.
3. The next 50 steps have no profiler. Each sample starts after an untimed world
   barrier/device sync, and ends after `Muon.step()` plus all-stream device sync.
   Original per-rank samples and per-step rank maxima are preserved. Stats include
   mean, median, p90/p95/p99, min/max, standard deviation and CV. The main summary
   uses the mean of per-step rank maxima. `max_rank_window_ms` includes untimed
   alignment overhead and must not be used as optimizer throughput.
4. Only after timing JSON is flushed does the profiler start, with a separate
   warmup and active window. Every rank continues the same number of steps even
   if `--profile-ranks 0,8` profiles selected ranks only. Trace overhead never
   enters the latency statistics. `PROFILE_STEPS=0` disables profiling explicitly.

Each device step is synchronized intentionally: this measures isolated optimizer
latency including internal communication/compute overlap, not cross-step overlap.
Profiling follows the same synchronized step boundaries. CUDA emits Chrome JSON;
NPU uses torch_npu Level1 and its TensorBoard handler's native trace/operator data.

## Artifacts and analysis

```text
muon_step_results/<RUN_ID>/
  plan-node<N>.json                     # exact commands and order
  environment-node<N>.json              # package versions, driver/device probe, selected runtime settings
  summary.csv / summary.json           # each repeat, baseline speedup, effective overlap
  ablations.csv                        # matched single-factor pairs within each repeat
  r00_<case>/
    workload.json                      # inventory/hash, source YAML, effective flags, git diffs
    warmup.json
    timing-rank00000.json              # durable before profiling starts
    result.json                        # all ranks, statistics, peaks, environment
    failure-rank00000.json              # if a worker fails
    torchrun/                          # stdout/stderr per worker
    profiles/rank00000/                 # per-rank trace
```

Do not treat `--smoke` results as Qwen performance. Do not pool individual steps
from different repeats as independent experiments. `summary.csv` preserves
per-repeat speedups; inspect variation between repeats and compare matched
factor settings. No outliers are trimmed. A failed profiler leaves timing files,
but does not create a completed `result.json`.

Rebuild the summary after copying node artifacts:

```bash
bash run_muon_step_st.sh --summarize-only --run-id muon-ablation-001
```

## Optional exact first-training-step input capture

For the exact weights, actual post-reduction/post-clip gradients and missing-grad
mask of a real training step, launch the existing training task through the capture
wrapper (same torchrun arguments on both nodes). It attaches a one-shot pre-step
hook to HP Muon. Set the normal training overrides including local model/data,
`model.optimizer.use_hyper_optimizer=true` and the selected FSDP backend.

```bash
cd VeOmni
python -m torch.distributed.run --nproc-per-node=16 \
  -m scripts.muon_step_benchmark.capture --output /experiment/snapshots/fsdp \
  --task tasks/train_text.py -- configs/hyperparallel/qwen3_muon_2k_16.yaml \
  --model.model_path /models/Qwen3-30B-A3B --data.train_path /data/train \
  --model.accelerator.fsdp_config.fsdp_backend hyper \
  --model.optimizer.use_hyper_optimizer true --train.max_steps 1
```

Repeat with the HSDP YAML and `--output /experiment/snapshots/hsdp`.
Capture needs real forward/backward once, enough HBM, disk and host RAM; the 128K
configuration may OOM before the hook. Snapshot tensors are rank-local CPU files,
and `manifest.json` is published after every rank writes its file. Capture supports
only empty first-step momentum state and a common Replicate/Shard(0) layout.
It refuses to overwrite files. For multi-node capture use shared snapshot storage,
or combine every node's rank files before replay (manifest comes from rank0).

```bash
cd /experiment
SNAPSHOT_ROOT=/experiment/snapshots bash run_muon_step_st.sh
```

Replay checks backend, world size, rank mapping, shapes, dtypes and NS/momentum/LR
settings. Weights/gradients are captured exactly; gradients then stay fixed across
warmup and measurement. This is still optimizer-only replay, not resumed training.

## Small infrastructure check

```bash
NPROC_PER_NODE=4 bash run_muon_step_st.sh --smoke --repeats 1 \
  --warmup-min 8 --warmup-max 100 --stability-window 4 --stable-windows 1 \
  --measure-steps 3 --profile-steps 2
```

This uses 24 small matrices to check launch, both meshes, switches, statistics and
trace generation. It skips model/VeOmni imports and is clearly labelled in output.
It is not a performance acceptance test. CPU `--device cpu --smoke` is supported
for control flow, but cannot exercise accelerator overlap.

To check the real VeOmni meta-wrap and optimizer factory on fewer devices,
append `--model-config scripts/muon_step_benchmark/tiny_qwen3_moe` to the smoke
command. The bundled tiny model is only for infrastructure verification. The
`smoke` field remains true in the workload and result metadata. Optional capture
replay can likewise use `--smoke` for matching small-world snapshots.

See [validation record](VALIDATION.md) for executed checks and untested coverage.
