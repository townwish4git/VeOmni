# Validation record

Validated on `t129` (Ascend 910B3), 2026-10-23, using an isolated editable
installation of both source trees. HyperParallel source: `81439e916d11b09ec8cab71506dd7e17ec99152b`.
VeOmni base: Cui-yshoho `feat/hyper-parallel-backend` at
`3b17d580aba07d7e5bb26b9ffcaa3ebd48e745aa`; benchmark branch:
`test/hyper-muon-step-ablation`.

Environment: Python 3.11, Torch 2.9.0, torch_npu 2.9.0, transformers 5.16.1,
NumPy 1.26.4, CANN 8.5.0. Only four available devices were used.

Passed checks:

- `make quality` with repository-locked ruff 0.13.2, plus ruff check/format of the new scripts,
  `git diff --check`, Python compilation and `bash -n`.
- `python -m pytest -q tests/optim/test_muon_unit.py -k TestMuonKwargsFromOptimizerConfig`:
  **14 passed**, including new Hyper config/factory forwarding and rejection cases.
- Import-free dry-run, 4+8 factorial cases, 36 distinct repeat/case jobs,
  warmup stability decisions, uneven shard sizing, percentile/statistics calculations,
  baseline speedups and **16** matched single-factor pairs for a complete repeat.
- The Bash file copied to the common parent correctly resolves the two repositories
  and relative model-config paths.
- Full four-device, 24-matrix smoke: **12 completed cases / 48 parsed NPU traces**.
  Four cases executed the multibucket pipeline; two HSDP dedup-on controls reported
  expected fallback; six serial cases reported disabled. Both momentum modes ran.
- Final copied-Bash entry with the bundled tiny Qwen3-MoE: **2 completed cases /
  8 parsed NPU traces**, covering FSDP 4 and HSDP 2×2. Real VeOmni meta wrapping and
  its optimizer factory selected 14 Muon parameters, including 3D expert weights,
  with the source YAML's five custom-coefficient NS iterations. HSDP executed
  pipeline buckets; the small FSDP case correctly reported single-bucket fallback.
- A separate four-device pre-step hook capture/replay check passed bitwise equality
  of local weights/gradients and preservation of a missing-gradient mask.

The small smoke uses relaxed stability tolerances to test infrastructure; its
latencies are not Qwen performance results. No 16-device full-size model,
128K forward/backward, two-host SSH fan-out, CUDA run, numerical acceptance or
convergence experiment was performed. The default benchmark uses the stricter
5% CV / 3% median-drift thresholds documented in the README.

The final copied-Bash NPU command was:

```bash
cd /home/townwish/workspace/projects/veomni-muon-step-st-20261023
cp VeOmni/scripts/run_muon_step_st.sh .
ASCEND_RT_VISIBLE_DEVICES=0,4,6,7 \
PYTHON="$PWD/env/bin/python" \
MODEL_CONFIG="$PWD/VeOmni/scripts/muon_step_benchmark/tiny_qwen3_moe" \
OUTPUT_DIR="$PWD/results" NPROC_PER_NODE=4 REPEATS=1 \
WARMUP_MIN=8 WARMUP_MAX=60 MEASURE_STEPS=3 PROFILE_STEPS=2 BUDGET_MIB=1 \
MASTER_PORT=29740 HCCL_BASE_PORT=44000 RUN_ID=tiny-veomni-final-bash-v2 \
bash run_muon_step_st.sh --smoke \
  --stability-window 4 --stable-windows 1 --stability-cv 0.8 --stability-drift 0.8 \
  --profile-warmup 1 \
  --cases fsdp_dedup1_overlap1_vanilla hsdp_dedup0_overlap1_vanilla
```

The full matrix smoke command, from the same directory's `VeOmni/`, was:

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 \
../env/bin/python -m scripts.muon_step_benchmark.run \
  --smoke --fsdp-backend hyper --device npu --nproc-per-node 4 --repeats 1 \
  --warmup-min 8 --warmup-max 60 --stability-window 4 --stable-windows 1 \
  --stability-cv 0.8 --stability-drift 0.8 --measure-steps 3 \
  --profile-warmup 1 --profile-steps 2 --output ../results \
  --run-id smoke-hyper-4card-v3 --budget-mib 1
```

Warnings observed: CANN installation ownership, deprecated TypedStorage,
NPU base-format normalization and the NPU profiler normalizing an unspecified
export type to text. All required `trace_view.json` files were checked after
parsing. A shared-host HCCL bind conflict during an earlier check motivated
per-case HCCL port ranges; the final check passed with explicit independent ports.

Raw logs and artifacts remain under the isolated validation directory above.
