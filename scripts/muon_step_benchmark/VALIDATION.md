# Validation record — direct Bash / torchrun entry

Host: `t129` (Ascend 910B3), 2026-10-23. Source branch:
`test/hyper-muon-step-ablation`, based on Cui-yshoho VeOmni
`3b17d580aba07d7e5bb26b9ffcaa3ebd48e745aa`. HyperParallel source:
`81439e916d11b09ec8cab71506dd7e17ec99152b`.

This revision replaces the Python suite launcher with direct Bash → torchrun →
`tasks/train_muon_step.py`. The default workload retains the actual wrapped
VeOmni model tree, materializes Muon shards only, and leaves non-Muon parameters
on meta. All non-Muon training stages are dummy.

Environment: Python 3.11 workers, Torch/torch_npu 2.9.0, transformers 5.16.1,
NumPy 1.26.4, CANN 8.5.0. torchrun came from the matching Torch 2.9 Python 3.10
installation; `PYTHON_EXEC` selected the Python 3.11 workers. Source imports were
selected through the Bash entry's PYTHONPATH and checked by workers.

Checks:

- `make quality` with repository-locked ruff 0.13.2, explicit ruff checks on
  benchmark helpers, Python compilation, `bash -n`, and `git diff --check`.
- `python -m pytest -q tests/optim/test_muon_unit.py -k
  'TestMuonBashLaunch or TestMuonKwargsFromOptimizerConfig'`: **19 passed**.
  Coverage includes the 36-job matrix/unique ports, identical node ordering,
  direct invocation of a torchrun stand-in while PYTHON is `/bin/false`,
  reporting without resolving torchrun, dummy stages preserving gradients/skipping unrelated updates, and Hyper
  config/factory forwarding. These checks do not need accelerator execution.
- Bash copied to the outer directory, including relative executable and model
  paths, passed dry-run checks without any Python planning process.
- Two-process CPU/Gloo matrix control: **4 completed FSDP cases**, each with
  a separate unprofiled timing window and CPU Chrome trace on both ranks.
- Four-card direct Bash entry with actual tiny Qwen3-MoE: **2 completed cases /
  8 parsed NPU traces**, covering FSDP 4 and HSDP 2×2, vanilla momentum and
  requested overlap. The HSDP dedup-off case executed pipeline buckets; the
  small FSDP case reported single-bucket fallback. The source YAML supplied
  five custom-coefficient NS iterations. Non-Muon AdamW groups were constructed
  from meta parameters and remained dummy.
- Full real-model four-card matrix: **12 completed cases / 48 parsed NPU traces**,
  with **16** matched single-factor pairs. Every workload retained the original
  wrapped Qwen3-MoE model class, 14 Muon tensors and non-Muon meta parameters.
- A separate four-card 24-matrix FSDP control confirmed actual multibucket pipeline
  execution through the new direct training entry, with at least two buckets per
  rank and **4 parsed NPU traces**.

The two-case NPU command was:

```bash
cd /home/townwish/workspace/projects/veomni-muon-step-st-20261023
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
PYTHON="$PWD/env/bin/python" \
TORCHRUN=/home/miniconda3/envs/twx_pt29/bin/torchrun \
OUTPUT_DIR="$PWD/results" \
MODEL_CONFIG="$PWD/VeOmni/scripts/muon_step_benchmark/tiny_qwen3_moe" \
MASTER_PORT=29900 HCCL_BASE_PORT=48000 NPROC_PER_NODE=4 REPEATS=1 DEVICE=npu \
PROFILE_STEPS=2 PROFILE_WARMUP=1 WARMUP_MIN=8 WARMUP_MAX=60 \
STABILITY_WINDOW=4 STABLE_WINDOWS=1 STABILITY_CV=0.8 STABILITY_DRIFT=0.8 \
MEASURE_STEPS=3 BUDGET_MIB=1 RUN_ID=direct-npu-model-v1 \
bash run_muon_step_st.sh --smoke \
  --cases fsdp_dedup1_overlap1_vanilla hsdp_dedup0_overlap1_vanilla
```

The full matrix uses the same command with no `--cases`, `PROFILE_STEPS=1`,
`MASTER_PORT=29940`, `HCCL_BASE_PORT=50000`, and
`RUN_ID=direct-npu-full-matrix-v1`.

The FSDP multibucket control uses the same Bash entry with MODEL_CONFIG unset,
`MASTER_PORT=29980`, `HCCL_BASE_PORT=55000`, `PROFILE_STEPS=2`,
`RUN_ID=direct-npu-fsdp-pipeline-v1`, and
`--smoke --cases fsdp_dedup1_overlap1_legacy`.

Warnings: CANN installation ownership, deprecated TypedStorage, NPU base-format
normalization, and profiler export-type normalization to text. The first attempt
at a real-model CPU check correctly encountered HP's accelerator-only fully_shard
constraint; the CLI now rejects that combination before distributed setup.
CPU matrix smoke remains supported.

Smoke runs use relaxed stability tolerances for infrastructure checks, not
performance acceptance. The production defaults remain 5% CV / 3% median drift.
No 16-device full-size model, 128K forward/backward, two-host SSH fan-out, CUDA,
numerical acceptance or convergence experiment was run for this revision.
Raw logs and results remain under the isolated validation directory above.
