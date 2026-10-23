# GPU Muon ST validation

The ST now defaults to CUDA/NCCL and uses `torch.profiler` with CPU + CUDA
activities and per-rank Chrome trace export. Device/backend import hooks,
Ascend profiling branches, HCCL port configuration and Ascend environment probes
were removed from the ST entry, launchers and reporting helper. CPU/Gloo remains
available only for infrastructure smoke. Python interpreter and import-path
selection remain entirely owned by the caller environment.

Source branch: `test/hyper-muon-step-ablation`, based on Cui-yshoho VeOmni
`3b17d580aba07d7e5bb26b9ffcaa3ebd48e745aa`. HyperParallel source used for
hardware-independent checks: `81439e916d11b09ec8cab71506dd7e17ec99152b`.

## Executed checks

On t129, Python 3.11 / Torch 2.9.0, this command passed **29 tests**:

```bash
python -m pytest -q tests/optim/test_muon_unit.py \
  -k 'TestSingleMuonExperimentScript or TestMuonBashLaunch or TestMuonKwargsFromOptimizerConfig'
```

Coverage includes:

- Default CUDA selection and absence of device-extension import/profiler branches.
- CUDA profiling requests CPU + CUDA activities from torch.profiler, runs the
  expected warmup/active iteration count, and exports profiles/rank00000/trace.json.
  This interface test uses a mocked profiler; it does not run GPU kernels.
- Direct single-case torchrun launch, YAML/flag mapping, environment overrides,
  matching node result directories and preservation of prior experiment data.
- Exact preservation of PYTHON/PYTHON_EXEC/PYTHONPATH and caller working directory,
  including when those environment variables are unset.
- Manual matrix planning, per-case rendezvous ports, identical node order,
  dummy training stages and optimizer config forwarding.

A real two-process **CPU/Gloo** smoke completed and produced two timing samples
per rank, result.json, and two Chrome traces under:

```text
results/single-case/hsdp_dedup0_overlap0_vanilla/gpu-only-cpu-v1/
```

The caller supplied its environment explicitly for this test; script.sh did not
set the Python-related variables:

```bash
cd /home/townwish/workspace/projects/veomni-muon-step-st-20261023
NNODES=1 NPROC_PER_NODE=2 NODE_RANK=0 MASTER_PORT=30160 \
DEVICE=cpu SMOKE=1 RUN_ID=gpu-only-cpu-v1 \
TORCHRUN=/home/miniconda3/envs/twx_pt29/bin/torchrun \
PYTHON_EXEC="$PWD/env/bin/python" PYTHONPATH="$PWD/hyper-parallel:$PWD/VeOmni" \
OUTPUT_DIR="$PWD/results/single-case" \
WARMUP_MIN=4 WARMUP_MAX=40 STABILITY_WINDOW=2 STABLE_WINDOWS=1 \
STABILITY_CV=1 STABILITY_DRIFT=1 MEASURE_STEPS=2 PROFILE_STEPS=1 PROFILE_WARMUP=1 \
bash VeOmni/script.sh --hsdp_dedup=0 --vanilla_momentum=1
```

Static validation passed `make quality` with repository-locked ruff 0.13.2,
explicit ruff check/format for the benchmark helpers, Python compilation,
`bash -n` for both launchers and `git diff --check`.

## Unverified coverage

No CUDA GPU was available for runtime validation. Actual CUDA kernels, NCCL
collectives, GPU overlap performance and CUDA hardware profiling remain
unverified for this revision. No 16-GPU full-size model, physical two-host GPU
experiment or convergence test was run. Earlier accelerator results from a
different backend do not establish CUDA performance or correctness.

Smoke runs use relaxed stability thresholds for control-flow verification;
production defaults remain 5% CV / 3% median drift. Raw logs are retained in the
isolated validation directory shown above.
