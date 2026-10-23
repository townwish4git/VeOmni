#!/usr/bin/env bash
# Copy this file to the common parent of VeOmni/ and hyper-parallel/ if desired.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../veomni/optim/optimizer.py" ]]; then
    DEFAULT_VEOMNI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    DEFAULT_VEOMNI_ROOT="$SCRIPT_DIR/VeOmni"
fi
VEOMNI_ROOT="${VEOMNI_ROOT:-$DEFAULT_VEOMNI_ROOT}"
HYPER_ROOT="${HYPER_ROOT:-$(dirname "$VEOMNI_ROOT")/hyper-parallel}"
VEOMNI_ROOT="$(cd "$VEOMNI_ROOT" && pwd)"
HYPER_ROOT="$(cd "$HYPER_ROOT" && pwd)"
PYTHON="${PYTHON:-python3}"
if [[ "$PYTHON" == */* && "$PYTHON" != /* ]]; then PYTHON="$PWD/$PYTHON"; fi
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29540}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$VEOMNI_ROOT")/muon_step_results}"
EXPLICIT_RUN_ID="${RUN_ID:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
# Each node must use the same explicit RUN_ID for manual multi-node launch.
if [[ "$NNODES" != 1 && -z "$EXPLICIT_RUN_ID" ]]; then
    echo 'For manual multi-node launch set RUN_ID identically on every node.' >&2
    exit 2
fi
if [[ ! -f "$VEOMNI_ROOT/scripts/muon_step_benchmark/run.py" ]]; then
    echo "Benchmark not found in VEOMNI_ROOT=$VEOMNI_ROOT" >&2
    exit 2
fi
if [[ "$OUTPUT_DIR" != /* ]]; then OUTPUT_DIR="$PWD/$OUTPUT_DIR"; fi
if [[ -n "${MODEL_CONFIG:-}" && "$MODEL_CONFIG" != /* ]]; then MODEL_CONFIG="$PWD/$MODEL_CONFIG"; fi
if [[ -n "${SNAPSHOT_ROOT:-}" && "$SNAPSHOT_ROOT" != /* ]]; then SNAPSHOT_ROOT="$PWD/$SNAPSHOT_ROOT"; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
cd "$VEOMNI_ROOT"
command=("$PYTHON" -m scripts.muon_step_benchmark.run
    --veomni-root "$VEOMNI_ROOT" --repo-root "$VEOMNI_ROOT" --hyper-root "$HYPER_ROOT"
    --python "$PYTHON" --output "$OUTPUT_DIR" --run-id "$RUN_ID"
    --nnodes "$NNODES" --node-rank "$NODE_RANK" --nproc-per-node "$NPROC_PER_NODE"
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --hccl-base-port "${HCCL_BASE_PORT:-40000}"
    --scenario "${SCENARIO:-all}" --device "${DEVICE:-npu}"
    --fsdp-backend "${FSDP_BACKEND:-hyper}" --repeats "${REPEATS:-3}"
    --warmup-min "${WARMUP_MIN:-20}" --warmup-max "${WARMUP_MAX:-200}"
    --measure-steps "${MEASURE_STEPS:-50}" --profile-steps "${PROFILE_STEPS:-5}"
    --budget-mib "${BUDGET_MIB:-512}")
if [[ -n "${MODEL_CONFIG:-}" ]]; then command+=(--model-config "$MODEL_CONFIG"); fi
if [[ -n "${SNAPSHOT_ROOT:-}" ]]; then command+=(--snapshot-root "$SNAPSHOT_ROOT"); fi
if [[ -n "${HOSTS:-}" ]]; then command+=(--hosts "$HOSTS"); fi
# Native CLI options appended last override the defaults above.
exec "${command[@]}" "$@"
