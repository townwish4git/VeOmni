#!/usr/bin/env bash
# Run one Muon experiment locally; run the same command on every node.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/veomni" ]]; then
    DEFAULT_VEOMNI_ROOT="$SCRIPT_DIR"
else
    DEFAULT_VEOMNI_ROOT="$SCRIPT_DIR/VeOmni"
fi
VEOMNI_ROOT="$(cd "${VEOMNI_ROOT:-$DEFAULT_VEOMNI_ROOT}" && pwd)"
HYPER_ROOT="$(cd "${HYPER_ROOT:-$(dirname "$VEOMNI_ROOT")/hyper-parallel}" && pwd)"
SCENARIO="${SCENARIO:-hsdp}"
DEVICE="${DEVICE:-cuda}"
HSDP_DEDUP="${HSDP_DEDUP:-1}"
VANILLA_MOMENTUM="${VANILLA_MOMENTUM:-0}"
FSDP_OVERLAP="${FSDP_OVERLAP:-0}"
DRY_RUN=0
SMOKE="${SMOKE:-0}"
EXTRA=()
while (( $# )); do
    case "$1" in
        --hsdp_dedup=*) HSDP_DEDUP="${1#*=}"; shift ;;
        --vanilla_momentum=*) VANILLA_MOMENTUM="${1#*=}"; shift ;;
        --fsdp_overlap=*) FSDP_OVERLAP="${1#*=}"; shift ;;
        --scenario=*) SCENARIO="${1#*=}"; shift ;;
        --hsdp_dedup) HSDP_DEDUP="$2"; shift 2 ;;
        --vanilla_momentum) VANILLA_MOMENTUM="$2"; shift 2 ;;
        --fsdp_overlap) FSDP_OVERLAP="$2"; shift 2 ;;
        --scenario) SCENARIO="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --smoke) SMOKE=1; shift ;;
        --dedup|--dedup=*|--momentum|--momentum=*|--overlap|--overlap=*|--case|--case=*|--output|--output=*)
            echo 'Use --hsdp_dedup, --vanilla_momentum, --fsdp_overlap and OUTPUT_DIR to name the experiment consistently' >&2
            exit 2 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done
for value in "$HSDP_DEDUP" "$VANILLA_MOMENTUM" "$FSDP_OVERLAP" "$SMOKE"; do
    if [[ "$value" != 0 && "$value" != 1 ]]; then echo "Switches must be 0 or 1, got $value" >&2; exit 2; fi
done
case "$SCENARIO" in
    hsdp) DEFAULT_CONFIG="$VEOMNI_ROOT/configs/hyperparallel/qwen3_muon_128k_hsdp_16.yaml" ;;
    fsdp) DEFAULT_CONFIG="$VEOMNI_ROOT/configs/hyperparallel/qwen3_muon_2k_16.yaml" ;;
    *) echo 'SCENARIO must be hsdp or fsdp' >&2; exit 2 ;;
esac
CONFIG_FILE="${CONFIG_FILE:-$DEFAULT_CONFIG}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-${node_rank:-}}"
MODEL_CONFIG="${MODEL_CONFIG:-${model_config:-}}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29540}"
for value in "$NNODES" "$NPROC_PER_NODE" "$MASTER_PORT"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then echo "Expected positive integer, got $value" >&2; exit 2; fi
done
if (( NNODES == 1 )); then NODE_RANK="${NODE_RANK:-0}"; MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"; fi
: "${NODE_RANK:?Set NODE_RANK=0 on the first node and NODE_RANK=1 on the second node}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the first node communication IP on both nodes}"
if [[ ! "$NODE_RANK" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then echo 'Invalid NODE_RANK' >&2; exit 2; fi
if (( ! SMOKE )) && [[ -z "$MODEL_CONFIG" && -z "${SNAPSHOT_ROOT:-}" ]]; then
    echo 'Set MODEL_CONFIG to the actual training model config directory' >&2; exit 2
fi
if [[ -n "$MODEL_CONFIG" && -n "${SNAPSHOT_ROOT:-}" ]]; then echo 'Choose MODEL_CONFIG or SNAPSHOT_ROOT' >&2; exit 2; fi
for variable in CONFIG_FILE MODEL_CONFIG SNAPSHOT_ROOT; do
    if [[ -n "${!variable:-}" && "${!variable}" != /* ]]; then printf -v "$variable" '%s' "$PWD/${!variable}"; fi
done
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MOMENTUM=legacy
if (( VANILLA_MOMENTUM )); then MOMENTUM=vanilla; fi
CASE_NAME="${SCENARIO}_dedup${HSDP_DEDUP}_overlap${FSDP_OVERLAP}_${MOMENTUM}"
RUN_ID="${RUN_ID:-run0}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$VEOMNI_ROOT")/muon_step_results}"
if [[ "$OUTPUT_DIR" != /* ]]; then OUTPUT_DIR="$PWD/$OUTPUT_DIR"; fi
CASE_DIR="$OUTPUT_DIR/$CASE_NAME/$RUN_ID"
command=("${TORCHRUN:-torchrun}" --nnodes="$NNODES" --nproc-per-node="$NPROC_PER_NODE" --node-rank="$NODE_RANK"
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --max-restarts=0
    --log-dir "$CASE_DIR/torchrun/node${NODE_RANK}" --tee=3
    "$VEOMNI_ROOT/tasks/train_muon_step.py" "$CONFIG_FILE"
    --veomni-root "$VEOMNI_ROOT" --hyper-root "$HYPER_ROOT" --scenario "$SCENARIO"
    --case "$CASE_NAME" --output "$CASE_DIR" --dedup "$HSDP_DEDUP" --overlap "$FSDP_OVERLAP" --momentum "$MOMENTUM")
# Environment overrides are explicit argv pairs, never evaluated shell text.
for mapping in MODEL_CONFIG:model-config SNAPSHOT_ROOT:snapshot-root DEVICE:device FSDP_BACKEND:fsdp-backend \
    SEED:seed BUDGET_MIB:budget-mib WARMUP_MIN:warmup-min WARMUP_MAX:warmup-max \
    STABILITY_WINDOW:stability-window STABLE_WINDOWS:stable-windows STABILITY_CV:stability-cv \
    STABILITY_DRIFT:stability-drift MEASURE_STEPS:measure-steps PROFILE_STEPS:profile-steps \
    PROFILE_WARMUP:profile-warmup PROFILE_RANKS:profile-ranks TIMEOUT_SECONDS:timeout-seconds TRIAL:trial; do
    variable="${mapping%%:*}"; option="${mapping#*:}"
    if [[ -n "${!variable:-}" ]]; then command+=("--$option" "${!variable}"); fi
done
if (( SMOKE )); then command+=(--smoke); fi
if (( ${#EXTRA[@]} )); then command+=("${EXTRA[@]}"); fi
printf 'Result directory: %s\n' "$CASE_DIR"
printf '%q ' "${command[@]}"; printf '\n'
if (( DRY_RUN )); then exit 0; fi
mkdir -p "$CASE_DIR"
# Per-node markers protect repeat data while allowing both nodes on shared storage.
PLAN="$CASE_DIR/launch-node${NODE_RANK}.sh"
if ! (set -o noclobber; printf '#!/usr/bin/env bash\n' > "$PLAN") 2>/dev/null; then
    echo "Existing experiment for this node: $PLAN. Set a new RUN_ID on both nodes." >&2; exit 2
fi
printf '%q ' "${command[@]}" >> "$PLAN"; printf '\n' >> "$PLAN"
exec "${command[@]}"
