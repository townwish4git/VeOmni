#!/usr/bin/env bash
# Move this file beside VeOmni/ and hyper-parallel/ if desired.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../veomni/optim/optimizer.py" ]]; then
    DEFAULT_VEOMNI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    DEFAULT_VEOMNI_ROOT="$SCRIPT_DIR/VeOmni"
fi
VEOMNI_ROOT="$(cd "${VEOMNI_ROOT:-$DEFAULT_VEOMNI_ROOT}" && pwd)"
HYPER_ROOT="$(cd "${HYPER_ROOT:-$(dirname "$VEOMNI_ROOT")/hyper-parallel}" && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# HOSTS enables one-machine coordination; otherwise launch locally with NODE_RANK.
HOSTS="${HOSTS:-}"
NNODES="${NNODES:-2}"; NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
EXPLICIT_NODE_RANK="${NODE_RANK:-}"; NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"; MASTER_PORT="${MASTER_PORT:-29540}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$VEOMNI_ROOT")/muon_step_results}"
EXPLICIT_RUN_ID="${RUN_ID:-}"; RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
SCENARIO="${SCENARIO:-all}"; DEVICE="${DEVICE:-cuda}"; FSDP_BACKEND="${FSDP_BACKEND:-hyper}"
REPEATS="${REPEATS:-3}"; SEED="${SEED:-42}"; BUDGET_MIB="${BUDGET_MIB:-512}"
WARMUP_MIN="${WARMUP_MIN:-20}"; WARMUP_MAX="${WARMUP_MAX:-200}"
STABILITY_WINDOW="${STABILITY_WINDOW:-10}"; STABLE_WINDOWS="${STABLE_WINDOWS:-3}"
STABILITY_CV="${STABILITY_CV:-0.05}"; STABILITY_DRIFT="${STABILITY_DRIFT:-0.03}"
MEASURE_STEPS="${MEASURE_STEPS:-50}"; PROFILE_STEPS="${PROFILE_STEPS:-5}"
PROFILE_WARMUP="${PROFILE_WARMUP:-2}"; PROFILE_RANKS="${PROFILE_RANKS:-all}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
SMOKE="${SMOKE:-0}"; DRY_RUN=0; SUMMARIZE_ONLY=0
SELECTED=(); WORKER_FLAGS=()
while (( $# )); do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --smoke) SMOKE=1; shift ;;
        --summarize-only) SUMMARIZE_ONLY=1; shift ;;
        --profile-memory|--record-shapes|--with-stack) WORKER_FLAGS+=("$1"); shift ;;
        --cases)
            shift
            while (( $# )) && [[ "$1" != --* ]]; do SELECTED+=("$1"); shift; done
            if (( ${#SELECTED[@]} == 0 )); then echo '--cases needs at least one case' >&2; exit 2; fi ;;
        --model-config) MODEL_CONFIG="$2"; shift 2 ;;
        --snapshot-root) SNAPSHOT_ROOT="$2"; shift 2 ;;
        --scenario) SCENARIO="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; EXPLICIT_RUN_ID="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --fsdp-backend) FSDP_BACKEND="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --budget-mib) BUDGET_MIB="$2"; shift 2 ;;
        --warmup-min) WARMUP_MIN="$2"; shift 2 ;;
        --warmup-max) WARMUP_MAX="$2"; shift 2 ;;
        --stability-window) STABILITY_WINDOW="$2"; shift 2 ;;
        --stable-windows) STABLE_WINDOWS="$2"; shift 2 ;;
        --stability-cv) STABILITY_CV="$2"; shift 2 ;;
        --stability-drift) STABILITY_DRIFT="$2"; shift 2 ;;
        --measure-steps) MEASURE_STEPS="$2"; shift 2 ;;
        --profile-steps) PROFILE_STEPS="$2"; shift 2 ;;
        --profile-warmup) PROFILE_WARMUP="$2"; shift 2 ;;
        --profile-ranks) PROFILE_RANKS="$2"; shift 2 ;;
        --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ "$OUTPUT_DIR" != /* ]]; then OUTPUT_DIR="$PWD/$OUTPUT_DIR"; fi
if [[ -n "${MODEL_CONFIG:-}" && "$MODEL_CONFIG" != /* ]]; then MODEL_CONFIG="$PWD/$MODEL_CONFIG"; fi
if [[ -n "${SNAPSHOT_ROOT:-}" && "$SNAPSHOT_ROOT" != /* ]]; then SNAPSHOT_ROOT="$PWD/$SNAPSHOT_ROOT"; fi
SUITE_DIR="$OUTPUT_DIR/$RUN_ID"
if (( SUMMARIZE_ONLY )); then
    exec "${PYTHON:-python3}" -m scripts.muon_step_benchmark.report "$SUITE_DIR"
fi
if [[ -n "$HOSTS" ]]; then
    IFS=, read -r -a HOST_ARRAY <<< "$HOSTS"
    NNODES="${#HOST_ARRAY[@]}"
    for host in "${HOST_ARRAY[@]}"; do
        if [[ -z "$host" || "$host" == -* || "$host" == *[[:space:]]* ]]; then
            echo "Invalid SSH host: $host" >&2; exit 2
        fi
    done
    : "${PYTHON:?Set PYTHON to the Python interpreter path used on every host}"
fi
for value in "$NNODES" "$NPROC_PER_NODE" "$REPEATS" "$MASTER_PORT"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then echo "Expected positive integer: $value" >&2; exit 2; fi
done
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then echo 'SEED must be a nonnegative integer' >&2; exit 2; fi
# Four workers on an eight-GPU host use the back four physical GPUs.
if [[ "$DEVICE" == cuda ]] && (( NPROC_PER_NODE == 4 )); then
    export CUDA_VISIBLE_DEVICES=4,5,6,7
fi
if [[ ! "$NODE_RANK" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then echo 'Invalid NODE_RANK' >&2; exit 2; fi
if [[ "$SCENARIO" != all && "$SCENARIO" != fsdp && "$SCENARIO" != hsdp ]]; then echo 'Invalid SCENARIO' >&2; exit 2; fi
if (( NNODES > 1 )); then
    if [[ -z "$HOSTS" && -z "$EXPLICIT_NODE_RANK" ]]; then
        echo 'Set NODE_RANK explicitly: 0 on the first node, 1 on the second node' >&2; exit 2
    fi
    if [[ -z "$MASTER_ADDR" ]]; then
        echo 'Set MASTER_ADDR to the first node IP, using the same value on both nodes' >&2; exit 2
    fi
    if [[ -z "$HOSTS" && -z "$EXPLICIT_RUN_ID" ]]; then
        echo 'Set the same explicit RUN_ID on every node' >&2; exit 2
    fi
else
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
if (( ! SMOKE && NNODES * NPROC_PER_NODE != 8 && NNODES * NPROC_PER_NODE != 16 )); then
    echo "Expected 8 or 16 total ranks, got $((NNODES * NPROC_PER_NODE)) (NNODES=$NNODES, NPROC_PER_NODE=$NPROC_PER_NODE)" >&2
    exit 2
fi
if [[ "$SCENARIO" != fsdp ]] && (( NNODES * NPROC_PER_NODE % 2 )); then echo 'HSDP needs an even world size' >&2; exit 2; fi
if (( ! SMOKE )) && [[ -z "${MODEL_CONFIG:-}" && -z "${SNAPSHOT_ROOT:-}" ]]; then
    echo 'Set MODEL_CONFIG to the actual training model config directory, or SNAPSHOT_ROOT' >&2; exit 2
fi
if [[ -n "${MODEL_CONFIG:-}" && -n "${SNAPSHOT_ROOT:-}" ]]; then echo 'Choose MODEL_CONFIG or SNAPSHOT_ROOT' >&2; exit 2; fi
CASES=()
for topology in fsdp hsdp; do
    if [[ "$SCENARIO" != all && "$SCENARIO" != "$topology" ]]; then continue; fi
    for dedup in 1 0; do
        if [[ "$topology" == fsdp && "$dedup" == 0 ]]; then continue; fi
        for overlap in 0 1; do
            for momentum in legacy vanilla; do
                name="${topology}_dedup${dedup}_overlap${overlap}_${momentum}"
                if (( ${#SELECTED[@]} )); then
                    matched=0
                    for selected in "${SELECTED[@]}"; do [[ "$selected" != "$name" ]] || matched=1; done
                    (( matched )) || continue
                fi
                CASES+=("$name")
            done
        done
    done
done
if (( ${#SELECTED[@]} )); then
    for selected in "${SELECTED[@]}"; do
        matched=0
        for name in "${CASES[@]}"; do [[ "$name" != "$selected" ]] || matched=1; done
        if (( ! matched )); then echo "Unknown case: $selected" >&2; exit 2; fi
    done
fi
TOTAL=$(( REPEATS * ${#CASES[@]} ))
if (( MASTER_PORT < 1024 || MASTER_PORT + TOTAL > 65535 )); then echo 'Rendezvous port range exceeds bounds' >&2; exit 2; fi
if [[ -n "$HOSTS" ]]; then
    PIDS=(); DONE=()
    if (( ! DRY_RUN )); then
        mkdir -p "$SUITE_DIR/coordinator"
        (set -o noclobber; printf '#!/usr/bin/env bash\n' > "$SUITE_DIR/dispatch.sh") || exit 2
    fi
    stop_nodes() {
        for (( child=0; child<${#PIDS[@]}; child++ )); do
            if (( ! DONE[child] )); then kill -TERM "${PIDS[$child]}" 2>/dev/null || true; fi
        done
    }
    trap 'stop_nodes; exit 130' INT TERM HUP
    for (( node=0; node<NNODES; node++ )); do
        remote=(env HOSTS= NNODES="$NNODES" NODE_RANK="$node" RUN_ID="$RUN_ID"
            VEOMNI_ROOT="$VEOMNI_ROOT" HYPER_ROOT="$HYPER_ROOT" PYTHON="$PYTHON")
        for variable in PYTHONPATH PYTHON_EXEC PATH LD_LIBRARY_PATH NPROC_PER_NODE MASTER_ADDR MASTER_PORT OUTPUT_DIR \
            SCENARIO DEVICE FSDP_BACKEND REPEATS SEED BUDGET_MIB WARMUP_MIN WARMUP_MAX STABILITY_WINDOW STABLE_WINDOWS \
            STABILITY_CV STABILITY_DRIFT MEASURE_STEPS PROFILE_STEPS PROFILE_WARMUP PROFILE_RANKS TIMEOUT_SECONDS \
            SMOKE MODEL_CONFIG SNAPSHOT_ROOT OMP_NUM_THREADS CUDA_VISIBLE_DEVICES NCCL_SOCKET_IFNAME NCCL_IB_HCA \
            NCCL_IB_DISABLE NCCL_ALGO PYTORCH_CUDA_ALLOC_CONF CUDA_LAUNCH_BLOCKING; do
            # Preserve configured empty values as well as nonempty ones.
            if [[ "${!variable+x}" == x ]]; then remote+=("$variable=${!variable}"); fi
        done
        remote+=(bash "$VEOMNI_ROOT/scripts/run_muon_step_st.sh")
        if (( ${#WORKER_FLAGS[@]} )); then remote+=("${WORKER_FLAGS[@]}"); fi
        if (( ${#SELECTED[@]} )); then remote+=(--cases "${SELECTED[@]}"); fi
        printf -v remote_env '%q ' "${remote[@]}"
        printf -v remote_cwd '%q' "$PWD"
        remote_command="cd -- $remote_cwd && exec $remote_env"
        dispatch=(ssh -o BatchMode=yes "${HOST_ARRAY[$node]}" "$remote_command")
        printf '%q ' "${dispatch[@]}"; printf '\n'
        if (( ! DRY_RUN )); then
            printf '%q ' "${dispatch[@]}" >> "$SUITE_DIR/dispatch.sh"; printf '\n' >> "$SUITE_DIR/dispatch.sh"
            "${dispatch[@]}" > "$SUITE_DIR/coordinator/node${node}.log" 2>&1 &
            PIDS+=("$!"); DONE+=(0)
        fi
    done
    if (( DRY_RUN )); then exit 0; fi
    failed=0; remaining="${#PIDS[@]}"
    while (( remaining )); do
        for (( node=0; node<${#PIDS[@]}; node++ )); do
            if (( DONE[node] )); then continue; fi
            if kill -0 "${PIDS[$node]}" 2>/dev/null; then continue; fi
            if ! wait "${PIDS[$node]}"; then failed=1; break; fi
            DONE[$node]=1; remaining=$((remaining - 1))
        done
        if (( failed )); then stop_nodes; break; fi
        if (( remaining )); then sleep 1; fi
    done
    trap - INT TERM HUP
    for (( node=0; node<NNODES; node++ )); do
        printf -v remote_directory '%q' "$SUITE_DIR"
        rsync -a --ignore-existing "${HOST_ARRAY[$node]}:$remote_directory/" "$SUITE_DIR/" || failed=1
    done
    if (( failed )); then echo "Node execution or result retrieval failed; see $SUITE_DIR/coordinator" >&2; exit 1; fi
    exec "$PYTHON" -m scripts.muon_step_benchmark.report "$SUITE_DIR"
fi
PLAN="$SUITE_DIR/plan-node${NODE_RANK}.sh"
if (( ! DRY_RUN )); then
    mkdir -p "$SUITE_DIR"
    (set -o noclobber; printf '#!/usr/bin/env bash\n' > "$PLAN") || exit 2
    if [[ "$DEVICE" == cuda ]] && (( NPROC_PER_NODE == 4 )); then
        printf 'export CUDA_VISIBLE_DEVICES=4,5,6,7\n' >> "$PLAN"
    fi
fi
LAUNCHER_PID=
stop_local_job() {
    if [[ -n "$LAUNCHER_PID" ]]; then kill -TERM "$LAUNCHER_PID" 2>/dev/null || true; fi
    exit 130
}
trap stop_local_job INT TERM HUP
index=0
for (( trial=0; trial<REPEATS; trial++ )); do
    ORDER=("${CASES[@]}")
    state=$(( (SEED + trial) & 2147483647 ))
    for (( i=${#ORDER[@]}-1; i>0; i-- )); do
        state=$(( (1103515245 * state + 12345) & 2147483647 ))
        j=$(( state % (i + 1) )); swap="${ORDER[$i]}"; ORDER[$i]="${ORDER[$j]}"; ORDER[$j]="$swap"
    done
    for name in "${ORDER[@]}"; do
        IFS=_ read -r topology dedup_token overlap_token momentum <<< "$name"
        dedup="${dedup_token#dedup}"; overlap="${overlap_token#overlap}"
        config_cards=16
        if (( NNODES * NPROC_PER_NODE == 8 )); then config_cards=8; fi
        if [[ "$topology" == fsdp ]]; then config="$VEOMNI_ROOT/configs/hyperparallel/qwen3_muon_2k_${config_cards}.yaml";
        else config="$VEOMNI_ROOT/configs/hyperparallel/qwen3_muon_128k_hsdp_${config_cards}.yaml"; fi
        printf -v trial_tag 'r%02d' "$trial"
        case_dir="$SUITE_DIR/${trial_tag}_${name}"
        command=("${PYTHON:-python3}" -m torch.distributed.run --nnodes="$NNODES" --nproc-per-node="$NPROC_PER_NODE" --node-rank="$NODE_RANK"
            --master-addr="$MASTER_ADDR" --master-port="$((MASTER_PORT + index))" --max-restarts=0
            --log-dir "$case_dir/torchrun/node${NODE_RANK}" --tee=3 "$VEOMNI_ROOT/tasks/train_muon_step.py" "$config"
            --veomni-root "$VEOMNI_ROOT" --hyper-root "$HYPER_ROOT" --output "$case_dir"
            --case "$name" --scenario "$topology" --momentum "$momentum" --dedup "$dedup" --overlap "$overlap"
            --trial "$trial" --device "$DEVICE" --fsdp-backend "$FSDP_BACKEND" --seed "$SEED" --budget-mib "$BUDGET_MIB"
            --warmup-min "$WARMUP_MIN" --warmup-max "$WARMUP_MAX" --stability-window "$STABILITY_WINDOW"
            --stable-windows "$STABLE_WINDOWS" --stability-cv "$STABILITY_CV" --stability-drift "$STABILITY_DRIFT"
            --measure-steps "$MEASURE_STEPS" --profile-steps "$PROFILE_STEPS" --profile-warmup "$PROFILE_WARMUP"
            --profile-ranks "$PROFILE_RANKS" --timeout-seconds "$TIMEOUT_SECONDS")
        if [[ -n "${MODEL_CONFIG:-}" ]]; then command+=(--model-config "$MODEL_CONFIG"); fi
        if [[ -n "${SNAPSHOT_ROOT:-}" ]]; then command+=(--snapshot-root "$SNAPSHOT_ROOT"); fi
        if (( SMOKE )); then command+=(--smoke); fi
        if (( ${#WORKER_FLAGS[@]} )); then command+=("${WORKER_FLAGS[@]}"); fi
        invocation=("${command[@]}")
        printf '%q ' "${invocation[@]}"; printf '\n'
        if (( ! DRY_RUN )); then
            printf '%q ' "${invocation[@]}" >> "$PLAN"; printf '\n' >> "$PLAN"
            "${invocation[@]}" &
            LAUNCHER_PID=$!
            if wait "$LAUNCHER_PID"; then status=0; else status=$?; fi
            LAUNCHER_PID=
            if (( status )); then exit "$status"; fi
        fi
        index=$((index + 1))
    done
done
if (( ! DRY_RUN && NODE_RANK == 0 )); then
    "${PYTHON:-python3}" -m scripts.muon_step_benchmark.report "$SUITE_DIR"
fi
