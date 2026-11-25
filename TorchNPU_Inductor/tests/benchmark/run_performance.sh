#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run_performance.sh --model PATH --workspace PATH --results PATH --sharegpt PATH [options]

Options:
  --devices IDS   Comma-separated NPU IDs (default: 0)
  --tp N          Tensor parallel size; must match the number of devices (default: 1)
  --cache-root PATH  Existing cache directory containing default/ and inductor/
  --enable-prefix-caching  Enable prefix caching instead of disabling it
  --enable-expert-parallel  Enable expert parallel
  --quantization NAME  Override quantization method
  --trust-remote-code  Allow checkpoint-provided model code
  --kv-cache-dtype TYPE  Override KV Cache dtype, for example int8
  --tokenizer-mode MODE  Override tokenizer mode
  --max-model-len N  Override maximum model context length
  --gpu-memory-utilization N  Override vLLM memory utilization fraction
  --repeats N     Number of default/Inductor rounds (default: 3)
  --capture-size N  Max cudagraph capture size in tokens for BOTH tracks (default: vLLM's 32).
                   Expands to sizes [1,2,4,8,16,24,32] then powers of two up to N.
  --qps LIST      Comma-separated ShareGPT request rates (default: 1,4,16,inf).
                   Use 'none' to skip ShareGPT loads.
  --skip-long     Skip the random 8192/32768 long-context loads.
  --timeout SEC   Timeout for each server process, including its startup wait (default: 21600)
  -h, --help      Show this help
EOF
}

MODEL=
WORKSPACE=
ROOT=
SHAREGPT=
DEVICES=0
TP=1
CACHE_ROOT=
PREFIX_CACHING=--no-enable-prefix-caching
EXTRA_ARGS=()
REPEATS=3
TIMEOUT=21600
CAPTURE_SIZE=
CAPTURE_ARGS=()
QPS_LIST="1 4 16 inf"
SKIP_LONG=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL=$2; shift 2 ;;
    --workspace) WORKSPACE=$2; shift 2 ;;
    --results) ROOT=$2; shift 2 ;;
    --sharegpt) SHAREGPT=$2; shift 2 ;;
    --devices) DEVICES=$2; shift 2 ;;
    --tp) TP=$2; shift 2 ;;
    --cache-root) CACHE_ROOT=$2; shift 2 ;;
    --enable-prefix-caching) PREFIX_CACHING=--enable-prefix-caching; shift ;;
    --enable-expert-parallel) EXTRA_ARGS+=(--enable-expert-parallel); shift ;;
    --quantization) EXTRA_ARGS+=(--quantization "$2"); shift 2 ;;
    --trust-remote-code) EXTRA_ARGS+=(--trust-remote-code); shift ;;
    --kv-cache-dtype) EXTRA_ARGS+=(--kv-cache-dtype "$2"); shift 2 ;;
    --tokenizer-mode) EXTRA_ARGS+=(--tokenizer-mode "$2"); shift 2 ;;
    --max-model-len) EXTRA_ARGS+=(--max-model-len "$2"); shift 2 ;;
    --gpu-memory-utilization) EXTRA_ARGS+=(--gpu-memory-utilization "$2"); shift 2 ;;
    --repeats) REPEATS=$2; shift 2 ;;
    --capture-size) CAPTURE_SIZE=$2; shift 2 ;;
    --qps) QPS_LIST=${2//,/ }; shift 2 ;;
    --skip-long) SKIP_LONG=1; shift ;;
    --timeout) TIMEOUT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n $MODEL && -n $WORKSPACE && -n $ROOT && -n $SHAREGPT ]] || { usage >&2; exit 2; }
[[ $ROOT = /* && ! -e $ROOT ]] || { echo 'Require a new absolute result directory' >&2; exit 2; }
[[ -f $MODEL/config.json ]] || { echo "Missing model config: $MODEL/config.json" >&2; exit 2; }
[[ -f $WORKSPACE/env.sh ]] || { echo "Missing workspace environment: $WORKSPACE/env.sh" >&2; exit 2; }
[[ -f $SHAREGPT ]] || { echo "Missing ShareGPT dataset: $SHAREGPT" >&2; exit 2; }
[[ $TP =~ ^[1-9][0-9]*$ && $REPEATS =~ ^[1-9][0-9]*$ && $TIMEOUT =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid TP, repeats or timeout' >&2; exit 2; }
# Build the capture-size ladder only when asked, so an unset --capture-size leaves
# vLLM's defaults untouched (Default then sends no --compilation-config at all).
if [[ -n $CAPTURE_SIZE ]]; then
  [[ $CAPTURE_SIZE =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid capture size' >&2; exit 2; }
  LADDER="1 2 4 8 16 24 32"
  for p in 64 128 256 512 1024 2048 4096; do
    (( p <= CAPTURE_SIZE )) || break
    LADDER="$LADDER $p"
  done
  [[ $CAPTURE_SIZE == 32 || " $LADDER " == *" $CAPTURE_SIZE "* ]] || { echo '--capture-size must be 32 or a power of two >= 64' >&2; exit 2; }
  CAPTURE_ARGS=(--max-cudagraph-capture-size "$CAPTURE_SIZE" --cudagraph-capture-sizes $LADDER)
fi
IFS=',' read -r -a CARDS <<< "$DEVICES"
[[ ${#CARDS[@]} -eq $TP ]] || { echo 'The number of devices must equal TP' >&2; exit 2; }

SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$WORKSPACE/env.sh" || exit 2
PYTHON=${PY:-python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1} PYTHONUNBUFFERED=1
mkdir -p "$ROOT/performance" "$ROOT/_locks"
CACHE_ROOT=${CACHE_ROOT:-$ROOT/cache}
LOCKS=$ROOT/_locks
SERVER_PID=
BASE_URL=
COMPARISONS=()

"$PYTHON" "$SCRIPTS/runtime.py" check-devices --devices "$DEVICES"
"$PYTHON" "$SCRIPTS/runtime.py" snapshot --workspace "$WORKSPACE" --out "$ROOT/environment.json"

free_port() {
  "$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'
}

stop_server() {
  if [[ -n $SERVER_PID ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
  fi
}
trap 'exit 130' INT TERM
trap stop_server EXIT

start_server() {
  local track=$1 round=$2 port
  port=$(free_port)
  BASE_URL=http://127.0.0.1:$port
  local prefix=$ROOT/performance/$track.repeat$round
  "$PYTHON" "$SCRIPTS/runtime.py" exec --workspace "$WORKSPACE" \
    --devices "$DEVICES" --lock-dir "$LOCKS" --cache-dir "$CACHE_ROOT/$track" \
    --out "$prefix.runtime.json" --timeout "$TIMEOUT" -- \
    "$PYTHON" "$SCRIPTS/server.py" run --model "$MODEL" --track "$track" \
    --port "$port" --tp "$TP" "$PREFIX_CACHING" "${CAPTURE_ARGS[@]}" "${EXTRA_ARGS[@]}" \
    --metadata "$prefix.server.json" -- --seed 0 > "$prefix.server.log" 2>&1 &
  SERVER_PID=$!
  "$PYTHON" "$SCRIPTS/server.py" wait --base-url "$BASE_URL" --server-pid "$SERVER_PID" --timeout "$TIMEOUT"
}

run_loads() {
  local track=$1 round=$2 qps input_len
  [[ $QPS_LIST == none ]] || for qps in $QPS_LIST; do
    "$PYTHON" "$SCRIPTS/performance.py" run --model "$MODEL" --base-url "$BASE_URL" \
      --dataset-name sharegpt --dataset-path "$SHAREGPT" --request-rate "$qps" \
      --out "$ROOT/performance/$track/sharegpt_qps_${qps}_repeat${round}.json"
  done
  [[ $SKIP_LONG == 1 ]] || for input_len in 8192 32768; do
    "$PYTHON" "$SCRIPTS/performance.py" run --model "$MODEL" --base-url "$BASE_URL" \
      --dataset-name random --random-input-len "$input_len" --request-rate inf \
      --out "$ROOT/performance/$track/random_${input_len}_1024_repeat${round}.json"
  done
}

compare_loads() {
  local round=$1 qps input_len output
  [[ $QPS_LIST == none ]] || for qps in $QPS_LIST; do
    output=$ROOT/performance/compare/sharegpt_qps_${qps}_repeat${round}.json
    "$PYTHON" "$SCRIPTS/performance.py" compare \
      --reference "$ROOT/performance/default/sharegpt_qps_${qps}_repeat${round}.json" \
      --candidate "$ROOT/performance/inductor/sharegpt_qps_${qps}_repeat${round}.json" \
      --out "$output"
    COMPARISONS+=("$output")
  done
  [[ $SKIP_LONG == 1 ]] || for input_len in 8192 32768; do
    output=$ROOT/performance/compare/random_${input_len}_1024_repeat${round}.json
    "$PYTHON" "$SCRIPTS/performance.py" compare \
      --reference "$ROOT/performance/default/random_${input_len}_1024_repeat${round}.json" \
      --candidate "$ROOT/performance/inductor/random_${input_len}_1024_repeat${round}.json" \
      --out "$output"
    COMPARISONS+=("$output")
  done
}

for ((round=1; round<=REPEATS; round++)); do
  for track in default inductor; do
    echo "PERFORMANCE_START model=$MODEL round=$round track=$track devices=$DEVICES"
    start_server "$track" "$round"
    run_loads "$track" "$round"
    stop_server
    if [[ $track == inductor ]]; then
      evidence=$ROOT/performance/inductor.repeat$round.evidence.json
      "$PYTHON" "$SCRIPTS/runtime.py" evidence \
        --run-config "$ROOT/performance/inductor.repeat$round.server.json" \
        --log "$ROOT/performance/inductor.repeat$round.server.log" \
        --cache-dir "$CACHE_ROOT/inductor" --out "$evidence"
      COMPARISONS+=("$evidence")
    fi
  done
  compare_loads "$round"
done
"$PYTHON" "$SCRIPTS/report.py" "${COMPARISONS[@]}" --out "$ROOT/performance/SUMMARY.md"
cat "$ROOT/performance/SUMMARY.md"
