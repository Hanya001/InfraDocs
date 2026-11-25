#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run_correctness.sh --model PATH --workspace PATH --results PATH [options]

Options:
  --devices IDS   Comma-separated NPU IDs (default: 0)
  --tp N          Tensor parallel size per DP rank (default: 1)
  --dp N          Data parallel size; devices must equal TP * DP (default: 1)
  --cache-root PATH  Existing cache directory containing eager/ and inductor/
  --embedding     Generate and compare embeddings instead of text
  --enable-prefix-caching  Enable prefix caching instead of disabling it
  --enable-expert-parallel  Enable expert parallel
  --quantization NAME  Override quantization method
  --trust-remote-code  Allow checkpoint-provided model code
  --kv-cache-dtype TYPE  Override KV Cache dtype, for example int8
  --tokenizer-mode MODE  Override tokenizer mode
  --max-model-len N  Override maximum model context length
  --gpu-memory-utilization N  Override vLLM memory utilization fraction
  --repeats N     Generation repetitions for each track (default: 2)
  --timeout SEC   Timeout for each engine process (default: 7200)
  -h, --help      Show this help
EOF
}

MODEL=
WORKSPACE=
ROOT=
DEVICES=0
TP=1
DP=1
CACHE_ROOT=
OPERATION=generate
PREFIX_CACHING=--no-enable-prefix-caching
EXTRA_ARGS=()
REPEATS=2
TIMEOUT=7200
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL=$2; shift 2 ;;
    --workspace) WORKSPACE=$2; shift 2 ;;
    --results) ROOT=$2; shift 2 ;;
    --devices) DEVICES=$2; shift 2 ;;
    --tp) TP=$2; shift 2 ;;
    --dp) DP=$2; shift 2 ;;
    --cache-root) CACHE_ROOT=$2; shift 2 ;;
    --embedding) OPERATION=embedding; shift ;;
    --enable-prefix-caching) PREFIX_CACHING=--enable-prefix-caching; shift ;;
    --enable-expert-parallel) EXTRA_ARGS+=(--enable-expert-parallel); shift ;;
    --quantization) EXTRA_ARGS+=(--quantization "$2"); shift 2 ;;
    --trust-remote-code) EXTRA_ARGS+=(--trust-remote-code); shift ;;
    --kv-cache-dtype) EXTRA_ARGS+=(--kv-cache-dtype "$2"); shift 2 ;;
    --tokenizer-mode) EXTRA_ARGS+=(--tokenizer-mode "$2"); shift 2 ;;
    --max-model-len) EXTRA_ARGS+=(--max-model-len "$2"); shift 2 ;;
    --gpu-memory-utilization) EXTRA_ARGS+=(--gpu-memory-utilization "$2"); shift 2 ;;
    --repeats) REPEATS=$2; shift 2 ;;
    --timeout) TIMEOUT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n $MODEL && -n $WORKSPACE && -n $ROOT ]] || { usage >&2; exit 2; }
[[ $ROOT = /* && ! -e $ROOT ]] || { echo 'Require a new absolute result directory' >&2; exit 2; }
[[ -f $MODEL/config.json ]] || { echo "Missing model config: $MODEL/config.json" >&2; exit 2; }
[[ -f $WORKSPACE/env.sh ]] || { echo "Missing workspace environment: $WORKSPACE/env.sh" >&2; exit 2; }
[[ $TP =~ ^[1-9][0-9]*$ && $DP =~ ^[1-9][0-9]*$ && $REPEATS =~ ^[1-9][0-9]*$ && $TIMEOUT =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid TP, DP, repeats or timeout' >&2; exit 2; }
[[ $DP -eq 1 || $OPERATION != embedding ]] || { echo '--embedding requires DP=1' >&2; exit 2; }
IFS=',' read -r -a CARDS <<< "$DEVICES"
[[ ${#CARDS[@]} -eq $((TP * DP)) ]] || { echo 'The number of devices must equal TP * DP' >&2; exit 2; }

SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$WORKSPACE/env.sh" || exit 2
PYTHON=${PY:-python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1} PYTHONUNBUFFERED=1
mkdir -p "$ROOT/correctness" "$ROOT/_locks"
CACHE_ROOT=${CACHE_ROOT:-$ROOT/cache}
LOCKS=$ROOT/_locks
SERVER_PID=
BASE_URL=
EVIDENCE_ARGS=()

"$PYTHON" "$SCRIPTS/runtime.py" check-devices --devices "$DEVICES"
"$PYTHON" "$SCRIPTS/runtime.py" snapshot --workspace "$WORKSPACE" --out "$ROOT/environment.json"

free_port() {
  "$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'
}

wait_for_device_release() {
  local deadline=$((SECONDS + 300))
  while (( SECONDS < deadline )); do
    if "$PYTHON" "$SCRIPTS/runtime.py" check-devices --devices "$DEVICES" >/dev/null 2>&1; then
      echo "DEVICES_RELEASED $DEVICES"
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for NPU devices to be released: $DEVICES" >&2
  "$PYTHON" "$SCRIPTS/runtime.py" check-devices --devices "$DEVICES"
}

stop_server() {
  if [[ -n $SERVER_PID ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
    wait_for_device_release
  fi
}
trap 'exit 130' INT TERM
trap stop_server EXIT

start_server() {
  local track=$1 port
  port=$(free_port)
  BASE_URL=http://127.0.0.1:$port
  local prefix=$ROOT/correctness/$track
  "$PYTHON" "$SCRIPTS/runtime.py" exec --workspace "$WORKSPACE" \
    --devices "$DEVICES" --lock-dir "$LOCKS" --cache-dir "$CACHE_ROOT/$track" \
    --out "$prefix.runtime.json" --timeout "$TIMEOUT" -- \
    "$PYTHON" "$SCRIPTS/server.py" run --model "$MODEL" --track "$track" \
    --port "$port" --tp "$TP" --dp "$DP" --max-model-len 8192 "$PREFIX_CACHING" "${EXTRA_ARGS[@]}" \
    --metadata "$prefix.server.json" -- --seed 0 > "$prefix.server.log" 2>&1 &
  SERVER_PID=$!
  "$PYTHON" "$SCRIPTS/server.py" wait --base-url "$BASE_URL" --server-pid "$SERVER_PID" --timeout "$TIMEOUT"
}

for track in eager inductor; do
  echo "CORRECTNESS_START model=$MODEL track=$track devices=$DEVICES"
  if [[ $DP -eq 1 ]]; then
    "$PYTHON" "$SCRIPTS/runtime.py" exec --workspace "$WORKSPACE" \
      --devices "$DEVICES" --lock-dir "$LOCKS" --cache-dir "$CACHE_ROOT/$track" \
      --out "$ROOT/correctness/$track.runtime.json" --timeout "$TIMEOUT" -- \
      "$PYTHON" "$SCRIPTS/correctness.py" "$OPERATION" --model "$MODEL" --track "$track" \
      --tp "$TP" --dp "$DP" --repeats "$REPEATS" --out "$ROOT/correctness/$track" "$PREFIX_CACHING" "${EXTRA_ARGS[@]}"
    continue
  fi
  start_server "$track"
  "$PYTHON" "$SCRIPTS/correctness.py" generate --model "$MODEL" --track "$track" \
    --tp "$TP" --dp "$DP" --repeats "$REPEATS" --out "$ROOT/correctness/$track" \
    --base-url "$BASE_URL" --server-metadata "$ROOT/correctness/$track.server.json" \
    --request-timeout "$TIMEOUT"
  stop_server
done

if [[ $DP -gt 1 ]]; then
  EVIDENCE_ARGS=(--log "$ROOT/correctness/inductor.server.log")
fi
"$PYTHON" "$SCRIPTS/runtime.py" evidence \
  --run-config "$ROOT/correctness/inductor/run_config.json" \
  "${EVIDENCE_ARGS[@]}" \
  --cache-dir "$CACHE_ROOT/inductor" --out "$ROOT/correctness/inductor_evidence.json"

TYPE=generation
FILE=acc_track.jsonl
if [[ $OPERATION == embedding ]]; then
  TYPE=embedding
  FILE=embeddings.json
fi
"$PYTHON" "$SCRIPTS/correctness.py" compare --type "$TYPE" \
  --reference "$ROOT/correctness/eager/$FILE" \
  --candidate "$ROOT/correctness/inductor/$FILE" \
  --reference-label eager --candidate-label inductor \
  --out "$ROOT/correctness/compare.json"
"$PYTHON" "$SCRIPTS/report.py" "$ROOT/correctness/compare.json" \
  "$ROOT/correctness/inductor_evidence.json" --out "$ROOT/correctness/SUMMARY.md"
cat "$ROOT/correctness/SUMMARY.md"
