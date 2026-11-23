#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run_accuracy.sh --model PATH --workspace PATH --results PATH \
         --aisbench-home PATH --mmlu PATH --gsm8k PATH [options]

Options:
  --devices IDS   Comma-separated NPU IDs (default: 0)
  --tp N          Tensor parallel size; must match the number of devices (default: 1)
  --cache-root PATH  Existing cache directory containing eager/ and inductor/
  --tasks LIST    Comma-separated AISBench tasks (default: mmlu,gsm8k)
  --num-prompts N Prompts per task (default: 200)
  --fixed-data-root PATH  Reuse fixed200/fixed1000_500 without sampling
  --mmlu-num-prompts N  Override MMLU prompt count
  --gsm8k-num-prompts N  Override GSM8K prompt count
  --gpqa-dataset PATH  Directory containing gpqa_diamond.csv
  --gpqa-num-prompts N  GPQA Diamond prompt count (must be 198)
  --allow-response-anomalies  Record anomaly metrics and continue remaining tasks
  --enable-prefix-caching  Enable prefix caching instead of disabling it
  --enable-expert-parallel  Enable expert parallel
  --quantization NAME  Override quantization method
  --trust-remote-code  Allow checkpoint-provided model code
  --kv-cache-dtype TYPE  Override KV Cache dtype, for example int8
  --tokenizer-mode MODE  Override tokenizer mode
  --max-model-len N  Override maximum model context length
  --gpu-memory-utilization N  Override vLLM memory utilization fraction
  --timeout SEC   Timeout for each server process (default: 10800)
  -h, --help      Show this help
EOF
}

MODEL=
WORKSPACE=
ROOT=
AISBENCH_HOME=
MMLU=
GSM8K=
DEVICES=0
TP=1
CACHE_ROOT=
TASKS=mmlu,gsm8k
NUM_PROMPTS=200
FIXED_DATA_ROOT=
MMLU_NUM_PROMPTS=
GSM8K_NUM_PROMPTS=
GPQA_DATASET=
GPQA_NUM_PROMPTS=
ALLOW_RESPONSE_ANOMALIES=
PREFIX_CACHING=--no-enable-prefix-caching
EXTRA_ARGS=()
TIMEOUT=10800
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL=$2; shift 2 ;;
    --workspace) WORKSPACE=$2; shift 2 ;;
    --results) ROOT=$2; shift 2 ;;
    --aisbench-home) AISBENCH_HOME=$2; shift 2 ;;
    --mmlu) MMLU=$2; shift 2 ;;
    --gsm8k) GSM8K=$2; shift 2 ;;
    --devices) DEVICES=$2; shift 2 ;;
    --tp) TP=$2; shift 2 ;;
    --cache-root) CACHE_ROOT=$2; shift 2 ;;
    --tasks) TASKS=$2; shift 2 ;;
    --num-prompts) NUM_PROMPTS=$2; shift 2 ;;
    --fixed-data-root) FIXED_DATA_ROOT=$2; shift 2 ;;
    --mmlu-num-prompts) MMLU_NUM_PROMPTS=$2; shift 2 ;;
    --gsm8k-num-prompts) GSM8K_NUM_PROMPTS=$2; shift 2 ;;
    --gpqa-dataset) GPQA_DATASET=$2; shift 2 ;;
    --gpqa-num-prompts) GPQA_NUM_PROMPTS=$2; shift 2 ;;
    --allow-response-anomalies) ALLOW_RESPONSE_ANOMALIES=--allow-response-anomalies; shift ;;
    --enable-prefix-caching) PREFIX_CACHING=--enable-prefix-caching; shift ;;
    --enable-expert-parallel) EXTRA_ARGS+=(--enable-expert-parallel); shift ;;
    --quantization) EXTRA_ARGS+=(--quantization "$2"); shift 2 ;;
    --trust-remote-code) EXTRA_ARGS+=(--trust-remote-code); shift ;;
    --kv-cache-dtype) EXTRA_ARGS+=(--kv-cache-dtype "$2"); shift 2 ;;
    --tokenizer-mode) EXTRA_ARGS+=(--tokenizer-mode "$2"); shift 2 ;;
    --max-model-len) EXTRA_ARGS+=(--max-model-len "$2"); shift 2 ;;
    --gpu-memory-utilization) EXTRA_ARGS+=(--gpu-memory-utilization "$2"); shift 2 ;;
    --timeout) TIMEOUT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n $MODEL && -n $WORKSPACE && -n $ROOT && -n $AISBENCH_HOME && -n $MMLU && -n $GSM8K ]] || { usage >&2; exit 2; }
[[ $ROOT = /* && ! -e $ROOT ]] || { echo 'Require a new absolute result directory' >&2; exit 2; }
[[ -f $MODEL/config.json ]] || { echo "Missing model config: $MODEL/config.json" >&2; exit 2; }
[[ -f $WORKSPACE/env.sh ]] || { echo "Missing workspace environment: $WORKSPACE/env.sh" >&2; exit 2; }
[[ -d $AISBENCH_HOME && -d $MMLU && -d $GSM8K ]] || { echo 'Invalid AISBench or dataset directory' >&2; exit 2; }
[[ $TP =~ ^[1-9][0-9]*$ && $NUM_PROMPTS =~ ^[1-9][0-9]*$ && $TIMEOUT =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid TP, prompt count or timeout' >&2; exit 2; }
IFS=',' read -r -a CARDS <<< "$DEVICES"
[[ ${#CARDS[@]} -eq $TP ]] || { echo 'The number of devices must equal TP' >&2; exit 2; }

SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$WORKSPACE/env.sh" || exit 2
PYTHON=${PY:-python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1} PYTHONUNBUFFERED=1
FIXED_ARGS=()
if [[ -n $FIXED_DATA_ROOT ]]; then
  [[ -d $FIXED_DATA_ROOT ]] || { echo 'Invalid fixed data directory' >&2; exit 2; }
  FIXED_ARGS=(--fixed-data-root "$FIXED_DATA_ROOT")
fi
if [[ -n $MMLU_NUM_PROMPTS ]]; then FIXED_ARGS+=(--mmlu-num-prompts "$MMLU_NUM_PROMPTS"); fi
if [[ -n $GSM8K_NUM_PROMPTS ]]; then FIXED_ARGS+=(--gsm8k-num-prompts "$GSM8K_NUM_PROMPTS"); fi
if [[ -n $GPQA_NUM_PROMPTS ]]; then FIXED_ARGS+=(--gpqa-num-prompts "$GPQA_NUM_PROMPTS"); fi
if [[ -n $GPQA_DATASET ]]; then FIXED_ARGS+=(--gpqa-dataset "$GPQA_DATASET"); fi
mkdir -p "$ROOT/accuracy" "$ROOT/_locks"
CACHE_ROOT=${CACHE_ROOT:-$ROOT/cache}
LOCKS=$ROOT/_locks
SERVER_PID=
BASE_URL=

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
  local track=$1 port
  port=$(free_port)
  BASE_URL=http://127.0.0.1:$port
  local prefix=$ROOT/accuracy/$track
  "$PYTHON" "$SCRIPTS/runtime.py" exec --workspace "$WORKSPACE" \
    --devices "$DEVICES" --lock-dir "$LOCKS" --cache-dir "$CACHE_ROOT/$track" \
    --out "$prefix.runtime.json" --timeout "$TIMEOUT" -- \
    "$PYTHON" "$SCRIPTS/server.py" run --model "$MODEL" --track "$track" \
    --port "$port" --tp "$TP" "$PREFIX_CACHING" "${EXTRA_ARGS[@]}" \
    --metadata "$prefix.server.json" -- --seed 0 > "$prefix.server.log" 2>&1 &
  SERVER_PID=$!
  "$PYTHON" "$SCRIPTS/server.py" wait --base-url "$BASE_URL" --server-pid "$SERVER_PID"
}

for track in eager inductor; do
  echo "ACCURACY_START model=$MODEL track=$track devices=$DEVICES"
  start_server "$track"
  "$PYTHON" "$SCRIPTS/accuracy.py" run --model "$MODEL" --base-url "$BASE_URL" \
    --label "$track" --out "$ROOT/accuracy/$track" \
    --vllm-ascend-source "$WORKSPACE/vllm-ascend" --benchmark-home "$AISBENCH_HOME" \
    --mmlu-dataset "$MMLU" --gsm8k-dataset "$GSM8K" \
    --tasks "$TASKS" --num-prompts "$NUM_PROMPTS" "${FIXED_ARGS[@]}" $ALLOW_RESPONSE_ANOMALIES
  stop_server
  if [[ $track == inductor ]]; then
    "$PYTHON" "$SCRIPTS/runtime.py" evidence --log "$ROOT/accuracy/inductor.server.log" \
      --run-config "$ROOT/accuracy/inductor.server.json" \
      --cache-dir "$CACHE_ROOT/inductor" --out "$ROOT/accuracy/inductor_evidence.json"
  fi
done

"$PYTHON" "$SCRIPTS/accuracy.py" compare \
  --reference "$ROOT/accuracy/eager/accuracy_results.json" \
  --candidate "$ROOT/accuracy/inductor/accuracy_results.json" \
  --tasks "$TASKS" --out "$ROOT/accuracy/compare.json"
"$PYTHON" "$SCRIPTS/report.py" "$ROOT/accuracy/compare.json" \
  "$ROOT/accuracy/inductor_evidence.json" --out "$ROOT/accuracy/SUMMARY.md"
cat "$ROOT/accuracy/SUMMARY.md"
