#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run_offline_performance.sh --model PATH --workspace PATH --results PATH [options]

Options:
  --devices IDS   Comma-separated NPU IDs (default: 0)
  --tp N          Tensor parallel size; must match the number of devices (default: 1)
  --cache-root PATH  Existing cache directory containing default/ and inductor/
  --sharegpt PATH ShareGPT JSON path; required for throughput
  --tests LIST    Comma-separated latency,throughput (default: both)
  --repeats N     Number of default/Inductor rounds (default: 3)
  --load-format NAME  Override model weight loading format, for example dummy
  --enable-expert-parallel  Enable expert parallel
  --quantization NAME  Override quantization method
  --trust-remote-code  Allow checkpoint-provided model code
  --kv-cache-dtype TYPE  Override KV Cache dtype, for example int8
  --tokenizer-mode MODE  Override tokenizer mode
  --max-model-len N  Override maximum model context length
  --gpu-memory-utilization N  Override vLLM memory utilization fraction
  --timeout SEC   Timeout for each benchmark process (default: 21600)
  -h, --help      Show this help
EOF
}

MODEL=
WORKSPACE=
ROOT=
DEVICES=0
TP=1
CACHE_ROOT=
SHAREGPT=
TESTS="latency throughput"
REPEATS=3
TIMEOUT=21600
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL=$2; shift 2 ;;
    --workspace) WORKSPACE=$2; shift 2 ;;
    --results) ROOT=$2; shift 2 ;;
    --devices) DEVICES=$2; shift 2 ;;
    --tp) TP=$2; shift 2 ;;
    --cache-root) CACHE_ROOT=$2; shift 2 ;;
    --sharegpt) SHAREGPT=$2; shift 2 ;;
    --tests) TESTS=${2//,/ }; shift 2 ;;
    --repeats) REPEATS=$2; shift 2 ;;
    --load-format) EXTRA_ARGS+=(--load-format "$2"); shift 2 ;;
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
[[ -n $MODEL && -n $WORKSPACE && -n $ROOT ]] || { usage >&2; exit 2; }
[[ $ROOT = /* && ! -e $ROOT ]] || { echo 'Require a new absolute result directory' >&2; exit 2; }
[[ -f $MODEL/config.json ]] || { echo "Missing model config: $MODEL/config.json" >&2; exit 2; }
[[ -f $WORKSPACE/env.sh ]] || { echo "Missing workspace environment: $WORKSPACE/env.sh" >&2; exit 2; }
[[ $TP =~ ^[1-9][0-9]*$ && $REPEATS =~ ^[1-9][0-9]*$ && $TIMEOUT =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid TP, repeats or timeout' >&2; exit 2; }
IFS=',' read -r -a CARDS <<< "$DEVICES"
[[ ${#CARDS[@]} -eq $TP ]] || { echo 'The number of devices must equal TP' >&2; exit 2; }
for TEST in $TESTS; do
  [[ $TEST == latency || $TEST == throughput ]] || { echo "Unknown test: $TEST" >&2; exit 2; }
done
if [[ " $TESTS " == *" throughput "* ]]; then
  [[ -n $SHAREGPT && -f $SHAREGPT ]] || { echo 'Throughput requires --sharegpt with an existing file' >&2; exit 2; }
fi

SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$WORKSPACE/env.sh" || exit 2
PYTHON=${PY:-python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1} PYTHONUNBUFFERED=1
mkdir -p "$ROOT/offline_performance"/{default,inductor,compare} "$ROOT/_locks"
CACHE_ROOT=${CACHE_ROOT:-$ROOT/cache}
LOCKS=$ROOT/_locks
COMPARISONS=()

"$PYTHON" "$SCRIPTS/runtime.py" check-devices --devices "$DEVICES"
"$PYTHON" "$SCRIPTS/runtime.py" snapshot --workspace "$WORKSPACE" --out "$ROOT/environment.json"

run_one() {
  local track=$1 test=$2 round=$3
  local output=$ROOT/offline_performance/$track/$test.repeat$round.json
  local prefix=$ROOT/offline_performance/$track.$test.repeat$round
  local dataset_args=()
  if [[ $test == throughput ]]; then
    dataset_args=(--dataset-path "$SHAREGPT")
  fi
  "$PYTHON" "$SCRIPTS/runtime.py" exec --workspace "$WORKSPACE" \
    --devices "$DEVICES" --lock-dir "$LOCKS" --cache-dir "$CACHE_ROOT/$track" \
    --out "$prefix.runtime.json" --timeout "$TIMEOUT" -- \
    "$PYTHON" "$SCRIPTS/offline_performance.py" run --benchmark "$test" --track "$track" \
    --model "$MODEL" --tp "$TP" --out "$output" "${dataset_args[@]}" \
    -- "${EXTRA_ARGS[@]}" 2>&1 | tee "$prefix.log"
  if [[ $track == inductor ]]; then
    "$PYTHON" "$SCRIPTS/runtime.py" evidence --log "$prefix.log" \
      --cache-dir "$CACHE_ROOT/inductor" --out "$prefix.evidence.json"
  fi
}

for ((round=1; round<=REPEATS; round++)); do
  for track in default inductor; do
    for test in $TESTS; do
      echo "OFFLINE_PERFORMANCE_START model=$MODEL round=$round track=$track test=$test devices=$DEVICES"
      run_one "$track" "$test" "$round"
    done
  done
  for test in $TESTS; do
    output=$ROOT/offline_performance/compare/$test.repeat$round.json
    "$PYTHON" "$SCRIPTS/offline_performance.py" compare --benchmark "$test" \
      --reference "$ROOT/offline_performance/default/$test.repeat$round.json" \
      --candidate "$ROOT/offline_performance/inductor/$test.repeat$round.json" \
      --out "$output"
    COMPARISONS+=("$output")
  done
done
"$PYTHON" "$SCRIPTS/offline_performance.py" summary "${COMPARISONS[@]}" \
  --out "$ROOT/offline_performance/SUMMARY.md"
