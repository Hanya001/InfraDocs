#!/usr/bin/env bash
# Final stage: after every small job is done, run DeepSeek-V4.1-Flash-1017 with the
# official 950DT DP8/TP1/EP8 recipe, only if all eight NPUs are idle.
# DeepSeek runs ONE round only (user decision). Order of attempts: BF16 KV with the official context length,
# then BF16 KV with a reduced context, then FP8 KV. The ladder stops at the first attempt whose default
# track completed; the next attempt only starts when the default track itself failed to run.
set -uo pipefail
ORCH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ORCH/common.sh"
if [[ -n ${DRY_RUN:-} ]]; then log "DRY_RUN deepseek stage skipped"; exit 0; fi
mkdir -p "$BATCH/logs" "$BATCH/status"

DS_MODEL=/mnt/weight/DeepSeek-V4.1-Flash-1017
DS_DEVICES=0,1,2,3,4,5,6,7
# Environment from the official 950DT single-node example.
export HCCL_OP_EXPANSION_MODE=CCU_SCHED HCCL_BUFFSIZE=1600 HCCL_CONNECT_TIMEOUT=1200 HCCL_EXEC_TIMEOUT=2040
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 OMP_PROC_BIND=false OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

log "DEEPSEEK_STAGE_START"
idle=0
for _ in $(seq 1 30); do
  if python "$SCRIPTS/runtime.py" check-devices --devices "$DS_DEVICES" >/dev/null 2>&1; then idle=1; break; fi
  sleep 20
done
if (( ! idle )); then
  log "DEEPSEEK_DEFERRED reason=NPU_busy"
  echo 75 > "$BATCH/status/deepseek.status"
  exit 75
fi

# The official example starts the container with --shm-size=512g for the Engram host
# table. This container was started with 16g, so enlarge /dev/shm for this stage and
# restore the original size afterwards.
shm_before=$(df -B1 --output=size /dev/shm | tail -1 | tr -d ' ')
if mount -o remount,size=512G /dev/shm 2>"$BATCH/logs/deepseek_shm.err"; then
  log "SHM_REMOUNTED before=${shm_before}B now=$(df -h /dev/shm | tail -1 | awk '{print $2}')"
else
  log "SHM_REMOUNT_FAILED $(cat "$BATCH/logs/deepseek_shm.err")"
  echo 76 > "$BATCH/status/deepseek.status"
  exit 76
fi
restore_shm() { mount -o remount,size="$shm_before" /dev/shm && log "SHM_RESTORED" || log "SHM_RESTORE_FAILED"; }
trap restore_shm EXIT

default_loads() { ls "$BATCH/$1"/perf/performance/default/*_repeat1.json 2>/dev/null | wc -l; }

run_attempt() {
  local name=$1 kv=$2; shift 2
  local -a kv_args=()
  [[ -n $kv ]] && kv_args=(--kv-cache-dtype "$kv")
  log "DEEPSEEK_ATTEMPT_START name=$name kv=${kv:-bf16} extra=$*"
  bash "$SCRIPTS/run_performance.sh" --model "$DS_MODEL" --workspace "$WS" \
    --results "$BATCH/$name/perf" --sharegpt "$SHAREGPT" \
    --devices "$DS_DEVICES" --tp 1 --dp 8 --enable-expert-parallel \
    --quantization deepseek_v4_fp8 --tokenizer-mode deepseek_v41 --trust-remote-code \
    --capture-size "$CAPTURE" --repeats 1 --timeout 14400 \
    --max-num-seqs 64 --gpu-memory-utilization 0.9 \
    --additional-config '{"multistream_overlap_shared_expert":true,"combine_quant_mode":4}' \
    --vllm-arg --block-size --vllm-arg 128 \
    --vllm-arg --max-num-batched-tokens --vllm-arg 4096 \
    --vllm-arg --engram-config --vllm-arg '{"cpu_offload":true,"dp_shared_memory":true}' \
    --vllm-arg --no-swa-bounded-replay \
    "${kv_args[@]}" "$@" \
    > "$BATCH/logs/perf_$name.log" 2>&1
  local rc=$?
  echo "$rc" > "$BATCH/status/perf_$name.status"
  log "DEEPSEEK_ATTEMPT_END name=$name rc=$rc default_loads=$(default_loads "$name")/6"
  return "$rc"
}

A=DeepSeek-V4.1-Flash-1017_bf16kv
B=DeepSeek-V4.1-Flash-1017_bf16kv_ctx65536
C=DeepSeek-V4.1-Flash-1017_fp8kv

run_attempt "$A" "" --max-model-len 1048576
rc=$?
if (( rc == 0 )); then log "DEEPSEEK_DONE variant=$A"; echo 0 > "$BATCH/status/deepseek.status"; exit 0; fi
if (( $(default_loads "$A") == 6 )); then log "DEEPSEEK_PARTIAL variant=$A default complete, inductor failed, no retry"; echo "$rc" > "$BATCH/status/deepseek.status"; exit "$rc"; fi

run_attempt "$B" "" --max-model-len 65536 --gpu-memory-utilization 0.8
rc=$?
if (( rc == 0 )); then log "DEEPSEEK_DONE variant=$B"; echo 0 > "$BATCH/status/deepseek.status"; exit 0; fi
if (( $(default_loads "$B") == 6 )); then log "DEEPSEEK_PARTIAL variant=$B default complete, inductor failed, no retry"; echo "$rc" > "$BATCH/status/deepseek.status"; exit "$rc"; fi

run_attempt "$C" fp8 --max-model-len 1048576
rc=$?
if (( rc == 0 )); then log "DEEPSEEK_DONE variant=$C"; echo 0 > "$BATCH/status/deepseek.status"; exit 0; fi
log "DEEPSEEK_FAILED all_variants"
echo "$rc" > "$BATCH/status/deepseek.status"
exit "$rc"
