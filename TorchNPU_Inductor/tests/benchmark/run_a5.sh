#!/usr/bin/env bash
# 在现有 A5 容器内运行。参数为新的结果根目录，以及可选的单个模型名称。
set -uo pipefail
RUN_ROOT=${1:?Usage: bash run_a5.sh /absolute/new/result/root [model-name]}
SELECT=${2:-all}
BASE=${BENCH_BASE:-/home/z00896543/validation/a51112_20261116_195508}
WORKSPACE=${BENCH_WORKSPACE:-$BASE/pin/inductor-meta-worktree}
SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case ${MODEL_SET:-20261117} in
  20261117)
    RUNS=(
      'Qwen3-0.6B|/mnt/weight/Qwen/Qwen3-0.6B|1|0|'
      'Qwen3-0.6B-W8A8|/mnt/weight/Qwen3-0.6B-W8A8|1|0|'
      'Qwen3-Embedding-0.6B|/mnt/weight/Qwen3-Embedding-0.6B|1|0|'
      'Qwen3.5-0.8B|/mnt/weight/Qwen3.5-0.8B|1|0|'
      'Qwen3-8B-W8A8|/mnt/weight/Qwen3-8B-W8A8|1|0|'
      'Qwen2.5-VL-7B-Instruct|/mnt/weight/Qwen2.5-VL-7B-Instruct|1|0|'
      'DeepSeek-V2-Lite|/mnt/weight/DeepSeek-V2-Lite|1|0|'
      'DeepSeek-V2-Lite-W8A8|/mnt/weight/DeepSeek-V2-Lite-W8A8|1|0|'
      'Qwen3-30B-A3B-W8A8|/mnt/weight/Qwen3-30B-A3B-W8A8|2|0|'
      'Qwen3.8-27B|/mnt/weight/Qwen3.8-27B|2|0|'
      'Qwen3.8-27B-w8a8|/mnt/weight/Qwen3.8-27B-w8a8|2|0|'
      'Qwen3.8-27B-FP8|/mnt/weight/Qwen3.8-27B-FP8|2|0|'
      'Qwen3.6-35B-A3B|/mnt/weight/Qwen3.6-35B-A3B|2|0|'
      'Qwen3.6-35B-A3B-w8a8|/mnt/weight/Qwen3.6-35B-A3B-w8a8|2|0|'
      'Qwen3.6-35B-A3B-w4a8-mxfp|/mnt/weight/Qwen3.6-35B-A3B-w4a8-mxfp|2|0|'
      'Qwen3.6-35B-A3B-w8a8-mxfp8|/mnt/weight/Qwen3.6-35B-A3B-w8a8-mxfp8|2|0|'
    )
    ;;
  20261118)
    RUNS=(
      'Qwen3-0.6B|/mnt/weight/Qwen/Qwen3-0.6B|1|0|'
      'Qwen3-0.6B-W8A8|/mnt/weight/Qwen3-0.6B-W8A8|1|0|'
      'Qwen3-Embedding-0.6B|/mnt/weight/Qwen3-Embedding-0.6B|1|0|'
      'Qwen3.5-0.8B|/mnt/weight/Qwen3.5-0.8B|1|0|'
      'Qwen2.5-VL-7B-Instruct|/mnt/weight/Qwen2.5-VL-7B-Instruct|1|0|'
      'Qwen3-8B-W8A8|/mnt/weight/Qwen3-8B-W8A8|1|0|'
      'DeepSeek-V2-Lite-W8A8|/mnt/weight/DeepSeek-V2-Lite-W8A8|1|0|'
      'DeepSeek-V2-Lite|/mnt/weight/DeepSeek-V2-Lite|1|0|'
      'Qwen3.8-27B-w8a8|/mnt/weight/Qwen3.8-27B-w8a8|2|0|'
      'Qwen3.8-27B-w8a8-mxfp8|/mnt/weight/weights/Qwen3.8-27B-w8a8-mxfp8|2|0|'
      'Qwen3.8-27B-FP8|/mnt/weight/Qwen3.8-27B-FP8|2|0|'
      'Qwen3-30B-A3B-W8A8|/mnt/weight/Qwen3-30B-A3B-W8A8|2|0|'
      'Qwen3.8-27B|/mnt/weight/Qwen3.8-27B|2|0|'
      'Qwen3.6-35B-A3B-w4a8-mxfp|/mnt/weight/Qwen3.6-35B-A3B-w4a8-mxfp|2|0|'
      'Qwen3.6-35B-A3B-w8a8|/mnt/weight/Qwen3.6-35B-A3B-w8a8|2|0|'
      'Qwen3.6-35B-A3B-w8a8-mxfp8|/mnt/weight/Qwen3.6-35B-A3B-w8a8-mxfp8|2|0|'
      'Qwen3.6-35B-A3B|/mnt/weight/Qwen3.6-35B-A3B|2|0|'
      'Kimi-K3-expert16-top8-100B-CountBenchQA|/mnt/weight/Kimi-K3-expert16-top8-100B-CountBenchQA|4|0|'
      'Qwen3.8-Flash-Next|/mnt/weight/Qwen3.8-Flash-Next|8|0|'
      'GLM-5.3-Flash-w8a8|/mnt/weight/GLM-5.3-Flash-w8a8|8|0|'
      'GLM-5.2-w4a8c8|/mnt/weight/GLM-5.2-w4a8c8|8|0|'
      'DeepSeek-V4.1-Flash|/mnt/weight/DeepSeek-V4.1-Flash-1017|8|1|deepseek_v4'
    )
    ;;
  *)
    printf 'Unknown MODEL_SET: %s\n' "$MODEL_SET" >&2
    exit 2
    ;;
esac
[[ $RUN_ROOT = /* ]] || { printf 'RUN_ROOT must be absolute\n' >&2; exit 2; }
mkdir -p "$RUN_ROOT"
cd "$WORKSPACE" || exit 1
exec 9>"$BASE/results/a5_bench_matrix_${BENCH_LOCK_SUFFIX:-$SELECT}.lock"
flock -n 9 || { printf 'Another bench213 A5 run holds the lock for %s\n' "$SELECT" >&2; exit 2; }
MATCHED=0
FAILED=0
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name weights tp enable_ep tokenizer_mode <<< "$spec"
  tp=${FORCE_TP:-$tp}
  [[ $SELECT = all || $SELECT = "$name" ]] || continue
  MATCHED=1
  if [[ ! -f $weights/config.json ]] || ! compgen -G "$weights/*.safetensors" > /dev/null; then
    mkdir -p "$RUN_ROOT/_skipped"
    printf 'time=%s model=%s weights=%s reason=missing_or_incomplete\n' "$(date -Is)" "$name" "$weights" >> "$RUN_ROOT/_skipped/$name.log"
    printf 'MODEL_EVENT skipped model=%s reason=missing_or_incomplete\n' "$name"
    continue
  fi
  model_dir=$RUN_ROOT/$name
  [[ ! -e $model_dir ]] || { printf 'Refusing to overwrite %s\n' "$model_dir" >&2; exit 2; }
  mkdir -p "$model_dir"
  if [[ -n ${BASELINE_EAGER_DIR:-} ]]; then
    mkdir -p "$model_dir/eager"
    cp "$BASELINE_EAGER_DIR/acc_track.jsonl" "$model_dir/eager/"
    cp "$BASELINE_EAGER_DIR/acc_self_compare.json" "$model_dir/eager/"
  fi
  printf 'MODEL_EVENT start model=%s tp=%s time=%s\n' "$name" "$tp" "$(date -Is)"
  npu-smi info > "$model_dir/npu_before.txt"
  model_failed=0
  for track in ${TRACKS:-eager default inductor}; do
    runtime=${REUSE_RUNTIME:-$model_dir/$track/runtime}
    mkdir -p "$runtime/log"
    cache_base=${BENCH_CACHE_ROOT:-$runtime/log}
    if [[ -d $cache_base && -n $(ls -A "$cache_base") ]]; then cache_state=warm; else cache_state=cold; fi
    printf 'cache_state=%s\ncache_base=%s\n' "$cache_state" "$cache_base" > "$model_dir/$track.cache_state.txt"
    unset VLLM_CACHE_ROOT
    source "$WORKSPACE/env.sh" > "$model_dir/$track.env.log" 2>&1
    env_rc=$?
    export VLLM_CACHE_ROOT=$cache_base/vllm_cache
    export TORCHINDUCTOR_CACHE_DIR=$cache_base/inductor
    export TRITON_CACHE_DIR=$cache_base/triton
    export TORCH_COMPILE_DEBUG_DIR=$runtime/log
    mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    if [[ $env_rc != 0 ]]; then
      printf 'MODEL_EVENT environment_failed model=%s track=%s rc=%s\n' "$name" "$track" "$env_rc"
      model_failed=1
      break
    fi
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCHINDUCTOR_AUTOGRAD_CACHE=0
    export PYTHONUNBUFFERED=1
    DEVICES=${BENCH_DEVICES:-2,3}
    IFS=',' read -r -a device_list <<< "$DEVICES"
    case $tp in
      1) devices=${device_list[0]} ;;
      2|4|8) devices=$DEVICES ;;
      *) printf 'Unsupported tensor parallel size for this run: %s\n' "$tp" >&2; exit 2 ;;
    esac
    export ASCEND_RT_VISIBLE_DEVICES="$devices"
    export BENCH_MODEL="$weights" BENCH_TP="$tp" BENCH_TRACK="$track" BENCH_OUT="$model_dir/$track"
    script=${BENCH_SCRIPT:-bench_one.py}
    if [[ $name = Qwen3-Embedding-0.6B && -z ${BENCH_SCRIPT:-} ]]; then script=embedding_one.py; fi
    export BENCH_ENABLE_EXPERT_PARALLEL="${enable_ep:-0}" BENCH_TOKENIZER_MODE="${tokenizer_mode:-}"
    export BENCH_VERIFY_REPEATS=${BENCH_VERIFY_REPEATS:-2}
    mkdir -p "$BENCH_OUT"
    cp env.sh "$BENCH_OUT/env.sh.snapshot"
    {
      printf 'time=%s\nworkspace=%s\npython=%s\n' "$(date -Is)" "$PWD" "$PY"
      printf 'model=%s\ntrack=%s\ntp=%s\ndevices=%s\n' "$weights" "$track" "$tp" "$ASCEND_RT_VISIBLE_DEVICES"
      printf 'model_runner_v2=%s\noptimization_level=%s\n' "${VLLM_USE_V2_MODEL_RUNNER:-0}" "O2"
      printf 'verify_repeats=%s\nautograd_cache=%s\n' "$BENCH_VERIFY_REPEATS" "$TORCHINDUCTOR_AUTOGRAD_CACHE"
      printf 'enable_expert_parallel=%s\ntokenizer_mode=%s\n' "$BENCH_ENABLE_EXPERT_PARALLEL" "$BENCH_TOKENIZER_MODE"
      printf 'mtp=%s\nmtp_tokens=%s\n' "${BENCH_MTP:-0}" "${BENCH_MTP_TOKENS:-1}"
      printf 'enforce_eager=%s\n' "${BENCH_ENFORCE_EAGER:-0}"
      git rev-parse HEAD
      git -C pytorch rev-parse HEAD
      git -C vllm-ascend rev-parse HEAD
      git -C vllm-ascend diff --stat
      git -C triton-ascend rev-parse HEAD
      "$PY" -I -c 'import importlib.metadata as m; print({p:m.version(p) for p in ["torch","torch-npu","triton-ascend","vllm","vllm-ascend"]})'
      printf 'runtime=%s\ncache_state=%s\ncache_base=%s\nvllm_cache_root=%s\ninductor_cache_dir=%s\ntriton_cache_dir=%s\n' "$runtime" "$cache_state" "$cache_base" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
      sha256sum "$SCRIPTS/$script" "$SCRIPTS/compare_tracks.py" "$SCRIPTS/run_a5.sh" "$SCRIPTS/process_supervisor.py" "$WORKSPACE/env.sh"
    } > "$BENCH_OUT/environment.txt" 2>&1
    git -C vllm-ascend diff > "$BENCH_OUT/vllm-ascend.patch"
    printf 'MODEL_EVENT track_start model=%s track=%s time=%s\n' "$name" "$track" "$(date -Is)"
    "$PY" -u "$SCRIPTS/process_supervisor.py" --timeout 3600 -- "$PY" -u "$SCRIPTS/$script" 2>&1 | tee "$model_dir/$track.runlog"
    rc=${PIPESTATUS[0]}
    printf '%s\n' "$rc" > "$BENCH_OUT/exit_code.txt"
    printf 'MODEL_EVENT track_end model=%s track=%s rc=%s time=%s\n' "$name" "$track" "$rc" "$(date -Is)"
    if [[ $rc != 0 ]]; then model_failed=1; break; fi
  done
  if [[ -f $model_dir/eager/acc_track.jsonl && -f $model_dir/inductor/acc_track.jsonl ]]; then
    "$PY" "$SCRIPTS/compare_tracks.py" "$model_dir" | tee "$model_dir/compare.log"
  fi
  npu-smi info > "$model_dir/npu_after.txt"
  printf 'MODEL_EVENT done model=%s failed=%s time=%s\n' "$name" "$model_failed" "$(date -Is)"
  [[ $model_failed = 0 ]] || FAILED=1
done
[[ $MATCHED = 1 ]] || { printf 'Unknown model: %s\n' "$SELECT" >&2; exit 2; }
exit "$FAILED"
