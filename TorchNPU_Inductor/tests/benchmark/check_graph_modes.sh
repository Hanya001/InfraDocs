#!/usr/bin/env bash
set -euo pipefail
BASE=/home/z00896543/validation/a51112_20261116_195508
ROOT=$BASE/results/20261117_inductor_matrix_01
SCRIPTS=$BASE/scripts/bench213_matrix
MODEL=${1:?usage: check_graph_modes.sh Qwen3-0.6B|Qwen3.5-0.8B}
for mode in NONE PIECEWISE FULL FULL_DECODE_ONLY FULL_AND_PIECEWISE; do
  mode_root=$ROOT/graph_modes/$MODEL/$mode
  if [[ -f $mode_root/$MODEL/inductor/correctness_complete.json || -f $mode_root/$MODEL/inductor/perf_metrics.json ]]; then
    echo "GRAPH_MODE_SKIP_COMPLETED model=$MODEL mode=$mode"
    continue
  fi
  echo "GRAPH_MODE_START model=$MODEL mode=$mode time=$(date -Is)"
  if BASELINE_EAGER_DIR=$ROOT/$MODEL/eager TRACKS=inductor BENCH_DEVICES=2,3 BENCH_CUDAGRAPH_MODE=$mode \
    BENCH_VERIFY_REPEATS=2 \
    bash "$SCRIPTS/run_model.sh" "$mode_root" "$MODEL"; then
    echo "GRAPH_MODE_DONE model=$MODEL mode=$mode time=$(date -Is)"
  else
    echo "GRAPH_MODE_FAILED model=$MODEL mode=$mode rc=$? time=$(date -Is)"
  fi
done
if [[ $MODEL = Qwen3-0.6B ]]; then
  tp2_root=$ROOT/tp2_smoke/$MODEL
  if [[ ! -f $tp2_root/$MODEL/inductor/correctness_complete.json && ! -f $tp2_root/$MODEL/inductor/perf_metrics.json ]]; then
    echo "TP2_SMOKE_START time=$(date -Is)"
    FORCE_TP=2 TRACKS="eager inductor" BENCH_DEVICES=2,3 \
      BENCH_CUDAGRAPH_MODE=PIECEWISE BENCH_VERIFY_REPEATS=2 \
      bash "$SCRIPTS/run_model.sh" "$tp2_root" "$MODEL"
    echo "TP2_SMOKE_DONE time=$(date -Is)"
  fi
fi
