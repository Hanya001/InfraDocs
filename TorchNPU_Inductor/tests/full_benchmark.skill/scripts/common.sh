#!/usr/bin/env bash
# Every knob below can be overridden from the environment, e.g.
#   BATCH=... ROUNDS=2 MAX_CARDS=8 CAPTURE=1024 DEEPSEEK=1 bash ./run_all.sh
V=${V:-/home/z00896543/validation/214build_20261120_1546}
WS=$V/inductor-meta-worktree
# server copy of tests/benchmark (sync it first and verify md5)
SCRIPTS=${SCRIPTS:-$V/scripts/benchmark_20261128_v9}
# results/<batch>; must be a new directory
BATCH=${BATCH:-$V/results/perf_cap1024_YYYYMMDD_NN}
SHAREGPT=$V/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
CAPTURE=${CAPTURE:-1024}
ROUNDS=${ROUNDS:-2}
MAX_CARDS=${MAX_CARDS:-4}
# Cards the scheduler may pick, in preference order; override to keep a batch away from cards
# another user (or another session) is using, e.g. CARD_ORDER="3 2".
CARD_ORDER=${CARD_ORDER:-"7 6 5 4 3 2 1 0"}
[[ -f /usr/share/zoneinfo/Asia/Shanghai ]] && export TZ=Asia/Shanghai
source "$WS/env.sh"
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
log() { echo "$(date -Is) $*"; }
