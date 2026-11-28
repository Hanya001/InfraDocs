#!/usr/bin/env bash
# Run the batch from this directory in the foreground of the calling session and stream its logs
# live: the scheduler's master.log and every logs/perf_<job>.log are followed as soon as they
# exist (job logs with a [name] prefix), plus a status block every PROGRESS seconds (default 300).
# Meant to be the command of a trackable background task: its output is the batch log.
#   BATCH=... ROUNDS=3 MAX_CARDS=4 CAPTURE=1024 DEEPSEEK=0 bash ./run_tracked.sh
set -u
ORCH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROGRESS=${PROGRESS:-300}
source "$ORCH/common.sh"
mkdir -p "$BATCH/logs"
cd "$ORCH"
bash ./run_all.sh > "$BATCH/logs/master.log" 2>&1 &
MASTER=$!
echo "MASTER_PID=$MASTER batch=$BATCH rounds=$ROUNDS max_cards=$MAX_CARDS capture=$CAPTURE progress_every=${PROGRESS}s"
# --pid makes every follower exit on its own when the scheduler is gone
( tail -n +1 -F --pid="$MASTER" "$BATCH/logs/master.log" 2>/dev/null | grep --line-buffered -v '^\[env.sh\]' ) &
FOLLOWED=" "
elapsed=0
while kill -0 "$MASTER" 2>/dev/null; do
  for file in "$BATCH"/logs/perf_*.log; do
    [[ -e $file ]] || continue
    [[ $FOLLOWED == *" $file "* ]] && continue
    FOLLOWED="$FOLLOWED$file "
    name=$(basename "$file" .log); name=${name#perf_}
    ( tail -n +1 -F --pid="$MASTER" "$file" 2>/dev/null | sed -u "s|^|[$name] |" ) &
  done
  sleep 10
  kill -0 "$MASTER" 2>/dev/null || break
  elapsed=$((elapsed + 10))
  if (( elapsed >= PROGRESS )); then
    elapsed=0
    echo "===== $(date -Is) progress ====="
    bash ./status.sh 2>&1 | grep -v '^\[env.sh\]' | sed '/^== JOB LOG/,$d'
  fi
done
wait "$MASTER"
rc=$?
sleep 3
echo "===== $(date -Is) SCHEDULER_EXITED rc=$rc ====="
exit "$rc"
