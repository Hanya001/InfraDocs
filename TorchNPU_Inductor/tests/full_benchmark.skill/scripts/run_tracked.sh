#!/usr/bin/env bash
# Run the batch from this directory in the foreground of the calling session and print progress
# every PROGRESS seconds (default 300) until the scheduler exits. Meant to be the command of a
# trackable background task: its output is the batch log.
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
while kill -0 "$MASTER" 2>/dev/null; do
  sleep "$PROGRESS" &
  wait $! 2>/dev/null
  kill -0 "$MASTER" 2>/dev/null || break
  echo "===== $(date -Is) progress ====="
  bash ./status.sh 2>&1 | grep -v '^\[env.sh\]'
done
wait "$MASTER"
rc=$?
echo "===== $(date -Is) SCHEDULER_EXITED rc=$rc ====="
grep -v '^\[env.sh\]' "$BATCH/logs/master.log" | tail -n 15
exit "$rc"
