#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" 2>/dev/null
echo "NOW $(date -Is)"
echo "== FINISHED (rc)"
for f in "$BATCH"/status/*.status; do [[ -e $f ]] && echo "  $(basename "$f" .status | sed 's/^perf_//') rc=$(cat "$f")"; done
echo "== CARD LOCKS (running jobs)"
for c in $CARD_ORDER; do
  [[ -d $BATCH/cardlocks/card_$c ]] && echo "  card $c -> $(cat "$BATCH/cardlocks/card_$c/job" 2>/dev/null)"
done
echo "== MASTER LOG (last 8, scheduler lines only)"
grep -E '^[0-9]{4}-' "$BATCH/logs/master.log" 2>/dev/null | tail -n 8
echo "== JOB LOG PROGRESS (running jobs)"
for f in $(ls -t "$BATCH"/logs/perf_*.log 2>/dev/null | head -5); do
  name=$(basename "$f" .log)
  [[ -f $BATCH/status/$name.status ]] && continue
  echo "--- $name mtime=$(date -r "$f" +%m-%dT%H:%M:%S) size=$(stat -c %s "$f")"
  grep -E '^(\[[^]]*\] )?(PERFORMANCE_START|DEVICES_|SUPERVISOR_STOP|Timed out|Traceback)' "$f" | tail -n 2 | cut -c1-200
  grep -cE '^(\[[^]]*\] )?PERFORMANCE_START' "$f" | sed 's/^/  server_starts=/'
  tail -n 1 "$f" | cut -c1-200
done
echo "== FILES_TOUCHED_LAST_30MIN=$(find "$BATCH" -type f -newermt '-30 minutes' 2>/dev/null | grep -v '/cache/' | wc -l)"
echo "== NPU HBM used/total MB"
npu-smi info | grep -oE '[0-9]+ +/ +98304' | awk -F'/' '{gsub(/ /,"",$1); print "  card" n++ " " $1 "/98304"}'
echo "== TEST PROCESSES=$(pgrep -fc 'run_performance.sh|vllm.entrypoints.openai.api_server|vllm bench serve' || true)"
