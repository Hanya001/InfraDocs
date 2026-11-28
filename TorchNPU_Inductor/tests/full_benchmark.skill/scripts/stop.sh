#!/usr/bin/env bash
# Stop the batch started by run_all.sh: SIGKILL the scheduler's whole process tree, then verify.
# Processes are matched by script name (run_all.sh / run_performance.sh / deepseek.sh) whatever
# their path; the bracketed patterns never match this script's own shell. Test servers are killed
# outright: an aborted measurement needs no graceful shutdown, and waiting only slows this down.
# This stops every run_all.sh / run_performance.sh in the container, not just one batch.
ORCH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ORCH/common.sh" 2>/dev/null
descendants() {
  local child
  for child in $(ps -o pid= --ppid "$1" 2>/dev/null); do echo "$child"; descendants "$child"; done
}
roots=$(ps -eo pid=,args= | grep -E "[r]un_all\.sh|[r]un_performance\.sh|[d]eepseek\.sh" | awk '{print $1}')
if [[ -z $roots ]]; then
  echo "NO_RUNNING_BATCH"
else
  list=$(for p in $roots; do echo "$p"; descendants "$p"; done | sort -un | tr '\n' ' ')
  echo "KILLING $(echo $list | wc -w) processes (roots: $(echo $roots | tr '\n' ' '))"
  kill -KILL $list 2>/dev/null
  sleep 3
  left=$(ps -eo pid=,args= | grep -E "[r]un_performance\.sh|[v]llm\.entrypoints|[E]ngineCore|[s]erver\.py run|[r]untime\.py exec|[v]llm bench serve" | awk '{print $1}' | tr '\n' ' ')
  echo "LEFT: ${left:-none}"
fi
rm -rf "$BATCH"/cardlocks/*
echo "NPU_HBM_USED_MB: $(npu-smi info | grep -oE '[0-9]+ +/ +98304' | awk -F'/' '{gsub(/ /,"",$1); printf "%s ", $1}')"
# The DeepSeek stage enlarges /dev/shm; restore the 16G container default if it was left enlarged.
shm_bytes=$(df -B1 --output=size /dev/shm | tail -1 | tr -d " ")
if (( shm_bytes > 34359738368 )); then
  mount -o remount,size=16G /dev/shm && echo "SHM_RESTORED_16G" || echo "SHM_RESTORE_FAILED"
fi
