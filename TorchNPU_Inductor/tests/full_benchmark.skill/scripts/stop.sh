#!/usr/bin/env bash
# Stop the batch started by run_all.sh: terminate the launcher's whole process tree, then verify.
ORCH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ORCH/common.sh" 2>/dev/null
PATTERN="$(basename "$ORCH")/(run_all|deepseek).sh"

descendants() {
  local child
  for child in $(ps -o pid= --ppid "$1" 2>/dev/null); do
    echo "$child"
    descendants "$child"
  done
}

alive() {
  local state
  state=$(ps -o stat= -p "$1" 2>/dev/null) || return 1
  [[ -n $state && $state != Z* ]]
}

masters=$(pgrep -f "$PATTERN" | grep -v -x -e "$$" -e "$PPID" || true)
if [[ -z $masters ]]; then
  echo "NO_RUNNING_BATCH"
else
  list=$(for m in $masters; do echo "$m"; descendants "$m"; done | sort -un | tr '\n' ' ')
  echo "STOPPING $(echo $list | wc -w) processes (masters: $(echo $masters | tr '\n' ' '))"
  ps -o pid=,etime=,args= -p "$(echo $list | tr ' ' ',')" 2>/dev/null | cut -c1-170
  kill -TERM $list 2>/dev/null
  left=$list
  for _ in $(seq 1 30); do
    sleep 2
    left=""
    for p in $list; do alive "$p" && left+=" $p"; done
    [[ -z $left ]] && break
  done
  if [[ -n $left ]]; then
    echo "SIGKILL after 60s: $left"
    kill -KILL $left 2>/dev/null
    sleep 5
  fi
  left=""
  for p in $list; do alive "$p" && left+=" $p"; done
  echo "LEFT: ${left:-none}"
fi
rm -rf "$BATCH"/cardlocks/*
busy=0
for _ in $(seq 1 15); do
  busy=$(npu-smi info | grep -oE '[0-9]+ +/ +98304' | awk -F'/' '{gsub(/ /,"",$1); if ($1+0 > 8192) n++} END {print n+0}')
  [[ $busy -eq 0 ]] && break
  sleep 4
done
echo "NPU_HBM_USED_MB: $(npu-smi info | grep -oE '[0-9]+ +/ +98304' | awk -F'/' '{gsub(/ /,"",$1); printf "%s ", $1}')"
echo "CARDS_OVER_8192MB: $busy"
echo "REMAINING_TEST_PROCESSES:"
# The DeepSeek stage enlarges /dev/shm; restore the 16G container default if it was left enlarged.
shm_bytes=$(df -B1 --output=size /dev/shm | tail -1 | tr -d " ")
if (( shm_bytes > 34359738368 )); then
  mount -o remount,size=16G /dev/shm && echo "SHM_RESTORED_16G ($(df -h /dev/shm | tail -1 | awk '{print $2}'))" || echo "SHM_RESTORE_FAILED"
fi
ps -eo pid,etime,stat,args | grep -E "run_performance|vllm.entrypoints|EngineCore|server.py|runtime.py" | grep -v -E "grep|defunct|stop.sh" | cut -c1-170
