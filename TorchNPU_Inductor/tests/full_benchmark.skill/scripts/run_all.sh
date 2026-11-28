#!/usr/bin/env bash
# Master scheduler: runs every job with run_performance.sh, picking idle cards at
# launch time (preferred order 7..0), never using more than MAX_CARDS cards at once.
set -uo pipefail
ORCH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ORCH/common.sh"
mkdir -p "$BATCH/logs" "$BATCH/status" "$BATCH/cardlocks"
exec 9> "$BATCH/.master.lock"
flock -n 9 || { echo "MASTER_ALREADY_RUNNING"; exit 1; }

if pgrep -f "run_performance.sh.*$BATCH" >/dev/null; then
  echo "JOBS_STILL_RUNNING for $BATCH, refusing to start:" >&2
  pgrep -af "run_performance.sh.*$BATCH" | cut -c1-200 >&2
  exit 1
fi
rm -rf "$BATCH"/cardlocks/*
cp "$ORCH"/*.sh "$ORCH/jobs.txt" "$ORCH/summarize.py" "$BATCH/" 2>/dev/null
(cd "$SCRIPTS" && md5sum -- * .gitattributes .gitignore) > "$BATCH/benchmark_scripts_md5.txt"

# Jobs come from jobs.txt next to this script (one "name|model|tp|extra" per line,
# blank lines and #-comments ignored). TP1 jobs first, TP2 last.
JOBS=()
while IFS= read -r line; do
  [[ -z $line || $line == \#* ]] && continue
  JOBS+=("$line")
done < "$ORCH/jobs.txt"
(( ${#JOBS[@]} )) || { echo "no jobs in $ORCH/jobs.txt" >&2; exit 2; }

release() { local c; for c in "$@"; do rm -rf "$BATCH/cardlocks/card_$c"; done; }
count_cards() { local n=0 c; for c in $CARD_ORDER; do [[ -d $BATCH/cardlocks/card_$c ]] && n=$((n + 1)); done; echo "$n"; }
card_idle() { python "$SCRIPTS/runtime.py" check-devices --devices "$1" >/dev/null 2>&1; }

# acquire NEED cards -> CARDS=(...); on failure releases whatever it took
acquire() {
  local need=$1 c
  CARDS=()
  for c in $CARD_ORDER; do
    (( ${#CARDS[@]} >= need )) && break
    if mkdir "$BATCH/cardlocks/card_$c" 2>/dev/null; then
      if card_idle "$c"; then CARDS+=("$c"); else rm -rf "$BATCH/cardlocks/card_$c"; fi
    fi
  done
  (( ${#CARDS[@]} >= need )) && return 0
  release "${CARDS[@]}"
  return 1
}

run_job() {
  local name=$1 model=$2 tp=$3 extra=$4 cards=$5 rc
  local -a extra_args=()
  [[ -n $extra ]] && read -r -a extra_args <<< "$extra"
  log "JOB_START name=$name model=$model tp=$tp cards=$cards"
  if [[ -n ${DRY_RUN:-} ]]; then
    # dry run: stand in for the real job so scheduling and card locks can be checked
    sleep "$DRY_RUN"
    rc=0
  else
    # Job output goes to its own log file and, with the job name as prefix, to this script's
    # stdout, so a caller that runs "bash ./run_all.sh 2>&1 | tee -a master.log" sees every job
    # log live. PIPESTATUS[0] keeps run_performance.sh's exit code.
    bash "$SCRIPTS/run_performance.sh" --model "$model" --workspace "$WS" \
      --results "$BATCH/$name/perf" --sharegpt "$SHAREGPT" --devices "$cards" --tp "$tp" \
      --capture-size "$CAPTURE" --repeats "$ROUNDS" "${extra_args[@]}" \
      2>&1 | sed -u "s|^|[$name] |" | tee -a "$BATCH/logs/perf_$name.log"
    rc=${PIPESTATUS[0]}
  fi
  echo "$rc" > "$BATCH/status/perf_$name.status"
  log "JOB_END name=$name rc=$rc"
  return "$rc"
}

launch() {
  local name=$1 model=$2 tp=$3 extra=$4 cards=$5 c
  for c in ${cards//,/ }; do echo "$name" > "$BATCH/cardlocks/card_$c/job"; done
  (
    trap "release ${cards//,/ }" EXIT
    run_job "$name" "$model" "$tp" "$extra" "$cards"
  ) &
  RUNNING[$!]="$name"
}

declare -A RUNNING=()
pending=("${JOBS[@]}")
log "MASTER_START host=$(hostname) jobs=${#JOBS[@]} max_cards=$MAX_CARDS capture=$CAPTURE rounds=$ROUNDS"

while (( ${#pending[@]} || ${#RUNNING[@]} )); do
  for pid in "${!RUNNING[@]}"; do
    name=${RUNNING[$pid]}
    if [[ -f $BATCH/status/perf_$name.status ]] || ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null
      log "JOB_REAP name=$name rc=$(cat "$BATCH/status/perf_$name.status" 2>/dev/null || echo unknown)"
      unset "RUNNING[$pid]"
    fi
  done
  launched=0
  if (( ${#pending[@]} )); then
    used=$(count_cards)
    for ((i = 0; i < ${#pending[@]}; i++)); do
      IFS='|' read -r name model tp extra <<< "${pending[$i]}"
      (( used + tp > MAX_CARDS )) && continue
      if acquire "$tp"; then
        # CANN rejects a descending ASCEND_RT_VISIBLE_DEVICES, so hand over ascending ids
        cards=$(printf '%s\n' "${CARDS[@]}" | sort -n | paste -sd,)
        pending=("${pending[@]:0:i}" "${pending[@]:i+1}")
        launch "$name" "$model" "$tp" "$extra" "$cards"
        log "JOB_LAUNCH name=$name tp=$tp cards=$cards used=$((used + tp))/$MAX_CARDS pending=${#pending[@]} running=${#RUNNING[@]}"
        launched=1
        break
      fi
    done
  fi
  (( launched )) || sleep 20
done

summary=""
for f in "$BATCH"/status/*.status; do [[ -e $f ]] && summary+=" $(basename "$f" .status | sed 's/^perf_//')=$(cat "$f")"; done
log "SMALL_END$summary"

# Final stage: DeepSeek-V4.1 only if all eight NPUs are idle. DEEPSEEK=0 skips it.
if [[ ${DEEPSEEK:-1} == 1 ]]; then
  bash "$ORCH/deepseek.sh"
  log "MASTER_END deepseek_rc=$?"
else
  log "MASTER_END deepseek skipped (DEEPSEEK=0)"
fi
