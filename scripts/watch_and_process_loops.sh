#!/usr/bin/env bash
set -euo pipefail
LOG_ROOT=log
OUT_BASE=${1:-DATA/analysis_outputs/rdagent_20round_auto_20260329_181653}
PROCESSED="$OUT_BASE/.processed_loops"
LOG_POLL_INTERVAL=${2:-10}
STABLE_WAIT=${3:-10}
mkdir -p "$OUT_BASE"
:>"$OUT_BASE/watch_process.log"
touch "$PROCESSED"
echo "$(date) start per-loop watcher -> $OUT_BASE" >> "$OUT_BASE/watch_process.log"
last_seen_latest=""
while true; do
  latest=$(ls -1dt "$LOG_ROOT"/* 2>/dev/null | head -n1 || true)
  if [ -n "$latest" ] && [ "$latest" != "$last_seen_latest" ]; then
    echo "$(date) detected latest log: $latest" >> "$OUT_BASE/watch_process.log"
    last_seen_latest="$latest"
  fi
  if [ -n "$latest" ]; then
    for loop in $(ls -1 "$latest" | egrep '^Loop_' | sort || true); do
      if grep -Fxq "$latest/$loop" "$PROCESSED"; then
        continue
      fi
      loop_dir="$latest/$loop"
      echo "$(date) new loop detected: $loop_dir" >> "$OUT_BASE/watch_process.log"
      # wait until there is at least one .pkl file and it becomes stable
      t0=$(date +%s)
      while true; do
        pkl_count=$(find "$loop_dir" -maxdepth 4 -type f -name '*.pkl' | wc -l || true)
        if [ "$pkl_count" -gt 0 ]; then
          # check stable: wait STABLE_WAIT seconds and ensure no new pkls appeared
          prev_count=$pkl_count
          sleep "$STABLE_WAIT"
          cur_count=$(find "$loop_dir" -maxdepth 4 -type f -name '*.pkl' | wc -l || true)
          if [ "$cur_count" -eq "$prev_count" ]; then
            break
          fi
        else
          sleep 2
        fi
        # timeout safety: don't wait forever (optional: 10 minutes)
        if [ $(( $(date +%s) - t0 )) -gt 600 ]; then
          echo "$(date) timeout waiting for pkls in $loop_dir" >> "$OUT_BASE/watch_process.log"
          break
        fi
      done
      # run collector (non-clean) to add/refresh this loop's analysis
      echo "$(date) running collector for $latest -> $OUT_BASE" >> "$OUT_BASE/watch_process.log"
      python3 scripts/collect_rdagent_rounds.py --log-dir "$latest" --out-base "$OUT_BASE" >> "$OUT_BASE/watch_process.log" 2>&1 || echo "collector failed for $loop_dir" >> "$OUT_BASE/watch_process.log"
      echo "$latest/$loop" >> "$PROCESSED"
      # notify
      if command -v osascript >/dev/null 2>&1; then
        osascript -e "display notification \"Collected $loop from $(basename $latest)\" with title \"rd-agent\""
      fi
    done
  fi
  sleep "$LOG_POLL_INTERVAL"
done
