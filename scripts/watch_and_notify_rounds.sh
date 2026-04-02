#!/usr/bin/env bash
set -euo pipefail
OUT_BASE=${1:-DATA/analysis_outputs/rdagent_20round_auto_20260329_181653}
LOG="$OUT_BASE/notify_watcher.log"
SEEN_FILE="$OUT_BASE/.seen_rounds"
mkdir -p "$OUT_BASE"
:>"$LOG"
if [ ! -f "$SEEN_FILE" ]; then
  touch "$SEEN_FILE"
fi
echo "$(date) start notify watcher for $OUT_BASE" >> "$LOG"
while true; do
  # list round directories
  if ls -1q "$OUT_BASE" 2>/dev/null | grep -q '^round_'; then
    for r in $(ls -1q "$OUT_BASE" | grep '^round_' | sort); do
      if ! grep -Fxq "$r" "$SEEN_FILE"; then
        echo "$(date) new round: $r" >> "$LOG"
        # macOS notification
        osascript -e "display notification \"New round collected: $r\" with title \"rd-agent\""
        echo "$r" >> "$SEEN_FILE"
      fi
    done
  fi
  sleep 10
done
