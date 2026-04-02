#!/usr/bin/env bash
set -euo pipefail

LOG_PREFIX="[docker-entrypoint]"
echo "$LOG_PREFIX starting with args: $*"

run_update_data() {
  echo "$LOG_PREFIX Updating qlib data to today..."
  # Forward extra args (like --delay, --end_date) to the collector command.
  # Build args array excluding the token "--update_data" and "--fin_factor".
  forward_args=()
  skip_next=false
  for t in "$@"; do
    if [ "$t" = "--update_data" ] || [ "$t" = "--fin_factor" ]; then
      continue
    fi
    forward_args+=("$t")
  done

  # If user did not provide --end_date, append today's date
  has_end_date=false
  for t in "${forward_args[@]}"; do
    case "$t" in
      --end_date)
        has_end_date=true
        ;;
      --end_date=*)
        has_end_date=true
        ;;
    esac
  done

  end_date_arg=()
  if [ "$has_end_date" = false ]; then
    end_date_arg=(--end_date "$(date +%F)")
  fi

  python /Users/apple/github/qlib/scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir /Users/apple/github/qlib/DATA/qlib_data/cn_data "${forward_args[@]}" "${end_date_arg[@]}"
}

run_fin_factor() {
  echo "$LOG_PREFIX Running rdagent fin_factor (capped at 20 rounds)..."
  # ensure cap is set
  export RDAGENT_MAX_ROUNDS=20
  python /Users/apple/github/qlib/scripts/run_fin_factor_with_cap.py
}

handled=false
if [ "$#" -gt 0 ]; then
  for arg in "$@"; do
    case "$arg" in
      --update_data)
        run_update_data
        handled=true
        ;;
      --fin_factor)
        run_fin_factor
        handled=true
        ;;
      *)
        ;;
    esac
  done
fi

# Fallback to environment variable `RDAGENT_FLAGS` if no args provided
if [ "$handled" = false ] && [ -n "${RDAGENT_FLAGS-}" ]; then
  echo "$LOG_PREFIX RDAGENT_FLAGS='$RDAGENT_FLAGS'"
  for token in $RDAGENT_FLAGS; do
    case "$token" in
      --update_data)
        run_update_data
        handled=true
        ;;
      --fin_factor)
        run_fin_factor
        handled=true
        ;;
    esac
  done
fi

if [ "$handled" = false ]; then
  echo "$LOG_PREFIX No action requested; entering idle sleep."
fi

# keep container alive
exec sleep infinity
