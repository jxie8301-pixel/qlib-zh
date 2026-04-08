#!/bin/bash
API_BASE="https://open.bigmodel.cn/api/paas/v4"
MODEL="glm-4-flash"

# Require OPENAI_API_KEY to be provided externally (do NOT hardcode keys in scripts)
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set. Export OPENAI_API_KEY before running this script."
  exit 1
fi

while true; do
  export OPENAI_API_BASE="$API_BASE"
  export CHAT_MODEL="$MODEL"
  export RDAGENT_MAX_ROUNDS=20
  unset FORCE_LOCAL_STUB
  echo "[fin_loop] starting rdagent fin_factor: $(date)"
  rdagent fin_factor && echo "[fin_loop] success at $(date)" && break || echo "[fin_loop] failed at $(date), retrying in 10s"
  sleep 10
done
