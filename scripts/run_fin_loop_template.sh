#!/bin/bash
API_BASE="https://open.bigmodel.cn/api/paas/v4"
TOKEN="086fd27dd6184f80bc04172021661edf.CPUClMiDSGqqVwpe"
MODEL="glm-4-flash"
while true; do
  export OPENAI_API_BASE="$API_BASE"
  export OPENAI_API_KEY="$TOKEN"
  export CHAT_MODEL="$MODEL"
  export RDAGENT_MAX_ROUNDS=20
  unset FORCE_LOCAL_STUB
  echo "[fin_loop] starting rdagent fin_factor: $(date)"
  rdagent fin_factor && echo "[fin_loop] success at $(date)" && break || echo "[fin_loop] failed at $(date), retrying in 10s"
  sleep 10
done
