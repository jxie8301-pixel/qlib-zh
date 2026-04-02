#!/usr/bin/env bash
# Ensure rdagent template files are installed into running rdagent container
set -euo pipefail

CONTAINER=${1:-qlib-rdagent-1}
SCRIPT_PATH=/Users/apple/github/qlib/scripts/install_rdagent_templates.py

echo "Running install script in container $CONTAINER (if present)" >&2
docker exec -u 0 "$CONTAINER" bash -lc "python3 $SCRIPT_PATH || true"

echo "done"
