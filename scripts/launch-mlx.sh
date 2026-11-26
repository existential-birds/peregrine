#!/bin/bash
# Detached mlx-vlm server launcher for bench.py.
# Usage: launch-mlx.sh <name> [extra mlx_vlm.server flags...]
# Writes .server.pid; logs to logs/server-<name>.log. bench.py measures
# startup from its own t0, so this returns as soon as the server is spawned.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:-default}"
shift || true

bash scripts/stop-mlx.sh || true

HAS_MODEL=0
for arg in "$@"; do
  if [ "$arg" = "--model" ]; then
    HAS_MODEL=1
    break
  fi
done

MODEL_ARGS=()
if [ "$HAS_MODEL" -eq 0 ]; then
  MODEL_ARGS=(--model mlx-community/Qwen3.8-27B-4bit)
fi

SERVER_BIN="${MLX_VLM_SERVER:-/Users/ka/.local/bin/mlx_vlm.server}"
LOG="logs/server-${NAME}.log"
mkdir -p logs
nohup /usr/bin/python3 -c 'import os, sys; os.setsid(); os.execv(sys.argv[1], sys.argv[1:])' \
  "$SERVER_BIN" \
  ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} \
  --port 8080 \
  "$@" >"$LOG" 2>&1 </dev/null &
echo $! > .server.pid
echo "launched ${NAME} (pid $(cat .server.pid)), log: ${LOG}"
