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

LOG="logs/server-${NAME}.log"
mkdir -p logs
setsid nohup mlx_vlm.server \
  --model mlx-community/Qwen3.8-27B-4bit \
  --port 8080 \
  "$@" >"$LOG" 2>&1 &
echo $! > .server.pid
echo "launched ${NAME} (pid $(cat .server.pid)), log: ${LOG}"
