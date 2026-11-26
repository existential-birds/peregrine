#!/bin/bash
# Drain-then-stop for the mlx-vlm server (never force-kill mid-batch:
# SIGTERM, 30 s grace for in-flight requests, then KILL). Stops by PID.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .server.pid ] || exit 0
PID="$(cat .server.pid)"
if kill -0 "$PID" 2>/dev/null; then
  kill -TERM "$PID" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  kill -0 "$PID" 2>/dev/null && kill -KILL "$PID" 2>/dev/null || true
fi
rm -f .server.pid
