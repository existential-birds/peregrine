#!/bin/bash
# macOS memory probe for bench.py: prints the server's phys_footprint in
# bytes. On Apple unified memory this includes Metal wired allocations,
# unlike Linux cgroup counters. Prints nothing (0) if no server is running.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .server.pid ] || exit 0
PID="$(cat .server.pid)"
kill -0 "$PID" 2>/dev/null || exit 0
KB="$(footprint "$PID" 2>/dev/null | awk -F': ' '/phys_footprint:/{print $2; exit}')"
[ -n "$KB" ] || exit 0
echo $((KB * 1024))
