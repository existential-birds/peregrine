#!/bin/bash
# macOS memory probe for bench.py: prints the server's phys_footprint in
# bytes via libproc proc_pid_rusage (RUSAGE_INFO_V4). On Apple unified memory
# ri_phys_footprint includes Metal wired allocations (matching `footprint`),
# runs in <1 ms, and never suspends the target task.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .server.pid ] || exit 0
PID="$(cat .server.pid)"
kill -0 "$PID" 2>/dev/null || exit 0
/usr/bin/python3 - "$PID" <<'PY' 2>/dev/null || exit 0
import ctypes, sys
pid = int(sys.argv[1])
buf = (ctypes.c_uint64 * 40)()
rc = ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(pid, 4, ctypes.byref(buf))
if rc == 0 and buf[9] > 0:
    print(buf[9])
PY
