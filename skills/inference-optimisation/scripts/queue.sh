#!/usr/bin/env bash
# Serialized GPU queue: runs bench.py on each config in order, one GPU job at a time across all callers.
#   queue.sh ctl-a cand ctl-b            # interleave controls around each candidate
#   BENCH_FLAGS="--long 131072" queue.sh cand
# Detach long queues so they survive agent/terminal restarts:
#   nohup bash queue.sh ctl-a cand ctl-b > queue.log 2>&1 < /dev/null & disown
# Run from your experiment directory (the one holding configs/ and ledger.jsonl).
set -uo pipefail
LOCK=${QUEUE_LOCK:-/tmp/inference-hillclimb.lock}
BENCH_PY="${BENCH:-$(dirname "$0")/bench.py}"
/usr/bin/python3 - "$LOCK" "$BENCH_PY" "$@" <<'PY'
import fcntl, os, shlex, subprocess, sys

lock_path, bench_py, *names = sys.argv[1:]
extra = shlex.split(os.environ.get("BENCH_FLAGS", ""))
with open(lock_path, "w") as lf:
    fcntl.flock(lf, fcntl.LOCK_EX)
    for n in names:
        log_path = f"runs-{n}.log"
        with open(log_path, "w") as out:
            subprocess.run(["/usr/bin/python3", bench_py, "run", f"configs/{n}.json", *extra], stdout=out, stderr=subprocess.STDOUT)
        tail = open(log_path, "rb").read()[-300:].decode("utf-8", "replace").replace("\n", " ")[:200]
        print(f"done {n}: {tail}", flush=True)
    print("QUEUE_DONE", *names, flush=True)
PY
