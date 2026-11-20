#!/usr/bin/env python3
"""Sample RSS and the call stack of a long-running process (here: the nest reporter).

  sample_stacks.py OUT_PREFIX PERIOD_S MAX_SAMPLES -- PROGRAM ARGS...

perf is unavailable (perf_event_paranoid=4) and ptrace_scope=1 forbids attaching, so gdb
starts PROGRAM as its own child. This script sends SIGINT to the inferior every PERIOD_S
seconds; gdb stops it (`handle SIGINT stop nopass`: the program never sees the signal),
prints a backtrace, and continues. RSS comes from /proc/PID/status at each sample.
Writes OUT_PREFIX.rss.tsv (t, VmRSS kB, VmHWM kB) and OUT_PREFIX.gdb.log (backtraces).
stdout of PROGRAM goes to OUT_PREFIX.stdout.
"""
import os
import signal
import subprocess
import sys
import time

prefix, period, nmax = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
prog = sys.argv[sys.argv.index("--") + 1:]
# No startup shell (it re-splits arguments with spaces); sh -c exec keeps the pid and
# redirects the program's stdout.
cmds = ["set debuginfod enabled off", "set startup-with-shell off", "set pagination off",
        "set confirm off", "set print frame-arguments none",
        "handle SIGINT stop print nopass", "run"]
prog = ["/bin/sh", "-c", 'exec "$0" "$@" > "' + prefix + '.stdout"'] + prog
for _ in range(nmax):
    cmds += ["bt 60", "continue"]
args = ["gdb", "-batch", "-nx"]
for c in cmds:
    args += ["-ex", c]
args += ["--args"] + prog
log = open(prefix + ".gdb.log", "w")
g = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)


def inferior():
    try:
        out = subprocess.run(["pgrep", "-P", str(g.pid)], capture_output=True, text=True).stdout.split()
        return int(out[0]) if out else None
    except Exception:
        return None


rss = open(prefix + ".rss.tsv", "w")
rss.write("t_s\tVmRSS_kB\tVmHWM_kB\tsample\n")
t0 = time.time()
n = 0
pid = None
while g.poll() is None:
    time.sleep(period)
    pid = pid or inferior()
    if pid is None:
        continue
    try:
        st = dict(l.split(":", 1) for l in open(f"/proc/{pid}/status"))
        r = st["VmRSS"].split()[0]
        h = st["VmHWM"].split()[0]
    except Exception:
        continue
    n += 1
    rss.write(f"{time.time() - t0:.1f}\t{r}\t{h}\t{n}\n")
    rss.flush()
    log.flush()
    if n <= nmax:
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            break
g.wait()
print(f"exit {g.returncode}; {n} samples; {prefix}.rss.tsv {prefix}.gdb.log")
