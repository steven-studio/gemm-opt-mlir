#!/usr/bin/env python3
"""Join sample_stacks.py output: per sample, RSS and the generalized_gemm call chain.
  summarize_stacks.py PREFIX
Prints t, RSS (GB), HWM (GB), chain outermost -> innermost (project frames only)."""
import re
import sys

prefix = sys.argv[1]
rss = [l.rstrip("\n").split("\t") for l in open(prefix + ".rss.tsv")][1:]
traces, cur = [], None
for line in open(prefix + ".gdb.log", errors="replace"):
    if "received signal SIGINT" in line:
        cur = []
        traces.append(cur)
    elif line.startswith("#") and cur is not None:
        m = re.search(r"(mlir::generalized_gemm::[^\s(]*(?:\([^)]*\))?[^\s(]*)", line)
        if m:
            name = m.group(1).replace("mlir::generalized_gemm::", "")
            name = re.sub(r"\(anonymous namespace\)::", "", name)
            name = re.sub(r"\(.*", "", name)
            name = re.sub(r"std::_Function_handler<.*", "lambda", name)
            cur.append(name)
        elif " in main " in line:
            cur.append("main")
        elif "libz3" in line and (not cur or cur[-1] != "z3"):
            cur.append("z3")
for i, (t, r, h, n) in enumerate(rss):
    chain = traces[i] if i < len(traces) else []
    chain = [c for c in reversed(chain) if c != "main"]
    dedup = []
    for c in chain:
        if not dedup or dedup[-1] != c:
            dedup.append(c)
    print(f"{float(t):7.0f}s rss={int(r)/2**20:6.2f}G hwm={int(h)/2**20:6.2f}G  " + " > ".join(dedup[-7:]))
