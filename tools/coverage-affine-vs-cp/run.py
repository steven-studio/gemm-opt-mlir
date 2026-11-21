#!/usr/bin/env python3
"""Runs the coverage corpus (cases.py) through CP, MLIR affine raising and Polly, and
classifies each (case, tool) into one of four classes:

  ACCEPT_PARALLEL  the tool models the whole nest and finds (or certifies) parallel
                   iterations: Polly/MLIR mark a loop parallel; CP proves the output cells
                   injective over (P, I, J) with the fold order verified (K the reduction)
  ACCEPT_SERIAL    the tool models the whole nest but finds no parallel loop (CP: fold order
                   verified, output injectivity not proved)
  REJECT           the tool refuses or models only part of the nest; the reason is kept
  ERROR            the tool crashed, timed out or printed nothing usable

  python3 run.py OUTDIR [--cases a,b] [--cp REPORTER] [--cp-main REPORTER]

Raw outputs (stdout/stderr of every tool) are kept under OUTDIR/<case>/.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cases import CASES  # noqa: E402

HOME = os.path.expanduser("~")
MLIR_OPT = f"{HOME}/work/llvm-project/build/bin/mlir-opt"
CLANG = f"{HOME}/work/clang24-build/bin/clang"
POLLY_OPT = f"{HOME}/work/polly24-build/opt-asserts/bin/opt"
CP_COMMON = ("--function=f --index-bit-width=16 --max-level=1 --candidates-per-group=1 "
             "--rlimit=2000000 --timeout-ms=20000 --max-points=512")
POLLY_DEFAULT = "-polly -polly-process-unprofitable -polly-ast-detect-parallel -polly-print-ast"
POLLY_PERMISSIVE = POLLY_DEFAULT + " -polly-allow-nonaffine -polly-allow-nonaffine-loops -polly-allow-nonaffine-branches"
REMARKS = "-pass-remarks-missed=polly-detect -pass-remarks-analysis=polly-detect"


def run(cmd, out, err, timeout=1800, shell=False):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, stdout=open(out, "w"), stderr=open(err, "w"), timeout=timeout, shell=shell)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    return rc, round(time.time() - t0, 2)


def depth(mlir):
    return mlir.count("scf.for")


# ---- CP ---------------------------------------------------------------------------------
def cp(reporter, src, d, flags, tag, strip_table=False):
    if strip_table:
        flags = re.sub(r"--table=\S+", "", flags)
        flags = flags.replace(",roots 2 and 3 are separate", "")
    args = [x for x in re.split(r"\s+(?=--)", (CP_COMMON + " " + flags).strip()) if x]
    out, err = os.path.join(d, f"{tag}.json"), os.path.join(d, f"{tag}.err")
    rc, secs = run([reporter, src] + args, out, err)
    res = {"cmd": " ".join([reporter, src] + [shlex.quote(a) for a in args]), "rc": rc, "seconds": secs}
    if rc != 0:
        res.update(cls="ERROR", reason=open(err).read().strip()[-300:] or f"rc {rc}")
        return res
    try:
        n = json.load(open(out))[0]["nests"][0]
    except Exception as e:  # noqa: BLE001
        res.update(cls="ERROR", reason=f"no record: {e}")
        return res
    g = n.get("gemm") or {}
    tiers = g.get("tiers", {})
    facts = tiers.get("facts", {})
    res.update(status=n.get("status"), fold=n.get("ordered_fold_verified"), refusals=n.get("refusals", []),
               assumptions=n.get("assumptions", []),
               computation_symbolic=tiers.get("computation", {}).get("symbolic", {}).get("status"),
               domain_symbolic=tiers.get("domain", {}).get("symbolic", {}).get("status"),
               coordinate=tiers.get("coordinate"), logical=tiers.get("theorem13", {}).get("logical_tier"),
               output_injective=facts.get("output_injective", {}).get("status"),
               single_contribution=facts.get("single_contribution", {}).get("status"),
               levels=[(l.get("axis"), str(l.get("lower"))[:60], str(l.get("upper"))[:90]) for l in g.get("levels", [])])
    if n.get("status") in ("VERIFIED", "VERIFIED_NOT_MAPPED") and n.get("ordered_fold_verified"):
        if res["output_injective"] == "PROVEN":
            res.update(cls="ACCEPT_PARALLEL",
                       reason="fold verified; output cells injective over (P,I,J) proved: P/I/J parallel, K reduction")
        else:
            res.update(cls="ACCEPT_SERIAL",
                       reason=f"fold verified; output injectivity {res['output_injective']}: no parallel axis certified")
    else:
        res.update(cls="REJECT", reason=(n.get("refusals") or [n.get("status")])[0])
    return res


# ---- MLIR affine --------------------------------------------------------------------------
def affine(src, d, nloops, manual=False):
    tag = "affine_manual" if manual else "affine"
    out, err = os.path.join(d, f"{tag}.mlir"), os.path.join(d, f"{tag}.err")
    passes = ["--affine-parallelize"] if manual else ["--raise-scf-to-affine", "--affine-raise-from-memref",
                                                     "--affine-parallelize"]
    cmd = [MLIR_OPT, src] + passes
    rc, secs = run(cmd, out, err)
    res = {"cmd": " ".join(cmd), "rc": rc, "seconds": secs}
    if rc != 0:
        res.update(cls="ERROR", reason=open(err).read().strip()[-300:])
        return res
    text = open(out).read()
    scf = text.count("scf.for")
    afor = len(re.findall(r"\baffine\.for\b", text))
    apar = len(re.findall(r"\baffine\.parallel\b", text))
    loads = len(re.findall(r"\bmemref\.load\b", text))
    stores = len(re.findall(r"\bmemref\.store\b", text))
    res.update(scf_for=scf, affine_for=afor, affine_parallel=apar, opaque_loads=loads, opaque_stores=stores)
    if scf:
        res.update(cls="REJECT", reason=f"{scf} of {nloops} loops not raised to affine (bound or step not affine)")
    elif apar:
        res.update(cls="ACCEPT_PARALLEL", reason=f"{apar} affine.parallel, {afor} affine.for"
                   + (f"; {loads} load(s)/{stores} store(s) left non-affine" if loads + stores else ""))
    else:
        res.update(cls="ACCEPT_SERIAL", reason=f"all {afor} loops affine, none parallelized"
                   + (f"; {loads} load(s)/{stores} store(s) left non-affine" if loads + stores else ""))
    return res


# ---- Polly --------------------------------------------------------------------------------
def polly(csrc, d, nloops, flags, tag):
    ll = os.path.join(d, "c.ll")
    if not os.path.exists(ll):
        rc, _ = run([CLANG, "-O1", "-S", "-emit-llvm", "-Xclang", "-disable-llvm-passes", csrc, "-o", ll],
                    os.path.join(d, "clang.out"), os.path.join(d, "clang.err"))
        if rc != 0:
            return {"cls": "ERROR", "reason": "clang: " + open(os.path.join(d, "clang.err")).read()[-300:]}
    out, err = os.path.join(d, f"{tag}.out"), os.path.join(d, f"{tag}.err")
    cmd = f"{POLLY_OPT} -passes='default<O3>' {flags} {REMARKS} -disable-output {ll}"
    rc, secs = run(cmd, out, err, shell=True)
    res = {"cmd": cmd, "rc": rc, "seconds": secs}
    if rc != 0:
        res.update(cls="ERROR", reason=open(err).read().strip()[-300:])
        return res
    text = open(out).read() + open(err).read()
    asts = text.split(":: isl ast ::")[1:]
    remarks = sorted(set(m.strip() for m in re.findall(r"remark: [^\n]*?:\s*(.*)", text)))
    res["remarks"] = remarks[:12]
    if not asts:
        res.update(cls="REJECT", reason="no SCoP: " + ("; ".join(remarks[:3]) or "no remark"))
        return res
    best = None
    for a in asts:
        dims = [len(m.split(",")) for m in re.findall(r"Stmt\w*\(([^)]*)\)", a)]
        dim = max(dims) if dims else 0
        if best is None or dim > best[0]:
            best = (dim, a)
    dim, a = best
    rtc = re.search(r"\nif \((.*)\)\n", a)
    res.update(scop_dims=dim, runtime_check=rtc.group(1) if rtc else None,
               known_parallel="#pragma known-parallel" in a or "#pragma omp parallel" in a)
    partial = any("Invalid Scop candidate" in r for r in remarks)
    if dim < nloops and partial:
        res.update(cls="REJECT", reason=f"SCoP covers {dim} of {nloops} loops (inner part only): "
                   + ("; ".join(remarks[:2]) or "no remark"))
    elif dim < nloops:
        # The whole nest is one SCoP, with inner loops boxed into one non-affine statement
        # (over-approximated accesses): modelled, but only the outer dims are scheduled.
        par = res["known_parallel"]
        res.update(cls="ACCEPT_PARALLEL" if par else "ACCEPT_SERIAL",
                   reason=f"whole nest a SCoP with {nloops - dim} inner loop(s) boxed as one non-affine statement; "
                   + ("an outer loop known-parallel" if par else "no loop known-parallel"))
    elif res["known_parallel"]:
        res.update(cls="ACCEPT_PARALLEL", reason="whole nest a SCoP, a loop known-parallel"
                   + (f"; run-time check: {res['runtime_check'][:80]}" if res["runtime_check"] not in (None, "1") else ""))
    else:
        res.update(cls="ACCEPT_SERIAL", reason="whole nest a SCoP, no loop known-parallel"
                   + (f"; run-time check: {res['runtime_check'][:80]}" if res["runtime_check"] not in (None, "1") else ""))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--cases", default="")
    ap.add_argument("--cp", default=os.path.join(HERE, "../../build/research/generalized-gemm/generalized-gemm-nest-report"))
    ap.add_argument("--cp-main", default=f"{HOME}/work/gemm-opt-mlir/build/research/generalized-gemm/generalized-gemm-nest-report")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    want = set(filter(None, a.cases.split(",")))
    results = []
    for c in CASES:
        if want and c["name"] not in want:
            continue
        d = os.path.join(a.outdir, c["name"])
        os.makedirs(d, exist_ok=True)
        src, csrc = os.path.join(d, "src.mlir"), os.path.join(d, "src.c")
        open(src, "w").write(c["mlir"])
        open(csrc, "w").write(c["c"])
        n = depth(c["mlir"])
        r = {"name": c["name"], "category": c["category"], "note": c["note"], "loops": n,
             "cp": cp(os.path.abspath(a.cp), src, d, c["cp"], "cp"),
             "cp_main": cp(a.cp_main, src, d, c["cp"], "cp_main", strip_table=True),
             "mlir_affine": affine(src, d, n),
             "mlir_affine_manual": None,
             "polly": polly(csrc, d, n, POLLY_DEFAULT, "polly"),
             "polly_permissive": polly(csrc, d, n, POLLY_PERMISSIVE, "polly_permissive")}
        if c.get("affine"):
            msrc = os.path.join(d, "manual_affine.mlir")
            open(msrc, "w").write(c["affine"])
            r["mlir_affine_manual"] = affine(msrc, d, n, manual=True)
        results.append(r)
        print(f"{c['name']:20s} CP={r['cp']['cls']:15s} CPmain={r['cp_main']['cls']:15s} "
              f"affine={r['mlir_affine']['cls']:15s} affine_hand={(r['mlir_affine_manual'] or {}).get('cls', '-'):15s} "
              f"polly={r['polly']['cls']:15s} "
              f"polly+={r['polly_permissive']['cls']}", flush=True)
    meta = {"mlir_opt": MLIR_OPT, "clang": CLANG, "polly_opt": POLLY_OPT, "cp": os.path.abspath(a.cp),
            "cp_main": a.cp_main, "cp_common": CP_COMMON, "polly_default": POLLY_DEFAULT,
            "polly_permissive": POLLY_PERMISSIVE}
    with open(os.path.join(a.outdir, "results.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)
    print("raw:", os.path.join(a.outdir, "results.json"))


if __name__ == "__main__":
    main()
