#!/usr/bin/env python3
"""Runs a module from scatter_gemm_triton.py against the nest itself.

  python3 test_scatter_gemm_triton.py SRC.mlir RECORD.json MODULE.py [--function f] [--json OUT]

The reference interprets the scf source literally, one scalar operation at a time in program
order, in float64 (independent of the module's re-emitted index computation). Scenarios:
  injective   the address tables' cells pairwise distinct (when they can be)
  repeated    two cells equal: repeated targets
  random      cells uniform over the record's range
  oob         a cell past the table range that reaches an out-of-bounds row: refused
  alias       C sharing storage with A: refused
Each prints the path the module took and the largest error against the reference.
"""

import argparse
import importlib.util
import itertools
import json
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import scatter_gemm_triton as gen  # noqa: E402


def interpret(nest, arrays):
    """The nest in program order, float64. arrays: name -> numpy array (copied)."""
    env = {}
    order = [lp["iv"] for lp in nest.loops]
    loops = {lp["iv"]: lp for lp in nest.loops}

    def ival(name):
        if name in nest.const:
            return nest.const[name]
        if name in env:
            return env[name]
        if name in nest.expr:
            op, a, b = nest.expr[name]
            x = ival(a)
            if op == "id":
                return x
            y = ival(b)
            ops = {"addi": lambda: x + y, "subi": lambda: x - y, "muli": lambda: x * y,
                   "divui": lambda: x // y, "divsi": lambda: x // y, "remui": lambda: x % y, "remsi": lambda: x % y}
            return ops[op]()
        if name in nest.loads:
            mem, idx = nest.loads[name]
            at = tuple(ival(e) for e in idx)
            arr = arrays[mem]
            if any(i < 0 or i >= d for i, d in zip(at, arr.shape)):
                raise IndexError(f"{mem}{list(at)} out of bounds")
            return int(arr[at])
        if name in nest.argnames:
            return int(arrays[name])
        raise KeyError(name)

    ranges = [range(nest.const[loops[iv]["lo"]], nest.const[loops[iv]["hi"]], nest.const[loops[iv]["step"]])
              for iv in order]
    cname, cidx = nest.C
    aname, aidx = nest.A
    bname, bidx = nest.B
    for point in itertools.product(*ranges):
        env = dict(zip(order, point))
        h = tuple(ival(e) for e in cidx)
        f = tuple(ival(e) for e in aidx)
        g = tuple(ival(e) for e in bidx)
        for arr, at, nm in ((arrays[cname], h, cname), (arrays[aname], f, aname), (arrays[bname], g, bname)):
            if any(i < 0 or i >= d for i, d in zip(at, arr.shape)):
                raise IndexError(f"{nm}{list(at)} out of bounds")
        arrays[cname][h] += arrays[aname][f] * arrays[bname][g]
    return arrays[cname]


def load(path):
    spec = importlib.util.spec_from_file_location("scatter_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("record")
    ap.add_argument("module")
    ap.add_argument("--function", default="f")
    ap.add_argument("--json")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    nest = gen.Nest(a.src, a.function)
    mod = load(a.module)
    rnd = random.Random(a.seed)
    tables = {p["argument"]: p for p in mod.PREMISES if p["kind"] == "table"}
    distinct = {p["argument"] for p in mod.PREMISES if p["kind"] == "distinct"}
    plan = mod.PLAN
    results = []

    def make(kind):
        """Arguments (torch, on the GPU) and the float64 copies the reference reads."""
        args, ref = [], {}
        for n, (name, ty) in enumerate(nest.args):
            if ty.startswith("memref"):
                dims, el = gen.memref_shape(ty)
                if el == "f32":
                    t = torch.randn(*dims, generator=torch.Generator().manual_seed(rnd.randrange(1 << 30)))
                    ref[name] = t.double().numpy().copy()
                    args.append(t.cuda())
                else:
                    p = tables.get(n)
                    lo, hi = (p["lower"], p["upper"]) if p else (0, 0)
                    cells = int(np.prod(dims))
                    vals = [rnd.randint(lo, hi) for _ in range(cells)]
                    if kind == "injective" and hi - lo + 1 >= cells:
                        vals = rnd.sample(range(lo, hi + 1), cells)
                    elif kind == "repeated" and cells >= 2:
                        vals = [rnd.randint(lo, hi) for _ in range(cells)]
                        vals[1] = vals[0]
                    elif kind == "oob" and n in tables:
                        vals[0] = hi + 1 + rnd.randint(0, 2)
                    arr = np.array(vals, dtype=np.int64).reshape(dims)
                    ref[name] = arr.copy()
                    args.append(torch.tensor(arr, dtype=torch.int64).cuda())
            else:
                p = next((q for q in mod.PREMISES if q["kind"] == "range" and q["argument"] == n), None)
                v = p["upper"] if p else 1
                ref[name] = v
                args.append(v)
        return args, ref

    for kind in ("injective", "repeated", "random", "oob", "alias"):
        if kind == "injective" and not distinct and not any(
                p["upper"] - p["lower"] + 1 >= int(np.prod(gen.memref_shape(nest.types[nest.argnames[t]])[0]))
                for t, p in tables.items()):
            results.append({"scenario": kind, "skipped": "no table can be injective at this size"})
            print(f"{kind:10s} skipped: no table can be injective at this size")
            continue
        args, ref = make(kind)
        cname = nest.C[0]
        ci = nest.argnames.index(cname)
        if kind == "alias":
            ai = nest.argnames.index(nest.A[0])
            if args[ai].numel() >= args[ci].numel():
                args[ci] = args[ai].reshape(-1)[:args[ci].numel()].reshape(args[ci].shape)
        expect_ref = None
        try:
            expect_ref = interpret(nest, {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ref.items()})
        except IndexError as e:
            expect_ref = e
        try:
            path = getattr(mod, a.function)(*args)
            torch.cuda.synchronize()
            out = args[ci].double().cpu().numpy()
            if isinstance(expect_ref, Exception):
                r = {"scenario": kind, "path": path, "ok": False, "why": f"ran, but the nest is undefined: {expect_ref}"}
            else:
                err = float(np.abs(out - expect_ref).max())
                r = {"scenario": kind, "path": path, "max_abs_err": err, "ok": bool(np.allclose(out, expect_ref, atol=1e-4, rtol=1e-4))}
        except ValueError as e:
            ok = (kind == "oob" and "undefined" in str(e)) or (kind == "alias" and "overlap" in str(e))
            r = {"scenario": kind, "path": "refused", "why": str(e), "ok": ok}
        tvals = {nest.argnames[t]: args[t].reshape(-1).tolist() for t in tables}
        r["tables"] = tvals
        r["plan"] = plan
        results.append(r)
        print(f"{kind:10s} tables={tvals} path={r['path']:13s} ok={r['ok']} "
              + (f"err={r['max_abs_err']:.1e}" if "max_abs_err" in r else r.get("why", "")))
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)
    return 0 if all(r.get("ok", True) for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
