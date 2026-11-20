#!/usr/bin/env python3
"""Compute-bound sizes: multi-version buckets vs native fixed BLOCK_M.

  python3 bench_large.py RECORD.json OUTDIR [--shapes 2048x2048,2048x4096] [--experts 8,64]
                         [--warmup 3] [--iters 30]

No verified record exists at these sizes: the record's cost is ~2 kB per traced event
(events = E*N*K*sum_{m<=CAP} m; see reports/2026-11-20-r2), so N = K = 2048 needs tens of
GB even at CAP = 1. This script therefore takes everything except the shape from RECORD
(read_record: the update, the views, the group axis, the scope of M_p, the fold-order
permission) and overrides E, N, K through the generator's own API (make_buckets/generate),
writing a header that says the shape is NOT covered by the record. The kernels and the
dispatcher are the generator's, unchanged; the variants are produced with the generator's
own options (default thresholds, --fixed-block-m). Results are checked against a float64
reference before timing; they are performance evidence, not verification evidence.

Variants: multi (16,64 thresholds: BLOCK_M 16/64/128), fixed128 / fixed64 / fixed16
(--fixed-block-m), split3_128 (harness-composed control: the fixed128 kernel launched once
per multi bucket, same work items as fixed128). Timings per (shape, E, workload, variant):
CUDA-event span of the launches, profiler device time, end-to-end grouped_gemm (with the
run-time guards, and without them: check_inputs=False), all raw samples saved.
"""

import argparse
import dataclasses
import json
import os
import random
import sys

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import grouped_gemm_triton as gen  # noqa: E402
from bench_block_m import load, make_variants, measure  # noqa: E402

OVERRIDE = ("\n# SHAPE OVERRIDE (bench_large.py): E = {E}, N = {N}, K = {K} replace the record's "
            "E = {e0}, N = {n0}, K = {k0}.\n# This shape is NOT covered by the record: performance "
            "only. Everything else is the record's.")


def emitter(record, E, N, K):
    with open(record) as f:
        g0 = gen.read_record(json.load(f))

    def emit(path, extra):
        g = dataclasses.replace(g0, groups=E, n=N, k=K, buckets=[])
        fixed = None
        if extra[:1] == ["--fixed-block-m"]:
            fixed = int(extra[1])
        g.buckets = gen.make_buckets(g, [16, 64], fixed)
        mode = OVERRIDE.format(E=E, N=N, K=K, e0=g0.groups, n0=g0.n, k0=g0.k)
        if fixed:
            mode += f"\n# Fixed BLOCK_M = {fixed} (--fixed-block-m)."
        with open(path, "w") as f:
            f.write(gen.generate(g, mode))
        return 0
    return emit, g0


def workloads(E, hi, seed=7):
    rnd = random.Random(seed)
    small = [rnd.randint(1, 16) for _ in range(E)]
    mixed = [rnd.randint(1, hi) for _ in range(E)]
    skew = [hi if p % 4 == 0 else rnd.randint(1, 16) for p in range(E)]
    mid = [rnd.randint(17, 64) for _ in range(E)]
    edges = [(16, 17, 64, 65)[p % 4] for p in range(E)]
    return {"small_1_16": small, "mid_17_64": mid, "mixed_1_128": mixed,
            "skew_3small_1max": skew, "edges_16_17_64_65": edges, "all_max": [hi] * E}


def inputs(E, N, K, rows, seed):
    g = torch.Generator().manual_seed(seed)
    T = max(1, sum(rows))
    x = (torch.randn(T, K, generator=g) / K ** 0.5).cuda()
    w = torch.randn(E, K, N, generator=g).cuda()
    ids = torch.randint(0, T, (T,), generator=g, dtype=torch.int32).cuda()
    offs, cur = [], 0
    for m in rows:
        offs.append(cur)
        cur += m
    c0 = torch.randn(T, N, generator=g).cuda()
    return (x, w, c0, ids, torch.tensor(offs, dtype=torch.int32).cuda(),
            torch.tensor(rows, dtype=torch.int32).cuda())


def reference(x, w, c0, ids, offs, rows):
    ref = c0.double().clone()
    for p, (o, m) in enumerate(zip(offs.tolist(), rows.tolist())):
        if m:
            ref[o:o + m] += x[ids[o:o + m].long()].double() @ w[p].double()
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("outdir")
    ap.add_argument("--shapes", default="2048x2048,2048x4096", help="NxK,...")
    ap.add_argument("--experts", default="8,64")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    out = {"record": os.path.abspath(a.record), "gpu": torch.cuda.get_device_name(),
           "torch": torch.__version__, "triton": triton.__version__, "warmup": a.warmup,
           "iters": a.iters, "note": "shape overridden: not covered by the record", "configs": []}
    for shape in a.shapes.split(","):
        N, K = map(int, shape.split("x"))
        for E in map(int, a.experts.split(",")):
            tag = f"e{E}_n{N}_k{K}"
            d = os.path.join(a.outdir, tag)
            os.makedirs(d, exist_ok=True)
            emit, g0 = emitter(a.record, E, N, K)
            variants = make_variants(a.record, d, emit)
            hi = variants["multi"].M_SCOPE[1]
            cfg = {"E": E, "N": N, "K": K, "scope": [g0.m_scope[0], hi],
                   "variants": {k: {"buckets": m.BUCKETS, "harness_composed": k.startswith("split")}
                                for k, m in variants.items()}, "results": []}
            for wname, rows in workloads(E, hi).items():
                base = inputs(E, N, K, rows, seed=1234)
                ref = reference(*base)
                flops = 2.0 * sum(rows) * N * K
                for vname, mod in variants.items():
                    r = measure(mod, base, ref, a.warmup, a.iters, e2e_unchecked=True)
                    r = {"workload": wname, "rows": rows, "variant": vname, "useful_gflop": flops / 1e9, **r}
                    cfg["results"].append(r)
                    print(f"{tag} {wname:18s} {vname:10s} ok={r['allclose']} err={r['max_abs_err']:.1e} "
                          f"L={r['n_launches']} evt={r['kernel_event_us']['median']:.0f}us "
                          f"dev={r['kernel_profiler_device_us_per_iter']:.0f}us "
                          f"e2e={r['e2e_us']['median']:.0f}us e2e_nochk={r['e2e_unchecked_us']['median']:.0f}us "
                          f"{flops / r['e2e_us']['median'] / 1e6:.2f}TFLOP/s", flush=True)
                del base, ref
                torch.cuda.empty_cache()
            by = {(r["workload"], r["variant"]): r for r in cfg["results"]}
            for (wl, vn), r in by.items():
                if vn.startswith("split"):
                    r["same_work_items_as_fixed128"] = r["work_items"] == by[(wl, "fixed128")]["work_items"]
                for key in ("e2e_us", "kernel_event_us"):
                    r[f"speedup_vs_fixed128_{key}"] = (by[(wl, "fixed128")][key]["median"] / r[key]["median"])
            out["configs"].append(cfg)
            del variants
            torch.cuda.empty_cache()
    path = os.path.join(a.outdir, "bench_large.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("raw:", path)
    ok = all(r["allclose"] for c in out["configs"] for r in c["results"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
