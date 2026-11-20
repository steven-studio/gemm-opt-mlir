#!/usr/bin/env python3
"""Fixed BLOCK_M vs the generated multi-version (16/64/128) grouped GEMM.

  python3 bench_block_m.py RECORD.json OUTDIR [--warmup 25] [--iters 200]

Variants (all from the same record, same kernel template, same dispatcher function):
  multi     generator default --thresholds 16,64: buckets <=16 / 17..64 / >64,
            BLOCK_M 16 / 64 / 128 (the generated multi-version).
  fixed<B>  generator with --thresholds '' : one bucket [1, hi] (BLOCK_M 128 for scope
            [0,128]); pure generator output, no patching ("native").
  split<n>_<B>  control: the native fixed kernel launched once per multi bucket, same work
            items as fixed<B>, as many launches as multi (harness-composed).
  fixed16, fixed64
            the generator has no flag for a fixed small BLOCK_M (BLOCK_M is derived from the
            bucket's upper bound). These reuse the multi module's kernel for that BLOCK_M
            (the kernel body is the same template) and replace BUCKETS with one bucket
            [1, hi]: the dispatcher splits each group into ceil(M_p / BLOCK_M) items.
            This is a harness-side patch of the generated module, marked as such.

Each variant gets exactly the same inputs (x, w, ids, offsets, rows, C start value) per
workload and is checked against a float64 reference before timing.

Two timings per (workload, variant), all raw samples saved:
  kernel_us  device time of the launches only: the worklist and its device tensors are
             built once beforehand; per iteration, CUDA events around the bucket launches
             (span on the GPU, gaps between launches included). Also, summed device kernel
             durations from torch.profiler over all iterations.
  e2e_us     host wall clock of one `grouped_gemm(...)` call (run-time checks, .tolist()
             syncs, O(E) worklist, H2D copies of the work items, launches) + synchronize.
C is updated in place; repeated iterations accumulate into it, which does not change the
work done.
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import grouped_gemm_triton as gen  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_variants(record, outdir):
    v = {}
    p_multi = os.path.join(outdir, "kernels_multi.py")
    assert gen.main([record, "-o", p_multi]) == 0
    p_fixed = os.path.join(outdir, "kernels_fixed_native.py")
    assert gen.main([record, "-o", p_fixed, "--thresholds", ""]) == 0
    v["multi"] = load(p_multi, "k_multi")
    native = load(p_fixed, "k_fixed_native")
    # Named by its BLOCK_M: 128 for scope [0,128]; the bucket's pow2 bound otherwise.
    v[f"fixed{native.BUCKETS[0][3]}"] = native
    hi = v["multi"].M_SCOPE[1]
    # Control: the native fixed128 kernel, launched once per bucket of the multi module
    # (the same partition of the groups), so it does exactly fixed128's work items in as
    # many launches as multi. Harness-composed, marked as such.
    split = load(p_fixed, "k_split_128")
    (name128, _, _, bm128, bn128, bk128, w128), = split.BUCKETS
    if len(v["multi"].BUCKETS) > 1:
        split.BUCKETS = []
        for b in v["multi"].BUCKETS:
            nm = "split_" + b[0]
            split.KERNELS[nm] = split.KERNELS[name128]
            split.BUCKETS.append((nm, b[1], b[2], bm128, bn128, bk128, w128))
        v[f"split{len(split.BUCKETS)}_{bm128}"] = split
    for bm in (16, 64):
        if f"fixed{bm}" in v:
            continue
        mod = load(p_multi, f"k_fixed{bm}")
        src = [b for b in mod.BUCKETS if b[3] == bm]
        if not src:
            continue
        name, _, _, _, bn, bk, warps = src[0]
        mod.BUCKETS = [(name, 1, hi, bm, bn, bk, warps)]
        v[f"fixed{bm}"] = mod
    return v


def workloads(E, hi):
    w = {}
    if E == 4:
        w["mixed_8_40_100_0"] = [8, 40, 100, 0]
        w["mixed_1_17_65_128"] = [1, 17, 65, 128]
        w["edges_16_17_64_65"] = [16, 17, 64, 65]
        w["skew_1_1_1_128"] = [1, 1, 1, 128]
    elif E == 3:
        w["mixed_8_40_100"] = [8, 40, 100]
        w["mixed_0_8_40"] = [0, 8, 40]
        w["edges_16_17_64"] = [16, 17, 64]
        w["edges_64_65_128"] = [64, 65, 128]
    w["all_small_8"] = [8] * E
    w["all_16"] = [16] * E
    w["all_64"] = [64] * E
    w["all_max"] = [hi] * E
    return {k: [min(m, hi) for m in v] for k, v in w.items()}


def inputs(mod, rows, seed):
    E, K, N = mod.E, mod.K, mod.N
    g = torch.Generator().manual_seed(seed)
    T = max(1, sum(rows))
    x = torch.randn(T, K, generator=g).cuda()
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


def prepared_launches(mod, x, w, c, ids, offs, rows):
    """The dispatcher's launches with the worklist built once (kernel-only timing)."""
    lists = mod.build_worklist(rows.tolist())
    calls = []
    for name, lo, hi, bm, bn, bk, warps in mod.BUCKETS:
        gl, rl = lists[name]
        if not gl:
            continue
        wg = torch.tensor(gl, dtype=torch.int32, device="cuda")
        wr = torch.tensor(rl, dtype=torch.int32, device="cuda")
        grid = (len(gl), triton.cdiv(mod.N, bn))
        calls.append((mod.KERNELS[name], grid, wg, wr, bm, bn, bk, warps))

    def run():
        for k, grid, wg, wr, bm, bn, bk, warps in calls:
            k[grid](x, w, c, ids, offs, wg, wr, rows,
                    x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.stride(2),
                    c.stride(0), c.stride(1),
                    N=mod.N, K=mod.K, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps)
    info = [{"grid": list(g), "BLOCK_M": bm, "items": len(wg),
             "work": sorted(zip(wg.tolist(), wr.tolist()))} for _, g, wg, wr, bm, _, _, _ in calls]
    return run, info


def stats(xs):
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]
    return {"median": statistics.median(s), "mean": statistics.fmean(s), "p10": q(0.1),
            "p90": q(0.9), "min": s[0], "n": len(s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("outdir")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    variants = make_variants(a.record, a.outdir)
    any_mod = variants["multi"]
    NATIVE = next(k for k, m in variants.items() if k.startswith("fixed") and m.__name__ == "k_fixed_native")
    E, hi = any_mod.E, any_mod.M_SCOPE[1]
    out = {"record": os.path.abspath(a.record), "E": E, "N": any_mod.N, "K": any_mod.K,
           "scope": any_mod.M_SCOPE, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
           "triton": triton.__version__, "warmup": a.warmup, "iters": a.iters,
           "variants": {k: {"buckets": m.BUCKETS,
                            "provenance": ("native generator output" if k in ("multi", NATIVE) else
                                           "harness-composed: native single-bucket kernel, one launch per multi bucket"
                                           if k.startswith("split") else
                                           "harness-patched: BUCKETS replaced by one bucket, multi module's kernel"),
                            "patched": k not in ("multi", NATIVE)} for k, m in variants.items()},
           "results": []}
    for wname, rows in workloads(E, hi).items():
        base = inputs(any_mod, rows, seed=1234)
        ref = reference(*base)
        for vname, mod in variants.items():
            x, w, c0, ids, offs, rws = base
            # Correctness on these exact inputs first.
            c = c0.clone()
            mod.grouped_gemm(x, w, c, ids, offs, rws)
            torch.cuda.synchronize()
            err = (c.double() - ref).abs().max().item()
            ok = torch.allclose(c.double(), ref, atol=1e-4, rtol=1e-4)

            # Kernel-only.
            c = c0.clone()
            run, info = prepared_launches(mod, x, w, c, ids, offs, rws)
            for _ in range(a.warmup):
                run()
            torch.cuda.synchronize()
            kern = []
            for _ in range(a.iters):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                run()
                e.record()
                e.synchronize()
                kern.append(s.elapsed_time(e) * 1e3)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(a.iters):
                    run()
                torch.cuda.synchronize()
            dev_total = sum(ev.device_time_total for ev in prof.key_averages()
                            if "grouped_gemm" in ev.key)
            dev_per_iter = dev_total / a.iters

            # End to end: the dispatcher call.
            c = c0.clone()
            for _ in range(a.warmup):
                mod.grouped_gemm(x, w, c, ids, offs, rws)
            torch.cuda.synchronize()
            e2e = []
            for _ in range(a.iters):
                t0 = time.perf_counter()
                mod.grouped_gemm(x, w, c, ids, offs, rws)
                torch.cuda.synchronize()
                e2e.append((time.perf_counter() - t0) * 1e6)

            work = sorted((p, r0, l["BLOCK_M"]) for l in info for p, r0 in l["work"])
            r = {"workload": wname, "rows": rows, "variant": vname, "launches": info,
                 "n_launches": len(info), "work_items": work,
                 "max_abs_err": err, "allclose": ok,
                 "kernel_event_us": stats(kern), "kernel_profiler_device_us_per_iter": dev_per_iter,
                 "e2e_us": stats(e2e), "raw_kernel_event_us": kern, "raw_e2e_us": e2e}
            out["results"].append(r)
            print(f"{wname:22s} {vname:11s} ok={ok} err={err:.1e} launches={[(l['BLOCK_M'], l['items']) for l in info]} "
                  f"kernel(evt) med={r['kernel_event_us']['median']:.1f}us "
                  f"kernel(prof) {dev_per_iter:.1f}us e2e med={r['e2e_us']['median']:.1f}us")
    out["native_fixed"] = NATIVE
    # The control must do exactly the native fixed version's work items.
    by = {(r["workload"], r["variant"]): r for r in out["results"]}
    for (wl, vn), r in by.items():
        if vn.startswith("split"):
            same = r["work_items"] == by[(wl, NATIVE)]["work_items"]
            r["same_work_items_as_native_fixed"] = same
            print(f"{wl}: {vn} work items == {NATIVE}: {same}")
    path = os.path.join(a.outdir, "bench.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("raw:", path)
    return 0 if all(r["allclose"] for r in out["results"]) else 1


if __name__ == "__main__":
    sys.exit(main())
