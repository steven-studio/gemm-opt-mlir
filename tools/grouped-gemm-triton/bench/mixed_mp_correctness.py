#!/usr/bin/env python3
"""Mixed per-group M_p correctness of the generated Triton grouped GEMM.

  python3 mixed_mp_correctness.py RECORD.json OUTDIR [--seeds 3]

Generates the module from RECORD with the generator's own CLI (default thresholds 16,64),
then for each per-group row-count combination runs the generated `grouped_gemm`
dispatcher and compares against a float64 reference that keeps the gather / output index
semantics:  C[off[p] + i, :] += X[ids[off[p] + i], :] @ W[p]  for i < M_p, every other row
of C untouched (bit for bit).

Two layouts per combination:
  padded  - the record's own layout: off[p] = p*CAP, identity ids (the source loop nest
            with group p bounded by M_p).
  gather  - non-identity ids (random tokens with repeats), compact offsets with a gap
            of untouched rows between groups, groups in shuffled order in C.

The kernels are wrapped (not changed) so every launch is logged: bucket, grid, BLOCK_M and
the (group, first row) work items it received. From that log the BLOCK_M each group was
actually launched with is reported and checked against the bucket rule
(<=16 -> 16, 17..64 -> 64, >64 -> 128).
"""

import argparse
import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import grouped_gemm_triton as gen  # noqa: E402

ATOL = RTOL = 1e-4


def load_module(path):
    spec = importlib.util.spec_from_file_location("gen_mixed_mp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LaunchLog:
    """Proxy for a @triton.jit kernel: `k[grid](*args, **kw)` is logged, then launched."""

    def __init__(self, name, kernel, log):
        self.name, self.kernel, self.log = name, kernel, log

    def __getitem__(self, grid):
        launcher = self.kernel[grid]

        def run(*args, **kw):
            # Positional args 5 and 6 are the work item arrays (group, first row).
            wg, wr = args[5].tolist(), args[6].tolist()
            self.log.append({"bucket": self.name, "grid": list(grid), "BLOCK_M": kw["BLOCK_M"],
                             "BLOCK_N": kw["BLOCK_N"], "BLOCK_K": kw["BLOCK_K"],
                             "num_warps": kw["num_warps"], "items": list(zip(wg, wr))})
            return launcher(*args, **kw)
        return run


BUCKETS = None  # Set from the generated module: (name, lo, hi, BLOCK_M, ...).


def expected_block_m(m):
    """The bucket rule of the generated module (16/64/128 for the default thresholds and
    scope [0,128])."""
    return next(b[3] for b in BUCKETS if b[1] <= m <= b[2])


def run_case(mod, rows, layout, seed, log):
    E, K, N = mod.E, mod.K, mod.N
    cap = mod.M_SCOPE[1]
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(E, K, N, generator=g).to(dev)
    if layout == "padded":
        x = torch.randn(E * cap, K, generator=g).to(dev)
        ids = torch.arange(E * cap, dtype=torch.int32)
        offs = [p * cap for p in range(E)]
        total = E * cap
    else:
        T = 37  # tokens; ids repeat and are not sorted
        x = torch.randn(T, K, generator=g).to(dev)
        order = torch.randperm(E, generator=g).tolist()  # groups placed in shuffled order
        offs, cur = [0] * E, 3
        for p in order:
            offs[p] = cur
            cur += rows[p] + 2  # gap of 2 untouched rows after each group
        total = cur + 1
        ids = torch.randint(0, T, (total,), generator=g, dtype=torch.int32)
    c0 = torch.randn(total, N, generator=g).to(dev)
    ids, c = ids.to(dev), c0.clone()
    offs_t = torch.tensor(offs, dtype=torch.int32, device=dev)
    rows_t = torch.tensor(rows, dtype=torch.int32, device=dev)

    log.clear()
    mod.grouped_gemm(x, w, c, ids, offs_t, rows_t)
    torch.cuda.synchronize()

    ref = c0.double().clone()
    touched = torch.zeros(total, dtype=torch.bool, device=dev)
    for p in range(E):
        o, m = offs[p], rows[p]
        if m:
            ref[o:o + m] += x[ids[o:o + m].long()].double() @ w[p].double()
            touched[o:o + m] = True
    err = (c.double() - ref).abs().max().item()
    ok_vals = torch.allclose(c.double(), ref, atol=ATOL, rtol=RTOL)
    untouched = torch.equal(c[~touched], c0[~touched])

    # BLOCK_M per group from the launch log; every row of every group covered exactly once.
    per_group, cover = {}, {p: [] for p in range(E)}
    for launch in log:
        for p, r0 in launch["items"]:
            per_group.setdefault(p, set()).add(launch["BLOCK_M"])
            cover[p].extend(range(r0, min(r0 + launch["BLOCK_M"], rows[p])))
    block_m = [sorted(per_group.get(p, [])) for p in range(E)]
    want = [[expected_block_m(m)] if m else [] for m in rows]
    covered = all(sorted(cover[p]) == list(range(rows[p])) for p in range(E))
    return {
        "rows": rows, "layout": layout, "seed": seed, "offsets": offs,
        "block_m_per_group": block_m, "block_m_expected": want,
        "launches": [{k: v for k, v in l.items() if k != "items"} | {"n_items": len(l["items"]),
                      "items": l["items"]} for l in log],
        "max_abs_err": err, "allclose": ok_vals, "outside_rows_untouched": untouched,
        "rows_covered_once": covered, "block_m_as_expected": block_m == want,
        "pass": bool(ok_vals and untouched and covered and block_m == want),
    }


COMBOS = {
    3: [(8, 40, 100), (0, 8, 40), (16, 17, 64), (64, 65, 128), (15, 16, 17), (63, 64, 65),
        (1, 16, 17), (17, 64, 65), (0, 16, 17), (0, 64, 65), (17, 16, 65), (65, 64, 16),
        (128, 0, 1), (0, 0, 0)],
    4: [(8, 40, 100, 0), (0, 8, 40, 100), (16, 17, 64, 65), (15, 16, 17, 18), (63, 64, 65, 66),
        (1, 17, 65, 128), (17, 16, 65, 64), (0, 16, 17, 0), (0, 64, 65, 0), (128, 128, 1, 0)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("outdir")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    module_path = os.path.join(a.outdir, "kernels.py")
    rc = gen.main([a.record, "-o", module_path])
    if rc:
        return rc
    mod = load_module(module_path)
    log = []
    for b in mod.BUCKETS:
        mod.KERNELS[b[0]] = LaunchLog(b[0], mod.KERNELS[b[0]], log)
    global BUCKETS
    BUCKETS = mod.BUCKETS
    hi = mod.M_SCOPE[1]
    combos = COMBOS.get(mod.E)
    if combos is not None:  # Clip to the record's scope (a smaller CAP), dropping repeats.
        combos = list(dict.fromkeys(tuple(min(m, hi) for m in c) for c in combos))
    if combos is None:
        print(f"no combinations for E={mod.E}")
        return 2
    results = []
    for rows in combos:
        for layout in ("padded", "gather"):
            for seed in range(a.seeds):
                r = run_case(mod, list(rows), layout, seed, log)
                results.append(r)
                print(f"{'PASS' if r['pass'] else 'FAIL'} rows={rows} layout={layout} seed={seed} "
                      f"BLOCK_M/group={r['block_m_per_group']} max|err|={r['max_abs_err']:.2e} "
                      f"untouched={r['outside_rows_untouched']} covered={r['rows_covered_once']}")
    summary = {"record": os.path.abspath(a.record), "module": module_path, "E": mod.E, "N": mod.N,
               "K": mod.K, "scope": mod.M_SCOPE, "buckets": mod.BUCKETS, "atol": ATOL, "rtol": RTOL,
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
               "n_cases": len(results), "n_pass": sum(r["pass"] for r in results), "cases": results}
    import triton
    summary["triton"] = triton.__version__
    with open(os.path.join(a.outdir, "results.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n{summary['n_pass']}/{summary['n_cases']} pass; raw: {os.path.join(a.outdir, 'results.json')}")
    return 0 if summary["n_pass"] == summary["n_cases"] else 1


if __name__ == "__main__":
    sys.exit(main())
