#!/usr/bin/env python3
"""Correctness of the generated Triton grouped GEMM against PyTorch.

  python3 test_grouped_gemm_triton.py [RECORD] [--keep DIR]

RECORD defaults to moe_grouped.record.json next to this file (generalized-gemm-nest-report
on moe_grouped.mlir). The test generates the module, imports it, and checks:
  1. the record's own layout (padded [E, CAP, .], identity ids, one M for every group,
     as in moe_grouped.mlir) against a float64 evaluation of the source loop nest;
  2. the same layout with a different M_p per group, one in each bucket;
  3. an MoE layer: top-k routing, sorted_token_ids gather, compact output, empty expert;
  4. the dispatcher's refusals (count outside the scope, overlapping rows, an id outside X,
     C overlapping X / W / sorted_token_ids) and that padding ids outside every group pass;
  5. the generator's refusals on altered records (no group axis, no reordering, ...);
  6. --fixed-block-m 16/64/128: one kernel each, correct on the per-group cases.
"""

import copy
import importlib.util
import json
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import grouped_gemm_triton as gen  # noqa: E402

ATOL = RTOL = 1e-4
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}: {name}{(' - ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def load_module(path):
    spec = importlib.util.spec_from_file_location("generated_grouped_gemm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def source_semantics(a, b, c, rows):
    """The loop nest of moe_grouped.mlir with the rows of group p bounded by rows[p], in
    float64: C[p,i,j] += sum_k A[p,i,k] * B[p,k,j]."""
    out = c.double().clone()
    for p in range(a.shape[0]):
        m = rows[p]
        out[p, :m] += a[p, :m].double() @ b[p].double()
    return out


def padded_case(mod, rows, label):
    E, K, N = mod.E, mod.K, mod.N
    cap = mod.M_SCOPE[1]
    dev = "cuda"
    a = torch.randn(E, cap, K, device=dev)
    b = torch.randn(E, K, N, device=dev)
    c = torch.randn(E, cap, N, device=dev)  # The update starts from the cell's value.
    ref = source_semantics(a, b, c, rows)
    # The record's layout through the gather interface: rows of A and C are p*CAP + i.
    x = a.reshape(E * cap, K)
    cc = c.reshape(E * cap, N).clone()
    ids = torch.arange(E * cap, dtype=torch.int32, device=dev)
    offs = torch.arange(E, dtype=torch.int32, device=dev) * cap
    grows = torch.tensor(rows, dtype=torch.int32, device=dev)
    mod.grouped_gemm(x, b, cc, ids, offs, grows)
    torch.cuda.synchronize()
    got = cc.reshape(E, cap, N).double()
    err = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, atol=ATOL, rtol=RTOL)
    # Rows past M_p must be untouched (bit for bit).
    untouched = all(torch.equal(cc.reshape(E, cap, N)[p, rows[p]:], c[p, rows[p]:]) for p in range(E))
    check(f"{label}: rows {rows}", ok and untouched, f"max |err| {err:.2e}, rows past M_p untouched: {untouched}")


def moe_case(mod, tokens, topk, counts_wanted, label):
    """Top-k routing, made to give each expert the wanted number of (token, expert) pairs."""
    E, K, N = mod.E, mod.K, mod.N
    dev = "cuda"
    assert sum(counts_wanted) == tokens * topk
    # A routing with those counts: deal (expert) slots to tokens so a token never gets the
    # same expert twice.
    slots = [p for p, n in enumerate(counts_wanted) for _ in range(n)]
    experts = torch.tensor(slots).reshape(topk, tokens).t().contiguous()  # [T, topk]
    assert all(len(set(r.tolist())) == topk for r in experts), "routing repeats an expert"
    x = torch.randn(tokens, K, device=dev)
    w = torch.randn(E, K, N, device=dev)
    flat = experts.reshape(-1).to(dev)
    order = torch.argsort(flat, stable=True)
    sorted_token_ids = (order // topk).to(torch.int32)
    counts = torch.bincount(flat, minlength=E).to(torch.int32)
    offs = (torch.cumsum(counts, 0) - counts).to(torch.int32)
    out0 = torch.randn(tokens * topk, N, device=dev)
    out = out0.clone()
    mod.grouped_gemm(x, w, out, sorted_token_ids, offs, counts)
    torch.cuda.synchronize()
    # PyTorch reference: per expert, gather then matmul, in float64.
    ref = out0.double().clone()
    for p in range(E):
        o, m = int(offs[p]), int(counts[p])
        ref[o:o + m] += x[sorted_token_ids[o:o + m].long()].double() @ w[p].double()
    err = (out.double() - ref).abs().max().item()
    # And the MoE result itself: combine back to tokens, vs. a dense per-token loop.
    combined = torch.zeros(tokens, N, dtype=torch.float64, device=dev)
    combined.index_add_(0, sorted_token_ids.long(), out.double() - out0.double())
    dense = torch.zeros(tokens, N, dtype=torch.float64, device=dev)
    for t in range(tokens):
        for e in experts[t].tolist():
            dense[t] += x[t].double() @ w[e].double()
    err2 = (combined - dense).abs().max().item()
    ok = torch.allclose(out.double(), ref, atol=ATOL, rtol=RTOL) and err2 < 1e-3
    check(f"{label}: counts {counts.tolist()}", ok, f"max |err| {err:.2e}; combined vs per-token {err2:.2e}")
    return counts.tolist()


def expect_raise(name, fn, words):
    try:
        fn()
    except ValueError as e:
        check(name, words in str(e), str(e))
    else:
        check(name, False, "no error")


def expect_refusal(name, records, words):
    try:
        gen.read_record(records)
    except gen.Refusal as e:
        check(name, words in str(e), str(e))
    else:
        check(name, False, "accepted")


def main():
    args = sys.argv[1:]
    keep = None
    if "--keep" in args:
        i = args.index("--keep")
        keep = args[i + 1]
        del args[i:i + 2]
    record_path = args[0] if args else os.path.join(HERE, "moe_grouped.record.json")
    with open(record_path) as f:
        records = json.load(f)
    if not torch.cuda.is_available():
        print("UNSUPPORTED: no CUDA device")
        return 77
    torch.manual_seed(0)

    outdir = keep or tempfile.mkdtemp(prefix="grouped-gemm-triton-")
    os.makedirs(outdir, exist_ok=True)
    module_path = os.path.join(outdir, "moe_grouped_kernels.py")
    rc = gen.main([record_path, "-o", module_path])
    check("generator accepts the record", rc == 0)
    if rc:
        return 1
    mod = load_module(module_path)
    src = open(module_path).read()
    nb = len(mod.BUCKETS)
    lo, hi = mod.M_SCOPE
    want_buckets = 1 + sum(1 for t in (16, 64) if max(lo, 1) <= t < hi)
    check("one @triton.jit kernel per bucket", src.count("@triton.jit") == nb == want_buckets,
          f"{nb} buckets: {[(b[1], b[2], b[3]) for b in mod.BUCKETS]}")
    E = mod.E
    edges = sorted({0, 1, hi} | {b[1] for b in mod.BUCKETS} | {b[2] for b in mod.BUCKETS})

    # 1. The source's own semantics: one M for every group (the source's %m).
    for m in edges:
        padded_case(mod, [m] * E, f"record layout, uniform M = {m}")
    # 2. A different M_p per group: the edges of the buckets, and an empty group.
    per_group = [b[2] for b in mod.BUCKETS] + [b[1] for b in mod.BUCKETS]
    padded_case(mod, (per_group * E)[:E], "record layout, per-group M_p at the bucket edges")
    mixed = ([max(1, (b[1] + b[2]) // 2) for b in mod.BUCKETS] + [0] * E)[:E - 1] + [0]
    padded_case(mod, mixed, "record layout, per-group M_p with an empty group")
    lists = mod.build_worklist(mixed)
    items = [len(lists[b[0]][0]) for b in mod.BUCKETS]
    expect = [sum(-(-m // b[3]) for m in mixed if b[1] <= m <= b[2]) for b in mod.BUCKETS]
    check("worklist: each group in its bucket only, ceil(M_p/BLOCK_M) items each",
          items == expect, f"items per bucket {items}, groups {mixed}")

    # 3. MoE with the gather (top-2): expert loads spread over the buckets, then one with
    #    an empty expert. Counts are clipped to the scope and made even; T = sum / 2.
    def moe_counts(wanted):
        c = [min(x, hi) for x in wanted]
        if sum(c) % 2:
            c[c.index(max(c))] -= 1
        return c
    for wanted, label in (([10, 40, 78, 128], "MoE top-2 gather"),
                          ([5, 64, 59, 0], "MoE top-2 gather, empty expert")):
        c = moe_counts((wanted * E)[:E])
        t = sum(c) // 2
        if max(c) > t:
            check(label, False, f"counts {c} cannot be routed top-2 over {t} tokens")
            continue
        moe_case(mod, t, 2, c, label)

    # 4. Dispatcher refusals.
    dev = "cuda"
    x = torch.randn(300, mod.K, device=dev)
    w = torch.randn(mod.E, mod.K, mod.N, device=dev)
    c = torch.zeros(300, mod.N, device=dev)
    ids = torch.arange(300, dtype=torch.int32, device=dev)
    t32 = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)
    expect_raise("dispatcher refuses M_p above the scope",
                 lambda: mod.grouped_gemm(x, w, c, ids, t32([0, 140, 150, 160]), t32([hi + 1, 1, 1, 1])),
                 "outside the record's scope")
    expect_raise("dispatcher refuses overlapping rows of C",
                 lambda: mod.grouped_gemm(x, w, c, ids, t32([0, 5, 100, 200]), t32([10, 10, 10, 10])),
                 "same rows")

    # 4b. Run-time guards for what the record assumes or does not model: ids inside X,
    #     C disjoint from every buffer the kernels read. Padding ids outside every group
    #     are not read and must be accepted.
    xs = torch.randn(16, mod.K, device=dev)
    c16 = torch.zeros(16, mod.N, device=dev)
    offs4, rows4 = t32([0, 4, 8, 12]), t32([4, 4, 4, 4])
    bad = torch.arange(16, dtype=torch.int32, device=dev)
    bad[3] = 10 ** 6
    expect_raise("dispatcher refuses an id past the rows of X",
                 lambda: mod.grouped_gemm(xs, w, c16, bad, offs4, rows4), "sorted_token_ids reads row")
    neg = torch.arange(16, dtype=torch.int32, device=dev)
    neg[0] = -5
    expect_raise("dispatcher refuses a negative id",
                 lambda: mod.grouped_gemm(xs, w, c16, neg, offs4, rows4), "sorted_token_ids reads row")
    buf = torch.randn(64, max(mod.K, mod.N), device=dev)
    expect_raise("dispatcher refuses X overlapping C",
                 lambda: mod.grouped_gemm(buf[:, :mod.K], w, buf[:, :mod.N],
                                          torch.arange(64, dtype=torch.int32, device=dev),
                                          t32([0, 8, 16, 24]), t32([8, 8, 8, 8])), "C overlaps X")
    cw = torch.randn(mod.E * mod.K * mod.N, device=dev)
    expect_raise("dispatcher refuses W overlapping C",
                 lambda: mod.grouped_gemm(xs, cw.view(mod.E, mod.K, mod.N), cw.view(-1, mod.N),
                                          torch.arange(16, dtype=torch.int32, device=dev), offs4,
                                          t32([1, 1, 1, 1])), "C overlaps W")
    raw = torch.zeros(16 * mod.N, dtype=torch.float32, device=dev)
    expect_raise("dispatcher refuses sorted_token_ids overlapping C",
                 lambda: mod.grouped_gemm(xs, w, raw.view(16, mod.N), raw[:16].view(torch.int32),
                                          offs4, rows4), "C overlaps sorted_token_ids")
    pad = torch.arange(20, dtype=torch.int32, device=dev) % 16
    pad[[4, 5, 6, 7]] = 10 ** 6  # padding between groups: not read
    cpad0 = torch.randn(20, mod.N, device=dev)
    cpad = cpad0.clone()
    offs_p, rows_p = [0, 8, 12, 16], [4, 4, 4, 4]
    try:
        mod.grouped_gemm(xs, w, cpad, pad, t32(offs_p), t32(rows_p))
        torch.cuda.synchronize()
        ref = cpad0.double().clone()
        for p, (o, m) in enumerate(zip(offs_p, rows_p)):
            ref[o:o + m] += xs[pad[o:o + m].long()].double() @ w[p].double()
        check("dispatcher accepts out-of-range padding ids outside every group",
              torch.allclose(cpad.double(), ref, atol=ATOL, rtol=RTOL))
    except ValueError as e:
        check("dispatcher accepts out-of-range padding ids outside every group", False, str(e))

    # 6. --fixed-block-m: one native kernel over the whole scope.
    for bm in (16, 64, 128):
        path = os.path.join(outdir, f"moe_grouped_fixed{bm}.py")
        rc = gen.main([record_path, "-o", path, "--fixed-block-m", str(bm)])
        fm = load_module(path) if rc == 0 else None
        ok = (fm is not None and len(fm.BUCKETS) == 1 and fm.BUCKETS[0][3] == bm
              and open(path).read().count("@triton.jit") == 1
              and (fm.BUCKETS[0][1], fm.BUCKETS[0][2]) == (max(lo, 1), hi))
        check(f"--fixed-block-m {bm}: one kernel over [{max(lo, 1)}, {hi}]", ok,
              f"{fm.BUCKETS if fm else rc}")
        if fm is not None:
            padded_case(fm, (per_group * E)[:E], f"--fixed-block-m {bm}, per-group M_p at the bucket edges")
            padded_case(fm, mixed, f"--fixed-block-m {bm}, per-group M_p with an empty group")
    expect_refusal_cli = lambda argv: gen.main(argv)
    try:
        expect_refusal_cli([record_path, "-o", os.devnull, "--fixed-block-m", "24"])
        check("--fixed-block-m 24 refused", False, "accepted")
    except SystemExit as e:
        check("--fixed-block-m 24 refused", e.code == 2, f"exit {e.code}")

    # 5. Generator refusals.
    def altered(fn):
        r = copy.deepcopy(records)
        fn(r[0]["nests"][0])
        return r
    expect_refusal("no group axis", altered(lambda n: n["gemm"].update(group_axis=None)), "no group axis")
    expect_refusal("fold may not be reordered", altered(lambda n: (
        n.update(reassociation_permitted=False), n["reassociation"].update(permitted=False),
        n["gemm"]["reorder"]["float"].update(status="REFUTED"),
        n["gemm"]["reorder"]["integer"].update(status="UNKNOWN"))), "may not be reordered")
    expect_refusal("not verified", altered(lambda n: n.update(status="UNSUPPORTED")), "only a verified")
    expect_refusal("row count not uniform over groups", altered(
        lambda n: n["gemm"]["parametric_extents"][0].update(uniform_over_groups=False)), "uniform")
    expect_refusal("row count unscoped", altered(
        lambda n: n["gemm"]["parametric_extents"][0]["upper_form"]["parameters"][0].update(scope=None)),
        "scoped argument")
    expect_refusal("P read by a second dimension of A", altered(
        lambda n: n["gemm"]["views"]["A"]["dimensions"][1]["levels"].append({"level": 0, "coefficient": 1})),
        "view of A")

    print(f"\n{len(failures)} failure(s)" + (f": {failures}" if failures else ""))
    print(f"generated module: {module_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
