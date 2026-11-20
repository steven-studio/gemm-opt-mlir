#!/usr/bin/env python3
"""Generate Triton grouped-GEMM kernels from a Generalized GEMM nest record.

Input: the JSON of `generalized-gemm-nest-report` (schema generalized-gemm-nest/2) with the
additive fields `group_axis` and `parametric_extents`. Output: one Python module holding
several `@triton.jit` kernels (one per bucket of the per-group row count M_p) and a host
dispatcher that builds an O(E) worklist at run time and launches each bucket's kernel once.

What the record states, and what this tool adds on top of it
-----------------------------------------------------------
The record states, for every value of the row-count argument inside its scope, that the
nest computes  C[p,i,j] := C[p,i,j] + sum_k A[p,i,k] * B[p,k,j]  for p over the group axis,
i < M_p, j < N, k < K, in the order of its levels, and whether the fold may be reordered
(`reassociation_permitted` / `reorder`). It is VERIFIED_NOT_MAPPED: the backend in
systolic-mlir does not map a batch axis. This tool is a second backend that does, and it
reads nothing the checkers did not take over:

* Groups. The views put P in dimension 0 of A, B and C with coefficient 1 and nowhere else,
  so group p touches only C[p, ., .]: different groups write different cells and the groups
  may run in any order or at once (the roots are separate by the record's authorization).
* Rows. M_p is the scalar argument named in `parametric_extents`; the record covers every
  value in its scope, so each group may be launched with its own M_p as long as that value
  is in the scope. The dispatcher refuses a count outside it.
* Gather. The kernels address A[p,i,k] as X[sorted_token_ids[group_offsets[p] + i], k] and
  C[p,i,j] as C[group_offsets[p] + i, j]. That is a renaming of the cells of the record's
  memrefs, supplied by the caller: with group_offsets[p] = p*CAP and the identity ids it is
  exactly the record's layout. A is only read, so a token in several groups is harmless;
  C rows must not overlap between groups, which the dispatcher checks.
* Order. tl.dot reorders the K fold. The generator refuses a record whose fold may not be
  reordered.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

SCHEMA = "generalized-gemm-nest/2"


class Refusal(Exception):
    """The record does not support a grouped Triton kernel; the message says why."""


@dataclass
class Bucket:
    name: str
    lo: int  # Inclusive lower bound of M_p.
    hi: int  # Inclusive upper bound of M_p.
    block_m: int
    block_n: int
    block_k: int
    num_warps: int


@dataclass
class GroupedGemm:
    function: str
    nest: int
    location: str
    status: str
    groups: int  # E: extent of the group axis.
    n: int
    k: int
    m_symbol: str
    m_argument: int
    m_scope: Tuple[int, int]
    element: str
    statement: str
    reorder: str  # Why the fold may be reordered.
    buckets: List[Bucket] = field(default_factory=list)


# ---------------------------------------------------------------------------------------
# Reading the record
# ---------------------------------------------------------------------------------------

def _const(level: dict, key: str) -> Optional[int]:
    v = level.get(key)
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _check_view(gemm: dict, name: str, roles: Tuple[str, str], group_level: int,
                levels_of: dict) -> None:
    """View `name` must be [P, roles[0], roles[1]], each dimension one level, coefficient 1."""
    view = gemm["views"][name]
    dims = view.get("dimensions", [])
    if not view.get("affine") or len(dims) != 3:
        raise Refusal(f"the view of {name} is not an affine 3-dimensional view")
    want = [group_level, levels_of[roles[0]], levels_of[roles[1]]]
    for d, (dim, level) in enumerate(zip(dims, want)):
        reads = dim.get("levels", [])
        if (not dim.get("affine") or dim.get("parameters") or dim.get("constant", 0) != 0
                or len(reads) != 1 or reads[0]["level"] != level or reads[0]["coefficient"] != 1):
            raise Refusal(f"dimension {d} of the view of {name} is {dim.get('form')!r}, "
                          f"not level{level} alone")


def read_record(records: list, function: Optional[str] = None,
                nest: Optional[int] = None) -> GroupedGemm:
    found = []
    for entry in records:
        if function is not None and entry.get("function") != function:
            continue
        if entry.get("schema") != SCHEMA:
            raise Refusal(f"schema {entry.get('schema')!r}, expected {SCHEMA!r}")
        for n in entry.get("nests", []):
            if nest is None or n.get("nest") == nest:
                found.append(n)
    if len(found) != 1:
        raise Refusal(f"{len(found)} nests match; name one with --function/--nest")
    n = found[0]

    status = n.get("status")
    if status not in ("VERIFIED", "VERIFIED_NOT_MAPPED"):
        raise Refusal(f"nest status {status}: only a verified record is lowered")
    gemm = n.get("gemm")
    if not gemm:
        raise Refusal("the record has no GEMM")
    symbolic = gemm.get("tiers", {}).get("computation", {}).get("symbolic", {})
    if symbolic.get("status") != "VERIFIED":
        raise Refusal("the computation is not verified over the scope (symbolic tier)")
    acc = gemm["accumulation"]
    if (acc.get("reduction") not in ("add", "fadd") or acc.get("multiply") not in ("mul", "fmul")
            or not acc.get("accumulator_first") or acc.get("statement") != "C[h] := C[h] + A[f] * B[g]"):
        raise Refusal(f"the update is {acc.get('statement')!r} with "
                      f"{acc.get('reduction')}/{acc.get('multiply')}, not C += A*B")
    if gemm.get("guard") != "true" or gemm.get("frame", {}).get("other_writes", 1) != 0:
        raise Refusal("the nest has a guard or writes other cells")

    # The fold order: tl.dot sums K in its own order.
    # A float fold reordered changes bits (the record's float reorder fact is REFUTED for
    # it); it is allowed only by the programmer's permission, `reassoc` on every site.
    reorder = gemm.get("reorder", {})
    reassociation = n.get("reassociation", {})
    if n.get("reassociation_permitted") and reassociation.get("permitted"):
        why = "reassociation permitted by the source (" + reassociation.get("basis", "?") + ")"
    elif reorder.get("integer", {}).get("status") == "PROVEN":
        why = "integer reorder proven: " + reorder["integer"].get("detail", "")
    elif reorder.get("float", {}).get("status") == "PROVEN":
        why = "float reorder proven: " + reorder["float"].get("detail", "")
    else:
        raise Refusal("the fold may not be reordered, and tl.dot reorders it")

    group = gemm.get("group_axis")
    if not group:
        raise Refusal("no group axis (axis P) in the record")
    if _const(group, "extent") is None or group.get("lower") != 0:
        raise Refusal("the group axis has no constant extent starting at 0")

    axes = gemm["axes"]
    if any(len(axes.get(a, [])) != 1 for a in "PIJK") or len(gemm["levels"]) != 4:
        raise Refusal(f"axes {axes}: need exactly one level each for P, I, J, K")
    levels_of = {a: axes[a][0] for a in "PIJK"}
    if levels_of["P"] != group["level"]:
        raise Refusal("group_axis disagrees with axes")
    levels = gemm["levels"]
    for a in "JK":
        lv = levels[levels_of[a]]
        if lv.get("lower") != 0 or _const(lv, "extent") is None or lv.get("descending"):
            raise Refusal(f"the level of axis {a} has no constant extent from 0")
    if levels[levels_of["I"]].get("descending"):
        raise Refusal("the level of axis I descends")

    _check_view(gemm, "A", ("I", "K"), levels_of["P"], levels_of)
    _check_view(gemm, "B", ("K", "J"), levels_of["P"], levels_of)
    _check_view(gemm, "C", ("I", "J"), levels_of["P"], levels_of)

    extents = gemm.get("parametric_extents", [])
    if len(extents) != 1 or extents[0].get("axis") != "I":
        raise Refusal(f"need one symbolic extent, on axis I; the record has "
                      f"{[e.get('axis') for e in extents]}")
    ext = extents[0]
    if not ext.get("affine") or not ext.get("uniform_over_groups"):
        raise Refusal("the row count is not an affine symbol that is uniform over the groups")
    lower, upper = ext["lower_form"], ext["upper_form"]
    if lower["constant"] != 0 or lower["parameters"]:
        raise Refusal("the rows of a group do not start at 0")
    params = upper["parameters"]
    if (upper["constant"] != 0 or len(params) != 1 or params[0]["coefficient"] != 1
            or params[0]["source"] != "argument" or not params[0].get("scope")):
        raise Refusal("the row count is not one scoped argument of the function")
    scope = tuple(params[0]["scope"])

    return GroupedGemm(
        function=n["function"], nest=n["nest"], location=n.get("location", ""), status=status,
        groups=group["extent"], n=levels[levels_of["J"]]["extent"],
        k=levels[levels_of["K"]]["extent"], m_symbol=ext["symbol"],
        m_argument=params[0]["argument"], m_scope=(scope[0], scope[1]),
        element=acc.get("element", "?"), statement=acc["statement"], reorder=why)


# ---------------------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------------------

def _pow2_at_least(x: int) -> int:
    p = 16
    while p < x:
        p *= 2
    return p


def make_buckets(g: GroupedGemm, thresholds: List[int]) -> List[Bucket]:
    """Split the scope of M_p at the thresholds: [lo, t0], (t0, t1], ..., (tn, hi]."""
    lo, hi = g.m_scope
    lo = max(lo, 1)  # A group with no rows launches nothing.
    cuts = sorted(t for t in set(thresholds) if lo <= t < hi)
    bounds = []
    start = lo
    for t in cuts:
        bounds.append((start, t))
        start = t + 1
    bounds.append((start, hi))
    block_n = min(64, _pow2_at_least(g.n))
    block_k = min(32, _pow2_at_least(g.k))
    buckets = []
    for b_lo, b_hi in bounds:
        block_m = min(128, _pow2_at_least(b_hi))
        warps = 4 if block_m <= 64 else 8
        buckets.append(Bucket(f"m{b_lo}_{b_hi}", b_lo, b_hi, block_m, block_n, block_k, warps))
    return buckets


# ---------------------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------------------

KERNEL = '''
@triton.jit
def {name}(
    x_ptr, w_ptr, c_ptr, sorted_token_ids_ptr, group_offsets_ptr,
    work_group_ptr, work_row_ptr, group_rows_ptr,
    stride_xt, stride_xk, stride_we, stride_wk, stride_wn, stride_cr, stride_cn,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """M_p in [{lo}, {hi}]. C[p,i,j] += sum_k A[p,i,k] B[p,k,j], with
    A[p,i,k] = X[sorted_token_ids[group_offsets[p] + i], k] and C[p,i,j] = C[group_offsets[p] + i, j]."""
    item = tl.program_id(0)
    pid_n = tl.program_id(1)
    p = tl.load(work_group_ptr + item)
    row0 = tl.load(work_row_ptr + item)
    rows = tl.load(group_rows_ptr + p)
    base = tl.load(group_offsets_ptr + p)

    offs_m = row0 + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < rows
    mask_n = offs_n < N

    # The gather index, injected by the caller.
    tokens = tl.load(sorted_token_ids_ptr + base + offs_m, mask=mask_m, other=0)
    x_rows = x_ptr + tokens[:, None].to(tl.int64) * stride_xt
    w_grp = w_ptr + p.to(tl.int64) * stride_we

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        a = tl.load(x_rows + kk[None, :] * stride_xk,
                    mask=mask_m[:, None] & (kk[None, :] < K), other=0.0)
        b = tl.load(w_grp + kk[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                    mask=(kk[:, None] < K) & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")

    c_ptrs = c_ptr + (base + offs_m)[:, None].to(tl.int64) * stride_cr + offs_n[None, :] * stride_cn
    mask = mask_m[:, None] & mask_n[None, :]
    c = tl.load(c_ptrs, mask=mask, other=0.0)   # The update reads the cell: C starts from it.
    tl.store(c_ptrs, c + acc, mask=mask)
'''

HEADER = '''# Generated by tools/grouped-gemm-triton/grouped_gemm_triton.py. Do not edit.
#
# Record: function {function}, nest {nest} ({location}), status {status}
# Update: {statement}, element {element}; fold order: {reorder}
# Groups E = {groups} (axis P); N = {n}, K = {k}; rows {m_symbol} = argument {m_argument},
# scope [{m_lo}, {m_hi}]. Buckets: {bucket_list}.
import torch
import triton
import triton.language as tl

E = {groups}
N = {n}
K = {k}
M_SCOPE = ({m_lo}, {m_hi})
BUCKETS = {buckets_py}
'''

DISPATCH = '''

KERNELS = {{{kernel_map}}}


def build_worklist(group_rows):
    """O(E) on the host: one pass over the groups. Each group goes to the bucket of its
    row count and adds ceil(M_p / BLOCK_M) items (p, first row); that count is bounded by
    the scope, so the pass is O(E) for a fixed record."""
    rows = [int(r) for r in group_rows]
    if len(rows) != E:
        raise ValueError(f"{{len(rows)}} row counts for {{E}} groups")
    lists = {{b[0]: ([], []) for b in BUCKETS}}
    for p, m in enumerate(rows):
        if not (M_SCOPE[0] <= m <= M_SCOPE[1]):
            raise ValueError(f"group {{p}} has {{m}} rows, outside the record's scope {{M_SCOPE}}")
        if m == 0:
            continue
        for name, lo, hi, bm, _, _, _ in BUCKETS:
            if lo <= m <= hi:
                for r0 in range(0, m, bm):
                    lists[name][0].append(p)
                    lists[name][1].append(r0)
                break
    return lists


def grouped_gemm(x, w, c, sorted_token_ids, group_offsets, group_rows):
    """C[group_offsets[p] + i, :] += X[sorted_token_ids[group_offsets[p] + i], :] @ W[p]
    for p < E, i < group_rows[p]. x: [T, K]; w: [E, K, N]; c: [R, N] (updated in place);
    sorted_token_ids: int32 [R]; group_offsets, group_rows: int32 [E]."""
    assert w.shape == (E, K, N) and x.shape[1] == K and c.shape[1] == N
    assert x.dtype == w.dtype == c.dtype == torch.float32
    offs = group_offsets.tolist()
    rows = group_rows.tolist()
    # The groups write disjoint rows of C: the record's groups write disjoint cells.
    spans = sorted((o, o + m) for o, m in zip(offs, rows) if m > 0)
    for (a0, a1), (b0, _) in zip(spans, spans[1:]):
        if b0 < a1:
            raise ValueError("two groups write the same rows of C")
    if spans and (spans[0][0] < 0 or spans[-1][1] > c.shape[0] or spans[-1][1] > sorted_token_ids.numel()):
        raise ValueError("a group's rows fall outside C or sorted_token_ids")
    lists = build_worklist(rows)
    dev = c.device
    for name, lo, hi, bm, bn, bk, warps in BUCKETS:
        groups_of, rows_of = lists[name]
        if not groups_of:
            continue
        wg = torch.tensor(groups_of, dtype=torch.int32, device=dev)
        wr = torch.tensor(rows_of, dtype=torch.int32, device=dev)
        grid = (len(groups_of), triton.cdiv(N, bn))
        KERNELS[name][grid](
            x, w, c, sorted_token_ids, group_offsets, wg, wr, group_rows,
            x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.stride(2),
            c.stride(0), c.stride(1),
            N=N, K=K, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps)
    return c
'''


def generate(g: GroupedGemm) -> str:
    names = [f"grouped_gemm_{b.name}" for b in g.buckets]
    out = [HEADER.format(
        function=g.function, nest=g.nest, location=g.location, status=g.status,
        statement=g.statement, element=g.element, reorder=g.reorder, groups=g.groups,
        n=g.n, k=g.k, m_symbol=g.m_symbol, m_argument=g.m_argument,
        m_lo=g.m_scope[0], m_hi=g.m_scope[1],
        bucket_list=", ".join(f"[{b.lo},{b.hi}] BLOCK_M={b.block_m}" for b in g.buckets),
        buckets_py=repr([(b.name, b.lo, b.hi, b.block_m, b.block_n, b.block_k, b.num_warps)
                         for b in g.buckets]))]
    for name, b in zip(names, g.buckets):
        out.append(KERNEL.format(name=name, lo=b.lo, hi=b.hi))
    out.append(DISPATCH.format(kernel_map=", ".join(
        f"{b.name!r}: {name}" for name, b in zip(names, g.buckets))))
    return "".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("record", help="JSON from generalized-gemm-nest-report")
    ap.add_argument("-o", "--output", required=True, help="generated Python module")
    ap.add_argument("--function")
    ap.add_argument("--nest", type=int)
    ap.add_argument("--thresholds", default="16,64",
                    help="bucket cuts of M_p (default 16,64: <=16, 17..64, >64)")
    args = ap.parse_args(argv)
    with open(args.record) as f:
        records = json.load(f)
    try:
        g = read_record(records, args.function, args.nest)
    except Refusal as why:
        print(f"grouped-gemm-triton: refused: {why}", file=sys.stderr)
        return 2
    g.buckets = make_buckets(g, [int(t) for t in args.thresholds.split(",") if t])
    with open(args.output, "w") as f:
        f.write(generate(g))
    print(f"{args.output}: {len(g.buckets)} kernels for {g.function} nest {g.nest}: " +
          ", ".join(f"{g.m_symbol} in [{b.lo},{b.hi}] BLOCK_M={b.block_m}" for b in g.buckets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
