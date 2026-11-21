#!/usr/bin/env python3
"""Lowering of a nest whose output row goes through an index table: scatter-add
(C[ids[r]][j] += A[r][k] * B[p][k][j]) and block-sparse rows (C[blk[b]*B + i][j] += ...),
from the record `generalized-gemm-nest-report --auto` writes and the scf source it read.

  python3 scatter_gemm_triton.py SRC.mlir RECORD.json -o MODULE.py [--function f]

The record says what may be done (its `auto.plan`) and under which premises (its
`auto.premises`); the source gives the nest's own index computation, which the module
re-emits (as systolic-mlir's Track B does): the rows of A, B and C each logical output reads,
as torch expressions over the row counters and the tables. Nothing here is a claim: the
module checks, at every call, before it relies on the record:

  bounds     every table read, and every row of A, B and C the nest addresses, is inside
             its memref. Otherwise the nest as written is undefined: ValueError.
  separate   the record's root separations (byte spans of the arguments apart). Otherwise
             ValueError: the record's claims assume them, and no path here models aliasing.
  table      each table cell in the record's range [lower, upper], each scalar in its range.
             Otherwise the record does not speak of the input: the nest runs as written.
  distinct   (plan parallel_if_injective) the table's cells pairwise distinct, O(cells + range)
             by a count per value. Otherwise the plan's on_repeated_index path.

Paths:
  parallel       Triton tiles (BLOCK_M source rows sharing one B, BLOCK_N columns, the K fold
                 by tl.dot) add into C directly: rows are distinct, so no two tiles meet.
  csr_reduction  the same tiles write Y[s, :] for every source row s; then the two-layer
                 inverse map: a count per target row, its prefix sum (bucket offsets), the
                 sources sorted stably by target (each bucket in program order), and one
                 program per target row adding its bucket into C in that order. Each cell of
                 C is written once.
  as_written     the nest in program order, one source row at a time.

The K fold is reordered by tl.dot (and by torch.matmul on the as-written path): allowed only
when the record says the source permits reassociation. The tiles' and the as-written path's
sums over k then may differ in the last bits; the collisions' sums are in program order on
the csr path.
"""

import argparse
import json
import re
import sys


class Refusal(Exception):
    pass


# ----------------------------------------------------------------------------------------
# The scf subset: func.func, arith.constant, scf.for with constant steps, integer arith,
# memref.load / memref.store, arith.mulf / arith.addf. Anything else is refused.
# ----------------------------------------------------------------------------------------

TYPE_RE = re.compile(r"memref<([0-9x]+)x(f32|index|i32|i64)>|memref<(f32|index|i32|i64)>")


def memref_shape(t):
    m = re.fullmatch(r"memref<((?:\d+x)*)(f32|f64|index|i\d+)>", t.strip())
    if not m:
        raise Refusal(f"type {t}: a memref of static shape is wanted")
    dims = [int(x) for x in m.group(1).split("x") if x]
    return dims, m.group(2)


def parse(src, function):
    text = open(src).read()
    m = re.search(r"func\.func\s+@" + re.escape(function) + r"\((.*?)\)\s*\{", text, re.S)
    if not m:
        raise Refusal(f"no function @{function}")
    args = []
    for part in [p.strip() for p in m.group(1).split(",") if p.strip()]:
        name, ty = [x.strip() for x in part.split(":", 1)]
        args.append((name, ty))
    body = text[m.end():]
    depth, end = 1, 0
    for idx, ch in enumerate(body):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            end = idx
            break
    lines = [ln.strip() for ln in body[:end].splitlines() if ln.strip()]
    ops, loops, stack = [], [], []
    for ln in lines:
        if ln == "}":
            if stack:
                stack.pop()
            continue
        if ln == "return":
            continue
        mm = re.fullmatch(r"scf\.for\s+(%\w+)\s*=\s*(%\w+)\s+to\s+(%\w+)\s+step\s+(%\w+)\s*\{", ln)
        if mm:
            loops.append(dict(iv=mm.group(1), lo=mm.group(2), hi=mm.group(3), step=mm.group(4), depth=len(stack)))
            stack.append(mm.group(1))
            continue
        ops.append((ln, list(stack)))
    return args, loops, ops


class Nest:
    """The nest's integer computation, as expressions over the counters and the tables."""

    BIN = {"addi": "+", "subi": "-", "muli": "*", "divui": "//", "divsi": "//", "remui": "%", "remsi": "%"}

    def __init__(self, src, function):
        self.args, self.loops, ops = parse(src, function)
        self.argnames = [a for a, _ in self.args]
        self.types = dict(self.args)
        self.const, self.expr, self.loads, self.floats = {}, {}, {}, {}
        self.store = None
        for ln, _ in ops:
            self._op(ln)
        if self.store is None:
            raise Refusal("no memref.store in the nest")
        self._roles()

    def _op(self, ln):
        m = re.fullmatch(r"(%\w+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*(index|i\d+)", ln)
        if m:
            self.const[m.group(1)] = int(m.group(2))
            return
        m = re.fullmatch(r"(%\w+)\s*=\s*arith\.(\w+)\s+(%\w+),\s*(%\w+)\s*:\s*(index|i\d+)", ln)
        if m and m.group(2) in self.BIN:
            self.expr[m.group(1)] = (m.group(2), m.group(3), m.group(4))
            return
        m = re.fullmatch(r"(%\w+)\s*=\s*arith\.index_cast\s+(%\w+)\s*:.*", ln)
        if m:
            self.expr[m.group(1)] = ("id", m.group(2), None)
            return
        m = re.fullmatch(r"(%\w+)\s*=\s*memref\.load\s+(%\w+)\[(.*?)\]\s*:\s*(.+)", ln)
        if m:
            idx = [x.strip() for x in m.group(3).split(",") if x.strip()]
            self.loads[m.group(1)] = (m.group(2), idx)
            return
        m = re.fullmatch(r"(%\w+)\s*=\s*arith\.(mulf|addf)\s+(%\w+),\s*(%\w+)(\s+fastmath<[\w,]+>)?\s*:\s*f32", ln)
        if m:
            self.floats[m.group(1)] = (m.group(2), m.group(3), m.group(4), bool(m.group(5) and "reassoc" in m.group(5)))
            return
        m = re.fullmatch(r"memref\.store\s+(%\w+),\s*(%\w+)\[(.*?)\]\s*:\s*(.+)", ln)
        if m:
            if self.store is not None:
                raise Refusal("more than one memref.store")
            self.store = (m.group(1), m.group(2), [x.strip() for x in m.group(3).split(",")])
            return
        raise Refusal(f"an operation outside the supported subset: {ln}")

    # The MAC: C[h] = C[h] + A[f] * B[g].
    def _roles(self):
        val, cname, cidx = self.store
        if val not in self.floats or self.floats[val][0] != "addf":
            raise Refusal("the stored value is no arith.addf")
        _, x, y, _ = self.floats[val]
        z, mm = (x, y) if x in self.loads and self.loads[x][0] == cname else (y, x)
        if z not in self.loads or self.loads[z] != (cname, cidx) or mm not in self.floats or self.floats[mm][0] != "mulf":
            raise Refusal("the update is not C[h] = C[h] + A[f] * B[g] at one address")
        _, u, v, _ = self.floats[mm]
        if u not in self.loads or v not in self.loads:
            raise Refusal("the product's operands are not loads")
        (aname, aidx), (bname, bidx) = self.loads[u], self.loads[v]
        self.A, self.B, self.C = (aname, aidx), (bname, bidx), (cname, cidx)
        ivs = [lp["iv"] for lp in self.loops]
        deps = lambda idx: set().union(*[self.ivs_of(e) for e in idx]) if idx else set()
        kk = [iv for iv in ivs if iv in deps(aidx) and iv in deps(bidx) and iv not in deps(cidx)]
        jj = [iv for iv in ivs if iv in deps(bidx) and iv in deps(cidx) and iv not in deps(aidx)]
        if len(kk) != 1 or len(jj) != 1:
            raise Refusal("one K counter (A and B, not C) and one J counter (B and C, not A) are wanted")
        self.k, self.j = kk[0], jj[0]
        if aidx[-1] != self.k or bidx[-1] != self.j or bidx[-2] != self.k or cidx[-1] != self.j:
            raise Refusal("A[..., k], B[..., k, j] and C[..., j] are wanted")
        for e in aidx[:-1] + cidx[:-1] + bidx[:-2]:
            if self.ivs_of(e) & {self.j, self.k}:
                raise Refusal("a row index reads j or k")
        self.rows = [lp for lp in self.loops if lp["iv"] not in (self.j, self.k)]
        order = [lp["iv"] for lp in self.loops]
        if order[-2:] != [self.j, self.k]:
            raise Refusal("the row counters must be outside j, and j outside k")
        for lp in self.loops:
            for key in ("lo", "hi", "step"):
                if lp[key] not in self.const:
                    raise Refusal(f"loop {lp['iv']}: a {key} that is no constant")
            if self.const[lp["step"]] <= 0:
                raise Refusal("a loop with a step that is not positive")
        jl = next(lp for lp in self.loops if lp["iv"] == self.j)
        kl = next(lp for lp in self.loops if lp["iv"] == self.k)
        if any(self.const[lp[k]] != v for lp in (jl, kl) for k, v in (("lo", 0), ("step", 1))):
            raise Refusal("j and k must run from 0 by 1")
        self.N, self.K = self.const[jl["hi"]], self.const[kl["hi"]]
        cdims, cel = memref_shape(self.types[cname])
        adims, ael = memref_shape(self.types[aname])
        bdims, bel = memref_shape(self.types[bname])
        if {cel, ael, bel} != {"f32"}:
            raise Refusal("f32 operands are wanted")
        if cdims[-1] != self.N or adims[-1] != self.K or bdims[-2:] != [self.K, self.N]:
            raise Refusal("the extents of j and k must be the last dimensions of C, A and B")
        self.cdims, self.adims, self.bdims = cdims, adims, bdims
        self.reassoc = self.floats[val][3]

    def ivs_of(self, name, seen=None):
        if name in self.const or name in self.argnames:
            return set()
        if name in self.expr:
            _, a, b = self.expr[name]
            return self.ivs_of(a) | (self.ivs_of(b) if b else set())
        if name in self.loads:
            return set().union(set(), *[self.ivs_of(e) for e in self.loads[name][1]])
        return {name}  # a loop counter

    def tables(self):
        """Arguments read by integer loads (the index tables)."""
        out = []
        for _, (mem, _) in self.loads.items():
            dims, el = memref_shape(self.types[mem])
            if el != "f32" and mem not in out:
                out.append(mem)
        return out

    def py(self, name):
        """Python/torch text of an integer value over the row counters' tensors."""
        if name in self.const:
            return str(self.const[name])
        if name in self.expr:
            op, a, b = self.expr[name]
            if op == "id":
                return self.py(a)
            return f"({self.py(a)} {self.BIN[op]} {self.py(b)})"
        if name in self.loads:
            mem, idx = self.loads[name]
            dims, el = memref_shape(self.types[mem])
            if el == "f32":
                raise Refusal("an index reads a float")
            return f"_read({_arg(mem)}, {dims}, [{', '.join(self.py(e) for e in idx)}], '{mem}')"
        if name in self.argnames:
            return f"int({_arg(name)})"
        return _iv(name)


def _arg(name):
    return "a_" + name.lstrip("%")


def _iv(name):
    return "iv_" + name.lstrip("%")


def linear(nest, idx, dims):
    """Row-major linear index over the leading dimensions `dims` of the index list."""
    if not idx:
        return "0"
    out = nest.py(idx[0])
    for e, d in zip(idx[1:], dims[1:]):
        out = f"(({out}) * {d} + {nest.py(e)})"
    return out


# ----------------------------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------------------------

def read_record(path, function):
    data = json.load(open(path))
    for fn in data:
        if fn.get("function") != function:
            continue
        nests = fn.get("nests", [])
        if len(nests) != 1:
            raise Refusal(f"{len(nests)} nests in @{function}: one is wanted")
        n = nests[0]
        if "auto" not in n:
            raise Refusal("the record has no 'auto' object: run the reporter with --auto")
        if not (n.get("reassociation_permitted") and n.get("reassociation", {}).get("permitted")):
            raise Refusal("the fold may not be reordered, and tl.dot reorders it")
        return n
    raise Refusal(f"no record for @{function}")


MODULE = r'''# Generated by tools/scatter-gemm-triton/scatter_gemm_triton.py. Do not edit.
# Source {src}, function @{function}. Record: status {status}, ordered fold verified {fold}.
# Plan: {plan}
# Premises (checked at every call): {premises}
import torch
import triton
import triton.language as tl

ARGS = {argnames}
PLAN = {plan_py}
PREMISES = {premises_py}
N, K = {N}, {K}
A_DIMS, B_DIMS, C_DIMS = {adims}, {bdims}, {cdims}
ROWS = {rows}            # (counter, lower, upper, step), outermost first: the source rows
LAST_PATH = None         # The path the last call took, for the tests.
BLOCK_M, BLOCK_N, BLOCK_K = 16, {block_n}, {block_k}


@triton.jit
def _tiles(a_ptr, b_ptr, out_ptr, arow_ptr, orow_ptr, bsel_ptr, seg_start_ptr, seg_len_ptr,
           stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_om, stride_on,
           N: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
           BLOCK_K: tl.constexpr, ACCUMULATE: tl.constexpr):
    """Source rows [start, start + len) share one B; out[orow[s], j] (+)= sum_k A[arow[s], k] B[bsel, k, j]."""
    item = tl.program_id(0)
    pid_n = tl.program_id(1)
    start = tl.load(seg_start_ptr + item)
    length = tl.load(seg_len_ptr + item)
    offs = tl.arange(0, BLOCK_M)
    mask_m = offs < length
    arow = tl.load(arow_ptr + start + offs, mask=mask_m, other=0)
    orow = tl.load(orow_ptr + start + offs, mask=mask_m, other=0)
    bsel = tl.load(bsel_ptr + start)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        a = tl.load(a_ptr + arow[:, None].to(tl.int64) * stride_am + kk[None, :] * stride_ak,
                    mask=mask_m[:, None] & (kk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + bsel.to(tl.int64) * stride_be + kk[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(kk[:, None] < K) & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    ptrs = out_ptr + orow[:, None].to(tl.int64) * stride_om + offs_n[None, :] * stride_on
    mask = mask_m[:, None] & mask_n[None, :]
    if ACCUMULATE:
        acc += tl.load(ptrs, mask=mask, other=0.0)
    tl.store(ptrs, acc, mask=mask)


@triton.jit
def _buckets(c_ptr, y_ptr, targets_ptr, offsets_ptr, counts_ptr, sorted_src_ptr,
             stride_cm, stride_cn, stride_ym, stride_yn, N: tl.constexpr, BLOCK_N: tl.constexpr):
    """One target row: C[row] += Y[s] for the sources s of its bucket, in program order."""
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    row = tl.load(targets_ptr + pid)
    first = tl.load(offsets_ptr + row)
    count = tl.load(counts_ptr + row)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    c_ptrs = c_ptr + row.to(tl.int64) * stride_cm + offs_n * stride_cn
    c = tl.load(c_ptrs, mask=mask_n, other=0.0)
    for t in range(0, count):
        s = tl.load(sorted_src_ptr + first + t)
        c += tl.load(y_ptr + s.to(tl.int64) * stride_ym + offs_n * stride_yn, mask=mask_n, other=0.0)
    tl.store(c_ptrs, c, mask=mask_n)


def _read(table, dims, idx, name):
    """A table read of the nest, its index checked against the table's shape."""
    lin = None
    for e, d in zip(idx, dims):
        e = torch.as_tensor(e, device=table.device)
        if bool(((e < 0) | (e >= d)).any()):
            raise ValueError(f"the nest reads {{name}} outside its shape: undefined as written")
        lin = e if lin is None else lin * d + e
    return table.reshape(-1)[lin.long()]


def indices({arglist}):
    """The nest's own index computation, per source row (program order)."""
    dev = {cname}.device
    grids = torch.meshgrid(*[torch.arange(lo, hi, st, device=dev) for _, lo, hi, st in ROWS], indexing="ij")
{bind_ivs}
    crow = torch.as_tensor({crow}, device=dev).reshape(-1).long()
    arow = torch.as_tensor({arow}, device=dev).reshape(-1).long()
    bsel = torch.as_tensor({bsel}, device=dev).reshape(-1).long()
    S = grids[0].numel() if grids else 1
    return crow.expand(S).contiguous(), arow.expand(S).contiguous(), bsel.expand(S).contiguous()


def _span(t):
    start = t.data_ptr()
    return start, start + t.untyped_storage().nbytes()


def check(args, crow, arow, bsel):
    """Bounds (refuse), separations (refuse), ranges and distinctness (choose a path)."""
    c_rows = 1
    for d in C_DIMS[:-1]:
        c_rows *= d
    a_rows = 1
    for d in A_DIMS[:-1]:
        a_rows *= d
    b_sel = 1
    for d in B_DIMS[:-2]:
        b_sel *= d
    for v, hi, what in ((crow, c_rows, "C"), (arow, a_rows, "A"), (bsel, b_sel, "B")):
        if v.numel() and bool(((v < 0) | (v >= hi)).any()):
            raise ValueError(f"a row of {{what}} outside its memref: the nest as written is undefined")
    for p in PREMISES:
        if p["kind"] == "separate":
            x, y = (args[i] for i in p["arguments"])
            (x0, x1), (y0, y1) = _span(x), _span(y)
            if x0 < y1 and y0 < x1:
                raise ValueError(f"arguments {{p['arguments']}} overlap: the record assumes them apart")
    inside, injective = True, True
    for p in PREMISES:
        if p["kind"] == "table":
            t = args[p["argument"]]
            inside = inside and bool(((t >= p["lower"]) & (t <= p["upper"])).all())
        elif p["kind"] == "range":
            v = int(args[p["argument"]])
            inside = inside and p["lower"] <= v <= p["upper"]
    if inside:
        for p in PREMISES:
            if p["kind"] == "distinct":
                t = args[p["argument"]].reshape(-1).long()
                lo = next(q["lower"] for q in PREMISES if q["kind"] == "table" and q["argument"] == p["argument"])
                injective = injective and int(torch.bincount(t - lo).max()) <= 1
    return inside, injective


def _segments(bsel):
    """Runs of consecutive source rows with one B, cut into BLOCK_M rows."""
    b = bsel.tolist()
    starts, lens = [], []
    s = 0
    while s < len(b):
        e = s
        while e < len(b) and b[e] == b[s] and e - s < BLOCK_M:
            e += 1
        starts.append(s)
        lens.append(e - s)
        s = e
    dev = bsel.device
    return torch.tensor(starts, dtype=torch.int32, device=dev), torch.tensor(lens, dtype=torch.int32, device=dev)


def _tiles_into(a, b, out, arow, orow, bsel, accumulate):
    a2 = a.reshape(-1, K)
    b3 = b.reshape(-1, K, N)
    o2 = out.reshape(-1, N)
    starts, lens = _segments(bsel)
    grid = (starts.numel(), triton.cdiv(N, BLOCK_N))
    _tiles[grid](a2, b3, o2, arow.to(torch.int32), orow.to(torch.int32), bsel.to(torch.int32), starts, lens,
                 a2.stride(0), a2.stride(1), b3.stride(0), b3.stride(1), b3.stride(2), o2.stride(0), o2.stride(1),
                 N=N, K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, ACCUMULATE=accumulate)


def parallel(a, b, c, crow, arow, bsel):
    _tiles_into(a, b, c, arow, crow, bsel, True)


def csr_reduction(a, b, c, crow, arow, bsel):
    S = crow.numel()
    y = torch.empty(S, N, device=c.device, dtype=c.dtype)
    _tiles_into(a, b, y, arow, torch.arange(S, device=c.device), bsel, False)
    c2 = c.reshape(-1, N)
    counts = torch.bincount(crow, minlength=c2.shape[0])                 # layer 1: bucket sizes
    offsets = torch.cumsum(counts, 0) - counts                           # bucket offsets
    sorted_src = torch.sort(crow, stable=True).indices.to(torch.int32)   # layer 2: sources by cell
    targets = torch.nonzero(counts).reshape(-1).to(torch.int32)
    grid = (targets.numel(), triton.cdiv(N, BLOCK_N))
    if targets.numel():
        _buckets[grid](c2, y, targets, offsets.to(torch.int32), counts.to(torch.int32), sorted_src,
                       c2.stride(0), c2.stride(1), y.stride(0), y.stride(1), N=N, BLOCK_N=BLOCK_N)


def as_written(a, b, c, crow, arow, bsel):
    a2, b3, c2 = a.reshape(-1, K), b.reshape(-1, K, N), c.reshape(-1, N)
    for s in range(crow.numel()):
        c2[crow[s]] += a2[arow[s]] @ b3[bsel[s]]


def {function}({arglist}, check_inputs=True):
    """The nest, in place on {cname}. Returns the path taken."""
    global LAST_PATH
    args = [{arglist}]
    crow, arow, bsel = indices({arglist})
    inside, injective = check(args, crow, arow, bsel)
    fast, repeated = PLAN["fast_path"], PLAN["on_repeated_index"]
    if not inside:
        path = "as_written"
    elif fast == "parallel":
        path = "parallel"
    elif fast == "parallel_if_injective":
        path = "parallel" if injective else (repeated if repeated in ("csr_reduction",) else "as_written")
    elif fast == "csr_reduction":
        path = "csr_reduction"
    else:
        path = "as_written"
    {{"parallel": parallel, "csr_reduction": csr_reduction, "as_written": as_written}}[path](
        {aname}, {bname}, {cname}, crow, arow, bsel)
    LAST_PATH = path
    return path
'''


def generate(src, record_path, function):
    nest = Nest(src, function)
    rec = read_record(record_path, function)
    if not nest.reassoc:
        raise Refusal("the accumulation carries no reassoc")
    auto = rec["auto"]
    plan = auto["plan"]
    if not rec.get("ordered_fold_verified") and plan["fast_path"] != "as_written":
        raise Refusal("a plan other than as_written without a verified fold")
    premises = auto["premises"]
    for p in premises:  # The record's arguments are the function's: names here.
        p["names"] = [nest.argnames[i] for i in (p["arguments"] if "arguments" in p else [p["argument"]])]
    rows = [(lp["iv"], nest.const[lp["lo"]], nest.const[lp["hi"]], nest.const[lp["step"]]) for lp in nest.rows]
    bind = "\n".join(f"    {_iv(iv)} = grids[{q}].reshape(-1)" for q, (iv, *_r) in enumerate(rows)) or "    pass"
    aname, aidx = nest.A
    bname, bidx = nest.B
    cname, cidx = nest.C
    pyname = lambda n: _arg(n)
    arglist = ", ".join(_arg(a) for a in nest.argnames)
    return MODULE.format(
        src=src, function=function.lstrip("@"), status=rec["status"], fold=rec["ordered_fold_verified"],
        plan=json.dumps(plan), premises=json.dumps([{k: v for k, v in p.items() if k != "names"} for p in premises]),
        argnames=repr(nest.argnames), plan_py=repr(plan),
        premises_py=repr([{k: v for k, v in p.items() if k != "names"} for p in premises]),
        N=nest.N, K=nest.K, adims=nest.adims, bdims=nest.bdims, cdims=nest.cdims, rows=repr(rows),
        block_n=max(16, min(64, 1 << (nest.N - 1).bit_length())), block_k=max(16, min(32, 1 << (nest.K - 1).bit_length())),
        arglist=arglist, bind_ivs=bind,
        crow=linear(nest, cidx[:-1], nest.cdims[:-1]), arow=linear(nest, aidx[:-1], nest.adims[:-1]),
        bsel=linear(nest, bidx[:-2], nest.bdims[:-2]),
        aname=pyname(aname), bname=pyname(bname), cname=pyname(cname))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("record")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--function", default="f")
    a = ap.parse_args(argv)
    try:
        code = generate(a.src, a.record, a.function)
    except Refusal as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    with open(a.output, "w") as f:
        f.write(code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
