# Mixed per-group M_p: what composing the record per group needs, and what is checked

Repo HEAD 34c8e0e (main), tools/grouped-gemm-triton. Record: test/moe_grouped.record.json
(E=4, CAP=128, N=8, K=4, scope argument 3 in [0,128]). Probes: probe_conditions.py,
probe.log, probe_results.json, m*_*.ptx in this directory.

## What the record states (and only this)

For every value m of argument 3 in [0,128], one value for every group
(`uniform_over_groups: true`; NestRecord.h:137-141, docs/GENERALIZED_GEMM_SCOPE.md:1295-1297):
the nest computes C[h] := C[h] + A[f]*B[g] in the order of its levels, starting from the
cell's entry value, with views A=[P,I,K], B=[P,K,J], C=[P,I,J] (coefficient 1),
`frame.other_writes = 0`, `output_injective` PROVEN (C1, symbolic), `cell_key` PROVEN (K1),
domain and computation VERIFIED at the symbolic tier. Assumptions: roots 0/2 and 1/2 are
separate (authorized, not proved); index values are 16-bit bit vectors; runs that complete.
The float reorder fact is REFUTED; reordering is permitted only by the source's `reassoc`.

## Conditions for running group p with its own M_p

| # | Condition | What the code checks or proves | Kind of evidence |
|---|---|---|---|
| 1 | Group p's final cells depend only on A[p], B[p], C[p] and on m (so group p of the run with m = M_p is well defined independently of other groups) | Follows from the record's views (P only in dim 0, coefficient 1) + frame (no other writes) + per-cell statement. Generator checks the views statically: `_check_view` (grouped_gemm_triton.py:86, 168-170) and frame/guard (:131) | Derivable from proven record fields; the composition step itself is argued only in prose (docstring :18-25, README:32-36). No checker states or discharges it |
| 2 | Each M_p lies in the scope | `build_worklist` raises (generated module; template line :308) | Run-time check, tested |
| 3 | Gather A[p,i,k] -> X[ids[off[p]+i], k] is a renaming of reads only | A is only read (frame other_writes 0); many-to-one ids are harmless | Argued; tested (mixed_mp_correctness, MoE top-2 test) |
| 3a | ids within [0, X.rows) | **Not checked.** id 10^6 -> CUDA illegal memory access (device fault); id -5 -> accepted silently, out-of-bounds read (probe_results.json) | Gap |
| 4 | C renaming C[p,i,j] -> C[off[p]+i, j] is injective over all groups, inside C | Spans of groups sorted and checked for overlap and bounds (template :332, :334); within a group injective by construction (off + i) | Run-time check, tested |
| 5 | Every (row, column) of each group computed exactly once, no write race | Worklist `range(0, m, bm)` (:313), disjoint pid_n tiles, masks `offs_m < rows` (:253); launches serialized on one stream | Code reading + test (rows_covered_once over 144 cases); no proof |
| 6 | No cross-group read/write dependency | Writes only C rows of p; reads X, W[p], C rows of p. Holds if (4) and X, W do not overlap C | Partly: (4) checked; **X/W vs C overlap not checked** -- X aliasing C and W aliasing C both accepted (probe). The record covers this only by the authorized separation of roots 0/2, 1/2 |
| 7 | Accumulation C_final = C_entry + sum_k A*B, rows past M_p unchanged | Kernel loads C, adds acc, stores (:272-273); masks keep other rows bit-for-bit (tested) | Tested |
| 8 | The kernel's order of floating-point operations is one the record permits | Generator requires `reassociation.permitted` (:139-146). **PTX of all three kernels is 32/128/128 `fma.rn.f32`, 0 separate mul/add.** The source has `fastmath<reassoc>` only on addf, no `contract`: fused multiply-add (one rounding instead of two) is not a reassociation and is not permitted by the record | Gap at bit level. Results agree within 1e-4 (max |err| <= 1.14e-6 vs float64), not bitwise with the source order |
| 9 | Index arithmetic of the real tensors is within the record's model | Record claims are about 16-bit index values; real offsets/ids are int32 with int64 pointer math | Outside the record: the renamed layout is not modelled; no check |
| 10 | dtype f32 as in the record | `assert x.dtype == w.dtype == c.dtype == torch.float32` (:325), shape asserts (:324) | Run-time check |

## Verdict

Mixed M_p is supported by: (a) proven record facts per uniform m that make the per-group
decomposition derivable by hand (views, frame, output injectivity), (b) run-time checks of
scope and of disjoint, in-bounds C rows, (c) tests: 144/144 mixed-M_p cases (round 1) plus
the repo tests. It is **not** covered by any checker or mechanised proof, and three conditions
are unchecked at run time or outside the record's permission: id bounds (3a), X/W-vs-C
overlap (6), FMA contraction (8). Claiming "the record covers mixed M_p" is not supported;
"mixed M_p is correct up to fp32 rounding on the tested inputs, given in-range ids and
non-overlapping X/W/C" is.
