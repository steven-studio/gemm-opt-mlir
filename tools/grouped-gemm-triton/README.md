# grouped-gemm-triton

Generates Triton grouped-GEMM kernels (MoE experts) from a Generalized GEMM nest record.

```
grouped_gemm_triton.py RECORD.json -o kernels.py [--function F] [--nest N]
                       [--thresholds 16,64 | --fixed-block-m {16,32,64,128,256}]
```

`--fixed-block-m B` generates one kernel over the whole scope with `BLOCK_M = B` (no buckets;
each group becomes `ceil(M_p / B)` work items), the baseline for the bucketed default.

`RECORD.json` is the output of `generalized-gemm-nest-report` (or `gemm-opt --nest-report`)
on a nest whose outer loop is the group axis P and whose row loop is bounded by a scalar
argument of the function (the record's `M_p`) given a `--scope`. The generated module has:

* one `@triton.jit` kernel per bucket of `M_p` (default `<=16`, `17..64`, `>64`, clipped to
  the scope), with `BLOCK_M` 16 / 64 / 128 and `N`, `K` taken from the record;
* `grouped_gemm(x, w, c, sorted_token_ids, group_offsets, group_rows)`: computes
  `C[group_offsets[p] + i] += X[sorted_token_ids[group_offsets[p] + i]] @ W[p]` for
  `i < group_rows[p]`. The gather index comes from the caller. With `group_offsets[p] = p*CAP`
  and the identity ids it is the record's own padded layout;
* `build_worklist`: one O(E) pass over the groups that puts each group into its bucket as
  `ceil(M_p / BLOCK_M)` items. Each bucket's kernel is then launched once over its items.

## What it reads, and what it refuses

It reads only fields of the record: `status` (VERIFIED or VERIFIED_NOT_MAPPED), the symbolic
computation tier, the update `C += A*B`, `axes`/`levels` (one level each for P, I, J, K;
constant J and K from 0), the views (`[P,I,K]`, `[P,K,J]`, `[P,I,J]`, coefficient 1),
`group_axis`, and `parametric_extents` (one, on I, affine, `uniform_over_groups`, from 0 to
one scoped argument). `tl.dot` reorders the K fold, so a float record also needs
`reassociation.permitted` (the source's `reassoc`), and an integer record needs its
reorder `PROVEN`. Anything else is refused with a reason (exit 2).

Things added beyond the record, or assumed by it, are checked at run time before any launch:

* Each group gets its own `M_p`. The record covers every value in the scope; a count outside
  it raises an error.
* The gather and compact output rename cells. A is only read. The rows of C must not overlap
  between groups, and the dispatcher raises an error when they do.
* Every id the kernels read (`sorted_token_ids` at positions inside a group) must be a row of
  X: `0 <= id < X.shape[0]`. Padding positions outside every group are not read and not
  checked. One `aminmax` over the ids is read back with the row counts (one host sync); only
  when it is out of range are the read positions checked one by one.
* C is the only buffer written. It must share no byte with X, W, `sorted_token_ids`,
  `group_offsets` or `group_rows` (byte intervals from shape and strides; conservative). The
  record only authorizes A/C and B/C as separate; it does not prove it.

`grouped_gemm(..., check_inputs=False)` skips the last two checks (about 65 us per call on an
RTX 2080 SUPER at small sizes); the caller then owns those conditions.

Not checked, and not covered by the record: the kernels use `tl.dot`, which compiles to fused
multiply-add (`fma.rn.f32`); the source permits reassociation (`reassoc`) but not
contraction, so results match the source to fp32 rounding, not bit for bit.

Out of scope: row counts read from memory (`offsets[p]..offsets[p+1]`, dp[E]). The record
makes no claim for them (the gate is UNSUPPORTED), and the G3a/G3b checkers are unchanged.

## Test

```
test/make_records.sh            # regenerate the two record fixtures (~2 min, 2 GB each)
python3 test/test_grouped_gemm_triton.py test/moe_grouped.record.json
python3 test/test_grouped_gemm_triton.py test/moe_grouped_wide.record.json
```

It needs CUDA, torch and triton. It checks the generated kernels against PyTorch in float64:
the source loop nest in the record's layout (one M, and a different M_p per group at the
bucket edges), and an MoE top-2 layer with the gather, compact output and an empty expert.
It also checks the dispatcher's and the generator's refusals (including the id-range and
disjointness guards, and that out-of-range padding ids outside every group are accepted), and
the `--fixed-block-m 16/64/128` modules on the same per-group cases.

`bench/` holds benchmarks (`bench_block_m.py` at record sizes, `bench_large.py` at
compute-bound sizes with the shape overridden and marked as not covered by the record) and
`reports/`. The fixtures are small
(4x128x8x4 and 4x16x64x32) because the record's cost grows with the memrefs: 4x128x64x32 ran
out of 28 GB.
