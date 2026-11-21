# Coverage: CP vs MLIR affine vs Polly

One GEMM-like loop nest per case (`cases.py`), five categories: affine home ground, a shared
parametric bound, data-dependent bounds (per-group row counts, CSR row pointers), indirect
access (gather, scatter-add), non-linear indices (`i*i`, `i/2`, `i%2`, block-sparse rows).
Each case is run through:

| Tool | Input | Command |
|---|---|---|
| CP (this branch) | the scf source | `generalized-gemm-nest-report` with the case's `--scope` / `--table` / `--authorize` |
| CP (main) | the same, `--table` removed | the main checkout's reporter |
| MLIR affine (raise) | the same scf source | `mlir-opt --raise-scf-to-affine --affine-raise-from-memref --affine-parallelize` |
| MLIR affine (hand-written) | affine dialect written by hand, where the dialect can say it | `mlir-opt --affine-parallelize` |
| Polly default / permissive | the nest in C, extents 64 | clang 24 `-O1 -Xclang -disable-llvm-passes`, then `opt -passes='default<O3>' -polly -polly-process-unprofitable -polly-ast-detect-parallel -polly-print-ast` (permissive adds `-polly-allow-nonaffine{,-loops,-branches}`) |

    python3 run.py OUTDIR            # all cases; --cases a,b for some
    python3 make_table.py OUTDIR     # OUTDIR/coverage_table.md

Classes (run.py): ACCEPT_PARALLEL / ACCEPT_SERIAL / REJECT (with the tool's reason) /
ERROR. For CP, ACCEPT means VERIFIED or VERIFIED_NOT_MAPPED with the ordered fold verified,
and PARALLEL means output injectivity over (P, I, J) proved (K the reduction). For Polly, a
SCoP that covers only inner loops is REJECT; a whole-nest SCoP with inner loops boxed as one
non-affine statement is ACCEPT, parallel only if an outer loop is known-parallel.

CP's extents are small and static: its finite tier enumerates the scope, and a table of
`n` cells over `[lo, hi]` has `(hi - lo + 1)^n` points.
