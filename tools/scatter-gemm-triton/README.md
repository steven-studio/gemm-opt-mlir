# scatter-gemm-triton: scatter-add and block-sparse rows from an `--auto` record

    generalized-gemm-nest-report SRC.mlir --function=f --index-bit-width=16 --auto ... > REC.json
    python3 scatter_gemm_triton.py SRC.mlir REC.json -o module.py [--function f]
    python3 test/test_scatter_gemm_triton.py SRC.mlir REC.json module.py    # needs a GPU

For nests `C[row(i)][j] += A[arow(i)][k] * B[bsel(i)][k][j]` whose C row goes through an
index table: scatter-add `C[ids[r]][j]`, block-sparse rows `C[blk[b]*B + i][j]`.

## Where each step comes from

The reporter's `--auto` (research/generalized-gemm/NestReport.cpp) needs no premise flag:
- it infers the root separations, the index tables and their ranges, and the scalar ranges
  (the largest values that keep every index in bounds);
- it analyses the nest (pass A), and again with index invariants when C1 is not proved;
- it speculates that a table whose values reach the written address is injective (pass B,
  the premise `distinct`), and keeps pass B only when it proves C1;
- it writes `auto.premises` and `auto.plan`.

The plan's fast path is one of:

| fast path | when |
|---|---|
| `parallel` | C1 proved under the inferred premises. |
| `parallel_if_injective` | C1 proved only under `distinct`. `on_repeated_index` is `csr_reduction` when pass A (without `distinct`) verifies the fold, else `as_written`. |
| `csr_reduction` | The fold is verified without `distinct`, but C1 is not proved; for example, a table with more cells than values can never be injective. |
| `serial` | The fold is verified, but there is no parallel claim. |
| `as_written` | There is no claim. |

The module re-emits the nest's own index computation from the source (systolic-mlir's
Track B does the same) and checks at every call:
- **bounds:** refuse if violated;
- **separations:** refuse if violated;
- **table and scalar ranges:** if violated, the record does not apply and the nest runs as
  written;
- **distinctness:** an O(cells + range) count per value; if violated, the
  `on_repeated_index` path runs.

## Paths

| path | what runs |
|---|---|
| parallel | Triton tiles of up to BLOCK_M source rows that share a B. The K fold is computed by `tl.dot`, and each tile adds into C directly. |
| csr_reduction | The same tiles write `Y[s]`. Then comes the two-layer inverse map: a count per target row, its prefix sum, and the sources sorted stably by target. One program per target row adds its bucket in program order, so each cell of C is written once. |
| as_written | One source row at a time, in program order. |

`tl.dot` reorders the K fold, so a record without `reassociation_permitted` is refused.

## Scope

- The row loops and j/k must have constant bounds.
- The update must be `C[..., j] += A[..., k] * B[..., k, j]` in f32.
- The tile size is fixed (BLOCK_M 16).
- No performance claim is made; the tests check correctness against a literal float64
  interpreter of the source.
