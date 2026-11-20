# CP -> Triton round 3 (2026-11-20): run-time guards, --fixed-block-m, compute-bound bench

Repo /home/steven-studio/work/gemm-opt-mlir, main @ 7123002 + uncommitted changes in
tools/grouped-gemm-triton (grouped_gemm_triton.py, README.md, test/, bench/). GPU RTX 2080
SUPER, torch 2.12.0+cu130, triton 3.7.0.

In the repo this directory holds: README.md, speedup_table.md, large/bench_large.json (all
raw samples of the compute-bound bench), large/large.log, tests/*.log (repo tests on the
final code). Generated modules and the mixed-M_p results.json stay in
~/work/gemm-opt-mlir-validation/grouped-gemm-triton/cp-triton-20261120-r3/.

## Changes
- Dispatcher guards (generated module): `check_token_ids` (ids read by the kernels must be
  rows of X; padding outside every group is not checked), `check_disjoint` (C shares no byte
  with X, W, sorted_token_ids, group_offsets, group_rows), int32 index dtypes asserted.
  One host sync for offsets, rows and the ids' aminmax. `check_inputs=False` skips the two
  guards. Cost A/B vs HEAD dispatcher at 4x128x8x4, rows [8,40,100,0], 3 rounds x 300 calls:
  314/313/317 us -> 382/379/376 us (~+65 us); invisible at compute-bound sizes (e2e vs
  e2e_unchecked in large/bench_large.json).
- CLI `--fixed-block-m {16,32,64,128,256}` (exclusive with --thresholds).
- bench_block_m.py: fixed16/64/128 from the CLI (no patching); split control unchanged.
- bench_large.py: shape override through the generator API, header-marked as not covered by
  the record.

## Tests (final code; final_tests/)
    python3 tools/grouped-gemm-triton/test/test_grouped_gemm_triton.py <record> --keep DIR
moe_grouped 38/38, moe_grouped_wide 34/34, e4_m32_n64_k32 36/36 (was 22/18/22).
mixed_mp_correctness.py: E=4 60/60, E=3 84/84.

## Compute-bound bench (large/, large.log, speedup_table.md)
    python3 tools/grouped-gemm-triton/bench/bench_large.py tools/grouped-gemm-triton/test/moe_grouped.record.json $O/large
N=2048, K in {2048, 4096}, E in {8, 64}, M_p in the record's scope [0,128], 6 workloads,
30 iterations; every result allclose to float64 (max |err| <= 1.8e-5).
See speedup_table.md.
