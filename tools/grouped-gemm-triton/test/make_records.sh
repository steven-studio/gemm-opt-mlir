#!/bin/sh
# Regenerates the record fixtures from the sources (about 70-90 s and 2 GB each).
# Usage: make_records.sh [path/to/generalized-gemm-nest-report]
set -e
cd "$(dirname "$0")"
R=${1:-../../../build/research/generalized-gemm/generalized-gemm-nest-report}
common='--function=moe_grouped --index-bit-width=16 --max-level=1 --candidates-per-group=1 --rlimit=2000000 --timeout-ms=20000'
auth='roots 0 and 2 are separate,roots 1 and 2 are separate'
"$R" moe_grouped.mlir $common --scope=3:0:128 --authorize="$auth" > moe_grouped.record.json
"$R" moe_grouped_wide.mlir $common --scope=3:0:16 --authorize="$auth" > moe_grouped_wide.record.json
