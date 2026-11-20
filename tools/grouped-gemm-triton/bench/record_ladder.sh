#!/bin/bash
# Record one size under a cgroup memory cap; prints one TSV line.
#   record_ladder.sh OUTDIR E CAP N K [MEMMAX=22G] [TIMEOUT=3600] [REPORTER]
# Flags are the fixtures' (test/make_records.sh): --scope=3:0:CAP, the same --authorize.
set -u
OUT=$1 E=$2 CAP=$3 N=$4 K=$5 MEM=${6:-22G} TMO=${7:-3600}
HERE=$(cd "$(dirname "$0")" && pwd)
R=${8:-$HERE/../../../build/research/generalized-gemm/generalized-gemm-nest-report}
tag=e${E}_m${CAP}_n${N}_k${K}
mkdir -p "$OUT"
python3 "$HERE/make_source.py" $E $CAP $N $K "$OUT/$tag.mlir"
common="--function=moe_grouped --index-bit-width=16 --max-level=1 --candidates-per-group=1 --rlimit=2000000 --timeout-ms=20000 --scope=3:0:$CAP"
auth='roots 0 and 2 are separate,roots 1 and 2 are separate'
start=$(date +%s.%N)
systemd-run --user --scope -q -p MemoryMax=$MEM -p MemorySwapMax=0 \
  /usr/bin/time -v timeout $TMO "$R" "$OUT/$tag.mlir" $common --authorize="$auth" \
  > "$OUT/$tag.record.json" 2> "$OUT/$tag.stderr"
rc=$?
end=$(date +%s.%N)
peak=$(grep "Maximum resident" "$OUT/$tag.stderr" | awk '{print $NF}')
status=$(python3 -c "
import json,sys
try:
  n=json.load(open('$OUT/$tag.record.json'))[0]['nests'][0]; g=n.get('gemm') or {}
  print(n['status'], (g.get('tiers',{}).get('computation',{}).get('symbolic',{}) or {}).get('status'))
except Exception as e: print('NO_RECORD', type(e).__name__)
")
printf "%s\trc=%s\twall=%.1fs\tpeak_rss_kb=%s\t%s\n" $tag $rc $(echo "$end - $start" | bc) "${peak:-?}" "$status"
