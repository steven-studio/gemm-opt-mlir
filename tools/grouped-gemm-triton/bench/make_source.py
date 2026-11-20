#!/usr/bin/env python3
"""Write the moe_grouped source (tools/grouped-gemm-triton/test/moe_grouped.mlir) at another size.
  make_source.py E CAP N K OUT.mlir
Same nest, same reassoc, same scalar row argument %m; only the memref extents and the
P/J/K loop bounds change."""
import sys
E, CAP, N, K = map(int, sys.argv[1:5])
a, b, c = f"memref<{E}x{CAP}x{K}xf32>", f"memref<{E}x{K}x{N}xf32>", f"memref<{E}x{CAP}x{N}xf32>"
src = f"""// moe_grouped at E={E}, CAP={CAP}, N={N}, K={K} (bench/make_source.py).
func.func @moe_grouped(%a: {a}, %b: {b}, %c: {c}, %m: index) {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %ce = arith.constant {E} : index
  %ck = arith.constant {K} : index
  %cn = arith.constant {N} : index
  scf.for %p = %c0 to %ce step %c1 {{
    scf.for %i = %c0 to %m step %c1 {{
      scf.for %j = %c0 to %cn step %c1 {{
        scf.for %k = %c0 to %ck step %c1 {{
          %x = memref.load %a[%p, %i, %k] : {a}
          %y = memref.load %b[%p, %k, %j] : {b}
          %z = memref.load %c[%p, %i, %j] : {c}
          %mm = arith.mulf %x, %y : f32
          %s = arith.addf %z, %mm fastmath<reassoc> : f32
          memref.store %s, %c[%p, %i, %j] : {c}
        }}
      }}
    }}
  }}
  return
}}
"""
open(sys.argv[5], "w").write(src)
