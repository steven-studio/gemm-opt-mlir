// The source of the grouped GEMM the Triton generator reads a record of
// (tools/grouped-gemm-triton): the wide variant. E = 4 groups of at most CAP = 16 rows,
// N = 64, K = 32 (one bucket; it exercises the N and K extents of the kernels),
// laid out with a padded capacity per group. The rows of a group are a scalar argument
// %m: the record's symbol M_p, with the scope the generator buckets over. The P loop is
// the group axis. binary32 with `reassoc`: the source permits the tiled order of the fold.
func.func @moe_grouped(%a: memref<4x16x32xf32>, %b: memref<4x32x64xf32>, %c: memref<4x16x64xf32>, %m: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  scf.for %p = %c0 to %c4 step %c1 {
    scf.for %i = %c0 to %m step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        scf.for %k = %c0 to %c32 step %c1 {
          %x = memref.load %a[%p, %i, %k] : memref<4x16x32xf32>
          %y = memref.load %b[%p, %k, %j] : memref<4x32x64xf32>
          %z = memref.load %c[%p, %i, %j] : memref<4x16x64xf32>
          %mm = arith.mulf %x, %y : f32
          %s = arith.addf %z, %mm fastmath<reassoc> : f32
          memref.store %s, %c[%p, %i, %j] : memref<4x16x64xf32>
        }
      }
    }
  }
  return
}
