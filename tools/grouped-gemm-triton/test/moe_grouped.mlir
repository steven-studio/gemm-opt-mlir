// The source of the grouped GEMM the Triton generator reads a record of
// (tools/grouped-gemm-triton). E = 4 groups (experts) of at most CAP = 128 rows each, N = 8,
// K = 4 (the record of 4x128x64x32 ran out of 28 GB; its cost grows with the memrefs),
// laid out with a padded capacity per group. The rows of a group are a scalar argument
// %m: the record's symbol M_p, with the scope the generator buckets over. The P loop is
// the group axis. binary32 with `reassoc`: the source permits the tiled order of the fold.
func.func @moe_grouped(%a: memref<4x128x4xf32>, %b: memref<4x4x8xf32>, %c: memref<4x128x8xf32>, %m: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 4 : index
  %c64 = arith.constant 8 : index
  scf.for %p = %c0 to %c4 step %c1 {
    scf.for %i = %c0 to %m step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        scf.for %k = %c0 to %c32 step %c1 {
          %x = memref.load %a[%p, %i, %k] : memref<4x128x4xf32>
          %y = memref.load %b[%p, %k, %j] : memref<4x4x8xf32>
          %z = memref.load %c[%p, %i, %j] : memref<4x128x8xf32>
          %mm = arith.mulf %x, %y : f32
          %s = arith.addf %z, %mm fastmath<reassoc> : f32
          memref.store %s, %c[%p, %i, %j] : memref<4x128x8xf32>
        }
      }
    }
  }
  return
}
