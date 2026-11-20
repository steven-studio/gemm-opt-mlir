// Inputs of group_axis.test: the group axis P and the parametric row count M_p of the
// record (gemm.group_axis, gemm.parametric_extents).
// (a) batched: P constant, uniform rows
func.func @batched(%a: memref<2x4x5xi32>, %b: memref<2x5x3xi32>, %c: memref<2x4x3xi32>) {
  affine.for %p = 0 to 2 {
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 3 {
        affine.for %k = 0 to 5 {
          %x = affine.load %a[%p, %i, %k] : memref<2x4x5xi32>
          %y = affine.load %b[%p, %k, %j] : memref<2x5x3xi32>
          %z = affine.load %c[%p, %i, %j] : memref<2x4x3xi32>
          %m = arith.muli %x, %y : i32
          %s = arith.addi %z, %m : i32
          affine.store %s, %c[%p, %i, %j] : memref<2x4x3xi32>
        }
      }
    }
  }
  return
}
// (b) grouped: rows of group p are a scalar argument m (uniform parametric M_p)
func.func @grouped_param(%a: memref<2x8x5xi32>, %b: memref<2x5x3xi32>, %c: memref<2x8x3xi32>, %m: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  scf.for %p = %c0 to %c2 step %c1 {
    scf.for %i = %c0 to %m step %c1 {
      scf.for %j = %c0 to %c3 step %c1 {
        scf.for %k = %c0 to %c5 step %c1 {
          %x = memref.load %a[%p, %i, %k] : memref<2x8x5xi32>
          %y = memref.load %b[%p, %k, %j] : memref<2x5x3xi32>
          %z = memref.load %c[%p, %i, %j] : memref<2x8x3xi32>
          %mm = arith.muli %x, %y : i32
          %s = arith.addi %z, %mm : i32
          memref.store %s, %c[%p, %i, %j] : memref<2x8x3xi32>
        }
      }
    }
  }
  return
}
// (c) MoE grouped: rows of group p from offsets[p]..offsets[p+1] (data-dependent)
func.func @grouped_offsets(%a: memref<16x5xi32>, %b: memref<2x5x3xi32>, %c: memref<16x3xi32>, %off: memref<3xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  scf.for %p = %c0 to %c2 step %c1 {
    %q = arith.addi %p, %c1 : index
    %lo = memref.load %off[%p] : memref<3xindex>
    %hi = memref.load %off[%q] : memref<3xindex>
    scf.for %i = %lo to %hi step %c1 {
      scf.for %j = %c0 to %c3 step %c1 {
        scf.for %k = %c0 to %c5 step %c1 {
          %x = memref.load %a[%i, %k] : memref<16x5xi32>
          %y = memref.load %b[%p, %k, %j] : memref<2x5x3xi32>
          %z = memref.load %c[%i, %j] : memref<16x3xi32>
          %mm = arith.muli %x, %y : i32
          %s = arith.addi %z, %mm : i32
          memref.store %s, %c[%i, %j] : memref<16x3xi32>
        }
      }
    }
  }
  return
}
