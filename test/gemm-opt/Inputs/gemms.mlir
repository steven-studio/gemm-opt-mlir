// Inputs of the gemm-opt tests: three GEMMs (affine, affine without reassoc, scf.for) and a
// loop with no multiply-accumulate. Roots 0 and 1 are A and B, root 2 is C.

// One product over a constant box, in binary32 with reassoc on the accumulation.
func.func @single(%a: memref<16x12xf32>, %b: memref<12x8xf32>, %c: memref<16x8xf32> {llvm.noalias}) {
  affine.for %i = 0 to 16 {
    affine.for %j = 0 to 8 {
      affine.for %k = 0 to 12 {
        %z = affine.load %c[%i, %j] : memref<16x8xf32>
        %x = affine.load %a[%i, %k] : memref<16x12xf32>
        %y = affine.load %b[%k, %j] : memref<12x8xf32>
        %m = arith.mulf %x, %y : f32
        %s = arith.addf %z, %m fastmath<reassoc> : f32
        affine.store %s, %c[%i, %j] : memref<16x8xf32>
      }
    }
  }
  return
}

// The same product with no reassoc: only the ordered fold is the source's.
func.func @strict(%a: memref<16x12xf32>, %b: memref<12x8xf32>, %c: memref<16x8xf32> {llvm.noalias}) {
  affine.for %i = 0 to 16 {
    affine.for %j = 0 to 8 {
      affine.for %k = 0 to 12 {
        %z = affine.load %c[%i, %j] : memref<16x8xf32>
        %x = affine.load %a[%i, %k] : memref<16x12xf32>
        %y = affine.load %b[%k, %j] : memref<12x8xf32>
        %m = arith.mulf %x, %y : f32
        %s = arith.addf %z, %m : f32
        affine.store %s, %c[%i, %j] : memref<16x8xf32>
      }
    }
  }
  return
}

// The product written with scf.for, memref.load and memref.store.
func.func @scf_gemm(%a: memref<16x12xf32>, %b: memref<12x8xf32>, %c: memref<16x8xf32> {llvm.noalias}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    scf.for %j = %c0 to %c8 step %c1 {
      scf.for %k = %c0 to %c12 step %c1 {
        %z = memref.load %c[%i, %j] : memref<16x8xf32>
        %x = memref.load %a[%i, %k] : memref<16x12xf32>
        %y = memref.load %b[%k, %j] : memref<12x8xf32>
        %m = arith.mulf %x, %y : f32
        %s = arith.addf %z, %m fastmath<reassoc> : f32
        memref.store %s, %c[%i, %j] : memref<16x8xf32>
      }
    }
  }
  return
}

// A loop with no multiply-accumulate.
func.func @zero(%c: memref<16x8xf32>) {
  %zero = arith.constant 0.000000e+00 : f32
  affine.for %i = 0 to 16 {
    affine.for %j = 0 to 8 {
      affine.store %zero, %c[%i, %j] : memref<16x8xf32>
    }
  }
  return
}
