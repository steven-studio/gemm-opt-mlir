// RUN: %nestreport %s --function=rows --table=3:0:4 --index-bit-width=16 --max-level=1 --candidates-per-group=1 --rlimit=2000000 --timeout-ms=20000 --authorize="roots 0 and 2 are separate,roots 1 and 2 are separate,roots 2 and 3 are separate" | FileCheck %s --check-prefix=ROWS
// RUN: %nestreport %s --function=rows --index-bit-width=16 --max-level=1 --candidates-per-group=1 --rlimit=2000000 --timeout-ms=20000 --authorize="roots 0 and 2 are separate,roots 1 and 2 are separate" | FileCheck %s --check-prefix=MEMORY
// RUN: %nestreport %s --function=rows --table=3:0:4 --index-bit-width=16 --max-level=1 --candidates-per-group=1 --rlimit=2000000 --timeout-ms=20000 --authorize="roots 0 and 2 are separate,roots 1 and 2 are separate" | FileCheck %s --check-prefix=APART
// RUN: %nestreport %s --function=gather --table=3:0:2 --scope=4:0:2 --max-points=512 --index-bit-width=16 --max-level=1 --candidates-per-group=1 --rlimit=2000000 --timeout-ms=20000 --authorize="roots 0 and 2 are separate,roots 1 and 2 are separate,roots 2 and 3 are separate" | FileCheck %s --check-prefix=GATHER
// RUN: %nestreport %s --function=csr --table=3:0:4 --index-invariants --max-points=512 --index-bit-width=16 --max-level=1 --candidates-per-group=1 --rlimit=2000000 --timeout-ms=20000 --authorize="roots 0 and 2 are separate,roots 1 and 2 are separate,roots 2 and 3 are separate" | FileCheck %s --check-prefix=CSR
// RUN: not %nestreport %s --function=written --table=3:0:4 --index-bit-width=16 2>&1 | FileCheck %s --check-prefix=WRITTEN
// RUN: not %nestreport %s --function=rows --table=3:4:0 --index-bit-width=16 2>&1 | FileCheck %s --check-prefix=MALFORMED
//
// --table: a memref argument read only by memref.load, each cell an appended scalar argument
// with a scope; each load reads its cell by a select over the linear index, and the load is
// kept, so the checkers need the table apart from every root the nest writes.
//
// Per-group row counts read from memory (rows[p]): the bound of I is the select over p of
// the two cells, the record is verified as the scalar-bound record is (batches unmapped).
// ROWS: "argument 3 is an index table of 2 cells read as the arguments 4 to 5 (row-major), each in [0, 4] (--table)
// ROWS: "argument 4 of the function is in [0, 4]: the scope of every claim",
// ROWS-NEXT: "argument 5 of the function is in [0, 4]: the scope of every claim",
// ROWS: "symbolic": "SYMBOLIC_VALID"
// ROWS: "output_injective": {
// ROWS-NEXT: "detail": "C1 over the scope",
// ROWS-NEXT: "status": "PROVEN",
// ROWS: "logical_tier": "SYMBOLIC",
// ROWS: "ordered_fold_verified": true,
// ROWS: "status": "VERIFIED_NOT_MAPPED"
//
// Without --table the trace reads memory, and no claim is made.
// MEMORY: "no claim was made about the nest: the gate is UNSUPPORTED (the trace depends on memory contents)"
// MEMORY: "status": "NOT_VERIFIED"
//
// With --table and no separation of the table from C, the kept load stops the claim.
// APART: "reason": "not authorized: roots 2 and 3 are separate",
// APART: "status": "NOT_VERIFIED"
//
// A gather X[ids[p*2 + i]] with a scalar bound: I is the linear index, from 2p to 2p + m.
// GATHER: "ordered_fold_verified": true,
// GATHER: "status": "VERIFIED_NOT_MAPPED"
//
// CSR row pointers: rows [rowptr[p], rowptr[p+1]); no two groups share a row, proved with
// the loop invariant rowptr[p] <= r the invariant search finds.
// CSR: "output_injective": {
// CSR-NEXT: "detail": "C1 over the scope",
// CSR-NEXT: "status": "PROVEN",
// CSR: "status": "INDUCTIVE"
// CSR: "ordered_fold_verified": true,
// CSR: "status": "VERIFIED_NOT_MAPPED"
//
// WRITTEN: --table refused for written: argument 3 has a use that is no memref.load: memref.store
// MALFORMED: malformed --table

func.func @rows(%a: memref<2x4x2xf32>, %b: memref<2x2x2xf32>, %c: memref<2x4x2xf32>, %rows: memref<2xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %p = %c0 to %c2 step %c1 {
    %m = memref.load %rows[%p] : memref<2xindex>
    scf.for %i = %c0 to %m step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        scf.for %k = %c0 to %c2 step %c1 {
          %x = memref.load %a[%p, %i, %k] : memref<2x4x2xf32>
          %y = memref.load %b[%p, %k, %j] : memref<2x2x2xf32>
          %z = memref.load %c[%p, %i, %j] : memref<2x4x2xf32>
          %mm = arith.mulf %x, %y : f32
          %s = arith.addf %z, %mm fastmath<reassoc> : f32
          memref.store %s, %c[%p, %i, %j] : memref<2x4x2xf32>
        }
      }
    }
  }
  return
}
func.func @gather(%a: memref<3x2xf32>, %b: memref<2x2x2xf32>, %c: memref<2x2x2xf32>, %ids: memref<4xindex>, %m: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %p = %c0 to %c2 step %c1 {
    scf.for %i = %c0 to %m step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        scf.for %k = %c0 to %c2 step %c1 {
          %r0 = arith.muli %p, %c2 : index
          %r = arith.addi %r0, %i : index
          %t = memref.load %ids[%r] : memref<4xindex>
          %x = memref.load %a[%t, %k] : memref<3x2xf32>
          %y = memref.load %b[%p, %k, %j] : memref<2x2x2xf32>
          %z = memref.load %c[%p, %i, %j] : memref<2x2x2xf32>
          %mm = arith.mulf %x, %y : f32
          %s = arith.addf %z, %mm fastmath<reassoc> : f32
          memref.store %s, %c[%p, %i, %j] : memref<2x2x2xf32>
        }
      }
    }
  }
  return
}
func.func @csr(%a: memref<4x2xf32>, %w: memref<2x2x2xf32>, %c: memref<4x2xf32>, %rowptr: memref<3xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %p = %c0 to %c2 step %c1 {
    %lo = memref.load %rowptr[%p] : memref<3xindex>
    %p1 = arith.addi %p, %c1 : index
    %hi = memref.load %rowptr[%p1] : memref<3xindex>
    scf.for %r = %lo to %hi step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        scf.for %k = %c0 to %c2 step %c1 {
          %x = memref.load %a[%r, %k] : memref<4x2xf32>
          %y = memref.load %w[%p, %k, %j] : memref<2x2x2xf32>
          %z = memref.load %c[%r, %j] : memref<4x2xf32>
          %mm = arith.mulf %x, %y : f32
          %s = arith.addf %z, %mm fastmath<reassoc> : f32
          memref.store %s, %c[%r, %j] : memref<4x2xf32>
        }
      }
    }
  }
  return
}
func.func @written(%a: memref<2x4x2xf32>, %b: memref<2x2x2xf32>, %c: memref<2x4x2xf32>, %rows: memref<2xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  memref.store %c1, %rows[%c0] : memref<2xindex>
  return
}
