// RUN: %nestreport %s --function=scatter3 --auto --index-bit-width=16 --max-level=1 --candidates-per-group=1 --rlimit=2000000 --timeout-ms=20000 --max-points=512 | FileCheck %s --check-prefix=SCATTER3
// RUN: %nestreport %s --function=scatter4 --auto --index-bit-width=16 --max-level=1 --candidates-per-group=1 --rlimit=2000000 --timeout-ms=20000 --max-points=512 | FileCheck %s --check-prefix=SCATTER4
// RUN: %nestreport %s --function=blocksparse --auto --index-bit-width=16 --max-level=1 --candidates-per-group=1 --rlimit=2000000 --timeout-ms=20000 --max-points=512 | FileCheck %s --check-prefix=BLOCK
// RUN: %nestreport %s --function=dense --auto --index-bit-width=16 --max-level=1 --candidates-per-group=1 --rlimit=2000000 --timeout-ms=20000 --max-points=512 | FileCheck %s --check-prefix=DENSE
//
// --auto: no --table, --scope or --authorize. The premises are read off the function (root
// separations, index tables and their ranges), each one a guard for the caller, and a table
// whose values reach the written address is speculated injective, a premise kept only when
// the record with it proves that no two logical outputs share a cell.
//
// Four ids into three rows of C: they cannot be pairwise distinct, so no speculation; the
// record without that premise verifies the fold, and the plan is the CSR reduction.
// SCATTER3: "argument 3 is an index table of 4 cells read as the arguments 4 to 7 (row-major), each in [0, 2] (inferred by --auto)
// SCATTER3: "skipped": "argument 3 has 4 cells over 3 values: they cannot be pairwise distinct"
// SCATTER3: "fast_path": "csr_reduction",
// SCATTER3: "kind": "separate"
// SCATTER3: "kind": "table",
// SCATTER3-NOT: "kind": "distinct"
// SCATTER3: "ordered_fold_verified": true,
// SCATTER3: "status": "VERIFIED_NOT_MAPPED"
//
// Four ids into four rows: speculated injective; with the premise C1 is proved, and without
// it the fold is still verified, so a repeated id takes the CSR reduction.
// SCATTER4: "the arguments 4, 5, 6, 7 of the function are pairwise distinct: a premise of every claim
// SCATTER4: "plain": {
// SCATTER4-NEXT: "index_invariants": false,
// SCATTER4-NEXT: "ordered_fold_verified": true,
// SCATTER4-NEXT: "output_injective_proven": false,
// SCATTER4: "injective": {
// SCATTER4-NEXT: "index_invariants": false,
// SCATTER4-NEXT: "ordered_fold_verified": true,
// SCATTER4-NEXT: "output_injective_proven": true,
// SCATTER4: "fast_path": "parallel_if_injective",
// SCATTER4-NEXT: "on_index_out_of_bounds": "refuse",
// SCATTER4-NEXT: "on_premise_violation": "as_written",
// SCATTER4-NEXT: "on_repeated_index": "csr_reduction"
// SCATTER4: "kind": "distinct",
// SCATTER4: "distinct": {
// SCATTER4: "premise": true,
// SCATTER4: "output_injective": {
// SCATTER4-NEXT: "detail": "C1 over the scope",
// SCATTER4-NEXT: "status": "PROVEN",
//
// Block-sparse rows C[blk[b]*2 + i]: blk in [0, 1] (the largest value with blk*2 + 1 < 4).
// BLOCK: "argument 3 is an index table of 2 cells read as the arguments 4 to 5 (row-major), each in [0, 1] (inferred by --auto)
// BLOCK: "the arguments 4, 5 of the function are pairwise distinct
// BLOCK: "fast_path": "parallel_if_injective",
// BLOCK: "on_repeated_index": "csr_reduction"
// BLOCK: "status": "VERIFIED_NOT_MAPPED"
//
// A dense GEMM: only the separations are inferred, and the record is mapped.
// DENSE: "roots 0 and 2 are separate
// DENSE: "fast_path": "parallel",
// DENSE-NOT: "kind": "table"
// DENSE: "status": "VERIFIED"

func.func @scatter3(%a: memref<4x2xf32>, %b: memref<2x2x2xf32>, %c: memref<3x2xf32>, %ids: memref<4xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %p = %c0 to %c2 step %c1 {
    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        scf.for %k = %c0 to %c2 step %c1 {
          %r0 = arith.muli %p, %c2 : index
          %r = arith.addi %r0, %i : index
          %t = memref.load %ids[%r] : memref<4xindex>
          %x = memref.load %a[%r, %k] : memref<4x2xf32>
          %y = memref.load %b[%p, %k, %j] : memref<2x2x2xf32>
          %z = memref.load %c[%t, %j] : memref<3x2xf32>
          %mm = arith.mulf %x, %y : f32
          %s = arith.addf %z, %mm fastmath<reassoc> : f32
          memref.store %s, %c[%t, %j] : memref<3x2xf32>
        }
      }
    }
  }
  return
}

func.func @scatter4(%a: memref<4x2xf32>, %b: memref<2x2x2xf32>, %c: memref<4x2xf32>, %ids: memref<4xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %p = %c0 to %c2 step %c1 {
    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        scf.for %k = %c0 to %c2 step %c1 {
          %r0 = arith.muli %p, %c2 : index
          %r = arith.addi %r0, %i : index
          %t = memref.load %ids[%r] : memref<4xindex>
          %x = memref.load %a[%r, %k] : memref<4x2xf32>
          %y = memref.load %b[%p, %k, %j] : memref<2x2x2xf32>
          %z = memref.load %c[%t, %j] : memref<4x2xf32>
          %mm = arith.mulf %x, %y : f32
          %s = arith.addf %z, %mm fastmath<reassoc> : f32
          memref.store %s, %c[%t, %j] : memref<4x2xf32>
        }
      }
    }
  }
  return
}

func.func @blocksparse(%a: memref<4x2xf32>, %w: memref<2x2xf32>, %c: memref<4x2xf32>, %blk: memref<2xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %b = %c0 to %c2 step %c1 {
    %bb = memref.load %blk[%b] : memref<2xindex>
    %base = arith.muli %bb, %c2 : index
    %abase = arith.muli %b, %c2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %row = arith.addi %base, %i : index
      %arow = arith.addi %abase, %i : index
      scf.for %j = %c0 to %c2 step %c1 {
        scf.for %k = %c0 to %c2 step %c1 {
          %x = memref.load %a[%arow, %k] : memref<4x2xf32>
          %y = memref.load %w[%k, %j] : memref<2x2xf32>
          %z = memref.load %c[%row, %j] : memref<4x2xf32>
          %mm = arith.mulf %x, %y : f32
          %s = arith.addf %z, %mm fastmath<reassoc> : f32
          memref.store %s, %c[%row, %j] : memref<4x2xf32>
        }
      }
    }
  }
  return
}

func.func @dense(%a: memref<4x2xf32>, %b: memref<2x2xf32>, %c: memref<4x2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c2 step %c1 {
      scf.for %k = %c0 to %c2 step %c1 {
        %x = memref.load %a[%i, %k] : memref<4x2xf32>
        %y = memref.load %b[%k, %j] : memref<2x2xf32>
        %z = memref.load %c[%i, %j] : memref<4x2xf32>
        %mm = arith.mulf %x, %y : f32
        %s = arith.addf %z, %mm fastmath<reassoc> : f32
        memref.store %s, %c[%i, %j] : memref<4x2xf32>
      }
    }
  }
  return
}
