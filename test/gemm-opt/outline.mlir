// gemm-opt --outline-only: each outermost loop's private copy, analysed by nothing.
// RUN: %gemm-opt %S/Inputs/gemms.mlir --outline-only --function=scf_gemm | FileCheck %s
// CHECK: // function scf_gemm
// CHECK-NEXT: // nest 0 (scf.for, statement {{[0-9]+}})
// CHECK: func.func
// CHECK: scf.for
// CHECK: arith.mulf
// CHECK-NOT: // nest 1
