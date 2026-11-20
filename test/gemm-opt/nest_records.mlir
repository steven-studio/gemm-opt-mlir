// gemm-opt --nest-records (the default mode): the records file a backend reads. A JSON
// array, one function object per line, schema generalized-gemm-nest/2; each nest's record
// carries the binding (GemmOpt/NestBinding.h) a reader recomputes of the nest in front of it.
// The solver's resource limit, not its time limit, decides every query.
//
// RUN: %gemm-opt %S/Inputs/gemms.mlir "--authorize=roots 0 and 2 are separate,roots 1 and 2 are separate" --max-level=1 --candidates-per-group=1 --timeout-ms=120000 --rlimit=20000000 -o %t.json
// RUN: FileCheck %s --check-prefix=FILE < %t.json
// RUN: %gemm-opt %S/Inputs/gemms.mlir --nest-records "--authorize=roots 0 and 2 are separate,roots 1 and 2 are separate" --max-level=1 --candidates-per-group=1 --timeout-ms=120000 --rlimit=20000000 | diff %t.json -
//
// FILE: {{^\[$}}
// FILE-NEXT: {{^\{"authorizes_lowering":false,.*"function":"single",.*"schema":"generalized-gemm-nest/2",.*\},$}}
// FILE-NEXT: {{^\{.*"function":"strict",.*\},$}}
// FILE-NEXT: {{^\{.*"function":"scf_gemm",.*\},$}}
// FILE-NEXT: {{^\{.*"function":"zero",.*\}$}}
// FILE-NEXT: {{^\]$}}
// FILE-NOT: {{.}}
//
// Each record of a GEMM is VERIFIED with its binding at the module's 64-bit index; the loop
// that only stores a constant has no multiply-accumulate site.
// RUN: FileCheck %s --check-prefix=STATUS < %t.json
// STATUS: "function":"single"
// STATUS-SAME: "binding":{"index_width":64,"reason":"","scheme":"generalized-gemm-nest-binding/1","sha256":"{{[0-9a-f]+}}"}
// STATUS-SAME: "element":"f32"
// STATUS-SAME: "reduction":"fadd"
// STATUS-SAME: "status":"VERIFIED"
// STATUS: "function":"strict"
// STATUS-SAME: "status":"VERIFIED"
// STATUS: "function":"scf_gemm"
// STATUS-SAME: "status":"VERIFIED"
// STATUS: "function":"zero"
// STATUS-SAME: "status":"NO_GEMM"
//
// The records pass the gate a backend runs (GemmOpt/NestRecordGate.h): the affine GEMMs agree
// with the affine recognizer field by field, the scf.for GEMM is usable as a record by itself,
// and the loop with no GEMM is refused.
// RUN: %gemm-opt %S/Inputs/gemms.mlir --check-nest-records=%t.json | FileCheck %s --check-prefix=GATE
// GATE: // gemm-opt: records gate on {{.*}} at a 64-bit index
// GATE-NEXT: nest 0 of single: affine GEMM: record agrees: nest 0 of single (binding {{[0-9a-f]+}}, index width 64): VERIFIED
// GATE-NEXT: nest 0 of strict: affine GEMM: record agrees:
// GATE-NEXT: nest 0 of scf_gemm: record usable
// GATE-NEXT: nest 0 of zero: record refused: the record's status is NO_GEMM: the nest has no multiply-accumulate site
// GATE-NEXT: // 3 usable, 1 refused
//
// A record is about one version of one nest: change a loop bound and the same file is refused
// for the nests that changed, and only for them.
// RUN: sed -e 's/affine.for %%i = 0 to 16/affine.for %%i = 0 to 15/' %S/Inputs/gemms.mlir > %t.changed.mlir
// RUN: %gemm-opt %t.changed.mlir --check-nest-records=%t.json | FileCheck %s --check-prefix=STALE
// STALE: nest 0 of single: affine GEMM: record refused: the record's binding is not this nest's
// STALE-NEXT: nest 0 of strict: affine GEMM: record refused: the record's binding is not this nest's
// STALE-NEXT: nest 0 of scf_gemm: record usable
// STALE: // 1 usable, 3 refused
//
// Claims made at another index width are not claims about this program.
// RUN: %gemm-opt %S/Inputs/gemms.mlir --function=single --index-bit-width=16 "--authorize=roots 0 and 2 are separate,roots 1 and 2 are separate" --max-level=1 --candidates-per-group=1 --timeout-ms=120000 --rlimit=20000000 -o %t.16.json
// RUN: %gemm-opt %S/Inputs/gemms.mlir --function=single --check-nest-records=%t.16.json | FileCheck %s --check-prefix=WIDTH
// WIDTH: nest 0 of single: affine GEMM: record refused: {{.*}}16{{.*}}64
