# gemm-opt-mlir

Generalized GEMM recognition and middle-end analysis for MLIR.

`gemm-opt` reads an MLIR module (`func`, `affine`, `scf`, `arith`, `memref`, `cf`), finds the
multiply-accumulate loop nests in it — written with `affine.for`, `scf.for` or counting
`scf.while` loops, with subscripts that may be carried by `iter_args` — and proves what it
claims about each of them: termination, the coordinates I, J and K, the computation, the domain
and the order of the reduction. It writes **one verified record per loop nest**. It changes
nothing in the input and lowers nothing: a backend reads the records and decides.

The first backend is systolic-mlir, whose
`--affine-matmul-to-systolic="nest-records=FILE"` reads the file `gemm-opt` writes.

## Layout

```
include/GeneralizedGemm/   lib/GeneralizedGemm/   the Generalized GEMM analyses (checkers, proposers,
                                                  semantic model, nest outliner and records)
include/GemmOpt/           lib/GemmOpt/           shared with a backend: the nest binding, the affine
                                                  GEMM recognizer, the records gate
                           lib/GemmOpt/Analysis/  polynomial summation and termination analyses
tools/gemm-opt/                                   the command-line tool
research/generalized-gemm/                        the analyses' drivers, unit tests, oracles, lit suite
research/polynomial-*/                            the polynomial analyses' drivers and lit suites
test/gemm-opt/                                    gemm-opt's lit suite
test/polynomial-*/                                inputs and unit tests of the polynomial analyses
docs/                                             design notes (GENERALIZED_GEMM_*, POLYNOMIAL_*)
```

## Build

Requirements: an LLVM/MLIR build (developed against LLVM 24.0.0git, revision 7615da10b75c), Z3
(headers and library), Boost headers, Python 3, CMake ≥ 3.20, Ninja. The LLVM build must provide
`FileCheck`, `not`, `count` and `llvm-lit` in its `bin/`. Some CTest oracles also need a clang of
the same LLVM version (`-DGG_CLANG=...`).

```sh
LLVM_BUILD=$HOME/work/llvm-project/build
cmake -G Ninja -S . -B build \
  -DLLVM_DIR=$LLVM_BUILD/lib/cmake/llvm \
  -DMLIR_DIR=$LLVM_BUILD/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
ninja -C build                    # build/bin/gemm-opt and the research drivers
ninja -C build check-gemm-opt     # every lit suite
ctest --test-dir build -R '^generalized-gemm-unit$'   # unit tests / oracles (some take an hour)
```

## Usage

```sh
# The records file a backend reads (the default mode): a JSON array, one function per line.
gemm-opt input.mlir --authorize="roots 0 and 2 are separate,roots 1 and 2 are separate" \
         -o input.records.json

# The same records, indented for reading.
gemm-opt input.mlir --nest-report

# What the backend's records gate will say of each outermost loop.
gemm-opt input.mlir --check-nest-records=input.records.json

# Handing the records to systolic-mlir:
systolic-opt input.mlir --systolic-declare-fpga-fleet \
  --affine-matmul-to-systolic="nest-records=input.records.json record-driven=true"
```

The claims are made at `--index-bit-width`, or else at the module's data-layout index width
(64 bits without one) — the width a backend compares a record with. `--authorize` names the
pairs of roots (function arguments and allocations, as the record numbers them) that the caller
guarantees do not alias. Limits such as `--rlimit`, `--timeout-ms`, `--max-level` and
`--candidates-per-group` are listed by `gemm-opt --help`; the solver's resource limit, not its
time limit, decides each query, so the records are the same on a loaded machine.

## The interface with a backend

* Schema `generalized-gemm-nest/2` (records file) and `generalized-gemm-nest-binding/1`
  (binding of a nest). A record is used only for the nest whose binding (canonical text and
  SHA-256, `GemmOpt/NestBinding.h`) the reader computes itself.
* `lib/GemmOpt/NestBinding.cpp` and `lib/GemmOpt/NestRecordGate.cpp` must stay identical, apart
  from the namespace, to the backend's copies; `gemm-opt --check-nest-records` runs this
  project's copy of the gate.
* No record authorizes a lowering (`"authorizes_lowering": false`): the backend still runs every
  check of its own.

## Origin

Split from systolic-mlir (`feat/generalized-gemm-recognition`, commit `d21e842`) on 2026-11-20.
References in comments and docs to `test/Systolic/...`, `lib/Systolic/...`, `systolic-opt` and
`docs/AFFINE_GEMM_MAPPING.md` are to files of that repository.
