cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# gemm-opt-mlir: Generalized GEMM recognition and middle-end analysis for MLIR.
#
# It recognizes multiply-accumulate loop nests (affine.for, scf.for, scf.while),
# proves what it claims about them, and writes one verified record per loop
# nest (`gemm-opt --nest-records`). A backend, such as systolic-mlir, reads
# those records and does the lowering; nothing here knows any hardware.
#
#   cmake -G Ninja -S . -B build \
#     -DLLVM_DIR=$LLVM_BUILD/lib/cmake/llvm \
#     -DMLIR_DIR=$LLVM_BUILD/lib/cmake/mlir \
#     -DCMAKE_BUILD_TYPE=Release
#   ninja -C build check-gemm-opt
# ---------------------------------------------------------------------------

project(gemm-opt-mlir LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
enable_testing()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR} (LLVM ${LLVM_PACKAGE_VERSION})")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# The shared libraries: the nest binding, the affine GEMM recognizer and the
# record gate (lib/GemmOpt).
add_subdirectory(lib)

# The research analyses and their drivers, unit tests, oracles and lit suites.
# Each of these directories is also a standalone CMake project; added from
# here, every library is defined once (by the first project that needs it).
# Their drivers stay in their own binary directories, where their lit suites
# look for them.
add_subdirectory(research/polynomial-summation-mlir)
add_subdirectory(research/polynomial-termination)
add_subdirectory(research/generalized-gemm)

# gemm-opt, the command-line tool (build/bin/gemm-opt).
add_subdirectory(tools)
# The lit suites: gemm-opt's own and the research ones (check-gemm-opt).
add_subdirectory(test)
