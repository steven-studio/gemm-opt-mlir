# The binding of a loop nest (GemmOpt/NestBinding.h): the canonical text of a nest and its
# SHA-256. A record carries it, and a reader of records (gemm-opt --check-nest-records, or a
# backend such as systolic-mlir) uses a record only for the nest whose binding it computes
# itself. Keep this file identical to the backend's copy.
add_library(gemm-opt-nest-binding STATIC GemmOpt/NestBinding.cpp)
target_include_directories(gemm-opt-nest-binding PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(gemm-opt-nest-binding SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(gemm-opt-nest-binding PUBLIC
  MLIRArithDialect MLIRMemRefDialect MLIRFuncDialect MLIRIR MLIRSupport)

# The affine GEMM recognizer and the gate a record passes before a backend may use it.
add_library(gemm-opt-core STATIC
  GemmOpt/AffineMatmulAnalysis.cpp
  GemmOpt/AffineMatmulPartitioning.cpp
  GemmOpt/IntegerLattice.cpp
  GemmOpt/NestRecordGate.cpp)
target_link_libraries(gemm-opt-core PUBLIC
  gemm-opt-nest-binding
  MLIRAffineDialect MLIRAffineAnalysis MLIRAffineUtils MLIRPresburger
  MLIRArithDialect MLIRMemRefDialect MLIRSCFDialect MLIRFuncDialect
  MLIRViewLikeInterface MLIRIR MLIRSupport)

foreach(target gemm-opt-nest-binding gemm-opt-core)
  if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(${target} PUBLIC -fno-rtti)
  endif()
endforeach()
