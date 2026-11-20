add_executable(gemm-opt gemm-opt.cpp)
target_link_libraries(gemm-opt PRIVATE
  generalized-gemm-nest generalized-gemm-z3 gemm-opt-core
  MLIRAffineDialect MLIRArithDialect MLIRControlFlowDialect MLIRFuncDialect
  MLIRLLVMDialect MLIRMemRefDialect MLIRSCFDialect MLIRParser MLIRDLTIDialect)
target_compile_options(gemm-opt PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(gemm-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
