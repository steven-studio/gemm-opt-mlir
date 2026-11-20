# The lit suites of gemm-opt-mlir.
#
#   test/gemm-opt                 the gemm-opt tool (this directory's lit.cfg.py)
#   research/generalized-gemm     the Generalized GEMM analyses' drivers
#   research/polynomial-*         the polynomial summation and termination analyses
#
# check-gemm-opt runs all of them in one lit invocation.
find_program(GEMM_OPT_LIT NAMES llvm-lit lit HINTS "${LLVM_TOOLS_BINARY_DIR}" REQUIRED)
find_program(GEMM_OPT_FILECHECK FileCheck HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
find_program(GEMM_OPT_NOT not HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

set(GEMM_OPT_LIT_SUITES
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${PROJECT_BINARY_DIR}/research/generalized-gemm/lit"
  "${PROJECT_BINARY_DIR}/research/polynomial-summation-mlir/lit"
  "${PROJECT_BINARY_DIR}/research/polynomial-summation-mlir/p2a-lit"
  "${PROJECT_BINARY_DIR}/research/polynomial-termination/lit")

set(GEMM_OPT_LIT_ARGS "-sv" CACHE STRING "Arguments of lit in check-gemm-opt")
separate_arguments(GEMM_OPT_LIT_ARGS_LIST UNIX_COMMAND "${GEMM_OPT_LIT_ARGS}")

add_custom_target(check-gemm-opt
  COMMAND "${GEMM_OPT_LIT}" ${GEMM_OPT_LIT_ARGS_LIST} ${GEMM_OPT_LIT_SUITES}
  DEPENDS gemm-opt
    generalized-gemm-report generalized-gemm-termination-report generalized-gemm-coordinate-report
    generalized-gemm-synthesis-report generalized-gemm-computation-report
    generalized-gemm-domain-report generalized-gemm-nest-report
    polynomial-cfg-report polynomial-termination-report
  USES_TERMINAL
  COMMENT "Running the gemm-opt-mlir lit suites")
