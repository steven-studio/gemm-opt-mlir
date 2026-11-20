# -*- Python -*-
# The lit suite of the gemm-opt tool. The research suites have their own configurations
# (research/*/lit*); check-gemm-opt runs them all.
import os
import lit.formats

config.name = "gemm-opt"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir", ".test"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.gemm_opt_obj_root + "/test"
# Sources and inputs of the research builds, not lit tests of this suite.
config.excludes = ["Inputs", "CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py.in",
                   "polynomial-summation", "polynomial-summation-mlir",
                   "polynomial-summation-p2a", "polynomial-termination"]

config.substitutions.append(("%gemm-opt", os.path.join(config.gemm_opt_tools_dir, "gemm-opt")))
config.substitutions.append(("%nestreport", os.path.join(config.research_dir, "generalized-gemm-nest-report")))
config.substitutions.append(("%obj_root", config.gemm_opt_obj_root))
config.substitutions.append(("%python", '"%s"' % config.python))
config.substitutions.append(("%src_root", config.gemm_opt_src_root))
config.substitutions.append(("FileCheck", config.filecheck))
config.substitutions.append((" not ", " %s " % config.not_tool))
