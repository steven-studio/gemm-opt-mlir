//===- gemm-opt.cpp - Generalized GEMM recognition, one record per nest ---===//
//
// The command-line tool of gemm-opt-mlir. It reads an MLIR module (func, affine, scf,
// arith, memref, cf), recognizes the multiply-accumulate loop nests in it, and writes what
// it proved of each nest. Nothing in the input is changed, and nothing is lowered: a
// backend (for example systolic-mlir's `--affine-matmul-to-systolic="nest-records=FILE"`)
// reads the records and decides.
//
// Modes (one of them; --nest-records is the default):
//
//   --nest-records            The interchange file: a JSON array of function objects of
//                             the schema generalized-gemm-nest/2, one function per line.
//                             This is the file a backend's records gate reads.
//   --nest-report             The same records, indented for reading.
//   --outline-only            Each nest's private copy, or why it has none; no analysis.
//   --check-nest-records=FILE The gate a backend runs (GemmOpt/NestRecordGate.h) on FILE
//                             against this input: for each outermost loop, whether the
//                             record would be used, or the first check that fails.
//
// The index width of the claims is --index-bit-width, or else the module's data layout
// (64 bits without one), which is the width a backend compares a record with.
//
//===----------------------------------------------------------------------===//
#include "GemmOpt/AffineMatmulAnalysis.h"
#include "GemmOpt/NestBinding.h"
#include "GemmOpt/NestRecordGate.h"
#include "GeneralizedGemm/NestRecord.h"
#include "GeneralizedGemm/Z3Backend.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::generalized_gemm;
namespace cl = llvm::cl;

static cl::OptionCategory modeCategory("Modes");
static cl::OptionCategory analysisCategory("Analysis limits");

static cl::opt<std::string> input(cl::Positional, cl::init("-"), cl::desc("<input.mlir>"));
static cl::opt<std::string> outputFile("o", cl::init("-"), cl::desc("Output file (default: stdout)"),
                                       cl::value_desc("file"));

static cl::opt<bool> nestRecords("nest-records", cl::init(false), cl::cat(modeCategory),
                                 cl::desc("Write the records file a backend reads: a JSON array, one "
                                          "function object per line (the default mode)"));
static cl::opt<bool> nestReport("nest-report", cl::init(false), cl::cat(modeCategory),
                                cl::desc("Write the same records indented, for reading"));
static cl::opt<bool> outlineOnly("outline-only", cl::init(false), cl::cat(modeCategory),
                                 cl::desc("Print each nest's copy, or why it has none, and analyse nothing"));
static cl::opt<std::string> checkRecords(
    "check-nest-records", cl::init(""), cl::cat(modeCategory), cl::value_desc("file"),
    cl::desc("Run the backend's records gate on FILE against the input, nest by nest"));

static cl::opt<std::string> onlyFunction("function", cl::init(""), cl::desc("Analyse this function only"));
static cl::opt<unsigned> indexWidth(
    "index-bit-width", cl::init(0),
    cl::desc("Index width of the claims (default: the module's data layout, 64 without one)"));
static cl::opt<std::string> scopeText(
    "scope", cl::init(""), cl::desc("argument:lower:upper,... over the function's arguments, as signed decimals"));
static cl::opt<std::string> authorize(
    "authorize", cl::init(""), cl::desc("Comma separated separation names over the function's roots, e.g. "
                                        "\"roots 0 and 2 are separate\""));
static cl::opt<bool> indexInvariants(
    "index-invariants", cl::init(false),
    cl::desc("Search for inductive invariants over the integer state at each nest's loop headers (a "
             "subscript carried by iter_args)"));
static cl::opt<unsigned> timeoutMs("timeout-ms", cl::init(10000), cl::cat(analysisCategory));
static cl::opt<unsigned> resourceLimit("rlimit", cl::init(20000000), cl::cat(analysisCategory));
static cl::opt<uint64_t> maxPoints("max-points", cl::init(256), cl::cat(analysisCategory));
static cl::opt<unsigned> maxLevel("max-level", cl::init(3), cl::cat(analysisCategory));
static cl::opt<unsigned> maxCandidates("max-candidates", cl::init(5), cl::cat(analysisCategory));
static cl::opt<unsigned> maxRounds("max-rounds", cl::init(300), cl::cat(analysisCategory));
static cl::opt<uint64_t> referenceFill("reference-fill", cl::init(0), cl::cat(analysisCategory));
static cl::opt<uint64_t> maxBox("max-box", cl::init(1u << 20), cl::cat(analysisCategory));
static cl::opt<unsigned> candidatesPerGroup("candidates-per-group", cl::init(2), cl::cat(analysisCategory));
static cl::opt<uint64_t> replaySteps("replay-steps", cl::init(1u << 22), cl::cat(analysisCategory));
static cl::opt<uint64_t> replayNodes("replay-nodes", cl::init(1u << 23), cl::cat(analysisCategory));
static cl::opt<unsigned> maxQueries("max-queries", cl::init(4000), cl::cat(analysisCategory));

/// The index width the claims are made at: the option, or the module's data layout.
static unsigned claimsIndexWidth(ModuleOp module) {
  if (indexWidth != 0)
    return indexWidth;
  return DataLayout(module).getTypeSizeInBits(IndexType::get(module.getContext())).getFixedValue();
}

/// --check-nest-records: what a backend's gate says of each outermost loop. A loop in which
/// the affine recognizer finds a GEMM is checked as the dense lowering checks it (the record
/// by itself, then field by field against the recognizer); any other loop as a record-driven
/// lowering checks it (the record by itself, views without a closed form allowed).
static int checkNestRecords(ModuleOp module, llvm::raw_ostream &os) {
  std::string error;
  FailureOr<gemm_opt::NestRecordFile> records = gemm_opt::NestRecordFile::load(checkRecords, error);
  if (failed(records)) {
    llvm::errs() << "gemm-opt: --check-nest-records: " << error << "\n";
    return 2;
  }
  const unsigned width = claimsIndexWidth(module);
  const gemm_opt::BoundNests bound(module);

  SmallVector<gemm_opt::AffineMatmulCandidate> candidates;
  gemm_opt::collectAffineMatmulCandidates(module.getOperation(), candidates);
  SmallVector<gemm_opt::LogicalMatmul, 1> matmuls;
  for (gemm_opt::AffineMatmulCandidate candidate : candidates) {
    gemm_opt::AffineMatmulRecognitionOptions recognition;
    recognition.countersAsWritten = true;
    FailureOr<gemm_opt::LogicalMatmul> result = gemm_opt::recognizeLogicalMatmul(candidate, nullptr, nullptr,
                                                                                   recognition);
    if (failed(result) || llvm::any_of(matmuls, [&](const gemm_opt::LogicalMatmul &m) {
          return m.outputStore == result->outputStore;
        }))
      continue;
    matmuls.push_back(*result);
  }

  os << "// gemm-opt: records gate on " << checkRecords << " at a " << width << "-bit index\n";
  unsigned usable = 0, refused = 0;
  for (auto function : module.getOps<func::FuncOp>()) {
    if (function.isExternal() || (!onlyFunction.empty() && function.getSymName() != onlyFunction))
      continue;
    for (Operation *loop : gemm_opt::outermostLoops(function)) {
      const gemm_opt::BoundNests::Nest *nest = bound.find(loop);
      std::string where = nest ? llvm::formatv("nest {0} of {1}", nest->ordinal, function.getSymName()).str()
                               : std::string("a loop of ") + function.getSymName().str();
      if (!nest) {
        os << where << ": no binding\n";
        ++refused;
        continue;
      }
      // The gate's reasons name the nest themselves.
      auto reason = [&](llvm::StringRef why) {
        why.consume_front(where + ": ");
        return why.str();
      };
      bool sawGemm = false;
      for (const gemm_opt::LogicalMatmul &matmul : matmuls) {
        if (matmul.anchor != loop && !loop->isAncestor(matmul.anchor))
          continue;
        sawGemm = true;
        gemm_opt::NestRecordDecision decision = gemm_opt::checkNestRecord(*records, bound, matmul, width);
        if (decision.agrees) {
          os << where << ": affine GEMM: record agrees: " << decision.summary << "\n";
          ++usable;
        } else {
          os << where << ": affine GEMM: record refused: " << reason(decision.why) << "\n";
          ++refused;
        }
      }
      if (sawGemm)
        continue;
      gemm_opt::RecordAtNest at = gemm_opt::checkRecordAtNest(*records, bound, loop, width,
                                                              /*unmappedViews=*/true);
      if (at.usable) {
        os << where << ": record usable" << (at.viewsUnmapped ? " (views by tables)" : "") << "\n";
        ++usable;
      } else {
        os << where << ": record refused: " << reason(at.why) << "\n";
        ++refused;
      }
    }
  }
  os << "// " << usable << " usable, " << refused << " refused\n";
  return 0;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv,
                              "gemm-opt: Generalized GEMM recognition, one verified record per loop nest\n");
  if (unsigned(nestRecords) + unsigned(nestReport) + unsigned(outlineOnly) + unsigned(!checkRecords.empty()) >
      1) {
    llvm::errs() << "gemm-opt: give one of --nest-records, --nest-report, --outline-only and "
                    "--check-nest-records\n";
    return 2;
  }

  DialectRegistry registry;
  registry.insert<affine::AffineDialect, arith::ArithDialect, cf::ControlFlowDialect, DLTIDialect,
                  func::FuncDialect, LLVM::LLVMDialect, memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(input, &context);
  if (!module)
    return 2;

  std::error_code error;
  auto output = std::make_unique<llvm::ToolOutputFile>(outputFile, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "gemm-opt: " << outputFile << ": " << error.message() << "\n";
    return 2;
  }
  llvm::raw_ostream &os = output->os();

  if (!checkRecords.empty()) {
    int status = checkNestRecords(*module, os);
    if (status == 0)
      output->keep();
    return status;
  }

  NestAnalysisOptions options;
  options.indexBitWidth = claimsIndexWidth(*module);
  options.timeoutMs = timeoutMs;
  options.resourceLimit = resourceLimit;
  options.maxPoints = maxPoints;
  options.maxBox = maxBox;
  options.replaySteps = replaySteps;
  options.replayNodes = replayNodes;
  options.referenceFill = referenceFill;
  options.maxLevel = maxLevel;
  options.maxCandidates = maxCandidates;
  options.maxRounds = maxRounds;
  options.candidatesPerGroup = candidatesPerGroup;
  options.maxQueries = maxQueries;
  options.indexInvariants = indexInvariants;
  {
    llvm::SmallVector<llvm::StringRef> items;
    llvm::StringRef(scopeText).split(items, ',', -1, /*KeepEmpty=*/false);
    for (llvm::StringRef item : items) {
      llvm::SmallVector<llvm::StringRef> parts;
      item.split(parts, ':');
      NestScopeEntry entry;
      if (parts.size() != 3 || parts[0].getAsInteger(10, entry.argument) ||
          parts[1].getAsInteger(10, entry.lower) || parts[2].getAsInteger(10, entry.upper)) {
        llvm::errs() << "gemm-opt: malformed --scope\n";
        return 2;
      }
      options.scope.push_back(entry);
    }
  }
  Authorizations authorized;
  {
    llvm::SmallVector<llvm::StringRef> names;
    llvm::StringRef(authorize).split(names, ',', -1, /*KeepEmpty=*/false);
    for (llvm::StringRef name : names)
      authorized.insert(name.trim().str());
  }

  Z3Backend backend;
  std::vector<llvm::json::Value> functions;
  for (auto function : module->getOps<func::FuncOp>()) {
    if (function.isExternal())
      continue;
    if (!onlyFunction.empty() && function.getSymName() != onlyFunction)
      continue;
    FunctionOutline outline = outlineNests(function, authorized);
    if (outlineOnly) {
      os << "// function " << outline.function;
      if (!outline.unsupported.empty())
        os << ": not outlined: " << outline.unsupported;
      os << "\n";
      for (const OutlinedNest &nest : outline.nests) {
        os << "// nest " << nest.ordinal << " (" << nest.loop << ", statement " << nest.statement << ")";
        if (!nest.unsupported.empty()) {
          os << ": no copy: " << nest.unsupported << "\n";
          continue;
        }
        os << "\n";
        nest.module.get().print(os);
      }
      continue;
    }
    std::vector<NestRecord> records;
    for (const OutlinedNest &nest : outline.nests)
      records.push_back(analyzeNest(nest, options, backend));
    functions.push_back(toJSON(outline, records));
  }

  if (nestReport) {
    llvm::json::Array array;
    for (llvm::json::Value &f : functions)
      array.push_back(std::move(f));
    os << llvm::formatv("{0:2}", llvm::json::Value(std::move(array))) << "\n";
  } else if (!outlineOnly) {
    // The interchange form: one function object per line, keys sorted, no spaces.
    os << "[\n";
    for (size_t i = 0; i < functions.size(); ++i)
      os << functions[i] << (i + 1 < functions.size() ? ",\n" : "\n");
    os << "]\n";
  }
  output->keep();
  return 0;
}
