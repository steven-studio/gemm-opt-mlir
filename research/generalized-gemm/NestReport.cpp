//===- NestReport.cpp - One record per loop nest -------------------------===//
//
// Stage 1 of docs/GENERALIZED_GEMM_SCOPE.md. Prints a JSON array with one object per
// function: its roots, the order of its statements, how its nests meet on memory, and
// one record per outermost loop. Each nest is analysed on a private copy, by the
// importer and the checkers the domain driver uses, from any memory at its entry.
// Read-only: the input is not changed, and no record authorizes a lowering.
//
// Names given to --authorize and arguments given to --scope are the function's: roots
// as the importer numbers them in the whole function, and argument numbers.
//
//===----------------------------------------------------------------------===//
#include "GeneralizedGemm/NestRecord.h"
#include "GeneralizedGemm/Z3Backend.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::generalized_gemm;
namespace cl = llvm::cl;

namespace {

struct TableSpec {
  unsigned argument = 0;
  int64_t lower = 0, upper = 0;
};

/// What --table did to one function: the assumption text per table, or why it refused.
struct TableRewrite {
  std::string refusal;
  std::vector<std::string> assumptions;
  std::vector<NestScopeEntry> scope;
};

/// Reads each table argument's cells as appended scalar arguments (see --table). The
/// rewrite is on the parsed copy only. Refuses, changing nothing, where an argument is
/// not a memref of static shape with an integer or index element, or has a use other
/// than as the memref of a memref.load.
TableRewrite readTablesAsParameters(func::FuncOp function, llvm::ArrayRef<TableSpec> tables) {
  TableRewrite out;
  unsigned original = function.getNumArguments();
  for (const TableSpec &table : tables) {
    if (table.argument >= original) {
      out.refusal = "--table names argument " + std::to_string(table.argument) + ", which does not exist";
      return out;
    }
    auto type = dyn_cast<MemRefType>(function.getArgument(table.argument).getType());
    if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity() ||
        !(type.getElementType().isIndex() || type.getElementType().isSignlessInteger())) {
      out.refusal = "argument " + std::to_string(table.argument) +
                    " is no memref of static shape and identity layout with an index or integer element";
      return out;
    }
    for (Operation *user : function.getArgument(table.argument).getUsers()) {
      auto load = dyn_cast<memref::LoadOp>(user);
      if (!load || load.getMemRef() != function.getArgument(table.argument)) {
        out.refusal = "argument " + std::to_string(table.argument) + " has a use that is no memref.load: " +
                      user->getName().getStringRef().str();
        return out;
      }
    }
  }
  for (const TableSpec &table : tables) {
    Value memref = function.getArgument(table.argument);
    auto type = cast<MemRefType>(memref.getType());
    Type element = type.getElementType();
    int64_t cells = type.getNumElements();
    unsigned first = function.getNumArguments();
    llvm::SmallVector<Value> cellArgs;
    for (int64_t j = 0; j < cells; ++j) {
      if (failed(function.insertArgument(function.getNumArguments(), element, DictionaryAttr(), function.getLoc())))
        llvm::report_fatal_error("insertArgument failed");
      cellArgs.push_back(function.getArgument(function.getNumArguments() - 1));
    }
    llvm::SmallVector<memref::LoadOp> loads;
    for (Operation *user : memref.getUsers())
      loads.push_back(cast<memref::LoadOp>(user));
    llvm::ArrayRef<int64_t> shape = type.getShape();
    for (memref::LoadOp load : loads) {
      OpBuilder b(load->getContext());
      b.setInsertionPointAfter(load);
      Location loc = load.getLoc();
      // The linear index, row-major.
      Value linear = b.create<arith::ConstantIndexOp>(loc, 0);
      for (unsigned d = 0; d < shape.size(); ++d) {
        Value extent = b.create<arith::ConstantIndexOp>(loc, shape[d]);
        linear = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, linear, extent), load.getIndices()[d]);
      }
      Value value = cellArgs.back();
      for (int64_t j = cells - 1; j-- > 0;) {
        Value at = b.create<arith::ConstantIndexOp>(loc, j);
        Value is = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, linear, at);
        value = b.create<arith::SelectOp>(loc, is, cellArgs[j], value);
      }
      // Every use of the loaded value reads the select; the load stays, unused, so that
      // an index outside the table stops the run as before.
      load.getResult().replaceAllUsesWith(value);
    }
    for (int64_t j = 0; j < cells; ++j)
      out.scope.push_back({first + (unsigned)j, table.lower, table.upper});
    out.assumptions.push_back(
        "argument " + std::to_string(table.argument) + " is an index table of " + std::to_string(cells) +
        " cells read as the arguments " + std::to_string(first) + " to " + std::to_string(first + cells - 1) +
        " (row-major), each in [" + std::to_string(table.lower) + ", " + std::to_string(table.upper) +
        "] (--table): the claims are about the nest with each load of it reading that argument; the loads "
        "are kept, so the checkers need the table apart from every root the nest writes, which is what "
        "makes its cells one value through the nest");
  }
  return out;
}

} // namespace

static cl::opt<std::string> input(cl::Positional, cl::init("-"));
static cl::opt<std::string> onlyFunction("function", cl::init(""));
static cl::opt<unsigned> indexWidth("index-bit-width", cl::init(0));
static cl::opt<std::string> scopeText(
    "scope", cl::init(""), cl::desc("argument:lower:upper,... over the function's arguments, as signed decimals"));
static cl::opt<std::string> authorize(
    "authorize", cl::init(""), cl::desc("comma separated separation names over the function's roots"));
static cl::opt<unsigned> timeoutMs("timeout-ms", cl::init(10000));
static cl::opt<unsigned> resourceLimit("rlimit", cl::init(20000000));
static cl::opt<uint64_t> maxPoints("max-points", cl::init(256));
static cl::opt<unsigned> maxLevel("max-level", cl::init(3));
static cl::opt<unsigned> maxCandidates("max-candidates", cl::init(5));
static cl::opt<unsigned> maxRounds("max-rounds", cl::init(300));
static cl::opt<uint64_t> referenceFill("reference-fill", cl::init(0));
static cl::opt<uint64_t> maxBox("max-box", cl::init(1u << 20));
static cl::opt<unsigned> candidatesPerGroup("candidates-per-group", cl::init(2));
static cl::opt<uint64_t> replaySteps("replay-steps", cl::init(1u << 22));
static cl::opt<uint64_t> replayNodes("replay-nodes", cl::init(1u << 23));
static cl::opt<unsigned> maxQueries("max-queries", cl::init(4000));
static cl::opt<std::string> tableText(
    "table", cl::init(""),
    cl::desc("argument:lower:upper,... : a memref argument with a static shape read only by "
             "memref.load (an index table: per-group row counts, gather ids, CSR row pointers). "
             "Each cell becomes a scalar argument appended to the function, in row-major order, "
             "with the scope [lower, upper]; each load reads its cell by a select over the "
             "linearized index, and the load itself is kept for its bounds and its root: the "
             "checkers then need the table's root apart from every root the nest writes"));
static cl::opt<bool> indexInvariants(
    "index-invariants", cl::init(false),
    cl::desc("search for inductive invariants over the integer state at each nest's loop headers and hand "
             "what is proved to the analysis (a subscript carried by iter_args)"));
static cl::opt<bool> outlineOnly(
    "outline-only", cl::init(false),
    cl::desc("print each nest's copy, or why it has none, and analyse nothing"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "Stage 1: one record per loop nest\n");
  DialectRegistry registry;
  registry.insert<affine::AffineDialect, arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
                  LLVM::LLVMDialect, memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  context.loadAllAvailableDialects();
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(input, &context);
  if (!module)
    return 2;

  NestAnalysisOptions options;
  options.indexBitWidth = indexWidth;
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
        llvm::errs() << "malformed --scope\n";
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
      authorized.insert(name.str());
  }

  std::vector<TableSpec> tables;
  {
    llvm::SmallVector<llvm::StringRef> items;
    llvm::StringRef(tableText).split(items, ',', -1, /*KeepEmpty=*/false);
    for (llvm::StringRef item : items) {
      llvm::SmallVector<llvm::StringRef> parts;
      item.split(parts, ':');
      TableSpec spec;
      if (parts.size() != 3 || parts[0].getAsInteger(10, spec.argument) ||
          parts[1].getAsInteger(10, spec.lower) || parts[2].getAsInteger(10, spec.upper) ||
          spec.lower > spec.upper) {
        llvm::errs() << "malformed --table\n";
        return 2;
      }
      tables.push_back(spec);
    }
  }

  Z3Backend backend;
  llvm::json::Array functions;
  for (auto function : module->getOps<func::FuncOp>()) {
    if (function.isExternal())
      continue;
    if (!onlyFunction.empty() && function.getSymName() != onlyFunction)
      continue;
    NestAnalysisOptions functionOptions = options;
    std::vector<std::string> tableAssumptions;
    if (!tables.empty()) {
      TableRewrite rewrite = readTablesAsParameters(function, tables);
      if (!rewrite.refusal.empty()) {
        llvm::errs() << "--table refused for " << function.getSymName() << ": " << rewrite.refusal << "\n";
        return 2;
      }
      functionOptions.scope.insert(functionOptions.scope.end(), rewrite.scope.begin(), rewrite.scope.end());
      tableAssumptions = rewrite.assumptions;
    }
    FunctionOutline outline = outlineNests(function, authorized);
    if (outlineOnly) {
      llvm::outs() << "// function " << outline.function;
      if (!outline.unsupported.empty())
        llvm::outs() << ": not outlined: " << outline.unsupported;
      llvm::outs() << "\n";
      for (const OutlinedNest &nest : outline.nests) {
        llvm::outs() << "// nest " << nest.ordinal << " (" << nest.loop << ", statement " << nest.statement << ")";
        if (!nest.unsupported.empty()) {
          llvm::outs() << ": no copy: " << nest.unsupported << "\n";
          continue;
        }
        llvm::outs() << "\n";
        nest.module.get().print(llvm::outs());
      }
      continue;
    }
    std::vector<NestRecord> records;
    for (const OutlinedNest &nest : outline.nests) {
      records.push_back(analyzeNest(nest, functionOptions, backend));
      NestRecord &record = records.back();
      record.assumptions.insert(record.assumptions.begin(), tableAssumptions.begin(), tableAssumptions.end());
    }
    functions.push_back(toJSON(outline, records));
  }
  if (!outlineOnly)
    llvm::outs() << llvm::formatv("{0:2}", llvm::json::Value(std::move(functions))) << "\n";
  return 0;
}
