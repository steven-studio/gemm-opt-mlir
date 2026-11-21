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
  bool distinct = false; // Its cells pairwise distinct: a premise (the table is injective).
};

/// What --table did to one function: the assumption text per table, or why it refused.
struct TableRewrite {
  std::string refusal;
  std::vector<std::string> assumptions;
  std::vector<NestScopeEntry> scope;
  std::vector<std::vector<unsigned>> distinct; // The cell arguments of each injective table.
  struct Cells {
    unsigned table = 0, first = 0, count = 0;
  };
  std::vector<Cells> cells; // Per table: its first cell argument and its number of cells.
};

/// Reads each table argument's cells as appended scalar arguments (see --table). The
/// rewrite is on the parsed copy only. Refuses, changing nothing, where an argument is
/// not a memref of static shape with an integer or index element, or has a use other
/// than as the memref of a memref.load.
TableRewrite readTablesAsParameters(func::FuncOp function, llvm::ArrayRef<TableSpec> tables,
                                    llvm::StringRef origin = "--table") {
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
    out.cells.push_back({table.argument, first, (unsigned)cells});
    if (table.distinct) {
      std::vector<unsigned> group;
      for (int64_t j = 0; j < cells; ++j)
        group.push_back(first + (unsigned)j);
      out.distinct.push_back(std::move(group));
    }
    out.assumptions.push_back(
        "argument " + std::to_string(table.argument) + " is an index table of " + std::to_string(cells) +
        " cells read as the arguments " + std::to_string(first) + " to " + std::to_string(first + cells - 1) +
        " (row-major), each in [" + std::to_string(table.lower) + ", " + std::to_string(table.upper) +
        "] (" + origin.str() + "): the claims are about the nest with each load of it reading that argument; the loads "
        "are kept, so the checkers need the table apart from every root the nest writes, which is what "
        "makes its cells one value through the nest");
  }
  return out;
}

//===--- --auto: the premises read off the function, with no per-nest flag -------===//
//
// IR that comes down from a framework carries no --table, --scope or --authorize. With
// --auto, each is inferred, and each becomes a premise that the caller (the lowering) checks
// at run time before it relies on the record:
//   separate  every pair of memref arguments of which one is written: the roots apart;
//   table     a memref argument of static shape, integer element, read only by loads,
//             whose loaded values reach an index or a loop bound: its cells as parameters,
//             in [0, hi], hi the largest value that keeps every such index in bounds;
//   range     a scalar integer argument reaching an index or a loop bound: [0, hi], alike;
//   distinct  speculative, per nest: the cells of each table whose values reach the
//             address of a written memref are pairwise distinct (the table is injective).
//             Tried only when the nest's record without it does not prove that no two
//             logical outputs share a cell, and only when the table has no more cells than
//             values; the record with it is kept only if it proves that fact.
// The inference proposes; the checkers decide. A premise only narrows the inputs a record
// speaks of, so a wrong guess costs a refusal or a guard that fails, never a wrong claim.

struct Interval {
  int64_t lo = 0, hi = 0;
};

std::optional<Interval> valueRange(Value v, unsigned depth = 0) {
  if (depth > 24)
    return std::nullopt;
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto a = dyn_cast<IntegerAttr>(c.getValue()))
      return Interval{a.getInt(), a.getInt()};
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (auto loop = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp()))
      if (arg == loop.getInductionVar()) {
        auto lb = valueRange(loop.getLowerBound(), depth + 1), ub = valueRange(loop.getUpperBound(), depth + 1);
        if (lb && ub && ub->hi > lb->lo)
          return Interval{lb->lo, ub->hi - 1};
      }
    return std::nullopt;
  }
  Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp, arith::ExtUIOp>(op))
    return valueRange(op->getOperand(0), depth + 1);
  if (op->getNumOperands() != 2)
    return std::nullopt;
  auto a = valueRange(op->getOperand(0), depth + 1), b = valueRange(op->getOperand(1), depth + 1);
  if (!a || !b)
    return std::nullopt;
  if (isa<arith::AddIOp>(op))
    return Interval{a->lo + b->lo, a->hi + b->hi};
  if (isa<arith::SubIOp>(op))
    return Interval{a->lo - b->hi, a->hi - b->lo};
  if (isa<arith::MulIOp>(op)) {
    int64_t p[4] = {a->lo * b->lo, a->lo * b->hi, a->hi * b->lo, a->hi * b->hi};
    return Interval{*std::min_element(p, p + 4), *std::max_element(p, p + 4)};
  }
  if ((isa<arith::DivUIOp, arith::DivSIOp>(op)) && b->lo == b->hi && b->lo > 0 && a->lo >= 0)
    return Interval{a->lo / b->lo, a->hi / b->lo};
  if ((isa<arith::RemUIOp, arith::RemSIOp>(op)) && b->lo == b->hi && b->lo > 0 && a->lo >= 0)
    return Interval{0, std::min(a->hi, b->lo - 1)};
  return std::nullopt;
}

int64_t floorDiv(int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

/// The largest s with c * s + o inside every index it reaches, for every o in `o`: the
/// tightest bound over the uses that give one, or nullopt when none does.
std::optional<int64_t> upperFromUses(Value v, int64_t c, Interval o, unsigned depth = 0) {
  if (depth > 24 || c <= 0)
    return std::nullopt;
  std::optional<int64_t> best;
  auto take = [&](std::optional<int64_t> bound) {
    if (bound)
      best = best ? std::min(*best, *bound) : *bound;
  };
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    unsigned at = use.getOperandNumber();
    auto indexBound = [&](MemRefType type, unsigned firstIndex) -> std::optional<int64_t> {
      if (at < firstIndex)
        return std::nullopt;
      unsigned d = at - firstIndex;
      if (d >= type.getRank() || type.isDynamicDim(d))
        return std::nullopt;
      return floorDiv(type.getDimSize(d) - 1 - o.hi, c);
    };
    if (auto load = dyn_cast<memref::LoadOp>(user)) {
      take(indexBound(load.getMemRefType(), 1));
    } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
      take(indexBound(store.getMemRefType(), 2));
    } else if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp, arith::ExtUIOp>(user)) {
      take(upperFromUses(user->getResult(0), c, o, depth + 1));
    } else if (isa<arith::AddIOp>(user)) {
      if (auto other = valueRange(user->getOperand(1 - at)))
        take(upperFromUses(user->getResult(0), c, Interval{o.lo + other->lo, o.hi + other->hi}, depth + 1));
    } else if (isa<arith::SubIOp>(user) && at == 0) {
      if (auto other = valueRange(user->getOperand(1)))
        take(upperFromUses(user->getResult(0), c, Interval{o.lo - other->hi, o.hi - other->lo}, depth + 1));
    } else if (isa<arith::MulIOp>(user)) {
      if (auto k = valueRange(user->getOperand(1 - at)); k && k->lo == k->hi && k->lo > 0)
        take(upperFromUses(user->getResult(0), c * k->lo, Interval{o.lo * k->lo, o.hi * k->lo}, depth + 1));
    } else if (auto loop = dyn_cast<scf::ForOp>(user)) {
      // A bound of the loop: its counter stays below it and inside every index it reaches.
      if (at == 0 || at == 1)
        if (auto counter = upperFromUses(loop.getInductionVar(), 1, Interval{0, 0}, depth + 1))
          take(floorDiv(*counter + 1 - o.hi, c));
    }
  }
  return best;
}

/// Whether `v` reaches, through integer arithmetic, an index of an access to a memref the
/// function writes.
bool reachesWrittenAddress(Value v, const llvm::DenseSet<Value> &written, unsigned depth = 0) {
  if (depth > 24)
    return false;
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    if (auto load = dyn_cast<memref::LoadOp>(user)) {
      if (use.getOperandNumber() >= 1 && written.contains(load.getMemRef()))
        return true;
    } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
      if (use.getOperandNumber() >= 2)
        return true;
    } else if (isa<arith::ArithDialect>(user->getDialect()) && user->getNumResults() == 1) {
      if (reachesWrittenAddress(user->getResult(0), written, depth + 1))
        return true;
    }
  }
  return false;
}

struct AutoPremises {
  std::vector<TableSpec> tables;
  std::vector<unsigned> addressTables; // Tables whose values reach a written address.
  std::vector<NestScopeEntry> scope;   // Scalar arguments.
  std::vector<std::pair<unsigned, unsigned>> separate; // Function argument pairs.
  Authorizations authorized;
  std::vector<std::string> notes; // What was not inferred, and why.
};

AutoPremises inferPremises(func::FuncOp function) {
  AutoPremises out;
  llvm::DenseSet<Value> written;
  std::vector<unsigned> memrefs; // Function arguments of memref type, in order: the roots.
  for (BlockArgument arg : function.getArguments()) {
    if (!isa<MemRefType>(arg.getType()))
      continue;
    memrefs.push_back(arg.getArgNumber());
    for (Operation *user : arg.getUsers())
      if (auto store = dyn_cast<memref::StoreOp>(user); store && store.getMemRef() == arg)
        written.insert(arg);
  }
  for (size_t a = 0; a < memrefs.size(); ++a)
    for (size_t b = a + 1; b < memrefs.size(); ++b)
      if (written.contains(function.getArgument(memrefs[a])) || written.contains(function.getArgument(memrefs[b]))) {
        out.separate.push_back({memrefs[a], memrefs[b]});
        out.authorized.insert("roots " + std::to_string(a) + " and " + std::to_string(b) + " are separate");
      }
  for (BlockArgument arg : function.getArguments()) {
    unsigned n = arg.getArgNumber();
    if (auto type = dyn_cast<MemRefType>(arg.getType())) {
      if (written.contains(arg) || !type.hasStaticShape() || !type.getLayout().isIdentity() ||
          !(type.getElementType().isIndex() || type.getElementType().isSignlessInteger()))
        continue;
      bool onlyLoads = !arg.use_empty();
      std::optional<int64_t> hi;
      bool address = false;
      for (Operation *user : arg.getUsers()) {
        auto load = dyn_cast<memref::LoadOp>(user);
        if (!load || load.getMemRef() != arg) {
          onlyLoads = false;
          break;
        }
        if (auto bound = upperFromUses(load.getResult(), 1, Interval{0, 0}))
          hi = hi ? std::min(*hi, *bound) : *bound;
        address = address || reachesWrittenAddress(load.getResult(), written);
      }
      if (!onlyLoads)
        continue;
      if (!hi || *hi < 0) {
        out.notes.push_back("argument " + std::to_string(n) + ": an integer table whose values reach no bounded "
                            "index or loop bound; left as memory");
        continue;
      }
      out.tables.push_back(TableSpec{n, 0, *hi, false});
      if (address)
        out.addressTables.push_back(n);
    } else if (arg.getType().isIndex() || arg.getType().isSignlessInteger()) {
      if (auto hi = upperFromUses(arg, 1, Interval{0, 0}); hi && *hi >= 0)
        out.scope.push_back(NestScopeEntry{n, 0, *hi});
      else
        out.notes.push_back("argument " + std::to_string(n) + ": a scalar that reaches no bounded index or loop "
                            "bound; no range");
    }
  }
  return out;
}

llvm::json::Object recordSummary(const NestRecord &r) {
  return llvm::json::Object{{"status", stringifyNestStatus(r.status)},
                            {"ordered_fold_verified", r.orderedFoldVerified},
                            {"output_injective_proven", r.outputInjectiveProven},
                            {"index_invariants", r.indexInvariants.asked}};
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
static cl::opt<bool> distinctSearchOnly(
    "distinct-search-only", cl::init(false),
    cl::desc("a table given as argument:lower:upper:distinct steers only the coordinate search: the "
             "checkers judge over the scope without the premise that its cells are pairwise distinct"));
static cl::opt<bool> autoMode(
    "auto", cl::init(false),
    cl::desc("infer every premise from the function (root separations, index tables and their "
             "ranges, scalar ranges), retry with index invariants, speculate that tables in a written "
             "address are injective, and add to each nest an 'auto' object: the premises a caller must "
             "check at run time and the plan they allow. No other premise flag is needed"));
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
      if (parts.size() == 4 && parts[3] == "distinct") {
        spec.distinct = true;
        parts.pop_back();
      }
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
    Authorizations functionAuthorized = authorized;
    std::vector<TableSpec> functionTables = tables;
    AutoPremises premises;
    if (autoMode) {
      premises = inferPremises(function);
      functionTables.insert(functionTables.end(), premises.tables.begin(), premises.tables.end());
      functionAuthorized.insert(premises.authorized.begin(), premises.authorized.end());
      functionOptions.scope.insert(functionOptions.scope.end(), premises.scope.begin(), premises.scope.end());
    }
    TableRewrite rewrite;
    if (!functionTables.empty()) {
      rewrite = readTablesAsParameters(function, functionTables, autoMode ? "inferred by --auto" : "--table");
      if (!rewrite.refusal.empty()) {
        llvm::errs() << "--table refused for " << function.getSymName() << ": " << rewrite.refusal << "\n";
        return 2;
      }
      functionOptions.scope.insert(functionOptions.scope.end(), rewrite.scope.begin(), rewrite.scope.end());
      functionOptions.distinct = rewrite.distinct;
      functionOptions.countersFirst = true;
      functionOptions.distinctSearchOnly = distinctSearchOnly;
      tableAssumptions = rewrite.assumptions;
    }
    FunctionOutline outline = outlineNests(function, functionAuthorized);
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
    std::vector<llvm::json::Object> autos;
    for (const OutlinedNest &nest : outline.nests) {
      if (!autoMode) {
        records.push_back(analyzeNest(nest, functionOptions, backend));
        NestRecord &record = records.back();
        record.assumptions.insert(record.assumptions.begin(), tableAssumptions.begin(), tableAssumptions.end());
        continue;
      }
      llvm::json::Object passes;
      NestAnalysisOptions base = functionOptions;
      NestRecord plain = analyzeNest(nest, base, backend);
      passes["plain"] = recordSummary(plain);
      if (!(plain.orderedFoldVerified && plain.outputInjectiveProven)) {
        NestAnalysisOptions invariants = base;
        invariants.indexInvariants = true;
        NestRecord again = analyzeNest(nest, invariants, backend);
        passes["index_invariants"] = recordSummary(again);
        bool better = (again.orderedFoldVerified && !plain.orderedFoldVerified) ||
                      (again.orderedFoldVerified && again.outputInjectiveProven && !plain.outputInjectiveProven);
        if (better) {
          plain = std::move(again);
          base = invariants;
        }
      }
      // The record without the speculative premise: what covers inputs with repeated ids.
      bool withoutDistinct = plain.orderedFoldVerified;
      NestRecord chosen = std::move(plain);
      bool injective = false;
      llvm::json::Object speculation;
      llvm::json::Array speculated;
      if (!(chosen.orderedFoldVerified && chosen.outputInjectiveProven) && !premises.addressTables.empty()) {
        std::vector<std::vector<unsigned>> groups;
        std::string infeasible;
        for (unsigned t : premises.addressTables)
          for (const TableRewrite::Cells &c : rewrite.cells)
            if (c.table == t) {
              speculated.push_back((int64_t)t);
              int64_t values = 0;
              for (const TableSpec &spec : functionTables)
                if (spec.argument == t)
                  values = spec.upper - spec.lower + 1;
              if ((int64_t)c.count > values)
                infeasible += (infeasible.empty() ? "" : "; ") + std::string("argument ") + std::to_string(t) +
                              " has " + std::to_string(c.count) + " cells over " + std::to_string(values) +
                              " values: they cannot be pairwise distinct";
              std::vector<unsigned> group;
              for (unsigned j = 0; j < c.count; ++j)
                group.push_back(c.first + j);
              groups.push_back(std::move(group));
            }
        speculation["tables"] = std::move(speculated);
        if (!infeasible.empty()) {
          speculation["skipped"] = infeasible;
        } else {
          NestAnalysisOptions distinct = base;
          distinct.distinct = groups;
          NestRecord withPremise = analyzeNest(nest, distinct, backend);
          speculation["injective"] = recordSummary(withPremise);
          if (withPremise.orderedFoldVerified && withPremise.outputInjectiveProven) {
            chosen = std::move(withPremise);
            injective = true;
          }
        }
        speculation["kept"] = injective;
      }
      passes["speculation"] = std::move(speculation);
      // What a lowering may do, and what it must check first.
      std::string fast, collision = "n/a";
      if (injective) {
        fast = "parallel_if_injective";
        collision = withoutDistinct ? "csr_reduction" : "as_written";
      } else if (chosen.orderedFoldVerified && chosen.outputInjectiveProven) {
        fast = "parallel";
      } else if (chosen.orderedFoldVerified && !premises.addressTables.empty()) {
        fast = "csr_reduction";
      } else if (chosen.orderedFoldVerified) {
        fast = "serial";
      } else {
        fast = "as_written";
      }
      llvm::json::Array guards;
      for (const auto &[a, b] : premises.separate)
        guards.push_back(llvm::json::Object{{"kind", "separate"}, {"arguments", llvm::json::Array{(int64_t)a, (int64_t)b}}});
      for (const NestScopeEntry &e : premises.scope)
        guards.push_back(llvm::json::Object{{"kind", "range"}, {"argument", (int64_t)e.argument},
                                            {"lower", e.lower}, {"upper", e.upper}});
      for (const TableRewrite::Cells &c : rewrite.cells)
        for (const TableSpec &spec : functionTables)
          if (spec.argument == c.table)
            guards.push_back(llvm::json::Object{{"kind", "table"}, {"argument", (int64_t)c.table},
                                                {"cells", (int64_t)c.count}, {"first_cell_argument", (int64_t)c.first},
                                                {"lower", spec.lower}, {"upper", spec.upper}});
      if (injective)
        for (unsigned t : premises.addressTables)
          guards.push_back(llvm::json::Object{{"kind", "distinct"}, {"argument", (int64_t)t}, {"speculative", true}});
      llvm::json::Array notes;
      for (const std::string &n : premises.notes)
        notes.push_back(n);
      autos.push_back(llvm::json::Object{
          {"premises", std::move(guards)},
          {"passes", std::move(passes)},
          {"plan", llvm::json::Object{{"fast_path", fast},
                                      {"on_repeated_index", collision},
                                      {"on_premise_violation", "as_written"},
                                      {"on_index_out_of_bounds", "refuse"}}},
          {"notes", std::move(notes)}});
      chosen.assumptions.insert(chosen.assumptions.begin(), tableAssumptions.begin(), tableAssumptions.end());
      records.push_back(std::move(chosen));
    }
    llvm::json::Object functionJSON = toJSON(outline, records);
    if (autoMode)
      if (llvm::json::Array *nests = functionJSON.getArray("nests"))
        for (size_t n = 0; n < nests->size() && n < autos.size(); ++n)
          if (llvm::json::Object *nest = (*nests)[n].getAsObject())
            (*nest)["auto"] = std::move(autos[n]);
    functions.push_back(std::move(functionJSON));
  }
  if (!outlineOnly)
    llvm::outs() << llvm::formatv("{0:2}", llvm::json::Value(std::move(functions))) << "\n";
  return 0;
}
