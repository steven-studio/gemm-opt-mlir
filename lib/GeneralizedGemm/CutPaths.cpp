#include "GeneralizedGemm/CutPaths.h"

#include "GeneralizedGemm/SymbolicEvaluator.h"

#include "llvm/ADT/DenseSet.h"

#include <functional>
#include <set>
#include <tuple>

using namespace mlir::generalized_gemm;

namespace mlir::generalized_gemm {

bool containsFloat(Term term) {
  llvm::DenseSet<Term> seen;
  std::function<bool(Term)> visit = [&](Term n) {
    if (!seen.insert(n).second)
      return false;
    if (n->sort.kind == Sort::Float)
      return true;
    for (Term a : n->args)
      if (visit(a))
        return true;
    return false;
  };
  return visit(term);
}

Term postTerm(const SemanticProgram &program, const CutPath &path, Term term) {
  auto leaf = [&](Term n) -> Term {
    if (n->op == TermOp::State)
      return path.state[n->index];
    if (n->op == TermOp::Shadow)
      return path.shadow[n->index];
    if (n->op == TermOp::MemVar)
      return path.memory[n->index];
    return nullptr;
  };
  return program.terms->substitute(term, leaf);
}

std::string describePath(const SemanticProgram &program, const CutPath &path) {
  std::string out = "L" + std::to_string(path.from);
  for (unsigned index : path.transitions) {
    const Transition &t = program.transitions[index];
    out += " -" + std::to_string(t.slot) + "-> ";
    out += t.to == program.exitLocation() ? std::string("exit")
                                          : "L" + std::to_string(t.to);
  }
  return out;
}

PathSet enumerateCutPaths(const SemanticProgram &program,
                          llvm::ArrayRef<unsigned> cuts,
                          const PathLimits &limits) {
  PathSet set;
  TermContext &terms = *program.terms;
  size_t startTerms = terms.size();
  struct LimitReached {
    std::string reason;
  };
  std::vector<bool> isCut(program.locationCount, false);
  for (unsigned cut : cuts)
    isCut[cut] = true;
  std::vector<std::vector<unsigned>> outgoing(program.locationCount);
  for (unsigned i = 0; i < program.transitions.size(); ++i)
    outgoing[program.transitions[i].from].push_back(i);

  // Guards so far, and no retained stop at any block already left.
  auto liveCondition = [&](const SymbolicPath &path) {
    Term condition = path.condition;
    for (const SideCondition &side : path.sideConditions)
      if (!containsFloat(side.violated))
        condition =
            terms.logicalAnd(condition, terms.logicalNot(side.violated));
    return condition;
  };

  auto walkFrom = [&](unsigned source, bool fromEntry) {
    std::function<void(const SymbolicPath &)> walk = [&](const SymbolicPath &path) {
      if (path.transitions.size() >= limits.maxPathLength)
        throw LimitReached{"path length limit"};
      if (terms.size() - startTerms > limits.maxNewTerms)
        throw LimitReached{"term limit"};
      const std::vector<unsigned> &leaving = outgoing[path.end];
      if (leaving.empty())
        throw LimitReached{"location " + std::to_string(path.end) +
                           " has no transition"};
      // Each block's conditions once per way of reaching it. Every transition
      // of a block carries the same list.
      SymbolicPath probe = extendSymbolicPath(program, path, leaving.front());
      Term reached = liveCondition(path);
      for (unsigned i = path.sideConditions.size();
           i < probe.sideConditions.size(); ++i) {
        const SideCondition &side = probe.sideConditions[i];
        StopSite site;
        site.from = source;
        site.fromEntry = fromEntry;
        site.prefix = path.transitions;
        site.location = path.end;
        site.kind = side.kind;
        site.reason = side.reason;
        site.reached = reached;
        site.violated = side.violated;
        set.sites.push_back(std::move(site));
        if (set.sites.size() > limits.maxStopSites)
          throw LimitReached{"stop site limit"};
      }
      for (unsigned index : leaving) {
        SymbolicPath next = extendSymbolicPath(program, path, index);
        unsigned to = next.end;
        if (to == program.exitLocation() || isCut[to]) {
          CutPath done;
          done.id = set.paths.size();
          done.from = source;
          done.to = to;
          done.fromEntry = fromEntry;
          done.transitions = next.transitions;
          done.condition = liveCondition(next);
          done.state = next.state;
          done.shadow = next.shadow;
          done.memory = next.memory;
          set.paths.push_back(std::move(done));
          if (set.paths.size() > limits.maxPaths)
            throw LimitReached{"path limit"};
        } else {
          walk(next);
        }
      }
    };
    SymbolicPath start = startSymbolicPath(program, source);
    if (fromEntry)
      start.shadow.assign(program.state.size(), terms.boolean(false));
    walk(start);
  };

  try {
    walkFrom(program.entryLocation, /*fromEntry=*/true);
    for (unsigned cut : cuts)
      walkFrom(cut, /*fromEntry=*/false);
  } catch (const LimitReached &limit) {
    set.paths.clear();
    set.sites.clear();
    set.reason = limit.reason;
    return set;
  } catch (const TermError &error) {
    set.paths.clear();
    set.sites.clear();
    set.reason = error.what();
    return set;
  }
  set.complete = true;
  return set;
}

std::map<unsigned, Term> onceAssignedDefinitions(const SemanticProgram &program) {
  std::map<unsigned, Term> only;
  std::set<unsigned> many;
  for (const Transition &t : program.transitions)
    for (const auto &[id, term] : t.assignments) {
      if (term->op == TermOp::State && term->index == id)
        continue; // Kept.
      auto [slot, added] = only.emplace(id, term);
      if (!added && slot->second != term)
        many.insert(id);
    }
  std::map<unsigned, Term> out;
  for (const auto &[id, term] : only) {
    if (many.count(id) || term->sort.kind != Sort::BV || term->op == TermOp::State || containsFloat(term))
      continue;
    bool bad = false;
    std::function<void(Term)> scan = [&](Term n) {
      if (bad)
        return;
      if (n->op == TermOp::Load || n->op == TermOp::Shadow || n->op == TermOp::MemVar ||
          (n->op == TermOp::State && n->index == id)) {
        bad = true;
        return;
      }
      for (Term a : n->args)
        scan(a);
    };
    scan(term);
    if (!bad)
      out.emplace(id, term);
  }
  return out;
}

std::set<unsigned> steppedStates(const SemanticProgram &program) {
  std::set<unsigned> out;
  for (const Transition &t : program.transitions)
    for (const auto &[id, term] : t.assignments)
      if (!(term->op == TermOp::State && term->index == id) && llvm::is_contained(stateSupport(term), id))
        out.insert(id);
  return out;
}

std::vector<GuardBound> guardBounds(const SemanticProgram &program) {
  TermContext &terms = *program.terms;
  std::map<unsigned, Term> definitions = onceAssignedDefinitions(program);
  // Through the definitions, transitively (a definition may read a constant held in
  // state); a bounded number of rounds, as definitions may form no cycle but chains.
  auto read = [&](Term t) {
    for (unsigned round = 0; round < 8; ++round) {
      Term next = terms.substitute(t, [&](Term leaf) -> Term {
        if (leaf->op != TermOp::State)
          return nullptr;
        auto d = definitions.find(leaf->index);
        return d == definitions.end() ? nullptr : d->second;
      });
      if (next == t)
        break;
      t = next;
    }
    return t;
  };
  std::vector<GuardBound> out;
  std::set<std::tuple<unsigned, bool, Term>> seen;
  std::set<Term> visited;
  std::function<void(Term)> collect = [&](Term g) {
    if (!g || !visited.insert(g).second)
      return;
    if (g->op == TermOp::And || g->op == TermOp::Or || g->op == TermOp::Not || g->op == TermOp::Xor) {
      for (Term a : g->args)
        collect(a);
      return;
    }
    if (g->op != TermOp::ICmp || g->args.size() != 2)
      return;
    auto pred = static_cast<Predicate>(g->aux);
    Term a = g->args[0], b = g->args[1];
    auto add = [&](Term x, Predicate p, Term other) {
      // A state with one definition holds a constant or a term of other state: no loop.
      if (x->op != TermOp::State || other->sort != x->sort || definitions.count(x->index))
        return;
      Term bound = read(other);
      if (llvm::is_contained(stateSupport(bound), x->index))
        return;
      std::function<bool(Term)> loads = [&](Term n) {
        if (n->op == TermOp::Load || n->op == TermOp::Shadow)
          return true;
        return llvm::any_of(n->args, loads);
      };
      if (loads(bound) || containsFloat(bound))
        return;
      Term one = terms.bv(llvm::APInt(x->sort.width, 1));
      GuardBound gb;
      gb.state = x->index;
      switch (p) {
      case Predicate::Ult: case Predicate::Slt: gb.upper = true; gb.bound = bound; break;
      case Predicate::Ule: case Predicate::Sle: gb.upper = true; gb.bound = terms.binary(TermOp::Add, bound, one); break;
      case Predicate::Uge: case Predicate::Sge: gb.upper = false; gb.bound = bound; break;
      case Predicate::Ugt: case Predicate::Sgt: gb.upper = false; gb.bound = terms.binary(TermOp::Add, bound, one); break;
      default: return;
      }
      if (seen.insert({gb.state, gb.upper, gb.bound}).second)
        out.push_back(gb);
    };
    auto flip = [](Predicate p) {
      switch (p) {
      case Predicate::Ult: return Predicate::Ugt;
      case Predicate::Ule: return Predicate::Uge;
      case Predicate::Ugt: return Predicate::Ult;
      case Predicate::Uge: return Predicate::Ule;
      case Predicate::Slt: return Predicate::Sgt;
      case Predicate::Sle: return Predicate::Sge;
      case Predicate::Sgt: return Predicate::Slt;
      case Predicate::Sge: return Predicate::Sle;
      default: return p;
      }
    };
    add(a, pred, b);
    add(b, flip(pred), a);
  };
  for (const Transition &t : program.transitions)
    collect(t.guard);
  // Loop entries: a state that one transition sets from a term not reading it (its start)
  // while others step it from itself is bounded below by that start, for a loop that
  // counts up. A proposal, as the rest.
  std::map<unsigned, std::set<Term>> starts;
  std::set<unsigned> steps;
  for (const Transition &t : program.transitions)
    for (const auto &[id, term] : t.assignments) {
      if (term->op == TermOp::State && term->index == id)
        continue;
      if (llvm::is_contained(stateSupport(term), id))
        steps.insert(id);
      else
        starts[id].insert(term);
    }
  for (const auto &[id, terms_] : starts) {
    if (!steps.count(id) || terms_.size() != 1 || definitions.count(id))
      continue;
    Term start = read(*terms_.begin());
    std::function<bool(Term)> loads = [&](Term n) {
      if (n->op == TermOp::Load || n->op == TermOp::Shadow)
        return true;
      return llvm::any_of(n->args, loads);
    };
    if (start->sort.kind != Sort::BV || loads(start) || containsFloat(start) ||
        llvm::is_contained(stateSupport(start), id))
      continue;
    GuardBound gb;
    gb.state = id;
    gb.upper = false;
    gb.bound = start;
    gb.start = true;
    if (seen.insert({gb.state, gb.upper, gb.bound}).second)
      out.push_back(gb);
  }
  return out;
}

} // namespace mlir::generalized_gemm
