#include "GeneralizedGemm/CoordinateSynthesis.h"

#include "GeneralizedGemm/CandidateChecker.h"
#include "GeneralizedGemm/CutPaths.h"
#include "GeneralizedGemm/DomainModel.h"

#include <algorithm>
#include <map>
#include <set>

using namespace llvm;

namespace mlir::generalized_gemm {

namespace {

using Literal = SelectorLiteral;
using Clause = SelectorClause;

struct Mode {
  bool box, unit;
};

struct Layer {
  Mode mode;
  std::vector<unsigned> needed; // Axes that must hold an atom.
  unsigned batchAtoms, totalAtoms;
};

using Selection = std::array<std::vector<unsigned>, 4>;

class Search {
public:
  Search(const SemanticProgram &program, const AtomBank &bank,
         ArrayRef<const AccessDomain *> domains, const SynthesisLimits &limits,
         SmtBackend &backend)
      : program(program), bank(bank), domains(domains), limits(limits), backend(backend) {}

  SynthesisOutcome run();

private:
  std::optional<Selection> solve(const Layer &layer, bool &failed);
  CoordinateCandidate labels(const Selection &selection, unsigned domain) const;
  bool judge(const Selection &selection, Mode mode, bool &stuck);
  void pairClause(const Selection &selection, const CoordinateCandidate &c, unsigned d,
                  const Counterexample &missing, bool &added);
  void separated(uint64_t first, uint64_t second, unsigned d, ArrayRef<unsigned> axes,
                 Clause &clause) const;
  void accept(const Selection &selection);
  std::vector<uint32_t> kernelKey(const Selection &selection) const;
  void seed();
  void seedCounters();
  std::optional<std::vector<unsigned>> atomsFor(const std::vector<Partition> &target,
                                                ArrayRef<unsigned> given) const;

  const SemanticProgram &program;
  const AtomBank &bank;
  ArrayRef<const AccessDomain *> domains;
  const SynthesisLimits &limits;
  SmtBackend &backend;
  std::vector<Clause> blocked;
  std::set<std::vector<uint32_t>> kernels;
  SynthesisOutcome out;
};

static const char *axisName(unsigned axis) { return "PIJK" + axis; }

std::string selector(unsigned atom, unsigned axis) {
  return "s" + std::string(1, *axisName(axis)) + std::to_string(atom);
}

std::optional<Selection> Search::solve(const Layer &layer, bool &failed) {
  unsigned count = bank.atoms.size();
  std::string script;
  for (unsigned a = 0; a < count; ++a)
    for (unsigned axis = 0; axis < 4; ++axis)
      script += "(declare-const " + selector(a, axis) + " (_ BitVec 1))\n";
  auto on = [](const std::string &name) { return "(= " + name + " #b1)"; };
  for (unsigned a = 0; a < count; ++a)
    for (unsigned x = 0; x < 4; ++x)
      for (unsigned y = x + 1; y < 4; ++y)
        script += "(assert (not (and " + on(selector(a, x)) + " " + on(selector(a, y)) + ")))\n";
  auto clauseText = [&](const Clause &clause) {
    std::string text = "(assert (or false";
    for (const Literal &l : clause) {
      std::string atom = on(selector(l.atom, l.axis));
      text += l.positive ? " " + atom : " (not " + atom + ")";
    }
    return text + "))\n";
  };
  for (const Clause &clause : out.always)
    script += clauseText(clause);
  for (const Clause &clause : blocked)
    script += clauseText(clause);
  if (layer.mode.unit)
    for (const Clause &clause : out.whenUnit)
      script += clauseText(clause);
  if (layer.mode.box)
    for (const Clause &clause : out.whenBox)
      script += clauseText(clause);
  for (unsigned axis : layer.needed) {
    Clause some;
    for (unsigned a = 0; a < count; ++a)
      some.push_back({a, axis, true});
    script += clauseText(some);
  }
  auto sum = [&](ArrayRef<unsigned> axes) {
    std::string text = "(_ bv0 12)";
    for (unsigned a = 0; a < count; ++a)
      for (unsigned axis : axes)
        text = "(bvadd " + text + " ((_ zero_extend 11) " + selector(a, axis) + "))";
    return text;
  };
  auto bound = [](unsigned value) { return "(_ bv" + std::to_string(value) + " 12)"; };
  script += "(assert (bvule " + sum({AxisP}) + " " + bound(layer.batchAtoms) + "))\n";
  script += "(assert (bvule " + sum({AxisI, AxisJ, AxisK}) + " " + bound(layer.totalAtoms) + "))\n";
  script += "(check-sat)\n";
  SmtResponse response = backend.check(script, limits.solver, /*wantModel=*/true);
  if (response.answer == SmtAnswer::Unsat)
    return std::nullopt;
  if (response.answer == SmtAnswer::Unknown) {
    failed = true;
    out.stop = "the selector query: " + response.detail;
    return std::nullopt;
  }
  Selection selection;
  for (unsigned axis = 0; axis < 4; ++axis)
    for (unsigned a = 0; a < count; ++a) {
      auto found = response.model.find(selector(a, axis));
      if (found != response.model.end() && found->second == "1")
        selection[axis].push_back(a);
    }
  return selection;
}

CoordinateCandidate Search::labels(const Selection &selection, unsigned d) const {
  CoordinateCandidate candidate;
  std::array<std::vector<uint32_t> *, 4> slots{&candidate.p, &candidate.i, &candidate.j,
                                               &candidate.k};
  size_t events = domains[d]->table().events().size();
  for (unsigned axis = 0; axis < 4; ++axis) {
    std::map<std::vector<uint64_t>, uint32_t> ids;
    for (size_t e = 0; e < events; ++e) {
      std::vector<uint64_t> key;
      for (unsigned atom : selection[axis])
        key.push_back(bank.atoms[atom].values[d][e]);
      slots[axis]->push_back(ids.try_emplace(std::move(key), (uint32_t)ids.size()).first->second);
    }
  }
  return candidate;
}

void Search::separated(uint64_t first, uint64_t second, unsigned d, ArrayRef<unsigned> axes,
                       Clause &clause) const {
  for (unsigned a = 0; a < bank.atoms.size(); ++a)
    if (bank.atoms[a].values[d][first] != bank.atoms[a].values[d][second])
      for (unsigned axis : axes)
        clause.push_back({a, axis, true});
}

/// Theorem 6, box clause: narrow an absent point to one atom of each of two
/// axes, or else to the three axes together. Adding atoms cannot restore a point.
/// The point is absent from this batch only, and a coarser P joins batches: so the
/// clause applies only to assignments that keep every atom of this P.
void Search::pairClause(const Selection &selection, const CoordinateCandidate &c, unsigned d,
                        const Counterexample &missing, bool &added) {
  size_t events = c.p.size();
  auto witness = [&](const std::vector<uint32_t> &axis, uint32_t label) {
    for (size_t e = 0; e < events; ++e)
      if (c.p[e] == missing.labels[0] && axis[e] == label)
        return e;
    return events;
  };
  size_t ei = witness(c.i, missing.labels[1]), ej = witness(c.j, missing.labels[2]),
         ek = witness(c.k, missing.labels[3]);
  if (ei == events || ej == events || ek == events)
    return;
  Clause keptBatch;
  for (unsigned atom : selection[AxisP])
    keptBatch.push_back({atom, AxisP, false});
  struct Pair {
    unsigned x, y;
    size_t ex, ey;
  };
  for (Pair pair : {Pair{AxisI, AxisJ, ei, ej}, Pair{AxisI, AxisK, ei, ek},
                    Pair{AxisJ, AxisK, ej, ek}})
    for (unsigned a : selection[pair.x])
      for (unsigned b : selection[pair.y]) {
        uint64_t va = bank.atoms[a].values[d][pair.ex], vb = bank.atoms[b].values[d][pair.ey];
        bool seen = false;
        for (size_t e = 0; e < events && !seen; ++e)
          seen = c.p[e] == missing.labels[0] && bank.atoms[a].values[d][e] == va &&
                 bank.atoms[b].values[d][e] == vb;
        if (seen)
          continue;
        Clause clause = keptBatch;
        clause.push_back({a, pair.x, false});
        clause.push_back({b, pair.y, false});
        separated(pair.ex, pair.ey, d, {AxisP}, clause);
        out.whenBox.push_back(std::move(clause));
        added = true;
        return;
      }
  Clause clause = keptBatch;
  for (unsigned axis : {AxisI, AxisJ, AxisK})
    for (unsigned a : selection[axis])
      clause.push_back({a, axis, false});
  separated(ei, ej, d, {AxisP}, clause);
  separated(ei, ek, d, {AxisP}, clause);
  out.whenBox.push_back(std::move(clause));
  added = true;
}

/// True when every domain passes. Else adds clauses, and `stuck` means the mode
/// cannot work with these atoms at all.
bool Search::judge(const Selection &selection, Mode mode, bool &stuck) {
  bool passed = true;
  for (unsigned d = 0; d < domains.size(); ++d) {
    CoordinateCandidate candidate = labels(selection, d);
    CandidateReport report = inspectCandidate(domains[d]->table(), candidate);
    if (!report.malformed.empty()) {
      stuck = true;
      return false;
    }
    if (!report.decompositionValid) {
      passed = false;
      const Counterexample &failure = report.counterexamples.front();
      Clause clause;
      switch (failure.kind) {
      case CounterexampleKind::DependenceH:
        separated(failure.first, failure.second, d, {AxisP, AxisI, AxisJ}, clause);
        break;
      case CounterexampleKind::DependenceF:
        separated(failure.first, failure.second, d, {AxisP, AxisI, AxisK}, clause);
        break;
      case CounterexampleKind::DependenceG:
        separated(failure.first, failure.second, d, {AxisP, AxisK, AxisJ}, clause);
        break;
      default:
        stuck = true;
        out.stop = "the group fails before coordinates matter";
        return false;
      }
      if (clause.empty()) {
        // No atom separates events with different views: this bank cannot work.
        stuck = true;
        out.stop = "no atom separates two events with different accesses";
        return false;
      }
      out.always.push_back(std::move(clause));
      continue;
    }
    if (mode.unit && report.multiplicity != MultiplicityForm::Unit) {
      passed = false;
      std::map<std::array<uint32_t, 4>, uint64_t> first;
      for (uint64_t e = 0; e < candidate.p.size(); ++e) {
        auto [slot, added] = first.try_emplace(
            {candidate.p[e], candidate.i[e], candidate.j[e], candidate.k[e]}, e);
        if (added)
          continue;
        Clause clause;
        separated(slot->second, e, d, {AxisP, AxisI, AxisJ, AxisK}, clause);
        if (clause.empty()) {
          stuck = true; // These atoms cannot tell the two events apart.
          return false;
        }
        out.whenUnit.push_back(std::move(clause));
        break;
      }
      continue;
    }
    if (mode.box && report.cartesian != CartesianForm::Box) {
      passed = false;
      bool added = false;
      for (const Counterexample &c : report.counterexamples)
        if (c.kind == CounterexampleKind::MissingPoint)
          pairClause(selection, candidate, d, c, added);
      if (!added) {
        stuck = true;
        return false;
      }
    }
  }
  return passed;
}

std::vector<uint32_t> Search::kernelKey(const Selection &selection) const {
  std::vector<uint32_t> key;
  for (unsigned d = 0; d < domains.size(); ++d) {
    CoordinateCandidate c = labels(selection, d);
    for (const std::vector<uint32_t> *axis : {&c.p, &c.i, &c.j, &c.k}) {
      key.insert(key.end(), axis->begin(), axis->end());
      key.push_back(UINT32_MAX);
    }
  }
  return key;
}

void Search::accept(const Selection &selection) {
  // A distinct structure is a distinct equivalence of the events on each axis.
  if (!kernels.insert(kernelKey(selection)).second)
    return;
  CoordinateFamily family;
  family.origin = "selector search";
  bool fromPrograms = false;
  for (unsigned axis = 0; axis < 4; ++axis)
    for (unsigned atom : selection[axis]) {
      family.axes[axis].push_back(bank.atoms[atom].term);
      family.programs[axis].push_back(bank.atoms[atom].program);
      fromPrograms = fromPrograms || !bank.atoms[atom].program.empty();
    }
  if (fromPrograms)
    family.origin += " over statement programs";
  else
    family.programs = {};
  out.candidates.push_back(std::move(family));
}

/// The fewest atoms, in order, that determine `target` given `given`. A hint for
/// the views that G3a can already split.
std::optional<std::vector<unsigned>> Search::atomsFor(const std::vector<Partition> &target,
                                                      ArrayRef<unsigned> given) const {
  std::vector<unsigned> chosen(given.begin(), given.end());
  auto key = [&](ArrayRef<unsigned> list, unsigned d, size_t e) {
    std::vector<uint64_t> values;
    for (unsigned a : list)
      values.push_back(bank.atoms[a].values[d][e]);
    return values;
  };
  auto matches = [&](ArrayRef<unsigned> list) {
    for (unsigned d = 0; d < domains.size(); ++d) {
      std::map<std::vector<uint64_t>, uint32_t> forward;
      std::map<uint32_t, std::vector<uint64_t>> backward;
      for (size_t e = 0; e < target[d].size(); ++e) {
        auto values = key(list, d, e);
        auto [one, a] = forward.try_emplace(values, target[d][e]);
        auto [two, b] = backward.try_emplace(target[d][e], values);
        if (one->second != target[d][e] || two->second != values)
          return false;
      }
    }
    return true;
  };
  auto functionOfTarget = [&](unsigned atom) {
    for (unsigned d = 0; d < domains.size(); ++d) {
      std::map<uint32_t, uint64_t> seen;
      for (size_t e = 0; e < target[d].size(); ++e) {
        auto [slot, added] = seen.try_emplace(target[d][e], bank.atoms[atom].values[d][e]);
        if (slot->second != bank.atoms[atom].values[d][e])
          return false;
      }
    }
    return true;
  };
  for (unsigned a = 0; a < bank.atoms.size() && !matches(chosen); ++a)
    if (functionOfTarget(a) && std::find(chosen.begin(), chosen.end(), a) == chosen.end())
      chosen.push_back(a);
  if (!matches(chosen))
    return std::nullopt;
  std::vector<unsigned> added(chosen.begin() + given.size(), chosen.end());
  return added;
}

void Search::seed() {
  std::array<std::vector<Partition>, 4> targets;
  for (const AccessDomain *domain : domains) {
    const AccessTable &table = domain->table();
    CanonicalPartitions p;
    viewKernels(table, p.kerH, p.kerF, p.kerG);
    p = joinKernels(p.kerH, p.kerF, p.kerG);
    CoordinateCandidate c = canonicalCandidate(table, p);
    targets[AxisP].push_back(c.p);
    targets[AxisI].push_back(c.i);
    targets[AxisJ].push_back(c.j);
    targets[AxisK].push_back(c.k);
  }
  auto p = atomsFor(targets[AxisP], {});
  if (!p)
    return;
  Selection selection;
  selection[AxisP] = *p;
  for (unsigned axis : {AxisI, AxisJ, AxisK}) {
    auto atoms = atomsFor(targets[axis], *p);
    if (!atoms)
      return;
    selection[axis] = *atoms;
  }
  bool stuck = false;
  if (judge(selection, Mode{false, false}, stuck))
    accept(selection);
}

/// A second seed, only when the canonical one gives nothing: the family of the counters
/// themselves, each on the axis named by the views it moves. A counter that changes C's
/// cell and A's address (and not B's) is I; C and B, J; A and B, K; all three, P. Whether a
/// counter moves a view is read off the samples: two events agreeing on every other
/// counter, differing on this one, with different addresses of the view. When an index
/// table makes two logical outputs share a cell at some sample (a scatter-add whose ids
/// repeat), the canonical partition of C's cells mixes them; the counters do not. A
/// proposal like any other: the judge below and then every checker decide.
void Search::seedCounters() {
  std::set<unsigned> counters = steppedStates(program);
  std::vector<unsigned> leaves;
  for (unsigned a = 0; a < bank.atoms.size(); ++a) {
    if (!bank.atoms[a].term || bank.atoms[a].term->op != TermOp::State ||
        !counters.count(bank.atoms[a].term->index))
      continue;
    bool varies = false;
    for (unsigned d = 0; d < domains.size() && !varies; ++d)
      for (size_t e = 1; e < bank.atoms[a].values[d].size() && !varies; ++e)
        varies = bank.atoms[a].values[d][e] != bank.atoms[a].values[d][0];
    if (varies)
      leaves.push_back(a);
  }
  if (leaves.empty())
    return;
  Selection selection;
  for (unsigned a : leaves) {
    std::array<bool, 3> moves{false, false, false}; // h, f, g
    for (unsigned d = 0; d < domains.size(); ++d) {
      const auto &events = domains[d]->table().events();
      std::map<std::vector<uint64_t>, std::map<uint64_t, std::array<uint32_t, 3>>> byOthers;
      for (size_t e = 0; e < events.size(); ++e) {
        std::vector<uint64_t> others;
        for (unsigned b : leaves)
          if (b != a)
            others.push_back(bank.atoms[b].values[d][e]);
        std::array<uint32_t, 3> views{events[e].h, events[e].f, events[e].g};
        auto &row = byOthers[others];
        for (const auto &[value, seen] : row)
          if (value != bank.atoms[a].values[d][e])
            for (unsigned v = 0; v < 3; ++v)
              moves[v] = moves[v] || seen[v] != views[v];
        row.try_emplace(bank.atoms[a].values[d][e], views);
      }
    }
    int axis = moves[0] && moves[1] && moves[2] ? AxisP
               : moves[0] && moves[1] && !moves[2] ? AxisI
               : moves[0] && !moves[1] && moves[2] ? AxisJ
               : !moves[0] && moves[1] && moves[2] ? AxisK
                                                   : -1;
    if (axis < 0)
      return; // A counter that moves one view or none: no role, no seed.
    selection[axis].push_back(a);
  }
  if (selection[AxisI].empty() || selection[AxisJ].empty() || selection[AxisK].empty())
    return;
  bool stuck = false;
  if (judge(selection, Mode{false, false}, stuck))
    accept(selection);
}

SynthesisOutcome Search::run() {
  if (limits.countersFirst)
    seedCounters();
  seed();
  // Preference, by search order only: a box before a partial domain, unit
  // multiplicity, every axis present, few batch atoms, small tuples.
  std::vector<Mode> modes{{true, true}, {true, false}, {false, true}, {false, false}};
  std::vector<std::vector<unsigned>> needs{{AxisI, AxisJ, AxisK}, {AxisK}, {}};
  bool complete = true;
  // The limit is per mode: a run of box candidates must not crowd out a partial
  // decomposition, which is a different structure and may be the natural one.
  for (Mode mode : modes) {
    size_t startOfMode = out.candidates.size();
    bool modeFull = false;
    for (const auto &needed : needs) {
      if (modeFull)
        break;
      for (unsigned batch = 0; batch <= limits.maxBatchAtoms; ++batch) {
        Layer layer{mode, needed, batch, (unsigned)bank.atoms.size()};
        while (!modeFull) {
          if (out.candidates.size() - startOfMode >= limits.maxCandidates) {
            modeFull = true;
            complete = false;
            break;
          }
          if (out.rounds >= limits.maxRounds ||
              out.always.size() + out.whenUnit.size() + out.whenBox.size() >= limits.maxClauses) {
            out.clauses = out.always.size() + out.whenUnit.size() + out.whenBox.size();
            out.stop = "the round or clause limit";
            return std::move(out);
          }
          ++out.rounds;
          bool failed = false;
          auto selection = solve(layer, failed);
          if (failed)
            return std::move(out);
          if (!selection)
            break;
          bool stuck = false;
          if (judge(*selection, mode, stuck)) {
            accept(*selection);
            // Nothing a superset adds is wanted: move to another structure.
            Clause block;
            for (unsigned axis = 0; axis < 4; ++axis)
              for (unsigned atom : (*selection)[axis])
                block.push_back({atom, axis, false});
            if (block.empty())
              break;
            blocked.push_back(std::move(block));
          } else if (stuck) {
            complete = false;
            break;
          }
        }
      }
    }
  }
  out.clauses = out.always.size() + out.whenUnit.size() + out.whenBox.size();
  out.exhausted = complete;
  return std::move(out);
}

/// Inner levels change most often: the order is proposed from that, and tested.
std::optional<std::vector<unsigned>> levelOrder(const std::vector<std::vector<std::vector<uint64_t>>> &values) {
  unsigned count = values.size();
  std::vector<uint64_t> changes(count, 0);
  for (unsigned c = 0; c < count; ++c)
    for (const auto &domain : values[c])
      for (size_t e = 1; e < domain.size(); ++e)
        changes[c] += domain[e] != domain[e - 1];
  std::vector<unsigned> order(count);
  for (unsigned c = 0; c < count; ++c)
    order[c] = c;
  std::stable_sort(order.begin(), order.end(),
                   [&](unsigned a, unsigned b) { return changes[a] < changes[b]; });
  for (unsigned d = 0; d < values[0].size(); ++d)
    for (size_t e = 1; e < values[0][d].size(); ++e) {
      bool increased = false;
      for (unsigned c : order) {
        uint64_t before = values[c][d][e - 1], after = values[c][d][e];
        if (before == after)
          continue;
        increased = after == before + 1;
        break;
      }
      if (!increased)
        return std::nullopt;
    }
  return order;
}

} // namespace

SynthesisOutcome synthesizeCoordinates(const SemanticProgram &program, const AtomBank &bank,
                                       ArrayRef<const AccessDomain *> domains,
                                       const SynthesisLimits &limits, SmtBackend &backend) {
  return Search(program, bank, domains, limits, backend).run();
}

namespace {
std::optional<EnumerationOrder> proposeOrderAt(const SemanticProgram &program,
                                               const CoordinateFamily &family,
                                               ArrayRef<const AccessDomain *> domains, bool extended);
} // namespace

/// The unit forms first; where they bound no order, the extended ones (bounds the
/// program's guards and loop starts give, a level at a stride), so that an order proposed
/// before is proposed the same.
std::optional<EnumerationOrder> proposeOrder(const SemanticProgram &program,
                                             const CoordinateFamily &family,
                                             ArrayRef<const AccessDomain *> domains) {
  if (auto order = proposeOrderAt(program, family, domains, false))
    return order;
  return proposeOrderAt(program, family, domains, true);
}

namespace {
std::optional<EnumerationOrder> proposeOrderAt(const SemanticProgram &program,
                                               const CoordinateFamily &family,
                                               ArrayRef<const AccessDomain *> domains, bool extended) {
  TermContext &terms = *program.terms;
  std::vector<Term> components;
  for (const auto &axis : family.axes)
    components.insert(components.end(), axis.begin(), axis.end());
  if (components.empty() || domains.empty())
    return std::nullopt;
  unsigned width = components[0]->sort.width;
  for (Term term : components)
    if (term->sort.width != width)
      return std::nullopt;
  std::vector<std::vector<std::vector<uint64_t>>> values(components.size());
  for (unsigned c = 0; c < components.size(); ++c)
    for (const AccessDomain *domain : domains) {
      EventEvaluator evaluator(program, *domain);
      std::vector<uint64_t> list;
      for (size_t e = 0; e < evaluator.size(); ++e) {
        auto value = evaluator.value(components[c], e);
        if (!value)
          return std::nullopt;
        list.push_back(value->getZExtValue());
      }
      values[c].push_back(std::move(list));
    }
  auto order = levelOrder(values);
  if (!order)
    return std::nullopt;

  EnumerationOrder out;
  auto constant = [&](uint64_t v) { return terms.bv(APInt(width, v, false, true)); };
  std::vector<GuardBound> bounds = guardBounds(program);

  for (unsigned m = 0; m < order->size(); ++m) {
    unsigned c = (*order)[m];
    // Per domain and per prefix of outer levels: the range this level covers.
    uint64_t low = UINT64_MAX;
    std::vector<std::map<std::vector<uint64_t>, uint64_t>> tops(domains.size());
    for (unsigned d = 0; d < domains.size(); ++d)
      for (size_t e = 0; e < values[c][d].size(); ++e) {
        std::vector<uint64_t> prefix;
        for (unsigned outer = 0; outer < m; ++outer)
          prefix.push_back(values[(*order)[outer]][d][e]);
        uint64_t value = values[c][d][e];
        low = std::min(low, value);
        auto [slot, added] = tops[d].try_emplace(prefix, value + 1);
        slot->second = std::max(slot->second, value + 1);
      }
    OrderLevel level;
    level.value = components[c];
    level.lower = constant(low);
    // A bound the program's own guards give (CutPaths.h, guardBounds): checked against
    // every event, then read over outer levels as level variables. Nullptr where none
    // fits; the symbolic checker judges what is proposed.
    auto fromGuards = [&](bool upper) -> Term {
      if (components[c]->op != TermOp::State)
        return nullptr;
      std::map<unsigned, unsigned> outerOfState;
      for (unsigned q = 0; q < m; ++q)
        if (components[(*order)[q]]->op == TermOp::State)
          outerOfState[components[(*order)[q]]->index] = q;
      for (const GuardBound &gb : bounds) {
        if (gb.state != components[c]->index || gb.upper != upper || gb.bound->sort != components[c]->sort)
          continue;
        bool outerOnly = true;
        for (unsigned id : stateSupport(gb.bound))
          outerOnly = outerOnly && outerOfState.count(id);
        if (!outerOnly)
          continue;
        bool fits = true;
        for (unsigned d = 0; d < domains.size() && fits; ++d) {
          EventEvaluator evaluator(program, *domains[d]);
          std::map<std::vector<uint64_t>, uint64_t> want;
          for (size_t e = 0; e < values[c][d].size(); ++e) {
            std::vector<uint64_t> prefix;
            for (unsigned outer = 0; outer < m; ++outer)
              prefix.push_back(values[(*order)[outer]][d][e]);
            uint64_t v = values[c][d][e];
            auto [slot, added] = want.try_emplace(prefix, upper ? v + 1 : v);
            if (!added)
              slot->second = upper ? std::max(slot->second, v + 1) : std::min(slot->second, v);
          }
          for (size_t e = 0; e < values[c][d].size() && fits; ++e) {
            std::vector<uint64_t> prefix;
            for (unsigned outer = 0; outer < m; ++outer)
              prefix.push_back(values[(*order)[outer]][d][e]);
            // The bound encloses the events (V3 needs no more; coverage, which needs
            // the exact set, is proved or not by the simulation).
            auto got = evaluator.value(gb.bound, e);
            fits = got && (upper ? got->getZExtValue() >= want[prefix] : got->getZExtValue() <= want[prefix]);
          }
        }
        if (!fits)
          continue;
        return terms.substitute(gb.bound, [&](Term leaf) -> Term {
          if (leaf->op != TermOp::State)
            return nullptr;
          unsigned q = outerOfState.at(leaf->index);
          return terms.state(LevelVariableBase + q, components[(*order)[q]]->sort);
        });
      }
      return nullptr;
    };
    // bound = a * outer level + (parameter) + c, a in 1..8, fitted to the extremes per
    // prefix of every domain. Nullptr where none fits.
    auto affineOuter = [&](bool upper) -> Term {
      for (unsigned q = 0; q < m; ++q) {
        if (components[(*order)[q]]->sort != components[c]->sort)
          continue;
        for (uint64_t a = 1; a <= 8; ++a)
          for (int pi = -1; pi < (int)program.parameters.size(); ++pi) {
            if (pi >= 0 && program.parameters[pi].sort != components[c]->sort)
              continue;
            std::optional<uint64_t> offset;
            bool fits = true, any = false;
            for (unsigned d = 0; d < domains.size() && fits; ++d) {
              std::map<std::vector<uint64_t>, uint64_t> want;
              for (size_t e = 0; e < values[c][d].size(); ++e) {
                std::vector<uint64_t> prefix;
                for (unsigned outer = 0; outer < m; ++outer)
                  prefix.push_back(values[(*order)[outer]][d][e]);
                uint64_t v = values[c][d][e];
                auto [slot, added] = want.try_emplace(prefix, upper ? v + 1 : v);
                if (!added)
                  slot->second = upper ? std::max(slot->second, v + 1) : std::min(slot->second, v);
              }
              uint64_t pv = pi >= 0 ? domains[d]->parameters()[pi].getZExtValue() : 0;
              for (const auto &[prefix, v] : want) {
                uint64_t here = v - a * prefix[q] - pv;
                any = true;
                if (!offset)
                  offset = here;
                else if (*offset != here)
                  fits = false;
              }
            }
            if (!fits || !any)
              continue;
            Sort sort = components[c]->sort;
            Term y = terms.state(LevelVariableBase + q, sort);
            Term bound = a == 1 ? y : terms.binary(TermOp::Mul, terms.bv(APInt(width, a)), y);
            if (pi >= 0)
              bound = terms.binary(TermOp::Add, bound, terms.param(pi, sort));
            if (*offset != 0)
              bound = terms.binary(TermOp::Add, bound, constant(*offset));
            return bound;
          }
      }
      return nullptr;
    };
    {
      // A lower bound that moves with outer levels.
      std::set<uint64_t> lows;
      for (unsigned d = 0; d < domains.size(); ++d) {
        std::map<std::vector<uint64_t>, uint64_t> least;
        for (size_t e = 0; e < values[c][d].size(); ++e) {
          std::vector<uint64_t> prefix;
          for (unsigned outer = 0; outer < m; ++outer)
            prefix.push_back(values[(*order)[outer]][d][e]);
          auto [slot, added] = least.try_emplace(prefix, values[c][d][e]);
          if (!added)
            slot->second = std::min(slot->second, values[c][d][e]);
        }
        for (const auto &[prefix, v] : least)
          lows.insert(v);
      }
      if (extended && lows.size() > 1) {
        if (Term guarded = fromGuards(false))
          level.lower = guarded;
        else if (Term fitted = affineOuter(false))
          level.lower = fitted;
      }
    }
    // upper = base(parameters), or base + an outer level.
    bool fixed = true;
    std::vector<uint64_t> byDomain;
    for (const auto &map : tops) {
      std::set<uint64_t> distinct;
      for (const auto &[prefix, top] : map)
        distinct.insert(top);
      fixed = fixed && distinct.size() == 1;
      byDomain.push_back(distinct.empty() ? 0 : *distinct.begin());
    }
    bool found = false;
    if (fixed) {
      std::set<uint64_t> all(byDomain.begin(), byDomain.end());
      if (all.size() == 1) {
        level.upper = constant(*all.begin());
        found = true;
      }
      for (const Parameter &p : program.parameters) {
        if (found || p.sort.kind != Sort::BV || p.sort.width != width)
          continue;
        for (int64_t shift : {0, 1, -1, 2, -2}) {
          bool fits = true;
          for (unsigned d = 0; d < domains.size(); ++d)
            fits = fits && byDomain[d] ==
                               domains[d]->parameters()[p.id].getZExtValue() + (uint64_t)shift;
          if (!fits)
            continue;
          level.upper = terms.binary(TermOp::Add, terms.param(p.id, p.sort), constant(shift));
          found = true;
          break;
        }
      }
    } else {
      // A triangular level: the top follows an outer level by a constant.
      for (unsigned q = 0; q < m && !found; ++q) {
        std::set<uint64_t> shifts;
        for (unsigned d = 0; d < domains.size(); ++d)
          for (const auto &[prefix, top] : tops[d])
            shifts.insert(top - prefix[q]);
        if (shifts.size() == 1) {
          level.upper = constant(*shifts.begin());
          level.upperOuter = q;
          found = true;
        }
      }
    }
    if (!found && extended)
      if (Term guarded = fromGuards(true)) {
        level.upper = guarded;
        found = true;
      }
    if (!found && extended)
      if (Term fitted = affineOuter(true)) {
        level.upper = fitted;
        found = true;
      }
    if (!found)
      return std::nullopt;
    out.levels.push_back(level);
  }
  return out;
}
} // namespace

} // namespace mlir::generalized_gemm
