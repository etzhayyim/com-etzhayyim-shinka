"""ShinkaEvolutionCell — Shinka capability-evolution cell (Loop A, S0).

Per ADR-2606142200. A single LangGraph super-step graph that maps the
DeepMind co-scientist generate→debate→evolve cycle onto etzhayyim primitives,
driven by the Supervisor (the graph itself):

    propose (Generation) → reflect (Reflection/critic, Charter G1-G8 pre-scan)
      → cluster (Proximity, diversity) → rank (Ranking, Elo pairwise debate)
      → recombine (Evolution) → synthesize (Meta-review, PR draft) → emit (datoms)

The cell is deterministic and LLM-free at S0 so it runs and tests offline; the
Murakumo debate is a typed hook (`_murakumo_debate`) that fails OPEN to the
deterministic kernel (Murakumo-only invariant I3 — never a commercial call).

Invariants (enforced here + in test_cell.py):
  I1  every fact emitted is an append-only `:db/add` datom — `_datom` REFUSES
      to build a `:db/retract`. The evolution history cannot be rewritten.
  I2  `synthesize` emits a PR draft with member_signed=False / auto_merge=False.
      `is_committable()` is False until a member CACAO capability is attached.
  I3  inference resolves Murakumo-only; the offline kernel is the fail-open path.

This is the Loop-A engine. Its tournament winners stage `corpus_candidates`
(dry-run) that the Loop-B Maxwell RSi pipeline (collect_corpus → gate_candidates
→ train → eval → deploy) consumes — the flywheel — but writing them is an
operator/leash-gated step performed elsewhere, never by this cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

try:  # LangGraph is the canonical super-step runtime (sibling: himawari cells).
    from langgraph.graph import StateGraph  # type: ignore
except Exception:  # pragma: no cover - env without a working langgraph install
    # Fall back to a tiny in-process sequential executor so the cell still
    # imports and runs. The node functions are identical; only the driver differs.
    StateGraph = None  # type: ignore[assignment]

# --- Charter Rider G1-G8 scanner (reused; fail-open to a minimal local scan) ---
# Imports the canonical scanner from the standalone etzhayyim-organism actor.
# We import best-effort so this cell
# stays pure-stdlib runnable in a bare checkout.
try:  # pragma: no cover - depends on monorepo layout at runtime
    import pathlib as _pathlib
    import sys as _sys

    _ROOT = _pathlib.Path(__file__).resolve().parents[3]
    _CR_SRC = _ROOT.parent / "com-etzhayyim-organism" / "src"
    if _CR_SRC.exists() and str(_CR_SRC) not in _sys.path:
        _sys.path.insert(0, str(_CR_SRC))
    from etzhayyim_organism.sensors.charter_rider import scan as _charter_scan  # type: ignore

    def _scan_ok(text: str) -> bool:
        try:
            return bool(_charter_scan(text).ok)
        except Exception:
            return _local_scan_ok(text)
except Exception:  # bare checkout / scanner unavailable

    def _scan_ok(text: str) -> bool:
        return _local_scan_ok(text)


# Minimal local fallback — a conservative subset of Charter Rider §2 prohibited
# signals (ADR-2605192200). Real gating uses the canonical scanner above; this
# only guarantees the cell never passes an obviously-prohibited proposal offline.
_PROHIBITED_SIGNALS = (
    "runpod",
    "vertex ai",
    "aws bedrock",
    "commercial gpu",
    "weapon design",
    "covert force",
    "child sexual",
    "transfer() land",
    "setowner",
)


def _local_scan_ok(text: str) -> bool:
    low = text.lower()
    return not any(sig in low for sig in _PROHIBITED_SIGNALS)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

_DEFAULT_ELO = 1200.0
_ELO_K = 32.0


@dataclass
class Proposal:
    """A candidate mutation (new cell version / schema upgrade / corpus pair / code)."""

    pid: str
    kind: str  # e.g. "cell-impl" | "schema-upgrade" | "corpus-pair" | "code-fix"
    body: str
    rationale: str
    source_refs: list[str] = field(default_factory=list)  # Datom-log retrieval anchors
    charter_ok: bool | None = None
    review_score: float = 0.0  # Reflection correctness heuristic, 0..1
    elo: float = _DEFAULT_ELO
    cluster_id: int | None = None
    is_duplicate: bool = False

    def text(self) -> str:
        return f"{self.kind}\n{self.body}\n{self.rationale}"


@dataclass
class EvolutionState:
    """State threaded through the Shinka evolution super-step graph."""

    task: str
    context_refs: list[str] = field(default_factory=list)
    n_propose: int = 4
    proposals: list[Proposal] = field(default_factory=list)
    rejected: list[Proposal] = field(default_factory=list)  # charter-failed (kept as evidence)
    debates: list[dict[str, Any]] = field(default_factory=list)
    merged: Proposal | None = None
    meta_review: str = ""
    pr_draft: dict[str, Any] | None = None
    corpus_candidates: list[dict[str, str]] = field(default_factory=list)  # Loop-B feed (dry-run)
    datoms: list[dict[str, Any]] = field(default_factory=list)
    member_cacao: str | None = None  # opaque CACAO capability; None ⇒ not committable
    errorMsg: str | None = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _datom(e: str, a: str, v: Any, op: str = ":db/add") -> dict[str, Any]:
    """Build an append-only datom. I1: refuses anything but :db/add."""
    if op != ":db/add":
        raise ValueError(
            f"shinka evolution history is append-only (I1): refused op {op!r}; "
            "proposals/verdicts are facts, never retractions (ADR-2606142200)"
        )
    return {"e": e, "a": a, "v": v, "op": op}


def elo_update(ra: float, rb: float, a_won: bool, k: float = _ELO_K) -> tuple[float, float]:
    """Standard Elo update for a pairwise debate (AlphaGo-style ranking, co-scientist)."""
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    eb = 1.0 - ea
    sa = 1.0 if a_won else 0.0
    sb = 1.0 - sa
    return ra + k * (sa - ea), rb + k * (sb - eb)


def _stable_score(text: str) -> float:
    """Deterministic 0..1 quality proxy (LLM-free kernel stand-in for a judge).

    A stable hash over the proposal text — replayable in tests; at S1 the
    `rank` node swaps this for a Murakumo pairwise debate verdict.
    """
    h = 0
    for ch in text:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (h % 1000) / 1000.0


# --------------------------------------------------------------------------- #
# Node functions (Co-scientist agents) — pure: state-in, state-out
# --------------------------------------------------------------------------- #


def node_propose(state: EvolutionState, sampler: Any | None = None) -> EvolutionState:
    """Generation: emit n candidate mutations grounded in retrieved context.

    When a `sampler` (FleetSampler, Research Track A) is supplied, each proposal
    BODY is drawn via fleet best-of-N + Elo selection across the murakumo nodes
    (test-time compute), not the deterministic template. `sampler=None` keeps the
    offline kernel. Murakumo-only (I3) — the sampler resolves to fleet endpoints.
    """
    kinds = ("cell-impl", "schema-upgrade", "corpus-pair", "code-fix")
    for i in range(state.n_propose):
        kind = kinds[i % len(kinds)]
        pid = f"p{i}"
        if sampler is not None:
            prompt = f"[{kind}] propose mutation {i} for task: {state.task}"
            res = sampler.best_of_n(prompt, n=3)
            body = res.winner.text if res.winner else prompt
            rationale = (
                f"fleet best-of-3 winner from {res.winner.node if res.winner else 'kernel'}; "
                f"grounded in {len(state.context_refs)} Datom-log refs"
            )
        else:
            body = f"[{kind}] candidate {i} for task: {state.task}"
            rationale = f"grounded in {len(state.context_refs)} Datom-log refs; angle {i}"
        p = Proposal(
            pid=pid,
            kind=kind,
            body=body,
            rationale=rationale,
            source_refs=list(state.context_refs),
        )
        state.proposals.append(p)
        state.datoms.append(_datom(f"shinka:proposal/{pid}", ":proposal/kind", kind))
        state.datoms.append(_datom(f"shinka:proposal/{pid}", ":proposal/task", state.task))
    return state


def node_reflect(state: EvolutionState) -> EvolutionState:
    """Reflection (virtual peer review): Charter G1-G8 pre-scan + correctness score.

    Charter-failing proposals are moved to `rejected` but STILL recorded as datoms
    (I1) — a rejection is evidence, not a deletion.
    """
    survivors: list[Proposal] = []
    for p in state.proposals:
        p.charter_ok = _scan_ok(p.text())
        p.review_score = _stable_score(p.text())
        state.datoms.append(
            _datom(f"shinka:proposal/{p.pid}", ":proposal/charter-ok", p.charter_ok)
        )
        state.datoms.append(
            _datom(f"shinka:proposal/{p.pid}", ":proposal/review-score", round(p.review_score, 3))
        )
        if p.charter_ok:
            survivors.append(p)
        else:
            state.rejected.append(p)
            state.datoms.append(
                _datom(f"shinka:proposal/{p.pid}", ":proposal/status", "charter-rejected")
            )
    state.proposals = survivors
    return state


def node_cluster(state: EvolutionState) -> EvolutionState:
    """Proximity: cluster by kind for diversity; flag duplicates (keep best per cluster)."""
    by_kind: dict[str, list[Proposal]] = {}
    for p in state.proposals:
        by_kind.setdefault(p.kind, []).append(p)
    for cid, (kind, group) in enumerate(sorted(by_kind.items())):
        group.sort(key=lambda x: x.review_score, reverse=True)
        for rank_in_group, p in enumerate(group):
            p.cluster_id = cid
            p.is_duplicate = rank_in_group > 0  # keep the top of each cluster
            state.datoms.append(_datom(f"shinka:proposal/{p.pid}", ":proposal/cluster", cid))
    return state


def _murakumo_debate(
    a: Proposal, b: Proposal, infer: Callable[[str], str] | None
) -> bool:
    """Pairwise scientific debate. I3: Murakumo-only; fails OPEN to the kernel.

    Returns True iff `a` wins. With a live Murakumo `infer` hook this runs a
    structured debate; offline it falls back to the deterministic review proxy.
    """
    if infer is not None:
        try:
            verdict = infer(
                "Adversarially debate which proposal better advances the task "
                f"under the Charter. A:\n{a.text()}\n\nB:\n{b.text()}\n\n"
                "Answer exactly 'A' or 'B'."
            ).strip().upper()
            if verdict.startswith("A"):
                return True
            if verdict.startswith("B"):
                return False
        except Exception:
            pass  # fail open to kernel
    # Deterministic kernel: higher review score wins; tie broken by stable hash.
    if a.review_score != b.review_score:
        return a.review_score > b.review_score
    return _stable_score(a.pid) >= _stable_score(b.pid)


def node_rank(
    state: EvolutionState, infer: Callable[[str], str] | None = None
) -> EvolutionState:
    """Ranking: round-robin Elo tournament over the non-duplicate survivors."""
    contenders = [p for p in state.proposals if not p.is_duplicate]
    for i in range(len(contenders)):
        for j in range(i + 1, len(contenders)):
            a, b = contenders[i], contenders[j]
            a_won = _murakumo_debate(a, b, infer)
            a.elo, b.elo = elo_update(a.elo, b.elo, a_won)
            winner = a.pid if a_won else b.pid
            state.debates.append({"a": a.pid, "b": b.pid, "winner": winner})
            state.datoms.append(
                _datom(f"shinka:debate/{a.pid}-vs-{b.pid}", ":debate/winner", winner)
            )
    for p in contenders:
        state.datoms.append(
            _datom(f"shinka:proposal/{p.pid}", ":proposal/elo", round(p.elo, 1))
        )
    return state


def node_recombine(state: EvolutionState) -> EvolutionState:
    """Evolution: merge the top-2 Elo contenders into one stronger candidate."""
    contenders = sorted(
        (p for p in state.proposals if not p.is_duplicate),
        key=lambda x: x.elo,
        reverse=True,
    )
    if not contenders:
        return state
    top = contenders[0]
    if len(contenders) >= 2:
        second = contenders[1]
        merged = Proposal(
            pid="merged",
            kind=top.kind,
            body=f"{top.body}\n+ grafted from {second.pid}: {second.body}",
            rationale=f"recombination of top-Elo {top.pid}({top.elo:.0f}) "
            f"+ {second.pid}({second.elo:.0f})",
            source_refs=sorted(set(top.source_refs) | set(second.source_refs)),
        )
    else:
        merged = top
    merged.charter_ok = _scan_ok(merged.text())  # re-scan the recombinant (I-safety)
    merged.review_score = _stable_score(merged.text())
    state.merged = merged
    state.datoms.append(_datom("shinka:proposal/merged", ":proposal/source", top.pid))
    state.datoms.append(
        _datom("shinka:proposal/merged", ":proposal/charter-ok", merged.charter_ok)
    )
    return state


def node_synthesize(state: EvolutionState) -> EvolutionState:
    """Meta-review: PR draft (NEVER auto-merge, I2) + dry-run Loop-B corpus feed."""
    winner = state.merged
    n_kept = len([p for p in state.proposals if not p.is_duplicate])
    state.meta_review = (
        f"Shinka evolution over task {state.task!r}: "
        f"{len(state.proposals)} charter-clean proposals "
        f"({len(state.rejected)} rejected), {n_kept} contenders debated in "
        f"{len(state.debates)} matches; winner = "
        f"{winner.pid if winner else 'none'}."
    )
    if winner is not None and winner.charter_ok:
        state.pr_draft = {
            "title": f"shinka: {winner.kind} for {state.task}",
            "body": f"{state.meta_review}\n\n{winner.body}\n\nRationale: {winner.rationale}",
            "member_signed": False,  # I2 — requires a member CACAO capability
            "auto_merge": False,  # I2 — never autonomous
            "source_refs": winner.source_refs,
        }
        # Loop-B coupling (DRY-RUN): stage the winner as a Maxwell SFT pair.
        # Publishing it into an external corpus is the operator/leash-gated step
        # and is never done here.
        state.corpus_candidates.append(
            {
                "id": f"shinka/{state.task}/{winner.pid}",
                "instruction": f"Advance task: {state.task}",
                "completion": winner.body,
            }
        )
        state.datoms.append(_datom("shinka:pr/draft", ":pr/winner", winner.pid))
        state.datoms.append(_datom("shinka:pr/draft", ":pr/auto-merge", False))
    return state


# --------------------------------------------------------------------------- #
# The cell
# --------------------------------------------------------------------------- #


class ShinkaEvolutionCell:
    """Supervisor-driven generate→debate→evolve→synthesize super-step graph.

    `.solve(state)` runs the full beat for one task. `infer` is an optional
    Murakumo inference callable (resolved Murakumo-only by the host); `sampler`
    is an optional FleetSampler (Research Track A) that backs `propose` with
    fleet best-of-N test-time compute. When both are None the deterministic
    kernel drives every node (I3 fail-open).
    """

    def __init__(
        self,
        infer: Callable[[str], str] | None = None,
        sampler: Any | None = None,
    ) -> None:
        self.infer = infer
        self.sampler = sampler
        self.graph = self._build_graph()

    def _node_table(self) -> dict[str, Callable[[EvolutionState], EvolutionState]]:
        return {
            "propose": lambda s: node_propose(s, self.sampler),
            "reflect": node_reflect,
            "cluster": node_cluster,
            "rank": lambda s: node_rank(s, self.infer),
            "recombine": node_recombine,
            "synthesize": node_synthesize,
        }

    _ORDER = ("propose", "reflect", "cluster", "rank", "recombine", "synthesize")

    def _build_graph(self):
        if StateGraph is None:
            return None
        graph = StateGraph(EvolutionState)
        table = self._node_table()
        for name in self._ORDER:
            graph.add_node(name, table[name])
        graph.set_entry_point(self._ORDER[0])
        for a, b in zip(self._ORDER, self._ORDER[1:]):
            graph.add_edge(a, b)
        from langgraph.graph import END  # type: ignore

        graph.add_edge(self._ORDER[-1], END)
        return graph.compile()

    def _sequential(self, state: EvolutionState) -> EvolutionState:
        table = self._node_table()
        for name in self._ORDER:
            state = table[name](state)
        return state

    def solve(self, state: EvolutionState) -> EvolutionState:
        if self.graph is None:
            return self._sequential(state)
        return self.graph.invoke(state)

    @staticmethod
    def is_committable(state: EvolutionState) -> bool:
        """I2: a PR draft becomes committable ONLY with a member CACAO capability."""
        return bool(
            state.pr_draft
            and state.member_cacao
            and not state.pr_draft.get("auto_merge", False)
        )
