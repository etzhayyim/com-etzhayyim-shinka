"""Pure-logic tests for ShinkaEvolutionCell (S0).

Standalone-runnable (no pytest / no langgraph required):
    python3 20-actors/shinka/cells/shinka_engine/test_cell.py

Verifies the three constitutional invariants (I1 append-only, I2 no-auto-merge,
I3 Murakumo-only fail-open) plus the generate→debate→evolve pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cell import (  # noqa: E402
    EvolutionState,
    Proposal,
    ShinkaEvolutionCell,
    _datom,
    _local_scan_ok,
    elo_update,
    node_propose,
    node_reflect,
    node_synthesize,
)

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {name}")


# --- I1: append-only datoms ------------------------------------------------- #

def test_datom_add_only() -> None:
    d = _datom("e", ":a", 1)
    check("datom op is :db/add", d["op"] == ":db/add")
    raised = False
    try:
        _datom("e", ":a", 1, op=":db/retract")
    except ValueError:
        raised = True
    check("I1: :db/retract refused", raised)


def test_history_is_evidence() -> None:
    # A charter-rejected proposal must still appear as datoms (evidence, not deletion).
    # Force a rejection at the mechanism level (independent of which scanner is loaded):
    # patch _scan_ok so any body containing "REJECTME" fails the gate.
    import cell as cellmod

    orig = cellmod._scan_ok
    cellmod._scan_ok = lambda t: "REJECTME" not in t
    try:
        st = EvolutionState(task="t", n_propose=2)
        node_propose(st)
        st.proposals[0].body = "REJECTME charter-failing proposal"
        node_reflect(st)
    finally:
        cellmod._scan_ok = orig
    check("rejected proposal recorded", len(st.rejected) == 1)
    rej_pid = st.rejected[0].pid
    has_status = any(
        d["e"] == f"shinka:proposal/{rej_pid}" and d["v"] == "charter-rejected"
        for d in st.datoms
    )
    check("I1: rejection emitted as datom", has_status)
    check("all emitted datoms are :db/add", all(d["op"] == ":db/add" for d in st.datoms))


# --- I2: no autonomous merge ------------------------------------------------ #

def test_no_auto_merge() -> None:
    st = EvolutionState(task="improve-cell", n_propose=4)
    cell = ShinkaEvolutionCell()
    out = cell.solve(st)
    check("pr_draft produced", out.pr_draft is not None)
    check("I2: member_signed False", out.pr_draft["member_signed"] is False)
    check("I2: auto_merge False", out.pr_draft["auto_merge"] is False)
    check("I2: not committable without CACAO", ShinkaEvolutionCell.is_committable(out) is False)
    out.member_cacao = "cacao_b64_opaque_member_signature"
    check("I2: committable after CACAO attached", ShinkaEvolutionCell.is_committable(out) is True)


# --- I3: Murakumo-only fail-open -------------------------------------------- #

def test_infer_fail_open() -> None:
    def broken_infer(_prompt: str) -> str:
        raise RuntimeError("fleet unreachable")

    st = EvolutionState(task="t", n_propose=4)
    cell = ShinkaEvolutionCell(infer=broken_infer)
    out = cell.solve(st)  # must not raise — falls open to the deterministic kernel
    check("I3: solve survives broken infer", out.errorMsg is None)
    check("I3: tournament still ran", len(out.debates) > 0)


def test_charter_scan_local() -> None:
    check("clean text ok", _local_scan_ok("implement a Pregel cell") is True)
    check("runpod flagged", _local_scan_ok("rent a RunPod commercial gpu") is False)


# --- pipeline + Elo --------------------------------------------------------- #

def test_elo_update() -> None:
    ra, rb = elo_update(1200.0, 1200.0, a_won=True)
    check("winner gains", ra > 1200.0)
    check("loser drops", rb < 1200.0)
    check("zero-sum-ish", abs((ra - 1200.0) + (rb - 1200.0)) < 1e-6)


def test_full_pipeline() -> None:
    st = EvolutionState(task="upgrade-himawari-cell", context_refs=["datom:1", "datom:2"], n_propose=4)
    cell = ShinkaEvolutionCell()
    out = cell.solve(st)
    check("proposals survived", len(out.proposals) >= 1)
    check("clusters assigned", all(p.cluster_id is not None for p in out.proposals))
    check("merged winner exists", out.merged is not None)
    check("meta_review non-empty", len(out.meta_review) > 0)
    # Loop-B coupling is dry-run only: staged, not written.
    check("corpus candidate staged", len(out.corpus_candidates) == 1)
    check(
        "corpus candidate well-formed",
        set(out.corpus_candidates[0]) == {"id", "instruction", "completion"},
    )


def test_determinism() -> None:
    def run() -> tuple[str, float]:
        st = EvolutionState(task="same-task", n_propose=4)
        out = ShinkaEvolutionCell().solve(st)
        return out.merged.pid, round(out.merged.elo, 3)

    check("deterministic merged winner", run() == run())


def main() -> int:
    for fn in (
        test_datom_add_only,
        test_history_is_evidence,
        test_no_auto_merge,
        test_infer_fail_open,
        test_charter_scan_local,
        test_elo_update,
        test_full_pipeline,
        test_determinism,
    ):
        fn()
    print(f"shinka_engine: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
