"""Integration test: FleetSampler (Track A) backs Loop-A proposal generation.

Standalone-runnable:
    python3 src/shinka/engine/test_fleet_wiring.py

Proves that when a FleetSampler is supplied, ShinkaEvolutionCell.propose draws
proposal bodies via fleet best-of-N (test-time compute), and that the
orchestrator threads the sampler through — while the no-sampler path is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "shinka" / "engine"))

from cell import EvolutionState, ShinkaEvolutionCell, node_propose  # noqa: E402
from fleet_sampler import FleetSampler  # noqa: E402
from orchestrator import ShinkaOrchestrator  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {name}")


def test_propose_uses_fleet_when_sampler_present() -> None:
    sampler = FleetSampler()  # deterministic kernel; bodies tagged with node name
    st = EvolutionState(task="upgrade-cell", n_propose=4)
    node_propose(st, sampler)
    check("4 proposals", len(st.proposals) == 4)
    # fleet kernel bodies look like "[node#k] completion for: ..."; rationale cites node.
    check(
        "bodies are fleet-sampled",
        all("completion for:" in p.body for p in st.proposals),
    )
    check(
        "rationale cites fleet best-of-3",
        all("fleet best-of-3" in p.rationale for p in st.proposals),
    )


def test_propose_kernel_unchanged_without_sampler() -> None:
    st = EvolutionState(task="t", n_propose=4)
    node_propose(st)  # no sampler
    check("kernel bodies use 'candidate'", all("candidate" in p.body for p in st.proposals))
    check("kernel rationale uses 'angle'", all("angle" in p.rationale for p in st.proposals))


def test_cell_solve_with_sampler() -> None:
    cell = ShinkaEvolutionCell(sampler=FleetSampler())
    out = cell.solve(EvolutionState(task="add-field", n_propose=4))
    check("winner produced via fleet path", out.merged is not None)
    check("pr draft present", out.pr_draft is not None)
    check("still no auto-merge (I2)", out.pr_draft["auto_merge"] is False)


def test_orchestrator_threads_sampler() -> None:
    orch = ShinkaOrchestrator(corpus_count=125, sampler=FleetSampler())
    rec = orch.beat("fix-rule")
    check("beat ran with sampler", rec.seq == 1)
    check("winner present", rec.winner is not None)
    check("corpus advanced", rec.corpus_after == 126)
    check("all datoms :db/add", all(d["op"] == ":db/add" for d in rec.datoms))


def main() -> int:
    for fn in (
        test_propose_uses_fleet_when_sampler_present,
        test_propose_kernel_unchanged_without_sampler,
        test_cell_solve_with_sampler,
        test_orchestrator_threads_sampler,
    ):
        fn()
    print(f"fleet_wiring: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
