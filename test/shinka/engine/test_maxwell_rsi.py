"""Pure-logic tests for Shinka Loop-B (maxwell_rsi).

Standalone-runnable:
    python3 src/shinka/engine/test_maxwell_rsi.py

Covers the deploy gate (>=250 steps OR >=+5pp), Robin's hypothesis→experiment→
analyse→update loop (incl. honest EVO-X2-offline blocking, no fabricated runs),
and the flywheel ingest (dedup + Charter gate + corpus progress, dry-run only).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "shinka" / "engine"))

from maxwell_rsi import (  # noqa: E402
    CORPUS_M1_TARGET,
    CORPUS_TRAIN_FLOOR,
    DeployGate,
    RSiState,
    flywheel_ingest,
    run_rsi,
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


def test_gate_steps() -> None:
    g = DeployGate()
    check("250 steps passes", g.passes(250, 0.0) is True)
    check("249 steps + 0pp fails", g.passes(249, 0.0) is False)
    check("0 steps + 5pp passes", g.passes(0, 5.0) is True)
    check("0 steps + 4.9pp fails", g.passes(0, 4.9) is False)
    check("reason cites steps", "steps 250" in g.reason(250, 0.0))


def test_blocked_below_floor() -> None:
    # corpus below floor → blocked, no fabricated run.
    st = run_rsi(RSiState(corpus_pairs=125, evo_x2_online=True))
    check("below floor blocked", st.status == "blocked")
    check("no run executed", st.run["ran"] is False)
    check("no flip", st.decision["flip_available"] is False)


def test_blocked_evo_offline() -> None:
    # corpus over floor but EVO-X2 offline → honest block (live 2026-06-13 state).
    st = run_rsi(RSiState(corpus_pairs=600, evo_x2_online=False))
    check("evo offline blocked", st.status == "blocked")
    check("reason mentions EVO-X2", "EVO-X2" in st.run["reason"])
    check("provenance kind blocked", st.provenance["kind"] == "blocked")


def test_train_and_gate_pass() -> None:
    # enough corpus + online + a train hook that exceeds the step gate.
    st = run_rsi(
        RSiState(corpus_pairs=900, evo_x2_online=True),
        train_hook=lambda recipe: {"steps": 300},
        eval_hook=lambda run: 2.0,  # below pp gate, but steps carry it
    )
    check("trained then flipped", st.status == "flipped")
    check("flip_available True", st.decision["flip_available"] is True)
    check("reason cites steps", "steps 300" in st.decision["reason"])


def test_train_gate_via_microbench() -> None:
    st = run_rsi(
        RSiState(corpus_pairs=900, evo_x2_online=True),
        train_hook=lambda recipe: {"steps": 100},  # under step gate
        eval_hook=lambda run: 6.5,                  # over pp gate
    )
    check("flips via microbench", st.decision["flip_available"] is True)
    check("reason cites pp", "pp" in st.decision["reason"])


def test_train_gate_reject() -> None:
    st = run_rsi(
        RSiState(corpus_pairs=900, evo_x2_online=True),
        train_hook=lambda recipe: {"steps": 100},
        eval_hook=lambda run: 1.0,
    )
    check("rejected when both under gate", st.status == "rejected")
    check("flip_available False", st.decision["flip_available"] is False)


def test_train_hook_error_is_honest() -> None:
    def boom(_recipe):
        raise RuntimeError("ROCm OOM")

    st = run_rsi(RSiState(corpus_pairs=900, evo_x2_online=True), train_hook=boom)
    check("hook error → blocked", st.status == "blocked")
    check("hook error not fabricated", st.run["ran"] is False)


def test_flywheel_dedup_and_gate() -> None:
    cands = [
        {"id": "a", "instruction": "task1", "completion": "clean impl"},
        {"id": "a", "instruction": "task1", "completion": "dup"},          # dedup
        {"id": "b", "instruction": "task2", "completion": "REJECTME"},     # gated below
        {"id": "c", "instruction": "task3", "completion": "another clean"},
    ]
    # Force the Charter gate to reject id "b" deterministically.
    import maxwell_rsi as m

    orig = m._scan_ok
    m._scan_ok = lambda t: "REJECTME" not in t
    try:
        res = flywheel_ingest(cands, existing_ids=set(), current_count=125)
    finally:
        m._scan_ok = orig
    check("2 staged (a,c)", len(res.staged) == 2)
    check("1 dup skipped", res.skipped_dup == 1)
    check("1 charter-rejected", res.rejected_charter == 1)
    check("new count = 125 + 2", res.new_count == 127)
    check("not yet at floor", res.train_floor_reached is False)
    check("progress string", res.provenance["progress"] == f"127/{CORPUS_M1_TARGET}")


def test_flywheel_reaches_floor() -> None:
    cands = [{"id": f"p{i}", "instruction": f"t{i}", "completion": "clean"} for i in range(400)]
    res = flywheel_ingest(cands, existing_ids=set(), current_count=125)
    check("staged 400", len(res.staged) == 400)
    check("reaches train floor", res.new_count >= CORPUS_TRAIN_FLOOR)
    check("floor flag set", res.train_floor_reached is True)


def main() -> int:
    for fn in (
        test_gate_steps,
        test_blocked_below_floor,
        test_blocked_evo_offline,
        test_train_and_gate_pass,
        test_train_gate_via_microbench,
        test_train_gate_reject,
        test_train_hook_error_is_honest,
        test_flywheel_dedup_and_gate,
        test_flywheel_reaches_floor,
    ):
        fn()
    print(f"maxwell_rsi: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
