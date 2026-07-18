"""Pure-logic tests for run_beat (the Shinka beat runner / fleet entrypoint).

Standalone-runnable:
    python3 src/shinka/engine/test_run_beat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "shinka" / "engine"))

from orchestrator import ShinkaOrchestrator  # noqa: E402
from run_beat import DEFAULT_BACKLOG, format_digest, main, run  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {name}")


def test_run_beats() -> None:
    orch = ShinkaOrchestrator(corpus_count=125)
    recs = run(orch, list(DEFAULT_BACKLOG), beats=3)
    check("3 records", len(recs) == 3)
    check("seqs 1..3", [r.seq for r in recs] == [1, 2, 3])
    check("corpus advanced to 128", recs[-1].corpus_after == 128)
    check("heads chained (distinct)", len({r.head_cid for r in recs}) == 3)
    check("all beats draft-only (I2)", all(r.pr_draft["auto_merge"] is False for r in recs))
    check("train honestly blocked offline", all(r.train_status == "blocked" for r in recs))


def test_round_robin_backlog() -> None:
    orch = ShinkaOrchestrator(corpus_count=125)
    # more beats than backlog → wraps around
    recs = run(orch, ["a", "b"], beats=5)
    check("5 beats over 2 tasks", len(recs) == 5)
    check("seqs monotone", [r.seq for r in recs] == [1, 2, 3, 4, 5])


def test_empty_backlog_raises() -> None:
    raised = False
    try:
        run(ShinkaOrchestrator(), [], beats=2)
    except ValueError:
        raised = True
    check("empty backlog raises", raised)


def test_digest_format() -> None:
    orch = ShinkaOrchestrator(corpus_count=125)
    recs = run(orch, ["task-x"], beats=2)
    d = format_digest(recs)
    check("digest has beat lines", d.count("beat ") >= 2)
    check("digest footer present", "beats;" in d)
    check("digest shows head", "head " in d)


def test_main_offline_ok() -> None:
    check("main runs offline returns 0", main(["--beats", "2"]) == 0)
    check("main zero beats returns 0", main(["--beats", "0"]) == 0)


def main_tests() -> int:
    for fn in (
        test_run_beats,
        test_round_robin_backlog,
        test_empty_backlog_raises,
        test_digest_format,
        test_main_offline_ok,
    ):
        fn()
    print(f"run_beat: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main_tests())
