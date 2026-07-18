"""Pure-logic tests for shinka_orchestrator (S0).

Standalone-runnable:
    python3 src/shinka/engine/test_orchestrator.py

Covers the beat cycle, the flywheel coupling across beats, Murakumo narration
fail-open (I3), no-auto-merge (I2), and the ibuki replay/idempotence property
(I1: state restores byte-identically from the datom log; re-beating a seen task
stages no duplicate corpus pairs).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "shinka" / "engine"))

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


def test_single_beat() -> None:
    orch = ShinkaOrchestrator(corpus_count=125)
    rec = orch.beat("upgrade-himawari-cell", context_refs=["datom:1"])
    check("seq is 1", rec.seq == 1)
    check("winner present", rec.winner is not None)
    check("debates ran", rec.debates > 0)
    check("corpus advanced by 1", rec.corpus_after == 126)
    check("pr draft present", rec.pr_draft is not None)
    check("all datoms are :db/add", all(d["op"] == ":db/add" for d in rec.datoms))
    check("narration non-empty", len(rec.narration) > 0)


def test_train_blocked_honest() -> None:
    # 125 corpus + EVO-X2 offline → Loop B must report blocked, never a flip.
    orch = ShinkaOrchestrator(corpus_count=125, evo_x2_online=False)
    rec = orch.beat("t")
    check("train blocked", rec.train_status == "blocked")
    check("no flip", rec.flip_available is False)


def test_no_auto_merge() -> None:
    orch = ShinkaOrchestrator()
    rec = orch.beat("add-lexicon-field")
    check("I2: member_signed False", rec.pr_draft["member_signed"] is False)
    check("I2: auto_merge False", rec.pr_draft["auto_merge"] is False)
    check("I2: not committable w/o CACAO", ShinkaOrchestrator.is_committable(rec, None) is False)
    check(
        "I2: committable with CACAO",
        ShinkaOrchestrator.is_committable(rec, "cacao_b64_member_sig") is True,
    )


def test_narration_fail_open() -> None:
    def boom(_p: str) -> str:
        raise RuntimeError("fleet unreachable")

    orch = ShinkaOrchestrator(infer=boom)
    rec = orch.beat("t")  # must not raise
    check("I3: beat survives broken infer", rec.seq == 1)
    check("I3: template narration used", "beat 1" in rec.narration)


def test_flywheel_across_beats() -> None:
    orch = ShinkaOrchestrator(corpus_count=125)
    orch.beat("task-A")
    orch.beat("task-B")
    rec3 = orch.beat("task-C")
    check("three beats advanced corpus to 128", rec3.corpus_after == 128)
    check("3 distinct corpus ids seen", len(orch.seen_corpus_ids) == 3)


def test_replay_idempotent() -> None:
    # Run two beats, capture the log, replay into a fresh orchestrator.
    orch = ShinkaOrchestrator(corpus_count=125)
    orch.beat("task-A")
    orch.beat("task-B")
    log_snapshot = list(orch.log)
    seen_before = set(orch.seen_corpus_ids)
    count_before = orch.corpus_count

    resumed = ShinkaOrchestrator(corpus_count=0)  # wrong count on purpose
    resumed.replay(log_snapshot)
    check("replay restores beat_seq", resumed.beat_seq == 2)
    check("replay restores corpus count", resumed.corpus_count == count_before)
    check("replay restores seen ids", resumed.seen_corpus_ids == seen_before)

    # Re-beat a task whose winner id was already staged → idempotent (no dup).
    rec = resumed.beat("task-A")
    check("re-beat seq advances", rec.seq == 3)
    check("re-beat stages no duplicate corpus pair", rec.corpus_staged == 0)
    check("corpus count unchanged on dup", rec.corpus_after == count_before)


def main() -> int:
    for fn in (
        test_single_beat,
        test_train_blocked_honest,
        test_no_auto_merge,
        test_narration_fail_open,
        test_flywheel_across_beats,
        test_replay_idempotent,
    ):
        fn()
    print(f"orchestrator: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
