"""Pure-logic tests for reward (Research Track E verifier-grounded reward).

Standalone-runnable:
    python3 src/shinka/engine/test_reward.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "shinka" / "engine"))

from reward import (  # noqa: E402
    CHARTER_VETO,
    PreferencePair,
    RewardComponents,
    ScoredCandidate,
    aggregate_reward,
    build_preference_corpus,
    build_preference_pair,
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


def test_charter_hard_veto() -> None:
    r = aggregate_reward(RewardComponents(charter_ok=False, microbench_delta_pp=50, pr_outcome="MERGED"))
    check("charter fail → -inf despite great metrics", r == CHARTER_VETO)
    r2 = aggregate_reward(RewardComponents(charter_ok=True, microbench_delta_pp=0, pr_outcome="OPEN"))
    check("charter ok + neutral → 0", abs(r2) < 1e-9)


def test_pr_outcome_ordering() -> None:
    base = dict(charter_ok=True, microbench_delta_pp=0.0)
    merged = aggregate_reward(RewardComponents(**base, pr_outcome="MERGED"))
    open_ = aggregate_reward(RewardComponents(**base, pr_outcome="OPEN"))
    closed = aggregate_reward(RewardComponents(**base, pr_outcome="CLOSED"))
    check("MERGED > OPEN > CLOSED", merged > open_ > closed)
    check("MERGED = +0.5 (w_pr)", abs(merged - 0.5) < 1e-9)
    check("CLOSED = -0.5", abs(closed + 0.5) < 1e-9)


def test_microbench_saturating() -> None:
    big = aggregate_reward(RewardComponents(charter_ok=True, microbench_delta_pp=100))
    full = aggregate_reward(RewardComponents(charter_ok=True, microbench_delta_pp=10))
    check("microbench clamps at +10pp", abs(big - full) < 1e-9)
    check("+10pp → +0.5 (w_microbench)", abs(full - 0.5) < 1e-9)
    combo = aggregate_reward(
        RewardComponents(charter_ok=True, microbench_delta_pp=10, pr_outcome="MERGED")
    )
    check("merged + 10pp → 1.0", abs(combo - 1.0) < 1e-9)


def test_pair_margin() -> None:
    a = ScoredCandidate("a", RewardComponents(charter_ok=True, pr_outcome="MERGED"))      # +0.5
    b = ScoredCandidate("b", RewardComponents(charter_ok=True, pr_outcome="OPEN"))         # 0.0
    pair = build_preference_pair(a, b, margin=0.1)
    check("pair built", isinstance(pair, PreferencePair))
    check("chosen = a (merged)", pair.chosen == "a")
    check("rejected = b", pair.rejected == "b")
    check("margin 0.5", abs(pair.margin - 0.5) < 1e-9)


def test_pair_too_close() -> None:
    a = ScoredCandidate("a", RewardComponents(charter_ok=True, microbench_delta_pp=1.0))   # +0.05
    b = ScoredCandidate("b", RewardComponents(charter_ok=True, microbench_delta_pp=0.0))   # 0.0
    check("no pair when within margin", build_preference_pair(a, b, margin=0.1) is None)


def test_pair_charter_vetoed_loser() -> None:
    clean = ScoredCandidate("clean", RewardComponents(charter_ok=True, pr_outcome="OPEN")) # 0.0
    dirty = ScoredCandidate("dirty", RewardComponents(charter_ok=False, pr_outcome="MERGED"))
    pair = build_preference_pair(clean, dirty, margin=0.1)
    check("vetoed loser still pairs", pair is not None)
    check("chosen is the charter-clean one", pair.chosen == "clean")
    check("rejected is the vetoed one", pair.rejected == "dirty")
    check("margin is +inf vs veto", pair.margin == float("inf"))


def test_pair_both_vetoed() -> None:
    a = ScoredCandidate("a", RewardComponents(charter_ok=False))
    b = ScoredCandidate("b", RewardComponents(charter_ok=False))
    check("both vetoed → no pair", build_preference_pair(a, b) is None)


def test_corpus_basic() -> None:
    # one prompt-group: a clear winner + a clear loser + a near-tie
    groups = {
        "g1": [
            ScoredCandidate("win", RewardComponents(charter_ok=True, pr_outcome="MERGED")),   # +0.5
            ScoredCandidate("lose", RewardComponents(charter_ok=True, pr_outcome="CLOSED")),   # -0.5
            ScoredCandidate("tie", RewardComponents(charter_ok=True, microbench_delta_pp=9.0)),  # +0.45
        ]
    }
    corpus = build_preference_corpus(groups, margin=0.1)
    chosen = {p.chosen for p in corpus}
    rejected = {p.rejected for p in corpus}
    check("chosen is the winner", chosen == {"win"})
    check("loser pairs (margin cleared)", "lose" in rejected)
    check("near-tie excluded (within margin)", "tie" not in rejected)


def test_corpus_skips_singletons_and_vetoed_best() -> None:
    groups = {
        "single": [ScoredCandidate("a", RewardComponents(charter_ok=True))],
        "all_vetoed": [
            ScoredCandidate("x", RewardComponents(charter_ok=False)),
            ScoredCandidate("y", RewardComponents(charter_ok=False)),
        ],
    }
    check("no pairs from singletons or all-vetoed groups", build_preference_corpus(groups) == [])


def test_corpus_multi_group() -> None:
    groups = {
        "g1": [
            ScoredCandidate("g1win", RewardComponents(charter_ok=True, pr_outcome="MERGED")),
            ScoredCandidate("g1lose", RewardComponents(charter_ok=True, pr_outcome="CLOSED")),
        ],
        "g2": [
            ScoredCandidate("g2win", RewardComponents(charter_ok=True, microbench_delta_pp=10)),
            ScoredCandidate("g2dirty", RewardComponents(charter_ok=False, pr_outcome="MERGED")),
        ],
    }
    corpus = build_preference_corpus(groups)
    check("two pairs (one per group)", len(corpus) == 2)
    check("g2 chosen is the charter-clean one", any(p.chosen == "g2win" for p in corpus))


def main() -> int:
    for fn in (
        test_charter_hard_veto,
        test_pr_outcome_ordering,
        test_microbench_saturating,
        test_pair_margin,
        test_pair_too_close,
        test_pair_charter_vetoed_loser,
        test_pair_both_vetoed,
        test_corpus_basic,
        test_corpus_skips_singletons_and_vetoed_best,
        test_corpus_multi_group,
    ):
        fn()
    print(f"reward: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
