"""Pure-logic tests for adaptive (Track C × Track A difficulty-adaptive compute).

Standalone-runnable:
    python3 20-actors/shinka/cells/shinka_engine/test_adaptive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptive import DEFAULT_BUDGETS, adaptive_best_of_n  # noqa: E402
from fleet_sampler import FleetSampler  # noqa: E402
from matformer import Tier  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {name}")


EASY = "rename a variable"
HARD = "prove the distributed recursive invariant optimizing concurrent fleet cross-actor scheduling"


def test_easy_small_budget_e2b() -> None:
    r = adaptive_best_of_n(FleetSampler(), EASY)
    check("easy → E2B tier", r.tier is Tier.E2B)
    check("easy → small budget", r.k == DEFAULT_BUDGETS[Tier.E2B])
    check("samples == budget", len(r.result.samples) == r.k)
    check("winner present", r.winner is not None)


def test_hard_large_budget_e4b() -> None:
    r = adaptive_best_of_n(FleetSampler(), HARD)
    check("hard → E4B tier", r.tier is Tier.E4B)
    check("hard → large budget", r.k == DEFAULT_BUDGETS[Tier.E4B])
    check("more samples for hard task", len(r.result.samples) == DEFAULT_BUDGETS[Tier.E4B])


def test_hard_gets_more_compute_than_easy() -> None:
    easy = adaptive_best_of_n(FleetSampler(), EASY)
    hard = adaptive_best_of_n(FleetSampler(), HARD)
    check("hard spends more samples than easy", hard.k > easy.k)


def test_custom_budgets() -> None:
    r = adaptive_best_of_n(FleetSampler(), EASY, budgets={Tier.E2B: 2, Tier.E4B: 16})
    check("custom easy budget used", r.k == 2)
    check("custom budget sample count", len(r.result.samples) == 2)


def test_prior_fail_escalates_budget() -> None:
    # an easy task that failed earlier escalates to E4B → the larger budget
    r = adaptive_best_of_n(FleetSampler(), EASY, prior_fail=1.0)
    check("escalated to E4B", r.tier is Tier.E4B)
    check("escalation uses large budget", r.k == DEFAULT_BUDGETS[Tier.E4B])


def test_deterministic() -> None:
    a = adaptive_best_of_n(FleetSampler(), HARD)
    b = adaptive_best_of_n(FleetSampler(), HARD)
    check("deterministic winner", a.winner.text == b.winner.text)


def main() -> int:
    for fn in (
        test_easy_small_budget_e2b,
        test_hard_large_budget_e4b,
        test_hard_gets_more_compute_than_easy,
        test_custom_budgets,
        test_prior_fail_escalates_budget,
        test_deterministic,
    ):
        fn()
    print(f"adaptive: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
