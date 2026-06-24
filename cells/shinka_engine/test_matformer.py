"""Pure-logic tests for matformer (Research Track C elastic E2B/E4B routing).

Standalone-runnable:
    python3 20-actors/shinka/cells/shinka_engine/test_matformer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matformer import (  # noqa: E402
    TIER_COST,
    MatFormerRouter,
    Tier,
    estimate_difficulty,
    route,
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


def test_difficulty_range() -> None:
    d = estimate_difficulty("fix typo")
    check("short simple task is easy", d < 0.3)
    hard = estimate_difficulty(
        "prove the distributed invariant for the recursive multi-node elo tournament "
        "while optimizing concurrent cross-actor RNEA scheduling across the whole fleet"
    )
    check("long hard task is hard", hard >= 0.5)
    check("difficulty in [0,1]", 0.0 <= d <= 1.0 and 0.0 <= hard <= 1.0)


def test_prior_fail_escalates() -> None:
    base = estimate_difficulty("add a field")
    esc = estimate_difficulty("add a field", prior_fail=1.0)
    check("prior failure raises difficulty", esc > base)


def test_route() -> None:
    check("low → E2B", route(0.2) is Tier.E2B)
    check("high → E4B", route(0.8) is Tier.E4B)
    check("at threshold → E4B", route(0.5) is Tier.E4B)


def test_router_decide() -> None:
    r = MatFormerRouter()
    easy = r.decide("rename var")
    check("easy routed to E2B", easy.tier is Tier.E2B)
    check("E2B cost cheaper", easy.est_cost == TIER_COST[Tier.E2B])
    hard = r.decide("prove the distributed recursive invariant optimizing concurrent fleet scheduling")
    check("hard routed to E4B", hard.tier is Tier.E4B)
    check("reason mentions difficulty", "difficulty" in hard.reason)


def test_escalate() -> None:
    r = MatFormerRouter()
    # an easy task that failed at E2B → escalate forces E4B
    dec = r.escalate("add a field")
    check("escalation forces E4B", dec.tier is Tier.E4B)
    check("reason notes escalation", "escalated" in dec.reason)


def test_cost_savings() -> None:
    r = MatFormerRouter()
    # 3 easy + 1 hard → most go E2B → meaningful savings vs all-E4B
    for t in ("rename x", "tidy import", "fix typo"):
        r.decide(t)
    r.decide("prove distributed recursive invariant optimizing concurrent fleet cross-actor")
    sav = r.cost_savings()
    check("positive savings vs all-E4B", sav > 0.0)
    check("savings < 1", sav < 1.0)
    # all-hard → no savings
    r2 = MatFormerRouter()
    r2.decide("prove distributed recursive invariant optimizing concurrent fleet cross-actor rnea")
    check("all-E4B → ~0 savings", abs(r2.cost_savings()) < 1e-9)
    check("empty router → 0 savings", MatFormerRouter().cost_savings() == 0.0)


def main() -> int:
    for fn in (
        test_difficulty_range,
        test_prior_fail_escalates,
        test_route,
        test_router_decide,
        test_escalate,
        test_cost_savings,
    ):
        fn()
    print(f"matformer: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
