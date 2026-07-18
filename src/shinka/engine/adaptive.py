"""adaptive — difficulty-adaptive fleet test-time compute (Track C × Track A).

Per ADR-2606142200. Composes the MatFormer router (Track C) with the fleet
best-of-N sampler (Track A): route each task by difficulty and SIZE the test-time
compute to it — easy tasks draw a small sample budget at the cheap E2B tier, hard
tasks draw a larger budget at the full E4B tier. This is the right-size-compute
policy made operational: spend fleet samples where they buy quality, not uniformly.

Pure + Murakumo-only: the router decision is local; the sampler resolves to fleet
endpoints (or the deterministic kernel offline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from .matformer import MatFormerRouter, RouteDecision, Tier
except Exception:  # pragma: no cover - standalone import path
    from matformer import MatFormerRouter, RouteDecision, Tier


# Sample budget per tier: hard tasks (E4B) earn more fleet draws than easy (E2B).
DEFAULT_BUDGETS: dict[Tier, int] = {Tier.E2B: 4, Tier.E4B: 10}


@dataclass
class AdaptiveResult:
    decision: RouteDecision   # the MatFormer tier routing for this task
    k: int                    # sample budget actually used
    result: Any               # the FleetSampler BestOfNResult

    @property
    def winner(self):
        return self.result.winner

    @property
    def tier(self) -> Tier:
        return self.decision.tier


def adaptive_best_of_n(
    sampler: Any,
    prompt: str,
    router: MatFormerRouter | None = None,
    budgets: dict[Tier, int] | None = None,
    prior_fail: float = 0.0,
) -> AdaptiveResult:
    """Route `prompt` by difficulty, then run fleet best-of-N at the tier's budget.

    `sampler` is a FleetSampler (duck-typed: needs `.best_of_n(prompt, k)`).
    `prior_fail` lets a failed earlier attempt escalate to the E4B (larger) budget.
    """
    router = router or MatFormerRouter()
    budgets = budgets or DEFAULT_BUDGETS
    decision = router.decide(prompt, prior_fail=prior_fail)
    k = budgets[decision.tier]
    result = sampler.best_of_n(prompt, k)
    return AdaptiveResult(decision=decision, k=k, result=result)
