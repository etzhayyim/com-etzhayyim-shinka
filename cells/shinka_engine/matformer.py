"""matformer — Research Track C: MatFormer elastic inference (E2B / E4B routing).

Per ADR-2606142200 §Research Program Track C. Gemma 4 E4B nests an E2B submodel
(MatFormer / mix-n-match): the same weights serve a cheap ~2B-effective forward
pass (E2B) or the full ~4B pass (E4B). Right-size compute per task: route easy
turns to E2B, hard turns to E4B, and escalate to E4B after a failed E2B attempt.
This buys most of the quality at a fraction of the Joules — the metric is
quality/Joule per difficulty bucket.

E2B doubles as the cheap DRAFTER for Track B speculative decoding (the adaptive
drafter, TLT): the same nested submodel that serves easy turns drafts for the
E4B verifier — one weight, two roles.

Pure + Murakumo-only: routing is a local decision; the per-tier infer hooks
resolve to the fleet (E2B/E4B Ollama variants), not a commercial endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Tier(Enum):
    E2B = "E2B"  # nested cheap submodel (~2B effective)
    E4B = "E4B"  # full Maxwell (~4B effective)


# Relative compute / Joule per tier (E4B ≈ 2.2× E2B — the MatFormer cost gap).
TIER_COST: dict[Tier, float] = {Tier.E2B: 1.0, Tier.E4B: 2.2}

# Lexical signals that a task is hard enough to warrant the full E4B pass.
HARD_MARKERS: tuple[str, ...] = (
    "prove",
    "multi-node",
    "distributed",
    "concurren",
    "optimi",
    "rnea",
    "invariant",
    "cross-actor",
    "recursive",
    "elo tournament",
)

DEFAULT_THRESHOLD = 0.5


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def estimate_difficulty(task: str, prior_fail: float = 0.0) -> float:
    """Heuristic task difficulty in [0, 1] (length + hard-markers + prior failure).

    `prior_fail` (0..1) lets a failed E2B attempt escalate the next routing —
    the right-size-then-escalate policy.
    """
    words = len(task.split())
    length_term = min(words / 40.0, 1.0)
    kw = sum(1 for m in HARD_MARKERS if m in task.lower())
    kw_term = min(kw / 3.0, 1.0)
    # Hard-keyword presence dominates (a maxed kw_term alone clears the E4B
    # threshold); length nudges; a prior E2B failure adds escalation pressure.
    return _clamp(0.3 * length_term + 0.6 * kw_term + 0.3 * _clamp(prior_fail))


@dataclass
class RouteDecision:
    tier: Tier
    difficulty: float
    est_cost: float
    reason: str


def route(difficulty: float, threshold: float = DEFAULT_THRESHOLD) -> Tier:
    return Tier.E4B if difficulty >= threshold else Tier.E2B


@dataclass
class MatFormerRouter:
    """Routes tasks to the E2B/E4B tier and tracks cost vs an all-E4B baseline."""

    threshold: float = DEFAULT_THRESHOLD
    decisions: list[RouteDecision] = field(default_factory=list)

    def decide(self, task: str, prior_fail: float = 0.0) -> RouteDecision:
        d = estimate_difficulty(task, prior_fail)
        # A prior E2B failure is a hard escalation to E4B regardless of difficulty.
        escalated = prior_fail >= 0.5
        tier = Tier.E4B if (escalated or d >= self.threshold) else Tier.E2B
        dec = RouteDecision(
            tier=tier,
            difficulty=d,
            est_cost=TIER_COST[tier],
            reason=(
                "escalated to E4B after E2B failure"
                if escalated
                else f"difficulty {d:.2f} {'>=' if tier is Tier.E4B else '<'} {self.threshold:.2f}"
            ),
        )
        self.decisions.append(dec)
        return dec

    def escalate(self, task: str) -> RouteDecision:
        """Re-route a task that failed at E2B — forces E4B (prior_fail = 1.0)."""
        return self.decide(task, prior_fail=1.0)

    def cost_savings(self) -> float:
        """Fraction of compute saved vs running every decided task at E4B."""
        if not self.decisions:
            return 0.0
        routed = sum(d.est_cost for d in self.decisions)
        baseline = len(self.decisions) * TIER_COST[Tier.E4B]
        return 1.0 - routed / baseline
