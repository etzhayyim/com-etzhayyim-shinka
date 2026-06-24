"""speculative — Research Track B: speculative decoding + TLT adaptive drafter.

Per ADR-2606142200 §Research Program Track B. The MatFormer E2B submodel (Track C)
drafts a block of γ tokens; the full E4B Maxwell verifies them in one pass,
accepting the longest correct prefix plus one bonus token (Leviathan et al. 2023).
Expected speedup rises with the acceptance rate α and the draft/verify cost ratio.

The TLT refinement (MIT HAN Lab, arXiv 2511.16665): a static drafter goes stale as
Maxwell evolves (Loop B), collapsing α. `DrafterFreshness` tracks the weight-
generation lag and signals when to retrain the drafter on the fleet's idle cycles —
keeping the drafter aligned with the current Maxwell so α stays high.

Pure + Murakumo-only: the draft/verify callables resolve to fleet endpoints
(E2B/E4B Ollama variants); this module only does the accept/cost arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

try:
    from .matformer import TIER_COST, Tier
except Exception:  # pragma: no cover - standalone import path
    from matformer import TIER_COST, Tier

# Default draft/verify cost ratio = E2B / E4B (the MatFormer tier gap).
DEFAULT_COST_RATIO = TIER_COST[Tier.E2B] / TIER_COST[Tier.E4B]


def expected_tokens_per_step(alpha: float, gamma: int) -> float:
    """Expected tokens emitted per verification step (Leviathan et al. 2023).

    (1 - α^(γ+1)) / (1 - α), the geometric series of accepting up to γ drafted
    tokens plus the 1 guaranteed verifier token.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    if gamma < 1:
        raise ValueError(f"gamma must be >= 1, got {gamma}")
    if alpha == 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def speculative_speedup(
    alpha: float, gamma: int, cost_ratio: float = DEFAULT_COST_RATIO
) -> float:
    """Wall-clock speedup vs plain autoregressive decoding.

    expected_tokens / (γ·cost_ratio + 1): each block costs γ cheap drafts plus
    one full verify; > 1 means a net win.
    """
    return expected_tokens_per_step(alpha, gamma) / (gamma * cost_ratio + 1.0)


def verify_prefix(draft: list, target: list) -> int:
    """Number of drafted tokens accepted = length of the longest matching prefix."""
    n = 0
    for d, t in zip(draft, target):
        if d == t:
            n += 1
        else:
            break
    return n


@dataclass
class SpecDecodeResult:
    tokens_emitted: int
    verify_steps: int
    drafted: int
    accepted: int

    @property
    def accept_rate(self) -> float:
        return 0.0 if self.drafted == 0 else self.accepted / self.drafted

    @property
    def tokens_per_step(self) -> float:
        return 0.0 if self.verify_steps == 0 else self.tokens_emitted / self.verify_steps


def simulate_decode(
    target: list,
    draft_block: Callable[[list, int], list],
    gamma: int = 4,
) -> SpecDecodeResult:
    """Simulate speculative decoding of `target` with an oracle E4B verifier.

    `draft_block(prefix, gamma)` is the E2B drafter. The verifier (the target
    sequence) accepts the matching prefix and always emits one correction/bonus
    token. Returns emitted/verify-step/accept counts for the empirical speedup.
    """
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    i = 0
    emitted = 0
    steps = 0
    drafted = 0
    accepted = 0
    n = len(target)
    while i < n:
        block = list(draft_block(target[:i], gamma))[:gamma]
        drafted += len(block)
        k = verify_prefix(block, target[i:])
        accepted += k
        advance = min(k + 1, n - i)  # accepted prefix + 1 verifier token
        emitted += advance
        i += advance
        steps += 1
    return SpecDecodeResult(tokens_emitted=emitted, verify_steps=steps, drafted=drafted, accepted=accepted)


@dataclass
class DrafterFreshness:
    """TLT: track Maxwell-generation lag and signal when to retrain the drafter."""

    target_gen: int = 0   # Maxwell weight generation (bumped each Loop-B promote)
    drafter_gen: int = 0   # generation the drafter was last aligned to
    max_lag: int = 1       # tolerated lag before the drafter is considered stale

    def observe_target_update(self) -> None:
        self.target_gen += 1

    def retrain_drafter(self) -> None:
        """Align the drafter to the current Maxwell (runs on fleet idle cycles)."""
        self.drafter_gen = self.target_gen

    @property
    def lag(self) -> int:
        return self.target_gen - self.drafter_gen

    def is_stale(self) -> bool:
        return self.lag > self.max_lag
