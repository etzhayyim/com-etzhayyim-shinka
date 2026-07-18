"""distill_flywheel — Research Track F: the distillation flywheel (the core).

Per ADR-2606142200 §Research Program Track F. The recursive self-improvement:
SFT Maxwell on Loop-A tournament-winner traces so the ORCHESTRATION collapses
into the weights — each generation needs fewer debate rounds to reach the same
quality. The metric is rounds-to-quality over generations (should decrease).

The ADR's explicit hazard: "watch for and gate against reward-hacking / mode
collapse." This module is therefore as much a GUARD as a tracker:
  * mode-collapse → proposal diversity falls below a floor (the engine stops
    exploring and parrots a single mode).
  * reward-hacking → held-out quality regresses while the in-loop metric improves
    (the weight games the training signal).
A generation that trips either guard is HALTED (not promoted) — distillation
never advances on a degenerate generation.

Pure + Murakumo-only (it scores generations; the actual SFT runs in Loop B / on
EVO-X2, operator/leash-gated).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Generation:
    gen: int
    rounds_needed: int        # debate rounds to reach the quality target this gen
    held_out_quality: float   # quality on a HELD-OUT eval (not the training signal)
    diversity: float          # proposal diversity (0..1); low ⇒ mode collapse
    status: str               # "promoted" | "halted"
    reason: str


def projected_rounds(gen: int, base: int = 8, decay: float = 0.7, floor: int = 1) -> int:
    """Healthy-case rounds-to-quality model: base·decay^gen, floored (>=1)."""
    return max(floor, round(base * (decay ** gen)))


@dataclass
class DistillFlywheel:
    """Tracks generations and gates promotion on the collapse / reward-hacking guards."""

    diversity_floor: float = 0.3      # mode-collapse threshold
    regression_tol: float = 0.02      # tolerated held-out drop vs best-so-far
    generations: list[Generation] = field(default_factory=list)
    best_quality: float = 0.0         # best held-out quality among PROMOTED gens

    def advance(self, rounds_needed: int, held_out_quality: float, diversity: float) -> Generation:
        reasons: list[str] = []
        if diversity < self.diversity_floor:
            reasons.append(f"mode-collapse: diversity {diversity:.2f} < floor {self.diversity_floor:.2f}")
        if held_out_quality < self.best_quality - self.regression_tol:
            reasons.append(
                f"reward-hacking: held-out {held_out_quality:.2f} regressed vs best {self.best_quality:.2f}"
            )
        if reasons:
            status = "halted"
        else:
            status = "promoted"
            self.best_quality = max(self.best_quality, held_out_quality)
        g = Generation(
            gen=len(self.generations) + 1,
            rounds_needed=rounds_needed,
            held_out_quality=held_out_quality,
            diversity=diversity,
            status=status,
            reason="; ".join(reasons) if reasons else "ok",
        )
        self.generations.append(g)
        return g

    def promoted(self) -> list[Generation]:
        return [g for g in self.generations if g.status == "promoted"]

    def rounds_trend(self) -> list[int]:
        """Rounds-to-quality across PROMOTED generations (should be non-increasing)."""
        return [g.rounds_needed for g in self.promoted()]

    def is_converging(self) -> bool:
        """True iff rounds-to-quality is non-increasing over promoted generations."""
        trend = self.rounds_trend()
        return all(a >= b for a, b in zip(trend, trend[1:]))
