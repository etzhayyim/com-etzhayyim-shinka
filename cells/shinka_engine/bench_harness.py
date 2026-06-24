"""bench_harness — Research Track A standing evaluation (pass@k vs k).

Per ADR-2606142200 §Research Program (S1: "extend e7m bench micro into a standing
harness that scores orchestrated-Maxwell against the frontier snapshot"). Measures
the frontier-efficiency thesis on a task set:

  * pass@k (ceiling): given n fleet samples per task with c correct, the unbiased
    pass@k estimator — the probability that k random draws contain a correct one.
    This is the raw test-time-compute ceiling (no selection).
  * tournament-solve@k: draw k samples and let the Elo tournament (fleet_sampler
    best_of_n) PICK one winner; measure how often the winner is correct. This is
    what the engine actually ships — the gap to the pass@k ceiling shows how well
    the tournament exploits the samples.

Hypothesis under test (ADR): tournament-solve@(k≈10) on Maxwell approaches a
single frontier forward pass on the org's task distribution. Each run is a fact
that can be appended to the Datom log (caller's choice) — the harness itself is
pure and Murakumo-only (it only drives the provided fleet sampler).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

try:
    from .fleet_sampler import FleetSampler, pass_at_k
except Exception:  # pragma: no cover - standalone import path
    from fleet_sampler import FleetSampler, pass_at_k


@dataclass
class BenchTask:
    name: str
    prompt: str
    verify: Callable[[str], bool]  # verifiable check on a completion (e7m-bench style)


@dataclass
class KPoint:
    k: int
    pass_at_k: float          # raw-sampling ceiling, averaged over tasks
    tournament_solve: float   # best_of_n winner correct rate, averaged over tasks

    @property
    def exploitation(self) -> float:
        """Fraction of the pass@k ceiling the tournament actually captures."""
        return 0.0 if self.pass_at_k == 0 else self.tournament_solve / self.pass_at_k


@dataclass
class BenchReport:
    ks: list[int]
    n: int
    n_tasks: int
    points: list[KPoint] = field(default_factory=list)

    def point(self, k: int) -> KPoint:
        return next(p for p in self.points if p.k == k)

    def summary(self) -> str:
        rows = ", ".join(
            f"k={p.k}: ceil={p.pass_at_k:.2f} sel={p.tournament_solve:.2f}"
            for p in self.points
        )
        return f"bench[{self.n_tasks} tasks, n={self.n}] {rows}"


class BenchHarness:
    """Drive a fleet sampler over a task set and compute the pass@k vs k curve."""

    def __init__(self, sampler: FleetSampler, tasks: list[BenchTask]) -> None:
        self.sampler = sampler
        self.tasks = tasks

    def run(self, ks: tuple[int, ...] = (1, 2, 5, 10), n: int | None = None) -> BenchReport:
        if not self.tasks:
            raise ValueError("bench task set is empty")
        n = n or max(ks)
        if any(k > n for k in ks):
            raise ValueError(f"every k must be <= n ({n})")

        # Per task: one n-sample draw → correct count c (drives the pass@k ceiling).
        ceiling_c: list[tuple[int, int]] = []  # (n_drawn, c)
        for t in self.tasks:
            samples = self.sampler.sample(t.prompt, n)
            c = sum(1 for s in samples if t.verify(s.text))
            ceiling_c.append((len(samples), c))

        report = BenchReport(ks=list(ks), n=n, n_tasks=len(self.tasks))
        for k in ks:
            # pass@k ceiling, averaged over tasks (skip tasks that drew < k samples).
            ceil_vals = [
                pass_at_k(nd, c, k) for (nd, c) in ceiling_c if nd >= k
            ]
            ceil = sum(ceil_vals) / len(ceil_vals) if ceil_vals else 0.0
            # tournament-solve@k: best_of_n picks one winner; is it correct?
            solved = 0
            for t in self.tasks:
                res = self.sampler.best_of_n(t.prompt, k)
                if res.winner is not None and t.verify(res.winner.text):
                    solved += 1
            report.points.append(
                KPoint(k=k, pass_at_k=ceil, tournament_solve=solved / len(self.tasks))
            )
        return report
