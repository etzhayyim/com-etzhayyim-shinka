"""fleet_sampler — Research Track A: fleet test-time compute (best-of-N + Elo).

Per ADR-2606142200 §Research Program Track A. The murakumo fleet's worker nodes
are a natural test-time-compute substrate: draw N candidate completions in
parallel across the nodes (sampling diversity = different nodes/seeds), then
select the winner via an Elo pairwise-debate tournament (the same ranking
mechanism as the Loop-A `tournament` cell). The thesis: k≈10 Maxwell samples,
tournament-selected, approach a single frontier forward pass on the org's tasks.

This is the engine behind "Elo ペア討論を10ノード並列": `best_of_n` fans the
prompt across `FLEET_WORKER_NODES`, then runs a round-robin Elo tournament over
the samples to pick the winner.

Invariants:
  I3  Murakumo-only — `infer_by_node` resolves to per-node fleet endpoints
      (Ollama on each Mac mini); a node that errors is skipped (graceful
      degrade), and with no live infer the deterministic kernel drives both
      sampling and judging so tests run offline. Never a commercial call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Callable

try:  # reuse Loop-A Elo + deterministic score proxy
    from .cell import _stable_score, elo_update
except Exception:  # pragma: no cover - standalone import path
    from cell import _stable_score, elo_update


# The 10 murakumo worker nodes (Mac mini M4) per 50-infra/murakumo/fleet.edn.
# `judah` runs the LiteLLM gateway (no local Ollama generation) and is excluded
# from the generation roster; the other 9 each serve a local Ollama Maxwell.
FLEET_WORKER_NODES: tuple[str, ...] = (
    "zebulun",
    "issachar",
    "dan",
    "benjamin",
    "joseph",
    "levi",
    "naphtali",
    "simeon",
    "asher",
)


@dataclass
class Sample:
    node: str
    text: str
    elo: float = 1200.0


@dataclass
class BestOfNResult:
    winner: Sample | None
    samples: list[Sample] = field(default_factory=list)
    nodes_used: list[str] = field(default_factory=list)
    debates: int = 0


class FleetSampler:
    """Fan a prompt across fleet nodes, then Elo-tournament-select the winner.

    `infer_by_node`: {node_name: callable(prompt)->completion}. Missing/erroring
    nodes are skipped (I3 graceful degrade). `judge`: callable(a_text,b_text)->
    bool (True iff a wins); defaults to the deterministic kernel.
    """

    def __init__(
        self,
        infer_by_node: dict[str, Callable[[str], str]] | None = None,
        judge: Callable[[str, str], bool] | None = None,
        nodes: tuple[str, ...] = FLEET_WORKER_NODES,
    ) -> None:
        self.infer_by_node = infer_by_node or {}
        self.judge = judge or self._kernel_judge
        self.nodes = nodes

    @staticmethod
    def _kernel_judge(a: str, b: str) -> bool:
        sa, sb = _stable_score(a), _stable_score(b)
        if sa != sb:
            return sa > sb
        return a >= b  # stable tiebreak

    def sample(self, prompt: str, n: int) -> list[Sample]:
        """Draw n completions, round-robin across the fleet (skip failed nodes)."""
        samples: list[Sample] = []
        i = 0
        attempts = 0
        # Bound retries while still allowing n draws from a single healthy node
        # (round-robin may need to cycle back to it n times).
        max_attempts = n * len(self.nodes)
        while len(samples) < n and attempts < max_attempts:
            node = self.nodes[i % len(self.nodes)]
            i += 1
            attempts += 1
            infer = self.infer_by_node.get(node)
            if infer is not None:
                try:
                    text = infer(prompt)
                except Exception:
                    continue  # I3: node error → skip, try the next node
            else:
                # Deterministic kernel: a distinct completion per (node, draw).
                text = f"[{node}#{len(samples)}] completion for: {prompt}"
            samples.append(Sample(node=node, text=text))
        return samples

    def tournament(self, samples: list[Sample]) -> tuple[Sample | None, int]:
        """Round-robin Elo tournament (pairwise debate) over the samples."""
        debates = 0
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                a, b = samples[i], samples[j]
                a_won = self.judge(a.text, b.text)
                a.elo, b.elo = elo_update(a.elo, b.elo, a_won)
                debates += 1
        if not samples:
            return None, 0
        winner = max(samples, key=lambda s: s.elo)
        return winner, debates

    def best_of_n(self, prompt: str, n: int = 10) -> BestOfNResult:
        samples = self.sample(prompt, n)
        winner, debates = self.tournament(samples)
        return BestOfNResult(
            winner=winner,
            samples=samples,
            nodes_used=sorted({s.node for s in samples}),
            debates=debates,
        )


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021): 1 - C(n-c,k)/C(n,k).

    n = total samples, c = number correct, k = budget. Used to quantify Track A's
    "pass@k vs k" metric — how fleet sample budget trades into solve rate.
    """
    if k > n:
        raise ValueError(f"k ({k}) must be <= n ({n})")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)
