"""quantization — Research Track G: fleet quantization frontier (tok/s × quality).

Per ADR-2606142200 §Research Program Track G. Each fleet node has a fixed unified-
memory budget (Mac mini M4 = Metal/MLX; EVO-X2 = ROCm). Pick the Maxwell quant per
node that maximises tok/s × quality among the quants that FIT the budget — the
per-node tok/s × quality Pareto front.

Pure + stdlib; the throughput/quality figures are S0 placeholder estimates for the
Gemma 4 E4B family (replaced by measured `e7m bench` + tok/s numbers at S1).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantOption:
    name: str
    bits: float        # effective bits/weight
    mem_gb: float      # model footprint (unified memory)
    tok_s: float       # throughput estimate (tokens/sec on the node class)
    quality: float     # quality retention vs fp16 (0..1)

    @property
    def score(self) -> float:
        """The Track-G objective: tok/s × quality."""
        return self.tok_s * self.quality


# S0 placeholder Gemma 4 E4B quant menu (Metal/MLX-leaning). Q4_0 is intentionally
# dominated (lower tok_s AND quality than Q4_K_M) to exercise the Pareto filter.
DEFAULT_E4B_QUANTS: tuple[QuantOption, ...] = (
    QuantOption("MLX-4bit", 4.0, 2.4, 72.0, 0.960),
    QuantOption("Q4_K_M", 4.5, 2.6, 60.0, 0.965),
    QuantOption("Q4_0", 4.0, 2.6, 55.0, 0.950),    # dominated
    QuantOption("Q5_K_M", 5.5, 3.1, 50.0, 0.982),
    QuantOption("Q8_0", 8.0, 4.7, 32.0, 0.997),
    QuantOption("FP16", 16.0, 8.5, 18.0, 1.000),
)


@dataclass(frozen=True)
class NodeSpec:
    name: str
    unified_mem_gb: float   # memory available to the model (after OS/other services)
    backend: str            # "metal" | "rocm"


def fits(opt: QuantOption, mem_budget_gb: float) -> bool:
    return opt.mem_gb <= mem_budget_gb


def pareto_front(options: tuple[QuantOption, ...] | list[QuantOption]) -> list[QuantOption]:
    """Non-dominated quants on (tok_s ↑, quality ↑)."""
    front: list[QuantOption] = []
    for a in options:
        dominated = any(
            b is not a
            and b.tok_s >= a.tok_s
            and b.quality >= a.quality
            and (b.tok_s > a.tok_s or b.quality > a.quality)
            for b in options
        )
        if not dominated:
            front.append(a)
    return front


def select_quant(
    options: tuple[QuantOption, ...] | list[QuantOption], mem_budget_gb: float
) -> QuantOption | None:
    """Best tok/s × quality among quants that fit; None if none fit."""
    feasible = [o for o in options if fits(o, mem_budget_gb)]
    if not feasible:
        return None
    return max(feasible, key=lambda o: (o.score, o.quality, -o.mem_gb))


def recommend(
    node: NodeSpec, options: tuple[QuantOption, ...] | list[QuantOption] = DEFAULT_E4B_QUANTS
) -> QuantOption | None:
    """Per-node recommendation: the best-scoring quant that fits the node's memory."""
    return select_quant(options, node.unified_mem_gb)
