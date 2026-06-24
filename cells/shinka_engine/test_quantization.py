"""Pure-logic tests for quantization (Track G fleet quantization frontier).

Standalone-runnable:
    python3 20-actors/shinka/cells/shinka_engine/test_quantization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantization import (  # noqa: E402
    DEFAULT_E4B_QUANTS,
    NodeSpec,
    QuantOption,
    fits,
    pareto_front,
    recommend,
    select_quant,
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


def _names(opts):
    return {o.name for o in opts}


def test_score_and_fits() -> None:
    o = QuantOption("x", 4.0, 2.5, 50.0, 0.9)
    check("score = tok_s × quality", abs(o.score - 45.0) < 1e-9)
    check("fits within budget", fits(o, 3.0) is True)
    check("does not fit over budget", fits(o, 2.0) is False)


def test_pareto_excludes_dominated() -> None:
    front = pareto_front(DEFAULT_E4B_QUANTS)
    check("Q4_0 (dominated) excluded", "Q4_0" not in _names(front))
    check("MLX-4bit (fastest) on front", "MLX-4bit" in _names(front))
    check("FP16 (best quality) on front", "FP16" in _names(front))
    check("Q8_0 on front", "Q8_0" in _names(front))


def test_select_best_score() -> None:
    # large budget → highest tok/s × quality = MLX-4bit (72 × 0.96 = 69.1)
    best = select_quant(DEFAULT_E4B_QUANTS, mem_budget_gb=32.0)
    check("big budget → MLX-4bit", best.name == "MLX-4bit")


def test_select_tight_budget() -> None:
    # 2.5 GB → only MLX-4bit (2.4) fits among the small quants
    best = select_quant(DEFAULT_E4B_QUANTS, mem_budget_gb=2.5)
    check("tight budget → MLX-4bit", best.name == "MLX-4bit")
    # 2.0 GB → nothing fits
    check("too tight → None", select_quant(DEFAULT_E4B_QUANTS, 2.0) is None)


def test_recommend_per_node() -> None:
    # Mac mini M4: ~6 GB available after OS/ComfyUI; EVO-X2: 32 GB ROCm.
    mac = NodeSpec("dan", unified_mem_gb=6.0, backend="metal")
    evo = NodeSpec("evo-x2", unified_mem_gb=32.0, backend="rocm")
    rmac = recommend(mac)
    revo = recommend(evo)
    check("mac recommendation fits its budget", rmac.mem_gb <= 6.0)
    check("evo recommendation fits its budget", revo.mem_gb <= 32.0)
    check("both pick best-score MLX-4bit", rmac.name == "MLX-4bit" and revo.name == "MLX-4bit")
    # a memory-starved node still gets a fitting (smaller) quant or None
    starved = NodeSpec("tiny", unified_mem_gb=2.45, backend="metal")
    rs = recommend(starved)
    check("starved node → still MLX-4bit (2.4 fits)", rs.name == "MLX-4bit")
    check("sub-2.4 node → None", recommend(NodeSpec("nano", 2.0, "metal")) is None)


def main() -> int:
    for fn in (
        test_score_and_fits,
        test_pareto_excludes_dominated,
        test_select_best_score,
        test_select_tight_budget,
        test_recommend_per_node,
    ):
        fn()
    print(f"quantization: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
