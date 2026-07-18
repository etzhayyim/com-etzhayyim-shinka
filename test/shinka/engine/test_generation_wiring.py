"""Integration test: Track-F distillation generations wired into the orchestrator.

Standalone-runnable:
    python3 src/shinka/engine/test_generation_wiring.py

A generation = the beats accumulated since the last close, scored on a held-out
quality + diversity (from the eval harness). The flywheel guards halt a
mode-collapsing / reward-hacking generation; rounds-to-quality should shrink.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "shinka" / "engine"))

from orchestrator import ShinkaOrchestrator  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {name}")


def test_generation_rounds_count() -> None:
    orch = ShinkaOrchestrator()
    for _ in range(3):
        orch.beat("task")
    gen = orch.close_generation(held_out_quality=0.6, diversity=0.7)
    check("gen 1", gen.gen == 1)
    check("rounds == 3 beats", gen.rounds_needed == 3)
    check("promoted", gen.status == "promoted")
    # next generation counts only its own beats
    orch.beat("task")
    gen2 = orch.close_generation(held_out_quality=0.66, diversity=0.6)
    check("gen 2 rounds == 1", gen2.rounds_needed == 1)


def test_generation_emits_datoms() -> None:
    orch = ShinkaOrchestrator()
    orch.beat("t")
    head_before = orch.head_cid
    gen = orch.close_generation(0.7, 0.7)
    check("head advanced on close", orch.head_cid != head_before)
    gen_datoms = [d for d in orch.log if d["e"] == f"shinka:generation/{gen.gen}"]
    check("generation datoms emitted", len(gen_datoms) >= 4)
    check("all :db/add (I1)", all(d["op"] == ":db/add" for d in gen_datoms))
    check("status datom present", any(d["a"] == ":gen/status" for d in gen_datoms))


def test_mode_collapse_halts_generation() -> None:
    orch = ShinkaOrchestrator()
    orch.beat("t")
    orch.close_generation(0.6, 0.7)            # promoted (best=0.6)
    orch.beat("t")
    gen = orch.close_generation(0.72, 0.10)    # diversity below floor → halted
    check("collapse halts generation", gen.status == "halted")
    check("reason cites mode-collapse", "mode-collapse" in gen.reason)


def test_reward_hacking_halts_generation() -> None:
    orch = ShinkaOrchestrator()
    orch.beat("t")
    orch.close_generation(0.70, 0.7)           # promoted, best=0.70
    orch.beat("t")
    gen = orch.close_generation(0.60, 0.7)     # held-out regressed → halted
    check("regression halts generation", gen.status == "halted")
    check("reason cites reward-hacking", "reward-hacking" in gen.reason)


def test_convergence_trend() -> None:
    orch = ShinkaOrchestrator()
    # shrinking rounds across promoted generations: 3 → 2 → 1 beats
    for _ in range(3):
        orch.beat("t")
    orch.close_generation(0.60, 0.7)
    for _ in range(2):
        orch.beat("t")
    orch.close_generation(0.66, 0.7)
    orch.beat("t")
    orch.close_generation(0.70, 0.7)
    check("rounds-to-quality converging", orch.is_converging() is True)


def main() -> int:
    for fn in (
        test_generation_rounds_count,
        test_generation_emits_datoms,
        test_mode_collapse_halts_generation,
        test_reward_hacking_halts_generation,
        test_convergence_trend,
    ):
        fn()
    print(f"generation_wiring: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
