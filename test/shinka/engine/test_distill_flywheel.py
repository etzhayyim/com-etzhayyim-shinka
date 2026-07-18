"""Pure-logic tests for distill_flywheel (Track F distillation flywheel + guards).

Standalone-runnable:
    python3 src/shinka/engine/test_distill_flywheel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "shinka" / "engine"))

from distill_flywheel import DistillFlywheel, projected_rounds  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {name}")


def test_projected_rounds_decreasing() -> None:
    rs = [projected_rounds(g) for g in range(6)]
    check("rounds non-increasing", all(a >= b for a, b in zip(rs, rs[1:])))
    check("floored at 1", min(rs) >= 1)
    check("gen0 = base 8", rs[0] == 8)


def test_healthy_convergence() -> None:
    fw = DistillFlywheel()
    # improving quality, healthy diversity, shrinking rounds
    fw.advance(rounds_needed=8, held_out_quality=0.60, diversity=0.7)
    fw.advance(rounds_needed=5, held_out_quality=0.66, diversity=0.6)
    fw.advance(rounds_needed=3, held_out_quality=0.70, diversity=0.5)
    check("all promoted", len(fw.promoted()) == 3)
    check("rounds trend shrinks", fw.rounds_trend() == [8, 5, 3])
    check("is_converging", fw.is_converging() is True)
    check("best_quality tracks max", abs(fw.best_quality - 0.70) < 1e-9)


def test_mode_collapse_halts() -> None:
    fw = DistillFlywheel(diversity_floor=0.3)
    fw.advance(8, 0.60, 0.7)               # promoted
    g = fw.advance(3, 0.72, 0.10)          # diversity below floor → halted
    check("collapsed gen halted", g.status == "halted")
    check("reason cites mode-collapse", "mode-collapse" in g.reason)
    check("not counted as promoted", len(fw.promoted()) == 1)
    check("best_quality unchanged by halted gen", abs(fw.best_quality - 0.60) < 1e-9)


def test_reward_hacking_halts() -> None:
    fw = DistillFlywheel(regression_tol=0.02)
    fw.advance(8, 0.70, 0.7)               # promoted, best=0.70
    g = fw.advance(2, 0.60, 0.7)           # held-out regressed → halted
    check("regressed gen halted", g.status == "halted")
    check("reason cites reward-hacking", "reward-hacking" in g.reason)
    check("best_quality preserved", abs(fw.best_quality - 0.70) < 1e-9)


def test_small_regression_within_tol_ok() -> None:
    fw = DistillFlywheel(regression_tol=0.05)
    fw.advance(8, 0.70, 0.7)
    g = fw.advance(5, 0.67, 0.7)           # within tol → still promoted
    check("within-tol promoted", g.status == "promoted")


def test_trend_only_promoted() -> None:
    fw = DistillFlywheel()
    fw.advance(8, 0.60, 0.7)               # promoted
    fw.advance(3, 0.72, 0.10)              # halted (collapse) — excluded from trend
    fw.advance(4, 0.65, 0.6)               # promoted
    check("trend skips halted gen", fw.rounds_trend() == [8, 4])


def main() -> int:
    for fn in (
        test_projected_rounds_decreasing,
        test_healthy_convergence,
        test_mode_collapse_halts,
        test_reward_hacking_halts,
        test_small_regression_within_tol_ok,
        test_trend_only_promoted,
    ):
        fn()
    print(f"distill_flywheel: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
