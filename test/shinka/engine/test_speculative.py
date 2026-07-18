"""Pure-logic tests for speculative (Track B speculative decoding + TLT drafter).

Standalone-runnable:
    python3 src/shinka/engine/test_speculative.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "shinka" / "engine"))

from speculative import (  # noqa: E402
    DEFAULT_COST_RATIO,
    DrafterFreshness,
    expected_tokens_per_step,
    simulate_decode,
    speculative_speedup,
    verify_prefix,
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


def test_expected_tokens() -> None:
    check("alpha=1 → gamma+1", expected_tokens_per_step(1.0, 4) == 5.0)
    check("alpha=0 → 1", expected_tokens_per_step(0.0, 4) == 1.0)
    check("alpha=0.5,g=4 ≈ 1.9375", abs(expected_tokens_per_step(0.5, 4) - 1.9375) < 1e-9)
    check("monotone in alpha", expected_tokens_per_step(0.8, 4) > expected_tokens_per_step(0.3, 4))


def test_expected_guards() -> None:
    bad_a = bad_g = False
    try:
        expected_tokens_per_step(1.5, 4)
    except ValueError:
        bad_a = True
    try:
        expected_tokens_per_step(0.5, 0)
    except ValueError:
        bad_g = True
    check("alpha out of range raises", bad_a)
    check("gamma < 1 raises", bad_g)


def test_speedup() -> None:
    check("default cost ratio = E2B/E4B", abs(DEFAULT_COST_RATIO - 1.0 / 2.2) < 1e-9)
    # high acceptance + cheap drafter → net speedup > 1
    check("high alpha speeds up", speculative_speedup(0.9, 4) > 1.0)
    # zero acceptance → no win (< 1)
    check("zero alpha no win", speculative_speedup(0.0, 4) < 1.0)
    check("monotone in alpha", speculative_speedup(0.9, 4) > speculative_speedup(0.4, 4))


def test_verify_prefix() -> None:
    check("full match", verify_prefix([1, 2, 3], [1, 2, 3, 4]) == 3)
    check("partial", verify_prefix([1, 2, 9], [1, 2, 3, 4]) == 2)
    check("no match", verify_prefix([9], [1, 2]) == 0)


def test_simulate_perfect_drafter() -> None:
    target = list(range(20))
    # perfect drafter: predicts the true continuation
    def perfect(prefix, g):
        i = len(prefix)
        return target[i : i + g]

    res = simulate_decode(target, perfect, gamma=4)
    check("emitted == len(target)", res.tokens_emitted == 20)
    check("perfect accept_rate ~1", res.accept_rate > 0.95)
    check("few verify steps (≈ n/(g+1))", res.verify_steps <= 5)
    check("tokens_per_step high", res.tokens_per_step >= 4.0)


def test_simulate_broken_drafter() -> None:
    target = list(range(20))
    def broken(prefix, g):
        return [-1] * g  # never matches

    res = simulate_decode(target, broken, gamma=4)
    check("still emits full target", res.tokens_emitted == 20)
    check("broken accept_rate 0", res.accept_rate == 0.0)
    check("one token per step", res.tokens_per_step == 1.0)
    check("steps == n (no acceleration)", res.verify_steps == 20)


def test_drafter_freshness_tlt() -> None:
    f = DrafterFreshness(max_lag=1)
    check("fresh initially", f.is_stale() is False)
    f.observe_target_update()
    check("lag 1 not yet stale", f.is_stale() is False)
    f.observe_target_update()
    check("lag 2 stale", f.is_stale() is True)
    f.retrain_drafter()  # retrain on idle fleet cycles
    check("retrain clears staleness", f.is_stale() is False)
    check("lag 0 after retrain", f.lag == 0)


def main() -> int:
    for fn in (
        test_expected_tokens,
        test_expected_guards,
        test_speedup,
        test_verify_prefix,
        test_simulate_perfect_drafter,
        test_simulate_broken_drafter,
        test_drafter_freshness_tlt,
    ):
        fn()
    print(f"speculative: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
