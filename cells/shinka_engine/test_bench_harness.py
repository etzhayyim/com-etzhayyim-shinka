"""Pure-logic tests for bench_harness (Research Track A standing eval).

Standalone-runnable:
    python3 20-actors/shinka/cells/shinka_engine/test_bench_harness.py

Verifies the pass@k ceiling math (monotone in k), tournament-solve measurement,
the exploitation ratio, and input guards.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_harness import BenchHarness, BenchTask, KPoint  # noqa: E402
from fleet_sampler import FLEET_WORKER_NODES, FleetSampler  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {name}")


def _verify_correct(text: str) -> bool:
    return "CORRECT" in text


def _sampler_all(node_text: dict[str, str]) -> FleetSampler:
    return FleetSampler(infer_by_node={n: (lambda p, t=node_text[n]: t) for n in FLEET_WORKER_NODES})


def test_all_correct() -> None:
    sampler = _sampler_all({n: "CORRECT answer" for n in FLEET_WORKER_NODES})
    tasks = [BenchTask("t1", "p1", _verify_correct), BenchTask("t2", "p2", _verify_correct)]
    rep = BenchHarness(sampler, tasks).run(ks=(1, 2, 5, 10))
    check("all ceilings 1.0", all(p.pass_at_k == 1.0 for p in rep.points))
    check("all tournament-solve 1.0", all(p.tournament_solve == 1.0 for p in rep.points))
    check("exploitation 1.0", all(p.exploitation == 1.0 for p in rep.points))


def test_none_correct() -> None:
    sampler = _sampler_all({n: "wrong answer" for n in FLEET_WORKER_NODES})
    tasks = [BenchTask("t1", "p1", _verify_correct)]
    rep = BenchHarness(sampler, tasks).run(ks=(1, 2, 5, 10))
    check("all ceilings 0", all(p.pass_at_k == 0.0 for p in rep.points))
    check("all solve 0", all(p.tournament_solve == 0.0 for p in rep.points))
    check("exploitation 0 when ceiling 0", rep.point(10).exploitation == 0.0)


def test_partial_monotone() -> None:
    # exactly one node (dan) correct → c=1 out of n=10 → pass@k strictly increases.
    texts = {n: "wrong" for n in FLEET_WORKER_NODES}
    texts["dan"] = "CORRECT"
    sampler = _sampler_all(texts)
    rep = BenchHarness(sampler, [BenchTask("t", "p", _verify_correct)]).run(ks=(1, 2, 5, 10), n=10)
    ceils = [rep.point(k).pass_at_k for k in (1, 2, 5, 10)]
    check("ceiling strictly increasing in k", all(a < b for a, b in zip(ceils, ceils[1:])))
    check("pass@1 ~ 0.1", abs(rep.point(1).pass_at_k - 0.1) < 1e-9)
    check("pass@10 == 1.0 (c>=1, k==n)", rep.point(10).pass_at_k == 1.0)
    check("all values in [0,1]", all(0.0 <= p.pass_at_k <= 1.0 for p in rep.points))


def test_summary_and_report() -> None:
    sampler = _sampler_all({n: "CORRECT" for n in FLEET_WORKER_NODES})
    rep = BenchHarness(sampler, [BenchTask("t", "p", _verify_correct)]).run(ks=(1, 10))
    check("summary mentions k=10", "k=10" in rep.summary())
    check("report n_tasks", rep.n_tasks == 1)
    check("report has 2 points", len(rep.points) == 2)


def test_exploitation_property() -> None:
    kp = KPoint(k=5, pass_at_k=0.8, tournament_solve=0.4)
    check("exploitation = solve/ceiling", abs(kp.exploitation - 0.5) < 1e-9)


def test_guards() -> None:
    sampler = _sampler_all({n: "x" for n in FLEET_WORKER_NODES})
    empty_raised = False
    try:
        BenchHarness(sampler, []).run()
    except ValueError:
        empty_raised = True
    check("empty task set raises", empty_raised)

    kgt_raised = False
    try:
        BenchHarness(sampler, [BenchTask("t", "p", _verify_correct)]).run(ks=(1, 20), n=10)
    except ValueError:
        kgt_raised = True
    check("k>n raises", kgt_raised)


def main() -> int:
    for fn in (
        test_all_correct,
        test_none_correct,
        test_partial_monotone,
        test_summary_and_report,
        test_exploitation_property,
        test_guards,
    ):
        fn()
    print(f"bench_harness: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
