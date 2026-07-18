"""Pure-logic tests for fleet_sampler (Research Track A).

Standalone-runnable:
    python3 src/shinka/engine/test_fleet_sampler.py

Covers best-of-N fan-out across fleet nodes, the Elo tournament selection,
node-failure graceful degrade (I3), live-infer routing, and the pass@k estimator.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "shinka" / "engine"))

from fleet_sampler import (  # noqa: E402
    FLEET_WORKER_NODES,
    FleetSampler,
    pass_at_k,
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


def test_roster() -> None:
    check("9 generation nodes (judah excluded)", len(FLEET_WORKER_NODES) == 9)
    check("judah not in generation roster", "judah" not in FLEET_WORKER_NODES)


def test_best_of_n_kernel() -> None:
    fs = FleetSampler()
    res = fs.best_of_n("solve task X", n=10)
    check("drew 10 samples", len(res.samples) == 10)
    check("winner selected", res.winner is not None)
    check("winner has max elo", res.winner.elo == max(s.elo for s in res.samples))
    check("round-robin debates = C(10,2)", res.debates == 45)
    check("fanned across multiple nodes", len(res.nodes_used) >= 2)


def test_live_infer_routing() -> None:
    calls = {"n": 0}

    def infer(prompt: str) -> str:
        calls["n"] += 1
        return f"answer:{calls['n']}"

    fs = FleetSampler(infer_by_node={n: infer for n in FLEET_WORKER_NODES})
    res = fs.best_of_n("p", n=5)
    check("live infer called per sample", calls["n"] == 5)
    check("5 samples", len(res.samples) == 5)


def test_node_failure_degrade() -> None:
    # Only one healthy node; the rest raise. Sampler must still gather n.
    def healthy(_p: str) -> str:
        return "ok"

    def broken(_p: str) -> str:
        raise RuntimeError("ollama down")

    infer_by_node = {n: broken for n in FLEET_WORKER_NODES}
    infer_by_node["dan"] = healthy
    fs = FleetSampler(infer_by_node=infer_by_node)
    res = fs.best_of_n("p", n=3)
    check("I3: gathered n despite failures", len(res.samples) == 3)
    check("I3: only healthy node used", res.nodes_used == ["dan"])


def test_all_nodes_fail_bounded() -> None:
    def broken(_p: str) -> str:
        raise RuntimeError("down")

    fs = FleetSampler(infer_by_node={n: broken for n in FLEET_WORKER_NODES})
    res = fs.best_of_n("p", n=5)  # must not hang; bounded retries
    check("all-fail yields no samples", len(res.samples) == 0)
    check("all-fail winner None", res.winner is None)


def test_deterministic() -> None:
    a = FleetSampler().best_of_n("same", n=8)
    b = FleetSampler().best_of_n("same", n=8)
    check("deterministic winner node", a.winner.node == b.winner.node)
    check("deterministic winner text", a.winner.text == b.winner.text)


def test_pass_at_k() -> None:
    check("c=0 → 0", pass_at_k(10, 0, 5) == 0.0)
    check("all correct → 1", pass_at_k(10, 10, 5) == 1.0)
    check("monotone in k", pass_at_k(10, 1, 1) < pass_at_k(10, 1, 5))
    # n=10, c=1, k=10 → must solve: 1.0
    check("k=n with c≥1 → 1", pass_at_k(10, 1, 10) == 1.0)
    raised = False
    try:
        pass_at_k(5, 1, 6)
    except ValueError:
        raised = True
    check("k>n rejected", raised)


def main() -> int:
    for fn in (
        test_roster,
        test_best_of_n_kernel,
        test_live_infer_routing,
        test_node_failure_degrade,
        test_all_nodes_fail_bounded,
        test_deterministic,
        test_pass_at_k,
    ):
        fn()
    print(f"fleet_sampler: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
