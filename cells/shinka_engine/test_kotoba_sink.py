"""Pure-logic tests for kotoba_sink + its orchestrator wiring (S0).

Standalone-runnable:
    python3 20-actors/shinka/cells/shinka_engine/test_kotoba_sink.py

Covers the append-only commit-DAG (I1), tamper-evident parent chaining,
KotobaBridgeSink no-server-key refusal (leash), and the orchestrator head chain.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kotoba_sink import InMemorySink, KotobaBridgeSink  # noqa: E402
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


def _add(e: str, v: int) -> dict:
    return {"e": e, "a": ":x", "v": v, "op": ":db/add"}


def test_chain_and_determinism() -> None:
    s1 = InMemorySink()
    h1 = s1.transact([_add("a", 1)], expected_parent=None)
    h2 = s1.transact([_add("b", 2)], expected_parent=h1)
    check("head advances", h1 != h2)
    check("two txs recorded", len(s1.txs) == 2)
    check("cid shape", h1.startswith("bafy-shinka-"))
    # same sequence in a fresh sink → identical heads (cross-run reproducible)
    s2 = InMemorySink()
    g1 = s2.transact([_add("a", 1)], expected_parent=None)
    g2 = s2.transact([_add("b", 2)], expected_parent=g1)
    check("deterministic h1", g1 == h1)
    check("deterministic h2", g2 == h2)


def test_parent_mismatch_refused() -> None:
    s = InMemorySink()
    s.transact([_add("a", 1)], expected_parent=None)
    raised = False
    try:
        s.transact([_add("b", 2)], expected_parent="wrong-parent")
    except ValueError:
        raised = True
    check("tamper/concurrency parent mismatch refused", raised)


def test_append_only_refused() -> None:
    s = InMemorySink()
    raised = False
    try:
        s.transact([{"e": "a", "a": ":x", "v": 1, "op": ":db/retract"}])
    except ValueError:
        raised = True
    check("I1: :db/retract refused at sink", raised)


def test_bridge_refuses_without_poster() -> None:
    sink = KotobaBridgeSink(endpoint="http://127.0.0.1:8077")
    check("not committable w/o poster+cacao", sink.committable is False)
    raised = False
    try:
        sink.transact([_add("a", 1)])
    except RuntimeError as e:
        raised = "no-server-key" in str(e)
    check("no-server-key refusal", raised)


def test_bridge_with_host_poster() -> None:
    seen = {}

    def poster(endpoint, payload, cacao):
        seen["endpoint"] = endpoint
        seen["cacao"] = cacao
        seen["n"] = len(payload["datoms"])
        return "bafy-engine-cid-xyz"

    sink = KotobaBridgeSink(
        endpoint="http://127.0.0.1:8077", poster=poster, present_cacao="cacao_b64"
    )
    check("committable with poster+cacao", sink.committable is True)
    head = sink.transact([_add("a", 1)], expected_parent=None)
    check("returns engine cid", head == "bafy-engine-cid-xyz")
    check("presented cacao (not signed)", seen["cacao"] == "cacao_b64")
    check("posted to engine endpoint", seen["endpoint"].endswith(":8077"))


def test_orchestrator_head_chain() -> None:
    orch = ShinkaOrchestrator(corpus_count=125)
    r1 = orch.beat("task-A")
    r2 = orch.beat("task-B")
    check("beat head set", r1.head_cid is not None)
    check("head advances across beats", r1.head_cid != r2.head_cid)
    check("orchestrator head == last beat head", orch.head_cid == r2.head_cid)
    check("log surfaces sink datoms", len(orch.log) == len(orch.sink.datoms) > 0)


def test_orchestrator_replay_resumes_dag() -> None:
    orch = ShinkaOrchestrator(corpus_count=125)
    orch.beat("task-A")
    orch.beat("task-B")
    snapshot = list(orch.log)
    resumed = ShinkaOrchestrator(corpus_count=0)
    resumed.replay(snapshot)
    check("replay restored head", resumed.head_cid is not None)
    # resumed beat continues the DAG without a parent-mismatch raise
    r = resumed.beat("task-C")
    check("resumed beat seq=3", r.seq == 3)
    check("resumed head advanced", r.head_cid != resumed.sink.txs[0].head)


def main() -> int:
    for fn in (
        test_chain_and_determinism,
        test_parent_mismatch_refused,
        test_append_only_refused,
        test_bridge_refuses_without_poster,
        test_bridge_with_host_poster,
        test_orchestrator_head_chain,
        test_orchestrator_replay_resumes_dag,
    ):
        fn()
    print(f"kotoba_sink: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
