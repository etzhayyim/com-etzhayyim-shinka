"""Pure-logic tests for datom_rag (Track D Datom-log RAG grounding).

Standalone-runnable:
    python3 src/shinka/engine/test_datom_rag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "shinka" / "engine"))

from datom_rag import DatomStore, datom_cid  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {name}")


def _d(e: str, a: str, v: str) -> dict:
    return {"e": e, "a": a, "v": v, "op": ":db/add"}


DATOMS = [
    _d("himawari", ":actor/tier", "R0.1 solar pv manufacturing"),
    _d("maxwell", ":weight/base", "gemma 4 e4b fine-tune"),
    _d("murakumo", ":fleet/nodes", "ten mac mini m4 plus evo-x2"),
    _d("ibuki", ":organism/beat", "replay perceive feel decide narrate act checkpoint"),
]


def test_retrieve_relevant() -> None:
    store = DatomStore(DATOMS)
    hits = store.retrieve("maxwell weight gemma", k=5)
    check("retrieves the maxwell fact first", hits[0].datom["e"] == "maxwell")
    check("score positive", hits[0].score >= 2)


def test_retrieve_irrelevant_empty() -> None:
    store = DatomStore(DATOMS)
    check("no overlap → empty", store.retrieve("zzz quantum banana", k=5) == [])


def test_topk_and_order() -> None:
    store = DatomStore(DATOMS)
    hits = store.retrieve("fleet mac mini murakumo nodes", k=2)
    check("k limit respected", len(hits) <= 2)
    check("murakumo fact ranks top", hits[0].datom["e"] == "murakumo")
    # deterministic across constructions
    h2 = DatomStore(DATOMS).retrieve("fleet mac mini murakumo nodes", k=2)
    check("deterministic order", [f.cid for f in hits] == [f.cid for f in h2])


def test_ground_refs_verify() -> None:
    store = DatomStore(DATOMS)
    g = store.ground("ibuki organism beat cycle", k=3)
    check("ground returns refs", len(g.refs) >= 1)
    check("snippet non-empty", "beat" in g.snippet or len(g.snippet) > 0)
    # every cited CID must verify against the log (anti-hallucination)
    check("all refs verify", all(store.verify_citation(c) for c in g.refs))


def test_no_grounding_snippet() -> None:
    store = DatomStore(DATOMS)
    g = store.ground("no such content xyz", k=3)
    check("empty refs", g.refs == [])
    check("honest no-grounding snippet", "no grounding" in g.snippet)


def test_citation_anti_hallucination() -> None:
    store = DatomStore(DATOMS)
    real = datom_cid(DATOMS[0])
    check("real cid verifies", store.verify_citation(real) is True)
    check("fabricated cid rejected", store.verify_citation("bafy-shinka-deadbeef") is False)


def test_feeds_context_refs() -> None:
    # ground() output is shaped to feed EvolutionState.context_refs (list[str] of CIDs)
    store = DatomStore(DATOMS)
    refs = store.ground("maxwell murakumo fleet", k=3).refs
    check("refs is list of str", isinstance(refs, list) and all(isinstance(r, str) for r in refs))
    check("refs are CIDs", all(r.startswith("bafy-shinka-") for r in refs))


def main() -> int:
    for fn in (
        test_retrieve_relevant,
        test_retrieve_irrelevant_empty,
        test_topk_and_order,
        test_ground_refs_verify,
        test_no_grounding_snippet,
        test_citation_anti_hallucination,
        test_feeds_context_refs,
    ):
        fn()
    print(f"datom_rag: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
