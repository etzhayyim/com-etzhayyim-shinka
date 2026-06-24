"""Integration test: Track-D RAG grounding wired into the orchestrator beat.

Standalone-runnable:
    python3 20-actors/shinka/cells/shinka_engine/test_rag_wiring.py

Proves each beat grounds its task against the engine's own append-only log
(self-grounding) + an optional seed corpus, and that explicit context_refs
override the retrieval.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def _seed(task_words: str) -> list[dict]:
    return [{"e": "prior", "a": ":note/text", "v": task_words, "op": ":db/add"}]


def test_seed_grounding_first_beat() -> None:
    # seed corpus relevant to the task → beat 1 grounds despite empty log
    orch = ShinkaOrchestrator(seed_datoms=_seed("upgrade himawari cell schema lexicon"))
    rec = orch.beat("upgrade himawari cell")
    check("beat 1 grounded from seed", rec.grounded_refs >= 1)
    has_datom = any(d["a"] == ":beat/grounded-refs" for d in rec.datoms)
    check("grounded-refs datom emitted", has_datom)


def test_self_grounding_later_beats() -> None:
    # no seed: beat 1 has nothing to ground on; later beats ground on prior datoms
    orch = ShinkaOrchestrator()
    r1 = orch.beat("add lexicon field provenance validation")
    check("beat 1 no grounding (empty log)", r1.grounded_refs == 0)
    # beat 2 reuses words present in beat-1 datoms (task/proposal text) → grounds
    r2 = orch.beat("add lexicon field provenance validation")
    check("beat 2 self-grounds on prior log", r2.grounded_refs >= 1)


def test_explicit_refs_override() -> None:
    orch = ShinkaOrchestrator(seed_datoms=_seed("anything relevant here"))
    rec = orch.beat("some task", context_refs=["cid:explicit-1", "cid:explicit-2"])
    check("explicit refs honored (count=2)", rec.grounded_refs == 2)


def test_grounding_does_not_break_invariants() -> None:
    orch = ShinkaOrchestrator(seed_datoms=_seed("maxwell murakumo fleet evolution"))
    rec = orch.beat("evolve maxwell on murakumo fleet")
    check("all datoms :db/add (I1)", all(d["op"] == ":db/add" for d in rec.datoms))
    check("pr draft no auto-merge (I2)", rec.pr_draft["auto_merge"] is False)
    check("head chained", rec.head_cid is not None)


def main() -> int:
    for fn in (
        test_seed_grounding_first_beat,
        test_self_grounding_later_beats,
        test_explicit_refs_override,
        test_grounding_does_not_break_invariants,
    ):
        fn()
    print(f"rag_wiring: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
