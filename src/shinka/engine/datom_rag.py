"""datom_rag — Research Track D: retrieval-augmented grounding over the Datom log.

Per ADR-2606142200 §Research Program Track D. The append-only EAVT Datom log is a
CID-addressed, hallucination-resistant memory: retrieval returns only facts that
ACTUALLY exist in the log (each carries a content CID), so a grounded generation
can cite verifiable sources and the gap that frontier scale would otherwise buy
(world knowledge) is closed with the org's own immutable facts.

`DatomStore.ground(query)` returns the CID-anchored context that feeds Loop-A
`propose` (EvolutionState.context_refs). `verify_citation(cid)` rejects a citation
to a CID that is not in the log — the structural anti-hallucination guard.

Pure + stdlib; deterministic token-overlap ranking (no embeddings / vector DB,
consistent with the kotoba hybrid-search posture, ADR-2606012300).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

try:
    from .kotoba_sink import _cid
except Exception:  # pragma: no cover - standalone import path
    from kotoba_sink import _cid


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9:_-]+", text.lower()))


def _datom_text(d: dict[str, Any]) -> str:
    return f"{d.get('e','')} {d.get('a','')} {d.get('v','')}"


def datom_cid(d: dict[str, Any]) -> str:
    """Content CID for a datom (same FNV-1a basis as the commit-DAG sink)."""
    return _cid(f"{d.get('op',':db/add')}|{d.get('e','')}|{d.get('a','')}|{d.get('v','')}")


@dataclass
class RetrievedFact:
    datom: dict[str, Any]
    cid: str
    score: int  # token-overlap count with the query


@dataclass
class GroundedContext:
    query: str
    refs: list[str] = field(default_factory=list)      # CID anchors (→ context_refs)
    facts: list[RetrievedFact] = field(default_factory=list)

    @property
    def snippet(self) -> str:
        if not self.facts:
            return "(no grounding facts in the Datom log)"
        return "; ".join(_datom_text(f.datom) for f in self.facts)


class DatomStore:
    """A retrievable view over append-only datoms (CID-addressed, read-only)."""

    def __init__(self, datoms: list[dict[str, Any]]) -> None:
        self._datoms = list(datoms)
        self._cids = [datom_cid(d) for d in self._datoms]
        self._cid_set = set(self._cids)
        self._toks = [_tokens(_datom_text(d)) for d in self._datoms]

    def __len__(self) -> int:
        return len(self._datoms)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedFact]:
        """Top-k datoms by token overlap with the query (deterministic ties by CID)."""
        q = _tokens(query)
        scored: list[RetrievedFact] = []
        for d, cid, toks in zip(self._datoms, self._cids, self._toks):
            s = len(q & toks)
            if s > 0:
                scored.append(RetrievedFact(datom=d, cid=cid, score=s))
        scored.sort(key=lambda f: (-f.score, f.cid))
        return scored[:k]

    def ground(self, query: str, k: int = 5) -> GroundedContext:
        facts = self.retrieve(query, k)
        return GroundedContext(query=query, refs=[f.cid for f in facts], facts=facts)

    def verify_citation(self, cid: str) -> bool:
        """Anti-hallucination: True only if `cid` names a fact actually in the log."""
        return cid in self._cid_set
