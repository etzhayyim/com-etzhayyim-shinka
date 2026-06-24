"""kotoba_sink — append-only datom sink for the Shinka beat log (S0).

Per ADR-2606142200 (I1: append-only :db/add evidence) + ibuki ADR-2606101200/
2606111400 (content-addressed commit-DAG on the kotoba Datom log, crash-resume,
no-server-key). Abstracts WHERE a beat's datoms land:

  * InMemorySink — the offline default; a tamper-evident commit-DAG in memory
    (each transact chains on the previous head CID), used for tests + local dev.
  * KotobaBridgeSink — the live path to the kotoba engine (:8077 `datomic.transact`,
    ibuki kotoba_bridge.py pattern). It PRESENTS an opaque member CACAO capability
    (present-only, never signs — leash, ADR-2606111400) and delegates the actual
    HTTP to an injected host `poster`. With no host poster it REFUSES (no-server-
    key: it never holds a key or fabricates a write) — the loop stays dry-run.

Both enforce I1 structurally: a non-`:db/add` datom is refused at transact time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


def _canon(datoms: list[dict[str, Any]]) -> str:
    """Canonical, replayable serialization of a datom batch."""
    return "\n".join(f"{d['op']}|{d['e']}|{d['a']}|{d['v']}" for d in datoms)


def _cid(payload: str) -> str:
    """Deterministic content reference (S0 stand-in for a kotoba CIDv1).

    Stable across processes/runs (no Python hash randomization) so the commit-DAG
    is reproducible in tests; the live KotobaBridgeSink returns the engine's real
    CIDv1 instead.
    """
    h = 0xCBF29CE484222325
    for ch in payload:
        h ^= ord(ch)
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF  # FNV-1a 64-bit
    return f"bafy-shinka-{h:016x}"


def _assert_add_only(datoms: list[dict[str, Any]]) -> None:
    for d in datoms:
        if d.get("op") != ":db/add":
            raise ValueError(
                f"kotoba sink is append-only (I1): refused op {d.get('op')!r} "
                "(ADR-2606142200)"
            )


@dataclass
class TxRecord:
    seq: int
    parent: str | None
    head: str
    n: int  # datoms in this tx


class InMemorySink:
    """Tamper-evident in-memory commit-DAG over the beat datoms (offline default)."""

    def __init__(self) -> None:
        self.datoms: list[dict[str, Any]] = []
        self.head: str | None = None
        self.txs: list[TxRecord] = []

    def transact(
        self, datoms: list[dict[str, Any]], expected_parent: str | None = None
    ) -> str:
        """Append a batch, chaining on the current head. Returns the new head CID.

        Raises if `expected_parent` does not match the current head (tamper-/
        concurrency-evidence — the ibuki `expected_parent` chaining property).
        """
        _assert_add_only(datoms)
        if expected_parent != self.head:
            raise ValueError(
                f"commit-DAG parent mismatch: expected {expected_parent!r}, "
                f"head is {self.head!r}"
            )
        self.datoms.extend(datoms)
        new_head = _cid((self.head or "") + _canon(datoms))
        self.head = new_head
        self.txs.append(
            TxRecord(seq=len(self.txs) + 1, parent=expected_parent, head=new_head, n=len(datoms))
        )
        return new_head

    def load(self, datoms: list[dict[str, Any]]) -> str | None:
        """Restore a flat historical datom list (replay) by re-chaining it as one tx.

        Returns the resulting head so a resumed orchestrator can continue the DAG.
        """
        self.datoms = []
        self.head = None
        self.txs = []
        if datoms:
            return self.transact(list(datoms), expected_parent=None)
        return None


@dataclass
class KotobaBridgeSink:
    """Live kotoba-engine sink (operator/leash-gated; no-server-key).

    `endpoint` e.g. "http://127.0.0.1:8077". `present_cacao` is the opaque member
    capability the organism PRESENTS (never signs). `poster(endpoint, payload,
    present_cacao) -> head_cid` is the host-supplied transactor; absent ⇒ refuse.
    """

    endpoint: str
    poster: Callable[[str, dict[str, Any], str | None], str] | None = None
    present_cacao: str | None = None
    head: str | None = None
    txs: list[TxRecord] = field(default_factory=list)

    def transact(
        self, datoms: list[dict[str, Any]], expected_parent: str | None = None
    ) -> str:
        _assert_add_only(datoms)
        if self.poster is None:
            raise RuntimeError(
                "no-server-key: KotobaBridgeSink needs an injected host poster; "
                "the loop stays dry-run until a member-CACAO-capable host transacts "
                "(ADR-2606111400 leash)"
            )
        payload = {
            "tx": "datomic.transact",
            "datoms": datoms,
            "expected_parent": expected_parent,
        }
        head = self.poster(self.endpoint, payload, self.present_cacao)
        self.head = head
        self.txs.append(
            TxRecord(seq=len(self.txs) + 1, parent=expected_parent, head=head, n=len(datoms))
        )
        return head

    @property
    def committable(self) -> bool:
        """A live write is possible only with BOTH a host poster and a CACAO cap."""
        return self.poster is not None and bool(self.present_cacao)
