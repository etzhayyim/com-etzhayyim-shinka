"""run_beat — Shinka beat-cycle runner / fleet entrypoint (S0).

Per ADR-2606142200. The operational driver (ibuki autorun.py analogue) that runs
the ShinkaOrchestrator beat cycle for N iterations over a task backlog. Offline-
safe by DEFAULT: no live Murakumo infer, no FleetSampler, an InMemorySink — so it
runs anywhere and asserts nothing about the fleet. The fleet deployment injects a
Murakumo infer hook, a FleetSampler over the live nodes, and a KotobaBridgeSink
with a member CACAO capability (operator/leash-gated) — none of which this module
holds (no-server-key).

Invariants preserved: append-only (I1, via the sink), no auto-merge (I2, beats
emit PR drafts only), Murakumo-only (I3, hooks fail open to the kernel).

CLI:
    python3 run_beat.py --beats 3
    python3 run_beat.py --beats 5 --corpus-count 480 --evo-x2-online
"""

from __future__ import annotations

import argparse
from typing import Any

try:
    from .orchestrator import BeatRecord, ShinkaOrchestrator
except Exception:  # pragma: no cover - standalone import path
    from orchestrator import BeatRecord, ShinkaOrchestrator


# A small default self-evolution backlog (the "perceive" task source). The live
# deployment replaces this with the stalest-actor / open-gap query over the
# Datom log (cf. the social scheduler's `MATCH (a:Actor) ... ORDER BY staleness`).
DEFAULT_BACKLOG: tuple[str, ...] = (
    "upgrade L1 actor to L3 (richer EAVT schema + cells)",
    "add a lexicon field with provenance + validation",
    "harvest corpus pairs from a refactor-verified diff",
    "tighten a kaizen rule from PR-merge outcomes",
    "close a coverage gap flagged by a mirror actor",
)


def run(
    orch: ShinkaOrchestrator,
    tasks: list[str],
    beats: int,
) -> list[BeatRecord]:
    """Run `beats` iterations, round-robin over `tasks`. Returns the beat records."""
    if not tasks:
        raise ValueError("task backlog is empty")
    records: list[BeatRecord] = []
    for i in range(beats):
        records.append(orch.beat(tasks[i % len(tasks)]))
    return records


def format_digest(records: list[BeatRecord]) -> str:
    """Human-readable colony digest of the run (one line per beat + a footer)."""
    lines = []
    for r in records:
        committable = "committable" if False else "draft"  # always draft (I2)
        lines.append(
            f"beat {r.seq}: winner={r.winner} corpus→{r.corpus_after} "
            f"train={r.train_status} pr={committable} head={r.head_cid}"
        )
    if records:
        last = records[-1]
        lines.append(
            f"-- {len(records)} beats; corpus {records[0].corpus_after - records[0].corpus_staged}"
            f"→{last.corpus_after}; head {last.head_cid}"
        )
    return "\n".join(lines)


def build_orchestrator(args: argparse.Namespace) -> ShinkaOrchestrator:
    # Offline-safe: no infer, no sampler, default InMemorySink (no-server-key).
    return ShinkaOrchestrator(
        corpus_count=args.corpus_count,
        evo_x2_online=args.evo_x2_online,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Shinka self-evolution beat runner")
    p.add_argument("--beats", type=int, default=3, help="number of beats to run")
    p.add_argument("--corpus-count", type=int, default=125, help="starting Maxwell corpus size")
    p.add_argument(
        "--evo-x2-online",
        action="store_true",
        help="declare EVO-X2 ROCm reachable (default: offline, training blocks honestly)",
    )
    args = p.parse_args(argv)
    if args.beats <= 0:
        print("nothing to do (--beats <= 0)")
        return 0
    orch = build_orchestrator(args)
    records = run(orch, list(DEFAULT_BACKLOG), args.beats)
    print(format_digest(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
