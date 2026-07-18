"""End-to-end flywheel coupling: Loop-A winners → Loop-B corpus ingest.

Standalone-runnable:
    python3 src/shinka/engine/test_flywheel_e2e.py

Proves the ADR-2606142200 flywheel: ShinkaEvolutionCell (Loop A) produces
`corpus_candidates` from its tournament winner, which flywheel_ingest (Loop B)
gates and stages toward the Maxwell SFT corpus — all dry-run (I2: no write).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "shinka" / "engine"))

from cell import EvolutionState, ShinkaEvolutionCell  # noqa: E402
from maxwell_rsi import flywheel_ingest  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {name}")


def test_loopA_feeds_loopB() -> None:
    # Loop A: run the evolution cell over several tasks, collect winners.
    candidates: list[dict[str, str]] = []
    for task in ("upgrade-himawari-cell", "add-lexicon-field", "fix-kaizen-rule"):
        out = ShinkaEvolutionCell().solve(EvolutionState(task=task, n_propose=4))
        candidates.extend(out.corpus_candidates)
    check("Loop A produced winners", len(candidates) == 3)
    check(
        "each candidate is an SFT pair",
        all(set(c) == {"id", "instruction", "completion"} for c in candidates),
    )

    # Loop B: ingest into the corpus (dry-run), starting from the live 125/1000.
    res = flywheel_ingest(candidates, existing_ids=set(), current_count=125)
    check("all 3 staged (charter-clean, unique)", len(res.staged) == 3)
    check("corpus advanced 125 -> 128", res.new_count == 128)
    check("ingest is dry-run (corpus-only)", "dry-run" in res.provenance["status"])

    # Re-ingest the same winners → fully deduped (idempotent flywheel).
    res2 = flywheel_ingest(candidates, existing_ids={c["id"] for c in candidates}, current_count=128)
    check("re-ingest dedups all", len(res2.staged) == 0 and res2.skipped_dup == 3)


def main() -> int:
    test_loopA_feeds_loopB()
    print(f"flywheel_e2e: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
