"""Validate the shinka actor-manifest self-evolution registration.

Standalone-runnable:
    python3 20-actors/shinka/cells/shinka_engine/test_manifest.py

Checks the manifest parses, registers the self-evolution engine, and that its
declared loop-A nodes match the actual ShinkaEvolutionCell node order (the
registration cannot drift from the implementation).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cell import ShinkaEvolutionCell  # noqa: E402

MANIFEST = HERE.parents[1] / "actor-manifest.jsonld"  # 20-actors/shinka/actor-manifest.jsonld
ENGINE_DIR = HERE  # 20-actors/shinka/cells/shinka_engine

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {name}")


def test_manifest_parses() -> None:
    check("manifest exists", MANIFEST.exists())
    data = json.loads(MANIFEST.read_text())
    check("manifest is a JSON object", isinstance(data, dict))


def test_registration_present() -> None:
    data = json.loads(MANIFEST.read_text())
    check("self-evolution in requiredLoops", "self-evolution" in data.get("requiredLoops", []))
    eng = data.get("selfEvolutionEngine")
    check("selfEvolutionEngine block present", isinstance(eng, dict))
    check("cites the ADR", eng.get("adr") == "ADR-2606142200")
    check("engine path points to shinka_engine", eng.get("enginePath", "").endswith("shinka_engine"))


def test_loopA_matches_implementation() -> None:
    data = json.loads(MANIFEST.read_text())
    declared = data["selfEvolutionEngine"]["loopA"]
    actual = list(ShinkaEvolutionCell._ORDER)
    check("manifest loopA == cell node order (no drift)", declared == actual)


def test_invariants_and_gate_declared() -> None:
    eng = json.loads(MANIFEST.read_text())["selfEvolutionEngine"]
    inv = eng.get("invariants", [])
    check("append-only invariant declared", any("append-only" in s for s in inv))
    check("no-auto-merge/leash invariant declared", any("CACAO" in s or "no-auto-merge" in s for s in inv))
    check("murakumo-only invariant declared", any("murakumo" in s for s in inv))
    check("deploy gate declared", "250" in eng.get("deployGate", "") and "5pp" in eng.get("deployGate", ""))


def test_research_tracks_complete() -> None:
    tracks = json.loads(MANIFEST.read_text())["selfEvolutionEngine"]["researchTracks"]
    check("all 7 tracks A-G declared", sorted(tracks.keys()) == list("ABCDEFG"))
    # each declared track has a module file in the engine dir (sanity, not exhaustive)
    expect = {"A": "fleet_sampler.py", "B": "speculative.py", "C": "matformer.py",
              "D": "datom_rag.py", "E": "reward.py", "F": "distill_flywheel.py",
              "G": "quantization.py"}
    check("each track has its module", all((ENGINE_DIR / m).exists() for m in expect.values()))


def main() -> int:
    for fn in (
        test_manifest_parses,
        test_registration_present,
        test_loopA_matches_implementation,
        test_invariants_and_gate_declared,
        test_research_tracks_complete,
    ):
        fn()
    print(f"manifest: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
