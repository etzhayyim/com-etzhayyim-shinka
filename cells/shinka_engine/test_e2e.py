"""End-to-end smoke test: the whole Shinka engine composed in one run.

Standalone-runnable:
    python3 20-actors/shinka/cells/shinka_engine/test_e2e.py

Drives the full stack together — orchestrator beats (Loop A + Loop B + flywheel)
with a FleetSampler and seed RAG self-grounding, difficulty-adaptive fleet
compute, the pass@k eval harness, the Track-E preference corpus, the Track-F
generation lifecycle with guards, and per-node quantization — asserting the
constitutional invariants (I1/I2/I3) hold across the composition.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptive import adaptive_best_of_n  # noqa: E402
from bench_harness import BenchHarness, BenchTask  # noqa: E402
from fleet_sampler import FleetSampler  # noqa: E402
from matformer import Tier  # noqa: E402
from orchestrator import ShinkaOrchestrator  # noqa: E402
from quantization import NodeSpec, recommend  # noqa: E402
from reward import RewardComponents, ScoredCandidate, build_preference_corpus  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {name}")


def test_full_stack() -> None:
    # --- Supervisor: beats with fleet sampling + RAG self-grounding ---------- #
    seed = [{"e": "maxwell", "a": ":weight/base", "v": "gemma 4 e4b murakumo fleet", "op": ":db/add"}]
    orch = ShinkaOrchestrator(sampler=FleetSampler(), seed_datoms=seed, corpus_count=125)
    # distinct tasks → distinct tournament winners → distinct corpus pairs
    tasks = ["evolve maxwell weight", "tune murakumo fleet routing", "improve maxwell corpus harvest"]
    recs = [orch.beat(t) for t in tasks]

    check("3 beats ran", len(recs) == 3)
    check("corpus advanced 125→128", recs[-1].corpus_after == 128)
    check("heads chained (distinct)", len({r.head_cid for r in recs}) == 3)
    check("first beat grounded from seed", recs[0].grounded_refs >= 1)
    # I1: every datom in the whole log is append-only
    check("I1 append-only across log", all(d["op"] == ":db/add" for d in orch.log))
    # I2: no beat auto-merges
    check("I2 no auto-merge", all(r.pr_draft["auto_merge"] is False for r in recs))
    # I3: train honestly blocked offline (no fabricated EVO-X2 run)
    check("I3 train blocked offline", all(r.train_status == "blocked" for r in recs))

    # --- Track C×A: difficulty-adaptive fleet compute ----------------------- #
    easy = adaptive_best_of_n(FleetSampler(), "rename a variable")
    hard = adaptive_best_of_n(
        FleetSampler(), "prove the distributed recursive invariant optimizing concurrent fleet scheduling"
    )
    check("easy → E2B", easy.tier is Tier.E2B)
    check("hard → E4B with more samples", hard.tier is Tier.E4B and hard.k > easy.k)

    # --- Track A: pass@k eval harness --------------------------------------- #
    sampler = FleetSampler(infer_by_node={n: (lambda p: "CORRECT") for n in FleetSampler().nodes})
    rep = BenchHarness(sampler, [BenchTask("t", "p", lambda s: "CORRECT" in s)]).run(ks=(1, 10))
    check("pass@k harness runs", rep.point(10).pass_at_k == 1.0)

    # --- Track E: preference corpus from scored candidates ------------------ #
    groups = {
        "beat": [
            ScoredCandidate("win", RewardComponents(charter_ok=True, pr_outcome="MERGED")),
            ScoredCandidate("lose", RewardComponents(charter_ok=True, pr_outcome="CLOSED")),
        ]
    }
    pairs = build_preference_corpus(groups)
    check("preference pair built", len(pairs) == 1 and pairs[0].chosen == "win")

    # --- Track F: generation lifecycle with guards -------------------------- #
    g1 = orch.close_generation(held_out_quality=0.6, diversity=0.7)
    check("generation 1 promoted", g1.status == "promoted")
    check("generation rounds == 3 beats", g1.rounds_needed == 3)
    orch.beat("another")
    g2 = orch.close_generation(held_out_quality=0.72, diversity=0.1)  # mode collapse
    check("generation 2 halted (collapse guard)", g2.status == "halted")

    # --- Track G: per-node quantization ------------------------------------- #
    q = recommend(NodeSpec("dan", unified_mem_gb=6.0, backend="metal"))
    check("quant fits the node", q is not None and q.mem_gb <= 6.0)

    # whole-run I1 sanity after generations too
    check("I1 holds after generations", all(d["op"] == ":db/add" for d in orch.log))


def main() -> int:
    test_full_stack()
    print(f"e2e: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
