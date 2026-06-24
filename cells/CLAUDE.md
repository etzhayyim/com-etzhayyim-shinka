# 20-actors/shinka/cells — Shinka self-evolution engine (CLAUDE.md)

## Identity

- **Engine**: `shinka_engine` — the Shinka self-evolution engine (Loop A + Loop B + flywheel).
- **DID**: shares `did:web:shinka.etzhayyim.com` with the actor's social-evolution scheduler (`../actor-manifest.jsonld`).
- **ADR**: ADR-2606142200 (`90-docs/adr/2606142200-shinka-self-evolution-engine.md`, proposed).
- **External basis**: DeepMind co-scientist (generate→debate→evolve, Elo tournament) + Robin (Nature 2026, hypothesis→experiment→analyse→update) + MIT TLT (arXiv 2511.16665, Tracks B/E).
- **Status**: S0 landed + S1 adapters + training preflight — full engine (both loops + Supervisor + all 7 research tracks) + live-fleet adapters (`live_hooks.py`) + Loop-B readiness gate (`preflight.py`); **376 standalone tests green / 22 suites**; LLM-free deterministic kernel + typed Murakumo/fleet hooks. NOT operationally activated (live Murakumo infer / EVO-X2 training / registry flip / live kotoba transact are all operator/leash-gated; `gad` (EVO-X2, Ubuntu ROCm) reached over Tailscale SSH (IP 100.82.98.110, Ubuntu .16)).

## What this is (and is not)

`shinka_engine` evolves **actors / cells / code / hypotheses** (capability) and **the Maxwell weight** (weight) on the murakumo fleet. It is a **SIBLING** of the existing shinka social-evolution scheduler (which evolves social posting cadence) — same DID, different target.

It is **NOT** a frontier-beating chase: the thesis is `frontier-class = small weight × fleet test-time compute × tournament/verify × Datom-log retrieval → distilled back`. The baien edge invariant (ADR-2605241900, frontier-beating non-target) is untouched.

## Architecture

```
            Loop A — capability (co-scientist)                  Loop B — weight (Robin/RSi)
  propose → reflect → cluster → rank(Elo) → recombine → synthesize    hypothesis → experiment → analyse → update
     │  (FleetSampler best-of-N)              │ PR draft (no auto-merge)    collect → gate → train(EVO-X2) → eval → deploy
     └────────────── kotoba commit-DAG (:db/add) ──────────────┘            gate: ≥250 steps OR ≥+5pp
                          ▲ corpus_candidates (dry-run flywheel) ───────────────┘
                ShinkaOrchestrator (Supervisor, ibuki beat cycle):
       replay → perceive → decide(Loop A) → flywheel → maybe_train(Loop B) → narrate(Murakumo) → checkpoint → act
```

## Co-scientist agent → cell mapping

| Co-scientist | Cell / node | Function |
|---|---|---|
| Supervisor | `ShinkaOrchestrator.beat` | ibuki beat cycle; drives both loops + flywheel |
| Generation | `node_propose` | candidates; bodies via FleetSampler best-of-N (Track A) |
| Reflection | `node_reflect` | Charter G1-G8 pre-scan (reuses charter_rider.scan) + review score |
| Proximity | `node_cluster` | dedup/diversity, keep best per cluster |
| Ranking | `node_rank` | Elo pairwise debate (Murakumo hook, fail-open kernel) |
| Evolution | `node_recombine` | merge top-2 Elo into a recombinant (re-scanned) |
| Meta-review | `node_synthesize` | PR draft (never auto-merge) + dry-run corpus feed |

## Invariants (enforced in code + tests)

- **I1 append-only** — every fact is a `:db/add` datom; `cell._datom` and the sink refuse `:db/retract`. Rejections are evidence, not deletions. Commit-DAG is tamper-evident (`expected_parent` chaining).
- **I2 no autonomous merge** — `synthesize` emits a PR draft (`member_signed/auto_merge` False); committable ONLY with a member CACAO capability (ADR-2606111400). The engine presents, never signs.
- **I3 Murakumo-only** — all inference resolves to the fleet; every hook fails OPEN to the deterministic kernel (never a commercial GPU / vendor call).

## Modules

16 source modules; 22 standalone test suites (376 tests). Core + research tracks:

| Module | Role | Tests |
|---|---|---|
| `cell.py` | Loop A: ShinkaEvolutionCell + 6 nodes + Elo | 24 |
| `maxwell_rsi.py` | Loop B: DeployGate, Robin loop, flywheel_ingest | 29 |
| `orchestrator.py` | Supervisor beat cycle (ibuki), replay/resume, generations | 23 |
| `kotoba_sink.py` | append-only commit-DAG (InMemory + KotobaBridge) | 20 |
| `run_beat.py` | offline-safe beat runner / fleet entrypoint | 14 |
| `fleet_sampler.py` | Track A: fleet best-of-N + Elo, pass@k | 20 |
| `bench_harness.py` | Track A: pass@k vs k standing eval | 16 |
| `speculative.py` | Track B: spec-decode + TLT adaptive-drafter freshness | 26 |
| `matformer.py` | Track C: MatFormer E2B/E4B routing | 17 |
| `adaptive.py` | Track C×A: difficulty-adaptive fleet budget | 13 |
| `datom_rag.py` | Track D: CID-anchored Datom-log RAG grounding | 15 |
| `reward.py` | Track E: verifier-grounded reward + DPO corpus | 24 |
| `distill_flywheel.py` | Track F: rounds-to-quality + collapse guards | 16 |
| `quantization.py` | Track G: tok/s × quality Pareto per node | 15 |
| `live_hooks.py` | S1: Murakumo infer + kotoba poster (only net I/O) | 15 |
| `preflight.py` | S1: Loop-B readiness gate (gad/EVO-X2/corpus, Tailscale IP) | 18 |

Integration/coupling suites: `test_flywheel_e2e` (6, Loop A→B), `test_fleet_wiring`
(12), `test_rag_wiring` (8), `test_generation_wiring` (13), `test_manifest` (13,
loop-A drift guard vs the manifest), `test_e2e` (16, whole-stack smoke).

## Research tracks (ADR §Research Program)

A fleet test-time compute ✅ (fleet_sampler + bench_harness) · B adaptive-drafter speculative ✅ (speculative.py, TLT freshness) · C MatFormer E2B/E4B ✅ (matformer.py) · D Datom-log RAG ✅ (datom_rag.py) · E verifier-grounded reward ✅ (reward.py) · F distillation flywheel ✅ (distill_flywheel.py, collapse guards) · G fleet quantization ✅ (quantization.py). All 7 tracks have an S0 model + tests; S1 swaps the kernels/figures for live fleet measurements.

## Build & test

```bash
cd 20-actors/shinka/cells
for t in shinka_engine/test_*.py; do python3 "$t"; done
# 22 suites / 376 tests; pure-stdlib; no pytest/langgraph required
# (langgraph used if present, else the sequential super-step driver)
```

## Related files

- `90-docs/adr/2606142200-shinka-self-evolution-engine.md` — master ADR
- `90-docs/adr/2606061000-maxwell-default-llm-weight.md` — Maxwell weight (Loop B target)
- `90-docs/baien/maxwell-models.jsonl` — corpus/weight provenance (125/1000)
- `70-tools/scripts/maxwell/{collect_corpus,gate_candidates}.py` — upstream RSi pipeline
- `50-infra/murakumo/fleet.edn` — fleet roster (9 worker nodes + judah gateway)
- `../actor-manifest.jsonld` — shinka social-evolution scheduler (sibling)
- `20-actors/ibuki/` — beat-cycle + commit-DAG + leash pattern this reuses
