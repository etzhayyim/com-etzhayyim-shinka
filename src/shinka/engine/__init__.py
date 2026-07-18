"""shinka_engine — the Shinka self-evolution engine (ADR-2606142200).

Loop A (capability evolution): ShinkaEvolutionCell — the co-scientist
generate→debate→evolve→synthesize super-step graph.
Loop B (weight evolution): maxwell_rsi — Robin's hypothesis→experiment→analyse→
update over the Maxwell RSi pipeline (DeployGate, flywheel_ingest).
Supervisor: ShinkaOrchestrator — the ibuki-style beat cycle that drives both
loops and the flywheel between them.
"""

from .cell import (
    ShinkaEvolutionCell,
    EvolutionState,
    Proposal,
    elo_update,
)
from .maxwell_rsi import (
    DeployGate,
    RSiState,
    run_rsi,
    flywheel_ingest,
    FlywheelResult,
    CORPUS_TRAIN_FLOOR,
    CORPUS_M1_TARGET,
)
from .orchestrator import ShinkaOrchestrator, BeatRecord
from .kotoba_sink import InMemorySink, KotobaBridgeSink
from .fleet_sampler import (
    FleetSampler,
    FLEET_WORKER_NODES,
    pass_at_k,
    BestOfNResult,
)
from .bench_harness import BenchHarness, BenchTask, BenchReport, KPoint
from .reward import (
    RewardComponents,
    ScoredCandidate,
    PreferencePair,
    aggregate_reward,
    build_preference_pair,
    build_preference_corpus,
)
from .matformer import Tier, MatFormerRouter, estimate_difficulty, route
from .speculative import (
    expected_tokens_per_step,
    speculative_speedup,
    simulate_decode,
    DrafterFreshness,
)
from .datom_rag import DatomStore, GroundedContext, RetrievedFact, datom_cid
from .distill_flywheel import DistillFlywheel, Generation, projected_rounds
from .quantization import QuantOption, NodeSpec, select_quant, pareto_front, recommend
from .adaptive import adaptive_best_of_n, AdaptiveResult, DEFAULT_BUDGETS
from .live_hooks import murakumo_infer, kotoba_poster
from .preflight import (
    fleet_preflight,
    PreflightVerdict,
    tailscale_ssh_probe,
    rocm_http_probe,
    rsi_state_from_preflight,
)

__all__ = [
    # Loop A
    "ShinkaEvolutionCell",
    "EvolutionState",
    "Proposal",
    "elo_update",
    # Loop B
    "DeployGate",
    "RSiState",
    "run_rsi",
    "flywheel_ingest",
    "FlywheelResult",
    "CORPUS_TRAIN_FLOOR",
    "CORPUS_M1_TARGET",
    # Supervisor
    "ShinkaOrchestrator",
    "BeatRecord",
    # append-only sink (I1 commit-DAG)
    "InMemorySink",
    "KotobaBridgeSink",
    # Research Track A — fleet test-time compute
    "FleetSampler",
    "FLEET_WORKER_NODES",
    "pass_at_k",
    "BestOfNResult",
    # Research Track A — standing eval harness
    "BenchHarness",
    "BenchTask",
    "BenchReport",
    "KPoint",
    # Research Track E — verifier-grounded reward / preference
    "RewardComponents",
    "ScoredCandidate",
    "PreferencePair",
    "aggregate_reward",
    "build_preference_pair",
    "build_preference_corpus",
    # Research Track C — MatFormer elastic E2B/E4B routing
    "Tier",
    "MatFormerRouter",
    "estimate_difficulty",
    "route",
    # Research Track B — speculative decoding + TLT adaptive drafter
    "expected_tokens_per_step",
    "speculative_speedup",
    "simulate_decode",
    "DrafterFreshness",
    # Research Track D — Datom-log RAG grounding
    "DatomStore",
    "GroundedContext",
    "RetrievedFact",
    "datom_cid",
    # Research Track F — distillation flywheel + collapse/reward-hacking guards
    "DistillFlywheel",
    "Generation",
    "projected_rounds",
    # Research Track G — fleet quantization frontier
    "QuantOption",
    "NodeSpec",
    "select_quant",
    "pareto_front",
    "recommend",
    # Integration — difficulty-adaptive fleet compute (Track C × Track A)
    "adaptive_best_of_n",
    "AdaptiveResult",
    "DEFAULT_BUDGETS",
    # S1 — live-fleet adapters (Murakumo-only; the only place network I/O lives)
    "murakumo_infer",
    "kotoba_poster",
    # S1 — Loop-B training-readiness preflight (gad / EVO-X2 / corpus)
    "fleet_preflight",
    "PreflightVerdict",
    "tailscale_ssh_probe",
    "rocm_http_probe",
    "rsi_state_from_preflight",
]
