"""maxwell_rsi — Shinka Loop-B: Maxwell weight evolution (Robin → RSi pipeline).

Per ADR-2606142200. Maps Robin's continuous discovery loop
(hypothesis → experiment → analyse → updated-hypothesis) onto the existing
Maxwell RSi pipeline (collect_corpus → gate_candidates → train[EVO-X2 ROCm] →
eval[e7m bench micro] → deploy), with the deploy gate from the ADR:

    flip Maxwell `available: true`  iff  training_steps >= 250  OR  microbench_delta_pp >= +5.0

and the flywheel coupling: Loop-A tournament winners (ShinkaEvolutionCell
`corpus_candidates`) are ingested here as SFT pairs, gated, deduped, and counted
toward the M1 corpus target (currently 125/1000 per maxwell-models.jsonl).

Determinism / safety:
  * `experiment` (train) and `analyse` (eval) are typed HOOKS. At S0 they run a
    deterministic stand-in and honour the EVO-X2 availability flag — when the
    fleet GPU is offline (the live state as of 2026-06-13) the loop returns a
    `blocked` status instead of fabricating a training run. Murakumo-only (I3):
    no commercial GPU is ever invoked.
  * Ingest never WRITES the corpus file. It returns the staged, gated pairs and
    a provenance record; appending to 90-docs/baien/maxwell-sft-corpus.jsonl +
    flipping the registry is the operator/leash-gated step (I2), done elsewhere.
  * Provenance records are append-only (I1), shaped to maxwell-models.jsonl.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

try:  # reuse the Loop-A Charter gate (fail-open to local scan)
    from .cell import _scan_ok  # type: ignore
except Exception:  # pragma: no cover - standalone import path
    from cell import _scan_ok  # type: ignore


# Deploy gate thresholds (ADR-2606142200; recipe from ADR-2605250400).
GATE_MIN_STEPS = 250
GATE_MIN_DELTA_PP = 5.0

# Corpus targets (maxwell-models.jsonl / ADR-2606061000).
CORPUS_TRAIN_FLOOR = 500   # ≥500 pairs before the first EVO-X2 LoRA run
CORPUS_M1_TARGET = 1000    # ≥1000 pairs for the M1 milestone


@dataclass
class DeployGate:
    """The Maxwell `available: true` flip gate."""

    min_steps: int = GATE_MIN_STEPS
    min_delta_pp: float = GATE_MIN_DELTA_PP

    def passes(self, steps: int, microbench_delta_pp: float) -> bool:
        return steps >= self.min_steps or microbench_delta_pp >= self.min_delta_pp

    def reason(self, steps: int, microbench_delta_pp: float) -> str:
        if steps >= self.min_steps:
            return f"steps {steps} >= {self.min_steps}"
        if microbench_delta_pp >= self.min_delta_pp:
            return f"microbench +{microbench_delta_pp:.1f}pp >= +{self.min_delta_pp:.1f}pp"
        return (
            f"FAIL: steps {steps} < {self.min_steps} AND "
            f"microbench +{microbench_delta_pp:.1f}pp < +{self.min_delta_pp:.1f}pp"
        )


@dataclass
class RSiState:
    """State threaded through Robin's hypothesis→experiment→analyse→update loop."""

    base_model: str = "google/gemma-4-E4B"
    corpus_pairs: int = 125            # current corpus size (maxwell-models.jsonl)
    evo_x2_online: bool = False        # EVO-X2 ROCm reachable (offline as of 2026-06-13)
    recipe: dict[str, Any] | None = None
    run: dict[str, Any] | None = None  # experiment result
    eval: dict[str, Any] | None = None  # analyse result
    decision: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    status: str = "init"               # init|ready|blocked|trained|gated|flipped|rejected


def node_hypothesis(state: RSiState) -> RSiState:
    """Hypothesis: specify the next fine-tune recipe (reuses ADR-2605250400 LoRA)."""
    state.recipe = {
        "base": state.base_model,
        "method": "peft+trl LoRA",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lr": 2e-4,
        "precision": "bf16",
        "hardware": "EVO-X2 ROCm",  # I3 Murakumo-only — never commercial GPU
        "corpus_pairs": state.corpus_pairs,
    }
    # Cannot train below the corpus floor; record readiness honestly.
    if state.corpus_pairs < CORPUS_TRAIN_FLOOR:
        state.status = "blocked"
    else:
        state.status = "ready"
    return state


def node_experiment(
    state: RSiState, train_hook: Callable[[dict[str, Any]], dict[str, Any]] | None = None
) -> RSiState:
    """Experiment = a training run. Honours EVO-X2 availability; never fabricates."""
    if state.status == "blocked":
        state.run = {
            "ran": False,
            "reason": f"corpus {state.corpus_pairs} < train floor {CORPUS_TRAIN_FLOOR}",
            "steps": 0,
        }
        return state
    if not state.evo_x2_online:
        state.run = {
            "ran": False,
            "reason": "EVO-X2 ROCm offline (Tailscale/LAN unreachable)",
            "steps": 0,
        }
        state.status = "blocked"
        return state
    if train_hook is not None:
        try:
            state.run = {"ran": True, **train_hook(state.recipe or {})}
            state.status = "trained"
            return state
        except Exception as e:  # fail honest, not fabricated
            state.run = {"ran": False, "reason": f"train_hook error: {e}", "steps": 0}
            state.status = "blocked"
            return state
    # Deterministic stand-in: 1 SGD step per pair above the floor (bounded model).
    steps = max(0, state.corpus_pairs - CORPUS_TRAIN_FLOOR)
    state.run = {"ran": True, "steps": steps, "note": "deterministic stand-in"}
    state.status = "trained"
    return state


def node_analyse(
    state: RSiState, eval_hook: Callable[[dict[str, Any]], float] | None = None
) -> RSiState:
    """Analyse = e7m bench micro eval → microbench delta (pp vs gemma-4-e4b-it)."""
    if state.status == "blocked" or not (state.run and state.run.get("ran")):
        state.eval = {"evaluated": False, "delta_pp": 0.0}
        return state
    if eval_hook is not None:
        try:
            delta = float(eval_hook(state.run))
        except Exception:
            delta = 0.0
    else:
        # Stand-in: a saturating curve over steps (no fabrication of frontier gains).
        steps = int(state.run.get("steps", 0))
        delta = round(6.0 * (steps / (steps + 300.0)), 2) if steps else 0.0
    state.eval = {"evaluated": True, "delta_pp": delta, "bench": "e7m bench micro"}
    return state


def node_update(state: RSiState, gate: DeployGate | None = None) -> RSiState:
    """Updated-hypothesis: apply the deploy gate; decide flip; emit provenance.

    I2: the decision is advisory — it sets `flip_available` but the actual
    registry flip + corpus write is operator/leash-gated and performed elsewhere.
    """
    gate = gate or DeployGate()
    steps = int((state.run or {}).get("steps", 0))
    delta = float((state.eval or {}).get("delta_pp", 0.0))
    ran = bool((state.run or {}).get("ran"))
    if not ran:
        state.decision = {
            "flip_available": False,
            "reason": (state.run or {}).get("reason", "no training run"),
            "blocked": True,
        }
        state.status = "blocked"
    else:
        ok = gate.passes(steps, delta)
        state.decision = {
            "flip_available": ok,
            "reason": gate.reason(steps, delta),
            "blocked": False,
        }
        state.status = "flipped" if ok else "rejected"
    # Append-only provenance record (I1), shaped to maxwell-models.jsonl.
    state.provenance = {
        "run": "rsi-loopB",
        "kind": "train" if ran else "blocked",
        "base": state.base_model,
        "corpus_pairs": state.corpus_pairs,
        "steps": steps,
        "microbench_delta_pp": delta,
        "gate": f">={gate.min_steps} steps OR >=+{gate.min_delta_pp}pp",
        "flip_available": state.decision["flip_available"],
        "status": state.status,
        "note": "advisory; registry flip + corpus write are operator/leash-gated (I2)",
    }
    return state


def run_rsi(
    state: RSiState,
    gate: DeployGate | None = None,
    train_hook: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    eval_hook: Callable[[dict[str, Any]], float] | None = None,
) -> RSiState:
    """Drive the full Robin loop: hypothesis → experiment → analyse → update."""
    state = node_hypothesis(state)
    state = node_experiment(state, train_hook)
    state = node_analyse(state, eval_hook)
    state = node_update(state, gate)
    return state


# --------------------------------------------------------------------------- #
# Flywheel: Loop-A tournament winners → Loop-B SFT corpus (Wave-3 RSi feed)
# --------------------------------------------------------------------------- #


@dataclass
class FlywheelResult:
    staged: list[dict[str, str]] = field(default_factory=list)  # gated, deduped pairs
    rejected_charter: int = 0
    skipped_dup: int = 0
    new_count: int = 0          # corpus size after ingest (if written)
    train_floor_reached: bool = False
    m1_target_reached: bool = False
    provenance: dict[str, Any] | None = None


def flywheel_ingest(
    corpus_candidates: list[dict[str, str]],
    existing_ids: set[str] | None = None,
    current_count: int = 125,
) -> FlywheelResult:
    """Ingest Loop-A `corpus_candidates` into the Maxwell SFT corpus (DRY-RUN).

    Mirrors gate_candidates.py: dedup by id + Charter Rider gate. Returns the
    staged pairs and progress toward the corpus floor/target; it NEVER writes
    maxwell-sft-corpus.jsonl (operator/leash-gated, I2).
    """
    existing = set(existing_ids or set())
    res = FlywheelResult()
    seen = set(existing)
    for c in corpus_candidates:
        cid = c.get("id", "")
        text = f"{c.get('instruction', '')}\n{c.get('completion', '')}"
        if cid in seen:
            res.skipped_dup += 1
            continue
        if not _scan_ok(text):
            res.rejected_charter += 1
            continue
        seen.add(cid)
        res.staged.append(
            {
                "id": cid,
                "instruction": c.get("instruction", ""),
                "completion": c.get("completion", ""),
            }
        )
    res.new_count = current_count + len(res.staged)
    res.train_floor_reached = res.new_count >= CORPUS_TRAIN_FLOOR
    res.m1_target_reached = res.new_count >= CORPUS_M1_TARGET
    res.provenance = {
        "run": "flywheel-ingest",
        "kind": "corpus",
        "pairs": len(res.staged),
        "rejected_charter": res.rejected_charter,
        "skipped_dup": res.skipped_dup,
        "source": "Loop-A ShinkaEvolutionCell tournament winners",
        "gate": "id-dedup + charter_rider",
        "corpus_after": res.new_count,
        "progress": f"{res.new_count}/{CORPUS_M1_TARGET}",
        "status": "corpus-only (dry-run; write is operator/leash-gated, I2)",
    }
    return res
