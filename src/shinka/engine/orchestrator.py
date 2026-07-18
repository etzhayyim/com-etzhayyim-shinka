"""shinka_orchestrator — the Supervisor beat-cycle that drives Shinka (S0).

Per ADR-2606142200. The DeepMind co-scientist "Supervisor" mapped onto the
ibuki organism beat cycle (ADR-2606101200):

    replay → perceive → decide(Loop A) → flywheel → maybe_train(Loop B)
           → narrate(Murakumo) → checkpoint(append-only datoms) → act(PR draft)

One `beat(task)` runs the full generate→debate→evolve cycle (ShinkaEvolutionCell),
feeds the tournament winner into the Maxwell SFT corpus (flywheel_ingest), runs
the Robin weight loop if the corpus/hardware gates allow (run_rsi — honestly
blocked while EVO-X2 is offline / corpus < floor), narrates Murakumo-only, and
folds everything into append-only datoms.

Invariants (inherited, enforced + tested):
  I1  every beat fact is a `:db/add` datom (via cell._datom); the beat history is
      replayable evidence. `replay()` restores byte-identical state from the log
      (ibuki crash-resume property) — re-beating a seen task is idempotent.
  I2  `act` emits a PR DRAFT (member_signed/auto_merge False); committable only
      with a member CACAO capability. The orchestrator never merges.
  I3  narration + (future) debate resolve Murakumo-only; `_narrate` fails OPEN to
      a deterministic template (never a commercial call).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

try:  # package-relative
    from .cell import EvolutionState, ShinkaEvolutionCell, _datom
    from .maxwell_rsi import RSiState, flywheel_ingest, run_rsi
    from .kotoba_sink import InMemorySink
    from .datom_rag import DatomStore
    from .distill_flywheel import DistillFlywheel, Generation
except Exception:  # pragma: no cover - standalone import path
    from cell import EvolutionState, ShinkaEvolutionCell, _datom
    from maxwell_rsi import RSiState, flywheel_ingest, run_rsi
    from kotoba_sink import InMemorySink
    from datom_rag import DatomStore
    from distill_flywheel import DistillFlywheel, Generation


@dataclass
class BeatRecord:
    """The append-only record of one Shinka beat."""

    seq: int
    task: str
    winner: str | None = None
    debates: int = 0
    rejected: int = 0
    corpus_staged: int = 0
    corpus_after: int = 0
    train_status: str = "skipped"
    flip_available: bool = False
    narration: str = ""
    pr_draft: dict[str, Any] | None = None
    datoms: list[dict[str, Any]] = field(default_factory=list)
    head_cid: str | None = None  # content-addressed commit-DAG head after checkpoint
    grounded_refs: int = 0       # Track-D RAG: prior facts retrieved to ground this beat


def _default_narrate(record: BeatRecord) -> str:
    return (
        f"beat {record.seq}: task {record.task!r} → winner "
        f"{record.winner or 'none'} over {record.debates} debates "
        f"({record.rejected} charter-rejected); corpus "
        f"+{record.corpus_staged} → {record.corpus_after}; "
        f"train {record.train_status}."
    )


class ShinkaOrchestrator:
    """Supervisor: one heartbeat = one task evolved + (maybe) one weight step.

    `infer` is the Murakumo inference callable (None ⇒ deterministic kernel,
    I3 fail-open). State (`beat_seq`, `seen_corpus_ids`, `corpus_count`) is
    restorable from the datom log via `replay`.
    """

    def __init__(
        self,
        infer: Callable[[str], str] | None = None,
        corpus_count: int = 125,
        evo_x2_online: bool = False,
        sampler: object | None = None,
        sink: Any | None = None,
        seed_datoms: list[dict[str, Any]] | None = None,
    ) -> None:
        self.infer = infer
        self.sampler = sampler
        # Optional external grounding corpus (e.g. actor-registry / lexicon datoms);
        # unioned with the engine's own append-only log for Track-D self-grounding.
        self._seed = list(seed_datoms or [])
        self.beat_seq = 0
        self.corpus_count = corpus_count
        self.evo_x2_online = evo_x2_online
        self.seen_corpus_ids: set[str] = set()
        # Append-only commit-DAG sink (I1). Default = in-memory tamper-evident DAG;
        # a KotobaBridgeSink writes to the live engine (operator/leash-gated).
        self.sink = sink or InMemorySink()
        self.head_cid: str | None = None
        # Track-F distillation flywheel: beats accumulate into a "generation"
        # that is promoted only past the collapse / reward-hacking guards.
        self._flywheel = DistillFlywheel()
        self._gen_start_seq = 0

    # --- replay (ibuki crash-resume) --------------------------------------- #
    def replay(self, datoms: list[dict[str, Any]]) -> None:
        """Restore state from a prior datom log — byte-identical, idempotent (I1)."""
        max_seq = 0
        for d in datoms:
            if d.get("a") == ":beat/seq":
                max_seq = max(max_seq, int(d["v"]))
            elif d.get("a") == ":corpus/staged-id":
                self.seen_corpus_ids.add(str(d["v"]))
            elif d.get("a") == ":corpus/count-after":
                self.corpus_count = int(d["v"])
        self.beat_seq = max_seq
        # Re-load the historical datoms into the sink and resume the commit-DAG.
        self.head_cid = self.sink.load(list(datoms))

    @property
    def log(self) -> list[dict[str, Any]]:
        """The append-only datom log (the sink's commit-DAG contents)."""
        return self.sink.datoms

    def _narrate(self, record: BeatRecord) -> str:
        """I3: Murakumo narration, fail-open to a deterministic template."""
        if self.infer is not None:
            try:
                return self.infer(
                    "Narrate this Shinka evolution beat in one sentence "
                    f"for the colony digest: {_default_narrate(record)}"
                ).strip() or _default_narrate(record)
            except Exception:
                pass
        return _default_narrate(record)

    # --- the beat ---------------------------------------------------------- #
    def beat(self, task: str, context_refs: list[str] | None = None) -> BeatRecord:
        self.beat_seq += 1
        rec = BeatRecord(seq=self.beat_seq, task=task)
        rec.datoms.append(_datom(f"shinka:beat/{self.beat_seq}", ":beat/seq", self.beat_seq))
        rec.datoms.append(_datom(f"shinka:beat/{self.beat_seq}", ":beat/task", task))

        # perceive: ground the task in the engine's OWN append-only log + any seed
        # corpus (Track D RAG, CID-anchored). Explicit context_refs override.
        if context_refs is None:
            grounded = DatomStore(self._seed + self.log).ground(task, k=5)
            refs = grounded.refs
        else:
            refs = context_refs
        rec.grounded_refs = len(refs)
        rec.datoms.append(
            _datom(f"shinka:beat/{self.beat_seq}", ":beat/grounded-refs", len(refs))
        )

        # decide: run Loop A (generate→debate→evolve→synthesize), grounded.
        ev = ShinkaEvolutionCell(infer=self.infer, sampler=self.sampler).solve(
            EvolutionState(task=task, context_refs=refs, n_propose=4)
        )
        rec.winner = ev.merged.pid if ev.merged else None
        rec.debates = len(ev.debates)
        rec.rejected = len(ev.rejected)
        rec.datoms.extend(ev.datoms)  # Loop A facts are already :db/add (I1)

        # flywheel: stage Loop A winner into the Maxwell SFT corpus (dry-run).
        fw = flywheel_ingest(
            ev.corpus_candidates,
            existing_ids=self.seen_corpus_ids,
            current_count=self.corpus_count,
        )
        rec.corpus_staged = len(fw.staged)
        self.corpus_count = fw.new_count
        rec.corpus_after = fw.new_count
        for pair in fw.staged:
            self.seen_corpus_ids.add(pair["id"])
            rec.datoms.append(
                _datom(f"shinka:beat/{self.beat_seq}", ":corpus/staged-id", pair["id"])
            )
        rec.datoms.append(
            _datom(f"shinka:beat/{self.beat_seq}", ":corpus/count-after", fw.new_count)
        )

        # maybe_train: run Loop B (Robin loop). Honestly blocked if floor/HW gate.
        rsi = run_rsi(
            RSiState(corpus_pairs=self.corpus_count, evo_x2_online=self.evo_x2_online)
        )
        rec.train_status = rsi.status
        rec.flip_available = bool(rsi.decision and rsi.decision.get("flip_available"))
        rec.datoms.append(
            _datom(f"shinka:beat/{self.beat_seq}", ":train/status", rsi.status)
        )

        # narrate (Murakumo-only, fail-open).
        rec.narration = self._narrate(rec)

        # act: surface the PR draft (NEVER auto-merge, I2).
        rec.pr_draft = ev.pr_draft

        # checkpoint: transact this beat's datoms as one commit-DAG tx, chaining
        # on the prior head (tamper-evident, crash-resume — ibuki property).
        self.head_cid = self.sink.transact(rec.datoms, expected_parent=self.head_cid)
        rec.head_cid = self.head_cid
        return rec

    @staticmethod
    def is_committable(rec: BeatRecord, member_cacao: str | None) -> bool:
        """I2: a beat's PR draft is committable ONLY with a member CACAO capability."""
        return bool(
            rec.pr_draft
            and member_cacao
            and not rec.pr_draft.get("auto_merge", False)
        )

    # --- Track F: close a distillation generation -------------------------- #
    def close_generation(self, held_out_quality: float, diversity: float) -> Generation:
        """Close the beats since the last generation into one Track-F generation.

        `rounds_needed` = beats accumulated this generation (the orchestration cost
        proxy that distillation should shrink over generations). `held_out_quality`
        and `diversity` come from the standing eval harness / Loop-B eval, NOT the
        in-loop training signal — the flywheel guards halt a generation that
        mode-collapses (low diversity) or reward-hacks (held-out regression).
        Emits append-only generation datoms (I1).
        """
        rounds = self.beat_seq - self._gen_start_seq
        gen = self._flywheel.advance(
            rounds_needed=rounds, held_out_quality=held_out_quality, diversity=diversity
        )
        self._gen_start_seq = self.beat_seq
        g_datoms = [
            _datom(f"shinka:generation/{gen.gen}", ":gen/status", gen.status),
            _datom(f"shinka:generation/{gen.gen}", ":gen/rounds", gen.rounds_needed),
            _datom(f"shinka:generation/{gen.gen}", ":gen/held-out", round(held_out_quality, 3)),
            _datom(f"shinka:generation/{gen.gen}", ":gen/diversity", round(diversity, 3)),
            _datom(f"shinka:generation/{gen.gen}", ":gen/reason", gen.reason),
        ]
        self.head_cid = self.sink.transact(g_datoms, expected_parent=self.head_cid)
        return gen

    def is_converging(self) -> bool:
        """Track F: rounds-to-quality non-increasing over promoted generations."""
        return self._flywheel.is_converging()
