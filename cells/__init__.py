"""shinka self-evolution Pregel cell catalog (S0 scaffold; all outward paths gated).

Per ADR-2606142200 (Shinka self-evolution engine). This is the CAPABILITY-evolution
half (Loop A) — the Co-scientist generate→debate→evolve cycle mapped onto a single
LangGraph super-step graph driven by the Supervisor (shinka_orchestrator). It is a
SIBLING of, not a replacement for, the existing shinka social-evolution scheduler
(actor-manifest.jsonld pipelines); the two share the `shinka` DID and joucho cadence
but evolve different things (this one evolves actors/cells/code, that one evolves
social posting cadence).

Invariants (ADR-2606142200, non-negotiable — enforced in code + tests):
  I1 :db/add only — every proposal/debate/Elo update is an append-only datom; the
     evolution history is itself immutable evidence (no retraction).
  I2 No autonomous merge — `synthesize` emits a PR DRAFT with member_signed=False /
     auto_merge=False; an outward commit requires a member-Ed25519 CACAO capability
     (ADR-2606111400). The cell PRESENTS, it never signs.
  I3 Murakumo-only — all inference (propose/debate/rank) resolves to the Murakumo
     fleet; commercial GPU is constitutionally prohibited (ADR-2605215000). Inference
     fails OPEN to a deterministic template (never a fake commercial call).

S0 status: the engine graph + 7 node functions are implemented and run end-to-end
on a deterministic (LLM-free) kernel; Murakumo debate is a typed hook with template
fallback. Corpus emission into the Loop-B Maxwell flywheel is DRY-RUN only
(`corpus_candidates` staged in state, never written — operator/leash-gated).
"""
