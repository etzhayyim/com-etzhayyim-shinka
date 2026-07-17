# Shinka ASI outer loop

The operational ASI boundary is `com.etzhayyim.shinka.asi/beat-plan`.  It is
pure Clojure/ClojureScript: hosts execute returned effects, so the same plan is
usable by a Kotoba WASM component, a JVM actor, or a cloud-murakumo relay.

## Shared libraries

- `kotoba-lang/murakumo`: `murakumo.infer.evolution/dispatch-plan` turns a
  healthy fleet observation into bounded, reproducible cloud-murakumo jobs.
  Unknown or unhealthy nodes never receive work.
- `kotoba-lang/kotoba-fleet`: `kotoba.fleet.evolution/promotion-verdict`
  requires constitutional attestations, a content-addressed evidence CID, and
  either 250 benchmark steps or a +5pp measured improvement.

## Safety and maturity boundary

A successful evaluation is not a deployment.  The only possible promotion
result is `:human-signoff`; the host must submit it to the append-only
`kotoba-fleet` governor and obtain a member CACAO signature before any
materialization.  The ASI actor never calls git, changes model weights, writes
to a registry, or opens a network connection itself.

Activation must first supply a fresh healthy `:nodes` observation.  While the
fleet is degraded, every beat records a blocked state and emits no dispatch
effect.  This makes recovery observable without converting a stale dashboard
into autonomous work.

Run the actor contract tests with `bb test`.  Check a fresh fleet observation
before an operator dispatches a beat:

```sh
bb asi-check examples/asi-beat.edn
bb asi-check --live examples/asi-beat.edn
```

`--live` reads `MURAKUMO_CLOUD/infer/fleet` (default:
`https://api.murakumo.cloud`) and normalizes its public `status`, `roles`,
`ram-gb`, and shard-ceiling telemetry into inference capacity. It also checks
`/infer/models`: only an entry whose status is `serving` can be dispatched. A beat also requires stable candidate/benchmark IDs,
1–32 nonblank prompts, a positive model shape, and no more than 4096 tokens per
prompt. The command is plan-only. It exits `0` only when dispatch is ready;
expected fleet blocks, invalid input, or observation failures exit `2` and emit
no enqueue effect.

When a serving catalog entry advertises an OpenAI-compatible endpoint, Shinka
emits `:cloud-murakumo/chat-completions` jobs for that gateway rather than
inventing a public-fleet shard placement. The effect remains plan-only; a host
with the appropriate capability executes it and records the evidence result.
Before emitting that effect, Shinka also compares the requested model ID with
the gateway's own `/v1/models` response. A control-plane/runtime disagreement
is `:runtime-model-unverified`. It emits only an idempotent
`kotoba-fleet` reconciliation proposal that requires member CACAO; no model
catalog or inference endpoint is changed autonomously.

For the currently observed Qwen3.6 runtime, the reviewed recommendation is
`qwen3.6-35b-a3b` (40 layers, 21.2GB GGUF registry size), replacing the stale
`qwen-agentworld-35b-a3b` serving entry. The exact signed proposal is in
`examples/qwen3.6-model-reconciliation.edn`.

Inspect the exact catalog writes with:

```sh
bb asi-reconcile examples/qwen3.6-model-reconciliation.edn
```

Only a host that has an explicit member capability may add `--apply`; it
requires `MURAKUMO_CACAO` and performs the reviewed upsert/retire sequence.
