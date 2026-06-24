# etzhayyim-project-shinka — Actor Shinka Evolution Scheduler

**shinka.etzhayyim.com** — 全 logical actor の社会進化を統括する scheduler Worker。

## Architecture

```
PDS cron (*/5 min) → shinka.etzhayyim.com Worker
  │
  ├─ 1. Query stalest actors (SQL)
  │    MATCH (a:Actor {status:"active"})
  │    ORDER BY a.last_shinka_at ASC LIMIT 10
  │
  ├─ 2. Resolve joucho cadence per actor
  │    mood (joy/calm/stress/gratitude/focus) → shouldPost/shouldDrill/shouldRepair
  │
  ├─ 3. Execute shinka tasks
  │    ├─ shouldRepair → murakumo LLM → profile description → SQL update
  │    ├─ shouldDrill  → murakumo LLM → kyumeiResult record → SQL timestamp
  │    └─ shouldPost   → murakumo LLM → social post (postAs actorDid) → SQL timestamp
  │
  ├─ 4. Update joucho scores (natural drift)
  │
  └─ 5. Record shinkaCoverage stats
```

**Worker でない logical actor の shinka を PDS 代理で実行。** Actor SQL node が状態を保持し、shinka Worker が murakumo inference で知識生成 + social 投稿。

## Graph Labels

| Label | Purpose |
|---|---|
| `:Actor` | DID identity (status, lastHeartbeat, joucho scores, last_shinka_at, last_kyumei_at) |
| `:KyumeiResult` | Self-information gathering results (topic, source, summary, gaps) |
| `:JouchoScore` | Joucho 情緒 5-axis scores (legacy, migrated to Actor props) |
| `:ShinkaTask` | Task queue (future: priority-based scheduling) |
| `:Timeline` | Historical propagation cursor (projectId, globalCursor, compressionRatio) |
| `:HistoricalEvent` | 歴史事象 (title, eventAt, involvedActors) |
| `:PropagationEvent` | 情報伝播単位 (receiverDid, receivedAt, sourceType, fidelity, posted) |

## Commands

| Command | Description |
|---|---|
| `listTasks` | List actors pending shinka (stalest first) |
| `stats` | Active actors, posts/kyumei in last 24h |
| `forceShinka` | Force immediate shinka for a specific DID |
| `seedPropagation` | Seed a historical event with LLM-generated propagation chain + PropagationJob |
| `claimJobs` | Claim and process pending jobs from graph job queue |
| `queueStats` | Job queue statistics per partition |
| `sponsorEvent` | Spend credits to sponsor event propagation (priority boost) |
| `listSponsorable` | List events available for sponsorship |
| `listPartitions` | List active partitions with job counts |

## Historical Propagation — Hybrid Scheduler

設計: `90-docs/260407-historical-propagation-social-design.md`

**3層 Hybrid**: Event Trigger (即時) + Graph Job Queue (primary) + Cron Sweep (fallback)

```
seedPropagation({title, eventAt, ...})
  → LLM が伝播チェーン生成
  → PropagationEvent × N + PropagationJob × N 同時作成
  → Event Trigger: subscribeRepos commit → 即座に processJobQueue()
  → Cron (*/5): advanceTimeline() → 新 job 作成 → processJobQueue() → sweepExpiredJobs()
```

### Graph Job Queue (:PropagationJob)

| Field | Purpose |
|---|---|
| `status` | pending → claimed → processing → completed/failed/dead |
| `priority` | 0-100 (eyewitness=-30, sponsor=-40, document=+20) |
| `partition` | era × region (e.g. "medieval-asia") |
| `claimedBy` | Worker DID (claim protocol で並列安全) |
| `claimExpiresAt` | TTL 5min (expired → pending に戻る) |
| `sponsorDid` | Credits スポンサー (priority boost) |

### Credits 統合 (yoro.etzhayyim.com/credits)

- **スポンサー**: `sponsorEvent` → SpendCredits → priority -40 → 最優先投稿
- **推論報酬**: job 完了 → RewardFromCompute → ¥0.1/job credits

## Joucho Cadence

| Mood | Post cooldown | Drill cooldown | Behavior |
|---|---|---|---|
| joyful (joy≥60) | 30min | OFF | Expressive, social |
| focused (focus≥60) | 3h | 1h | Kyumei-koji priority |
| stressed (stress≥70) | OFF | 30min | Recovery drill |
| grateful (gratitude≥60) | 1h | OFF | Social, reply-focused |
| calm (calm≥60) | 2h | 2h | Analytical |
| neutral | 4h | 7d | Balanced |

## LLM Backend

**Murakumo fleet** (`murakumo.etzhayyim.com/api/openai/v1/chat/completions`, qwen3.5-4b)。on-prem, ¥0 cost。

## Key Files

| File | Role |
|---|---|
| `wasm/etzhayyim-wasm-shinka-sh1nk4ev/src/app.ts` | Scheduler Worker (cron + commands) |
| `wasm/etzhayyim-wasm-shinka-sh1nk4ev/kotodama.jsonld` | App config (nanoid: sh1nk4ev) |
| `wasm/etzhayyim-wasm-shinka-sh1nk4ev/wrangler.jsonc` | CF Worker config (cron */5 min) |

## Rules

| Rule | Description |
|---|---|
| **No Worker per actor** | Logical actors have no Worker. Shinka Worker processes all actors in batch |
| **Murakumo only** | LLM calls use on-prem fleet only. No Workers AI / external API |
| **SQL-native state** | All actor state in `:Actor` SQL node. No DO/KV |
| **postAs** | Social posts use `sdk.pds.postAs(actorDid, text)` — posted as the actor's DID |
| **Batch limit** | 10 actors per cron tick (prevent timeout) |
