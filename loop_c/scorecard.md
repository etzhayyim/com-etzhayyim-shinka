# Shinka Loop C R0 — family ranking (real fitness)

_2026-06-17 · 1/4 scoreable · ADR-2606172200 R0 · ranks the existing family, invents nothing_

| rank | candidate | family | score | evidence | route |
|---|---|---|---|---|---|
| 1 | maxwell-diffusion-1 | diffusion | 0.640 | t2-task-measured | propose-candidate |
| 2 | maxwell-1 | ar | — | t2-landscape-only | insufficient-evidence |
| 3 | oka-mmsheaf | sheaf | — | t0-feasible-only | insufficient-evidence |
| 4 | baien-1.58 | bitnet | — | t0-feasible-only | insufficient-evidence |

**Honesty:** only `maxwell-diffusion-1` has the measured task signal (e7m micro 0.80) → scoreable. `maxwell-1` has a measured loss-landscape + train-loss but no microbench → insufficient until benched. `oka-mmsheaf` / `baien-1.58` have no measured task fitness (oka = R0, no weights). No architecture is invented or promoted; promotion is member-CACAO-gated.
