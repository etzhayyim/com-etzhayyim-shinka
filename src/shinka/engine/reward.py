"""reward — Research Track E: verifier-grounded reward + preference pairs.

Per ADR-2606142200 §Research Program Track E. The reward for a Shinka-evolved
candidate is OUTCOME-GROUNDED, never synthetic:

    reward = Charter-gate pass (HARD veto)
           + microbench delta (e7m bench micro, pp vs gemma-4-e4b-it)
           + real PR-merge outcome (Kaizen already reads `gh pr state`:
             MERGED → +1, CLOSED → -1, OPEN → 0)

Charter is a hard gate, not a weighted term: a charter-failing candidate can
NEVER earn a positive reward (it is vetoed to -inf). The two graded terms are
both grounded in reality — a measured benchmark delta and a human/Council merge
decision — so the preference signal cannot be gamed by self-rated quality.

`build_preference_pair` turns two scored candidates into a (chosen, rejected)
DPO pair, feeding Loop B's preference tuning. The chosen side is always
charter-clean (I-safety); a charter-vetoed candidate can only be the rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

# gh pr state → grounded outcome signal (Kaizen ADR-2605240200 / ibuki Wave-4).
PR_OUTCOME: dict[str, float] = {"MERGED": 1.0, "CLOSED": -1.0, "OPEN": 0.0}

CHARTER_VETO = float("-inf")  # a charter-failing candidate is never preferable

# Default weights for the two grounded, graded terms (sum to 1.0).
W_MICROBENCH = 0.5
W_PR = 0.5
# +10pp microbench ≈ a full +1 on the microbench term (saturating).
MICROBENCH_FULL_PP = 10.0


@dataclass
class RewardComponents:
    """The grounded inputs to a candidate's reward."""

    charter_ok: bool
    microbench_delta_pp: float = 0.0  # e7m bench micro delta (pp)
    pr_outcome: str | None = None     # gh pr state: MERGED / CLOSED / OPEN


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def aggregate_reward(
    rc: RewardComponents,
    w_microbench: float = W_MICROBENCH,
    w_pr: float = W_PR,
) -> float:
    """Scalar reward in [-1, 1], or CHARTER_VETO (-inf) if the Charter gate fails."""
    if not rc.charter_ok:
        return CHARTER_VETO  # hard gate — never positive, never weighted around
    mb = _clamp(rc.microbench_delta_pp / MICROBENCH_FULL_PP)
    pr = PR_OUTCOME.get((rc.pr_outcome or "").upper(), 0.0)
    return w_microbench * mb + w_pr * pr


@dataclass
class ScoredCandidate:
    cid: str
    rc: RewardComponents


@dataclass
class PreferencePair:
    chosen: str
    rejected: str
    reward_chosen: float
    reward_rejected: float
    margin: float


def build_preference_pair(
    a: ScoredCandidate,
    b: ScoredCandidate,
    margin: float = 0.1,
) -> PreferencePair | None:
    """Build a (chosen, rejected) DPO pair from two scored candidates.

    Returns None when neither is charter-clean, or when both are charter-clean
    but their rewards are within `margin` (too close to teach a preference).
    The chosen side is ALWAYS charter-clean.
    """
    ra, rb = aggregate_reward(a.rc), aggregate_reward(b.rc)
    if ra == CHARTER_VETO and rb == CHARTER_VETO:
        return None  # nothing to prefer; both vetoed
    # Winner = higher reward (a charter veto = -inf always loses).
    if ra >= rb:
        win, lose, rw, rl = a, b, ra, rb
    else:
        win, lose, rw, rl = b, a, rb, ra
    if not win.rc.charter_ok:  # defensive — cannot happen given veto = -inf
        return None
    # Margin gate only applies when BOTH are finite (both charter-clean).
    if rl != CHARTER_VETO and (rw - rl) < margin:
        return None
    return PreferencePair(
        chosen=win.cid,
        rejected=lose.cid,
        reward_chosen=rw,
        reward_rejected=rl,
        margin=(rw - rl),  # may be +inf when the loser is charter-vetoed
    )


def build_preference_corpus(
    groups: dict[str, list[ScoredCandidate]],
    margin: float = 0.1,
) -> list[PreferencePair]:
    """Turn per-prompt scored candidates into a Loop-B DPO preference corpus.

    `groups` maps a prompt/task id → its candidate completions (e.g. the Loop-A
    samples for one beat). For each group the highest-reward, charter-clean
    candidate is the `chosen`, paired against every other candidate that clears
    the margin (a charter-vetoed candidate always pairs). Groups whose best is
    charter-vetoed, or with < 2 candidates, contribute nothing. The result feeds
    Loop B's preference tuning (Track E), grounded in real outcomes.
    """
    pairs: list[PreferencePair] = []
    for cands in groups.values():
        if len(cands) < 2:
            continue
        best = max(cands, key=lambda c: aggregate_reward(c.rc))
        if not best.rc.charter_ok:
            continue  # no charter-clean chosen in this group
        for other in cands:
            if other is best:
                continue
            p = build_preference_pair(best, other, margin)
            if p is not None:
                pairs.append(p)
    return pairs
