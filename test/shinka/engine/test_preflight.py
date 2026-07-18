"""Pure-logic tests for preflight (Loop-B training-readiness gate).

Standalone-runnable (no network):
    python3 src/shinka/engine/test_preflight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "shinka" / "engine"))

import inspect  # noqa: E402

from maxwell_rsi import CORPUS_TRAIN_FLOOR, run_rsi  # noqa: E402
from preflight import (  # noqa: E402
    fleet_preflight,
    rocm_http_probe,
    rsi_state_from_preflight,
    tailscale_ssh_probe,
)

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {name}")


_UP = lambda: True
_DOWN = lambda: False


def test_all_ready() -> None:
    v = fleet_preflight(CORPUS_TRAIN_FLOOR, gad_probe=_UP, rocm_probe=_UP)
    check("ready when all pass", v.ready is True)
    check("no blocked reasons", v.blocked_reasons == [])
    check("reason is 'ready'", v.reason == "ready")


def test_gad_offline_blocks() -> None:
    # the live state right now: corpus below floor + gad offline
    v = fleet_preflight(125, gad_probe=_DOWN, rocm_probe=_DOWN)
    check("not ready", v.ready is False)
    check("corpus flagged", any("corpus" in r for r in v.blocked_reasons))
    check("gad flagged", any("gad-tailscale" in r for r in v.blocked_reasons))
    check("rocm flagged", any("evo-x2-rocm" in r for r in v.blocked_reasons))


def test_missing_probe_degrades_honestly() -> None:
    # no probes supplied → unreachable, never assumed-ready
    v = fleet_preflight(CORPUS_TRAIN_FLOOR)
    check("missing probes → not ready", v.ready is False)
    check("gad unreachable by default", any("gad-tailscale" in r for r in v.blocked_reasons))


def test_corpus_floor_gate() -> None:
    below = fleet_preflight(CORPUS_TRAIN_FLOOR - 1, gad_probe=_UP, rocm_probe=_UP)
    check("below floor not ready", below.ready is False)
    at = fleet_preflight(CORPUS_TRAIN_FLOOR, gad_probe=_UP, rocm_probe=_UP)
    check("at floor ready", at.ready is True)


def test_rocm_probe_transport_injection() -> None:
    probe_up = rocm_http_probe(transport=lambda url: True)
    probe_down = rocm_http_probe(transport=lambda url: (_ for _ in ()).throw(OSError()))
    check("injected healthy probe", probe_up() is True)
    check("injected failing probe degrades to False", probe_down() is False)


def test_feeds_run_rsi_blocked() -> None:
    # not-ready preflight → RSiState → run_rsi honestly blocked, no flip
    v = fleet_preflight(900, gad_probe=_DOWN, rocm_probe=_DOWN)  # corpus ok, gad down
    st = rsi_state_from_preflight(v, corpus_pairs=900)
    check("evo_x2_online False from preflight", st.evo_x2_online is False)
    out = run_rsi(st)
    check("run_rsi blocked", out.status == "blocked")
    check("no flip", out.decision["flip_available"] is False)


def test_feeds_run_rsi_ready() -> None:
    # ready preflight + train hook clearing the gate → run_rsi proceeds + flips
    v = fleet_preflight(900, gad_probe=_UP, rocm_probe=_UP)
    st = rsi_state_from_preflight(v, corpus_pairs=900)
    check("evo_x2_online True from preflight", st.evo_x2_online is True)
    out = run_rsi(st, train_hook=lambda recipe: {"steps": 300}, eval_hook=lambda run: 1.0)
    check("run_rsi flips on step gate", out.decision["flip_available"] is True)


def test_defaults_use_tailscale_ip_not_stale_lan() -> None:
    # gad is Ubuntu now and its LAN IP changes -> probes target gad's stable
    # Tailscale IP (100.82.98.110), never the stale 192.168.1.70 LAN address.
    GAD_TS = "100.82.98.110"
    rocm_default = inspect.signature(rocm_http_probe).parameters["url"].default
    check("rocm probe default targets gad Tailscale IP", GAD_TS in rocm_default)
    check("rocm probe default is not the stale LAN IP", "192.168.1.70" not in rocm_default)
    ssh_default = inspect.signature(tailscale_ssh_probe).parameters["host"].default
    check("ssh probe default host is gad Tailscale IP", ssh_default == GAD_TS)


def main() -> int:
    for fn in (
        test_all_ready,
        test_gad_offline_blocks,
        test_missing_probe_degrades_honestly,
        test_corpus_floor_gate,
        test_rocm_probe_transport_injection,
        test_feeds_run_rsi_blocked,
        test_feeds_run_rsi_ready,
        test_defaults_use_tailscale_ip_not_stale_lan,
    ):
        fn()
    print(f"preflight: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
