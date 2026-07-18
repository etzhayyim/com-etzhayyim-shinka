"""preflight — Loop-B training-readiness gate (gad / EVO-X2 / corpus).

Per ADR-2606142200 (S1) + ADR-2606130900 (Maxwell RSi). Before Loop B can run a
real EVO-X2 fine-tune, three conditions must hold: the corpus has reached the
training floor, the EVO-X2 pod (`gad`, Ubuntu ROCm gfx1151) is reachable over the
Tailscale overlay, and its ROCm inference endpoint is healthy. This module turns
those into one structured, honest verdict that decides whether `run_rsi` may
proceed or must stay `blocked` — wiring `RSiState.evo_x2_online` to a real probe
instead of a hand-set flag.

`gad` is addressed by its **Tailscale IP** (`100.82.98.110`), NOT its LAN IP: the
pod was re-imaged Ubuntu 24.04 and its DHCP LAN address changes (currently .16),
whereas the Tailscale IP is stable per-node and reachable from any fleet host
(MagicDNS short-names do not resolve from every client, so the IP is used — same
convention as ~/.ssh/config and the rest of the fleet). Probes are INJECTABLE (the
live ones do network I/O, isolated like live_hooks); a missing probe degrades
honestly to "unreachable" rather than assuming readiness. Murakumo-only: the probes only ever
touch the fleet (gad over Tailscale), never a commercial host.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from .maxwell_rsi import CORPUS_TRAIN_FLOOR, RSiState
except Exception:  # pragma: no cover - standalone import path
    from maxwell_rsi import CORPUS_TRAIN_FLOOR, RSiState


@dataclass
class ProbeResult:
    name: str
    ok: bool
    detail: str


@dataclass
class PreflightVerdict:
    ready: bool
    results: list[ProbeResult] = field(default_factory=list)
    reason: str = ""

    @property
    def blocked_reasons(self) -> list[str]:
        return [f"{r.name}: {r.detail}" for r in self.results if not r.ok]


def tailscale_ssh_probe(
    host: str = "100.82.98.110", port: int = 22, timeout: float = 3.0
) -> Callable[[], bool]:
    """Build a probe that TCP-connects to `host:port` (gad's SSH over Tailscale).

    Defaults to gad's stable Tailscale IP (`100.82.98.110`). Returns a no-arg
    callable so the verdict stays lazy/injectable. The connection is the only
    network I/O; it targets the fleet (Tailscale), never a vendor.
    """

    def probe() -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    return probe


def rocm_http_probe(
    url: str = "http://100.82.98.110:11434/v1/models",
    transport: Callable[[str], bool] | None = None,
) -> Callable[[], bool]:
    """Build a probe for the EVO-X2 ROCm inference endpoint health (/v1/models).

    Defaults to gad's stable Tailscale IP (`100.82.98.110`; the Ubuntu pod's LAN
    IP changes, so the Tailscale IP is used). `transport(url) -> bool` is
    injectable for tests; the default does a minimal stdlib GET and treats a 200
    as healthy. Fleet host only. NOTE: the Ollama endpoint awaits Ubuntu
    re-provision (was a Windows Scheduled Task) — until then this probe fails,
    and the readiness gate stays honestly blocked.
    """

    def probe() -> bool:
        if transport is not None:
            try:
                return bool(transport(url))
            except Exception:
                return False
        try:  # pragma: no cover - live path, exercised only against the real pod
            import urllib.request

            with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 (fleet host)
                return resp.status == 200
        except Exception:
            return False

    return probe


def fleet_preflight(
    corpus_pairs: int,
    gad_probe: Callable[[], bool] | None = None,
    rocm_probe: Callable[[], bool] | None = None,
    train_floor: int = CORPUS_TRAIN_FLOOR,
) -> PreflightVerdict:
    """Probe the three training preconditions and return a structured verdict.

    A missing probe degrades to `False` (honest "unreachable") — readiness is
    never assumed. `ready` is true only when ALL probes pass.
    """
    results: list[ProbeResult] = []

    corpus_ok = corpus_pairs >= train_floor
    results.append(
        ProbeResult("corpus", corpus_ok, f"{corpus_pairs}/{train_floor} pairs")
    )

    gad_ok = bool(gad_probe()) if gad_probe is not None else False
    results.append(
        ProbeResult(
            "gad-tailscale",
            gad_ok,
            "reachable" if gad_ok else "unreachable (Tailscale offline)",
        )
    )

    rocm_ok = bool(rocm_probe()) if rocm_probe is not None else False
    results.append(
        ProbeResult(
            "evo-x2-rocm",
            rocm_ok,
            "healthy" if rocm_ok else "no /v1/models (pod down or unreachable)",
        )
    )

    ready = all(r.ok for r in results)
    reason = "ready" if ready else "; ".join(f"{r.name}: {r.detail}" for r in results if not r.ok)
    return PreflightVerdict(ready=ready, results=results, reason=reason)


def rsi_state_from_preflight(
    verdict: PreflightVerdict, corpus_pairs: int, base_model: str = "google/gemma-4-E4B"
) -> RSiState:
    """Build an RSiState whose `evo_x2_online` reflects the live preflight verdict.

    The gad/Tailscale + ROCm probes decide `evo_x2_online`; the corpus floor is
    already enforced inside the Robin loop (`node_hypothesis`). So a not-ready
    preflight yields a state that `run_rsi` honestly reports as `blocked`.
    """
    gad = next((r.ok for r in verdict.results if r.name == "gad-tailscale"), False)
    rocm = next((r.ok for r in verdict.results if r.name == "evo-x2-rocm"), False)
    return RSiState(
        base_model=base_model,
        corpus_pairs=corpus_pairs,
        evo_x2_online=bool(gad and rocm),
    )
