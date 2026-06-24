"""live_hooks — S1 live-fleet adapters (Murakumo infer + kotoba poster).

Per ADR-2606142200 (S1) + ADR-2605215000 (Murakumo-only inference). The S0 engine
keeps every network call behind an injected hook (`infer` for Loop-A/rank,
`poster` for KotobaBridgeSink) so it runs offline. This module IMPLEMENTS those
hooks for the live fleet — and is the ONLY place network I/O lives.

Two hard guards, enforced in code:
  * **Murakumo-only (I3 / Charter §2(i))** — `_assert_fleet_endpoint` refuses any
    URL whose host is not the loopback LiteLLM gateway, the EVO-X2 LAN address, or
    a `192.168.1.*` fleet node. A commercial endpoint (OpenAI/Vertex/Bedrock/…)
    cannot be reached through these hooks; it raises before any request.
  * **No-server-key (leash)** — `kotoba_poster` PRESENTS the opaque member CACAO
    capability in a header (present-only, never signs); without one the call still
    goes out but carries no authority, and KotobaBridgeSink already refuses to
    construct a committable transactor without it.

Network uses the stdlib only (urllib). All functions accept an injectable
`transport` so the contract is unit-tested with a fake — no live endpoint needed.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable
from urllib.parse import urlparse

# Murakumo fleet endpoint allowlist (ADR-2605215000 / CLAUDE.md GPU row).
MURAKUMO_GATEWAY = "http://127.0.0.1:4000"   # per-node LiteLLM loopback gateway
EVO_X2 = "192.168.1.70"                       # EVO-X2 ROCm pod (LiteLLM/Ollama)
_FLEET_HOSTS = {"127.0.0.1", "localhost", EVO_X2}


def _is_fleet_host(host: str) -> bool:
    return host in _FLEET_HOSTS or host.startswith("192.168.1.")


def _assert_fleet_endpoint(url: str) -> None:
    """I3: refuse any non-Murakumo endpoint (no commercial GPU, ADR-2605215000)."""
    host = urlparse(url).hostname or ""
    if not _is_fleet_host(host):
        raise ValueError(
            f"Murakumo-only: refused non-fleet endpoint {host!r} (ADR-2605215000); "
            "commercial GPU / vendor inference is constitutionally prohibited"
        )


# transport(url, headers, body_bytes) -> response dict (parsed JSON)
Transport = Callable[[str, dict[str, str], bytes], dict[str, Any]]


def _urllib_transport(url: str, headers: dict[str, str], body: bytes) -> dict[str, Any]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (fleet host only)
        return json.loads(resp.read().decode("utf-8"))


def murakumo_infer(
    endpoint: str = MURAKUMO_GATEWAY,
    model: str = "maxwell-1",
    transport: Transport | None = None,
) -> Callable[[str], str]:
    """Build the Loop-A `infer(prompt) -> str` hook against the Murakumo gateway.

    Targets the OpenAI-compatible `/v1/chat/completions` LiteLLM endpoint. Raises
    immediately for a non-fleet endpoint (I3). The returned callable raises on a
    transport/parse error so the engine's hooks fail OPEN to the deterministic
    kernel (the engine wraps every infer call in try/except).
    """
    _assert_fleet_endpoint(endpoint)
    send = transport or _urllib_transport
    url = endpoint.rstrip("/") + "/v1/chat/completions"

    def infer(prompt: str) -> str:
        body = json.dumps(
            {"model": model, "messages": [{"role": "user", "content": prompt}]}
        ).encode("utf-8")
        data = send(url, {"Content-Type": "application/json"}, body)
        return data["choices"][0]["message"]["content"]

    return infer


def kotoba_poster(
    transport: Transport | None = None,
) -> Callable[[str, dict[str, Any], str | None], str]:
    """Build the KotobaBridgeSink `poster(endpoint, payload, present_cacao) -> head`.

    POSTs the `datomic.transact` payload to the live kotoba engine and returns the
    engine's CIDv1 head. PRESENTS the member CACAO capability in a header
    (present-only, never signs — leash, ADR-2606111400). Refuses non-fleet hosts.
    """
    send = transport or _urllib_transport

    def poster(endpoint: str, payload: dict[str, Any], present_cacao: str | None) -> str:
        _assert_fleet_endpoint(endpoint)
        url = endpoint.rstrip("/") + "/datomic.transact"
        headers = {"Content-Type": "application/json"}
        if present_cacao:
            headers["X-CACAO"] = present_cacao  # presented, not a held key
        data = send(url, headers, json.dumps(payload).encode("utf-8"))
        return data["head"]

    return poster
