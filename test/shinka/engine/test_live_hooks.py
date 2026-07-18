"""Pure-logic tests for live_hooks (S1 fleet adapters), via a fake transport.

Standalone-runnable (no network):
    python3 src/shinka/engine/test_live_hooks.py

Verifies the Murakumo-only endpoint guard (I3), the OpenAI-compatible infer
contract, the kotoba poster + CACAO presentation (leash), and that these hooks
compose with the existing engine (ShinkaEvolutionCell, KotobaBridgeSink).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "shinka" / "engine"))

from cell import EvolutionState, ShinkaEvolutionCell  # noqa: E402
from kotoba_sink import KotobaBridgeSink  # noqa: E402
from live_hooks import kotoba_poster, murakumo_infer  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {name}")


def test_infer_contract() -> None:
    seen = {}

    def fake(url, headers, body):
        seen["url"] = url
        seen["body"] = json.loads(body.decode())
        return {"choices": [{"message": {"content": "the answer"}}]}

    infer = murakumo_infer(transport=fake)
    out = infer("what is 2+2?")
    check("returns content", out == "the answer")
    check("hits chat-completions endpoint", seen["url"].endswith("/v1/chat/completions"))
    check("targets the loopback gateway", "127.0.0.1:4000" in seen["url"])
    check("sends the prompt", seen["body"]["messages"][0]["content"] == "what is 2+2?")
    check("uses maxwell-1 by default", seen["body"]["model"] == "maxwell-1")


def test_murakumo_only_guard() -> None:
    bad = False
    try:
        murakumo_infer(endpoint="https://api.openai.com")
    except ValueError as e:
        bad = "Murakumo-only" in str(e)
    check("I3: commercial endpoint refused", bad)
    # EVO-X2 + 192.168.1.* fleet nodes are allowed
    ok = True
    try:
        murakumo_infer(endpoint="http://192.168.1.70:4000", transport=lambda *a: {"choices": [{"message": {"content": "x"}}]})
        murakumo_infer(endpoint="http://192.168.1.17:4000", transport=lambda *a: {"choices": [{"message": {"content": "x"}}]})
    except ValueError:
        ok = False
    check("fleet LAN endpoints allowed", ok)


def test_poster_and_cacao() -> None:
    seen = {}

    def fake(url, headers, body):
        seen["url"] = url
        seen["headers"] = headers
        return {"head": "bafy-engine-real-cid"}

    poster = kotoba_poster(transport=fake)
    head = poster("http://127.0.0.1:8077", {"tx": "datomic.transact", "datoms": []}, "cacao_b64")
    check("returns engine head", head == "bafy-engine-real-cid")
    check("posts to datomic.transact", seen["url"].endswith("/datomic.transact"))
    check("presents CACAO header (not signs)", seen["headers"].get("X-CACAO") == "cacao_b64")


def test_poster_guard_and_no_cacao() -> None:
    poster = kotoba_poster(transport=lambda u, h, b: {"head": "h"})
    bad = False
    try:
        poster("http://evil.example.com", {}, "cacao")
    except ValueError:
        bad = True
    check("I3: poster refuses non-fleet host", bad)
    # without a cacao, no X-CACAO header is sent (leash: no authority presented)
    seen = {}

    def fake(u, h, b):
        seen["headers"] = h
        return {"head": "h"}

    kotoba_poster(transport=fake)("http://127.0.0.1:8077", {}, None)
    check("no CACAO header when none presented", "X-CACAO" not in seen["headers"])


def test_composes_with_engine() -> None:
    # the live infer hook drives the Loop-A rank node (fake fleet)
    def fake(url, headers, body):
        return {"choices": [{"message": {"content": "A"}}]}  # always picks A

    infer = murakumo_infer(transport=fake)
    out = ShinkaEvolutionCell(infer=infer).solve(EvolutionState(task="t", n_propose=4))
    check("engine runs with live infer hook", out.merged is not None)

    # the live poster drives a committable KotobaBridgeSink
    sink = KotobaBridgeSink(
        endpoint="http://127.0.0.1:8077",
        poster=kotoba_poster(transport=lambda u, h, b: {"head": "bafy-x"}),
        present_cacao="cacao_b64",
    )
    check("bridge sink committable with live poster + cacao", sink.committable is True)
    check("bridge transact returns engine head", sink.transact([{"e": "a", "a": ":x", "v": 1, "op": ":db/add"}]) == "bafy-x")


def main() -> int:
    for fn in (
        test_infer_contract,
        test_murakumo_only_guard,
        test_poster_and_cacao,
        test_poster_guard_and_no_cacao,
        test_composes_with_engine,
    ):
        fn()
    print(f"live_hooks: passed={_passed} failed={_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
