"""Phase 5 hardening: overload, limits, timeouts, auth coverage, metrics, load.

These run against engine_mock.py, so they exercise the HTTP/protocol layer and the
engine_client IPC contract, never a real GPU.
"""

import asyncio
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

MODEL = "deepseek-v4.1-flash"


def body(**extra):
    return {"model": MODEL, "messages": [{"role": "user", "content": "Hi"}], "thinking": {"type": "disabled"}, **extra}


@contextmanager
def make_client(tokenizer_path, monkeypatch, delay_ms=5, **overrides):
    from starlette.testclient import TestClient

    from deepinfer_bridge.config import BridgeConfig
    from deepinfer_bridge.server import create_app

    monkeypatch.setenv("BRIDGE_MOCK_DELAY_MS", str(delay_ms))
    cfg = BridgeConfig(mock=True, ep=1, dp=1, tokenizer_path=tokenizer_path, startup_timeout_s=120, **overrides)
    with TestClient(create_app(cfg)) as c:
        yield c


def wait_outstanding_zero(client, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline and client.get("/stats").json()["outstanding"]:
        time.sleep(0.02)
    return client.get("/stats").json()["outstanding"]


NON_STREAM_OK = 'deepinfer_bridge_requests_total{endpoint="chat_completions",stream="false",status="ok"}'
STREAM_OK = 'deepinfer_bridge_requests_total{endpoint="chat_completions",stream="true",status="ok"}'


def sample(client, name: str) -> float:
    """Read one /metrics sample by its full name (including any labels).

    The metric registry is process-global, so tests compare deltas instead of absolute
    values -- otherwise a streaming request made by another test file would break them.
    """
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(name):
            head, _, value = line.rpartition(" ")
            if head == name:
                return float(value)
    return 0.0


# --- overload -------------------------------------------------------------------------


def test_overload_returns_503_with_retry_after(tokenizer_path, monkeypatch):
    with make_client(tokenizer_path, monkeypatch, delay_ms=300, max_outstanding=1) as c:
        barrier = threading.Barrier(2)

        def fire():
            barrier.wait()
            return c.post("/v1/chat/completions", json=body())

        with ThreadPoolExecutor(2) as pool:
            responses = [f.result() for f in [pool.submit(fire), pool.submit(fire)]]

    codes = sorted(r.status_code for r in responses)
    assert codes == [200, 503], [r.status_code for r in responses]
    rejected = next(r for r in responses if r.status_code == 503)
    assert rejected.headers["retry-after"] == "1"
    assert rejected.json()["error"]["type"] == "overloaded_error"


# --- request body limit ---------------------------------------------------------------


def test_oversized_body_is_rejected_413(tokenizer_path, monkeypatch):
    with make_client(tokenizer_path, monkeypatch, max_body_bytes=256) as c:
        payload = body(messages=[{"role": "user", "content": "x" * 5000}])
        r = c.post("/v1/chat/completions", json=payload)
    assert r.status_code == 413
    assert "exceeds" in r.json()["error"]["message"]


def test_body_within_the_limit_still_works(tokenizer_path, monkeypatch):
    with make_client(tokenizer_path, monkeypatch, max_body_bytes=4096) as c:
        assert c.post("/v1/chat/completions", json=body()).status_code == 200


# --- request timeout ------------------------------------------------------------------


def test_request_timeout_returns_504(tokenizer_path, monkeypatch):
    # The mock needs 400 ms per token; the budget is far shorter.
    with make_client(tokenizer_path, monkeypatch, delay_ms=400, request_timeout_s=0.05) as c:
        r = c.post("/v1/chat/completions", json=body())
    assert r.status_code == 504
    assert r.json()["error"]["type"] == "timeout_error"


def test_timeout_aborts_the_engine_request(tokenizer_path, monkeypatch):
    with make_client(tokenizer_path, monkeypatch, delay_ms=400, request_timeout_s=0.05) as c:
        c.post("/v1/chat/completions", json=body())
        assert wait_outstanding_zero(c) == 0


# --- auth on every route --------------------------------------------------------------


PROTECTED_GET = ["/v1/models", "/models", "/stats", "/metrics"]

# Each endpoint has its own request shape; a chat-shaped body is invalid for /v1/responses.
PROTECTED_POST = [
    ("/v1/chat/completions", body()),
    ("/v1/responses", {"model": MODEL, "input": "Hi", "reasoning": {"effort": "none"}}),
    ("/v1/messages", {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": "Hi"}]}),
]


def test_every_protected_route_requires_the_key(tokenizer_path, monkeypatch):
    with make_client(tokenizer_path, monkeypatch, api_key="secret") as c:
        for path in PROTECTED_GET:
            assert c.get(path).status_code == 401, path
        for path, payload in PROTECTED_POST:
            assert c.post(path, json=payload).status_code == 401, path


def test_every_protected_route_accepts_the_key(tokenizer_path, monkeypatch):
    headers = {"Authorization": "Bearer secret"}
    with make_client(tokenizer_path, monkeypatch, api_key="secret") as c:
        for path in PROTECTED_GET:
            assert c.get(path, headers=headers).status_code == 200, path
        for path, payload in PROTECTED_POST:
            r = c.post(path, json=payload, headers=headers)
            assert r.status_code == 200, (path, r.text)


def test_healthz_stays_open_so_a_supervisor_can_probe_it(tokenizer_path, monkeypatch):
    with make_client(tokenizer_path, monkeypatch, api_key="secret") as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/healthz", headers={"Authorization": "Bearer wrong"}).status_code == 200


def test_wrong_and_malformed_keys_are_rejected(tokenizer_path, monkeypatch):
    with make_client(tokenizer_path, monkeypatch, api_key="secret") as c:
        for header in ("Bearer wrong", "secret", "Bearer secret ", ""):
            r = c.get("/v1/models", headers={"Authorization": header})
            assert r.status_code == 401, header


def test_401_body_is_openai_shaped(tokenizer_path, monkeypatch):
    with make_client(tokenizer_path, monkeypatch, api_key="secret") as c:
        err = c.get("/v1/models").json()["error"]
    assert err["type"] == "authentication_error" and err["code"] == "invalid_api_key"


# --- metrics --------------------------------------------------------------------------


def test_metrics_endpoint_exposes_prometheus_text(tokenizer_path, monkeypatch):
    with make_client(tokenizer_path, monkeypatch) as c:
        before = sample(c, NON_STREAM_OK)
        assert c.post("/v1/chat/completions", json=body()).status_code == 200
        r = c.get("/metrics")
        after = sample(c, NON_STREAM_OK)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    text = r.text
    assert "# TYPE deepinfer_bridge_requests_total counter" in text
    assert "# TYPE deepinfer_bridge_in_flight gauge" in text
    assert "# TYPE deepinfer_bridge_ttft_seconds histogram" in text
    assert "deepinfer_bridge_ttft_seconds_bucket{" in text
    assert 'deepinfer_bridge_tokens_total{kind="prompt"}' in text
    assert after == before + 1


def test_metrics_track_streaming_requests_separately(tokenizer_path, monkeypatch):
    with make_client(tokenizer_path, monkeypatch) as c:
        before = sample(c, STREAM_OK)
        with c.stream("POST", "/v1/chat/completions", json=body(stream=True)) as r:
            list(r.iter_lines())
        after = sample(c, STREAM_OK)
        text = c.get("/metrics").text
    assert 'stream="true"' in text
    assert after == before + 1, "a streamed request is counted separately from a buffered one"


def test_in_flight_returns_to_zero_after_requests(tokenizer_path, monkeypatch):
    with make_client(tokenizer_path, monkeypatch) as c:
        c.post("/v1/chat/completions", json=body())
        assert sample(c, "deepinfer_bridge_in_flight") == 0


# --- stop sequences combined with max_tokens ------------------------------------------


def test_stop_and_max_tokens_together(tokenizer_path, monkeypatch):
    with make_client(tokenizer_path, monkeypatch) as c:
        r = c.post("/v1/chat/completions", json=body(stop=["Hello"], max_tokens=2))
        assert r.status_code == 200, r.text
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] in {"stop", "length"}
        assert wait_outstanding_zero(c) == 0


# --- abort on release -----------------------------------------------------------------


def test_release_aborts_an_unfinished_request_at_the_ipc_layer(tokenizer_path, monkeypatch):
    """The mechanism behind 'client disconnects -> engine work stops'.

    Driven at the EngineClient level because that is where the abort is actually sent;
    the HTTP layer only calls release().
    """
    from deepinfer_bridge.config import BridgeConfig
    from deepinfer_bridge.engine_client import EngineClient
    from deepinfer_bridge.engine_mock import mock_worker

    monkeypatch.setenv("BRIDGE_MOCK_DELAY_MS", "500")  # nothing is emitted within the window below
    cfg = BridgeConfig(mock=True, ep=1, dp=1, tokenizer_path=tokenizer_path, startup_timeout_s=120)

    async def scenario():
        client = EngineClient(cfg, mock_worker)
        await client.start()
        try:
            handle = client.submit([1, 2, 3], {"max_tokens": 1000, "temperature": 0.0, "top_p": 1.0, "top_k": -1})
            await asyncio.sleep(0.05)
            assert client.stats()["outstanding"] == 1

            client.release(handle)
            assert client.stats()["outstanding"] == 0

            for _ in range(100):  # the worker acknowledges the abort
                if client.aborts_acked:
                    break
                await asyncio.sleep(0.02)
            assert client.aborts_acked >= 1
        finally:
            await client.stop(0.0)

    asyncio.run(scenario())


def test_release_is_idempotent(tokenizer_path, monkeypatch):
    from deepinfer_bridge.config import BridgeConfig
    from deepinfer_bridge.engine_client import EngineClient
    from deepinfer_bridge.engine_mock import mock_worker

    cfg = BridgeConfig(mock=True, ep=1, dp=1, tokenizer_path=tokenizer_path, startup_timeout_s=120)

    async def scenario():
        client = EngineClient(cfg, mock_worker)
        await client.start()
        try:
            handle = client.submit([1, 2, 3], {"max_tokens": 5, "temperature": 0.0, "top_p": 1.0, "top_k": -1})
            client.release(handle)
            client.release(handle)  # must not drive the load counter negative
            assert client.stats()["outstanding"] == 0
            assert client.stats()["load_per_rank"] == [0]
        finally:
            await client.stop(0.0)

    asyncio.run(scenario())


# --- load -----------------------------------------------------------------------------


def test_hundred_concurrent_requests_leak_no_handles(tokenizer_path, monkeypatch):
    import httpx

    from deepinfer_bridge.config import BridgeConfig
    from deepinfer_bridge.server import create_app

    monkeypatch.setenv("BRIDGE_MOCK_DELAY_MS", "5")
    cfg = BridgeConfig(mock=True, ep=1, dp=1, tokenizer_path=tokenizer_path, startup_timeout_s=120)
    app = create_app(cfg)

    async def scenario():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
                responses = await asyncio.gather(*(ac.post("/v1/chat/completions", json=body()) for _ in range(100)))
                assert all(r.status_code == 200 for r in responses), sorted(r.status_code for r in responses)
                stats = (await ac.get("/stats")).json()
                assert stats["outstanding"] == 0
                assert stats["total_requests"] == 100
                assert stats["load_per_rank"] == [0]

    asyncio.run(scenario())


# --- import hygiene -------------------------------------------------------------------


def test_http_process_never_imports_torch_or_deepinfer():
    """The HTTP process must not initialise CUDA: per-rank CUDA_VISIBLE_DEVICES is only
    set inside the spawned workers, and importing torch here would defeat it."""
    import deepinfer_bridge.server  # noqa: F401

    assert "torch" not in sys.modules
    assert "deepinfer" not in sys.modules


def test_missing_tokenizer_is_a_clear_startup_error(tmp_path):
    from deepinfer_bridge.config import BridgeConfig
    from deepinfer_bridge.server import create_app

    cfg = BridgeConfig(mock=True, ep=1, dp=1, tokenizer_path=str(tmp_path / "nope.json"), startup_timeout_s=10)
    with pytest.raises((OSError, ValueError, RuntimeError)):
        create_app(cfg)
