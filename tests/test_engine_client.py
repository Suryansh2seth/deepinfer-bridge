"""EngineClient lifecycle: rank crash, health propagation and drain-on-shutdown.

These drive the real multiprocessing IPC with engine_mock.worker (spawn context), so they
cover the contract engine_real.py must also satisfy.
"""

import asyncio

import pytest

from deepinfer_bridge.config import BridgeConfig
from deepinfer_bridge.engine_client import EngineClient, EngineUnavailable
from deepinfer_bridge.engine_mock import mock_worker

SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 100}


def crashing_worker(rank, cfg, cmd_q, evt_q):
    """A rank that dies during startup, with no 'ready' event and no 'fatal' event.

    Must be module-level so the spawn context can unpickle it by reference.
    """
    raise SystemExit(3)


def make_cfg(tokenizer_path, **overrides):
    fields = {"mock": True, "ep": 1, "dp": 1, "tokenizer_path": tokenizer_path, "startup_timeout_s": 60}
    fields.update(overrides)
    return BridgeConfig(**fields)


def test_rank_crash_during_startup_is_reported(tokenizer_path):
    cfg = make_cfg(tokenizer_path, startup_timeout_s=30)
    client = EngineClient(cfg, crashing_worker)

    async def scenario():
        with pytest.raises(EngineUnavailable):
            await client.start()
        assert client.healthy is False
        assert client.fatal, "a dead rank must be recorded as fatal"
        # New work is refused rather than silently queued to a dead rank.
        with pytest.raises(EngineUnavailable):
            client.submit([1, 2, 3], SAMPLING)

    asyncio.run(scenario())


def test_submit_before_start_is_refused(tokenizer_path):
    client = EngineClient(make_cfg(tokenizer_path), mock_worker)
    with pytest.raises(EngineUnavailable):
        client.submit([1, 2, 3], SAMPLING)


def test_start_reports_ready_ranks_and_stop_kills_them(tokenizer_path):
    cfg = make_cfg(tokenizer_path)
    client = EngineClient(cfg, mock_worker)

    async def scenario():
        await client.start()
        assert client.healthy is True
        assert client.stats()["ranks_ready"] == [0]
        procs = list(client._procs)
        assert procs and all(p.is_alive() for p in procs)

        await client.stop(0.0)
        assert client.healthy is False
        assert not any(p.is_alive() for p in procs), "no orphan engine processes"

    asyncio.run(scenario())


def test_stop_drains_an_in_flight_request_instead_of_killing_it(tokenizer_path, monkeypatch):
    monkeypatch.setenv("BRIDGE_MOCK_DELAY_MS", "1")
    cfg = make_cfg(tokenizer_path)
    client = EngineClient(cfg, mock_worker)

    async def scenario():
        await client.start()
        try:
            handle = client.submit([1, 2, 3], SAMPLING)

            async def consume():
                """Stand-in for the HTTP handler: pump events, then release."""
                while True:
                    event = await handle.queue.get()
                    if event[0] == "error" or event[3]:
                        break
                client.release(handle)

            await asyncio.gather(consume(), client.stop(drain_timeout=15.0))
            assert handle.finished is True, "the request should have completed, not been aborted"
            assert handle.released is True
            assert client.stats()["outstanding"] == 0
        finally:
            await client.stop(0.0)

    asyncio.run(scenario())


def test_stop_with_zero_grace_does_not_hang(tokenizer_path, monkeypatch):
    monkeypatch.setenv("BRIDGE_MOCK_DELAY_MS", "2000")
    cfg = make_cfg(tokenizer_path)
    client = EngineClient(cfg, mock_worker)

    async def scenario():
        await client.start()
        client.submit([1, 2, 3], SAMPLING)  # deliberately left in flight
        await asyncio.wait_for(client.stop(drain_timeout=0.0), timeout=30)
        assert client.healthy is False

    asyncio.run(scenario())


def test_load_balancing_uses_the_least_loaded_rank(tokenizer_path, monkeypatch):
    """Routing is least-outstanding-requests; with 2 ranks the two first picks alternate."""
    import multiprocessing as mp
    import queue as pyqueue
    import threading

    cfg = BridgeConfig(mock=False, ep=2, cp=1, dp=2, tokenizer_path=tokenizer_path, startup_timeout_s=5)
    # No workers are started: routing is pure bookkeeping over self.load.
    client = EngineClient(cfg, mock_worker)
    client._cmd_qs = [mp.get_context("spawn").Queue() for _ in range(2)]
    client.load = [0, 0]
    client.healthy = True
    client._stopping = threading.Event()

    handles = [client.submit([1, 2, 3], SAMPLING) for _ in range(4)]
    assert [h.rank for h in handles] == [0, 1, 0, 1], "least-outstanding routing should alternate"
    assert client.stats()["load_per_rank"] == [2, 2]

    for h in handles:
        client.release(h)
    assert client.stats()["load_per_rank"] == [0, 0]
    for q in client._cmd_qs:  # drain the queued aborts so nothing is left behind
        while True:
            try:
                q.get_nowait()
            except pyqueue.Empty:
                break
