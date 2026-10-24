"""HTTP-process side of the engine: spawns one engine process per DP rank, routes requests,
streams results back into asyncio queues, tracks health.

IPC protocol (multiprocessing queues, spawn context):
  command  (HTTP -> rank): ("add", rid, token_ids:list[int], sampling:dict) | ("abort", rid) | ("stop",)
  event    (rank -> HTTP): ("ready", rank, info) | ("out", rid, tokens, finished, reason)
                           | ("aborted", rid) | ("fatal", rank, traceback_text)
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import multiprocessing as mp
import queue as pyqueue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.process import BaseProcess

from .config import BridgeConfig

log = logging.getLogger("deepinfer_bridge.engine")

DRAIN_POLL_S = 0.05  # how often to re-check the in-flight set while draining
STOP_JOIN_S = 15.0  # graceful exit budget per worker (DeepInfer teardown is not instant)
TERMINATE_JOIN_S = 5.0  # budget after escalating to SIGTERM


class EngineUnavailable(RuntimeError):
    pass


class EngineBusy(RuntimeError):
    pass


class EngineError(RuntimeError):
    pass


@dataclass
class Handle:
    rid: str
    rank: int
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    finished: bool = False
    released: bool = False


class EngineClient:
    def __init__(self, cfg: BridgeConfig, worker_target: Callable) -> None:
        self.cfg = cfg
        self._target = worker_target
        self._ids = itertools.count()
        self.handles: dict[str, Handle] = {}
        self.load = [0] * cfg.dp_size
        self.ready_ranks: set[int] = set()
        self.healthy = False
        self.fatal: str | None = None
        self.aborts_acked = 0
        self.total_requests = 0
        self._procs: list[BaseProcess] = []
        self._cmd_qs: list = []
        self._evt_q = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._ready = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ---- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        ctx = mp.get_context("spawn")
        self._evt_q = ctx.Queue()
        for rank in range(self.cfg.dp_size):
            cmd_q = ctx.Queue()
            # daemon=False: DeepInfer workers spawn their own child processes.
            proc = ctx.Process(target=self._target, args=(rank, self.cfg, cmd_q, self._evt_q), daemon=False)
            proc.start()
            self._cmd_qs.append(cmd_q)
            self._procs.append(proc)
        self._thread = threading.Thread(target=self._reader, name="engine-events", daemon=True)
        self._thread.start()
        try:
            await asyncio.wait_for(self._ready.wait(), self.cfg.startup_timeout_s)
        except asyncio.TimeoutError as exc:
            await self.stop()
            raise EngineUnavailable("engine did not become ready in time") from exc
        if self.fatal:
            await self.stop()
            raise EngineUnavailable(self.fatal)
        self.healthy = True
        log.info("engine ready: %d DP rank(s)", self.cfg.dp_size)

    async def stop(self, drain_timeout: float = 0.0) -> None:
        """Stop every worker. With drain_timeout > 0, first let in-flight requests finish.

        All ranks are lock-step: every rank calls has_unfinished_requests()/step() in the
        same sequence, and those contain collectives. They must therefore be asked to stop
        together and joined together -- stopping a subset would deadlock the rest.
        """
        loop = asyncio.get_running_loop()
        self.healthy = False

        if drain_timeout > 0 and self.handles:
            log.info("draining %d in-flight request(s), up to %.1fs", len(self.handles), drain_timeout)
            deadline = loop.time() + drain_timeout
            while self.handles and loop.time() < deadline:
                await asyncio.sleep(DRAIN_POLL_S)
            if self.handles:
                log.warning("drain deadline reached with %d request(s) still running", len(self.handles))

        self._stopping.set()
        # Abort anything left so no rank is still producing when the loop breaks.
        for handle in list(self.handles.values()):
            try:
                self._cmd_qs[handle.rank].put(("abort", handle.rid))
            except Exception:  # noqa: BLE001
                log.exception("failed to send abort for %s", handle.rid)
        self._fail_all("engine stopped")

        for q in self._cmd_qs:
            # A worker that already died cannot receive this; nothing to do about it.
            with contextlib.suppress(Exception):
                q.put(("stop",))
        for proc in self._procs:
            await loop.run_in_executor(None, proc.join, STOP_JOIN_S)
            if proc.is_alive():
                log.warning("engine rank pid=%s did not exit in %.0fs; terminating", proc.pid, STOP_JOIN_S)
                proc.terminate()
                await loop.run_in_executor(None, proc.join, TERMINATE_JOIN_S)

    # ---- request API -------------------------------------------------------
    def submit(self, token_ids: list[int], sampling: dict) -> Handle:
        """Admit a request. Raises EngineUnavailable / EngineBusy *before* any HTTP bytes are sent."""
        if not self.healthy:
            raise EngineUnavailable(self.fatal or "engine not ready")
        if len(self.handles) >= self.cfg.max_outstanding:
            raise EngineBusy("too many requests in flight")
        rank = min(range(len(self.load)), key=lambda r: (self.load[r], r))
        rid = f"r{next(self._ids)}"
        handle = Handle(rid=rid, rank=rank)
        self.handles[rid] = handle
        self.load[rank] += 1
        self.total_requests += 1
        self._cmd_qs[rank].put(("add", rid, token_ids, sampling))
        return handle

    def release(self, handle: Handle) -> None:
        """Idempotent. Aborts the request in the engine if it has not finished."""
        if handle.released:
            return
        handle.released = True
        self.handles.pop(handle.rid, None)
        self.load[handle.rank] = max(0, self.load[handle.rank] - 1)
        if not handle.finished and not self._stopping.is_set():
            try:
                self._cmd_qs[handle.rank].put(("abort", handle.rid))
            except Exception:  # noqa: BLE001
                log.exception("failed to send abort for %s", handle.rid)

    def stats(self) -> dict:
        return {
            "healthy": self.healthy,
            "outstanding": len(self.handles),
            "load_per_rank": list(self.load),
            "total_requests": self.total_requests,
            "aborts_acked": self.aborts_acked,
            "ranks_ready": sorted(self.ready_ranks),
            "fatal": self.fatal,
        }

    # ---- event plumbing ----------------------------------------------------
    def _reader(self) -> None:
        assert self._loop is not None
        assert self._evt_q is not None
        while not self._stopping.is_set():
            try:
                evt = self._evt_q.get(timeout=1.0)
            except pyqueue.Empty:
                for rank, proc in enumerate(self._procs):
                    if not proc.is_alive() and not self._stopping.is_set():
                        self._loop.call_soon_threadsafe(
                            self._on_fatal, f"engine rank {rank} exited (code {proc.exitcode})"
                        )
                        return
                continue
            self._loop.call_soon_threadsafe(self._on_event, evt)

    def _on_event(self, evt: tuple) -> None:
        kind = evt[0]
        if kind == "out":
            _, rid, tokens, finished, reason = evt
            handle = self.handles.get(rid)
            if handle is None:
                return  # released/aborted; drop late tokens
            if finished:
                handle.finished = True
            handle.queue.put_nowait(("out", tokens, finished, reason))
        elif kind == "ready":
            self.ready_ranks.add(evt[1])
            if len(self.ready_ranks) == self.cfg.dp_size:
                self._ready.set()
        elif kind == "aborted":
            self.aborts_acked += 1
        elif kind == "fatal":
            self._on_fatal(f"engine rank {evt[1]} failed:\n{evt[2]}")

    def _on_fatal(self, message: str) -> None:
        log.error(message)
        self.fatal = message
        self.healthy = False
        self._ready.set()  # unblock start(); it checks self.fatal
        self._fail_all(message)

    def _fail_all(self, message: str) -> None:
        for handle in list(self.handles.values()):
            handle.queue.put_nowait(("error", message))
