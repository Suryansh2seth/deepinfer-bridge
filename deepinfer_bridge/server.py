"""FastAPI app: OpenAI/Responses/Messages-compatible endpoints backed by DeepInfer.

Response processing (thinking / tool-call parsing, stop sequences, SSE chunk shapes, usage) is
delegated to deepseek-recipe's StreamProcessor; the engine only ever sees token IDs.
Structure follows deepseek-recipe's server-py example (MIT).

This module must never import torch or deepinfer: it runs in the HTTP process, and importing
either would initialise CUDA there and defeat the per-rank CUDA_VISIBLE_DEVICES setup that has
to happen inside the spawned engine workers.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import time
from uuid import uuid4

from anyio import CancelScope
from deepseek_recipe import (
    ConversionError,
    InferenceChunk,
    InferenceFinishReason,
    PromptUsage,
    StreamProcessor,
    Tokenizer,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

from . import __version__, metrics
from .config import BridgeConfig
from .engine_client import EngineBusy, EngineClient, EngineError, EngineUnavailable, Handle
from .protocol import (
    PROTOCOL_TYPES,
    PreparedRequest,
    RequestError,
    build_sampling,
    conversion_options,
    make_encoding,
    prepare_request,
)

log = logging.getLogger("deepinfer_bridge")

_FINISH = {"stop": InferenceFinishReason.Stop, "length": InferenceFinishReason.Length}

# How long a client should wait before retrying an overloaded request.
RETRY_AFTER_S = 1

# OpenAI-style error `type` per HTTP status, so clients can branch on it.
_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    404: "not_found_error",
    413: "invalid_request_error",
    429: "rate_limit_error",
    500: "internal_error",
    503: "overloaded_error",
    504: "timeout_error",
}


@dataclass
class RequestStats:
    """Per-request accounting for logs and metrics. Finalised at most once."""

    request_id: str
    endpoint: str
    stream: bool
    model: str
    prompt_tokens: int
    started_at: float = field(default_factory=time)
    first_token_at: float | None = None
    completion_tokens: int = 0
    finish_reason: str | None = None
    completed: bool = False
    finalized: bool = False

    def record_tokens(self, count: int) -> None:
        if count and self.first_token_at is None:
            self.first_token_at = time()
        self.completion_tokens += count


def resolve_tokenizer_path(cfg: BridgeConfig) -> str:
    if cfg.tokenizer_path:
        return cfg.tokenizer_path
    if cfg.mock:
        raise SystemExit("--mock needs --tokenizer-path (e.g. deepseek-recipe/static/tokenizers/v41/tokenizer.json)")
    from .engine_real import resolve_model_dir

    return os.path.join(resolve_model_dir(cfg.model_path), "tokenizer.json")


def sse_frame(event: str | None, data: str) -> str:
    prefix = f"event: {event}\n" if event is not None else ""
    return f"{prefix}data: {data}\n\n"


def _error_payload(message: str, status_code: int, code: str | None = None) -> dict:
    kind = _ERROR_TYPES.get(status_code, "internal_error")
    return {"error": {"message": message, "type": kind, "param": None, "code": code or kind}}


def _finalize(stats: RequestStats, status: str) -> None:
    """Emit the one structured log line and the metrics for a finished request."""
    if stats.finalized:
        return
    stats.finalized = True

    duration = time() - stats.started_at
    ttft = None if stats.first_token_at is None else stats.first_token_at - stats.started_at
    metrics.IN_FLIGHT.dec()
    metrics.DURATION.observe(duration)
    if ttft is not None:
        metrics.TTFT.observe(ttft)
    metrics.TOKENS.inc(stats.prompt_tokens, kind="prompt")
    metrics.TOKENS.inc(stats.completion_tokens, kind="completion")
    metrics.REQUESTS.inc(endpoint=stats.endpoint, stream="true" if stats.stream else "false", status=status)

    log.info(
        "request complete",
        extra={
            "event": {
                "request_id": stats.request_id,
                "endpoint": stats.endpoint,
                "model": stats.model,
                "stream": stats.stream,
                "status": status,
                "finish_reason": stats.finish_reason or "",
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.completion_tokens,
                "duration_ms": round(duration * 1000, 1),
                "ttft_ms": None if ttft is None else round(ttft * 1000, 1),
            }
        },
    )


class InferenceStreamingResponse(StreamingResponse):
    """Close the iterator and release the engine request after a disconnect or failed send.
    Needed because an async generator that never started ignores aclose()."""

    def __init__(self, *args, on_close=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._on_close = on_close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with CancelScope(shield=True):
                close = getattr(self.body_iterator, "aclose", None)
                if close is not None:
                    await close()
            if self._on_close:
                self._on_close()


async def token_chunks(
    client: EngineClient,
    handle: Handle,
    prompt_tokens: int,
    stats: RequestStats,
    deadline: float,
) -> AsyncIterator[InferenceChunk]:
    """Engine events -> recipe InferenceChunks. Leaving this generator (normally, on stop-sequence
    early exit, on client disconnect, or on timeout) releases the handle, which aborts unfinished
    work in the engine.

    The deadline is enforced here rather than with an outer timeout so that a stalled engine
    surfaces as a clean RequestError and the abort still happens through the `finally`.
    """
    loop = asyncio.get_running_loop()
    try:
        yield InferenceChunk.ready(
            prompt_usage=PromptUsage(prompt_tokens=prompt_tokens, prompt_cache_hit_tokens=0),
            system_fingerprint="deepinfer",
        )
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise RequestError("request exceeded the configured timeout", 504)
            try:
                event = await asyncio.wait_for(handle.queue.get(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise RequestError("request exceeded the configured timeout", 504) from exc
            if event[0] == "error":
                raise EngineError(event[1])
            _, tokens, finished, reason = event
            if tokens:
                stats.record_tokens(len(tokens))
                yield InferenceChunk.tokens(tokens)
            if finished:
                if reason and reason.startswith("error:"):
                    raise EngineError(reason)
                if reason == "abort":
                    raise EngineError("engine aborted the request (prompt too long for the engine limits?)")
                stats.finish_reason = reason
                stats.completed = True
                yield InferenceChunk.finish(_FINISH.get(reason, InferenceFinishReason.Stop))
                return
    finally:
        client.release(handle)


async def response_body(
    prepared: PreparedRequest,
    chunks: AsyncIterator[InferenceChunk],
    tokenizer: Tokenizer,
    stats: RequestStats,
) -> AsyncIterator[str]:
    _, response_type = PROTOCOL_TYPES[prepared.protocol]
    request_type, _ = PROTOCOL_TYPES[prepared.protocol]
    response_id = str(uuid4())
    model = prepared.model
    generator = request_type.chunk_generator(prepared.conversation_request, response_id, model)
    # The tokenizer lets recipe decode `tokens` chunks and count reasoning tokens.
    processor = StreamProcessor(generator, prepared.conversation_request.parsing_options, tokenizer)
    response = None if prepared.stream else response_type(response_id, model, int(time()), 0, 0)
    inference = None
    try:
        inference = aiter(chunks)
        while True:
            try:
                chunk = await anext(inference)
            except StopAsyncIteration:
                out = processor.finish()
            else:
                out = processor.push(chunk)
            for item in out:
                if prepared.stream:
                    yield sse_frame(response_type.sse_event_name(item), item.to_json())
                else:
                    assert response is not None  # built above whenever prepared.stream is False
                    response.accumulate(item)
            if processor.finished:
                break
        if prepared.stream:
            done = response_type.sse_done_data()
            if done is not None:
                yield sse_frame(None, done)
        else:
            assert response is not None
            yield response.to_json()
    except (EngineError, RequestError) as exc:
        # A stream may already have started, so the status line cannot change. Emit a final
        # error frame so an SSE client sees the reason instead of a silent truncation.
        status = exc.status_code if isinstance(exc, RequestError) else 500
        metrics.ERRORS.inc(kind="timeout" if status == 504 else "engine")
        if not prepared.stream:
            raise
        yield sse_frame(None, json.dumps(_error_payload(str(exc), status)))
        done = response_type.sse_done_data()
        if done is not None:
            yield sse_frame(None, done)
    finally:
        try:
            close = getattr(inference, "aclose", None)
            if close is not None:
                with CancelScope(shield=True):
                    await close()
        finally:
            processor.close()


def create_app(cfg: BridgeConfig) -> FastAPI:
    cfg.validate()
    tokenizer_path = resolve_tokenizer_path(cfg)
    cfg.tokenizer_path = tokenizer_path  # workers (mock) read it too
    tokenizer = Tokenizer.from_file(tokenizer_path)
    encoding = make_encoding(tokenizer)
    options = conversion_options(cfg.served_model_name)

    if cfg.mock:
        from .engine_mock import mock_worker as target
    else:
        from .engine_real import engine_worker as target
    client = EngineClient(cfg, target)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await client.start()
        try:
            yield
        finally:
            await client.stop(cfg.shutdown_grace_s)

    app = FastAPI(title="deepinfer-bridge", version=__version__, lifespan=lifespan)
    app.state.client = client
    app.state.config = cfg

    @app.middleware("http")
    async def auth(request: Request, call_next):
        # /healthz stays open so a supervisor can liveness-check an authenticated server.
        if cfg.api_key and request.url.path != "/healthz":
            provided = (request.headers.get("authorization") or "").encode()
            expected = f"Bearer {cfg.api_key}".encode()
            # compare_digest on bytes avoids both a timing side channel and a TypeError on
            # non-ASCII header values.
            if not hmac.compare_digest(provided, expected):
                return JSONResponse(
                    status_code=401,
                    content=_error_payload("invalid API key", 401, code="invalid_api_key"),
                )
        return await call_next(request)

    @app.exception_handler(ConversionError)
    async def conversion_error_handler(_r: Request, exc: ConversionError) -> Response:
        return Response(content=exc.body, status_code=exc.status_code, media_type="application/json")

    @app.exception_handler(RequestError)
    async def request_error_handler(_r: Request, exc: RequestError) -> Response:
        headers = {"Retry-After": str(RETRY_AFTER_S)} if exc.status_code in (429, 503) else None
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(str(exc), exc.status_code),
            headers=headers,
        )

    @app.get("/healthz")
    async def healthz() -> Response:
        ok = client.healthy
        return JSONResponse(
            status_code=200 if ok else 503, content={"status": "ok" if ok else "unavailable", **client.stats()}
        )

    @app.get("/stats")
    async def stats() -> dict:
        return client.stats()

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        snapshot = client.stats()
        metrics.ENGINE_HEALTHY.set(1 if snapshot["healthy"] else 0)
        for rank, load in enumerate(snapshot["load_per_rank"]):
            metrics.RANK_LOAD.set(load, rank=str(rank))
        return Response(metrics.REGISTRY.render(), media_type=metrics.CONTENT_TYPE)

    @app.get("/v1/models")
    @app.get("/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [{"id": cfg.served_model_name, "object": "model", "created": 0, "owned_by": "deepinfer"}],
        }

    def api_handler(protocol: str):
        async def handler(request: Request) -> Response:
            request_id = str(uuid4())
            deadline = asyncio.get_running_loop().time() + cfg.request_timeout_s

            declared = request.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > cfg.max_body_bytes:
                metrics.REJECTED.inc(reason="body_too_large")
                raise RequestError(f"request body exceeds {cfg.max_body_bytes} bytes", 413)
            body = await request.body()
            if len(body) > cfg.max_body_bytes:
                metrics.REJECTED.inc(reason="body_too_large")
                raise RequestError(f"request body exceeds {cfg.max_body_bytes} bytes", 413)

            prepared = await run_in_threadpool(
                prepare_request, protocol, body, options=options, encoding=encoding, context_limit=cfg.max_model_len
            )
            sampling = build_sampling(prepared, tokenizer, constrained=cfg.constrained_decoding)
            stats = RequestStats(
                request_id=request_id,
                endpoint=protocol,
                stream=prepared.stream,
                model=prepared.model,
                prompt_tokens=len(prepared.token_ids),
            )

            try:
                handle = client.submit(prepared.token_ids, sampling)
            except EngineBusy as exc:
                metrics.REJECTED.inc(reason="overload")
                raise RequestError(str(exc), 503) from exc
            except EngineUnavailable as exc:
                metrics.REJECTED.inc(reason="unavailable")
                raise RequestError(str(exc), 503) from exc

            metrics.IN_FLIGHT.inc()
            chunks = token_chunks(client, handle, len(prepared.token_ids), stats, deadline)
            output = response_body(prepared, chunks, tokenizer, stats)

            if prepared.stream:

                def on_close() -> None:
                    client.release(handle)
                    _finalize(stats, "ok" if stats.completed else "incomplete")

                return InferenceStreamingResponse(output, media_type="text/event-stream", on_close=on_close)

            try:
                content = "".join([part async for part in output])
            except EngineError as exc:
                metrics.ERRORS.inc(kind="engine")
                raise RequestError(str(exc), 500) from exc
            finally:
                client.release(handle)
                _finalize(stats, "ok" if stats.completed else "incomplete")
            return Response(content, media_type="application/json")

        return handler

    for path, protocol in (
        ("/v1/chat/completions", "chat_completions"),
        ("/v1/responses", "responses"),
        ("/v1/messages", "messages"),
    ):
        app.add_api_route(path, api_handler(protocol), methods=["POST"], name=protocol)
    return app
