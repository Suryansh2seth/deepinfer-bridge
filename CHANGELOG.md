# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-25

First release with the hardening pass and the full test suite. The HTTP/protocol layer is
tested; the DeepInfer engine integration is not (see **Not done** below).

### Added

- **Configuration validation** (`config.py::BridgeConfig.validate`). Rejects `--cp` not
  dividing `--ep`, a `--dp`/`--cp` world size that is not a multiple of `--ep` (the assert
  DeepInfer makes at `examples/offline_inference.py:29`), out-of-range limits, a non-empty
  requirement for `--api-key`, a non-loopback bind without a key, and `--mock` with more than
  one rank. Errors are raised at startup instead of failing minutes into weight loading.
- **Environment overrides for every field**: `DEEPINFER_BRIDGE_<FIELD_NAME>`. Precedence is
  CLI flag > environment > default. `BRIDGE_API_KEY` still works as a legacy alias.
- **New CLI flags**: `--request-timeout`, `--shutdown-grace`, `--max-body-bytes`,
  `--startup-timeout`, `--sampler-force-topk`, `--insecure`, `--log-format`, `--log-prompts`.
- **Per-request timeout** (default 600 s). Expiry aborts the engine request; buffered calls
  return 504, streams emit a final error frame.
- **Request body limit** (default 8 MiB) returning 413 before JSON parsing.
- **`Retry-After: 1`** on 429/503 responses, and OpenAI-style `error.type` values
  (`overloaded_error`, `timeout_error`, `authentication_error`, ...).
- **`GET /metrics`** in Prometheus text format (requests, in-flight, per-rank load, tokens,
  errors, aborts, rejected, TTFT and duration histograms). Hand-rolled in `metrics.py` —
  no new dependency.
- **Structured request logging**: one line per request with request id, endpoint, tokens,
  TTFT, duration, status and finish reason. `--log-format json` for machine consumption.
  Prompt and completion text is never logged unless `--log-prompts` is set.
- **Graceful shutdown**: `SIGTERM` drains in-flight requests for `--shutdown-grace` seconds
  (default 30), then aborts and joins every rank.
- **`docs/`**: `ARCHITECTURE.md` (with a sequence diagram including the abort path),
  `HARDWARE_BRINGUP.md` (Phases 0-4 runbook), `OPERATIONS.md` (monitoring, restart, upgrade).
- **Tests**: `test_config.py`, `test_metrics.py`, `test_hardening.py`,
  `test_engine_client.py` — 87 tests total, up from 13.
- **Lint and type configuration**: ruff and pyright in `pyproject.toml`, both clean.

### Fixed

- **`DEEPINFER_HF_MODEL_DIR` was ignored.** `engine_real.resolve_model_dir` now honours it
  for the default model id, matching `examples/_config.py::resolve_model_path`. Previously a
  local checkpoint could not be selected the way DeepInfer's own examples select it.
- **API-key comparison is now constant time** (`hmac.compare_digest` on bytes), and a
  non-ASCII header value can no longer raise `TypeError`.
- **`Gauge.dec` did not exist**, so every request that reached `_finalize` raised
  `AttributeError`. Caught by the tests added in this release.
- **`Histogram.render` iterated a dict without `.items()`**, raising `ValueError` whenever a
  histogram with observations was scraped.
- Trailing whitespace and an ambiguous `l` variable in the original test suite.

### Changed

- The HTTP process is asserted (by test) never to import `torch` or `deepinfer`.
- `engine_mock` raises a clear error when `tokenizer_path` is missing.

### Not done

- Phases 1-4 and 6 (single-rank bring-up, correctness features, multi-rank lock-step, real
  Harness integration, performance/cost) require SM100 or Ascend 950 hardware and have **not
  been executed**. See `REPORT.md` and `docs/HARDWARE_BRINGUP.md`.

## [0.1.0]

Initial release. Protocol layer only, verified against `engine_mock.py`:

- OpenAI-compatible endpoints: `POST /v1/chat/completions`, `/v1/responses`, `/v1/messages`;
  `GET /v1/models`, `/models`, `/healthz`, `/stats`.
- `deepseek-recipe` for request conversion, V4.1 prompt encoding, streaming decode, thinking
  and tool-call parsing, stop sequences and usage.
- One engine process per data-parallel rank over multiprocessing queues, least-outstanding
  routing, health tracking, abort-on-release, admission control.
- `engine_real.py` written against DeepInfer's incremental engine API; **never executed**.
- 13 tests, all passing against the mock engine.

[0.2.0]: https://github.com/Suryansh2seth/deepinfer-bridge/releases/tag/v0.2.0
[0.1.0]: https://github.com/Suryansh2seth/deepinfer-bridge/releases/tag/v0.1.0
