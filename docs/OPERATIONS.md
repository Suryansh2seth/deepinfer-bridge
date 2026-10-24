# Operations

Running, monitoring, restarting and upgrading the bridge. Written for whoever is on call.

---

## Health and readiness

| Endpoint | Auth | Meaning |
|---|---|---|
| `GET /healthz` | **exempt** | 200 when every rank is ready, 503 otherwise. Body carries the full stats object. |
| `GET /stats` | required | counters and per-rank load |
| `GET /metrics` | required | Prometheus text exposition |
| `GET /v1/models`, `GET /models` | required | model discovery for Harness |

`/healthz` is the only unauthenticated route, so a supervisor can probe a secured server
without holding the API key. Everything else needs
`Authorization: Bearer <api-key>` whenever `--api-key` is set — including `/metrics`, so
configure your Prometheus scrape job with the credential.

## Key metrics

| Metric | Type | Watch for |
|---|---|---|
| `deepinfer_bridge_engine_healthy` | gauge | 0 means at least one rank is dead; restart |
| `deepinfer_bridge_in_flight` | gauge | approaching `--max-outstanding` means clients will get 503 |
| `deepinfer_bridge_rejected_total{reason}` | counter | `overload` vs `unavailable` — different fixes |
| `deepinfer_bridge_ttft_seconds` | histogram | p95, not the mean; long-context prompts dominate |
| `deepinfer_bridge_request_duration_seconds` | histogram | total wall time per request |
| `deepinfer_bridge_errors_total{kind}` | counter | `engine` vs `timeout` |
| `deepinfer_bridge_aborts_total` | counter | rising with `in_flight` flat = clients giving up early |
| `deepinfer_bridge_rank_load{rank}` | gauge | an imbalanced series means a rank is stalling |
| `deepinfer_bridge_tokens_total{kind}` | counter | prompt vs completion, for cost accounting |

**`in_flight` is deliberately not clamped at zero.** A negative value means `_finalize` ran
more times than a request was admitted — a bug, made visible on purpose.

### Suggested alerts

- `deepinfer_bridge_engine_healthy == 0` for 1 minute → page.
- `rate(deepinfer_bridge_rejected_total{reason="overload"}[5m]) > 0` for 10 minutes → scale
  or raise `--max-outstanding` (after checking engine capacity).
- `histogram_quantile(0.95, rate(deepinfer_bridge_ttft_seconds_bucket[5m])) > <your SLO>`.
- `deepinfer_bridge_in_flight < 0` → page; accounting bug.

## Logs

Default format is `key=value`; `--log-format json` gives one JSON object per line.

```json
{"ts":"2026-10-25T01:02:45+0530","level":"INFO","logger":"deepinfer_bridge",
 "message":"request complete","request_id":"...","endpoint":"chat_completions",
 "model":"deepseek-v4.1-flash","stream":false,"status":"ok","finish_reason":"stop",
 "prompt_tokens":18,"completion_tokens":42,"duration_ms":812.4,"ttft_ms":96.2}
```

`status` is `ok` when the engine reported a finish reason, `incomplete` when the stream ended
without one (client disconnect, abort, timeout). **Prompts and completions are never logged**
unless `--log-prompts` is passed; leave it off in production.

## Restart procedure

Ranks are lock-step, so they must restart **together**. Never restart a single rank.

```bash
# 1. stop admitting work and drain
kill -TERM <pid>

# 2. confirm the drain completed and workers exited
#    (bridge logs "draining N in-flight request(s)"; workers are joined, then SIGTERM'd)
ps aux | grep -c '[p]ython -m deepinfer_bridge\|[d]eepinfer'

# 3. start again
python -m deepinfer_bridge --model-path ... --ep 8 --dp 8 --api-key "$BRIDGE_API_KEY"
```

`SIGTERM` drains for `--shutdown-grace` seconds (default 30) before aborting. Weight loading
means a restart is expensive: expect minutes before `/healthz` returns 200.

If a rank dies unexpectedly, `/healthz` goes 503, in-flight work fails, and new work is
refused with 503. There is no automatic recovery, by design — see ARCHITECTURE.md.

## Upgrade procedure

### Upgrading the bridge

```bash
git -C deepinfer-bridge pull
pip install -e ".[test]"
python -m pytest -q          # must be green before restarting the service
```

The test suite runs against `engine_mock.py` and needs the recipe tokenizer:

```bash
export DEEPSEEK_RECIPE_TOKENIZER=/path/to/deepseek-recipe/static/tokenizers/v41/tokenizer.json
```

Without it every test **skips** rather than fails — a green run that proves nothing. Check
for `13 skipped` vs `N passed`.

### Upgrading DeepInfer

`engine_real.py` mirrors DeepInfer's examples. When DeepInfer changes, re-check these three
places and update `_build_llm` / `_apply_env` / `_set_visible_devices`:

| Bridge | Mirror in DeepInfer |
|---|---|
| `engine_real._build_llm` | `examples/_config.py::create_llm` + `examples/offline_inference.py` |
| `engine_real._apply_env` | `examples/_config.py::setup_common_env` |
| `engine_real._set_visible_devices` | `examples/_config.py::setup_cuda_visible_devices` |
| `engine_real.resolve_model_dir` | `examples/_config.py::resolve_model_path` |

Then pin the new commit SHA in the README's version matrix and in
`docs/HARDWARE_BRINGUP.md`.

### Upgrading deepseek-recipe

The bridge pins `deepseek-recipe>=0.2,<0.3`. A minor bump can change request conversion,
`StreamProcessor` output or `so_args` semantics. Re-run the suite; the tool-call and
structured-output tests are the canaries. On a source build, remember OpenCV 4.x, Clang and
Rust (edition 2024, so Rust >= 1.85) are required.

## Capacity and overload

- `--max-outstanding` is the admission limit. Beyond it, requests get **503** with
  `Retry-After: 1` and `error.type == "overloaded_error"`.
- `--request-timeout` bounds one generation (default 600 s). On expiry the engine request is
  aborted; buffered calls get **504**, streams get a final `data: {"error": ...}` frame
  followed by `data: [DONE]`.
- `--max-body-bytes` (default 8 MiB) rejects oversized bodies with **413** before JSON parsing.

## Security notes

- `--api-key` is compared with `hmac.compare_digest` (constant time) on bytes.
- An empty `--api-key` is a startup **error** rather than silently meaning "no auth".
- Binding a non-loopback address with no key is refused unless `--insecure` is passed.
- This is a bearer check only. Put it behind a proxy for TLS, rate limiting and audit.

## WSL specifics

If `pip` stalls inside WSL, the cause is usually IPv6: `pypi.org` resolves to IPv6 first and
the address is unreachable, so each attempt waits for a timeout. Verify with:

```bash
timeout 10 curl -4 -sS -o /dev/null -w '%{http_code}\n' https://pypi.org/simple/   # 200
timeout 10 curl -6 -sS -o /dev/null -w '%{http_code}\n' https://pypi.org/simple/   # rc=7
```

Workarounds: `curl -4`, `pip --timeout 60 --retries 5`, or prefer IPv4 in `/etc/gai.conf`.

Building `deepseek-recipe` from source in WSL additionally needs `rustc >= 1.85` (the crates
use edition 2024), OpenCV 4.x dev headers, and libclang:

```bash
sudo apt install -y build-essential pkg-config clang libclang-dev libopencv-dev
pip install --no-binary deepseek-recipe deepseek-recipe==0.2.0
```

If `rustup` reports "no default toolchain is configured" but a toolchain directory exists,
it is likely a partial install whose binaries are 0 bytes. Remove and reinstall:

```bash
rm -rf ~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu
rustup toolchain install stable --profile minimal
rustup default stable
```
