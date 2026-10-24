# deepinfer-bridge

OpenAI-compatible HTTP bridge so **DeepSeek Harness (dsh)** can use **DeepInfer** as its engine.

```
dsh ──HTTP──► deepinfer-bridge (FastAPI) ──► DeepInfer engine process(es), one per DP rank
              │  deepseek-recipe: request → V4.1 prompt token IDs;  token IDs → thinking /
              │  tool_calls / content, SSE chunks, usage
              └─ routes to least-loaded rank, aborts on disconnect / stop sequence
```

Endpoints: `POST /v1/chat/completions` (what Harness uses), `/v1/responses`, `/v1/messages`,
`GET /v1/models` (+ `/models`), `GET /healthz`, `GET /stats`, `GET /metrics`.

## Status — read this first

| Part | State |
|---|---|
| Protocol layer (recipe encode/decode, SSE, thinking, tool calls, stop sequences, auth, abort, admission control, timeouts, metrics, config validation) | **Tested — 87 tests pass** against a mock engine using the real recipe wheel and tokenizer |
| `engine_real.py` (DeepInfer process) | **Written against DeepInfer's source, NEVER RUN.** Needs SM100 or Ascend 950. Treat every line as a hypothesis. |

**Bring-up Phases 1-4 and 6 (single-rank bring-up, correctness features, multi-rank
lock-step, Harness integration, performance/cost) have not been executed** — no SM100 or
Ascend 950 host was available. `docs/HARDWARE_BRINGUP.md` is the runbook for them, and
`REPORT.md` lists exactly which claims are verified and which are not.

Verified against:

| Component | Version / commit |
|---|---|
| deepinfer-bridge | 0.2.0 |
| deepseek-recipe (runtime dependency) | `0.2.0` from PyPI |
| deepseek-recipe (source read: tokenizer, reference drain loop) | `935817c66f5d6e64838f419692545f06787e485f` |
| DeepInfer (source read only — never executed) | `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` |
| Model checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash-1017` (not downloaded) |

## Install

```bash
# DeepInfer (per its README): Python 3.12, GCC>=13, CUDA 13 (or CANN 9.2)
export DEEPINFER_ACCELERATOR=nvidia && ./build.sh        # in the DeepInfer checkout
pip install -e .                                          # this bridge (pulls deepseek-recipe, fastapi, uvicorn)
```

Building `deepseek-recipe` from source (no wheel for your platform) needs Rust >= 1.85
(the crates use edition 2024), OpenCV 4.x and libclang:

```bash
sudo apt install -y build-essential pkg-config clang libclang-dev libopencv-dev
pip install --no-binary deepseek-recipe deepseek-recipe==0.2.0
```

## Run

```bash
# 8x SM100 on one node (EP=8 => 8 DP ranks, one process each)
export DEEPINFER_HF_MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017
python -m deepinfer_bridge \
  --model-path "$DEEPINFER_HF_MODEL_DIR" \
  --deepinfer-root /path/to/DeepInfer \
  --ep 8 --max-model-len 131072 --api-key "$BRIDGE_API_KEY" \
  --host 127.0.0.1 --port 8000 --log-format json
```

Loading weights can take many minutes; `GET /healthz` returns 503 until every rank is ready.

Try it without GPUs (canned "Hello world!" through the real protocol stack):

```bash
python -m deepinfer_bridge --mock --tokenizer-path /path/to/deepseek-recipe/static/tokenizers/v41/tokenizer.json
```

Smoke test:

```bash
curl -s localhost:8000/healthz | jq
curl -s localhost:8000/v1/models | jq
curl -sN localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","stream":true,"thinking":{"type":"disabled"},
       "messages":[{"role":"user","content":"hi"}]}'
```

## DeepSeek Harness setup

`npx @deepseek-ai/dsh web` → Settings → Models → **Add a custom provider**:

| Field | Value |
|---|---|
| Provider ID | `deepinfer` (lowercase; permanent) |
| Base URL | `http://127.0.0.1:8000/v1` |
| API Protocol | `openai-completions` |
| Credential | your `--api-key` value (any non-empty string if you set none) |
| Model ID | `deepseek-v4.1-flash` (= `--served-model-name`) |

Then pick `deepinfer / deepseek-v4.1-flash` for a session. Harness lists models via
`GET /models`; a custom model must be in that list to be selectable.

> **If you set `--api-key`, every route except `/healthz` requires the bearer token** —
> including `/models` and `/metrics`. Configure the credential, or model discovery returns
> 401 and the model will not appear.

## Flags worth knowing

| Flag | Meaning |
|---|---|
| `--ep / --cp / --dp` | Same as DeepInfer's `offline_inference.py`. `dp` defaults to `ep // cp`. Validated at startup: `--cp` must divide `--ep`, and `dp * cp` must be a multiple of `--ep`. |
| `--max-model-len` | Context window. Default 32768 is small for agents; raise it as memory allows. |
| `--max-num-seqs` | Concurrent sequences **per DP rank**. |
| `--max-outstanding` | Bridge admission limit; beyond it requests get HTTP 503 + `Retry-After`. |
| `--request-timeout` | Wall-clock budget per generation (default 600 s) → 504. |
| `--shutdown-grace` | Seconds to drain in-flight requests on SIGTERM (default 30). |
| `--max-body-bytes` | Reject larger bodies with 413 (default 8 MiB). |
| `--insecure` | Permit binding a non-loopback address with no `--api-key`. |
| `--log-format` | `text` (key=value) or `json`. |
| `--log-prompts` | Include prompt/completion text in logs. Off by default. |
| `--no-dspark`, `--nextn N` | DSpark speculative decoding (on, 5 drafts). |
| `--no-swa-bounded-replay` | Disable decoder SWA bounded replay. |
| `--no-constrained-decoding` | Don't map `response_format` json/regex to DeepInfer structured output. |

**Every field is also settable as an environment variable** `DEEPINFER_BRIDGE_<FIELD_NAME>`,
e.g. `DEEPINFER_BRIDGE_REQUEST_TIMEOUT_S=120`. Precedence: CLI flag > environment > default.
`BRIDGE_API_KEY` remains supported as a legacy alias.

Env vars passed to the engine via `setdefault`, so yours win: `DEEPINFER_ENABLE_MTP_VARLEN`,
`DEEPINFER_USE_NEXTN_DECODE`, `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE`,
`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS`. `DEEPINFER_HF_MODEL_DIR` is honoured for the default
model id, matching DeepInfer's own examples.

## Observability

`GET /metrics` exposes Prometheus text: request counts by endpoint/stream/status, in-flight
gauge, per-rank load, prompt/completion tokens, errors, aborts, rejections, and TTFT and
total-duration histograms. `GET /stats` returns the raw counters as JSON. Every request emits
one structured log line with request id, tokens, TTFT, duration and finish reason.

`deepinfer_bridge_in_flight` is intentionally **not** clamped at zero: a negative value
reveals a double-finalise bug rather than hiding it.

See `docs/OPERATIONS.md` for alerts and procedures.

## Design notes and limitations

- **Incremental engine API.** DeepInfer's `LLM.generate()` is blocking; the bridge instead
  drives `LLMEngine.add_request / step / abort_request`, so tokens stream and disconnects
  abort work.
- **Lock-step.** With DP>1, `has_unfinished_requests()` is an `all_reduce(MAX)` and `step()`
  contains collectives, so *every* rank runs the loop continuously (2 ms sleep between idle
  polls). Ranks must start, and be restarted, together. If idle ranks hang or spin, this is
  the first place to look. See `docs/ARCHITECTURE.md`.
- **No KV cache system / PD disaggregation / cross-replica balancing** — DeepInfer's release
  omits them and so does this bridge. Agents resend growing contexts, so prefill cost
  dominates; expect to add prefix reuse yourself if that matters.
- **Single node only.** `WORLD_SIZE` must be 1.
- **Not supported:** image input (400), `n>1`, `logprobs`, server-side tools (web_search),
  recipe's `tool_call`/`tool_call_required` decode-time constraints (DeepInfer has no matching
  grammar; tool calls are parsed from model output, as with the official API).
- **Structured output in thinking mode** passes `trigger_token_ids=[</think>]` so the grammar
  engages after reasoning. This follows `SamplingParams`' docstring but is **unverified**;
  disable with `--no-constrained-decoding` if it misbehaves.
- `engine_real._build_llm` copies `examples/_config.py` and `examples/offline_inference.py`;
  re-sync it when DeepInfer changes.
- Security: `--api-key` is a constant-time bearer check only. Bind to localhost or put it
  behind a proxy for TLS and rate limiting.

## Tests

```bash
pip install -e ".[test]"

# The suite needs the V4.1 tokenizer. Without it every test SKIPS.
curl -fsSL -o /tmp/v41-tokenizer.json \
  https://raw.githubusercontent.com/deepseek-ai/deepseek-recipe/935817c66f5d6e64838f419692545f06787e485f/static/tokenizers/v41/tokenizer.json
export DEEPSEEK_RECIPE_TOKENIZER=/tmp/v41-tokenizer.json

pytest -q -rs         # expect 87 passed, 0 skipped
```

**Check for `skipped`.** `tests/conftest.py` skips rather than fails when the tokenizer is
missing, so a green run with everything skipped proves nothing. CI asserts this explicitly.

Lint and types:

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pyright
```

`engine_real.py` is excluded from pyright because it imports `deepinfer`/`torch`, which only
exist on supported GPU hosts.

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| All tests skipped | tokenizer missing | set `DEEPSEEK_RECIPE_TOKENIZER` |
| `/healthz` stays 503 | weights loading, or a rank died | read the logs; `--startup-timeout` defaults to 1 h |
| Startup hangs with DP>1 | one rank failed to join; others block in a collective | ranks must fail together — check every rank's log |
| Requests 503 immediately | `--max-outstanding` reached | raise it or scale; check `rejected_total{reason="overload"}` |
| Requests 504 | `--request-timeout` too low for the context length | raise it; watch the TTFT histogram |
| 401 from `/models` | `--api-key` set but Harness has no credential | configure the credential in the provider |
| Slow first request | graph capture on the first batch | warm up with a throwaway request |
| `world_size % ep_size` assert | bad `--ep/--cp/--dp` | the bridge now rejects this at startup |
| pip stalls in WSL | IPv6 resolves but is unreachable | `curl -4`, or prefer IPv4 in `/etc/gai.conf` |
| `rustup`: "no default toolchain" | partial Rust install (0-byte binaries) | `rm -rf ~/.rustup/toolchains/stable-*`, then reinstall |

Protocol code adapted from deepseek-recipe's `server-py` (MIT); engine setup mirrors
DeepInfer's examples (Apache-2.0). See `LICENSE`.

## Development and contributing

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pyright
pytest -q -rs         # needs DEEPSEEK_RECIPE_TOKENIZER, see Tests above
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development environment, the checks that
must pass, and what "verified" means in this project. Security issues should go through
[`SECURITY.md`](SECURITY.md), **not** the public issue tracker.

More documentation: [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
[`OPERATIONS.md`](docs/OPERATIONS.md) · [`HARDWARE_BRINGUP.md`](docs/HARDWARE_BRINGUP.md) ·
[`REPORT.md`](REPORT.md) · [`CHANGELOG.md`](CHANGELOG.md) ·
[`CONTRIBUTING.md`](CONTRIBUTING.md) · [`SECURITY.md`](SECURITY.md)
