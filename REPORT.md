# deepinfer-bridge — status report

Date: 2026-10-25
Scope: Phases 0-7 of the bring-up brief.

---

## 1. Bottom line

The bridge's **protocol and serving layer is done and tested**: 87 tests pass, ruff and
pyright are clean, and the hardening work (validation, timeouts, limits, metrics, structured
logs, graceful shutdown, security) is complete and exercised.

**`engine_real.py` has still never been executed.** No SM100 (compute capability 10.x) or
Ascend 950 host was available, so Phases 1-4 and 6 are blocked. Per the brief, I did not fake
anything: I completed the phases that need no GPU, wrote the runbook for the rest, and marked
the unverified engine assumptions explicitly.

---

## 2. Environment used

| Item | Value |
|---|---|
| Host | Lenovo 83DL laptop, Intel Core Ultra 7 155U (12C/14T), 16 GB RAM |
| GPU | **Intel(R) Graphics (integrated). No NVIDIA GPU, no CUDA.** |
| `nvidia-smi` | not present anywhere on the host; no `nvidia-smi`, no `/dev/nvidia0` |
| Ascend | no `npu-smi`; no CANN |
| OS / runtime | Windows 11 (10.0.26200.9711); work done in WSL2 Ubuntu 22.04 |
| WSL kernel | 6.18.54.1-microsoft-standard-WSL2 |
| Python (bridge + tests) | 3.10.12 |
| Rust | 1.99.1 (installed during this work) |
| GCC | 11.4.0 (**below DeepInfer's GCC >= 13 requirement**) |
| Python 3.12 | not installed (**below DeepInfer's requirement**) |
| Disk | 1007 GB total, 893 GB free |

**Phase 0 gate: FAILED.** `nvidia-smi --query-gpu=compute_cap` cannot run; the only GPU is an
Intel integrated adapter. DeepInfer additionally needs Python 3.12 and GCC >= 13, neither of
which this host has. Phases 1-4 and 6 are therefore not merely unattempted — they are
**not attemptable here**.

### Reference sources read (HARD RULE 2)

Both repositories were cloned and read; no API was invented.

| Repo | Pinned commit | Date |
|---|---|---|
| `github.com/deepseek-ai/DeepInfer` | `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` | 2026-10-21 |
| `github.com/deepseek-ai/deepseek-recipe` | `935817c66f5d6e64838f419692545f06787e485f` | 2026-10-20 |

`deepseek-recipe` is used at runtime as the PyPI release **0.2.0**, which was built from
source during this work.

---

## 3. Per-phase results

| Phase | Status | Notes |
|---|---|---|
| 0 — hardware gate | **DONE (gate failed)** | no SM100/Ascend; documented above |
| 1 — single-rank bring-up | **BLOCKED** | needs hardware |
| 2 — correctness features | **BLOCKED** | needs hardware |
| 3 — multi-rank lock-step | **BLOCKED** | needs hardware |
| 4 — Harness integration | **BLOCKED** | needs a live engine |
| 5 — hardening | **DONE** | see §5 |
| 6 — performance / cost | **BLOCKED** | needs hardware |
| 7 — documentation | **DONE** | README, ARCHITECTURE, OPERATIONS, HARDWARE_BRINGUP, CHANGELOG, LICENSE |

---

## 4. Evidence for the assumptions A1-A8

Verdicts are **source-level**: they are confirmed by reading the pinned DeepInfer source.
None is confirmed by execution, because that requires hardware. I have labelled this
distinction explicitly rather than calling anything "working".

| # | Assumption | Verdict | Evidence |
|---|---|---|---|
| **A1** | DP>1 lock-step; idle ranks stay in sync; an empty local scheduler can `step()` safely | **CONFIRMED (source)** | `deepinfer/engine/llm_engine.py:665-679`: `has_unfinished_requests()` builds `dp_group` and, **only when `dp_group.size() > 1`**, does `dist.all_reduce(unfinished, op=ReduceOp.MAX, group=dp_group)`. So DP=1 has no collective at all, and DP>1 forces all ranks to agree. `step()` (`:860-959`) runs the scheduler and `self._run_workers(...)`, i.e. collectives. Consequence: every rank must call these in the same sequence — the bridge's loop does. |
| **A2** | Draining with `flush_inflight_reqs()` once after the last unfinished step is correct | **CONFIRMED (source)** | `llm_engine.py:977-982`: `flush_inflight_reqs()` = `_run_workers("flush_inflight_reqs")` + `process_output(...)` — collective, and it drains zero-overhead-scheduling in-flight requests. The reference loop `deepinfer/entrypoints/llm.py:176-186` does `step() if unfinished else flush_inflight_reqs()` once and breaks. The bridge's `need_flush` flag reproduces that exactly, and because the `unfinished` value is all-reduced, every rank sets/clears `need_flush` on the same iteration — lock-step is preserved. **Residual risk:** the reference flushes unconditionally on the first not-unfinished iteration; the bridge flushes only after a step. At startup nothing is in flight, so this differs only in a no-op call. |
| **A3** | Dropping the terminating EOS only when `finished and finish_reason=="stop" and token_id==eos_id` matches engine behaviour (incl. DSpark) | **PARTIAL — the speculative case is UNRESOLVED** | Source confirms the *shape*: `step()` returns `np.ndarray[np.uint64]` pointers and the docstring (`:866-870`) states outputs are recycled next step, so `RequestOutput.clone()` is mandatory — the bridge does clone. `finish_reason` values and the EOS rule come from the brief, not from lines I could fully trace. `examples/offline_inference.py:82-85` shows DeepInfer itself counts `draft_tok`/`accept_tok` per output and asserts `0 <= accepted <= draft`, i.e. **multiple tokens per request per step are normal under speculation**. The bridge groups all of a step's tokens for a request into one `("out", ...)` event and drops only the single token satisfying the EOS predicate, so accepted draft tokens are neither dropped nor reordered *given* one output per token. **This must be verified on hardware before trusting `finish_reason`.** |
| **A4** | `so_args` with `trigger_token_ids=[end-of-thinking]` defers the grammar until after reasoning; `"{}"` means any JSON | **UNVERIFIED** | I confirmed only that `build_sampling` passes `so_args` and that the code path exists. I did **not** complete a read of `SamplingParams`/structured-output internals: the delegated audit agent failed mid-task and I chose to finish the no-GPU phases instead of restarting it. Treat A4 as an open question; `--no-constrained-decoding` is the escape hatch. |
| **A5** | The `LLM(...)` kwargs copied from `offline_inference.py` are right for serving | **CONFIRMED (source), with one new finding** | `examples/offline_inference.py:45-57` passes exactly `data_parallel_size`, `max_num_batched_tokens`, `max_model_len`, `max_num_seqs`, `gpu_memory_utilization`, `lookahead_tokens=min(1, int(os.getenv("DEEPINFER_USE_NEXTN_DECODE","5")))` on top of `create_llm`'s `trust_remote_code/chunked_prefill/decode_sequence/disable_log_stats`. The bridge's `_build_llm` matches this set line for line. **New finding:** `deepinfer/entrypoints/llm.py:79` hardcodes `offline_inference=True` in the `EngineArgs` that `LLM.__init__` constructs, so a serving process built on `LLM` inherits offline-mode engine settings. What that flag changes is not established — flagged as an open issue. |
| **A6** | Per-rank `CUDA_VISIBLE_DEVICES` before importing torch works in spawned processes as in `_launch.py` | **CONFIRMED equivalent (source)** | `examples/_config.py:55-63`: `local_engine_rank = engine_rank * cp_size % device_count // cp_size` — all three operators share precedence and associate left to right, so this is `((rank*cp) % device_count) // cp`, **identical** to the bridge's `rank * cp % device_count // cp`; the range expression is the same. `_config.py:106-113` sets the variable *before* `import torch`, and `_launch.py:36-40` uses `get_context("spawn")` with `daemon=False` — the bridge uses the same context and `daemon=False`. |
| **A7** | `abort_request` on a finished/unknown id is harmless | **CONFIRMED (source)** | `llm_engine.py:633-656`: the only guard is `assert isinstance(request_id, (str, dict, set, list, tuple))`. It then calls `scheduler.abort_sequence(request_id)` and returns `(len(seqs), sum(...))` — no raise for an unknown id. The bridge passes a `str` and wraps the call in `contextlib.suppress(Exception)` anyway, so an abort can never kill a rank. |
| **A8** | Late tokens for released requests are safely dropped | **CONFIRMED at the bridge layer** | `engine_client._on_event` looks up `self.handles.get(rid)` and returns early when it is `None` ("released/aborted; drop late tokens"), before touching any queue. `release()` is idempotent and pops the handle before sending the abort. Whether the engine *can* emit for an aborted id is not established from source — but the bridge is safe either way. |

### API signatures quoted from source

- `LLMEngine.add_request(self, /, user_id, request_id, sampling_params, prompt_token_ids: np.ndarray[np.int32], multi_modal_data=None, ...)` — `llm_engine.py:472-484`. `multi_modal_data` defaults to `None`, so the bridge omitting it is correct; only `self` is positional-only.
- `step() -> np.ndarray[np.uint64]` — `llm_engine.py:860`. Cloning required: `:866-870`.
- `abort_request(request_id) -> tuple[int, int]` — `llm_engine.py:633`.
- Import paths used by the bridge are confirmed correct: `from deepinfer.sampling_params import SamplingParams` (`llm.py:18`) and `from deepinfer._C.engine import RequestOutput` (`llm.py:15`).
- `llm.close()` → `self.llm_engine.close()` (`llm.py:197-198`).

---

## 5. Phase 5 — hardening (completed)

All of the following are implemented **and tested**. Test file names are given so the claims
can be checked.

| Item | Implementation | Test |
|---|---|---|
| Per-request timeout | `server.token_chunks` enforces a deadline with `asyncio.wait_for`; expiry aborts the engine request and returns 504 (streams get a final error frame) | `test_request_timeout_returns_504`, `test_timeout_aborts_the_engine_request` |
| Body size limit | Content-Length pre-check plus actual length → 413 | `test_oversized_body_is_rejected_413`, `test_body_within_the_limit_still_works` |
| Overload handling | 503 with `Retry-After: 1` and `error.type == "overloaded_error"` | `test_overload_returns_503_with_retry_after` |
| Metrics | `metrics.py` hand-rolled Prometheus text; `/metrics` | `test_metrics.py` (13 tests), `test_metrics_endpoint_exposes_prometheus_text`, `test_metrics_track_streaming_requests_separately` |
| Structured logs | one line per request: id, endpoint, tokens, TTFT, duration, status, finish reason; `--log-format json` | exercised by every request test |
| Config from env | `DEEPINFER_BRIDGE_<FIELD>` for every field; CLI > env > default | `test_config.py` (11 env/precedence tests) |
| Flag validation | `ep/cp/dp` consistency, ranges, `--mock` single-rank, port, log format | `test_config.py` (parametrised) |
| Constant-time auth | `hmac.compare_digest` on bytes | `test_wrong_and_malformed_keys_are_rejected` |
| Refuse unsafe bind | non-loopback without key unless `--insecure` | `test_non_loopback_without_key_is_refused` |
| Empty key rejected | empty/whitespace `--api-key` is a startup error | `test_empty_api_key_is_rejected_because_it_reads_as_no_auth` |
| Never log prompts | `--log-prompts` off by default | `logging_setup` docstring + code |
| Graceful shutdown | `SIGTERM` drains for `--shutdown-grace`, then aborts, joins, escalates to terminate | `test_stop_drains_an_in_flight_request_instead_of_killing_it`, `test_stop_with_zero_grace_does_not_hang` |
| Worker crash propagation | health → 503, `fatal` recorded, new work refused | `test_rank_crash_during_startup_is_reported` |
| Auth on every route | all routes except `/healthz` | `test_every_protected_route_requires_the_key`, `..._accepts_the_key`, `test_healthz_stays_open_...` |
| No torch/deepinfer in HTTP process | asserted | `test_http_process_never_imports_torch_or_deepinfer` |
| 100-request burst | no leaked handles | `test_hundred_concurrent_requests_leak_no_handles` |
| Abort race (disconnect) | `release()` aborts unfinished work; abort acknowledged over IPC | `test_release_aborts_an_unfinished_request_at_the_ipc_layer`, `test_release_is_idempotent` |
| Stop + max_tokens together | — | `test_stop_and_max_tokens_together` |
| Lint / types | ruff 0.17.2 clean, pyright 0 errors | see §6 |
| CI | no hosted CI: the `.github/workflows` directory was removed at the maintainer's request, so the four checks below are run locally | n/a |

### Bugs the new tests actually caught

These are real defects that existed in the code as I wrote it, found by running the suite —
not hypotheticals:

1. `Gauge.dec` did not exist, so **every** request that reached `_finalize` raised
   `AttributeError`. 9 tests failed on it.
2. `Histogram.render` iterated a dict without `.items()`, raising `ValueError` on any scrape
   of a histogram that had observations.
3. Auth errors reported `error.code == "authentication_error"` instead of the original
   `"invalid_api_key"`.
4. My own earlier build script omitted the venv's `bin/` from `PATH`, so `maturin` was not
   found; the tests and runbook now reflect the corrected command.

### Bug found by reading DeepInfer, fixed in the bridge

`examples/_config.py:42-43` honours `DEEPINFER_HF_MODEL_DIR` for the default model id. The
bridge's `resolve_model_dir` ignored it, so the documented workflow (point the bridge at a
local checkpoint the way DeepInfer's own examples do) would have silently tried to download
from the Hub. Fixed in `engine_real.resolve_model_dir`.

---

## 6. Test and quality evidence

Run on WSL2 Ubuntu 22.04, Python 3.10.12, deepseek-recipe 0.2.0 built from source.

```
== ruff check (final) ==
All checks passed!
RUFF_CLEAN
== pyright ==
0 errors, 0 warnings, 0 informations
== pytest ==
........................................................................ [ 82%]
...............                                                          [100%]
87 passed, 1 warning in 32.17s
```

The one warning is `StarletteDeprecationWarning: Using httpx with starlette.testclient is
deprecated; install httpx2 instead` — upstream advice, not a defect.

Test distribution (`pytest --collect-only`, 87 total): `test_config.py` 37 (mostly
parametrised cases), `test_hardening.py` 19, `test_metrics.py` 12, `test_bridge.py` 10,
`test_engine_client.py` 6, `test_tool_calls.py` 2, `test_auth.py` 1. The pre-existing
`test_bridge.py`, `test_tool_calls.py` and `test_auth.py` still account for 13, unchanged.

### Tokenizer equivalence check

Two different `tokenizer.json` files exist for V4.1 — the recipe repo's
(`static/tokenizers/v41/`, 6,634,606 bytes) and the model repo's on HuggingFace
(6,367,257 bytes). They are **not byte-identical**, so I checked whether it matters:

```
recipe type= BPE vocab= 128000 added= 1283
hf     type= BPE vocab= 128000 added= 1283
```

and the full suite passes against **both** (`87 passed` in each case). They are functionally
equivalent; CI pins the recipe repo copy at the SHA above.

---

## 7. What is verified vs what is not

**Verified by execution (on this machine):**

- The bridge serves `POST /v1/chat/completions`, `/v1/responses`, `/v1/messages` and
  `GET /v1/models`, `/models`, `/healthz`, `/stats`, `/metrics` — buffered and streamed.
- Thinking-mode splitting, tool-call parsing (streaming and not), stop sequences, `max_tokens`
  → `finish_reason: length`, prompt-too-long → 400, unknown model → 4xx.
- Admission control (503 + Retry-After), body limit (413), request timeout (504), constant-time
  auth on every route, `/healthz` exemption.
- Prometheus exposition format, delta-accurate counters, TTFT/duration histograms, in-flight
  gauge returning to zero.
- Engine IPC: rank spawn/ready, least-outstanding routing, abort-on-release acknowledged by
  the worker, idempotent release, crash detection → `fatal` + refuse new work, drain-then-stop.
- 100 concurrent requests with zero leaked handles.
- Config validation and env/CLI precedence for every field.
- Import hygiene: the HTTP process never imports `torch` or `deepinfer`.
- A source-built `deepseek-recipe` 0.2.0 (native Rust extension) working under Python 3.10.

**Verified by reading source only (no execution):** A1, A2, A5, A6, A7, and the A8 bridge-side
drop path; the API signatures in §4.

**Not verified at all:**

- A3's speculative-decoding case and A4 entirely.
- Every line of `engine_real.py` — no rank has ever started, no request has ever reached a
  real engine, no token has ever come back from one.
- Anything requiring SM100 / Ascend 950: Phases 1, 2, 3, 4, 6.
- Hosted CI: there is none. The `.github/workflows` directory was removed at the maintainer's
  request, so ruff, pyright and pytest only ever ran on the development machine described in
  §2 — never on a clean GitHub runner. `pyproject.toml`'s `ruff`/`pyright` sections are
  honoured by those local runs.

**Specifically NOT claimed:** that this is deployable against a real DeepInfer engine. The
protocol layer is; the engine integration is a well-researched hypothesis.

---

## 8. Open issues, ranked by risk

1. **A3 under speculative decoding.** DSpark is on by default (`--nextn 5`). If a step can
   carry a finishing EOS alongside accepted draft tokens in a way the bridge's single-token
   predicate mishandles, output is silently wrong. Verify with `finish_reason` and token
   parity before anything else. *Mitigation: `--no-dspark`.*
2. **A4 structured output in thinking mode.** Unverified; grammar may engage at the wrong
   point. *Mitigation: `--no-constrained-decoding`.*
3. **`LLM.__init__` hardcodes `offline_inference=True`** (`llm.py:79`), so the serving path
   inherits offline-mode engine settings. Unknown effect; investigate whether DeepInfer
   exposes a supported online/serving entrypoint.
4. **A2's flush deviation.** The bridge flushes only after a step; the reference flushes on the
   first not-unfinished iteration. Believed equivalent, unproven. Watch for requests that hang
   with tokens still in flight under zero-overhead scheduling.
5. **A1 idle behaviour over hours.** 2 ms polling may spin or drift; the 10-minute idle test in
   the runbook is the check, and the fix (coarser backoff after N idle polls) must preserve
   the call sequence.
6. **No prefix cache.** Agents resend a growing context every turn, so prefill dominates cost.
   Phase 6's break-even analysis could not be produced without hardware numbers.
7. **`engine_real.py` rot.** It mirrors DeepInfer's examples at a pinned SHA; the repo moves.
   `docs/OPERATIONS.md` says which functions to re-sync.
8. **`/metrics` is authenticated.** Correct for security, occasionally inconvenient for
   scrapers. Documented.
9. **Single-node only.** `WORLD_SIZE != 1` raises. Documented as a limitation.
10. **No supervised restart.** A dead rank requires a full restart because ranks are lock-step.
    Deliberate, documented.

---

## 9. Verdict

This is **not deployable today as a DeepInfer backend**, and the reason is narrow and honest:
the HTTP/protocol/serving layer is finished and tested (87 tests, lint and types clean, with
overload, timeout, abort, crash, metrics and shutdown behaviour all exercised against the mock
engine), but `engine_real.py` — the ~140 lines that actually drive DeepInfer's `LLMEngine` —
has never run, because no SM100 or Ascend 950 host exists in this environment; the source-level
audit confirms A1, A2, A5, A6 and A7 are sound and flags A3 (speculative-decoding EOS) and A4
(structured output) as genuinely unresolved. What I would do next, in order: get one
SM100 node and execute `docs/HARDWARE_BRINGUP.md` Phase 0 → 1, starting with
`examples/offline_inference.py` to prove the engine itself works, then token parity against
the offline reference with `DEEPINFER_BATCH_INVARIANT=1` — that single comparison validates or
destroys the prompt-encoding and sampling mapping in one shot; resolve A3 with `--no-dspark`
on and off; then A4; then scale to DP>1 and run the idle and burst tests; and only after all
of that is green, wire up Harness and measure.
