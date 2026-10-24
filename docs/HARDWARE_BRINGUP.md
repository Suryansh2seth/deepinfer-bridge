# Hardware bring-up runbook (Phases 0-4)

This is the runbook for taking `deepinfer-bridge` from "protocol layer verified against a
mock" to "serving a real DeepSeek-V4.1-Flash checkpoint on real accelerators".

**Every command below is UNEXECUTED by the author of this repository.** It was written by
reading DeepInfer at commit `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` and
deepseek-recipe at `935817c66f5d6e64838f419692545f06787e485f`. No SM100 (compute capability
10.x) or Ascend 950 host was available. Treat this as a checklist to work through, not as
evidence that it works.

Anything in this document that *is* verified is marked **[verified]**, with the command that
proved it.

---

## Prerequisites

| Requirement | Why | How to check |
|---|---|---|
| NVIDIA SM100 (Blackwell), compute capability 10.x, or Huawei Ascend 950 | DeepInfer's kernels target these | `nvidia-smi --query-gpu=compute_cap --format=csv` |
| Python 3.12 | DeepInfer's pinned wheels | `python3.12 --version` |
| GCC >= 13 | DeepInfer C++ extensions | `gcc --version` |
| CUDA 13 (or CANN 9.2 for Ascend) | DeepInfer build | `nvcc --version` |
| DeepSeek-V4.1-Flash-1017 checkpoint locally | only supported model | see below |
| ~1 TB free disk | 48 safetensors shards + build tree | `df -h` |

The bridge itself only needs Python >= 3.10; **DeepInfer** is what needs 3.12 and GCC 13.
Run the bridge from the same interpreter that can import `deepinfer`.

> Reference host for this repo's mock testing (NOT valid for Phases 1-4): WSL2 Ubuntu
> 22.04, Python 3.10.12, GCC 11.4, Intel iGPU, 16 GB RAM. **[verified]**

---

## Phase 0 — environment and hardware gate

### 0.1 Confirm the accelerator

```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
```

For Ascend:

```bash
npu-smi info
```

**Gate:** `compute_cap` must show `10.x`. Anything else (e.g. `8.9`, `12.0`) is not a
supported target and Phases 1-4 cannot proceed. Stop here and report.

### 0.2 Confirm the toolchain

```bash
python3.12 --version     # expect 3.12.x
gcc --version            # expect >= 13
nvcc --version           # expect CUDA 13.x, or `cann` equivalent for Ascend
```

### 0.3 Get the checkpoint

```bash
export DEEPINFER_HF_MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017
ls "$DEEPINFER_HF_MODEL_DIR/config.json"     # must exist
ls "$DEEPINFER_HF_MODEL_DIR"/model-*.safetensors | wc -l
```

**The bridge honours `DEEPINFER_HF_MODEL_DIR`** exactly as `examples/_config.py` does: it is
consulted only when `--model-path` is left at its default
(`deepseek-ai/DeepSeek-V4.1-Flash-1017`). This was added during hardening — the released
version ignored it and would have tried to download from the Hub. **[verified: unit-tested
indirectly via config; the override itself is exercised by `engine_real.resolve_model_dir`,
which has not run]**

### 0.4 Build DeepInfer

```bash
git clone https://github.com/deepseek-ai/DeepInfer
cd DeepInfer
git checkout 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9
export DEEPINFER_ACCELERATOR=nvidia      # or: ascend
./build.sh
pip install -e .
```

### 0.5 Prove DeepInfer works BEFORE blaming the bridge

This step is not optional. If the engine cannot do offline inference, the bridge cannot
either.

```bash
cd DeepInfer
python examples/offline_inference.py --ep 8
```

Expect one line per DP rank of the form
`DP 0; INPUT: ...; OUTPUT: '...'; output_tokens=...; accept_rate=...`.

**Gate:** if this fails, stop. The bridge is not the problem. Report the error.

### 0.6 Install the bridge

```bash
cd deepinfer-bridge
pip install -e ".[test]"
```

---

## Phase 1 — single-rank bring-up (`--dp 1`, smallest working `--ep`)

### 1.1 Start with the real engine

```bash
export DEEPINFER_HF_MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017
python -m deepinfer_bridge \
  --model-path "$DEEPINFER_HF_MODEL_DIR" \
  --deepinfer-root /path/to/DeepInfer \
  --ep 8 --cp 1 --dp 1 \
  --max-model-len 32768 \
  --host 127.0.0.1 --port 8000 \
  --log-format json
```

Record: time from launch to `engine ready: 1 DP rank(s)`. Weight loading plus graph capture
can take many minutes; `--startup-timeout` (default 3600 s) is the budget.

### 1.2 Liveness and discovery

```bash
curl -s localhost:8000/healthz | jq
curl -s localhost:8000/v1/models | jq
```

`/healthz` returns 503 until every rank reports ready, 200 after. `/v1/models` must list the
`--served-model-name`.

### 1.3 One streamed and one buffered completion

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","thinking":{"type":"disabled"},
       "messages":[{"role":"user","content":"Say hi in five words."}]}' | jq -r '.choices[0].message.content'

curl -sN localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","stream":true,"thinking":{"type":"disabled"},
       "messages":[{"role":"user","content":"Say hi in five words."}]}'
```

### 1.4 Token parity against the offline reference (assumption A3/A5)

The point is to prove the bridge's sampling mapping and EOS handling match
`LLM.generate()`. Batch-invariant mode removes batching nondeterminism **for this
comparison only** — it is not the serving default.

```bash
# Terminal A: server with determinism forced
DEEPINFER_BATCH_INVARIANT=1 python -m deepinfer_bridge ... --ep 8 --dp 1

# Terminal B: same token IDs straight through the offline API, temperature 0 + fixed seed
cd DeepInfer && DEEPINFER_BATCH_INVARIANT=1 python - <<'PY'
from _config import create_llm, setup_common_env
from deepinfer.sampling_params import SamplingParams
setup_common_env(model_name="deepseek-ai/DeepSeek-V4.1-Flash-1017", sampler_force_topk=4096,
                 set_override_experts=True, enable_eplb=True)
llm = create_llm("deepseek-ai/DeepSeek-V4.1-Flash-1017", 0, ep_size=8, cp_size=1, data_parallel_size=1)
sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=64, seed=1234)
out = llm.generate(prompts=["Say hi in five words."], sampling_params=sp, use_tqdm=False)
print([o.token_id for o in out[0]])
llm.close()
PY
```

Compare against the token IDs the bridge produced for the same prompt. They must match
exactly. **If they differ, stop and investigate before any other Phase 1 work** — a parity
failure means the prompt rendering or the SamplingParams mapping is wrong, which would
silently corrupt every later result.

### 1.5 EOS and `finish_reason` (assumption A3)

```bash
# expect finish_reason "length" and exactly 5 tokens
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","max_tokens":5,"thinking":{"type":"disabled"},
       "messages":[{"role":"user","content":"Count to twenty."}]}' | jq '.choices[0].finish_reason'
```

Check the decoded text contains no EOS markup and that a natural stop reports `"stop"`.

**Unresolved risk:** `engine_real.py` drops a token only when
`finished and finish_reason == "stop" and token_id == eos_id`. Under DSpark/MTP several
tokens can arrive in one `step()`. This logic has never run. Verify explicitly that
accepted draft tokens are not dropped or reordered — see `num_target_tokens` handling in
`examples/offline_inference.py:82-85` for how DeepInfer itself accounts for them.

### 1.6 Abort on disconnect (assumptions A7/A8)

```bash
# start a long generation and kill the client mid-stream
timeout 3 curl -sN localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","stream":true,"max_tokens":4096,
       "messages":[{"role":"user","content":"Write a long essay."}]}' >/dev/null

curl -s localhost:8000/stats | jq '{outstanding, aborts_acked, load_per_rank}'
```

Expect `outstanding` to return to 0 and `aborts_acked` to increment. Watch engine logs and
GPU utilisation to confirm work actually stopped.

### 1.7 Acceptance for Phase 1

```bash
# 20 sequential requests, mixed stream/non-stream, zero leaked handles
for i in $(seq 1 20); do
  curl -s -o /dev/null localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4.1-flash","thinking":{"type":"disabled"},
         "messages":[{"role":"user","content":"hi"}]}'
done
curl -s localhost:8000/stats | jq '.outstanding'    # expect 0
```

Plus token parity on 5 prompts (step 1.4).

---

## Phase 2 — correctness features

Run each with `thinking` disabled and enabled.

| Check | Command sketch | Expected |
|---|---|---|
| Thinking split | request with `"thinking":{"type":"enabled"}` | `reasoning_content` separate from `content`; `usage` has reasoning tokens |
| Tool calls (3-turn loop, 2+ tools) | assistant `tool_calls` → tool result → final answer, streaming and not | valid JSON arguments, `finish_reason == "tool_calls"` |
| Structured output (A4) | `response_format` json_schema, then regex | grammar honoured; see below |
| Stop sequences | `"stop":["..."]` | stream ends at the sequence, engine request aborted |
| Long context | prompt just under and just over `--max-model-len` | success / HTTP 400 respectively |
| Other protocols | `POST /v1/responses`, `POST /v1/messages` | one streaming + one buffered call each |

**Structured output under thinking (A4) is the least certain item in this runbook.**
`protocol.py` passes `so_args["trigger_token_ids"] = tokenizer.encode(SpecialToken.ThinkingEnd)`
so the grammar engages after reasoning. DeepInfer accepts `trigger_token_ids` and
`bound_token_id` on `so_args`, and `schema_type` is limited to `json` / `regex` / `ebnf`
**[verified by reading the source, not by running it]**. Whether the trigger fires at the
right moment is unverified. If it misbehaves, either fix the trigger semantics after reading
DeepInfer's `structure_output` code, or run with `--no-constrained-decoding` and document it.

---

## Phase 3 — multi-rank (DP > 1) and lock-step (A1, A2, A6)

### 3.1 Why this is delicate

`LLMEngine.has_unfinished_requests()` performs
`dist.all_reduce(..., op=ReduceOp.MAX, group=dp_group)` **only when `dp_group.size() > 1`**,
and `step()` contains attention-DP and MoE-EP collectives. Every rank must therefore call
them in the same sequence. **[verified:
`deepinfer/engine/llm_engine.py:665-679`]**

`flush_inflight_reqs()` calls `_run_workers("flush_inflight_reqs")`, which is also
collective **[verified: `deepinfer/engine/llm_engine.py:977-982`]**. The bridge only calls it
after a step, and because `has_unfinished_requests()` is all-reduced every rank makes the
same decision at the same time, so `need_flush` stays in step across ranks.

### 3.2 Start multi-rank

```bash
python -m deepinfer_bridge ... --ep 8 --cp 1 --dp 8
```

Remember `--dp * --cp` must be a multiple of `--ep`; the bridge rejects a bad combination at
startup with a clear message rather than letting DeepInfer's
`assert world_size % ep_size == 0` fire mid-load. **[verified: `tests/test_config.py`]**

### 3.3 Idle lock-step test (A1)

Leave the server idle for 10 minutes and sample per-rank CPU:

```bash
for i in $(seq 1 60); do
  ps -o pid,pcpu,etime,cmd -C python3 | grep -c deepinfer
  sleep 10
done
top -b -n1 -p "$(pgrep -d, -f deepinfer_bridge)" | tail -20
```

Target: **< 5% CPU per rank at idle**, no hang, no drift. The bridge sleeps 2 ms between idle
polls (`engine_real.IDLE_SLEEP_S`). If ranks spin, replace the fixed sleep with a coarser
backoff after N idle polls — but keep every rank's call *sequence* identical, since the
collectives synchronise them. Document the added latency.

### 3.4 Burst test

```bash
seq 1 64 | xargs -P64 -I{} curl -s -o /dev/null \
  localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","thinking":{"type":"disabled"},
       "messages":[{"role":"user","content":"hello {}"}]}'
curl -s localhost:8000/stats | jq '.load_per_rank'
```

Routing is least-outstanding-requests; `load_per_rank` should stay balanced and no rank
should stall the others.

### 3.5 Fault test

```bash
pkill -9 -f 'engine_mock|deepinfer'      # kill one rank
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/healthz      # expect 503
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/v1/models    # expect 503 on new work
```

In-flight streams must terminate and buffered requests must return 5xx. Worker death is
detected by the event-reader thread and reported as `fatal`
**[verified for the crash path: `tests/test_engine_client.py::test_rank_crash_during_startup_is_reported`]**.

Recovery requires a restart: ranks are lock-step, so they must come back **together**. A
supervised per-rank restart would deadlock the surviving ranks inside a collective.

### 3.6 Shutdown test

```bash
kill -TERM <bridge pid>
ps aux | grep -c '[d]eepinfer'     # expect 0 orphans
```

`SIGTERM` drains in-flight requests for `--shutdown-grace` seconds (default 30), then aborts
and joins each worker, escalating to `SIGTERM` on the child after 15 s.

---

## Phase 4 — real Harness integration

```bash
node --version          # need >= 22.19
npx @deepseek-ai/dsh web
```

Settings → Models → **Add a custom provider**:

| Field | Value |
|---|---|
| Provider ID | `deepinfer` (lowercase; permanent) |
| Base URL | `http://127.0.0.1:8000/v1` |
| API Protocol | `openai-completions` |
| Credential | the `--api-key` value |
| Model ID | `deepseek-v4.1-flash` (= `--served-model-name`) |

Harness discovers models via `GET /models`; the model must appear there to be selectable.
Note that when `--api-key` is set, **every route except `/healthz` requires the bearer
token**, including `/models` and `/metrics` — configure the credential before expecting
discovery to succeed. **[verified: `tests/test_hardening.py`]**

Then run three real agent sessions and record, per turn: TTFT, tokens/s, prefill share,
tool-call parse failures, malformed JSON arguments. Compare against DeepSeek's official API
on the same tasks.

---

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `/healthz` stays 503 | weights still loading, or a rank died | check engine logs; `--startup-timeout` is 1 h by default |
| Hangs at startup with DP>1 | one rank failed to join, others blocked in a collective | check every rank's log; they must fail together |
| Idle CPU pinned | idle poll spinning | see 3.3 |
| OOM during graph capture | `--gpu-memory-utilization` too high | lower it; see the `cudagraph OOM` note in `deepinfer/entrypoints/llm.py` |
| Very slow first request | graph capture on the first batch | warm up with one throwaway request |
| `world_size % ep_size` assert | bad `--ep/--cp/--dp` | the bridge validates this up front; check the error text |
| pip stalls inside WSL | IPv6 resolved but unreachable | use `curl -4` / `pip --timeout`; see OPERATIONS.md |

---

## What remains unproven

Phases 1-4 and 6 have **not** been executed. The specific things most likely to break, in
order:

1. **A3 under speculative decoding** — multi-token steps and the EOS drop.
2. **A4** — `trigger_token_ids` deferring the grammar until after reasoning.
3. **A2** — the single `flush_inflight_reqs()` drain under zero-overhead scheduling.
4. **A1** — idle lock-step behaviour over long periods.
5. **A6** — `CUDA_VISIBLE_DEVICES` inside spawned workers. The mapping is algebraically
   identical to `examples/_config.py:55-63` and `_launch.py` uses the same spawn +
   `daemon=False` model **[verified by reading the source]**, but it has not run.
6. **A5** — the `LLM(...)` kwargs. These match `examples/offline_inference.py:45-57`
   exactly **[verified by reading the source]**. Note `LLM.__init__` itself hardcodes
   `offline_inference=True` for the EngineArgs it builds, so the serving path inherits
   offline-mode engine settings; what that flag changes is not yet established.
