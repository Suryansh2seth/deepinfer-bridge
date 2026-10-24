"""One DeepInfer engine process (one data-parallel rank).

Drives DeepInfer's incremental engine API (LLMEngine.add_request / step / abort_request)
instead of the blocking LLM.generate(), which is what makes online streaming possible.

IMPORTANT (lock-step): with DP > 1, `has_unfinished_requests()` and `step()` contain
collectives (attention-DP / MoE-EP). Every rank must call them in the same sequence, so every
rank runs this same loop continuously; idle ranks just poll with a short sleep between polls.
The collective itself synchronises ranks, so no extra signalling is needed.

Setup mirrors DeepInfer's examples/_config.py (create_llm / setup_common_env). If those change
upstream, update `_build_llm`.

NOT TESTED against real hardware by the author of this file (no SM100 / Ascend 950 available).
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
import traceback
from pathlib import Path

from .config import BridgeConfig

IDLE_SLEEP_S = 0.002

# Must match examples/_config.py::DEFAULT_MODEL_NAME, so the DEEPINFER_HF_MODEL_DIR override
# applies to exactly the same model id as it does in DeepInfer's own examples.
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


def resolve_model_dir(model_path: str) -> str:
    """HF id or local dir -> local dir containing config.json (done in the parent as well,
    so the HTTP process can find tokenizer.json without importing torch).

    Honours DEEPINFER_HF_MODEL_DIR for the default model id, mirroring
    examples/_config.py::resolve_model_path, so pointing the bridge at a local checkpoint
    works the same way it does for DeepInfer's own examples.
    """
    if model_path == DEFAULT_MODEL_NAME:
        override = os.environ.get("DEEPINFER_HF_MODEL_DIR")
        if override:
            model_path = override
    path = Path(model_path).expanduser()
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model_path))
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"HF checkpoint requires config.json: {path}")
    return str(path.resolve())


def _apply_env(cfg: BridgeConfig) -> None:
    env = os.environ
    env.setdefault("RANK", "0")
    env.setdefault("WORLD_SIZE", "1")
    if int(env["WORLD_SIZE"]) != 1:
        raise RuntimeError("deepinfer-bridge supports single-node deployments only")
    # Shared with examples/_config.py::setup_common_env
    env.setdefault("DEEPINFER_USE_NEXTN_DECODE", str(cfg.nextn if cfg.dspark else 0))
    env["DEEPINFER_SAMPLER_FORCE_TOPK"] = str(cfg.sampler_force_topk)
    env["DEEPINFER_EXPERT_COUNT_WINDOW_SIZE"] = "3"
    if cfg.eplb:
        env.setdefault("DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS", "32")
        env["DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL"] = "30"
        env["DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE"] = "1"
    # README feature switches
    if cfg.dspark:
        env.setdefault("DEEPINFER_ENABLE_MTP_VARLEN", "1")
    if cfg.swa_bounded_replay:
        env.setdefault("DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE", "compact")
        env.setdefault("DEEPINFER_YOCO_PREFILL_KEEP_TOKENS", "140")
    env.update(cfg.extra_env)


def _set_visible_devices(rank: int, cp: int, device_count: int) -> None:
    offset = int(os.environ.get("DEVICE_OFFSET", "0"))
    local = rank * cp % device_count // cp
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(offset + i) for i in range(local * cp, (local + 1) * cp))
    from deep_cluster.hardware import accelerator_kind

    if accelerator_kind() == "ascend":
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]


def _build_llm(rank: int, cfg: BridgeConfig, model_dir: str):
    if cfg.deepinfer_root and cfg.deepinfer_root not in sys.path:
        sys.path.insert(0, cfg.deepinfer_root)
    _apply_env(cfg)
    from deep_cluster.hardware import get_actual_num_gpus  # no torch/CUDA init yet

    _set_visible_devices(rank, cfg.cp, get_actual_num_gpus())
    import torch  # noqa: F401  (must come after CUDA_VISIBLE_DEVICES)
    from deepinfer.entrypoints.llm import LLM

    return LLM(
        model=model_dir,
        expert_parallel_size=cfg.ep,
        context_parallel_size=cfg.cp,
        engine_rank=rank,
        trust_remote_code=True,
        chunked_prefill=True,
        decode_sequence=False,
        disable_log_stats=False,
        data_parallel_size=cfg.dp_size,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        max_model_len=cfg.max_model_len,
        max_num_seqs=cfg.max_num_seqs,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        # Copied from examples/offline_inference.py
        lookahead_tokens=min(1, int(os.getenv("DEEPINFER_USE_NEXTN_DECODE", "5"))),
    )


def engine_worker(rank: int, cfg: BridgeConfig, cmd_q, evt_q) -> None:
    try:
        _run(rank, cfg, cmd_q, evt_q)
    except BaseException:  # noqa: BLE001
        evt_q.put(("fatal", rank, traceback.format_exc()))
        raise


def _run(rank: int, cfg: BridgeConfig, cmd_q, evt_q) -> None:
    import queue as pyqueue

    import numpy as np

    model_dir = resolve_model_dir(cfg.model_path)
    llm = _build_llm(rank, cfg, model_dir)

    from deepinfer._C.engine import RequestOutput  # pyright: ignore[reportMissingModuleSource]
    from deepinfer.sampling_params import SamplingParams

    engine = llm.llm_engine
    eos_id = engine.eos_id
    evt_q.put(("ready", rank, {"eos_id": eos_id}))

    need_flush = False
    stopping = False
    while not stopping:
        # 1) apply pending commands (local only; no collectives)
        while True:
            try:
                cmd = cmd_q.get_nowait()
            except pyqueue.Empty:
                break
            if cmd[0] == "add":
                _, rid, token_ids, sampling = cmd
                try:
                    engine.add_request(
                        user_id="0",
                        request_id=rid,
                        sampling_params=SamplingParams(**sampling),
                        prompt_token_ids=np.asarray(token_ids, dtype=np.int32),
                    )
                except Exception:  # noqa: BLE001  (bad request must not kill the rank)
                    evt_q.put(("out", rid, [], True, "error:" + traceback.format_exc(limit=1)))
            elif cmd[0] == "abort":
                # A7: aborting an unknown or already-finished id is harmless -- DeepInfer's
                # abort_request returns (0, 0) for it. Never let an abort kill the rank.
                with contextlib.suppress(Exception):
                    engine.abort_request(cmd[1])
                evt_q.put(("aborted", cmd[1]))
            elif cmd[0] == "stop":
                stopping = True
        if stopping:
            break

        # 2) one lock-step engine iteration (collective across DP ranks)
        if engine.has_unfinished_requests():
            outputs = engine.step()
            need_flush = True
        elif need_flush:
            outputs = engine.flush_inflight_reqs()  # drain zero-overhead-scheduling in-flight work
            need_flush = False
        else:
            time.sleep(IDLE_SLEEP_S)
            continue

        # 3) group this step's token outputs per request and ship them
        per_req: dict[str, list] = {}
        for raw in outputs:
            out = RequestOutput.clone(raw)  # step outputs are recycled on the next step
            slot = per_req.setdefault(out.request_id, [[], False, None])
            token = out.token_id
            if out.finished and out.finish_reason == "stop" and token == eos_id:
                pass  # the terminating EOS is not content
            else:
                slot[0].append(token)
            if out.finished:
                slot[1], slot[2] = True, out.finish_reason
        for rid, (tokens, finished, reason) in per_req.items():
            evt_q.put(("out", rid, tokens, finished, reason))

    llm.close()
