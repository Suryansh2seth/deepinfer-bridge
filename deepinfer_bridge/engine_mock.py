"""GPU-free stand-in for an engine rank. Same IPC protocol as engine_real; emits canned text.

Used by the tests and for trying the bridge against Harness on a laptop. Output is
"Hello world!" (preceded by the end-of-thinking token when the prompt opens a thinking block).
Per-token delay: env BRIDGE_MOCK_DELAY_MS (default 5).
"""

from __future__ import annotations

import os
import queue as pyqueue
import time
import traceback

from .config import BridgeConfig


def mock_worker(rank: int, cfg: BridgeConfig, cmd_q, evt_q) -> None:
    try:
        _run(rank, cfg, cmd_q, evt_q)
    except BaseException:  # noqa: BLE001
        evt_q.put(("fatal", rank, traceback.format_exc()))
        raise


def _run(rank: int, cfg: BridgeConfig, cmd_q, evt_q) -> None:
    from deepseek_recipe import SpecialToken, Tokenizer

    if not cfg.tokenizer_path:
        raise RuntimeError("the mock engine needs cfg.tokenizer_path (a tokenizer.json); pass --tokenizer-path")
    tok = Tokenizer.from_file(cfg.tokenizer_path)
    think_start = tok.encode(SpecialToken.ThinkingStart.default_str())
    think_end = tok.encode(SpecialToken.ThinkingEnd.default_str())
    reply = os.environ.get("BRIDGE_MOCK_REPLY", "Hello world!")
    delay = float(os.environ.get("BRIDGE_MOCK_DELAY_MS", "5")) / 1000.0
    reply_ids = tok.encode(reply)

    active: dict[str, dict] = {}
    evt_q.put(("ready", rank, {"mock": True}))
    while True:
        try:
            while True:
                cmd = cmd_q.get_nowait()
                if cmd[0] == "add":
                    _, rid, ids, sp = cmd
                    script = (think_end if ids[-len(think_start) :] == think_start else []) + reply_ids
                    active[rid] = {"script": script, "pos": 0, "max": sp["max_tokens"]}
                elif cmd[0] == "abort":
                    active.pop(cmd[1], None)
                    evt_q.put(("aborted", cmd[1]))
                elif cmd[0] == "stop":
                    return
        except pyqueue.Empty:
            pass
        if not active:
            time.sleep(0.002)
            continue
        time.sleep(delay)
        for rid in list(active):
            st = active[rid]
            token = st["script"][st["pos"]]
            st["pos"] += 1
            if st["pos"] >= st["max"] and st["pos"] < len(st["script"]):
                evt_q.put(("out", rid, [token], True, "length"))
                del active[rid]
            elif st["pos"] >= len(st["script"]):
                evt_q.put(("out", rid, [token], True, "stop"))
                del active[rid]
            else:
                evt_q.put(("out", rid, [token], False, None))
