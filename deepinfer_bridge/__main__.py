"""CLI entry point.

Precedence is CLI flag > environment variable > default. Every BridgeConfig field is
reachable from the environment as DEEPINFER_BRIDGE_<FIELD_NAME> (see config.py), so flags
that exist only for convenience can always be overridden in a container.
"""

from __future__ import annotations

import argparse
import logging
import os

from .config import ENV_PREFIX, BridgeConfig, ConfigError, apply_env_overrides

log = logging.getLogger("deepinfer_bridge")

# argparse.SUPPRESS means "absent from the namespace unless the user passed it", which is
# what lets an environment variable sit between the CLI and the default.
_UNSET = argparse.SUPPRESS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deepinfer-bridge", description="OpenAI-compatible bridge to DeepInfer")

    p.add_argument("--host", default=_UNSET)
    p.add_argument("--port", type=int, default=_UNSET)
    p.add_argument(
        "--served-model-name",
        dest="served_model_name",
        default=_UNSET,
        help="Model ID clients use (Harness 'Model ID')",
    )
    p.add_argument(
        "--api-key",
        dest="api_key",
        default=_UNSET,
        help=f"require this bearer token (env {ENV_PREFIX}API_KEY or BRIDGE_API_KEY)",
    )

    p.add_argument("--model-path", dest="model_path", default=_UNSET, help="HF id or local checkpoint dir")
    p.add_argument(
        "--tokenizer-path",
        dest="tokenizer_path",
        default=_UNSET,
        help="tokenizer.json (default: <model dir>/tokenizer.json)",
    )
    p.add_argument(
        "--deepinfer-root",
        dest="deepinfer_root",
        default=_UNSET,
        help="path to a DeepInfer checkout if it is not pip-installed",
    )
    p.add_argument(
        "--mock", action="store_true", default=_UNSET, help="no GPU: canned replies through the real protocol stack"
    )

    p.add_argument("--ep", type=int, default=_UNSET)
    p.add_argument("--cp", type=int, default=_UNSET)
    p.add_argument("--dp", type=int, default=_UNSET)

    p.add_argument("--max-model-len", dest="max_model_len", type=int, default=_UNSET)
    p.add_argument("--max-num-batched-tokens", dest="max_num_batched_tokens", type=int, default=_UNSET)
    p.add_argument("--max-num-seqs", dest="max_num_seqs", type=int, default=_UNSET, help="per DP rank")
    p.add_argument("--gpu-memory-utilization", dest="gpu_memory_utilization", type=float, default=_UNSET)

    p.add_argument("--max-outstanding", dest="max_outstanding", type=int, default=_UNSET)
    p.add_argument(
        "--startup-timeout",
        dest="startup_timeout_s",
        type=float,
        default=_UNSET,
        help="seconds to wait for every rank to report ready",
    )
    p.add_argument(
        "--request-timeout",
        dest="request_timeout_s",
        type=float,
        default=_UNSET,
        help="wall-clock budget for one generation, in seconds",
    )
    p.add_argument(
        "--shutdown-grace",
        dest="shutdown_grace_s",
        type=float,
        default=_UNSET,
        help="drain in-flight requests this long on SIGTERM before aborting",
    )
    p.add_argument(
        "--max-body-bytes",
        dest="max_body_bytes",
        type=int,
        default=_UNSET,
        help="reject larger request bodies with HTTP 413",
    )
    p.add_argument(
        "--insecure", action="store_true", default=_UNSET, help="allow binding a non-loopback address with no --api-key"
    )

    p.add_argument("--no-dspark", dest="dspark", action="store_false", default=_UNSET)
    p.add_argument("--nextn", type=int, default=_UNSET)
    p.add_argument("--no-swa-bounded-replay", dest="swa_bounded_replay", action="store_false", default=_UNSET)
    p.add_argument("--no-eplb", dest="eplb", action="store_false", default=_UNSET)
    p.add_argument("--no-constrained-decoding", dest="constrained_decoding", action="store_false", default=_UNSET)
    p.add_argument("--sampler-force-topk", dest="sampler_force_topk", type=int, default=_UNSET)

    p.add_argument("--log-level", dest="log_level", default=_UNSET)
    p.add_argument("--log-format", dest="log_format", choices=("text", "json"), default=_UNSET)
    p.add_argument(
        "--log-prompts",
        dest="log_prompts",
        action="store_true",
        default=_UNSET,
        help="include prompt/completion text in logs (off by default)",
    )
    return p


def resolve_config(argv: list[str] | None = None) -> BridgeConfig:
    """Merge defaults, environment and CLI into a validated BridgeConfig."""
    parser = build_parser()
    args = parser.parse_args(argv)
    provided = vars(args)

    cfg = apply_env_overrides(BridgeConfig())
    for key, value in provided.items():
        setattr(cfg, key, value)

    # Legacy single-purpose variable, kept working for existing deployments.
    if "api_key" not in provided and f"{ENV_PREFIX}API_KEY" not in os.environ:
        legacy = os.environ.get("BRIDGE_API_KEY")
        if legacy is not None:
            cfg.api_key = legacy or None

    if cfg.mock:
        # The mock engine has no parallelism; force one rank so validate() cannot trip.
        cfg.ep, cfg.cp, cfg.dp = 1, 1, 1

    try:
        cfg.validate()
    except ConfigError as exc:
        parser.error(str(exc))  # prints usage and exits 2
    return cfg


def main(argv: list[str] | None = None) -> None:
    cfg = resolve_config(argv)

    from .logging_setup import configure_logging

    configure_logging(cfg.log_level, cfg.log_format)

    import uvicorn

    from .server import create_app

    if cfg.max_num_batched_tokens < cfg.max_model_len:
        log.warning(
            "max_num_batched_tokens (%d) < max_model_len (%d): long prompts will prefill in "
            "several chunks (chunked_prefill is on), which raises TTFT for long contexts",
            cfg.max_num_batched_tokens,
            cfg.max_model_len,
        )
    # log_config=None keeps uvicorn from installing its own handlers over ours.
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level=cfg.log_level, log_config=None)


if __name__ == "__main__":
    main()
