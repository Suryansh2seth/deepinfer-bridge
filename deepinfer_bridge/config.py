"""Bridge configuration. Must stay picklable: it is sent to engine worker processes."""

from __future__ import annotations

import dataclasses
import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

# Every field can be overridden from the environment as DEEPINFER_BRIDGE_<FIELD_NAME>.
# CLI flags win over environment variables, which win over the defaults below.
ENV_PREFIX = "DEEPINFER_BRIDGE_"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})


class ConfigError(ValueError):
    """Invalid configuration. The message is user-facing."""


def is_loopback(host: str) -> bool:
    """True only for addresses that cannot be reached from another machine."""
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _coerce(annotation: str, raw: str) -> object:
    """Convert an environment string using the dataclass field's declared type."""
    if annotation == "bool":
        lowered = raw.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ConfigError(f"expected a boolean, got {raw!r}")
    if annotation == "int":
        return int(raw)
    if annotation == "float":
        return float(raw)
    if annotation in ("int | None", "Optional[int]"):
        return None if raw.strip() == "" else int(raw)
    if annotation in ("str | None", "Optional[str]"):
        return None if raw == "" else raw
    if annotation == "str":
        return raw
    raise ConfigError(f"cannot set this field from the environment (type {annotation})")


@dataclass
class BridgeConfig:
    # --- serving -----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    served_model_name: str = "deepseek-v4.1-flash"
    api_key: str | None = None  # if set, require "Authorization: Bearer <key>"
    max_outstanding: int = 256  # admission limit (requests in flight); beyond it -> HTTP 503
    startup_timeout_s: float = 3600.0  # weight loading + graph capture can be slow
    request_timeout_s: float = 600.0  # wall-clock budget for one generation
    shutdown_grace_s: float = 30.0  # drain in-flight requests this long before aborting
    max_body_bytes: int = 8 << 20  # reject larger request bodies with HTTP 413
    insecure: bool = False  # allow binding beyond loopback with no --api-key
    log_prompts: bool = False  # never log prompt/completion text by default
    log_level: str = "info"
    log_format: str = "text"  # "text" (key=value) or "json"

    # --- model -------------------------------------------------------------
    model_path: str = "deepseek-ai/DeepSeek-V4.1-Flash-1017"  # HF id or local dir
    tokenizer_path: str | None = None  # tokenizer.json; default: <model dir>/tokenizer.json
    deepinfer_root: str | None = None  # in-tree checkout of DeepInfer (added to sys.path)
    mock: bool = False  # no GPUs: serve canned output through the same protocol stack

    # --- parallelism (same meaning as examples/offline_inference.py) -------
    ep: int = 8
    cp: int = 1
    dp: int | None = None  # default: ep // cp. One engine process per DP rank.

    # --- engine limits -----------------------------------------------------
    max_model_len: int = 32768
    max_num_batched_tokens: int = 32768
    max_num_seqs: int = 32  # per DP rank
    gpu_memory_utilization: float = 0.85

    # --- DeepInfer features ------------------------------------------------
    dspark: bool = True  # DSpark speculative decoding
    nextn: int = 5  # max draft length
    swa_bounded_replay: bool = True
    eplb: bool = True
    sampler_force_topk: int = 4096
    constrained_decoding: bool = True  # map json/regex response formats to DeepInfer so_args
    extra_env: dict[str, str] = field(default_factory=dict)

    @property
    def dp_size(self) -> int:
        return self.dp if self.dp is not None else max(self.ep // self.cp, 1)

    @property
    def auth_required(self) -> bool:
        return bool(self.api_key)

    def validate(self) -> None:
        """Reject flag combinations that would fail late (or silently misbehave).

        Mirrors the asserts DeepInfer itself makes at startup (see
        examples/offline_inference.py: ``assert world_size % ep_size == 0``) so a bad
        combination surfaces as a clear message instead of a CUDA traceback several
        minutes into weight loading.
        """
        if self.ep < 1:
            raise ConfigError(f"--ep must be >= 1 (got {self.ep})")
        if self.cp < 1:
            raise ConfigError(f"--cp must be >= 1 (got {self.cp})")
        if self.ep % self.cp != 0:
            raise ConfigError(f"--cp ({self.cp}) must divide --ep ({self.ep}); dp defaults to ep // cp")
        if self.dp is not None:
            if self.dp < 1:
                raise ConfigError(f"--dp must be >= 1 (got {self.dp})")
            world_size = self.dp * self.cp
            if world_size % self.ep != 0:
                raise ConfigError(
                    f"--dp * --cp ({world_size}) must be a multiple of --ep ({self.ep}); "
                    "DeepInfer asserts world_size % ep_size == 0"
                )
        if self.max_model_len < 1:
            raise ConfigError(f"--max-model-len must be >= 1 (got {self.max_model_len})")
        if self.max_num_batched_tokens < 1:
            raise ConfigError(f"--max-num-batched-tokens must be >= 1 (got {self.max_num_batched_tokens})")
        if self.max_num_seqs < 1:
            raise ConfigError(f"--max-num-seqs must be >= 1 (got {self.max_num_seqs})")
        if self.max_outstanding < 1:
            raise ConfigError(f"--max-outstanding must be >= 1 (got {self.max_outstanding})")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ConfigError(f"--gpu-memory-utilization must be in (0, 1] (got {self.gpu_memory_utilization})")
        if self.startup_timeout_s <= 0:
            raise ConfigError(f"--startup-timeout must be > 0 (got {self.startup_timeout_s})")
        if self.request_timeout_s <= 0:
            raise ConfigError(f"--request-timeout must be > 0 (got {self.request_timeout_s})")
        if self.shutdown_grace_s < 0:
            raise ConfigError(f"--shutdown-grace must be >= 0 (got {self.shutdown_grace_s})")
        if self.max_body_bytes < 1:
            raise ConfigError(f"--max-body-bytes must be >= 1 (got {self.max_body_bytes})")
        if not 1 <= self.port <= 65535:
            raise ConfigError(f"--port must be in 1..65535 (got {self.port})")
        if self.nextn < 0:
            raise ConfigError(f"--nextn must be >= 0 (got {self.nextn})")
        if self.log_format not in ("text", "json"):
            raise ConfigError(f"--log-format must be 'text' or 'json' (got {self.log_format!r})")

        # An empty key would otherwise read as "no auth" and quietly expose the server.
        if self.api_key is not None and not self.api_key.strip():
            raise ConfigError("--api-key must be non-empty; omit it entirely to disable authentication")

        if not self.insecure and self.api_key is None and not is_loopback(self.host):
            raise ConfigError(
                f"refusing to bind {self.host!r} without --api-key: it would be reachable from the "
                "network with no authentication. Pass --api-key, bind 127.0.0.1, or pass --insecure."
            )

        if self.mock and self.dp_size > 1:
            raise ConfigError(f"--mock serves a single DP rank, but --ep/--cp imply dp={self.dp_size}")


def apply_env_overrides(cfg: BridgeConfig, environ: Mapping[str, str] | None = None) -> BridgeConfig:
    """Apply DEEPINFER_BRIDGE_<FIELD> overrides for every field. Mutates and returns cfg."""
    env = os.environ if environ is None else environ
    for f in dataclasses.fields(cfg):
        if f.name == "extra_env":  # a dict has no sensible single-variable encoding
            continue
        key = ENV_PREFIX + f.name.upper()
        if key in env:
            try:
                setattr(cfg, f.name, _coerce(str(f.type), env[key]))
            except (ValueError, ConfigError) as exc:
                raise ConfigError(f"invalid {key}={env[key]!r}: {exc}") from exc
    return cfg
