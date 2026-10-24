"""Configuration validation, environment overrides and CLI precedence.

These are pure unit tests: no engine, no tokenizer.
"""

import pytest

from deepinfer_bridge.__main__ import resolve_config
from deepinfer_bridge.config import (
    ENV_PREFIX,
    BridgeConfig,
    ConfigError,
    apply_env_overrides,
    is_loopback,
)


def test_defaults_validate():
    BridgeConfig().validate()  # must not raise


def test_cp_must_divide_ep():
    with pytest.raises(ConfigError, match="divide"):
        BridgeConfig(ep=8, cp=3).validate()


def test_dp_must_satisfy_the_deepinfer_world_size_assert():
    # DeepInfer asserts world_size % ep_size == 0 (examples/offline_inference.py).
    with pytest.raises(ConfigError, match="multiple of --ep"):
        BridgeConfig(ep=4, cp=1, dp=3).validate()


def test_valid_dp_combinations_pass():
    # world_size = dp * cp must be a multiple of ep.
    BridgeConfig(ep=8, cp=2, dp=4).validate()  # 4 * 2 == 8, 8 % 8 == 0
    BridgeConfig(ep=8, cp=1, dp=8).validate()  # 8 * 1 == 8
    BridgeConfig(ep=1, cp=1, dp=1).validate()


def test_dp_default_is_ep_over_cp():
    assert BridgeConfig(ep=8, cp=2).dp_size == 4
    assert BridgeConfig(ep=8, cp=1, dp=3).dp_size == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("ep", 0),
        ("cp", 0),
        ("max_model_len", 0),
        ("max_num_batched_tokens", 0),
        ("max_num_seqs", 0),
        ("max_outstanding", 0),
        ("max_body_bytes", 0),
        ("port", 0),
        ("port", 70000),
        ("nextn", -1),
        ("startup_timeout_s", 0),
        ("request_timeout_s", 0),
        ("shutdown_grace_s", -1),
        ("gpu_memory_utilization", 0.0),
        ("gpu_memory_utilization", 1.5),
    ],
)
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises(ConfigError):
        BridgeConfig(**{field: value}).validate()


def test_empty_api_key_is_rejected_because_it_reads_as_no_auth():
    with pytest.raises(ConfigError, match="non-empty"):
        BridgeConfig(api_key="   ").validate()


def test_non_loopback_without_key_is_refused():
    with pytest.raises(ConfigError, match="refusing to bind"):
        BridgeConfig(host="0.0.0.0").validate()


def test_non_loopback_with_key_or_insecure_is_allowed():
    BridgeConfig(host="0.0.0.0", api_key="k").validate()
    BridgeConfig(host="0.0.0.0", insecure=True).validate()


def test_loopback_hosts_need_no_key():
    for host in ("127.0.0.1", "::1", "localhost"):
        assert is_loopback(host)
        BridgeConfig(host=host).validate()
    assert not is_loopback("0.0.0.0")
    assert not is_loopback("10.0.0.5")


def test_mock_rejects_multiple_ranks():
    with pytest.raises(ConfigError, match="single DP rank"):
        BridgeConfig(mock=True, ep=8, cp=1).validate()


def test_bad_log_format_is_rejected():
    with pytest.raises(ConfigError, match="log-format"):
        BridgeConfig(log_format="xml").validate()


# --- environment overrides ------------------------------------------------------------


def test_env_overrides_every_field_type():
    cfg = apply_env_overrides(
        BridgeConfig(),
        {
            f"{ENV_PREFIX}PORT": "9999",
            f"{ENV_PREFIX}REQUEST_TIMEOUT_S": "12.5",
            f"{ENV_PREFIX}LOG_PROMPTS": "true",
            f"{ENV_PREFIX}INSECURE": "yes",
            f"{ENV_PREFIX}HOST": "0.0.0.0",
            f"{ENV_PREFIX}DP": "2",
            f"{ENV_PREFIX}API_KEY": "from-env",
            f"{ENV_PREFIX}LOG_FORMAT": "json",
        },
    )
    assert cfg.port == 9999
    assert cfg.request_timeout_s == 12.5
    assert cfg.log_prompts is True
    assert cfg.insecure is True
    assert cfg.host == "0.0.0.0"
    assert cfg.dp == 2
    assert cfg.api_key == "from-env"
    assert cfg.log_format == "json"


def test_env_empty_optional_becomes_none():
    cfg = apply_env_overrides(BridgeConfig(api_key="x", dp=4), {f"{ENV_PREFIX}API_KEY": "", f"{ENV_PREFIX}DP": ""})
    assert cfg.api_key is None
    assert cfg.dp is None


def test_env_boolean_accepts_false_spellings():
    for raw in ("0", "false", "no", "off", ""):
        cfg = apply_env_overrides(BridgeConfig(), {f"{ENV_PREFIX}LOG_PROMPTS": raw})
        assert cfg.log_prompts is False


def test_invalid_env_value_names_the_variable():
    with pytest.raises(ConfigError, match=f"{ENV_PREFIX}PORT"):
        apply_env_overrides(BridgeConfig(), {f"{ENV_PREFIX}PORT": "not-a-number"})
    with pytest.raises(ConfigError, match=f"{ENV_PREFIX}LOG_PROMPTS"):
        apply_env_overrides(BridgeConfig(), {f"{ENV_PREFIX}LOG_PROMPTS": "maybe"})


def test_extra_env_is_not_reachable_from_the_environment():
    # dict-typed field: no single-variable encoding, so it is deliberately skipped.
    cfg = apply_env_overrides(BridgeConfig(), {f"{ENV_PREFIX}EXTRA_ENV": "FOO=bar"})
    assert cfg.extra_env == {}


# --- CLI precedence -------------------------------------------------------------------


def test_cli_beats_env_beats_default(monkeypatch):
    monkeypatch.setenv(f"{ENV_PREFIX}PORT", "1111")
    assert resolve_config([]).port == 1111  # env beats default
    assert resolve_config(["--port", "2222"]).port == 2222  # CLI beats env
    monkeypatch.delenv(f"{ENV_PREFIX}PORT")
    assert resolve_config([]).port == 8000  # default


def test_no_dspark_flag_maps_to_the_field(monkeypatch):
    monkeypatch.delenv(f"{ENV_PREFIX}DSPARK", raising=False)
    assert resolve_config([]).dspark is True
    assert resolve_config(["--no-dspark"]).dspark is False
    # an env value is overridden only when the flag is actually passed
    monkeypatch.setenv(f"{ENV_PREFIX}DSPARK", "false")
    assert resolve_config([]).dspark is False


def test_legacy_bridge_api_key_still_works(monkeypatch):
    monkeypatch.delenv(f"{ENV_PREFIX}API_KEY", raising=False)
    monkeypatch.setenv("BRIDGE_API_KEY", "legacy")
    assert resolve_config([]).api_key == "legacy"


def test_env_api_key_wins_over_legacy(monkeypatch):
    monkeypatch.setenv(f"{ENV_PREFIX}API_KEY", "new")
    monkeypatch.setenv("BRIDGE_API_KEY", "legacy")
    assert resolve_config([]).api_key == "new"


def test_invalid_combination_exits_with_usage(capsys):
    with pytest.raises(SystemExit) as exc:
        resolve_config(["--cp", "3", "--ep", "8"])
    assert exc.value.code == 2
    assert "divide" in capsys.readouterr().err


def test_mock_forces_a_single_rank():
    cfg = resolve_config(["--mock", "--ep", "8"])
    assert (cfg.ep, cfg.cp, cfg.dp) == (1, 1, 1)
