import os
from pathlib import Path

import pytest

TOKENIZER = os.environ.get(
    "DEEPSEEK_RECIPE_TOKENIZER",
    str(Path(__file__).resolve().parents[2] / "deepseek-recipe/static/tokenizers/v41/tokenizer.json"),
)


@pytest.fixture(scope="session")
def tokenizer_path():
    if not Path(TOKENIZER).is_file():
        pytest.skip("set DEEPSEEK_RECIPE_TOKENIZER to deepseek-recipe/static/tokenizers/v41/tokenizer.json")
    return TOKENIZER


@pytest.fixture()
def client(tokenizer_path, monkeypatch):
    from starlette.testclient import TestClient

    from deepinfer_bridge.config import BridgeConfig
    from deepinfer_bridge.server import create_app

    monkeypatch.setenv("BRIDGE_MOCK_DELAY_MS", "5")
    cfg = BridgeConfig(mock=True, ep=1, dp=1, tokenizer_path=tokenizer_path, startup_timeout_s=120, max_model_len=4096)
    with TestClient(create_app(cfg)) as c:
        yield c
