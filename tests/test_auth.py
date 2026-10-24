def test_api_key_required(tokenizer_path):
    from starlette.testclient import TestClient

    from deepinfer_bridge.config import BridgeConfig
    from deepinfer_bridge.server import create_app

    cfg = BridgeConfig(mock=True, ep=1, dp=1, tokenizer_path=tokenizer_path, api_key="secret", startup_timeout_s=120)
    with TestClient(create_app(cfg)) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200
