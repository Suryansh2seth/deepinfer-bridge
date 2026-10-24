import json

DSML = (
    '<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]


def make_client(tokenizer_path, monkeypatch):
    from starlette.testclient import TestClient

    from deepinfer_bridge.config import BridgeConfig
    from deepinfer_bridge.server import create_app

    monkeypatch.setenv("BRIDGE_MOCK_REPLY", DSML)
    cfg = BridgeConfig(mock=True, ep=1, dp=1, tokenizer_path=tokenizer_path, startup_timeout_s=120)
    return TestClient(create_app(cfg))


def test_tool_call_is_parsed_by_recipe(tokenizer_path, monkeypatch):
    with make_client(tokenizer_path, monkeypatch) as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4.1-flash",
                "tools": TOOLS,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": "weather in Paris?"}],
            },
        )
        assert r.status_code == 200, r.text
        choice = r.json()["choices"][0]
        call = choice["message"]["tool_calls"][0]
        assert call["function"]["name"] == "get_weather"
        assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
        assert choice["finish_reason"] == "tool_calls"


def test_tool_call_streaming(tokenizer_path, monkeypatch):
    with make_client(tokenizer_path, monkeypatch) as c:
        with c.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4.1-flash",
                "tools": TOOLS,
                "stream": True,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": "weather in Paris?"}],
            },
        ) as r:
            frames = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: {")]
        name, args, finish = "", "", None
        for f in frames:
            for ch in f.get("choices", []):
                for tc in ch["delta"].get("tool_calls") or []:
                    name += tc["function"].get("name") or ""
                    args += tc["function"].get("arguments") or ""
                finish = ch.get("finish_reason") or finish
        assert name == "get_weather" and json.loads(args) == {"city": "Paris"} and finish == "tool_calls"
