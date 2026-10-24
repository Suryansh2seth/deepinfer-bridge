import json
import time

MODEL = "deepseek-v4.1-flash"


def chat(client, **extra):
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Hi"}], "thinking": {"type": "disabled"}, **extra}
    return client.post("/v1/chat/completions", json=body)


def test_health_and_models(client):
    assert client.get("/healthz").status_code == 200
    data = client.get("/v1/models").json()
    assert data["object"] == "list" and data["data"][0]["id"] == MODEL
    assert client.get("/models").json() == data  # Harness discovers via GET /models relative to base URL


def test_non_streaming(client):
    r = chat(client)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["choices"][0]["message"]["content"] == "Hello world!"
    assert j["choices"][0]["finish_reason"] == "stop"
    assert j["usage"]["prompt_tokens"] > 0 and j["usage"]["completion_tokens"] > 0


def test_streaming(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "stream": True,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "Hi"}],
        },
    ) as r:
        assert r.status_code == 200
        lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    text = "".join(
        (json.loads(ln[6:])["choices"][0]["delta"].get("content") or "")
        for ln in lines[:-1]
        if json.loads(ln[6:]).get("choices")
    )
    assert text == "Hello world!"


def test_length_finish(client):
    j = chat(client, max_tokens=2).json()
    assert j["choices"][0]["finish_reason"] == "length"


def test_thinking_mode_splits_reasoning(client):
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Hi"}], "thinking": {"type": "enabled"}}
    j = client.post("/v1/chat/completions", json=body).json()
    assert j["choices"][0]["message"]["content"].strip() == "Hello world!"


def test_unknown_model_is_a_clean_error(client):
    r = client.post("/v1/chat/completions", json={"model": "gpt-nope", "messages": [{"role": "user", "content": "x"}]})
    assert 400 <= r.status_code < 500


def test_prompt_too_long_is_rejected_before_inference(client):
    r = chat(client, messages=[{"role": "user", "content": "word " * 6000}])
    assert r.status_code == 400


def test_stop_sequence_aborts_engine_request(client):
    r = chat(client, stop=["Hello"])
    assert r.status_code == 200
    # the engine request is released (aborted if it was still running)
    deadline = time.time() + 5
    while time.time() < deadline and client.get("/stats").json()["outstanding"]:
        time.sleep(0.05)
    assert client.get("/stats").json()["outstanding"] == 0


def test_responses_endpoint(client):
    r = client.post("/v1/responses", json={"model": MODEL, "input": "Hi", "reasoning": {"effort": "none"}})
    assert r.status_code == 200, r.text


def test_no_leaked_requests(client):
    for _ in range(3):
        chat(client)
    assert client.get("/stats").json()["outstanding"] == 0
