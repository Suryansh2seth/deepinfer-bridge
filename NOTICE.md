# Third-party attribution

`deepinfer-bridge` is licensed under the MIT License (see [`LICENSE`](LICENSE)). It contains
code adapted from, and depends on, the following projects. Their notices are reproduced here
rather than appended to `LICENSE`, so that license-detection tooling reads the file as plain
MIT. The license terms themselves are unchanged.

## deepseek-recipe (MIT)

<https://github.com/deepseek-ai/deepseek-recipe>

- The HTTP/SSE response structure in `deepinfer_bridge/server.py` and the request preparation
  flow in `deepinfer_bridge/protocol.py` follow the `server-py` example from that repository.
- The `deepseek-recipe` package is a runtime dependency (`deepseek-recipe>=0.2,<0.3`), used
  for request conversion, V4.1 prompt encoding, streaming decode, tool-call and thinking
  parsing, and usage accounting.

Source revision read while developing this bridge:
`935817c66f5d6e64838f419692545f06787e485f`.

## DeepInfer (Apache License 2.0)

<https://github.com/deepseek-ai/DeepInfer>

- `deepinfer_bridge/engine_real.py` mirrors the engine construction and environment setup of
  DeepInfer's `examples/_config.py`, `examples/_launch.py` and
  `examples/offline_inference.py`.
- DeepInfer is **not** vendored here; it is imported at runtime on supported hardware.

Source revision read while developing this bridge:
`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`.

## Runtime dependencies

| Package | License |
|---|---|
| `fastapi` | MIT |
| `uvicorn` | BSD-3-Clause |
| `deepseek-recipe` | MIT |
| `pytest`, `httpx` (test extra) | MIT / BSD-3-Clause |
| `ruff` (dev extra) | MIT |
| `pyright` (dev extra) | MIT |

Dependency license texts are distributed with the packages themselves and are not vendored
into this repository.
