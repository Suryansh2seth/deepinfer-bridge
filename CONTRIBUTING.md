# Contributing

Thanks for looking at `deepinfer-bridge`. This document covers the development environment,
how to run the checks, and the expectations for a change.

## Read this first

The project has an unusual constraint, and it shapes every contribution:

> **The HTTP/protocol layer is tested. The DeepInfer engine integration (`engine_real.py`)
> has never been executed**, because it needs an NVIDIA SM100 (compute capability 10.x) or
> Huawei Ascend 950 host, and none was available.

So there are two kinds of change:

1. **Protocol-layer changes** — anything in `server.py`, `protocol.py`, `config.py`,
   `metrics.py`, `logging_setup.py`, `engine_client.py`, `engine_mock.py`. These can and
   must be verified against the mock engine. Add tests.
2. **Engine-integration changes** — `engine_real.py`. These cannot be verified without the
   hardware. If you have that hardware, evidence (logs, `curl` output, token parity) is
   extremely welcome. If you do not, say so plainly in the pull request rather than
   implying it works.

Please do not describe something as working because it looks correct. Run it, or say you
could not.

## Development setup

The bridge needs Python >= 3.10. `deepseek-recipe` is the one dependency that can be
awkward: it is a Rust extension, and on platforms without a prebuilt wheel it must be
compiled.

```bash
git clone https://github.com/Suryansh2seth/deepinfer-bridge
cd deepinfer-bridge

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

If `pip install` has to build `deepseek-recipe` from source you additionally need
**Rust >= 1.85** (the crates use edition 2024), **OpenCV 4.x** and **libclang**:

```bash
# Debian/Ubuntu
sudo apt install -y build-essential pkg-config clang libclang-dev libopencv-dev
pip install --no-binary deepseek-recipe deepseek-recipe==0.2.0
```

### The tokenizer is required, and its absence is silent

`tests/conftest.py` **skips** rather than fails when the V4.1 tokenizer is missing. A run
with everything skipped is green and worthless. Fetch it and confirm it is used:

```bash
curl -fsSL -o /tmp/v41-tokenizer.json \
  https://raw.githubusercontent.com/deepseek-ai/deepseek-recipe/935817c66f5d6e64838f419692545f06787e485f/static/tokenizers/v41/tokenizer.json

export DEEPSEEK_RECIPE_TOKENIZER=/tmp/v41-tokenizer.json
pytest -q -rs              # expect "87 passed" and NO "skipped"
```

## Running the checks

All four must pass before a pull request is ready:

```bash
ruff check .               # lint
ruff format --check .      # formatting
pyright                    # types (0 errors expected)
pytest -q -rs              # tests
```

`engine_real.py` is excluded from pyright because it imports `deepinfer`/`torch`, which
only exist on supported GPU hosts. That exclusion is a deliberate, documented gap, not an
oversight.

## What a good change looks like

- **Every behaviour change needs a test.** The suite is mock-based, so almost everything in
  the protocol layer is testable without hardware.
- **Never weaken or delete a test to make it pass.** Fix the code. If a test is genuinely
  wrong, say why in the pull request.
- **Keep the HTTP process free of `torch` and `deepinfer`.** Importing either initialises
  CUDA in the wrong process and breaks per-rank `CUDA_VISIBLE_DEVICES`. There is a test
  guarding this; keep it passing.
- **No new heavy dependencies without justification.** `metrics.py` hand-rolls the
  Prometheus text format specifically to avoid adding `prometheus_client`.
- **Keep Python 3.10 compatibility** for the bridge. (`anyio`, not `asyncio.timeout`.)
- **Do not add features DeepInfer cannot support** (image input, `n>1`, `logprobs`,
  server-side tools). Return a clear 4xx instead.
- **Metrics against untested hardware are not acceptable.** Benchmarks belong in a pull
  request description only when they were actually measured, with the hardware and command
  shown.

## Documentation

If you change a flag, an endpoint, a metric or a failure mode, update the matching document:

| Document | Covers |
|---|---|
| `README.md` | overview, install, run, flags, limitations, troubleshooting |
| `docs/ARCHITECTURE.md` | component responsibilities, request sequence, lock-step |
| `docs/OPERATIONS.md` | monitoring, alerting, restart, upgrade |
| `docs/HARDWARE_BRINGUP.md` | the Phases 0-4 runbook for real hardware |
| `REPORT.md` | current status, the A1-A8 assumption audit, open issues |
| `CHANGELOG.md` | user-visible changes |

The status tables are the point of this project: they exist so a reader can tell immediately
what has been proven and what has not. Please keep them accurate.

## Commit messages

Conventional-commit style prefixes are used:

```
feat: add --max-outstanding Retry-After header
fix: compare API keys in constant time
docs: clarify the tokenizer skip trap
test: cover overload rejection on the mock engine
chore: prepare 0.2.0 for public release
```

Explain *why* in the body when the change is not self-evident.

## Reporting problems

- Bugs and feature requests: open an issue.
- Security issues: **do not** open a public issue — see [`SECURITY.md`](SECURITY.md).
