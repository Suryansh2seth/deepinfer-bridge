# Security Policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Use GitHub's private vulnerability reporting instead:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Describe the issue, and include a reproduction if you can.

That channel is private to the maintainers until an advisory is published, so no contact
details need to be shared in public and nothing sensitive is exposed in an issue tracker.

Please include, where relevant:

- The affected version or commit.
- The configuration in use (flags / environment variables), with any real API key redacted.
- A minimal request or script that demonstrates the problem.
- The impact you believe it has.

## Supported versions

| Version | Supported |
|---|---|
| 0.2.x | Yes |
| 0.1.x | No — superseded by 0.2.0 |

## Scope

**In scope — the bridge's own code** (`deepinfer_bridge/`):

- Authentication bypass of the `--api-key` bearer check, or a timing side channel in it.
- Request smuggling, header injection, or a path that reaches the engine without admission
  control.
- A way to make the bridge bind beyond loopback without authentication (`--insecure` is an
  explicit opt-in, so using it is not a vulnerability).
- Denial of service that bypasses `--max-outstanding`, `--max-body-bytes` or
  `--request-timeout` — for example unbounded memory growth from a crafted request.
- Leakage of prompt or completion content into logs or `/metrics` when `--log-prompts` is
  not set.
- Any path that lets the HTTP process import `torch`/`deepinfer` and thereby initialise CUDA
  where it must not.

**Out of scope — reported upstream instead:**

- **DeepInfer itself** (<https://github.com/deepseek-ai/DeepInfer>). Kernel bugs, memory
  corruption in the engine, or a crash reachable only by a valid request are DeepInfer's, not
  the bridge's. The bridge does not vendor it.
- **deepseek-recipe** (<https://github.com/deepseek-ai/deepseek-recipe>). Request
  conversion, tokenizer and streaming decode live there.
- FastAPI / Starlette / uvicorn.

## Known limitations that are not vulnerabilities

These are documented behaviours rather than defects. Please do not report them as security
issues:

- **`--api-key` is a bearer check and nothing more.** There is no TLS, no rate limiting, no
  audit log, no per-user authorisation. The README says to bind to localhost or put the
  bridge behind a proxy. Anyone who can reach the port with the key has full access.
- **`/healthz` is unauthenticated by design**, so a supervisor can liveness-check a secured
  server. It exposes counters and per-rank load, not prompt content.
- **`/metrics` and `/stats` are authenticated** when a key is set. Scrape jobs need the
  credential.
- **`--insecure`** deliberately permits a non-loopback bind with no key. Passing it is the
  operator's decision.
- **No cross-replica balancing, no KV-cache service, no prefill/decode disaggregation.**
  Those are absent from DeepInfer's release and deliberately absent here.
- **`engine_real.py` has never been executed.** See `REPORT.md`. That is a correctness risk,
  not a vulnerability report — though a bug there could certainly have security consequences,
  and reports are welcome.

## Handling of secrets

- The bridge never logs prompt or completion text unless `--log-prompts` is passed.
- `.env` is gitignored; only `.env.example` with placeholder values is committed.
- An empty `--api-key` is a startup error rather than a silent downgrade to "no auth".
