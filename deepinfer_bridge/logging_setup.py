"""Logging configuration: one line per request, either key=value or JSON.

Request logs carry structured fields through ``extra={"event": {...}}``. Both formats
render them, so an operator gets readable lines by default and machine-parseable lines
with ``--log-format json``. Prompts and completions are never included unless
``--log-prompts`` is passed.
"""

from __future__ import annotations

import json
import logging

TEXT_FORMAT = "text"
JSON_FORMAT = "json"


def _event_of(record: logging.LogRecord) -> dict:
    event = getattr(record, "event", None)
    return event if isinstance(event, dict) else {}


class KeyValueFormatter(logging.Formatter):
    """`2026-01-01 00:00:00 INFO deepinfer_bridge: msg key=value ...`"""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        event = _event_of(record)
        if event:
            base += " " + " ".join(f"{k}={_flat(v)}" for k, v in event.items())
        return base


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the structured fields merged at the top level."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_event_of(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"), default=str)


def _flat(value: object) -> str:
    if isinstance(value, str):
        return value if value.isidentifier() or value.replace(".", "").isalnum() else repr(value)
    return str(value)


def configure_logging(level: str = "info", fmt: str = TEXT_FORMAT) -> None:
    """Install the bridge's root handler. Safe to call more than once."""
    formatter: logging.Formatter = JsonFormatter() if fmt == JSON_FORMAT else KeyValueFormatter()
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level.upper())
    # uvicorn installs its own handlers; make them inherit ours instead of duplicating.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
