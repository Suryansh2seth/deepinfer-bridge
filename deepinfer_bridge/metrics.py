"""Minimal Prometheus-style metrics for the bridge. No third-party dependency.

Only the three shapes the bridge needs are implemented: monotonic counters, gauges and
fixed-bucket histograms, rendered in the Prometheus text exposition format (v0.0.4).
A dependency such as ``prometheus_client`` would work, but this keeps the serving process
tiny and removes a version-pinning risk from the release matrix.

Metrics are process-global and thread-safe: the event loop and the engine event-reader
thread both touch them.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from typing import TypeVar

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Latency buckets in seconds; shared by TTFT and whole-request duration.
DEFAULT_BUCKETS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)

Labels = tuple[tuple[str, str], ...]


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_labels(labels: Labels) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape(v)}"' for k, v in labels)
    return "{" + inner + "}"


def _render_value(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))


def _normalize(label_names: Sequence[str], labels: dict[str, str]) -> Labels:
    unknown = set(labels) - set(label_names)
    if unknown:
        raise ValueError(f"unknown label(s) {sorted(unknown)}; expected {list(label_names)}")
    return tuple((name, str(labels.get(name, ""))) for name in label_names)


class _Metric:
    kind = "untyped"

    def __init__(self, name: str, doc: str, label_names: Sequence[str] = ()) -> None:
        self.name = name
        self.doc = doc
        self.label_names = tuple(label_names)
        self._lock = threading.Lock()

    def _header(self) -> list[str]:
        return [f"# HELP {self.name} {self.doc}", f"# TYPE {self.name} {self.kind}"]

    def render(self) -> list[str]:
        raise NotImplementedError


_M = TypeVar("_M", bound=_Metric)


class Counter(_Metric):
    """Monotonically increasing total."""

    kind = "counter"

    def __init__(self, name: str, doc: str, label_names: Sequence[str] = ()) -> None:
        super().__init__(name, doc, label_names)
        self._values: dict[Labels, float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = _normalize(self.label_names, labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        with self._lock:
            return self._values.get(_normalize(self.label_names, labels), 0.0)

    def render(self) -> list[str]:
        with self._lock:
            items = sorted(self._values.items())
        lines = self._header()
        lines += [f"{self.name}{_render_labels(k)} {_render_value(v)}" for k, v in items]
        return lines


class Gauge(_Metric):
    """A value that can go up and down."""

    kind = "gauge"

    def __init__(self, name: str, doc: str, label_names: Sequence[str] = ()) -> None:
        super().__init__(name, doc, label_names)
        self._values: dict[Labels, float] = {}

    def set(self, value: float, **labels: str) -> None:
        key = _normalize(self.label_names, labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = _normalize(self.label_names, labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: str) -> None:
        self.inc(-amount, **labels)

    def value(self, **labels: str) -> float:
        with self._lock:
            return self._values.get(_normalize(self.label_names, labels), 0.0)

    def render(self) -> list[str]:
        with self._lock:
            items = sorted(self._values.items())
        lines = self._header()
        lines += [f"{self.name}{_render_labels(k)} {_render_value(v)}" for k, v in items]
        return lines


class Histogram(_Metric):
    """Cumulative histogram with fixed buckets, plus _sum and _count."""

    kind = "histogram"

    def __init__(
        self,
        name: str,
        doc: str,
        label_names: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> None:
        super().__init__(name, doc, label_names)
        self.buckets = tuple(sorted(buckets))
        self._counts: dict[Labels, list[int]] = {}
        self._sums: dict[Labels, float] = {}
        self._totals: dict[Labels, int] = {}

    def observe(self, value: float, **labels: str) -> None:
        key = _normalize(self.label_names, labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * len(self.buckets))
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    counts[i] += 1
            self._sums[key] = self._sums.get(key, 0.0) + float(value)
            self._totals[key] = self._totals.get(key, 0) + 1

    def render(self) -> list[str]:
        with self._lock:
            keys = sorted(set(self._counts) | set(self._totals))
            snapshot = {
                k: (list(self._counts.get(k, [])), self._sums.get(k, 0.0), self._totals.get(k, 0)) for k in keys
            }
        lines = self._header()
        for key, (counts, total_sum, total_count) in snapshot.items():
            for bound, count in zip(self.buckets, counts, strict=True):
                labelled = key + (("le", _render_value(bound)),)
                lines.append(f"{self.name}_bucket{_render_labels(labelled)} {count}")
            labelled = key + (("le", "+Inf"),)
            lines.append(f"{self.name}_bucket{_render_labels(labelled)} {total_count}")
            lines.append(f"{self.name}_sum{_render_labels(key)} {_render_value(total_sum)}")
            lines.append(f"{self.name}_count{_render_labels(key)} {total_count}")
        return lines


class Registry:
    """Holds every metric exposed on /metrics."""

    def __init__(self) -> None:
        self._metrics: list[_Metric] = []

    def register(self, metric: _M) -> _M:
        """Register and return the metric, preserving its concrete type."""
        self._metrics.append(metric)
        return metric

    def render(self) -> str:
        lines: list[str] = []
        for metric in self._metrics:
            lines += metric.render()
        return "\n".join(lines) + "\n"

    def __iter__(self) -> Iterator[_Metric]:
        return iter(self._metrics)


REGISTRY = Registry()

# --- request lifecycle ---------------------------------------------------------------
REQUESTS = REGISTRY.register(
    Counter(
        "deepinfer_bridge_requests_total",
        "Completed requests by endpoint, streaming mode and terminal status.",
        ("endpoint", "stream", "status"),
    )
)
IN_FLIGHT = REGISTRY.register(Gauge("deepinfer_bridge_in_flight", "Requests currently admitted to the engine."))
REJECTED = REGISTRY.register(
    Counter(
        "deepinfer_bridge_rejected_total",
        "Requests refused before reaching the engine, by reason.",
        ("reason",),
    )
)
ERRORS = REGISTRY.register(Counter("deepinfer_bridge_errors_total", "Errors raised while serving, by kind.", ("kind",)))
ABORTS = REGISTRY.register(
    Counter("deepinfer_bridge_aborts_total", "Engine requests aborted because the client went away or timed out.")
)

# --- tokens and timing ---------------------------------------------------------------
TOKENS = REGISTRY.register(Counter("deepinfer_bridge_tokens_total", "Tokens processed, by kind.", ("kind",)))
TTFT = REGISTRY.register(Histogram("deepinfer_bridge_ttft_seconds", "Time to first output token, in seconds."))
DURATION = REGISTRY.register(
    Histogram("deepinfer_bridge_request_duration_seconds", "Total request duration, in seconds.")
)

# --- engine state --------------------------------------------------------------------
ENGINE_HEALTHY = REGISTRY.register(Gauge("deepinfer_bridge_engine_healthy", "1 when every engine rank is ready."))
RANK_LOAD = REGISTRY.register(
    Gauge("deepinfer_bridge_rank_load", "Outstanding requests per data-parallel rank.", ("rank",))
)
