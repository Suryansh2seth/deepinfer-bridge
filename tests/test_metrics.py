"""Metrics primitives and the Prometheus text format they render.

Tests build their own metric objects rather than touching the module-level REGISTRY, so
they cannot be disturbed by (or disturb) other tests that exercise the live app.
"""

import pytest

from deepinfer_bridge.metrics import DEFAULT_BUCKETS, Counter, Gauge, Histogram, Registry


def parse(text: str) -> dict[str, float]:
    """Turn exposition text into {sample_name{labels}: value}."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        out[name] = float(value)
    return out


# --- counters -------------------------------------------------------------------------


def test_counter_accumulates_and_renders():
    c = Counter("test_total", "doc")
    c.inc()
    c.inc(2.5)
    assert c.value() == 3.5
    rendered = "\n".join(c.render())
    assert "# TYPE test_total counter" in rendered
    assert "# HELP test_total doc" in rendered
    assert parse(rendered)["test_total"] == 3.5


def test_counter_labels_are_independent():
    c = Counter("test_total", "doc", ("endpoint",))
    c.inc(endpoint="a")
    c.inc(3, endpoint="b")
    assert c.value(endpoint="a") == 1
    assert c.value(endpoint="b") == 3
    assert c.value(endpoint="never-seen") == 0


def test_counter_rejects_unknown_labels():
    c = Counter("test_total", "doc", ("endpoint",))
    with pytest.raises(ValueError, match="unknown label"):
        c.inc(bogus="x")


def test_declared_but_unset_labels_render_as_empty_strings():
    c = Counter("test_total", "doc", ("endpoint", "status"))
    c.inc(endpoint="chat")
    assert 'test_total{endpoint="chat",status=""} 1' in "\n".join(c.render())


def test_label_values_are_escaped():
    c = Counter("test_total", "doc", ("model",))
    c.inc(model='we"ird\\path\nnewline')
    line = next(line for line in c.render() if line.startswith("test_total{"))
    assert '\\"' in line and "\\\\" in line and "\\n" in line


# --- gauges ---------------------------------------------------------------------------


def test_gauge_set_inc_dec():
    g = Gauge("test_gauge", "doc")
    g.set(5)
    g.inc()
    g.dec(2)
    assert g.value() == 4
    assert "# TYPE test_gauge gauge" in "\n".join(g.render())


def test_gauge_dec_can_expose_a_leak_as_a_negative_value():
    # A negative in-flight gauge means finalize() ran more often than admit(); the metric
    # is deliberately not clamped so that bug is visible.
    g = Gauge("test_gauge", "doc")
    g.dec()
    assert g.value() == -1


def test_gauge_labels_are_independent():
    g = Gauge("test_gauge", "doc", ("rank",))
    g.set(1, rank="0")
    g.set(7, rank="1")
    rendered = parse("\n".join(g.render()))
    assert rendered['test_gauge{rank="0"}'] == 1
    assert rendered['test_gauge{rank="1"}'] == 7


# --- histograms -----------------------------------------------------------------------


def test_histogram_buckets_are_cumulative():
    h = Histogram("test_seconds", "doc", buckets=DEFAULT_BUCKETS)
    h.observe(0.03)
    h.observe(3.0)
    text = "\n".join(h.render())
    samples = parse(text)

    # 0.03 is <= every bound; 3.0 is <= only the bounds from 5.0 upwards.
    assert samples['test_seconds_bucket{le="0.05"}'] == 1
    assert samples['test_seconds_bucket{le="2.5"}'] == 1
    assert samples['test_seconds_bucket{le="5"}'] == 2
    assert samples['test_seconds_bucket{le="+Inf"}'] == 2
    assert samples["test_seconds_count"] == 2
    assert samples["test_seconds_sum"] == pytest.approx(3.03)
    assert "# TYPE test_seconds histogram" in text


def test_histogram_with_no_observations_renders_only_headers():
    h = Histogram("test_seconds", "doc")
    lines = h.render()
    assert lines == ["# HELP test_seconds doc", "# TYPE test_seconds histogram"]


def test_histogram_respects_labels():
    h = Histogram("test_seconds", "doc", ("endpoint",))
    h.observe(1.0, endpoint="chat")
    samples = parse("\n".join(h.render()))
    assert samples['test_seconds_count{endpoint="chat"}'] == 1


# --- registry -------------------------------------------------------------------------


def test_registry_renders_every_metric_and_ends_with_newline():
    reg = Registry()
    reg.register(Counter("a_total", "doc"))
    reg.register(Gauge("b_gauge", "doc"))
    reg.register(Histogram("c_seconds", "doc"))
    text = reg.render()
    assert text.endswith("\n")
    assert "# TYPE a_total counter" in text
    assert "# TYPE b_gauge gauge" in text
    assert "# TYPE c_seconds histogram" in text
    assert len(list(reg)) == 3
