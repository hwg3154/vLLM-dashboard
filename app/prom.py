"""Minimal Prometheus text-exposition parser + histogram helpers.

No prometheus_client dependency: we only need to read, never to serve.
"""
from __future__ import annotations

import math
import re
from typing import Any

_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')

Sample = tuple[dict[str, str], float]


def _unescape(v: str) -> str:
    return v.replace(r"\\", "\\").replace(r"\"", '"').replace(r"\n", "\n")


def parse(text: str) -> dict[str, list[Sample]]:
    """Parse exposition text into {metric_name: [(labels, value), ...]}."""
    out: dict[str, list[Sample]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] == "#":
            continue
        if "{" in line:
            name, _, rest = line.partition("{")
            labels_s, _, tail = rest.rpartition("}")
            labels = {k: _unescape(v) for k, v in _LABEL_RE.findall(labels_s)}
            val_s = tail.strip().split(" ")[0]
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, val_s, labels = parts[0], parts[1], {}
        try:
            value = float(val_s)
        except ValueError:
            continue
        out.setdefault(name.strip(), []).append((labels, value))
    return out


def first(prom: dict[str, list[Sample]], *names: str, **match: str) -> float | None:
    """First value of the first metric name that exists, optionally label-filtered."""
    for name in names:
        for labels, value in prom.get(name, ()):
            if all(labels.get(k) == v for k, v in match.items()):
                return value
    return None


def total(prom: dict[str, list[Sample]], *names: str) -> float | None:
    """Sum across all label sets (e.g. multi-engine deployments)."""
    for name in names:
        samples = prom.get(name)
        if samples:
            return sum(v for _, v in samples)
    return None


def by_label(prom: dict[str, list[Sample]], name: str, key: str) -> dict[str, float]:
    """Group a metric family by one label, summing duplicates."""
    out: dict[str, float] = {}
    for labels, value in prom.get(name, ()):
        k = labels.get(key)
        if k is not None:
            out[k] = out.get(k, 0.0) + value
    return out


def label_of(prom: dict[str, list[Sample]], name: str, key: str) -> str | None:
    for labels, _ in prom.get(name, ()):
        if key in labels:
            return labels[key]
    return None


def info_labels(prom: dict[str, list[Sample]], name: str) -> dict[str, str]:
    """Labels of an *_info style gauge (value 1, everything encoded as labels)."""
    for labels, _ in prom.get(name, ()):
        return labels
    return {}


# --- histograms ---------------------------------------------------------

def snapshot_histogram(prom: dict[str, list[Sample]], base: str) -> dict[str, Any] | None:
    """Collect cumulative buckets/count/sum for one histogram family."""
    buckets: dict[float, float] = {}
    for labels, value in prom.get(base + "_bucket", ()):
        le = labels.get("le")
        if le is None:
            continue
        try:
            edge = float(le)
        except ValueError:
            continue
        buckets[edge] = buckets.get(edge, 0.0) + value
    if not buckets:
        return None
    return {
        "buckets": buckets,
        "count": total(prom, base + "_count") or 0.0,
        "sum": total(prom, base + "_sum") or 0.0,
    }


def diff_histogram(cur: dict[str, Any], base: dict[str, Any] | None) -> dict[str, Any]:
    """Subtract a baseline snapshot so quantiles cover the dashboard's session."""
    if not base:
        return cur
    buckets = {
        le: max(0.0, c - base["buckets"].get(le, 0.0)) for le, c in cur["buckets"].items()
    }
    return {
        "buckets": buckets,
        "count": max(0.0, cur["count"] - base["count"]),
        "sum": max(0.0, cur["sum"] - base["sum"]),
    }


def quantile(hist: dict[str, Any], q: float) -> float | None:
    """Linear-interpolated quantile from cumulative buckets.

    vLLM's *_sum/_count mean is badly skewed by outliers (a request that lands
    while the engine is asleep waits for the whole wake-up), so percentiles off
    the buckets are the honest number.
    """
    edges = sorted(hist["buckets"])
    if not edges:
        return None
    counts = [hist["buckets"][e] for e in edges]
    observed = counts[-1]
    if observed <= 0:
        return None
    target = q * observed
    prev_edge, prev_count = 0.0, 0.0
    for edge, count in zip(edges, counts):
        if count >= target:
            if math.isinf(edge):
                return prev_edge if prev_edge else None
            if count == prev_count:
                return edge
            span = edge - prev_edge
            return prev_edge + span * (target - prev_count) / (count - prev_count)
        prev_edge, prev_count = edge, count
    return prev_edge


def summarize(hist: dict[str, Any] | None) -> dict[str, Any] | None:
    if not hist or not hist["count"]:
        return None
    return {
        "n": hist["count"],
        "mean": hist["sum"] / hist["count"] if hist["count"] else None,
        "p50": quantile(hist, 0.50),
        "p90": quantile(hist, 0.90),
        "p99": quantile(hist, 0.99),
    }
