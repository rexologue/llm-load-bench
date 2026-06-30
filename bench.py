#!/usr/bin/env python3
"""
Simple OpenAI-compatible LLM serving benchmark for vLLM and SGLang.

Output schema per run:
{
  "meta": {...},
  "recorder": {...},
  "backend": {...}
}

The runner measures client-side streaming timings itself and also scrapes the
backend /metrics endpoint before/after each run. Backend metric names differ
between vLLM and SGLang, so this file maps them into the same logical fields.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import httpx
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: httpx. Install with: pip install -r requirements.txt") from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: PyYAML. Install with: pip install -r requirements.txt") from exc


###############################################################################
# Small utilities
###############################################################################


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return data


def write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = copy.deepcopy(base or {})
    if not override:
        return out
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")
    return base_url


def stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def flatten_payload_for_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Raw HTTP uses OpenAI-compatible JSON. extra_body is an SDK concept, so flatten it."""
    out = copy.deepcopy(payload or {})
    extra = out.pop("extra_body", None)
    if isinstance(extra, dict):
        out = deep_merge(out, extra)
    return out


###############################################################################
# Statistics
###############################################################################


def percentile(sorted_values: List[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def stat_block(values: Iterable[Any], round_digits: int = 6) -> Dict[str, Any]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    vals.sort()

    def r(x: Optional[float]) -> Optional[float]:
        return None if x is None else round(float(x), round_digits)

    return {
        "count": len(vals),
        "min": r(vals[0]),
        "max": r(vals[-1]),
        "mean": r(sum(vals) / len(vals)),
        "p50": r(percentile(vals, 0.50)),
        "p90": r(percentile(vals, 0.90)),
        "p95": r(percentile(vals, 0.95)),
        "p99": r(percentile(vals, 0.99)),
    }


###############################################################################
# Prometheus parsing and backend metric normalization
###############################################################################


PROM_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)(?:\s+\d+)?$"
)
LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_prom_value(raw: str) -> float:
    if raw == "+Inf" or raw == "Inf":
        return float("inf")
    if raw == "-Inf":
        return float("-inf")
    if raw == "NaN":
        return float("nan")
    return float(raw)


def parse_labels(raw: Optional[str]) -> Tuple[Tuple[str, str], ...]:
    if not raw:
        return tuple()
    labels = []
    for k, v in LABEL_RE.findall(raw):
        labels.append((k, bytes(v, "utf-8").decode("unicode_escape")))
    return tuple(sorted(labels))


def labels_to_dict(labels: Tuple[Tuple[str, str], ...]) -> Dict[str, str]:
    return dict(labels)


def label_filter_matches(labels: Tuple[Tuple[str, str], ...], label_filter: Dict[str, str]) -> bool:
    if not label_filter:
        return True
    d = labels_to_dict(labels)
    return all(str(d.get(k)) == str(v) for k, v in label_filter.items())


MODEL_LABEL_KEYS = ("model_name", "model", "served_model_name", "model_id")


def normalize_label_filter(label_filter: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not isinstance(label_filter, dict):
        return {}
    return {str(k): str(v) for k, v in label_filter.items() if v is not None}


def build_label_filter_candidates(label_filter: Dict[str, str], model: Optional[str] = None) -> List[Dict[str, str]]:
    """Build tolerant label filters.

    Priority:
    1. exact user filter
    2. same filter with model/model_name/served_model_name aliases
    3. model-only aliases
    4. no filter fallback

    This avoids silent metric-nulling when config uses model=... while backend
    exposes model_name=..., which is common for vLLM/SGLang.
    """
    requested = normalize_label_filter(label_filter)
    candidates: List[Dict[str, str]] = []
    seen: set[Tuple[Tuple[str, str], ...]] = set()

    def add(item: Optional[Dict[str, Any]]) -> None:
        normalized = normalize_label_filter(item)
        key = tuple(sorted(normalized.items()))
        if key not in seen:
            seen.add(key)
            candidates.append(normalized)

    add(requested)

    if requested:
        for src_key in MODEL_LABEL_KEYS:
            if src_key not in requested:
                continue
            value = requested[src_key]
            non_model_labels = {k: v for k, v in requested.items() if k not in MODEL_LABEL_KEYS}

            for dst_key in MODEL_LABEL_KEYS:
                aliased = dict(non_model_labels)
                aliased[dst_key] = value
                add(aliased)

            for dst_key in MODEL_LABEL_KEYS:
                add({dst_key: value})

    if model:
        for dst_key in MODEL_LABEL_KEYS:
            add({dst_key: str(model)})

    add({})
    return candidates


def metric_series_count(snapshot: Optional[PromSnapshot], name: str, label_filter: Dict[str, str]) -> int:
    if snapshot is None:
        return 0
    return sum(
        1
        for (metric, labels), _ in snapshot.series.items()
        if metric == name and label_filter_matches(labels, label_filter)
    )


def present_metric_candidates(snapshot: Optional[PromSnapshot], mapping: Dict[str, List[str]]) -> Dict[str, List[str]]:
    if snapshot is None:
        return {}
    out: Dict[str, List[str]] = {}
    for logical_name, names in mapping.items():
        present = [name for name in names if snapshot.has_metric(name)]
        if present:
            out[logical_name] = present
    return out


def collect_model_label_values(snapshot: Optional[PromSnapshot], max_items: int = 30) -> Dict[str, List[str]]:
    if snapshot is None:
        return {}
    values: Dict[str, set[str]] = defaultdict(set)
    for (_, labels), _ in snapshot.series.items():
        d = labels_to_dict(labels)
        for key in MODEL_LABEL_KEYS:
            if key in d:
                values[key].add(d[key])
    return {k: sorted(v)[:max_items] for k, v in values.items()}


def merge_label_filters(base: Dict[str, str], extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    out = dict(base or {})
    if not extra:
        return out
    for k, v in extra.items():
        k = str(k)
        v = str(v)
        if k in out and out[k] != v:
            return None
        out[k] = v
    return out


def sum_present(*values: Optional[float]) -> Optional[float]:
    present = [float(v) for v in values if v is not None]
    return sum(present) if present else None


class PromSnapshot:
    def __init__(self, text: str = "") -> None:
        self.series: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
        self.parse(text)

    def parse(self, text: str) -> None:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = PROM_LINE_RE.match(line)
            if not match:
                continue
            name, raw_labels, raw_value = match.groups()
            try:
                value = parse_prom_value(raw_value)
            except ValueError:
                continue
            if math.isnan(value):
                continue
            labels = parse_labels(raw_labels)
            self.series[(name, labels)] = value

    def has_metric(self, name: str) -> bool:
        return any(metric == name for metric, _ in self.series.keys())

    def metric_names(self) -> List[str]:
        return sorted({metric for metric, _ in self.series.keys()})

    def get_exact(self, name: str, labels: Tuple[Tuple[str, str], ...]) -> Optional[float]:
        return self.series.get((name, labels))

    def aggregate(self, name: str, label_filter: Dict[str, str], mode: str = "sum") -> Optional[float]:
        vals = [v for (metric, labels), v in self.series.items() if metric == name and label_filter_matches(labels, label_filter)]
        if not vals:
            return None
        if mode == "max":
            return max(vals)
        if mode == "mean":
            return sum(vals) / len(vals)
        return sum(vals)


def delta_value(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if after is None:
        return None
    if before is None:
        return after
    d = after - before
    if d < 0:
        return after
    return d


def counter_delta(
    before: Optional[PromSnapshot],
    after: Optional[PromSnapshot],
    candidates: List[str],
    label_filter: Dict[str, str],
    mode: str = "sum",
) -> Tuple[Optional[float], Optional[str]]:
    if after is None:
        return None, None
    for name in candidates:
        if not after.has_metric(name):
            continue
        a = after.aggregate(name, label_filter, mode=mode)
        b = before.aggregate(name, label_filter, mode=mode) if before is not None else None
        return delta_value(b, a), name
    return None, None


def gauge_after(
    after: Optional[PromSnapshot],
    candidates: List[str],
    label_filter: Dict[str, str],
    mode: str = "sum",
) -> Tuple[Optional[float], Optional[str]]:
    if after is None:
        return None, None
    for name in candidates:
        if after.has_metric(name):
            return after.aggregate(name, label_filter, mode=mode), name
    return None, None

def counter_delta_smart(
    before: Optional[PromSnapshot],
    after: Optional[PromSnapshot],
    candidates: List[str],
    label_filter: Dict[str, str],
    model: Optional[str] = None,
    mode: str = "sum",
) -> Tuple[Optional[float], Optional[str], Dict[str, str]]:
    if after is None:
        return None, None, {}

    for name in candidates:
        if not after.has_metric(name):
            continue
        for candidate_filter in build_label_filter_candidates(label_filter, model):
            if metric_series_count(after, name, candidate_filter) <= 0:
                continue
            a = after.aggregate(name, candidate_filter, mode=mode)
            b = before.aggregate(name, candidate_filter, mode=mode) if before is not None else None
            return delta_value(b, a), name, candidate_filter

    return None, None, {}


def gauge_after_smart(
    after: Optional[PromSnapshot],
    candidates: List[str],
    label_filter: Dict[str, str],
    model: Optional[str] = None,
    mode: str = "sum",
) -> Tuple[Optional[float], Optional[str], Dict[str, str]]:
    if after is None:
        return None, None, {}

    for name in candidates:
        if not after.has_metric(name):
            continue
        for candidate_filter in build_label_filter_candidates(label_filter, model):
            if metric_series_count(after, name, candidate_filter) <= 0:
                continue
            return after.aggregate(name, candidate_filter, mode=mode), name, candidate_filter

    return None, None, {}


def histogram_summary_smart(
    before: Optional[PromSnapshot],
    after: Optional[PromSnapshot],
    candidates: List[str],
    label_filter: Dict[str, str],
    model: Optional[str] = None,
    extra_label_filter: Optional[Dict[str, Any]] = None,
    round_digits: int = 6,
) -> Dict[str, Any]:
    empty = {
        "source_metric": None,
        "source_label_filter": {},
        "count": 0,
        "mean": None,
        "p50": None,
        "p90": None,
        "p95": None,
        "p99": None,
    }
    if after is None:
        return empty

    for base in candidates:
        bucket_name = f"{base}_bucket"
        if not after.has_metric(bucket_name):
            continue

        for candidate_filter in build_label_filter_candidates(label_filter, model):
            effective_filter = merge_label_filters(candidate_filter, extra_label_filter)
            if effective_filter is None:
                continue
            if metric_series_count(after, bucket_name, effective_filter) <= 0:
                continue

            result = histogram_summary(
                before=before,
                after=after,
                candidates=[base],
                label_filter=effective_filter,
                round_digits=round_digits,
            )
            result["source_label_filter"] = effective_filter
            return result

    return empty


def histogram_label_values(after: Optional[PromSnapshot], base: str, label_name: str) -> List[str]:
    if after is None:
        return []
    bucket_name = f"{base}_bucket"
    values: set[str] = set()
    for (metric, labels), _ in after.series.items():
        if metric != bucket_name:
            continue
        d = labels_to_dict(labels)
        if label_name in d:
            values.add(d[label_name])
    return sorted(values)


def histogram_summaries_by_label(
    before: Optional[PromSnapshot],
    after: Optional[PromSnapshot],
    candidates: List[str],
    label_filter: Dict[str, str],
    model: Optional[str] = None,
    label_name: str = "stage",
    allowed_values: Optional[List[str]] = None,
) -> Dict[str, Any]:
    empty = {
        "source_metric": None,
        "label": label_name,
        "by_label": {},
    }
    if after is None:
        return empty

    allowed = set(allowed_values) if allowed_values else None

    for base in candidates:
        bucket_name = f"{base}_bucket"
        if not after.has_metric(bucket_name):
            continue

        values = histogram_label_values(after, base, label_name)
        if allowed is not None:
            values = [v for v in values if v in allowed]

        by_label: Dict[str, Any] = {}
        for value in values:
            summary = histogram_summary_smart(
                before=before,
                after=after,
                candidates=[base],
                label_filter=label_filter,
                model=model,
                extra_label_filter={label_name: value},
            )
            if summary.get("source_metric") is not None:
                by_label[value] = summary

        if by_label:
            return {
                "source_metric": base,
                "label": label_name,
                "by_label": by_label,
            }

    return empty


def parse_le(labels: Tuple[Tuple[str, str], ...]) -> Optional[float]:
    d = labels_to_dict(labels)
    raw = d.get("le")
    if raw is None:
        return None
    if raw == "+Inf" or raw == "Inf":
        return float("inf")
    try:
        return float(raw)
    except ValueError:
        return None


def histogram_quantile_from_buckets(cumulative_buckets: Dict[float, float], q: float) -> Optional[float]:
    finite_items = sorted((le, c) for le, c in cumulative_buckets.items() if le is not None)
    if not finite_items:
        return None
    total = None
    for le, c in finite_items:
        if math.isinf(le):
            total = c
            break
    if total is None:
        total = finite_items[-1][1]
    if total <= 0:
        return None
    rank = q * total
    prev_le = 0.0
    prev_count = 0.0
    for le, count in finite_items:
        if count >= rank:
            if math.isinf(le):
                return prev_le
            bucket_count = count - prev_count
            if bucket_count <= 0:
                return le
            frac = (rank - prev_count) / bucket_count
            return prev_le + (le - prev_le) * frac
        if not math.isinf(le):
            prev_le = le
        prev_count = count
    return finite_items[-1][0]


def histogram_summary(
    before: Optional[PromSnapshot],
    after: Optional[PromSnapshot],
    candidates: List[str],
    label_filter: Dict[str, str],
    round_digits: int = 6,
) -> Dict[str, Any]:
    empty = {
        "source_metric": None,
        "count": 0,
        "mean": None,
        "p50": None,
        "p90": None,
        "p95": None,
        "p99": None,
    }
    if after is None:
        return empty

    for base in candidates:
        bucket_name = f"{base}_bucket"
        if not after.has_metric(bucket_name):
            continue

        buckets: Dict[float, float] = defaultdict(float)
        for (metric, labels), after_value in after.series.items():
            if metric != bucket_name or not label_filter_matches(labels, label_filter):
                continue
            le = parse_le(labels)
            if le is None:
                continue
            before_value = before.get_exact(metric, labels) if before is not None else None
            d = delta_value(before_value, after_value)
            if d is None:
                continue
            buckets[le] += d

        count_name = f"{base}_count"
        sum_name = f"{base}_sum"
        count_after = after.aggregate(count_name, label_filter, mode="sum")
        count_before = before.aggregate(count_name, label_filter, mode="sum") if before is not None else None
        total_count = delta_value(count_before, count_after)
        if total_count is None:
            total_count = buckets.get(float("inf"), 0.0)

        sum_after = after.aggregate(sum_name, label_filter, mode="sum")
        sum_before = before.aggregate(sum_name, label_filter, mode="sum") if before is not None else None
        total_sum = delta_value(sum_before, sum_after)

        if not buckets or not total_count or total_count <= 0:
            return {**empty, "source_metric": base}

        def r(x: Optional[float]) -> Optional[float]:
            return None if x is None else round(float(x), round_digits)

        return {
            "source_metric": base,
            "count": int(total_count),
            "mean": r(safe_div(total_sum, total_count)),
            "p50": r(histogram_quantile_from_buckets(buckets, 0.50)),
            "p90": r(histogram_quantile_from_buckets(buckets, 0.90)),
            "p95": r(histogram_quantile_from_buckets(buckets, 0.95)),
            "p99": r(histogram_quantile_from_buckets(buckets, 0.99)),
        }

    return empty


BACKEND_METRICS = {
    "vllm": {
        # Counters: current docs may expose names with or without _total depending
        # on version / compatibility layer. Keep both.
        "prompt_tokens_total": ["vllm:prompt_tokens_total", "vllm:prompt_tokens"],
        "generation_tokens_total": ["vllm:generation_tokens_total", "vllm:generation_tokens"],
        "request_success_total": [
            "vllm:request_success_total",
            "vllm:request_success",
            "vllm:num_requests_success_total",
        ],
        "preemptions_total": ["vllm:num_preemptions_total", "vllm:num_preemptions"],

        # Prefix cache. Preferred vLLM Prometheus value is hits_delta / queries_delta.
        "cache_hits_total": [
            "vllm:prefix_cache_hits_total",
            "vllm:prefix_cache_hits",
            "vllm:gpu_prefix_cache_hits_total",
            "vllm:gpu_prefix_cache_hits",
        ],
        "cache_queries_total": [
            "vllm:prefix_cache_queries_total",
            "vllm:prefix_cache_queries",
            "vllm:gpu_prefix_cache_queries_total",
            "vllm:gpu_prefix_cache_queries",
        ],
        "external_cache_hits_total": [
            "vllm:external_prefix_cache_hits_total",
            "vllm:external_prefix_cache_hits",
        ],
        "external_cache_queries_total": [
            "vllm:external_prefix_cache_queries_total",
            "vllm:external_prefix_cache_queries",
        ],
        "prompt_tokens_cached_total": [
            "vllm:prompt_tokens_cached_total",
            "vllm:prompt_tokens_cached",
        ],

        # Legacy/direct gauges. Keep as fallback/debug only.
        "cache_hit_rate": [
            "vllm:gpu_prefix_cache_hit_rate",
            "vllm:prefix_cache_hit_rate",
            "vllm:cache_hit_rate",
        ],
        "gpu_prefix_cache_hit_rate": ["vllm:gpu_prefix_cache_hit_rate"],
        "cpu_prefix_cache_hit_rate": ["vllm:cpu_prefix_cache_hit_rate"],

        "cache_usage": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
        "running": ["vllm:num_requests_running"],
        "waiting": ["vllm:num_requests_waiting"],
        "waiting_by_reason": ["vllm:num_requests_waiting_by_reason"],
        "swapped": ["vllm:num_requests_swapped"],

        # Latency histograms.
        "ttft": ["vllm:time_to_first_token_seconds"],
        "itl": ["vllm:inter_token_latency_seconds"],
        "tpot": [
            "vllm:request_time_per_output_token_seconds",
            "vllm:time_per_output_token_seconds",
        ],
        "e2e": ["vllm:e2e_request_latency_seconds"],
        "queue": ["vllm:request_queue_time_seconds", "vllm:time_in_queue_requests"],
        "prefill": ["vllm:request_prefill_time_seconds"],
        "decode": ["vllm:request_decode_time_seconds"],
        "inference": ["vllm:request_inference_time_seconds"],

        # Token/request histograms.
        "request_prompt_tokens": ["vllm:request_prompt_tokens"],
        "request_generation_tokens": ["vllm:request_generation_tokens"],
        "prefill_kv_computed_tokens": ["vllm:request_prefill_kv_computed_tokens"],
    },
    "sglang": {
        "prompt_tokens_total": ["sglang:prompt_tokens_total"],
        "generation_tokens_total": ["sglang:generation_tokens_total"],
        "cached_tokens_total": ["sglang:cached_tokens_total"],
        "realtime_tokens_total": ["sglang:realtime_tokens_total"],

        "request_success_total": ["sglang:num_requests_total"],
        "aborted_requests_total": ["sglang:num_aborted_requests_total"],

        # SGLang exposes cache hit rate directly, but we also compute a run-local
        # fallback from cached_tokens_total.
        "cache_hit_rate": ["sglang:cache_hit_rate"],
        "gpu_prefix_cache_hit_rate": [],
        "cpu_prefix_cache_hit_rate": [],

        "token_usage": ["sglang:token_usage"],
        "num_used_tokens": ["sglang:num_used_tokens"],
        "max_total_num_tokens": ["sglang:max_total_num_tokens"],
        "gen_throughput": ["sglang:gen_throughput"],
        "running": ["sglang:num_running_reqs"],
        "waiting": ["sglang:num_queue_reqs"],
        "swapped": ["sglang:num_retracted_reqs"],

        # Latency histograms.
        "ttft": ["sglang:time_to_first_token_seconds"],
        "itl": ["sglang:inter_token_latency_seconds"],
        "tpot": ["sglang:time_per_output_token_seconds"],
        "e2e": ["sglang:e2e_request_latency_seconds"],
        "queue": ["sglang:queue_time_seconds"],
        "per_stage": ["sglang:per_stage_req_latency_seconds"],

        # Request token histograms. Names differ across builds; keep fallbacks.
        "request_prompt_tokens": ["sglang:request_prompt_tokens", "sglang:prompt_tokens"],
        "request_generation_tokens": ["sglang:request_generation_tokens", "sglang:generation_tokens"],
    },
}


def infer_backend_type(snapshot: Optional[PromSnapshot], configured: str) -> str:
    configured = (configured or "auto").lower()
    if configured in {"vllm", "sglang"}:
        return configured
    names = snapshot.metric_names() if snapshot is not None else []
    if any(n.startswith("vllm:") for n in names):
        return "vllm"
    if any(n.startswith("sglang:") for n in names):
        return "sglang"
    return "auto"


def summarize_backend_metrics(
    before: Optional[PromSnapshot],
    after: Optional[PromSnapshot],
    backend_type: str,
    metrics_url: str,
    label_filter: Dict[str, str],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    actual_type = infer_backend_type(after, backend_type)
    mapping = BACKEND_METRICS.get(actual_type, {})

    scrape_info = {
        "ok": after is not None,
        "backend_type": actual_type,
        "metrics_url": metrics_url,
        "series_count": len(after.series) if after is not None else 0,
        "label_filter": label_filter,
        "label_filter_candidates": build_label_filter_candidates(label_filter, model),
        "model_label_values": collect_model_label_values(after),
        "present_candidate_metrics": present_metric_candidates(after, mapping),
    }

    if not mapping or after is None:
        return {
            "scrape": scrape_info,
            "tokens": {},
            "requests": {},
            "cache": {},
            "scheduler": {},
        }

    def c(key: str) -> Tuple[Optional[float], Optional[str], Dict[str, str]]:
        return counter_delta_smart(
            before=before,
            after=after,
            candidates=mapping.get(key, []),
            label_filter=label_filter,
            model=model,
            mode="sum",
        )

    def g(key: str, mode: str = "sum") -> Tuple[Optional[float], Optional[str], Dict[str, str]]:
        return gauge_after_smart(
            after=after,
            candidates=mapping.get(key, []),
            label_filter=label_filter,
            model=model,
            mode=mode,
        )

    def h(key: str) -> Dict[str, Any]:
        return histogram_summary_smart(
            before=before,
            after=after,
            candidates=mapping.get(key, []),
            label_filter=label_filter,
            model=model,
        )

    prompt_delta, prompt_metric, prompt_filter = c("prompt_tokens_total")
    gen_delta, gen_metric, gen_filter = c("generation_tokens_total")
    cached_delta, cached_metric, cached_filter = c("cached_tokens_total")
    prompt_cached_delta, prompt_cached_metric, prompt_cached_filter = c("prompt_tokens_cached_total")
    realtime_delta, realtime_metric, realtime_filter = c("realtime_tokens_total")

    success_delta, success_metric, success_filter = c("request_success_total")
    aborted_delta, aborted_metric, aborted_filter = c("aborted_requests_total")
    preempt_delta, preempt_metric, preempt_filter = c("preemptions_total")

    hit_rate_after, hit_rate_metric, hit_rate_filter = g("cache_hit_rate", mode="mean")
    gpu_hit_rate_after, gpu_hit_rate_metric, gpu_hit_rate_filter = g("gpu_prefix_cache_hit_rate", mode="mean")
    cpu_hit_rate_after, cpu_hit_rate_metric, cpu_hit_rate_filter = g("cpu_prefix_cache_hit_rate", mode="mean")

    hits_delta, hits_metric, hits_filter = c("cache_hits_total")
    queries_delta, queries_metric, queries_filter = c("cache_queries_total")
    external_hits_delta, external_hits_metric, external_hits_filter = c("external_cache_hits_total")
    external_queries_delta, external_queries_metric, external_queries_filter = c("external_cache_queries_total")

    computed_hit_rate = safe_div(hits_delta, queries_delta)
    external_hit_rate = safe_div(external_hits_delta, external_queries_delta)

    total_hits_delta = sum_present(hits_delta, external_hits_delta)
    total_queries_delta = sum_present(queries_delta, external_queries_delta)
    total_hit_rate_with_external = safe_div(total_hits_delta, total_queries_delta)

    # Fallback cache ratio from token counters.
    # For vLLM this usually comes from prompt_tokens_cached.
    # For SGLang this usually comes from cached_tokens_total.
    cache_token_delta = prompt_cached_delta if prompt_cached_delta is not None else cached_delta
    cached_over_prompt_delta = safe_div(cache_token_delta, prompt_delta)

    cached_over_input_like_delta = None
    if cache_token_delta is not None and prompt_delta is not None:
        denom = cache_token_delta + prompt_delta
        cached_over_input_like_delta = safe_div(cache_token_delta, denom)

    fallback_cached_rate = None
    fallback_cached_rate_source = None
    fallback_cached_rate_metric = None
    fallback_cached_rate_filter: Dict[str, str] = {}

    if cached_over_prompt_delta is not None and 0.0 <= cached_over_prompt_delta <= 1.0:
        fallback_cached_rate = cached_over_prompt_delta
        fallback_cached_rate_source = "cached_tokens_over_prompt_tokens_delta"
        fallback_cached_rate_metric = f"{prompt_cached_metric or cached_metric}/{prompt_metric}"
        fallback_cached_rate_filter = prompt_cached_filter or cached_filter or prompt_filter
    elif cached_over_input_like_delta is not None:
        fallback_cached_rate = cached_over_input_like_delta
        fallback_cached_rate_source = "cached_tokens_over_cached_plus_prompt_delta"
        fallback_cached_rate_metric = f"{prompt_cached_metric or cached_metric}/({prompt_cached_metric or cached_metric}+{prompt_metric})"
        fallback_cached_rate_filter = prompt_cached_filter or cached_filter or prompt_filter

    normalized_hit_rate = None
    normalized_hit_rate_source = None
    normalized_hit_rate_source_metric = None
    normalized_hit_rate_source_label_filter: Dict[str, str] = {}

    if actual_type == "vllm":
        # vLLM Prometheus path should prefer run-local counters.
        if computed_hit_rate is not None:
            normalized_hit_rate = computed_hit_rate
            normalized_hit_rate_source = "counter_delta"
            normalized_hit_rate_source_metric = f"{hits_metric}/{queries_metric}"
            normalized_hit_rate_source_label_filter = hits_filter or queries_filter
        elif total_hit_rate_with_external is not None:
            normalized_hit_rate = total_hit_rate_with_external
            normalized_hit_rate_source = "counter_delta_with_external"
            normalized_hit_rate_source_metric = f"({hits_metric}+{external_hits_metric})/({queries_metric}+{external_queries_metric})"
            normalized_hit_rate_source_label_filter = hits_filter or queries_filter or external_hits_filter or external_queries_filter
        elif hit_rate_after is not None:
            normalized_hit_rate = hit_rate_after
            normalized_hit_rate_source = "direct_gauge_legacy"
            normalized_hit_rate_source_metric = hit_rate_metric
            normalized_hit_rate_source_label_filter = hit_rate_filter
        elif fallback_cached_rate is not None:
            normalized_hit_rate = fallback_cached_rate
            normalized_hit_rate_source = fallback_cached_rate_source
            normalized_hit_rate_source_metric = fallback_cached_rate_metric
            normalized_hit_rate_source_label_filter = fallback_cached_rate_filter

    elif actual_type == "sglang":
        # SGLang exposes a direct gauge. If gauge is zero but cached token counter
        # moved, prefer the run-local computed fallback.
        direct_zero_but_cached_moved = (
            hit_rate_after is not None
            and hit_rate_after <= 0.0
            and cache_token_delta is not None
            and cache_token_delta > 0
            and fallback_cached_rate is not None
        )

        if hit_rate_after is not None and not direct_zero_but_cached_moved:
            normalized_hit_rate = hit_rate_after
            normalized_hit_rate_source = "direct_gauge"
            normalized_hit_rate_source_metric = hit_rate_metric
            normalized_hit_rate_source_label_filter = hit_rate_filter
        elif fallback_cached_rate is not None:
            normalized_hit_rate = fallback_cached_rate
            normalized_hit_rate_source = fallback_cached_rate_source
            normalized_hit_rate_source_metric = fallback_cached_rate_metric
            normalized_hit_rate_source_label_filter = fallback_cached_rate_filter

    else:
        if computed_hit_rate is not None:
            normalized_hit_rate = computed_hit_rate
            normalized_hit_rate_source = "counter_delta"
            normalized_hit_rate_source_metric = f"{hits_metric}/{queries_metric}"
            normalized_hit_rate_source_label_filter = hits_filter or queries_filter
        elif hit_rate_after is not None:
            normalized_hit_rate = hit_rate_after
            normalized_hit_rate_source = "direct_gauge"
            normalized_hit_rate_source_metric = hit_rate_metric
            normalized_hit_rate_source_label_filter = hit_rate_filter
        elif fallback_cached_rate is not None:
            normalized_hit_rate = fallback_cached_rate
            normalized_hit_rate_source = fallback_cached_rate_source
            normalized_hit_rate_source_metric = fallback_cached_rate_metric
            normalized_hit_rate_source_label_filter = fallback_cached_rate_filter

    cache_usage_after, cache_usage_metric, cache_usage_filter = g("cache_usage", mode="max")
    token_usage_after, token_usage_metric, token_usage_filter = g("token_usage", mode="max")
    used_tokens_after, used_tokens_metric, used_tokens_filter = g("num_used_tokens", mode="sum")
    max_total_tokens_after, max_total_tokens_metric, max_total_tokens_filter = g("max_total_num_tokens", mode="sum")
    gen_throughput_after, gen_throughput_metric, gen_throughput_filter = g("gen_throughput", mode="sum")

    running_after, running_metric, running_filter = g("running", mode="sum")
    waiting_after, waiting_metric, waiting_filter = g("waiting", mode="sum")
    swapped_after, swapped_metric, swapped_filter = g("swapped", mode="sum")

    stage_latency = histogram_summaries_by_label(
        before=before,
        after=after,
        candidates=mapping.get("per_stage", []),
        label_filter=label_filter,
        model=model,
        label_name="stage",
    )
    stage_by_label = stage_latency.get("by_label", {})
    prefill_stage = {k: v for k, v in stage_by_label.items() if str(k).startswith("prefill")}
    decode_stage = {k: v for k, v in stage_by_label.items() if str(k).startswith("decode")}

    out = {
        "scrape": scrape_info,
        "tokens": {
            "prompt_total_delta": prompt_delta,
            "prompt_total_source_metric": prompt_metric,
            "prompt_total_source_label_filter": prompt_filter,
            "generation_total_delta": gen_delta,
            "generation_total_source_metric": gen_metric,
            "generation_total_source_label_filter": gen_filter,
            "cached_total_delta": cached_delta,
            "cached_total_source_metric": cached_metric,
            "cached_total_source_label_filter": cached_filter,
            "prompt_cached_total_delta": prompt_cached_delta,
            "prompt_cached_total_source_metric": prompt_cached_metric,
            "prompt_cached_total_source_label_filter": prompt_cached_filter,
            "realtime_total_delta": realtime_delta,
            "realtime_total_source_metric": realtime_metric,
            "realtime_total_source_label_filter": realtime_filter,
            "total_delta": (prompt_delta or 0) + (gen_delta or 0) if prompt_delta is not None or gen_delta is not None else None,
        },
        "requests": {
            "success_total_delta": success_delta,
            "success_total_source_metric": success_metric,
            "success_total_source_label_filter": success_filter,
            "aborted_total_delta": aborted_delta,
            "aborted_total_source_metric": aborted_metric,
            "aborted_total_source_label_filter": aborted_filter,
        },
        "cache": {
            # Preferred normalized field. Existing dashboards should keep using this.
            "hit_rate": normalized_hit_rate,
            "hit_rate_source": normalized_hit_rate_source,
            "hit_rate_source_metric": normalized_hit_rate_source_metric,
            "hit_rate_source_label_filter": normalized_hit_rate_source_label_filter,

            # Direct gauges.
            "hit_rate_after": hit_rate_after,
            "hit_rate_after_source_metric": hit_rate_metric,
            "hit_rate_after_source_label_filter": hit_rate_filter,
            "gpu_prefix_hit_rate_after": gpu_hit_rate_after,
            "gpu_prefix_hit_rate_source_metric": gpu_hit_rate_metric,
            "gpu_prefix_hit_rate_source_label_filter": gpu_hit_rate_filter,
            "cpu_prefix_hit_rate_after": cpu_hit_rate_after,
            "cpu_prefix_hit_rate_source_metric": cpu_hit_rate_metric,
            "cpu_prefix_hit_rate_source_label_filter": cpu_hit_rate_filter,

            # Counter-window view. For vLLM this is the main run-local value.
            "hit_rate_from_counter_delta": computed_hit_rate,
            "hits_delta": hits_delta,
            "hits_source_metric": hits_metric,
            "hits_source_label_filter": hits_filter,
            "queries_delta": queries_delta,
            "queries_source_metric": queries_metric,
            "queries_source_label_filter": queries_filter,

            # External KV connector cache, if exposed.
            "external_hit_rate_from_counter_delta": external_hit_rate,
            "external_hits_delta": external_hits_delta,
            "external_hits_source_metric": external_hits_metric,
            "external_hits_source_label_filter": external_hits_filter,
            "external_queries_delta": external_queries_delta,
            "external_queries_source_metric": external_queries_metric,
            "external_queries_source_label_filter": external_queries_filter,
            "hit_rate_with_external_from_counter_delta": total_hit_rate_with_external,

            # Token-counter fallback/debug.
            "cache_token_delta": cache_token_delta,
            "cached_over_prompt_delta": cached_over_prompt_delta,
            "cached_over_input_like_delta": cached_over_input_like_delta,
            "fallback_cached_rate": fallback_cached_rate,
            "fallback_cached_rate_source": fallback_cached_rate_source,
            "fallback_cached_rate_source_metric": fallback_cached_rate_metric,
            "fallback_cached_rate_source_label_filter": fallback_cached_rate_filter,

            # Capacity usage.
            "usage_after": cache_usage_after,
            "usage_source_metric": cache_usage_metric,
            "usage_source_label_filter": cache_usage_filter,
            "token_usage_after": token_usage_after,
            "token_usage_source_metric": token_usage_metric,
            "token_usage_source_label_filter": token_usage_filter,
        },
        "scheduler": {
            "running_after": running_after,
            "running_source_metric": running_metric,
            "running_source_label_filter": running_filter,
            "waiting_after": waiting_after,
            "waiting_source_metric": waiting_metric,
            "waiting_source_label_filter": waiting_filter,
            "swapped_after": swapped_after,
            "swapped_source_metric": swapped_metric,
            "swapped_source_label_filter": swapped_filter,
            "used_tokens_after": used_tokens_after,
            "used_tokens_source_metric": used_tokens_metric,
            "used_tokens_source_label_filter": used_tokens_filter,
            "max_total_tokens_after": max_total_tokens_after,
            "max_total_tokens_source_metric": max_total_tokens_metric,
            "max_total_tokens_source_label_filter": max_total_tokens_filter,
            "preemptions_delta": preempt_delta,
            "preemptions_source_metric": preempt_metric,
            "preemptions_source_label_filter": preempt_filter,
            "gen_throughput_after": gen_throughput_after,
            "gen_throughput_source_metric": gen_throughput_metric,
            "gen_throughput_source_label_filter": gen_throughput_filter,
        },
        "ttft_sec": h("ttft"),
        "tpot_sec": h("tpot"),
        "itl_sec": h("itl"),
        "e2e_sec": h("e2e"),
        "queue_sec": h("queue"),
        "prefill_sec": h("prefill"),
        "decode_sec": h("decode"),
        "inference_sec": h("inference"),
        "prefill_kv_computed_tokens": h("prefill_kv_computed_tokens"),
        "stage_latency_sec": {
            "source_metric": stage_latency.get("source_metric"),
            "label": stage_latency.get("label", "stage"),
            "by_stage": stage_by_label,
        },
        "prefill_stage_sec": {
            "source_metric": stage_latency.get("source_metric"),
            "by_stage": prefill_stage,
        },
        "decode_stage_sec": {
            "source_metric": stage_latency.get("source_metric"),
            "by_stage": decode_stage,
        },
        "prompt_tokens": h("request_prompt_tokens"),
        "completion_tokens": h("request_generation_tokens"),
    }
    return out


async def scrape_metrics(client: httpx.AsyncClient, metrics_url: str, headers: Dict[str, str]) -> Optional[PromSnapshot]:
    try:
        resp = await client.get(metrics_url, headers=headers)
        resp.raise_for_status()
        return PromSnapshot(resp.text)
    except Exception:
        return None


###############################################################################
# Messages and token counting
###############################################################################


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def rough_render_chat(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = content_to_text(msg.get("content", ""))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def normalize_message(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    role = raw.get("role") or raw.get("speaker") or raw.get("from")
    content = raw.get("content", raw.get("text", raw.get("message", "")))
    if role in {"human", "customer", "client"}:
        role = "user"
    elif role in {"bot", "model", "ai", "agent"}:
        role = "assistant"
    if role not in {"system", "user", "assistant", "tool"}:
        return None
    return {"role": role, "content": content}


def extract_messages_from_json(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        raw_messages = obj
    elif isinstance(obj, dict):
        for key in ("messages", "conversation", "turns", "dialog", "dialogue"):
            if isinstance(obj.get(key), list):
                raw_messages = obj[key]
                break
        else:
            raise ValueError("JSON must be a list of messages or contain messages/conversation/turns/dialog/dialogue")
    else:
        raise ValueError("messages JSON must be a list or object")
    out = []
    for item in raw_messages:
        msg = normalize_message(item)
        if msg is not None:
            out.append(msg)
    return out


@dataclass
class Dialogue:
    path: Path
    messages: List[Dict[str, Any]]


class MessageRepository:
    def __init__(self, system_path: Optional[Path], messages_path: Path) -> None:
        self.system_message = self._load_system(system_path)
        self.dialogues = self._load_dialogues(messages_path)
        if not self.dialogues:
            raise ValueError(f"No messages loaded from {messages_path}")

    def _load_system(self, path: Optional[Path]) -> Optional[Dict[str, Any]]:
        if not path:
            return None
        if not path.exists():
            raise FileNotFoundError(f"system_path not found: {path}")
        return {"role": "system", "content": path.read_text(encoding="utf-8")}

    def _load_dialogues(self, path: Path) -> List[Dialogue]:
        files: List[Path]
        if path.is_dir():
            files = sorted(path.glob("*.json"))
        elif path.is_file():
            files = [path]
        else:
            raise FileNotFoundError(f"messages_dir/messages_path not found: {path}")

        dialogues: List[Dialogue] = []
        for file in files:
            try:
                obj = json.loads(file.read_text(encoding="utf-8"))
                messages = extract_messages_from_json(obj)
                # system is common and external; drop per-file system messages to avoid double system prompts.
                messages = [m for m in messages if m.get("role") != "system"]
                if messages:
                    dialogues.append(Dialogue(path=file, messages=messages))
            except Exception as exc:
                print(f"[warn] skip {file}: {exc}", file=sys.stderr)
        return dialogues

    def with_system(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.system_message is None:
            return list(messages)
        return [copy.deepcopy(self.system_message)] + copy.deepcopy(messages)


class TokenCounter:
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        headers: Dict[str, str],
        cfg: Dict[str, Any],
    ) -> None:
        self.client = client
        self.base_url = base_url
        self.model = model
        self.headers = headers
        self.enabled = bool(cfg.get("enabled", True))
        self.mode = str(cfg.get("mode", "endpoint"))
        self.add_generation_prompt = bool(cfg.get("add_generation_prompt", True))
        self.chat_template_kwargs = cfg.get("chat_template_kwargs")
        self.timeout_sec = float(cfg.get("timeout_sec", 60))
        self.strategy: Optional[str] = None
        self.cache: Dict[str, int] = {}
        self.stats = Counter()

    async def count(self, messages: List[Dict[str, Any]]) -> int:
        key = stable_hash(messages)
        if key in self.cache:
            self.stats["cache_hit"] += 1
            return self.cache[key]
        self.stats["cache_miss"] += 1

        if not self.enabled or self.mode == "rough":
            count = self.rough_count(messages)
            self.cache[key] = count
            return count

        try:
            count = await self.endpoint_count(messages)
            self.cache[key] = count
            return count
        except Exception as exc:
            self.stats["endpoint_fail"] += 1
            print(f"[warn] /tokenize failed, using rough count: {exc}", file=sys.stderr)
            count = self.rough_count(messages)
            self.cache[key] = count
            return count

    def rough_count(self, messages: List[Dict[str, Any]]) -> int:
        # Intentionally simple fallback: ~4 chars/token, plus tiny role/template overhead.
        text = rough_render_chat(messages)
        return max(1, int(len(text) / 4.0))

    async def endpoint_count(self, messages: List[Dict[str, Any]]) -> int:
        url = f"{self.base_url}/tokenize"
        rendered = rough_render_chat(messages)

        chat_payload = {
            "model": self.model,
            "messages": messages,
            "add_generation_prompt": self.add_generation_prompt,
            "return_token_strs": False,
        }
        if self.chat_template_kwargs is not None:
            chat_payload["chat_template_kwargs"] = self.chat_template_kwargs

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        if self.strategy == "chat_with_model":
            candidates.append(("chat_with_model", chat_payload))
        elif self.strategy == "chat_without_model":
            p = copy.deepcopy(chat_payload)
            p.pop("model", None)
            candidates.append(("chat_without_model", p))
        elif self.strategy == "prompt_with_model":
            candidates.append(("prompt_with_model", {"model": self.model, "prompt": rendered, "add_special_tokens": False}))
        elif self.strategy == "text_native":
            candidates.append(("text_native", {"text": rendered}))
        elif self.strategy == "prompt_native":
            candidates.append(("prompt_native", {"prompt": rendered}))
        else:
            candidates.extend(
                [
                    ("chat_with_model", chat_payload),
                    ("chat_without_model", {k: v for k, v in chat_payload.items() if k != "model"}),
                    ("prompt_with_model", {"model": self.model, "prompt": rendered, "add_special_tokens": False}),
                    ("text_native", {"text": rendered}),
                    ("prompt_native", {"prompt": rendered}),
                ]
            )

        last_error = None
        for name, payload in candidates:
            try:
                resp = await self.client.post(url, headers=self.headers, json=payload, timeout=self.timeout_sec)
                if resp.status_code >= 400:
                    last_error = f"{name}: HTTP {resp.status_code} {resp.text[:300]}"
                    continue
                data = resp.json()
                count = self.extract_token_count(data)
                if count is None:
                    last_error = f"{name}: cannot extract count from {str(data)[:300]}"
                    continue
                self.strategy = name
                return count
            except Exception as exc:
                last_error = f"{name}: {exc}"
        raise RuntimeError(last_error or "tokenize endpoint did not work")

    def extract_token_count(self, data: Any) -> Optional[int]:
        if not isinstance(data, dict):
            return None
        for key in ("count", "token_count", "num_tokens"):
            if isinstance(data.get(key), int):
                return int(data[key])
            if isinstance(data.get(key), float):
                return int(data[key])
        for key in ("tokens", "input_ids", "output_ids"):
            if isinstance(data.get(key), list):
                return len(data[key])
        meta = data.get("meta_info")
        if isinstance(meta, dict):
            for key in ("prompt_tokens", "input_tokens"):
                if isinstance(meta.get(key), (int, float)):
                    return int(meta[key])
        return None

    def meta(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "strategy": self.strategy,
            "cache_entries": len(self.cache),
            "cache_hit": int(self.stats.get("cache_hit", 0)),
            "cache_miss": int(self.stats.get("cache_miss", 0)),
            "endpoint_fail": int(self.stats.get("endpoint_fail", 0)),
        }


###############################################################################
# Test cases
###############################################################################


@dataclass
class RequestSpec:
    request_id: str
    run_name: str
    mode: str
    messages: List[Dict[str, Any]]
    dialogue_file: str
    dialogue_index: int
    message_count: int
    prompt_tokens_est: int
    target_tokens: Optional[int] = None
    target_reached: Optional[bool] = None
    turn_index: Optional[int] = None
    mix_component: Optional[str] = None
    mix_component_mode: Optional[str] = None
    payload_override: Optional[Dict[str, Any]] = None


def expand_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand list-valued profile axes.

    One expanded item is one benchmark profile and produces one JSONL row.
    Example:
      target_tokens=[7000, 10000], concurrency=[1, 5], total_requests=100
    becomes four profiles, each aggregated over its own 100 requests.
    """
    expanded: List[Dict[str, Any]] = []
    for raw in runs:
        concurrencies = raw.get("concurrency", 1)
        targets = raw.get("target_tokens", None)
        conc_list = concurrencies if isinstance(concurrencies, list) else [concurrencies]
        target_list = targets if isinstance(targets, list) else [targets]
        for c, t in itertools.product(conc_list, target_list):
            run = copy.deepcopy(raw)
            run["_base_name"] = str(raw.get("name", raw.get("mode", "run")))
            run["concurrency"] = int(c)
            if t is None:
                run.pop("target_tokens", None)
                suffix = f"c{c}"
            else:
                run["target_tokens"] = int(t)
                suffix = f"ctx{int(t)}_c{c}"
            base_name = run["_base_name"]
            run["name"] = f"{base_name}_{suffix}"
            run["_profile_key"] = {
                "base_name": base_name,
                "mode": run.get("mode", "fixed_context"),
                "target_tokens": run.get("target_tokens"),
                "concurrency": int(c),
                "total_requests": int(run.get("total_requests", 1)),
            }
            expanded.append(run)
    return expanded


def repeated_dialogue_prefix(
    dialogue_messages: List[Dict[str, Any]],
    message_count: int,
) -> List[Dict[str, Any]]:
    if message_count <= 0:
        return []

    if not dialogue_messages:
        raise ValueError("Cannot build fixed_context from an empty dialogue")

    n = len(dialogue_messages)
    return [copy.deepcopy(dialogue_messages[i % n]) for i in range(message_count)]


async def build_fixed_context_spec(
    repo: MessageRepository,
    counter: TokenCounter,
    run_name: str,
    dialogue_index: int,
    target_tokens: int,
) -> RequestSpec:
    dialogue = repo.dialogues[dialogue_index % len(repo.dialogues)]

    if not dialogue.messages:
        raise ValueError(f"Empty dialogue cannot be used for fixed_context: {dialogue.path}")

    def make_messages(dialogue_message_count: int) -> List[Dict[str, Any]]:
        repeated = repeated_dialogue_prefix(dialogue.messages, dialogue_message_count)
        return repo.with_system(repeated)

    # Find an upper bound that reaches target_tokens.
    # If the real dialogue is too short, we keep repeating it.
    hi = max(1, len(dialogue.messages))
    hi_messages = make_messages(hi)
    hi_tokens = await counter.count(hi_messages)

    while hi_tokens < target_tokens:
        hi *= 2
        hi_messages = make_messages(hi)
        hi_tokens = await counter.count(hi_messages)

    # Binary search the smallest repeated-prefix length that reaches target_tokens.
    lo = 0
    best_n = hi
    best_tokens = hi_tokens

    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = make_messages(mid)
        tokens = await counter.count(candidate)

        if tokens >= target_tokens:
            best_n = mid
            best_tokens = tokens
            hi = mid - 1
        else:
            lo = mid + 1

    messages = make_messages(best_n)

    return RequestSpec(
        request_id=str(uuid.uuid4()),
        run_name=run_name,
        mode="fixed_context",
        messages=messages,
        dialogue_file=str(dialogue.path),
        dialogue_index=dialogue_index % len(repo.dialogues),
        message_count=len(messages),
        prompt_tokens_est=best_tokens,
        target_tokens=target_tokens,
        target_reached=True,
    )


async def build_dialog_replay_pool(
    repo: MessageRepository,
    counter: TokenCounter,
    run_name: str,
    request_after_roles: List[str],
) -> List[RequestSpec]:
    pool: List[RequestSpec] = []
    allowed = set(request_after_roles or ["user"])
    for d_idx, dialogue in enumerate(repo.dialogues):
        for i in range(1, len(dialogue.messages) + 1):
            if dialogue.messages[i - 1].get("role") not in allowed:
                continue
            messages = repo.with_system(dialogue.messages[:i])
            tokens = await counter.count(messages)
            pool.append(
                RequestSpec(
                    request_id=str(uuid.uuid4()),
                    run_name=run_name,
                    mode="dialog_replay",
                    messages=messages,
                    dialogue_file=str(dialogue.path),
                    dialogue_index=d_idx,
                    message_count=len(messages),
                    prompt_tokens_est=tokens,
                    turn_index=i - 1,
                )
            )
    if not pool:
        raise ValueError("dialog_replay produced zero request points; check request_after_roles and messages roles")
    return pool


def allocate_weighted_counts(total: int, components: List[Dict[str, Any]]) -> List[int]:
    """Deterministically split total requests by component weights."""
    if total <= 0:
        return [0 for _ in components]
    weights = [float(c.get("weight", 1.0)) for c in components]
    if any(w < 0 for w in weights):
        raise ValueError("mix component weights must be non-negative")
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("mix requires at least one positive component weight")

    raw = [(total * w / weight_sum) for w in weights]
    counts = [int(math.floor(x)) for x in raw]
    remainder = total - sum(counts)
    fractions = sorted(((raw[i] - counts[i], i) for i in range(len(components))), reverse=True)
    for _, idx in fractions[:remainder]:
        counts[idx] += 1
    return counts


async def build_component_specs(
    repo: MessageRepository,
    counter: TokenCounter,
    run_name: str,
    component: Dict[str, Any],
    count: int,
    seed: int,
    dialogue_offset: int,
) -> List[RequestSpec]:
    """Build specs for one mix component.

    Supported component modes intentionally match top-level modes:
    - fixed_context
    - dialog_replay
    """
    mode = str(component.get("mode", "fixed_context"))
    component_name = str(component.get("name", mode))
    shuffle = bool(component.get("shuffle", False))
    specs: List[RequestSpec]

    if count <= 0:
        return []

    if mode == "fixed_context":
        if "target_tokens" not in component:
            raise ValueError(f"mix component {component_name}: fixed_context requires target_tokens")
        target = int(component["target_tokens"])
        specs = [
            await build_fixed_context_spec(repo, counter, run_name, dialogue_offset + i, target)
            for i in range(count)
        ]
    elif mode == "dialog_replay":
        pool = await build_dialog_replay_pool(
            repo,
            counter,
            run_name,
            list(component.get("request_after_roles", ["user"])),
        )
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(pool)
        specs = [copy.deepcopy(pool[i % len(pool)]) for i in range(count)]
        for spec in specs:
            spec.request_id = str(uuid.uuid4())
    else:
        raise ValueError(f"Unknown mix component mode: {mode}")

    payload_override = component.get("payload")
    for spec in specs:
        spec.mode = "mix"
        spec.mix_component = component_name
        spec.mix_component_mode = mode
        spec.payload_override = copy.deepcopy(payload_override) if isinstance(payload_override, dict) else None
    return specs


async def build_mix_specs(
    repo: MessageRepository,
    counter: TokenCounter,
    run: Dict[str, Any],
    seed: int,
) -> List[RequestSpec]:
    run_name = str(run.get("name", "mix"))
    total_requests = int(run.get("total_requests", 1))
    components = run.get("components") or run.get("mix")
    if not isinstance(components, list) or not components:
        raise ValueError(f"Run {run_name}: mix requires non-empty components list")

    counts = allocate_weighted_counts(total_requests, components)
    specs: List[RequestSpec] = []
    dialogue_offset = 0

    for idx, (component, count) in enumerate(zip(components, counts)):
        if not isinstance(component, dict):
            raise ValueError(f"Run {run_name}: mix component #{idx} must be an object")
        part = await build_component_specs(
            repo=repo,
            counter=counter,
            run_name=run_name,
            component=component,
            count=count,
            seed=seed + idx,
            dialogue_offset=dialogue_offset,
        )
        specs.extend(part)
        dialogue_offset += count

    if bool(run.get("shuffle", True)):
        rng = random.Random(seed)
        rng.shuffle(specs)

    # Make request ids unique after shuffling/copying.
    for spec in specs:
        spec.request_id = str(uuid.uuid4())
    return specs


async def build_specs(
    repo: MessageRepository,
    counter: TokenCounter,
    run: Dict[str, Any],
    seed: int,
) -> List[RequestSpec]:
    mode = str(run.get("mode", "fixed_context"))
    run_name = str(run.get("name", mode))
    total_requests = int(run.get("total_requests", 1))
    shuffle = bool(run.get("shuffle", False))

    if mode == "fixed_context":
        if "target_tokens" not in run:
            raise ValueError(f"Run {run_name}: fixed_context requires target_tokens")
        target = int(run["target_tokens"])
        specs = [await build_fixed_context_spec(repo, counter, run_name, i, target) for i in range(total_requests)]
    elif mode == "dialog_replay":
        pool = await build_dialog_replay_pool(repo, counter, run_name, list(run.get("request_after_roles", ["user"])))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(pool)
        specs = [copy.deepcopy(pool[i % len(pool)]) for i in range(total_requests)]
        for spec in specs:
            spec.request_id = str(uuid.uuid4())
    elif mode == "mix":
        specs = await build_mix_specs(repo, counter, run, seed)
    else:
        raise ValueError(f"Unknown run mode: {mode}")

    if shuffle and mode != "dialog_replay":
        rng = random.Random(seed)
        rng.shuffle(specs)
    return specs


###############################################################################
# Streaming request recorder
###############################################################################


@dataclass
class RequestResult:
    request_id: str
    ok: bool
    error: Optional[str]
    status_code: Optional[int]
    ttft_sec: Optional[float]
    tpot_sec: Optional[float]
    e2e_sec: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    cached_prompt_tokens: Optional[int]
    observed_chunks: int
    itl_samples: List[float]
    dialogue_file: str
    prompt_tokens_est: int
    target_tokens: Optional[int]
    target_reached: Optional[bool]
    message_count: int
    mix_component: Optional[str] = None
    mix_component_mode: Optional[str] = None
    usage_present: bool = False
    completion_from_usage: bool = False


def extract_usage(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    usage = data.get("usage")
    return usage if isinstance(usage, dict) else None


def extract_cached_tokens(usage: Optional[Dict[str, Any]]) -> Optional[int]:
    if not usage:
        return None
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
    if isinstance(details, dict):
        for key in ("cached_tokens", "cache_read_input_tokens"):
            if isinstance(details.get(key), (int, float)):
                return int(details[key])
    return None


def extract_delta_payload(data: Dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return ""
    delta = choice0.get("delta") or choice0.get("message") or {}
    if not isinstance(delta, dict):
        return ""
    parts: List[str] = []
    for key in ("content", "reasoning_content"):
        val = delta.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
    tool_calls = delta.get("tool_calls")
    if tool_calls:
        parts.append(json.dumps(tool_calls, ensure_ascii=False))
    return "".join(parts)


async def send_chat_request(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    model: str,
    payload_base: Dict[str, Any],
    spec: RequestSpec,
    timeout_sec: float,
    max_itl_samples: int,
) -> RequestResult:
    payload = flatten_payload_for_request(payload_base)
    if spec.payload_override:
        payload = deep_merge(payload, spec.payload_override)
        payload = flatten_payload_for_request(payload)
    payload["model"] = model
    payload["messages"] = spec.messages
    payload.setdefault("stream", True)

    start = time.perf_counter()
    first_token_at: Optional[float] = None
    last_token_at: Optional[float] = None
    prev_token_at: Optional[float] = None
    end = None
    observed_chunks = 0
    itl_samples: List[float] = []
    usage: Optional[Dict[str, Any]] = None

    try:
        async with client.stream("POST", url, headers=headers, json=payload, timeout=timeout_sec) as resp:
            status = resp.status_code
            if status >= 400:
                text = await resp.aread()
                end = time.perf_counter()
                return RequestResult(
                    request_id=spec.request_id,
                    ok=False,
                    error=f"HTTP {status}: {text[:500].decode('utf-8', errors='replace')}",
                    status_code=status,
                    ttft_sec=None,
                    tpot_sec=None,
                    e2e_sec=end - start,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    cached_prompt_tokens=None,
                    observed_chunks=0,
                    itl_samples=[],
                    dialogue_file=spec.dialogue_file,
                    prompt_tokens_est=spec.prompt_tokens_est,
                    target_tokens=spec.target_tokens,
                    target_reached=spec.target_reached,
                    message_count=spec.message_count,
                    mix_component=spec.mix_component,
                    mix_component_mode=spec.mix_component_mode,
                )

            async for line in resp.aiter_lines():
                if not line:
                    continue
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                maybe_usage = extract_usage(data)
                if maybe_usage is not None:
                    usage = maybe_usage
                delta_text = extract_delta_payload(data)
                if delta_text:
                    now = time.perf_counter()
                    observed_chunks += 1
                    if first_token_at is None:
                        first_token_at = now
                    if prev_token_at is not None and len(itl_samples) < max_itl_samples:
                        itl_samples.append(now - prev_token_at)
                    prev_token_at = now
                    last_token_at = now
        end = time.perf_counter()

        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        if usage:
            for src, dst in (("prompt_tokens", "prompt"), ("completion_tokens", "completion"), ("total_tokens", "total")):
                val = usage.get(src)
                if isinstance(val, (int, float)):
                    if dst == "prompt":
                        prompt_tokens = int(val)
                    elif dst == "completion":
                        completion_tokens = int(val)
                    else:
                        total_tokens = int(val)

        # Track provenance so the summary can flag when token-derived metrics
        # (TPOT, output tok/s) are approximate rather than from real usage.
        usage_present = usage is not None
        completion_from_usage = completion_tokens is not None

        if completion_tokens is None:
            completion_tokens = observed_chunks if observed_chunks > 0 else None
        if prompt_tokens is None:
            prompt_tokens = spec.prompt_tokens_est
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        ttft = first_token_at - start if first_token_at is not None else None
        e2e = end - start
        tpot = None
        if first_token_at is not None and last_token_at is not None and completion_tokens and completion_tokens > 1:
            tpot = (last_token_at - first_token_at) / max(1, completion_tokens - 1)
        elif itl_samples:
            tpot = sum(itl_samples) / len(itl_samples)

        return RequestResult(
            request_id=spec.request_id,
            ok=True,
            error=None,
            status_code=status,
            ttft_sec=ttft,
            tpot_sec=tpot,
            e2e_sec=e2e,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_prompt_tokens=extract_cached_tokens(usage),
            observed_chunks=observed_chunks,
            itl_samples=itl_samples,
            dialogue_file=spec.dialogue_file,
            prompt_tokens_est=spec.prompt_tokens_est,
            target_tokens=spec.target_tokens,
            target_reached=spec.target_reached,
            message_count=spec.message_count,
            mix_component=spec.mix_component,
            mix_component_mode=spec.mix_component_mode,
            usage_present=usage_present,
            completion_from_usage=completion_from_usage,
        )
    except Exception as exc:
        end = time.perf_counter()
        return RequestResult(
            request_id=spec.request_id,
            ok=False,
            error=str(exc),
            status_code=None,
            ttft_sec=None,
            tpot_sec=None,
            e2e_sec=end - start,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            cached_prompt_tokens=None,
            observed_chunks=observed_chunks,
            itl_samples=itl_samples,
            dialogue_file=spec.dialogue_file,
            prompt_tokens_est=spec.prompt_tokens_est,
            target_tokens=spec.target_tokens,
            target_reached=spec.target_reached,
            message_count=spec.message_count,
            mix_component=spec.mix_component,
            mix_component_mode=spec.mix_component_mode,
        )


async def run_requests(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    model: str,
    payload: Dict[str, Any],
    specs: List[RequestSpec],
    concurrency: int,
    timeout_sec: float,
    progress_every: int,
    max_itl_samples: int,
) -> List[RequestResult]:
    queue: asyncio.Queue[RequestSpec] = asyncio.Queue()
    for spec in specs:
        queue.put_nowait(spec)
    results: List[RequestResult] = []
    lock = asyncio.Lock()
    done_count = 0

    async def worker(worker_id: int) -> None:
        nonlocal done_count
        while True:
            try:
                spec = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            result = await send_chat_request(client, url, headers, model, payload, spec, timeout_sec, max_itl_samples)
            async with lock:
                results.append(result)
                done_count += 1
                if progress_every and done_count % progress_every == 0:
                    ok_count = sum(1 for r in results if r.ok)
                    print(f"  progress {done_count}/{len(specs)} ok={ok_count} errors={done_count-ok_count}")
            queue.task_done()

    workers = [asyncio.create_task(worker(i)) for i in range(max(1, concurrency))]
    await asyncio.gather(*workers)
    return results


###############################################################################
# Summary
###############################################################################


def summarize_recorder(results: List[RequestResult], duration_sec: float) -> Dict[str, Any]:
    ok = [r for r in results if r.ok]
    errors = [r for r in results if not r.ok]
    total_prompt = sum(r.prompt_tokens or 0 for r in ok)
    total_completion = sum(r.completion_tokens or 0 for r in ok)
    total_tokens = sum(r.total_tokens or 0 for r in ok)

    error_counter = Counter((r.error or "unknown")[:240] for r in errors)
    status_counter = Counter(str(r.status_code) for r in results)

    all_itl = []
    for r in ok:
        all_itl.extend(r.itl_samples)

    target_reached = [r.target_reached for r in results if r.target_reached is not None]
    component_counter = Counter((r.mix_component or "default") for r in results)
    component_mode_counter = Counter((r.mix_component_mode or "default") for r in results)

    # Token-accounting transparency: when usage is absent, completion_tokens (and
    # therefore TPOT and completion_tokens_per_sec) fall back to chunk counts and
    # are only approximate. Surface how often that happened.
    usage_present_count = sum(1 for r in ok if r.usage_present)
    completion_from_usage_count = sum(1 for r in ok if r.completion_from_usage)

    return {
        "components": {
            "by_name": dict(component_counter),
            "by_mode": dict(component_mode_counter),
        },
        "requests": {
            "total": len(results),
            "success": len(ok),
            "error": len(errors),
            "success_rate": safe_div(len(ok), len(results)),
            "status_codes": dict(status_counter),
            "target_reached_count": sum(1 for x in target_reached if x),
            "target_not_reached_count": sum(1 for x in target_reached if x is False),
        },
        "throughput": {
            "duration_sec": round(duration_sec, 6),
            "requests_per_sec": round(safe_div(len(ok), duration_sec) or 0, 6),
            "errors_per_sec": round(safe_div(len(errors), duration_sec) or 0, 6),
            "prompt_tokens_per_sec": round(safe_div(total_prompt, duration_sec) or 0, 6),
            "completion_tokens_per_sec": round(safe_div(total_completion, duration_sec) or 0, 6),
            "total_tokens_per_sec": round(safe_div(total_tokens, duration_sec) or 0, 6),
            "note": "throughput counts only successful requests over wall-clock duration (goodput)",
        },
        "token_accounting": {
            "usage_present_rate": safe_div(usage_present_count, len(ok)),
            "completion_from_usage_rate": safe_div(completion_from_usage_count, len(ok)),
            "note": "if completion_from_usage_rate < 1.0, some completion_tokens (hence TPOT and output tok/s) fall back to chunk counts and are approximate",
        },
        "ttft_sec": stat_block(r.ttft_sec for r in ok),
        "tpot_sec": stat_block(r.tpot_sec for r in ok),
        "itl_sec": stat_block(all_itl),
        "e2e_sec": stat_block(r.e2e_sec for r in ok),
        "prompt_tokens": stat_block(r.prompt_tokens for r in ok),
        "completion_tokens": stat_block(r.completion_tokens for r in ok),
        "total_tokens": stat_block(r.total_tokens for r in ok),
        "cached_prompt_tokens": stat_block(r.cached_prompt_tokens for r in ok),
        "observed_chunks": stat_block(r.observed_chunks for r in ok),
        "message_count": stat_block(r.message_count for r in ok),
        "errors": {
            "top": [{"error": k, "count": v} for k, v in error_counter.most_common(10)]
        },
    }


def public_payload_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = flatten_payload_for_request(payload)
    out.pop("messages", None)
    return out


def clean_component_meta(components: Any) -> List[Dict[str, Any]]:
    if not isinstance(components, list):
        return []
    out: List[Dict[str, Any]] = []
    for idx, comp in enumerate(components):
        if not isinstance(comp, dict):
            continue
        item = {
            "name": str(comp.get("name", f"component_{idx}")),
            "mode": str(comp.get("mode", "fixed_context")),
            "weight": float(comp.get("weight", 1.0)),
            "target_tokens": comp.get("target_tokens"),
            "request_after_roles": comp.get("request_after_roles"),
            "payload": public_payload_meta(comp.get("payload") or {}),
        }
        out.append(item)
    return out


def profile_meta(run: Dict[str, Any], run_name: str, mode: str, concurrency: int, total_requests: int) -> Dict[str, Any]:
    components = run.get("components") or run.get("mix")
    profile = {
        "name": run_name,
        "base_name": run.get("_base_name", run_name),
        "key": run.get("_profile_key"),
        "mode": mode,
        "target_tokens": run.get("target_tokens"),
        "concurrency": concurrency,
        "total_requests": total_requests,
        "is_mixed": mode == "mix",
    }
    if mode == "mix":
        profile["components"] = clean_component_meta(components)
        profile["component_counts_expected"] = dict(
            zip(
                [c.get("name", f"component_{i}") for i, c in enumerate(components or []) if isinstance(c, dict)],
                allocate_weighted_counts(total_requests, [c for c in (components or []) if isinstance(c, dict)]),
            )
        )
    return profile


###############################################################################
# Main runner
###############################################################################


async def run_benchmark(config: Dict[str, Any], config_path: Path) -> None:
    base_url = normalize_base_url(str(config.get("base_url") or config.get("vllm_base_url") or "http://127.0.0.1:8000"))
    model = str(config.get("model"))
    if not model:
        raise ValueError("Config requires model")

    backend_type = str(config.get("backend_type", "auto")).lower()
    api_key = config.get("api_key")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    messages_path = Path(config.get("messages_dir") or config.get("messages_path") or "./messages")
    system_path_raw = config.get("system_path")
    system_path = Path(system_path_raw) if system_path_raw else None

    if not messages_path.is_absolute():
        messages_path = (config_path.parent / messages_path).resolve()
    if system_path is not None and not system_path.is_absolute():
        system_path = (config_path.parent / system_path).resolve()

    output_jsonl = Path(config.get("output_jsonl") or config.get("output_file") or "./bench_results.jsonl")
    if not output_jsonl.is_absolute():
        output_jsonl = (config_path.parent / output_jsonl).resolve()
    if output_jsonl.exists() and bool(config.get("overwrite_output", False)):
        output_jsonl.unlink()

    request_timeout_sec = float(config.get("request_timeout_sec", 300))
    progress_every = int(config.get("progress_every", 20))
    max_itl_samples = int(config.get("max_itl_samples_in_memory", 1_000_000))
    seed = int(config.get("seed", 12345))

    metrics_cfg = config.get("metrics") or {}
    metrics_enabled = bool(metrics_cfg.get("enabled", config.get("metrics_enabled", True)))
    metrics_url = str(metrics_cfg.get("url") or config.get("metrics_url") or f"{base_url}/metrics")
    label_filter = metrics_cfg.get("label_filter") or {}
    if not isinstance(label_filter, dict):
        label_filter = {}

    payload_global = config.get("payload") or {}
    runs = expand_runs(config.get("runs") or [])
    if not runs:
        raise ValueError("Config requires at least one run")

    connection_limit = int(config.get("http_connection_limit", 0))
    limits = httpx.Limits(max_connections=None if connection_limit <= 0 else connection_limit, max_keepalive_connections=None if connection_limit <= 0 else connection_limit)

    async with httpx.AsyncClient(limits=limits, timeout=request_timeout_sec) as client:
        repo = MessageRepository(system_path, messages_path)
        tokenizer_cfg = config.get("tokenizer") or {}
        if "chat_template_kwargs" not in tokenizer_cfg:
            flattened_payload = flatten_payload_for_request(payload_global)
            if "chat_template_kwargs" in flattened_payload:
                tokenizer_cfg = deep_merge(tokenizer_cfg, {"chat_template_kwargs": flattened_payload["chat_template_kwargs"]})
        counter = TokenCounter(client, base_url, model, headers, tokenizer_cfg)

        print(f"Loaded dialogues: {len(repo.dialogues)} from {messages_path}")
        print(f"Runs expanded: {len(runs)}")
        print(f"Output: {output_jsonl}")

        for idx, run in enumerate(runs, start=1):
            run_name = str(run.get("name", f"run_{idx}"))
            mode = str(run.get("mode", "fixed_context"))
            concurrency = int(run.get("concurrency", 1))
            total_requests = int(run.get("total_requests", 1))
            payload = deep_merge(payload_global, run.get("payload") or {})
            payload = flatten_payload_for_request(payload)
            payload.setdefault("stream", True)

            print(f"\n[{idx}/{len(runs)}] {run_name}: mode={mode} concurrency={concurrency} total_requests={total_requests}")
            specs = await build_specs(repo, counter, run, seed + idx)
            if len(specs) != total_requests:
                print(f"  built specs: {len(specs)}")

            before = await scrape_metrics(client, metrics_url, headers) if metrics_enabled else None
            started_at = utc_now()
            t0 = time.perf_counter()
            results = await run_requests(
                client=client,
                url=f"{base_url}/v1/chat/completions",
                headers=headers,
                model=model,
                payload=payload,
                specs=specs,
                concurrency=concurrency,
                timeout_sec=request_timeout_sec,
                progress_every=progress_every,
                max_itl_samples=max_itl_samples,
            )
            duration_sec = time.perf_counter() - t0
            ended_at = utc_now()
            after = await scrape_metrics(client, metrics_url, headers) if metrics_enabled else None

            recorder = summarize_recorder(results, duration_sec)
            backend = summarize_backend_metrics(before, after, backend_type, metrics_url, label_filter, model=model) if metrics_enabled else {
                "scrape": {"ok": False, "backend_type": backend_type, "metrics_url": metrics_url, "disabled": True},
                "tokens": {},
                "cache": {},
                "scheduler": {},
            }

            result_obj = {
                "meta": {
                    "run_id": str(uuid.uuid4()),
                    "run_name": run_name,
                    "profile": profile_meta(run, run_name, mode, concurrency, total_requests),
                    "run_index": idx,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "duration_sec": round(duration_sec, 6),
                    "backend_type_configured": backend_type,
                    "base_url": base_url,
                    "metrics_url": metrics_url,
                    "model": model,
                    "mode": mode,
                    "concurrency": concurrency,
                    "total_requests": total_requests,
                    "target_tokens": run.get("target_tokens"),
                    "request_after_roles": run.get("request_after_roles"),
                    "messages_path": str(messages_path),
                    "system_path": str(system_path) if system_path else None,
                    "dialogues_loaded": len(repo.dialogues),
                    "payload": public_payload_meta(payload),
                    "tokenizer": counter.meta(),
                },
                "recorder": recorder,
                "backend": backend,
            }
            write_jsonl(output_jsonl, result_obj)

            print(
                "  done "
                f"ok={recorder['requests']['success']}/{recorder['requests']['total']} "
                f"ttft_p95={recorder['ttft_sec']['p95']} "
                f"tpot_p95={recorder['tpot_sec']['p95']} "
                f"e2e_p95={recorder['e2e_sec']['p95']} "
                f"out_tok/s={recorder['throughput']['completion_tokens_per_sec']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple vLLM/SGLang OpenAI-compatible benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    asyncio.run(run_benchmark(config, config_path))


if __name__ == "__main__":
    main()
