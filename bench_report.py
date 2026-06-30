#!/usr/bin/env python3
"""
Aggregate vLLM/SGLang benchmark JSONL reports, render comparison plots, and
(optionally) build a single self-contained Confluence-ready HTML page with a
navigation table on top and a "Обратно" back-link above every graph.

Plots tree:
    out/
      ctx_sweep/{ttft,tpot,itl,e2e,throughput,reliability}/c{N}.png
      dialog_replay/{ttft,tpot,itl,e2e,throughput,reliability}.png
      mixed_callcenter/...
      decode_7k_512/summary_bars.png
      summary.csv
      confluence_report.html        # when --html is passed

Series (lines) come from report file names; override with --labels labels.json.

Usage:
    python bench_report.py --reports-dir ./results --out-dir ./plots
    python bench_report.py --reports-dir ./results --out-dir ./plots \
        --labels labels.json --html --html-img embed
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


###############################################################################
# Loading
###############################################################################


def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_dict(value, name, sep=sep))
        else:
            out[name] = value
    return out


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] {path.name}:{line_no} invalid JSON, skipped: {exc}", file=sys.stderr)
                continue
            rows.append(flatten_dict(obj))
    return rows


def prettify_stem(stem: str) -> str:
    name = re.sub(r"_bench$", "", stem)
    return name.replace("_", " ").strip()


def load_reports(reports_dir: Path, labels: Dict[str, str]) -> pd.DataFrame:
    files = sorted(reports_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *.jsonl files found in {reports_dir}")

    frames: List[pd.DataFrame] = []
    for file in files:
        rows = read_jsonl(file)
        if not rows:
            print(f"[warn] {file.name}: no rows", file=sys.stderr)
            continue
        df = pd.DataFrame(rows)
        stem = file.stem
        df["series"] = labels.get(stem, prettify_stem(stem))
        df["series_file"] = stem
        frames.append(df)
        print(f"[load] {file.name}: {len(df)} rows -> series='{df['series'].iloc[0]}'")

    if not frames:
        raise ValueError("All report files were empty")
    return pd.concat(frames, ignore_index=True)


###############################################################################
# Profile / axis logic
###############################################################################

COL_BASE_NAME = "meta.profile.key.base_name"
COL_CONCURRENCY = "meta.profile.key.concurrency"
COL_TARGET = "meta.profile.key.target_tokens"

LATENCY_METRICS = {  # kind -> short label
    "ttft": "TTFT",
    "e2e": "E2E latency",
    "tpot": "TPOT",
    "itl": "ITL",
}

SCALAR_PANELS = {
    "throughput": [
        ("recorder.throughput.completion_tokens_per_sec", "output tok/s"),
        ("recorder.throughput.total_tokens_per_sec", "total tok/s"),
        ("recorder.throughput.requests_per_sec", "req/s"),
    ],
    "reliability": [
        ("recorder.requests.success_rate", "success rate"),
        ("recorder.throughput.errors_per_sec", "errors/s"),
        ("backend.cache.hit_rate", "prefix-cache hit rate"),
    ],
}

# Order and full names used for the HTML navigation table columns.
COLUMN_ORDER = ["ttft", "e2e", "tpot", "itl", "throughput", "reliability"]
COLUMN_FULL = {
    "ttft": "Time To First Token",
    "e2e": "End-2-End Latency",
    "tpot": "Time Per Output Token",
    "itl": "Inter Token Latency",
    "throughput": "Throughput",
    "reliability": "Reliability",
}
COLUMN_SHORT = {
    "ttft": "TTFT", "e2e": "E2E", "tpot": "TPOT", "itl": "ITL",
    "throughput": "tok/s", "reliability": "rel",
}

DEFAULT_SECTION_TITLES = {
    "ctx_sweep": "Fixed Context",
    "dialog_replay": "Dialogue Replay",
    "decode_7k_512": "Decode",
    "short_answer_7k": "Short Answer",
    "mixed_callcenter": "Mixed",
}


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def profile_column(df: pd.DataFrame) -> str:
    col = _first_present(df, [COL_BASE_NAME, "meta.profile.name", "meta.mode", "meta.run_name"])
    if col is None:
        raise KeyError("Cannot find a profile-identifying column in reports")
    return col


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def choose_time_unit(values_sec: pd.Series) -> Tuple[str, float]:
    v = pd.to_numeric(values_sec, errors="coerce").dropna()
    if v.empty:
        return "ms", 1000.0
    mx = float(v.max())
    if mx <= 10.0:
        return "ms", 1000.0
    if mx <= 120.0:
        return "s", 1.0
    return "min", 1.0 / 60.0


def section_title(profile: str) -> str:
    return DEFAULT_SECTION_TITLES.get(profile, prettify_stem(profile))


def ru_threads(n: int) -> str:
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        w = "поток"
    elif 2 <= n10 <= 4 and not (12 <= n100 <= 14):
        w = "потока"
    else:
        w = "потоков"
    return f"{n} {w}"


###############################################################################
# Colors: stable per-series across every figure
###############################################################################


class SeriesStyle:
    def __init__(self, series_names: List[str]) -> None:
        names = sorted(set(series_names))
        cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20", max(len(names), 1))
        markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H", "p", "8"]
        self.color = {n: cmap(i) for i, n in enumerate(names)}
        self.marker = {n: markers[i % len(markers)] for i, n in enumerate(names)}

    def c(self, name: str):
        return self.color.get(name, "tab:gray")

    def m(self, name: str) -> str:
        return self.marker.get(name, "o")


###############################################################################
# Plotters
###############################################################################


def _fmt_tick(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _annotate(ax, xs, ys, color, enabled: bool) -> None:
    if not enabled or len(xs) > 8:
        return
    for x, y in zip(xs, ys):
        if y is None or (isinstance(y, float) and not math.isfinite(y)):
            continue
        ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7, color=color)


def plot_latency_family(series_frames, x_col, metric_key, metric_label, percentiles, style,
                        x_label, title, out_path, annotate=False, log_x=False, dpi=130) -> bool:
    y_cols = [f"recorder.{metric_key}_sec.{p}" for p in percentiles]
    all_y = [pd.to_numeric(df[yc], errors="coerce") for df in series_frames.values() for yc in y_cols if yc in df.columns]
    if not all_y:
        return False
    unit_name, unit_mult = choose_time_unit(pd.concat(all_y, ignore_index=True))

    ncols = 2 if len(percentiles) > 1 else 1
    nrows = math.ceil(len(percentiles) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.5 * ncols, 4.2 * nrows), squeeze=False)
    x_all = sorted({x for df in series_frames.values()
                    for x in pd.to_numeric(df[x_col], errors="coerce").dropna().tolist()})

    plotted = False
    for idx, p in enumerate(percentiles):
        ax = axes[idx // ncols][idx % ncols]
        yc = f"recorder.{metric_key}_sec.{p}"
        for name, df in series_frames.items():
            if yc not in df.columns or x_col not in df.columns:
                continue
            sub = df[[x_col, yc]].copy()
            sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
            sub[yc] = pd.to_numeric(sub[yc], errors="coerce") * unit_mult
            sub = sub.dropna().sort_values(x_col)
            if sub.empty:
                continue
            plotted = True
            ax.plot(sub[x_col], sub[yc], color=style.c(name), marker=style.m(name),
                    markersize=6, linewidth=2, label=name, zorder=3)
            _annotate(ax, sub[x_col].tolist(), sub[yc].tolist(), style.c(name), annotate)
        ax.set_title(f"{metric_label} {p}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{metric_label}, {unit_name}")
        if log_x:
            ax.set_xscale("log")
        if x_all:
            ax.set_xticks(x_all)
            ax.set_xticklabels([_fmt_tick(v) for v in x_all])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for e in range(len(percentiles), nrows * ncols):
        axes[e // ncols][e % ncols].axis("off")
    if not plotted:
        plt.close(fig)
        return False
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def plot_scalar_panel(series_frames, x_col, columns, style, x_label, title, out_path,
                      annotate=False, log_x=False, dpi=130) -> bool:
    present = [(c, l) for c, l in columns if any(c in df.columns for df in series_frames.values())]
    if not present:
        return False
    ncols = min(len(present), 2) if len(present) > 1 else 1
    nrows = math.ceil(len(present) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.5 * ncols, 4.2 * nrows), squeeze=False)
    x_all = sorted({x for df in series_frames.values()
                    for x in pd.to_numeric(df[x_col], errors="coerce").dropna().tolist()})

    plotted = False
    for idx, (col, lbl) in enumerate(present):
        ax = axes[idx // ncols][idx % ncols]
        for name, df in series_frames.items():
            if col not in df.columns or x_col not in df.columns:
                continue
            sub = df[[x_col, col]].copy()
            sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub = sub.dropna().sort_values(x_col)
            if sub.empty:
                continue
            plotted = True
            ax.plot(sub[x_col], sub[col], color=style.c(name), marker=style.m(name),
                    markersize=6, linewidth=2, label=name, zorder=3)
            _annotate(ax, sub[x_col].tolist(), sub[col].tolist(), style.c(name), annotate)
        ax.set_title(lbl)
        ax.set_xlabel(x_label)
        ax.set_ylabel(lbl)
        if log_x:
            ax.set_xscale("log")
        if x_all:
            ax.set_xticks(x_all)
            ax.set_xticklabels([_fmt_tick(v) for v in x_all])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for e in range(len(present), nrows * ncols):
        axes[e // ncols][e % ncols].axis("off")
    if not plotted:
        plt.close(fig)
        return False
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def plot_bars(df, columns, style, title, out_path, dpi=130) -> bool:
    present = [(c, l) for c, l in columns if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()]
    if not present:
        return False
    series_names = sorted(df["series"].unique().tolist())
    ncols = min(len(present), 3) if len(present) > 1 else 1
    nrows = math.ceil(len(present) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 4.0 * nrows), squeeze=False)
    plotted = False
    for idx, (col, lbl) in enumerate(present):
        ax = axes[idx // ncols][idx % ncols]
        vals, names, colors = [], [], []
        for name in series_names:
            v = pd.to_numeric(df[df["series"] == name][col], errors="coerce").dropna()
            if v.empty:
                continue
            vals.append(float(v.iloc[0])); names.append(name); colors.append(style.c(name))
        if not vals:
            ax.axis("off"); continue
        plotted = True
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(lbl)
        ax.grid(True, axis="y", alpha=0.3)
    for e in range(len(present), nrows * ncols):
        axes[e // ncols][e % ncols].axis("off")
    if not plotted:
        plt.close(fig); return False
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


###############################################################################
# Driver: render plots and collect structured entries for the HTML index
###############################################################################


def split_series(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {name: sub for name, sub in df.groupby("series")}


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "x"


def _entry(profile, kind, path, concurrency=None):
    c = None if concurrency is None else int(concurrency)
    anchor = f"{_safe(profile)}__{kind}" + (f"__c{c}" if c is not None else "")
    return {"profile": profile, "section": section_title(profile), "kind": kind,
            "concurrency": c, "anchor": anchor, "path": Path(path)}


def render_profile(profile_df, profile_name, style, out_dir, percentiles, annotate, dpi, entries) -> None:
    has_target = COL_TARGET in profile_df.columns and pd.to_numeric(profile_df[COL_TARGET], errors="coerce").notna().any()
    conc_col = _first_present(profile_df, [COL_CONCURRENCY, "meta.concurrency"])
    target_col = COL_TARGET if has_target else None
    n_target = profile_df[target_col].nunique() if target_col else 0
    n_conc = profile_df[conc_col].nunique() if conc_col else 0
    pout = out_dir / _safe(profile_name)

    def latency_and_scalars(sf, x_col, x_label, title_suffix, fname_for, concurrency):
        for mkey, mlabel in LATENCY_METRICS.items():
            p = fname_for(mkey)
            if plot_latency_family(sf, x_col, mkey, mlabel, percentiles, style, x_label,
                                   f"{profile_name} · {mlabel}{title_suffix}", p, annotate, _log(x_col), dpi):
                entries.append(_entry(profile_name, mkey, p, concurrency))
        for panel, cols in SCALAR_PANELS.items():
            p = fname_for(panel)
            if plot_scalar_panel(sf, x_col, cols, style, x_label,
                                 f"{profile_name} · {panel}{title_suffix}", p, annotate, _log(x_col), dpi):
                entries.append(_entry(profile_name, panel, p, concurrency))

    def _log(x_col):
        vals = pd.to_numeric(profile_df[x_col], errors="coerce").dropna()
        if vals.empty:
            return False
        ratio = vals.max() / max(vals.min(), 1)
        return ratio >= (8 if x_col == target_col else 16)

    # Case A: target_tokens AND concurrency both vary -> facet by concurrency, x=target.
    if target_col and n_target > 1 and conc_col and n_conc > 1:
        for c in sorted(pd.to_numeric(profile_df[conc_col], errors="coerce").dropna().unique()):
            c = int(c)
            sub = profile_df[pd.to_numeric(profile_df[conc_col], errors="coerce") == c]
            sf = split_series(sub)
            latency_and_scalars(sf, target_col, "Context tokens", f" · concurrency={c}",
                                lambda k, c=c: pout / k / f"c{c}.png", c)
        return

    # Case B: concurrency varies -> x=concurrency, one figure per metric.
    if conc_col and n_conc > 1:
        sf = split_series(profile_df)
        latency_and_scalars(sf, conc_col, "Concurrency", "", lambda k: pout / f"{k}.png", None)
        return

    # Case C: target varies, single concurrency -> x=target, one figure per metric.
    if target_col and n_target > 1:
        sf = split_series(profile_df)
        latency_and_scalars(sf, target_col, "Context tokens", "", lambda k: pout / f"{k}.png", None)
        return

    # Case D: single operating point -> grouped bars across series.
    bar_cols = [(f"recorder.{k}_sec.p95", f"{lbl} p95") for k, lbl in LATENCY_METRICS.items()]
    bar_cols += SCALAR_PANELS["throughput"] + SCALAR_PANELS["reliability"]
    p = pout / "summary_bars.png"
    if plot_bars(profile_df, bar_cols, style, f"{profile_name} · single point", p, dpi):
        entries.append(_entry(profile_name, "bars", p, None))


###############################################################################
# Summary CSV
###############################################################################


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    pcol = profile_column(df)
    conc_col = _first_present(df, [COL_CONCURRENCY, "meta.concurrency"])
    target_col = COL_TARGET if COL_TARGET in df.columns else None
    keep = {"series": "series", pcol: "profile"}
    if conc_col:
        keep[conc_col] = "concurrency"
    if target_col:
        keep[target_col] = "target_tokens"
    metric_cols = {
        "recorder.ttft_sec.p50": "ttft_p50_s", "recorder.ttft_sec.p95": "ttft_p95_s",
        "recorder.ttft_sec.p99": "ttft_p99_s", "recorder.tpot_sec.p95": "tpot_p95_s",
        "recorder.e2e_sec.p95": "e2e_p95_s",
        "recorder.throughput.completion_tokens_per_sec": "output_tok_s",
        "recorder.throughput.total_tokens_per_sec": "total_tok_s",
        "recorder.throughput.requests_per_sec": "req_s",
        "recorder.throughput.errors_per_sec": "errors_s",
        "recorder.requests.success_rate": "success_rate",
        "recorder.requests.total": "requests",
        "recorder.token_accounting.completion_from_usage_rate": "completion_from_usage_rate",
        "backend.cache.hit_rate": "cache_hit_rate",
    }
    cols = {**keep, **{c: n for c, n in metric_cols.items() if c in df.columns}}
    out = df[list(cols.keys())].rename(columns=cols).copy()
    sort_cols = [c for c in ["profile", "concurrency", "target_tokens", "series"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


###############################################################################
# Confluence HTML
###############################################################################


def _img_src(path: Path, mode: str, assets_dir: Path, anchor: str) -> str:
    if mode == "embed":
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{data}"
    # link mode: copy to a flat assets dir with a unique name; reference the basename
    # (Confluence matches a pasted <img src="name.png"> to an attachment of that name).
    assets_dir.mkdir(parents=True, exist_ok=True)
    flat = f"{anchor}.png"
    shutil.copyfile(path, assets_dir / flat)
    return flat


def build_html(entries: List[dict], out_dir: Path, title: str, img_mode: str, html_name: str) -> Path:
    assets_dir = out_dir / "html_assets"
    # Group entries by profile, preserving first-seen order.
    profiles: List[str] = []
    by_profile: Dict[str, List[dict]] = {}
    for e in entries:
        by_profile.setdefault(e["profile"], []).append(e)
        if e["profile"] not in profiles:
            profiles.append(e["profile"])

    esc = html.escape
    parts: List[str] = []
    parts.append(f'<div id="bench-report"><a id="index"></a>')
    parts.append(
        "<style>"
        "#bench-report table{border-collapse:collapse;margin:8px 0 24px;width:auto}"
        "#bench-report th,#bench-report td{border:1px solid #c9d1d9;padding:6px 12px;text-align:center;font-size:14px}"
        "#bench-report th{background:#2c3440;color:#fff}"
        "#bench-report td.rowhdr{font-weight:600;background:#f4f5f7;text-align:left}"
        "#bench-report h2{margin-top:28px}"
        "#bench-report .back{font-size:13px;margin:4px 0 8px}"
        "#bench-report img{max-width:100%;height:auto;border:1px solid #e1e4e8;border-radius:6px}"
        "#bench-report .figsec{margin:18px 0 30px}"
        "</style>"
    )
    parts.append(f"<h1>{esc(title)}</h1>")

    # ---- Navigation tables (one per profile) ----
    for prof in profiles:
        ents = by_profile[prof]
        cols_present = [k for k in COLUMN_ORDER if any(e["kind"] == k for e in ents)]
        # Map (concurrency, kind) -> anchor
        cell = {(e["concurrency"], e["kind"]): e["anchor"] for e in ents}
        concs = sorted({e["concurrency"] for e in ents if e["concurrency"] is not None})
        bars = [e for e in ents if e["kind"] == "bars"]

        parts.append(f"<h2>{esc(section_title(prof))}</h2>")
        parts.append("<table><thead><tr>")
        first_hdr = "Concurrency" if concs else "Test"
        parts.append(f"<th>{first_hdr}</th>")
        for k in cols_present:
            parts.append(f"<th>{esc(COLUMN_FULL[k])}</th>")
        parts.append("</tr></thead><tbody>")

        rows = [(c, c) for c in concs] if concs else [(None, None)]
        for cval, _ in rows:
            rowname = ru_threads(cval) if cval is not None else section_title(prof)
            parts.append(f'<tr><td class="rowhdr">{esc(rowname)}</td>')
            for k in cols_present:
                anchor = cell.get((cval, k))
                if anchor:
                    parts.append(f'<td><a href="#{anchor}">{esc(COLUMN_SHORT[k])}</a></td>')
                else:
                    parts.append("<td></td>")
            parts.append("</tr>")
        if bars:
            parts.append(f'<tr><td class="rowhdr">summary</td>'
                         f'<td colspan="{max(len(cols_present),1)}">'
                         f'<a href="#{bars[0]["anchor"]}">open</a></td></tr>')
        parts.append("</tbody></table>")

    # ---- Figure sections ----
    def sort_key(e):
        return (profiles.index(e["profile"]),
                -1 if e["concurrency"] is None else e["concurrency"],
                COLUMN_ORDER.index(e["kind"]) if e["kind"] in COLUMN_ORDER else 99)

    for e in sorted(entries, key=sort_key):
        kind_name = COLUMN_FULL.get(e["kind"], e["kind"].title())
        suffix = f" · {ru_threads(e['concurrency'])}" if e["concurrency"] is not None else ""
        heading = f"{section_title(e['profile'])} · {kind_name}{suffix}"
        src = _img_src(e["path"], img_mode, assets_dir, e["anchor"])
        parts.append(f'<div class="figsec"><h3 id="{e["anchor"]}">{esc(heading)}</h3>')
        parts.append('<div class="back"><a href="#index">(Обратно)</a></div>')
        parts.append(f'<img src="{src}" alt="{esc(heading)}"/></div>')

    parts.append("</div>")
    out_path = out_dir / html_name
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


###############################################################################
# Confluence Storage Format (for self-hosted Server / Data Center)
#
# On-prem Confluence strips data-URI images on paste, so we reference images as
# page ATTACHMENTS and use the native anchor macro for in-page navigation. This
# XML is exactly what the REST API consumes (representation="storage"), so it can
# be published in one shot by confluence_publish.py.
###############################################################################


def _anchor_macro(name: str) -> str:
    return (f'<ac:structured-macro ac:name="anchor">'
            f'<ac:parameter ac:name="">{html.escape(name)}</ac:parameter>'
            f'</ac:structured-macro>')


def _anchor_link(anchor: str, text: str) -> str:
    return (f'<ac:link ac:anchor="{html.escape(anchor)}">'
            f'<ac:link-body>{html.escape(text)}</ac:link-body></ac:link>')


def build_confluence_storage(entries: List[dict], out_dir: Path, title: str,
                             storage_name: str, img_width: int = 980) -> Path:
    """Emit Confluence Storage Format XML + flat PNG attachments in html_assets/."""
    assets_dir = out_dir / "html_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    profiles: List[str] = []
    by_profile: Dict[str, List[dict]] = {}
    for e in entries:
        by_profile.setdefault(e["profile"], []).append(e)
        if e["profile"] not in profiles:
            profiles.append(e["profile"])

    esc = html.escape
    p: List[str] = []
    p.append(f"<p>{_anchor_macro('index')}</p>")
    p.append(f"<h1>{esc(title)}</h1>")

    # ---- Navigation tables ----
    for prof in profiles:
        ents = by_profile[prof]
        cols_present = [k for k in COLUMN_ORDER if any(e["kind"] == k for e in ents)]
        cell = {(e["concurrency"], e["kind"]): e["anchor"] for e in ents}
        concs = sorted({e["concurrency"] for e in ents if e["concurrency"] is not None})
        bars = [e for e in ents if e["kind"] == "bars"]

        p.append(f"<h2>{esc(section_title(prof))}</h2>")
        p.append("<table><tbody><tr>")
        p.append(f"<th>{'Concurrency' if concs else 'Test'}</th>")
        for k in cols_present:
            p.append(f"<th>{esc(COLUMN_FULL[k])}</th>")
        p.append("</tr>")

        rows = concs if concs else [None]
        for cval in rows:
            rowname = ru_threads(cval) if cval is not None else section_title(prof)
            p.append(f"<tr><td>{esc(rowname)}</td>")
            for k in cols_present:
                anchor = cell.get((cval, k))
                p.append(f"<td>{_anchor_link(anchor, COLUMN_SHORT[k]) if anchor else ''}</td>")
            p.append("</tr>")
        if bars:
            p.append(f'<tr><td>summary</td><td colspan="{max(len(cols_present),1)}">'
                     f'{_anchor_link(bars[0]["anchor"], "open")}</td></tr>')
        p.append("</tbody></table>")

    # ---- Figure sections (image = attachment) ----
    def sort_key(e):
        return (profiles.index(e["profile"]),
                -1 if e["concurrency"] is None else e["concurrency"],
                COLUMN_ORDER.index(e["kind"]) if e["kind"] in COLUMN_ORDER else 99)

    for e in sorted(entries, key=sort_key):
        kind_name = COLUMN_FULL.get(e["kind"], e["kind"].title())
        suffix = f" · {ru_threads(e['concurrency'])}" if e["concurrency"] is not None else ""
        heading = f"{section_title(e['profile'])} · {kind_name}{suffix}"
        fname = f"{e['anchor']}.png"
        shutil.copyfile(e["path"], assets_dir / fname)
        p.append(f"<p>{_anchor_macro(e['anchor'])}</p>")
        p.append(f"<h3>{esc(heading)}</h3>")
        p.append(f"<p>{_anchor_link('index', '(Обратно)')}</p>")
        p.append(f'<ac:image ac:width="{img_width}"><ri:attachment ri:filename="{esc(fname)}"/></ac:image>')

    out_path = out_dir / storage_name
    out_path.write_text("\n".join(p), encoding="utf-8")
    return out_path


###############################################################################
# Main
###############################################################################


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate benchmark JSONL reports, plot, and build Confluence output")
    ap.add_argument("--reports-dir", required=True)
    ap.add_argument("--out-dir", default="./plots")
    ap.add_argument("--labels", default=None, help="JSON mapping {file_stem: display_name}")
    ap.add_argument("--percentiles", nargs="+", default=["p50", "p95", "p99", "mean"])
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--html", action="store_true", help="Also build a Confluence-ready HTML page")
    ap.add_argument("--html-img", choices=["embed", "link"], default="embed",
                    help="embed=base64 in HTML (single self-contained paste); "
                         "link=flat PNGs in html_assets/, referenced by filename (upload as attachments)")
    ap.add_argument("--html-name", default="confluence_report.html")
    ap.add_argument("--confluence", action="store_true",
                    help="Build Confluence Storage Format XML + attachment PNGs (for self-hosted Server/DC)")
    ap.add_argument("--confluence-name", default="confluence_storage.xml")
    ap.add_argument("--title", default="LLM serving benchmark")
    args = ap.parse_args()

    reports_dir = Path(args.reports_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8")) if args.labels else {}

    df = load_reports(reports_dir, labels)
    pcol = profile_column(df)
    df = coerce_numeric(df, [c for c in [COL_CONCURRENCY, COL_TARGET, "meta.concurrency"] if c in df.columns])
    style = SeriesStyle(df["series"].unique().tolist())

    entries: List[dict] = []
    profiles = [p for p in df[pcol].dropna().unique()]
    print(f"\nProfiles found: {profiles}")
    for prof in profiles:
        prof_df = df[df[pcol] == prof].dropna(axis=1, how="all")
        render_profile(prof_df, str(prof), style, out_dir, args.percentiles, args.annotate, args.dpi, entries)

    summary = build_summary(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nWrote {len(entries)} figures + summary.csv under {out_dir}")

    if args.html:
        html_path = build_html(entries, out_dir, args.title, args.html_img, args.html_name)
        print(f"Wrote HTML: {html_path}  (img mode: {args.html_img})")

    if args.confluence:
        xml_path = build_confluence_storage(entries, out_dir, args.title, args.confluence_name)
        n_assets = len(list((out_dir / "html_assets").glob("*.png")))
        print(f"Wrote Confluence storage: {xml_path}")
        print(f"  + {n_assets} attachment PNGs in {out_dir / 'html_assets'}")
        print(f"  publish with: python confluence_publish.py --base-url <URL> --page-id <ID> "
              f"--token <PAT> --body {xml_path} --assets {out_dir / 'html_assets'}")


if __name__ == "__main__":
    main()
