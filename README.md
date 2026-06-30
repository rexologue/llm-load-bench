# LLM serving benchmark — runner + report

Two files:

- **`benchmark.py`** — the OpenAI-compatible vLLM/SGLang benchmark runner (unchanged load model: closed-loop, fixed concurrency), with two additive transparency fixes baked in.
- **`bench_report.py`** — aggregates the JSONL reports into comparison plots and, optionally, a single Confluence-ready HTML page.

---

## benchmark.py — what changed vs your version

Only additive fields. Nothing in the load model, profiles, or existing output keys was altered, so old reports and the notebook still parse.

1. **Token accounting transparency.** `TPOT` and `completion_tokens_per_sec` depend on `completion_tokens`, which silently falls back to counting stream chunks when the backend doesn't return `usage`. New block in each report:
   ```json
   "recorder": {
     "token_accounting": {
       "usage_present_rate": 1.0,
       "completion_from_usage_rate": 1.0
     }
   }
   ```
   If `completion_from_usage_rate < 1.0`, those token-derived metrics are approximate on that run.

2. **Goodput / error visibility.** `recorder.throughput` now also has `errors_per_sec`, and a `note` clarifying that throughput counts only successful requests over wall-clock duration. So a throughput drop under overload is distinguishable from failures.

Run unchanged:
```bash
python benchmark.py --config config.yaml
```

> Note: `ctx_sweep` (fixed_context) builds context by repeating one dialogue, so its prefix-cache hit_rate is inflated vs real distinct calls. Fine for engine-vs-engine comparison (symmetric), but take absolute cache numbers from `dialog_replay`/`mixed`.

---

## bench_report.py — plots

```bash
python bench_report.py --reports-dir ./results --out-dir ./plots --labels labels.json
```

One `*.jsonl` in `--reports-dir` = one series (line) on every plot. Names come from the file stem; override with `--labels` (see `labels.example.json`).

Profiles and axes are detected from the data:
- target_tokens **and** concurrency vary (`ctx_sweep`) -> x = context tokens, one figure per concurrency -> `ctx_sweep/ttft/c20.png`
- only concurrency varies (`dialog_replay`, `mixed_callcenter`) -> x = concurrency -> `dialog_replay/ttft.png`
- only target varies -> x = context tokens
- single point (`decode_7k_512`) -> grouped bars across series

Each profile gets latency families (`ttft/tpot/itl/e2e`, percentiles configurable via `--percentiles p50 p95 p99 mean`), plus **throughput** (output/total tok/s, req/s) and **reliability** (success_rate, errors/s, cache hit_rate) panels — the throughput plots the notebook never had. Colors are stable per series across all figures. `summary.csv` holds the key numbers.

Other flags: `--annotate` (values on points, auto-suppressed when >8 points), `--dpi`.

---

## bench_report.py — Confluence HTML

Add `--html` to also emit a single page with your navigation layout: a per-profile
table (rows = concurrency with Russian pluralization, columns = TTFT / E2E / TPOT /
ITL / Throughput / Reliability) whose cells jump to the matching graph, and an
**(Обратно)** back-link above every graph.

```bash
python bench_report.py --reports-dir ./results --out-dir ./plots --labels labels.json --html
```

### The image problem — two modes

`--html-img embed` (default): images are inlined as base64 data-URIs. The HTML is
fully self-contained — one paste, no attachments. Most Confluence Cloud instances
convert pasted data-URI images into attachments automatically. Downside: the file
gets big (~150-250 KB per figure; a full 11-concurrency sweep can be 10-20 MB),
and a very large paste can choke the editor.

`--html-img link`: figures are copied flat into `out/html_assets/` with unique
names, and the HTML references them by bare filename (`<img src="ctx_sweep__ttft__c20.png">`).
Confluence matches each `<img>` to a page **attachment** of the same name. Workflow:
upload everything in `html_assets/` as attachments to the page, then paste the HTML.

```bash
# leaner page for large sweeps
python bench_report.py --reports-dir ./results --out-dir ./plots --html --html-img link --dpi 110
```

Other HTML flags: `--html-name confluence_report.html`, `--title "SG Lang vs vLLM na 2x4090"`.

If your instance strips data-URIs or the embedded page is too heavy, switch to
`link` mode and lower `--dpi`.
