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

---

## Self-hosted Confluence (Server / Data Center) — the on-prem path

On-prem Confluence strips `data:` image URIs on paste, so `--html-img embed` will
**not** show images there. Use attachment-based output instead. Two routes:

### A. One command via REST (recommended)

```bash
# 1) build storage-format XML + flat attachment PNGs
python bench_report.py --reports-dir ./results --out-dir ./plots --labels labels.json \
    --confluence --title "SG Lang vs vLLM на 2x4090"

# 2) create an empty page in the UI, grab its numeric pageId, then publish:
python confluence_publish.py \
    --base-url https://confluence.corp.local \
    --page-id 123456789 \
    --token "$CONFLUENCE_TOKEN" \
    --body ./plots/confluence_storage.xml \
    --assets ./plots/html_assets
```

`--confluence` emits `confluence_storage.xml` (native Confluence Storage Format:
`<ac:image><ri:attachment>` for figures, anchor macros for the nav table and the
**(Обратно)** links) plus one flat PNG per figure in `html_assets/`.

`confluence_publish.py` uploads every PNG as a page attachment and replaces the
page body — one shot, no manual image dragging. Auth is a **Personal Access
Token** (Confluence Server/DC 7.9+: Profile → Settings → Personal Access Tokens);
pass it via `--token` or the `CONFLUENCE_TOKEN` env var. Use `--dry-run` first to
see the plan without writing, and `--verify-tls false` for self-signed corp certs.

> The page must already exist (publisher overwrites its body); point `--page-id` at it.

### B. No API access (manual, still no per-image dragging)

If you can't get a token: open the page attachments view and bulk-upload everything
in `html_assets/` at once (multi-select drag), then paste the contents of
`confluence_storage.xml` via the page's source/storage editor. The `<img>`
references resolve against the attachments by filename.

### C. HTML macro (only if your admin enabled it)

If the `{html}` macro is enabled, you can paste the `--html --html-img embed`
output inside it and images render from base64. Most locked-down corporate
instances keep this macro disabled, so prefer route A.
