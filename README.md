# METRIC MIRAGE

### The business metric truth engine

> **What important business reality is hidden behind the numbers everyone is looking at?**

Metric Mirage is a forensic analytics workstation built with Python, Pandas, SciPy, Statsmodels, SQLite and Streamlit. It does not just chart KPIs. It tests whether a headline number means what people think it means: whether it reverses inside segments (Simpson's paradox), is carried by a handful of extreme orders, is distorted by missing or duplicated data, or only moved because the customer mix changed.

Load the built-in public retail dataset, or **upload your own CSV or Excel file**.

> **Disclaimer.** Metric Mirage reports statistical patterns and investigation leads. It does not establish causal truths. Every finding lists alternative explanations and limitations, and the tool says "no finding" when there is none.

<!-- Replace these with your own screenshots (save them in docs/images/). -->
![Command Center](docs/images/command-center.png)
![Contradiction Lab](docs/images/contradiction-lab.png)
![Evidence Room](docs/images/evidence-room.png)

---

## Why this is not just another dashboard

| An ordinary dashboard | Metric Mirage |
|---|---|
| Shows the number | Interrogates whether the number means what readers assume |
| Slices only where the viewer thinks to look | Decomposes a change into a *mix* effect and a *rate* effect, exactly |
| Reports a difference | States whether it is a real reversal, a mix shift, or just a difference in size |
| Charts are the output | Evidence is the output: every finding records its metric, method, sample size, uncertainty and parameters |
| Implicitly claims truth | Lists alternative explanations and limitations for every finding |
| Cannot be audited | Findings can be re-run from stored parameters, and SQL and Python must agree |

The design rule: **the system must be able to say "no finding"**, and it is tested against datasets with known answers to prove it does.

---

## Features

| Page | What it does |
|---|---|
| **Command Center** | Dataset health, headline metrics, and which analyses the loaded data supports |
| **Upload Data** | Bring your own CSV or Excel file: automatic cleaning report, date handling, column mapping, preview |
| **Metric Integrity** | Missing data by column, time and segment; duplicates; outliers (IQR and MAD); mean vs median; change-point candidates; rate and leakage warnings |
| **Contradiction Lab** | Aggregate-versus-segment analysis with an exact mix/rate decomposition, confidence intervals and FDR correction |
| **Cohort Observatory** | Retention heatmaps, repeat purchasing, revenue concentration (Gini), value tiers, funnels when stage data exists |
| **Experiment Lab** | Proportion and mean comparisons, practical significance, power and sample size, SRM and balance checks, adjusted association, causal-claim tiers |
| **Evidence Room** | Saved investigation cards, re-run from stored parameters, Markdown and HTML reports |

Analyses that the loaded data cannot support are switched off with a stated reason, for example no `customer_id` means no cohort analysis. Nothing is invented.

---

## The central method

For a metric `M = sum(numerator) / sum(denominator)`, write it as a weighted average over segments, `M = sum_k w_k * m_k`, where `w_k` is segment *k*'s share of the denominator and `m_k` its value. The change between two periods splits **exactly** (midpoint weights) into:

```
mix effect  = sum_k (w1_k - w0_k) * (m0_k + m1_k) / 2        who is in the data
rate effect = sum_k (w0_k + w1_k) / 2 * (m1_k - m0_k)        what happens inside each segment
```

plus an entry/exit term for segments that exist in only one period. The verdict comes from the signs and sizes of these pieces:

| Verdict | Meaning |
|---|---|
| Simpson-type reversal | Overall moves with the mix effect while every evaluable segment moves the other way |
| Partial reversal | The average within-segment movement opposes the overall change |
| Driven by a shift in mix | Composition explains most of the change, no reversal |
| Hidden movement | Flat aggregate, but at least one segment changed significantly |
| Heterogeneous / consistent | No aggregation paradox |
| Insufficient data | Too few evaluable segments, so no claim is made |

A difference in magnitude is never called a paradox. Segment changes get delta-method standard errors and Benjamini-Hochberg FDR correction, and the engine says when a reversal is arithmetically real but statistically unsupported. In the built-in teaching example the overall rate rises from 25% to 38% while both segments fall by 2 points; at that sample size neither decline is distinguishable from noise, so the finding is rated *descriptive* rather than a statistical lead. The same structure at 20 times the data becomes a statistical lead, and a test proves it.

### Investigation cards

Every finding is a structured record: title, metric and comparison, exact numbers, affected segment or period, method, sample size, uncertainty, a supporting chart, alternative explanations, limitations, a recommended next step, and the parameters needed to reproduce it. There is deliberately no "confidence score", because any such number would be arbitrary.

---

## Data

### Public dataset

**UCI Online Retail II**: transactions from a UK online retailer, December 2009 to December 2011.

1. Download `online_retail_II.xlsx` from the UCI Machine Learning Repository and check the licence and citation terms there.
2. Place it at `data/raw/online_retail_II.xlsx`. It is not included in this repository.

> **Citation:** *[paste the official UCI citation text here]*

What the source does **not** contain: cost, margin, campaigns, acquisition cost or retention events. Profit and acquisition-cost analyses are therefore not possible, and the tool does not pretend otherwise.

Handling decisions (documented in `docs/METRICS.md`):

- The workbook's two sheets overlap by 22,523 identical rows; the duplicate copy is removed, leaving 1,044,848 rows.
- Returns (invoices starting with `C`), non-product stock codes (postage, fees, adjustments) and zero-price rows are classified and counted, never silently dropped.
- Rows without a customer ID stay in revenue totals and are excluded from customer metrics. Both figures are reported.
- Currency is assumed to be GBP.

### Simulated teaching datasets

Six reproducible datasets with known ground truth: a constructed Simpson's paradox, a mix-shift-only case, a magnitude-only case that **must not** be flagged, a null case, a heavy-tail case and a randomized A/B test. Everything derived from them is labelled **SIMULATED** and listed separately in reports, so it can never be mistaken for a discovery about real data.

### Your own data

Open **Upload Data**, choose a file (or try the built-in simulated sample), confirm what each column means, review the preview, and press *Use this dataset*. Every page then works on your data.

- **Formats:** `.csv`, `.tsv`, `.txt` (comma, semicolon, tab or pipe separated), `.xlsx`. Encoding and delimiter are detected.
- **Cleaning is reported, never silent:** renamed headers, removed empty rows and columns, values set to missing because they were not numeric.
- **Dates are never guessed:** ISO dates, times, AM/PM, month names and Excel/Unix numeric dates are understood. Ambiguous dates such as `03/04/2021` stop the pipeline and ask you to choose day-first or month-first.
- **Smart typing:** currency symbols, thousands separators, `(negative)` brackets and decimal commas are understood; ID-like columns and codes with leading zeros stay text so they are never treated as metrics.
- **Line items:** tick one option to combine several rows per order into one row per order.
- **Privacy:** the file stays in your browser session on the machine running the app. It is never written to disk.

See [`docs/UPLOAD_GUIDE.md`](docs/UPLOAD_GUIDE.md) for the full guide and troubleshooting table.

---

## Install and run

Tested with Python 3.13. From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m data.synthetic            # simulated datasets
python -m data.prepare_retail       # one-time, several minutes: cleans and classifies the workbook
python -m data.build_sqlite         # builds data/processed/retail.db and runs 32 SQL/Python parity checks
streamlit run app.py
```

Without the retail file the app still runs on the simulated datasets, and the Upload Data page works on its own.

### Tests

```bash
pytest -v
```

219 tests, including:

- **Known-answer tests** for the Simpson example (overall 25% to 38%, mix effect +0.15, rate effect -0.02, total +0.13)
- A **magnitude-only** case and a **null** case that must not be flagged
- Zero denominators, missing and duplicated data, invalid inputs
- **SQL vs Python parity** on every shared metric, and SELECT-only SQL safety
- **58 ingestion tests** covering encodings, delimiters, Excel, date ambiguity, number formats and column mapping, run on pandas 2.2 and 3.x
- Headless renders of every page on simulated and uploaded data, and a driven upload flow

---

## Architecture

```
app.py                  entry point and navigation
ui/                     logic.py (testable), state.py (caching, upload sessions), theme.py, views/ (seven pages)
core/                   schema, contracts, ingest, data_loader, metrics, contradiction, integrity,
                        cohorts, experiments, sql_store, evidence, reports
viz/                    chart specs (plain JSON) and a themed Plotly builder
data/                   synthetic generator, sample upload file, retail preparation, SQLite build
sql/                    the readable SQL files the app displays and runs
tests/  docs/
```

Design choices:

- **One metric-definition module** is the single source of truth, so "revenue" never means three different things.
- **Every analysis engine is a pure function** over a DataFrame and column names. That is why uploaded data flows through the same engines as the built-in data.
- **SQL does real work:** invoice lines and an `orders` view (one row per sale invoice) make the grain explicit. Window functions (`LAG` for purchase gaps, running totals, cohort assignment) are tested against independent Pandas calculations.
- **Safe SQL:** free-form queries run only on a read-only connection with a SELECT-only authorizer, a row cap and a time limit.
- **Chart specs are plain JSON**, so a saved finding can redraw its chart without the original data.

---

## Results on the real data

These come from actual runs on the Online Retail II data. They are observations, not business conclusions.

- 22.5% of invoice lines have no customer ID, but they carry only 13.1% of gross sales.
- 11,812 rows (1.13%) exactly repeat an earlier row. They are kept and flagged, not confirmed errors.
- Order values are heavily skewed (mean 498.5, median 301.8). The top 1% of orders carry 17.7% of order value.
- The top 10% of identified customers hold 63.8% of identified revenue (bootstrap 95% CI 59.6% to 67.8%), Gini 0.74.
- **Average order value, Q4 2010 vs Q4 2011:** +62.3 (95% CI -14.2 to +138.7, p = 0.110). With a matched window the result is +62.7 (p = 0.112). The verdict is "heterogeneous, no reversal" and no country survives FDR correction. **The rise is not distinguishable from noise.**
- A level-shift candidate in monthly order value (2011-08, p = 0.024) disappeared (p = 0.117) once the incomplete final month was excluded. This is why the tool drops partial periods.
- SQL and Python agree on all 32 parity checks.

---

## Limitations

- Findings are statistical leads, not causal conclusions. Randomization in the Experiment Lab is the analyst's declaration; the tool cannot verify it.
- Standard errors assume independent rows. Orders from the same customer are not independent, so real uncertainty is larger than shown.
- About two years of data: seasonality cannot be separated from trend, and cohorts are left- and right-censored.
- The non-product stock-code heuristic misclassifies a few genuine items (about 0.013% of gross sales).
- The SQL layer and the retail-specific metrics apply to the built-in retail dataset only. Uploaded data uses the Pandas engines.
- Segment p-values are corrected within one analysis, not across every segmentation a user might try.

## Roadmap

- Cluster-robust standard errors for order-level comparisons
- Seasonality-aware period comparisons
- A scanner that tests every metric and segment combination with a global multiple-comparisons correction
- Funnel analysis for uploaded event data

---

## Licence and attribution

Code: MIT, see [`LICENSE`](LICENSE).
Data: the UCI Online Retail II dataset is subject to its own licence and citation requirements; see the UCI repository page.

Built by **Vaibhav Gawali**, a B.Sc. Computer Science graduate focused on data analysis and machine learning.
