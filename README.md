# FinRisk AI

### AI-Powered Financial Analytics & Credit Risk Intelligence Platform

**From Financial Data to Intelligent Insights, Risk Detection and Actionable Decisions**

An interactive analytics platform that turns raw consumer-credit records into
measured findings, risk signals and evidence-linked next steps. Built for the
**AICTE / IBM SkillsBuild Data Analytics with AI Internship 2026** with
**BharatCares**.

```
RAW DATA → VALIDATION → CLEANING → FEATURE ENGINEERING → ANALYTICS
        → KPIs → SEGMENTATION → INSIGHTS → RISK / OPPORTUNITY → ACTION
```

---

## 1. Project overview

FinRisk AI analyses 30,000 credit-card customers across six monthly observation
periods and answers four questions in order:

1. **What is happening?** Adaptive KPIs and portfolio trends.
2. **Why is it happening?** Driver analysis triangulated across four independent
   statistical methods.
3. **What is the risk or opportunity?** Signals measured from historical outcomes,
   plus behaviour-based customer segments.
4. **What action could be considered?** Findings expressed as
   FACT → INSIGHT → IMPLICATION → ACTION, each carrying the numbers it came from.

The analytics engine is fully separated from the dashboard. Nothing under `src/`
(except one caching decorator) imports Streamlit, so every calculation is testable
headlessly and the interface could be replaced without touching the analysis.

---

## 2. Problem statement

A lender holds months of billing, repayment and delinquency records, but in flat
tabular form the data answers almost nothing a credit committee actually asks:
how much is outstanding, who is slipping, **why** some customers default while
similar ones do not, where risk concentrates, and where the safe growth is.

The gap is not a shortage of data. It is the absence of a reproducible path from
raw records to a defensible decision. FinRisk AI closes that gap with an auditable
pipeline whose every displayed number traces back to a function call on real data.

---

## 3. Objectives

- Build a reproducible pipeline from source file to analysis-ready dataset.
- Profile data quality, including *semantic* defects rather than only blanks.
- Engineer credit-behaviour features the raw columns only imply.
- Compute KPIs that adapt to whichever columns genuinely exist.
- Analyse the six-month panel for real period-over-period movement.
- Identify drivers as **associations**, never as proven causes.
- Group customers by measured behaviour, with every methodological choice decided
  by measurement rather than convention.
- Generate findings that cannot be constructed without supporting evidence.
- Present all of it in an interface that reads as a financial product.

---

## 4. Implemented features

### Analytics engine

| Capability | Detail |
|---|---|
| Ingestion | `.xls` / `.xlsx` / `.csv`, evidence-based header detection, encoding fallback |
| Validation | 16 contract checks; errors block, warnings inform |
| Data quality | 4 scored dimensions; outliers reported but deliberately unscored |
| Cleaning | 6 steps, each recording its rationale; **zero rows deleted** |
| Features | 37 engineered, 20 marked model-safe |
| KPIs | 30 registered, adaptively omitted when unsupported |
| EDA | Univariate, bivariate, multivariate, outliers, with small-sample guards |
| Trends | Period-over-period across 6 months; forecasting explicitly refused |
| Drivers | Rank correlation, Mann-Whitney U, chi-square + Cramér's V, tree importance |
| Segmentation | K-Means on 7 behavioural features; scaler and K chosen by measurement |
| Findings | FACT → INSIGHT → IMPLICATION → ACTION with mandatory evidence |

### Dashboard

- Dark financial-intelligence interface with a custom grouped sidebar.
- Compact KPI panels that show a change indicator **only** when a real comparison
  exists.
- Global filters that flow through every KPI, chart and table on a page.
- Empty, loading and error states rendered as product notices, never tracebacks.
- Consistent Plotly theme registered once and inherited by every chart.

---

## 5. Architecture

```
app.py                      Orchestration and routing only
│
├── src/                    Analytics engine (no Streamlit)
│   ├── config.py           Settings from env; secrets read on demand
│   ├── logging_setup.py    Namespaced logging + secret redaction
│   ├── schema.py           Column roles, code maps, provenance
│   ├── data_loader.py      Discovery, loading, validation, profiling
│   ├── data_quality.py     Four-dimension quality assessment
│   ├── data_processing.py  Cleaning with an audit trail; panel reshape
│   ├── feature_engineering.py  37 features with safe division
│   ├── kpi_engine.py       Adaptive KPI registry
│   ├── analytics.py        EDA with insufficient-data guards
│   ├── trends.py           Panel trend analysis
│   ├── drivers.py          Four triangulated driver methods
│   ├── segmentation.py     K-Means clustering, profiling, naming, PCA
│   ├── findings.py         Evidence-enforced finding structure
│   ├── filtering.py        Pure filter logic
│   ├── signals.py          Finding / signal derivation
│   ├── visualization.py    Generic chart primitives
│   ├── pipeline.py         Orchestration (only Streamlit touchpoint)
│   └── ui/                 Presentation layer
│       ├── theme.py        Palette, CSS, Plotly template
│       ├── components.py   Layout primitives
│       ├── kpi_cards.py    KPI panels
│       ├── insight_cards.py Findings, signals, action chains
│       ├── charts.py       Dashboard chart compositions
│       ├── navigation.py   Custom sidebar
│       ├── filter_bar.py   Filter widgets
│       ├── context.py      Per-run page context
│       └── pages/          One module per page
│
├── scripts/                download_data.py, generate_data_dictionary.py
├── tests/                  984 tests
└── data/raw, data/processed
```

**Key design rule.** The page context is assembled **once** per rerun and shared by
every visual. That is what guarantees a filter change moves the whole page rather
than some charts only.

---

## 6. Dataset

| | |
|---|---|
| **Name** | Default of Credit Card Clients |
| **Source** | UCI Machine Learning Repository, dataset 350 |
| **URL** | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| **Licence** | CC BY 4.0 |
| **Size** | 30,000 customers × 25 columns |
| **Period** | April – September 2005 (six monthly observations per customer) |
| **Currency** | New Taiwan Dollar (NT$) |
| **Target** | `default payment next month` (22.12% positive) |

**Citation.** Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining
techniques for the predictive accuracy of probability of default of credit card
clients. *Expert Systems with Applications*, 36(2), 2473–2480.

### What this dataset does not contain

It has **no income, savings, expense, employment or credit-score variable**.
Debt-to-income, savings rate and expense ratio therefore **cannot be computed and
are not approximated**. Credit utilisation is used as the closest valid substitute,
and `LIMIT_BAL` is described as a **credit limit** throughout — never as income and
never as wealth.

---

## 7. Technologies

Python 3.11 · pandas 3.0.6 · numpy · scipy · scikit-learn · Plotly 7.1 ·
Streamlit 1.64 · pyarrow · pytest

All versions are pinned in `requirements.txt` and were verified together, including
pandas 3.x interop under `FutureWarning`-as-error. Segmentation added **no new
dependencies** — scikit-learn was already present.

---

## 8. Installation

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate         # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fetch the dataset (public, no account needed)
python scripts/download_data.py
```

---

## 9. Environment variables

Copy `.env.example` to `.env` and edit as needed. **Every setting has a working
default, so the platform runs with no `.env` at all.**

```powershell
Copy-Item .env.example .env
```

| Variable | Default | Purpose |
|---|---|---|
| `AI_PROVIDER` | `none` | `none` \| `openai` \| `gemini` \| `ollama` |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` | empty | Only read for that provider |
| `RANDOM_STATE` | `42` | Seed for reproducibility, including clustering |
| `MIN_ROWS_FOR_ANALYSIS` | `30` | Below this, analyses are suppressed |
| `HIGH_UTILISATION_THRESHOLD` | `0.80` | Drives the utilisation band and KPI |
| `OUTLIER_IQR_MULTIPLIER` | `1.5` | IQR fence for outlier reporting |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |

`.env` is git-ignored. No key is ever written to a log: a redaction filter masks
credential-shaped strings as a second line of defence.

---

## 10. Running the project

### Launch the dashboard

```powershell
.\run_dashboard.ps1
```

The browser opens on http://localhost:8501. Use `-Port 8502` for a different port
and `-Headless` to start the server without opening a browser.

This launcher exists because `streamlit` is installed **inside `.venv`, not on the
system PATH**. From a shell where the environment is not active, the bare command
fails:

```text
streamlit : The term 'streamlit' is not recognized as the name of a cmdlet ...
```

The launcher resolves the interpreter in `.venv` explicitly, so no activation step
is needed. If you prefer the standard command, activate the environment first:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Or call the interpreter directly, without activating:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

> **Note on `.streamlit/config.toml`:** `server.headless` is deliberately left
> unset. Setting it to `true` there suppresses the browser launch for *every* run,
> so the server starts and serves normally — health returns 200 — while no window
> ever appears. Automated runs pass `--server.headless true` on the command line
> instead.

> **First run:** Streamlit prompts for an email address on its first interactive
> run and waits on stdin, which can look like a hang. `run_dashboard.ps1` declines
> the prompt once by seeding `~/.streamlit/credentials.toml` with an empty email.
> To clear an address already stored there, edit that file.

### Other entry points

```powershell
.\.venv\Scripts\python.exe -m src.pipeline                     # run pipeline, print summary
.\.venv\Scripts\python.exe scripts\download_data.py --verify   # verify local dataset
.\.venv\Scripts\python.exe scripts\generate_data_dictionary.py # regenerate data_dictionary.md
.\.venv\Scripts\python.exe -m pytest                           # full test suite
.\.venv\Scripts\python.exe -m pytest -m "not integration"      # unit tests only
```

### When a section reports "Analysis unavailable"

The dashboard renders failures as product notices rather than tracebacks
(`client.showErrorDetails = false`). The full traceback is never discarded — read
it in `logs/finrisk.log`.

---

## 11. Dashboard pages

| Page | Status | Contents |
|---|---|---|
| **Executive Overview** | ✅ | KPI row, Credit Behaviour Overview across six periods, Key Intelligence, Risk Signals, Opportunities, Recommended Next Steps |
| **Financial Analytics** | ✅ | Filter bar plus four tabs: Credit Behaviour, Payment Behaviour & Delinquency, Driver Analysis, Customer Profile |
| **Customer Segmentation** | ✅ | K-Means behavioural segments, measured K selection, PCA behaviour map, segment profiles and interpretation |
| **Data Explorer** | ✅ | Dataset overview, column information, summary statistics, searchable paginated table, CSV export |
| **Data Quality** | ✅ | Score with dimension breakdown, real semantic findings, extreme values, cleaning audit trail |
| **Methodology** | ✅ | Pipeline diagram, provenance, panel structure, KPI formulas, feature definitions, limitations |
| Credit Risk | ⬜ | Predictive model, explainability, fairness audit — risk-model phase |
| AI Insights | ⬜ | LLM narration over computed results — AI phase |
| Recommendations | ⬜ | Full rule library — recommendation phase |

Unimplemented pages state their planned scope. **They show no mock charts and no
sample numbers**, so everything visible elsewhere can be trusted as real.

---

## 12. KPI definitions

30 KPIs are registered; each is omitted entirely, with a recorded reason, when its
source columns are absent. Headline definitions:

| KPI | Formula | Notes |
|---|---|---|
| **Total Customers** | count of distinct `client_id` | |
| **Observed Default Rate** | `mean(default_next_month) × 100` | Recorded historical outcome, **not** a prediction |
| **Average / Median Credit Limit** | `mean` / `median(credit_limit)` | A **credit limit**, not income. Compare both: the distribution is right-skewed |
| **Total Credit Exposure Limit** | `sum(credit_limit)` | Maximum owed if every limit were fully drawn |
| **Total / Average Outstanding Balance** | `sum` / `mean(bill_amt_m1)` | Most recent statement |
| **Average Credit Utilisation** | `mean(bill_amt_m1 ÷ credit_limit) × 100` | **Derived analytical measure.** Closest available substitute for debt-to-income |
| **High-Utilisation Customers** | `count(utilisation_latest > threshold) ÷ total` | Threshold from `HIGH_UTILISATION_THRESHOLD` (0.80) |
| **Over-Limit Customers** | `count(utilisation_max_6m > 1.0) ÷ total` | Balance exceeded the limit in ≥1 month |
| **Median Repayment Ratio** | `median` of each customer's median `pay_amt_m<m> ÷ bill_amt_m<m+1>` | Payment settles the **previous** month's statement. Median used because the raw ratio is unbounded |
| **Average Repayment Ratio (capped)** | `mean(repayment_ratio_capped_mean) × 100` | Each month clipped at 200% so a near-zero prior balance cannot distort the average |
| **Full Payers** | `count(repayment_ratio_median ≥ 0.95) ÷ total` | Transactors rather than borrowers |
| **Revolving Borrowers** | `count(repayment_ratio_median < 0.30) ÷ total` | Carry a balance forward |
| **Currently Delinquent** | `count(current_delinquency ≥ 1) ÷ total` | Most recent observed month |
| **Ever Delinquent** | `count(any pay_status_m* ≥ 1) ÷ total` | Any of the six months |
| **Average Delinquent Months** | `mean(delinquent_months_count)` | Frequency, out of six |
| **Average Worst Delay** | `mean(max_delinquency)` | Severity, in months |
| **Median Balance Trend** | `median(bill_trend_slope)` | NT$ per month across the six periods |

**Persistent delinquency** is defined as the count of monthly observations with a
documented delay code of 1 or more, out of the six periods held per customer. Codes
`-2` and `0` occur frequently but are **not defined** in the source paper, so
delinquency measures deliberately exclude them.

Not shown, because the columns do not exist: average income, median income,
debt-to-income, savings rate, expense ratio.

---

## 13. Analytics implemented

**Univariate** — distributions, frequency tables, summary statistics with a
skewness interpretation.
**Bivariate** — Spearman correlation (default, because monetary columns are heavily
skewed), group comparison, category outcome rates, Cohen's *d*.
**Multivariate** — cross-tabulation, segment profiling.
**Outliers** — IQR and z-score, with a MAD fallback when the IQR collapses.
**Trends** — mean/median/total balance, payments, utilisation, delinquency rate and
active accounts across six periods, with direction, change, and a MAD-based anomaly
flag.
**Drivers** — four methods triangulated, ranked by cross-method agreement then
effect size.
**Segmentation** — K-Means on behavioural features, with scaler and K chosen by
measurement (see §13a).

---

## 13a. Customer segmentation methodology

Unsupervised K-Means clustering that groups customers by **measured credit
behaviour**, then reports separately what outcome each group experienced.

### Features used (7)

`credit_limit` · `utilisation_mean_6m` · `repayment_ratio_capped_mean` ·
`delinquent_months_count` · `bill_trend_slope` · `utilisation_volatility` ·
`months_zero_payment`

### Features excluded, and why

| Excluded | Reason |
|---|---|
| `default_next_month` | **Outcome leakage.** Including it would make the segments a restatement of the answer rather than a description of behaviour. |
| `sex`, `marriage`, `education`, `age` | **Fairness.** Segmenting a credit portfolio on protected attributes is discriminatory; excluded from every modelling path in this project. |
| `client_id`, bookkeeping flags | Identifiers carry no behavioural signal. |

Requesting any of these raises an error rather than silently succeeding. A
permutation test in the suite shuffles the target and confirms the cluster
assignments do not move.

### Scaler selected by measurement

| Scaler | Best K | Silhouette | Davies-Bouldin | Smallest segment |
|---|---|---|---|---|
| Standard | 6 | 0.2879 | 1.2033 | 4.74% |
| **Robust (adopted)** | **3** | **0.4035** | **1.0001** | **12.07%** |

Robust scaling won on both metrics. The clustering features are strongly skewed
(utilisation volatility 3.43, balance trend 2.35), and standard scaling divides by
a standard deviation that a long tail inflates, compressing the bulk of the
distribution into a narrow band. The choice is empirical, not conventional.

### K selected by measurement

Silhouette is the primary criterion, supported by the inertia elbow and
Davies-Bouldin. Swept K = 2…8:

| K | Silhouette | Davies-Bouldin | Smallest segment |
|---|---|---|---|
| 2 | 0.3525 | 1.2222 | 17.14% |
| **3** | **0.4035** | **1.0001** | **12.07%** |
| 4 | 0.3350 | 1.0388 | 2.85% |
| 5 | 0.3508 | 0.9805 | 1.83% |
| 6 | 0.3014 | 1.0373 | 1.81% |
| 7 | 0.2698 | 1.0755 | 1.76% |
| 8 | 0.2832 | 1.0882 | 1.48% |

Silhouette peaks at **K = 3**. The elbow sits at K = 4 and Davies-Bouldin is best at
K = 5, so the criteria disagree — normal for continuous behavioural data with no
naturally separated groups. The full curve is displayed so the disagreement is
visible rather than hidden. K ≥ 4 also yields segments under 3%, which describe
outliers rather than usable populations.

### Feature set chosen on redundancy, not silhouette

Dropping the two trajectory features leaves five, which scores a **higher**
silhouette (0.5672 at its own best K = 2, against 0.4035 for seven at K = 3) yet
collapses the book into a coarser split: its largest segment holds 87.5% of
customers against 72.6% for the seven-feature set. **Silhouette rises mechanically
as dimensionality falls**, so it is only comparable across K within a fixed feature
space — never across feature spaces of different size. The set was chosen instead
on redundancy (strongest pair correlation 0.756, below the 0.80 threshold),
behavioural coverage, and partition balance.

The reduced set used for this comparison is named in code as
`COMPARISON_FEATURE_SUBSET`, so the figures above can be reproduced rather than
taken on trust.

### Silhouette is scored on a stratified sample

Silhouette is quadratic in row count, so on 30,000 customers it is evaluated on a
fixed-seed sample of ~5,000 rather than the full book. The sample is **stratified by
segment**: each segment contributes a share proportional to its real size, floored
at two rows. A plain random subset can miss a small segment entirely, leaving one
label and an undefined score — the stratified sample makes that outcome impossible
by construction instead of catching it after the fact. Inertia, Davies-Bouldin and
Calinski-Harabasz use all rows. When a partition genuinely cannot be scored the
metric is reported as unavailable; no value is substituted.

### Missing values

`repayment_ratio_capped_mean` is undefined for **1,406 customers (4.69%)** whose
prior statements were all zero or negative. Dropping them would bias the
segmentation toward active accounts, so the median is imputed and disclosed. An
undefined ratio usually means an inactive or in-credit account rather than poor
repayment, and an imputed value is **not** an observed customer value.

### Segments discovered

| Segment | Customers | Share | Observed default rate |
|---|---|---|---|
| Mid-Range / Typical-Behaviour | 21,775 | 72.6% | 16.51% |
| Growing-Balance / Volatile-Usage | 4,604 | 15.3% | 18.68% |
| Delinquent / Low-Repayment | 3,621 | 12.1% | **60.20%** |

Names are composed at runtime from each centroid's standardised distance from the
portfolio mean, so every word traces to a measured value. No judgemental labels
("good", "bad", "safe" customer) are used anywhere.

The 43.7-point spread in observed default rate is notable precisely **because the
outcome was excluded from clustering**. Behaviour alone separates the portfolio
that sharply; the rate was measured afterwards and is descriptive only.

### PCA limitation

The behaviour map is a 2D PCA projection **for visualisation only**. K-Means ran on
the full 7-dimensional scaled matrix. The two components retain **71.9%** of the
variance (44.56% + 27.37%), so roughly 28% of the structure the model used is not
visible. Points that appear to overlap may be well separated on the discarded
dimensions — the silhouette and Davies-Bouldin scores are the evidence about
cluster quality, not the plot.

### Segmentation limitations

- Unsupervised and descriptive: it assigns groups, it does not score, rank or
  predict any individual.
- Observed default rates are measured **after** clustering and are descriptive
  only. No causal link between membership and outcome is claimed.
- **72.6% of customers fall into one undifferentiated segment.** This is a genuine
  property of the result: behaviour here is largely continuous rather than
  naturally clustered, so a clean split of the whole book does not exist. The page
  states this directly and offers a K selector to compare alternatives.
- K-Means assumes roughly spherical, similarly sized clusters, which credit
  behaviour only partly satisfies.
- Fitted on the **whole portfolio**, deliberately. Segments are a property of the
  book; refitting per filter would make them shift underfoot and prevent
  comparison between views. When filters are active the page shows how the
  selection distributes across the fixed segments.

---

## 14. Key insights from the data

All measured, all reproducible:

- **Observed default rate 22.12%** (6,636 of 30,000). Predicting "no default" for
  everyone would score 77.88% accuracy while finding no defaulters — which is why
  accuracy is not used as a headline metric.
- **Payment delay is the sharpest single split.** Customers with no recorded delay
  default at **13.83%**; those 3+ months behind default at **71.92%** — roughly
  5.2×.
- **The portfolio deteriorated across the window.** Mean balance rose 31.8%
  (NT$38,872 → NT$51,223), utilisation rose 33.0% (31.9% → 42.4%), and the share
  behind on payment rose **121.4%** (10.26% → 22.73%).
- **Default falls monotonically as credit limit rises**: 31.79% → 24.72% → 17.35% →
  13.98% across quartile bands.
- **Utilisation matters, but not linearly.** Low (0–30%) defaults at 16.93% versus
  27.25% at high (60–80%), then plateaus. Accounts in credit are reported
  separately, since they are not low-utilisation borrowers.
- **62.4% are revolving borrowers**; **26.8%** typically clear their statement and
  default at 14.69% against 22.12% overall.
- **Behaviour alone separates outcome by 43.7 points.** Clustering on behaviour
  with the outcome excluded still produced segments ranging from 16.51% to 60.20%
  observed default.

---

## 15. Limitations

- **Historical and region-specific.** Taiwan, April–September 2005.
- **No income data**, so no debt-to-income or income banding (see §6).
- **Six periods only.** Period-over-period comparison is valid; forecasting,
  seasonality decomposition and daily/weekly resampling are not offered.
- **Associations, not causes.** Observational data; no causal claim is made.
- **No predictive model yet.** Everything labelled a risk signal, and every
  segment default rate, is a measured historical outcome rather than a prediction.
- **Narrow target.** Default on the single following month's payment.
- **Fairness.** Sex and marital status are excluded from driver analysis, from
  clustering, and from planned model features; they appear only in descriptive and
  fairness views.
- **Quality score is project-defined**, not DAMA or ISO 8000.

---

## 16. Testing

```powershell
python -m pytest                       # 984 tests
python -m pytest -m "not integration"  # skip tests needing the dataset
```

Coverage includes loading edge cases, division-by-zero guards, KPI
hand-verification against the raw file, filter propagation, empty and invalid
filter combinations, banned-language checks that fail if causal or prediction
wording leaks into generated text, chart empty states, segmentation determinism and
**target-leakage prevention** (including a permutation test), and full page
rendering through Streamlit's `AppTest`.

---

## 17. Future scope

1. Credit-risk model with ROC-AUC/PR-AUC, tuned threshold and fairness audit.
2. Explainability via permutation importance and optional SHAP.
3. AI narration over computed results, with numeric post-validation.
4. Full recommendation rule library.
5. Alternative clustering approaches (Gaussian mixtures, HDBSCAN) to test whether
   the dominant mid-range segment subdivides meaningfully under a method that does
   not assume spherical clusters.
6. PostgreSQL data source behind the existing interface.
7. Docker packaging.

---

## 18. Author

B.Tech CSE (Artificial Intelligence & Machine Learning) student.
Submitted for the **AICTE / IBM SkillsBuild Data Analytics with AI Internship
2026** in partnership with **BharatCares**.

---

## 19. Data source attribution

Dataset: [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients),
UCI Machine Learning Repository (dataset 350), licensed **CC BY 4.0**.
Original research: Yeh & Lien (2009), *Expert Systems with Applications* 36(2),
2473–2480.
