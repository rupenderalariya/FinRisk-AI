# PROJECT_PLAN.md

# FinRisk AI

### AI-Powered Financial Analytics & Credit Risk Intelligence Platform

**"From Financial Data to Intelligent Insights, Risk Detection and Actionable Decisions"**

| | |
|---|---|
| **Programme** | AICTE / IBM SkillsBuild Data Analytics with AI Internship 2026 |
| **Partner Organisation** | BharatCares |
| **Track** | Data Analytics with AI — Foundation to Implementation |
| **Plan status** | Phase 1 complete — awaiting approval before implementation |
| **Plan date** | 21 September 2026 |

---

## 0. What has actually been verified so far

This plan is grounded in commands that were really executed in this workspace. Nothing below is assumed.

**Workspace**

| Check | Result |
|---|---|
| Workspace path | `c:\Users\rupen\Desktop\Project\FinRisk Ai` |
| Existing files | Empty — no prior code, no dataset, no notebooks |
| Python | 3.11.9 (`C:\Users\rupen\AppData\Local\Programs\Python\Python311\python.exe`) |
| Installed packages at start | `certifi`, `charset-normalizer`, `idna`, `pip 26.2.1`, `requests`, `setuptools`, `urllib3` — a clean interpreter |
| Node.js | v24.13.0 (available, not required by the chosen stack) |
| Git | 2.51.1.windows.1 |
| OS / Shell | Windows / PowerShell |

**Installed during Phase 1 diligence** (needed to read the candidate dataset): `pandas 3.0.6`, `numpy`, `xlrd 2.0.2`, `openpyxl`.

> ⚠️ **Open decision for Phase 2:** pip resolved **pandas 3.0.6**, which is very new. Streamlit, Plotly, and SHAP may not yet be fully compatible with the pandas 3.x API. Phase 2 will create a **virtual environment** and empirically test whether the full stack works on pandas 3.x; if it does not, `requirements.txt` will pin `pandas>=2.2,<3.0`. This will be resolved by testing, not by guessing.

---

## 1. Problem Statement

A consumer-credit lender holds months of billing, repayment, and delinquency records for tens of thousands of card-holders, but the data sits as a flat table of coded columns. In that form it answers almost nothing a credit committee actually asks:

- How much balance is outstanding, and is the book growing or shrinking?
- Which clients are already slipping, and how badly?
- **Why** do some clients default while others with similar limits do not?
- Which behavioural groups concentrate the risk?
- Which clients are low-risk and under-served, i.e. where is the growth?
- What should be done next week, and on what evidence?

The gap is not a shortage of data. It is the absence of a reproducible path from raw records to a defensible decision. Manual spreadsheet work is slow, inconsistent between analysts, and offers no way to weigh one driver against another.

**FinRisk AI closes that gap** with an auditable pipeline that carries raw credit records through cleaning, EDA, KPIs, behavioural segmentation, a predictive risk model, model explanation, and finally to written, evidence-linked recommendations — presented in one interactive dashboard.

---

## 2. Objectives

**Primary**

1. Build a reproducible pipeline from raw source file to analysis-ready dataset, with every transformation expressed in code.
2. Profile data quality and surface it in the UI — including *semantically* invalid values, not just blanks.
3. Engineer credit-behaviour features that the raw columns only imply (utilisation, repayment behaviour, delinquency persistence, balance trajectory).
4. Compute a KPI layer that adapts to whichever columns are genuinely present.
5. Analyse the 6-month panel structure for real trends in balances, repayments, and delinquency.
6. Identify drivers statistically, and describe them as *associations*, never as proven causes.
7. Segment clients by behaviour with unsupervised learning, then interpret the clusters from their measured profiles.
8. Train and honestly evaluate a credit-default classifier, using metrics fit for an imbalanced target.
9. Explain the model globally and for individual clients.
10. Generate recommendations in a strict **FACT → INSIGHT → RISK/OPPORTUNITY → ACTION** chain, each traceable to a computed number.
11. Add an optional LLM narration layer that receives *only* computed results — and degrade cleanly to analytics-only when no API key is present.

**Secondary**

12. Ship a UI that reads as a financial-analytics product, not a student demo.
13. Handle errors so that no raw Python traceback ever reaches the user.
14. Audit the model for demographic bias and document the finding.
15. Produce README, data dictionary, and project report good enough for both academic submission and a portfolio.

**Explicit non-objectives**

- Not a loan-origination or underwriting system.
- Not a real-time transaction scorer.
- Not a source of financial advice. Outputs are analytical findings for review by a qualified human.

---

## 3. Target Users

| User | What they need | Where the app serves them |
|---|---|---|
| Credit Risk Manager | Where risk is concentrating and why | Credit Risk, Driver Analysis |
| Portfolio / Collections Lead | Which behavioural groups to prioritise | Segmentation, Recommendations |
| Business / BI Analyst | Trustworthy KPIs and trends, plus the data behind them | Executive Overview, Data Explorer, Financial Analytics |
| Executive / Sponsor | A one-screen answer and a short narrative | Executive Overview, AI Insights |
| Data Scientist / Reviewer | Model validity, leakage checks, fairness, reproducibility | Credit Risk, Methodology |
| Internship Evaluator | Evidence of the full analytics lifecycle | Methodology + PROJECT_REPORT.md |

---

## 4. Dataset Strategy

### 4.1 Constraints applied

The internship brief requires a dataset **different from the masterclass datasets**, publicly available, legitimately sourced, with a recorded source URL. It must be rich enough for real analytics, and must not be fabricated. Reproducibility was weighted heavily: a marker must be able to fetch the same data with one command.

### 4.2 Candidates evaluated

Three credit-domain candidates were assessed. Availability was tested with real HTTP requests.

**Candidate A — Default of Credit Card Clients (UCI ID 350), Taiwan, 2005**
`https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients`
Direct download returned **HTTP 200**, 5,539,494 bytes. **Fully downloaded and inspected** — every figure in §4.4 is measured.

**Candidate B — Give Me Some Credit (Kaggle, 2011)**
`https://www.kaggle.com/c/GiveMeSomeCredit`
~150,000 rows, 10 predictors, target `SeriousDlqin2yrs`, ~6.7% positive. An unauthenticated request to the data page returned **HTTP 404** — Kaggle competition data requires a logged-in account and acceptance of the competition rules, and the 2011 competition is closed. Row counts here are from public documentation; **this dataset was not downloaded.** It is flat (one snapshot per borrower), so it supports **no** temporal analysis.

**Candidate C — Statlog German Credit Data (UCI ID 144)**
`https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data`
Direct download returned **HTTP 200**. 1,000 rows / 20 attributes per UCI documentation; **not downloaded.** At 1,000 rows it is thin for segmentation plus a held-out test set, and it is the single most-used teaching dataset in credit risk, which hurts originality.

### 4.3 Comparison

Scores are 1–5, reasoned rather than measured, except where §0/§4.4 gives a hard number.

| Criterion | A — UCI Credit Card Default | B — Give Me Some Credit | C — German Credit |
|---|:--:|:--:|:--:|
| Rows | **30,000** (verified) | ~150,000 | 1,000 |
| Columns | **25** (verified) | 12 | 21 |
| Data quality | **5** — 0 nulls, 0 exact dupes (verified) | 3 — real nulls in income & dependents | 4 |
| Analytics potential | **5** — 6-month panel, utilisation, repayment | 3 — single snapshot | 2 |
| **Temporal analysis** | **5** — 6 monthly periods per client | **1** — none possible | **1** — none possible |
| ML potential | **5** — 22.1% positive class (verified) | 5 — but severe 6.7% imbalance | 3 — small sample |
| Explainability value | **5** — features map to real credit levers | 4 | 4 |
| Visualisation potential | **5** — panel + demographic + behavioural | 3 | 3 |
| Originality | **4** — common, but panel-based feature work is not | 3 | **1** — heavily overused |
| **Reproducibility** | **5** — verified one-command public download | **2** — login + rules acceptance | 5 |
| Internship suitability | **5** | 3 | 3 |
| **Total** | **49** | 30 | 29 |

### 4.4 Recommendation — Candidate A

**Default of Credit Card Clients (UCI ID 350).** Measured properties from the real file:

| Property | Verified value |
|---|---|
| Shape | **30,000 rows × 25 columns** |
| Missing values | **0** across all columns |
| Exact duplicate rows | **0** |
| Duplicates ignoring `ID` | **35** |
| Target | `default payment next month` |
| Class balance | 23,364 non-default / **6,636 default** |
| Observed default rate | **22.12%** |
| `LIMIT_BAL` | NT$10,000 – 1,000,000; mean 167,484; median 140,000 |
| `AGE` | 21 – 79; mean 35.49; median 34 |
| `BILL_AMT1` | −165,580 – 964,511; median 22,381.5 |
| Negative `BILL_AMT` values | 590 / 669 / 655 / 675 / 655 / 688 across months 1–6 |
| `PAY_0..PAY_6` observed codes | −2, −1, 0, 1 … 8 |
| `EDUCATION` observed codes | 0, 1, 2, 3, 4, 5, 6 |
| `MARRIAGE` observed codes | 0, 1, 2, 3 |
| Period covered | April – September 2005 (per UCI documentation) |
| Currency | New Taiwan Dollar (NT$) |
| File format | `.xls`, header on the **second** row (row 1 is a merged banner) |

**Why it wins.** It is the only candidate with a genuine time dimension: `BILL_AMT1..6`, `PAY_AMT1..6`, and `PAY_0..PAY_6` are six monthly observations per client. That single property makes Trend Analysis, balance-trajectory features, and delinquency-persistence features real work on real data rather than a fabricated time axis. A 22.12% positive rate is imbalanced enough to demand proper metrics but healthy enough to train on. And the download is public and verified, so the whole project reproduces from a single command.

### 4.5 Honest limitations of this dataset — and how the plan adapts

This is the most important section of the plan. The master brief lists example features such as Debt-to-Income, Savings Rate, and Expense Ratio. **This dataset contains no income, savings, expense, employment, or credit-score column.** Those features therefore **cannot be computed and will not be faked.**

| Brief suggests | Available? | Plan's response |
|---|---|---|
| Debt-to-Income | ❌ no income column | **Not computed.** Replaced by Credit Utilisation = balance ÷ credit limit, which is real. |
| Savings Rate | ❌ no savings column | **Not computed.** UI states the field is absent. |
| Expense Ratio | ❌ no expense column | **Not computed.** |
| Credit Utilisation | ✅ `BILL_AMT` ÷ `LIMIT_BAL` | **Computed** — a first-class feature. |
| Payment history | ✅ `PAY_0..PAY_6` | **Computed** — delinquency count, max severity, persistence. |
| Credit score | ❌ absent | **Not computed.** The model's predicted probability serves as an internal risk score and is labelled as model output. |
| Employment status | ❌ absent | Omitted. |

Further points to be carried into the UI and the report, not buried:

1. **`LIMIT_BAL` is a credit limit, not income.** It will be used as a coarse proxy for *credit capacity* and always labelled as such. The "risk by income group" view in the brief becomes **risk by credit-limit band**.
2. **Zero missing values changes the data-quality story.** A module that only counts nulls would report a perfect score and be useless. So the Data Quality module targets *semantic* defects that genuinely exist here: undocumented `EDUCATION` codes (0, 5, 6), undocumented `MARRIAGE` code (0), `PAY_*` codes −2 and 0 which the source paper does not define, negative `BILL_AMT` values (real credit balances/overpayments — flagged, not deleted), and the 35 rows that duplicate once `ID` is dropped.
3. **Panel, not a continuous series.** Six monthly snapshots with no calendar date column. Trend analysis is therefore period-over-period comparison across months 1–6. **No forecasting will be offered** — six points cannot support it.
4. **Demographic attributes are a fairness hazard.** `SEX`, `MARRIAGE`, `EDUCATION`, and `AGE` are present. Using `SEX` or `MARRIAGE` to price credit risk is discriminatory and in many jurisdictions unlawful. **Default behaviour: protected attributes are excluded from model features** and used *only* for a bias audit that compares error rates across groups. This is a deliberate design choice and will be documented.
5. **2005 Taiwan data.** Findings are historical and region-specific; they do not transfer to present-day portfolios.
6. **Behavioural labels are not real-world default certainty.** The target is "defaulted on the next month's payment", a narrow definition.

### 4.6 Provenance

- **Source:** UCI Machine Learning Repository, Dataset 350 — Default of Credit Card Clients
- **Landing page:** `https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients`
- **Direct file:** `https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip`
- **Original citation:** Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480.
- **Licence:** UCI ML Repository, CC BY 4.0 — permits use with attribution.
- **Local raw copy:** `data/raw/default of credit card clients.xls` (already downloaded; `data/raw/` will be git-ignored, and a `scripts/download_data.py` will re-fetch it on demand)

---

## 5. Architecture

Layered, one direction of dependency: UI → services → domain logic → data. No Streamlit import below the UI layer, which keeps the analytics importable from tests, notebooks, and the final consolidated submission file.

```
┌───────────────────────────────────────────────────────────────┐
│  PRESENTATION — Streamlit (app.py + pages/)                   │
│  7 pages · sidebar filters · KPI cards · Plotly charts        │
│  loading + empty + error states                               │
└───────────────────────────┬───────────────────────────────────┘
                            │  cached calls only
┌───────────────────────────▼───────────────────────────────────┐
│  ORCHESTRATION — src/pipeline.py                              │
│  runs load→clean→engineer once, cached via st.cache_data      │
└───────────────────────────┬───────────────────────────────────┘
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
┌───────────────┐  ┌─────────────────┐  ┌────────────────────┐
│ DATA          │  │ ANALYTICS       │  │ INTELLIGENCE       │
│ data_loader   │  │ kpi_engine      │  │ risk_model         │
│ data_quality  │  │ analytics       │  │ explainability     │
│ data_processing│ │ trends          │  │ segmentation       │
│ feature_eng   │  │ drivers         │  │ recommendations    │
│ schema        │  │                 │  │ ai_insights        │
└───────┬───────┘  └────────┬────────┘  └─────────┬──────────┘
        │                   │                     │
        ▼                   ▼                     ▼
┌───────────────┐   ┌──────────────┐   ┌──────────────────────┐
│ data/raw      │   │ models/*.pkl │   │ LLM provider         │
│ data/processed│   │              │   │ (optional, env-cfg)  │
└───────────────┘   └──────────────┘   └──────────────────────┘
```

**Rules enforced during implementation**

- `src/` never imports `streamlit`, except `pipeline.py` for its cache decorators.
- Every module is pure-function-first: dataframe in, dataframe or dataclass out.
- A `schema.py` holds a single `ColumnRegistry` describing which logical roles are present, so KPIs and features can ask *"do I have a balance column?"* instead of hard-coding `BILL_AMT1`. This is what makes the KPI engine genuinely adaptive.
- No analytical result is ever hard-coded. Numbers shown in the UI come from a function call.
- The LLM layer receives a compact JSON of computed metrics, never the dataframe.

---

## 6. Technology Stack

| Layer | Choice | Reasoning |
|---|---|---|
| Language | Python 3.11.9 | Already present; broad library support |
| Dataframes | pandas, numpy | Standard; version pinned in Phase 2 after a compatibility test |
| Excel read | xlrd (`.xls`) | Required for this specific legacy format; source is `.xls` |
| ML | scikit-learn | Covers Logistic Regression, Random Forest, Gradient Boosting, KMeans, metrics, pipelines |
| Statistics | scipy | Chi-square, Mann-Whitney for group comparison |
| Explainability | sklearn permutation importance (**required**), SHAP (**optional**) | SHAP is valuable but heavy and version-sensitive. It will be attempted and wrapped in a graceful fallback to permutation importance — never a hard dependency. |
| Charts | Plotly | Interactive, tooltips, works natively in Streamlit. Matplotlib only for static report exports; seaborn omitted as unnecessary. |
| App | **Streamlit** | Confirmed as the right call — this is an analytics product, the whole stack stays Python, and a marker runs it with one command. A React/FastAPI split would double the code and add no analytical value. |
| Persistence | CSV/Parquet + joblib | Simple, inspectable. A `DataSource` interface keeps a future PostgreSQL swap cheap. |
| Config | python-dotenv | `.env` for secrets, `.env.example` committed |
| LLM | Provider abstraction: OpenAI / Gemini / Ollama | Selected by `AI_PROVIDER`; app fully functional with none configured |
| Tests | pytest | Standard |

---

## 7. Data Pipeline

```
data/raw/default of credit card clients.xls   (30,000 × 25, verified)
        │
        ▼ [1] LOAD — read_excel(header=1); normalise names to snake_case;
        │        rename "default payment next month" → default_next_month
        ▼ [2] VALIDATE — assert expected columns & row count; fail loud with a clear message
        ▼ [3] PROFILE QUALITY — nulls, dupes, dtypes, invalid codes, outliers → DataQualityReport
        ▼ [4] CLEAN — map coded ints to labels (sex, education, marriage);
        │        fold undocumented education 0/5/6 → "unknown"; marriage 0 → "unknown";
        │        keep negative bills (valid credit balances) but flag them
        ▼ [5] RESHAPE — build a tidy long panel: (client_id, month 1..6, bill, pay, pay_status)
        ▼ [6] ENGINEER — utilisation, repayment ratio, delinquency, trajectory  (§8)
        ▼ [7] PERSIST — data/processed/clients_wide.parquet + panel_long.parquet
        ▼ [8] ANALYSE — EDA · KPIs · trends · drivers · segmentation · model
        ▼ [9] SERVE — Streamlit dashboard
```

Determinism: fixed `RANDOM_STATE = 42` everywhere, stratified splits, no dependence on row order. Running the pipeline twice yields byte-identical processed files.

---

## 8. Feature Engineering

Each feature below is derivable from columns confirmed to exist. Month index 1 = September 2005 (most recent) through 6 = April 2005, matching the source's numbering.

**Utilisation** — real, and the core capacity signal
- `utilisation_m{1..6}` = `bill_amt{m}` / `limit_bal`
- `utilisation_latest`, `utilisation_mean_6m`, `utilisation_max_6m`
- `is_high_utilisation` = `utilisation_latest > 0.80`

**Repayment behaviour**
- `repayment_ratio_m{m}` = `pay_amt{m}` / `bill_amt{m+1}` (payment made against the *previous* month's statement — the correct alignment)
- `repayment_ratio_mean`, `months_zero_payment` (count of `pay_amt == 0`)
- `is_revolver` — consistently pays a small fraction of the balance
- `is_full_payer` — consistently clears the statement

**Delinquency, from `pay_0..pay_6`** (code ≥ 1 means months of payment delay)
- `delinquent_months_count` = number of periods with code ≥ 1
- `max_delinquency` = worst code observed
- `current_delinquency` = `pay_0`
- `ever_delinquent`, `delinquency_trend` (worsening / stable / improving)

**Balance trajectory** — enabled purely by the panel structure
- `bill_trend_slope` = OLS slope of balance across the 6 months
- `bill_volatility` = standard deviation of balances
- `balance_growth_6m` = (`bill_amt1` − `bill_amt6`) / |`bill_amt6`|
- `avg_bill_6m`, `total_paid_6m`

**Capacity band (proxy, clearly labelled)**
- `limit_band` = quantile bucket of `limit_bal` — described everywhere as *credit-limit band*, never as income
- `age_band` = 21–29 / 30–39 / 40–49 / 50–59 / 60+

Guards: every ratio uses safe division (zero or negative denominators → NaN, never `inf`); each feature function checks its inputs exist via the column registry and is skipped with a logged reason if not.

---

## 9. KPI Engine

A registry of `KPI(name, fn, required_roles, format, help)` objects. The engine evaluates only those KPIs whose required columns are present, so an unsupported KPI is silently omitted rather than rendered as `N/A`.

| KPI | Basis | Status |
|---|---|---|
| Total Clients | row count | ✅ verified 30,000 |
| Observed Default Rate | mean of target | ✅ verified 22.12% |
| Average / Median Credit Limit | `limit_bal` | ✅ verified 167,484 / 140,000 |
| Total Outstanding Balance | Σ `bill_amt1` | ✅ |
| Average Credit Utilisation | engineered | ✅ |
| High-Utilisation Client Share | > 80% | ✅ |
| Total Repayments (6m) | Σ all `pay_amt` | ✅ |
| Average Repayment Ratio | engineered | ✅ |
| Currently Delinquent Share | `pay_0 ≥ 1` | ✅ |
| Ever-Delinquent Share | any `pay_* ≥ 1` | ✅ |
| Average Max Delinquency | engineered | ✅ |
| Average Age | `age` | ✅ verified 35.49 |
| Portfolio Balance Trend | panel slope | ✅ |
| High-Risk Client Count | model probability ≥ tuned threshold | ✅ — **labelled "model-derived"** |
| Average Income / Savings Rate / DTI | — | ❌ **omitted, no source column** |

Every KPI card carries a tooltip with its formula. All currency displays as NT$.

---

## 10. Analytics Methodology

**EDA** — distributions for all numerics, frequency tables for categoricals, correlation matrix (Spearman alongside Pearson, since the money columns are heavily skewed — verified: `bill_amt1` mean 51,223 vs median 22,382), outlier detection via IQR and z-score reported *without* silent deletion.

**Trend analysis** — across the six periods: mean balance, total repayment, mean utilisation, delinquency rate. Reported with direction, absolute and percentage change, and month-over-month deltas. Anomalies flagged when a period deviates beyond 2 standard deviations of the series. No forecast is produced, and the UI says why.

**Driver analysis** — four independent lenses, deliberately triangulated so no single method carries the claim:
1. Correlation of each numeric feature with the target
2. Group comparison: default rate by band, with chi-square for categoricals and Mann-Whitney U for numerics, reporting effect size next to the p-value
3. Random Forest impurity importance
4. Permutation importance on held-out data (the most trustworthy of the four)

Language is fixed at the template level: *"associated with"*, *"identified as an important predictive feature"*. The word "causes" is not used. A standing caveat notes that these are observational associations.

**Segmentation** — KMeans on standardised behavioural features (utilisation, repayment ratio, delinquency count, balance trend, limit). `k` swept over 2–8, chosen by silhouette score with an elbow plot shown; the chosen `k` is justified from the measured scores, not fixed in advance. **The target is excluded from clustering inputs** so segments describe behaviour, not the answer. Clusters are labelled *after* inspecting their measured profiles, then reported with size, mean limit, mean utilisation, mean repayment ratio, mean delinquency, and observed default rate.

---

## 11. ML Methodology

**Task** — binary classification of `default_next_month`; 30,000 rows, 22.12% positive (verified).

**Leakage check** — `pay_0` is the September repayment status and the target is the October default. This is legitimately prior information, not leakage. It will be documented, because `pay_0` is likely to dominate importance and a reader deserves to know why.

**Features** — engineered behavioural features plus `limit_bal` and `age`. `sex` and `marriage` excluded by default on fairness grounds (§4.5); `education` retained but reviewed in the bias audit. A config flag can include protected attributes for comparison, defaulting to off.

**Split** — stratified 80/20 hold-out, plus stratified 5-fold CV on the training portion. Preprocessing lives inside an `sklearn.Pipeline` so scaling is fit on training folds only.

**Models** — escalating, simplest first:
1. `DummyClassifier` — the floor any real model must beat
2. **Logistic Regression** (scaled, `class_weight="balanced"`) — interpretable baseline
3. Random Forest
4. Gradient Boosting / HistGradientBoosting

**Evaluation** — accuracy is explicitly *not* the headline: predicting "no default" for everyone scores 77.88% on this data, which is why the report leads with:
- **ROC-AUC** (primary) and **PR-AUC** (better suited to imbalance)
- Recall on the default class — the costly error is a missed default
- Precision, F1, full confusion matrix
- KS statistic and a calibration curve with Brier score
- **Decision threshold tuned explicitly** against a stated cost assumption rather than left at 0.5, with the assumption written down

Whatever the metrics turn out to be, they get reported as measured. No target figure is being aimed for.

---

## 12. Explainability

- **Global:** logistic-regression coefficients as odds ratios; tree feature importances; permutation importance on the test split (primary).
- **Local:** per-client contribution breakdown in the prediction interface.
- **SHAP:** attempted for TreeExplainer on a sample; if import or compute fails, the UI degrades to permutation importance with a visible note. Never a hard requirement.
- **Fairness audit:** recall, precision, and selection rate compared across `sex`, `age_band`, and `education` groups, reported plainly even if unflattering.

Every prediction surface carries a fixed disclaimer: the output is a model-estimated probability from 2005 historical data, not a guaranteed financial outcome, and not advice.

---

## 13. Risk, Opportunity, Recommendations

**Risk tiers** — from model predicted probability, binned Low / Medium / High at documented cut-points, every view labelled **"model-derived"** to separate it from the observed historical target. Breakdowns by credit-limit band, utilisation band, delinquency history, age band, and segment. The distinction between *observed* default (22.12%, historical fact) and *predicted* risk (model output) is maintained rigorously throughout.

**Opportunities** — derived only from measured findings: segments with low predicted risk and high headroom (low utilisation, high limit), clients with improving delinquency trends, consistent full-payers, and under-utilised low-risk accounts. Each opportunity card names the metric and value it came from.

**Recommendation engine** — rule objects that each declare a trigger condition on computed metrics and emit the four-part chain. A rule that does not trigger produces nothing; there is no filler. Every recommendation renders with the supporting number visible, and is framed as an option for review ("consider…", "review…") rather than an instruction.

---

## 14. AI Insights Layer

**Contract:** the LLM receives a compact JSON of already-computed results — KPI values, trend statistics, top features with importances, segment profiles, risk distribution, model metrics. **It never receives the dataframe** and is never asked to calculate anything.

**Prompt discipline:** the system prompt forbids inventing numbers and instructs the model to use only supplied values. Generated text is post-validated — numerals in the output are checked against the supplied payload, and a mismatch flags the response rather than displaying it silently.

**Providers:** `AI_PROVIDER` selects `openai` | `gemini` | `ollama` | `none`. Keys come from `.env` only; `.env.example` ships with placeholder values; `.env` is git-ignored. Failures (missing key, network error, rate limit, timeout) are caught and surfaced as a calm inline notice.

**Analytics-only mode is a first-class path, not a fallback.** With no key configured, every page still works and the AI page shows deterministic template-generated summaries built from the same computed payload. This will be tested explicitly.

---

## 15. Dashboard Structure

| # | Page | Question answered | Core content |
|---|---|---|---|
| 1 | Executive Overview | What is happening? | KPI cards, portfolio health, risk donut, balance/delinquency trend, top findings, alerts, opportunities, actions |
| 2 | Data Explorer | Can I trust the data? | Preview, filters, quality report, invalid-code findings, summary stats, CSV download |
| 3 | Financial Analytics | What are the patterns? | Limit & balance distributions, utilisation analysis, repayment behaviour, delinquency profile, correlations, period trends |
| 4 | Customer Segmentation | Who are the groups? | Elbow/silhouette, cluster scatter, segment profile table, comparison radar |
| 5 | Credit Risk | Who is at risk and why? | Risk distribution, drivers, model comparison, ROC/PR curves, confusion matrix, importances, single-client predictor, fairness audit |
| 6 | AI Insights | What does it mean? | Executive summary, findings, risks, opportunities, FACT→INSIGHT→RISK→ACTION cards |
| 7 | Methodology | How was this done? | Dataset & source, cleaning decisions, KPI formulas, analytics and ML approach, AI guardrails, **limitations** |

Global sidebar filters (limit band, age band, education, delinquency status, risk tier) apply consistently, show active-filter count, and display a proper empty state when a combination yields too few rows — with analyses that need a minimum sample size suppressed and explained rather than computed on 3 records.

**Design:** restrained professional palette (neutral base, one accent, semantic red/amber/green reserved for risk only), consistent 8px spacing scale, system font stack, clear hierarchy, colour-blind-safe sequential scales, tooltips carrying formulas, text alternatives for chart takeaways, keyboard-navigable controls, and skeleton loaders during computation. No gradients-for-decoration, no animation, no oversized hero text.

---

## 16. Security Considerations

- `.env` git-ignored; `.env.example` holds placeholders only. No key ever appears in code, logs, error messages, or the report.
- API keys read only via `os.environ`; the UI shows provider status as configured/not-configured without echoing any value.
- The dataset is public and already anonymised — `ID` is a surrogate key, and no name, address, or account number exists. No re-identification attempt is made.
- Uploaded files (if the optional upload path is enabled): extension and size checks, `.xls`/`.xlsx`/`.csv` only, parsed with pandas without `eval`. No `pickle` loading of user-supplied files.
- Model artefacts in `models/` are produced locally by this project only; `joblib.load` is never pointed at untrusted input.
- No outbound network call except the explicit dataset download and, if configured, the LLM request — which carries aggregate statistics only, never client-level rows.
- `.gitignore` covers `.env`, `__pycache__/`, `.venv/`, `data/raw/`, `data/processed/`, `*.pkl`, `.ipynb_checkpoints/`.

---

## 17. Testing Strategy

`pytest`, with small synthetic fixtures for unit tests (fast, no file dependency) and a handful of integration tests against the real file, skipped automatically if it is absent.

| File | Covers |
|---|---|
| `test_data_processing.py` | header-row handling, column normalisation, validation failure on a bad schema, code→label mapping, panel reshape |
| `test_data_quality.py` | null/duplicate/invalid-code detection, quality score bounds |
| `test_feature_engineering.py` | utilisation & repayment maths, **division by zero**, negative balances, missing-column skip path |
| `test_kpis.py` | each KPI against hand-computed fixtures; adaptive omission when columns are absent |
| `test_segmentation.py` | determinism under fixed seed, k selection, degenerate input |
| `test_model.py` | trains and beats the dummy baseline, metric ranges valid, threshold logic, no-leakage assertion on the feature list |
| `test_recommendations.py` | rules fire only when triggered; every output has all four parts and a source metric |
| `test_ai_insights.py` | analytics-only mode works with no key; provider failure is caught; **no numeric fabrication** |

Edge cases deliberately included: empty dataframe, single row, one-class target, all-identical column, filter returning zero rows.

---

## 18. Deployment Strategy

Primary target is local, because that is how it will be marked:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_data.py
streamlit run app.py
```

Streamlit Community Cloud is documented as the optional hosted path (repo + `requirements.txt` + secrets via the dashboard, never committed). A `Dockerfile` is listed as future scope, not claimed. `requirements.txt` will pin exact versions that were actually tested on this machine.

---

## 19. Documentation Strategy

| Artefact | Content |
|---|---|
| `README.md` | All 20 required sections, dataset source URL, tested install steps, honest feature list |
| `PROJECT_PLAN.md` | This document |
| `data_dictionary.md` | **Generated programmatically** from the real dataframe — column, dtype, meaning, missing %, example value, used-for-analytics flag, used-for-ML flag. Generated, not typed by hand, so it cannot drift from reality. |
| `reports/PROJECT_REPORT.md` | All 25 required sections, structured for DOCX/PDF conversion, with clearly marked screenshot placeholders |
| `screenshots/` | Real captures added after the app runs; placeholders until then |
| Docstrings | Every public function: purpose, args, returns, and assumptions |
| In-app Methodology page | User-facing version of the method and limitations |

Two standing rules: the README will not claim anything unimplemented, and the report will contain no number that was not produced by running the code.

---

## 20. Implementation Phases

Each phase ends with a run, a check for errors, and fixes. No phase starts before the previous one is stable.

| Phase | Work | Done when | Status |
|---|---|---|---|
| **1** | Inspect workspace, evaluate datasets, verify download, write this plan | plan approved | ✅ **Complete** |
| **2** | venv, project structure, `requirements.txt`, **resolve the pandas 3.x compatibility question**, `.gitignore`, `.env.example` | full stack imports cleanly | ✅ **Complete** |
| **3** | `data_loader.py`, `schema.py`, `download_data.py`, generate `data_dictionary.md` | loads 30,000 × 25 reproducibly | ✅ **Complete** |
| **4** | `data_quality.py`, `data_processing.py`, panel reshape | processed parquet written; quality report lists the real defects | ✅ **Complete** |
| **5** | `feature_engineering.py`, `analytics.py`, `kpi_engine.py`, `trends.py`, `drivers.py`, `findings.py`, `visualization.py`, `pipeline.py` | KPIs match hand checks; trends computed over 6 periods | ✅ **Complete** |
| **6** | `app.py`, theme, navigation, filters, Executive Overview, Data Explorer, Financial Analytics, Data Quality, Methodology | app runs; pages live | ✅ **Complete** |
| **7** | `segmentation.py` + Customer Segmentation page | k justified by measured silhouette; profiles interpreted | ✅ **Complete** |
| 8 | `risk_model.py` + page 5 | models trained, honest metrics, beats dummy | ⬜ |
| 9 | `explainability.py`, fairness audit, single-client predictor | global + local explanation working; SHAP fallback verified | ⬜ |
| 10 | `ai_insights.py`, provider abstraction | works **with and without** an API key | ⬜ |
| 11 | `recommendations.py` + page 6 | every recommendation traceable to a metric | ⬜ |
| 12 | Error-path hardening, page 7 Methodology | no traceback reaches the UI | ⬜ |
| 13 | `README.md`, `reports/PROJECT_REPORT.md`, screenshots | numbers all sourced from real runs | ⬜ |
| 14 | Consolidated single-file submission build, final validation sweep | §30 checklist fully green | ⬜ |

### Implementation status — data foundation & analytics engine

Phases 2–5 are built, run and tested. Measured results from the live pipeline:

| Item | Result |
|---|---|
| Pipeline runtime | 1.4 s for 30,000 rows |
| Shape | 30,000 × 25 raw → 30,000 × 68 processed |
| Long panel | 180,000 client-month rows |
| Rows removed by cleaning | **0** (nothing deleted; suspect values flagged instead) |
| Features created | 37 |
| Features refused (no source column) | 5 |
| KPIs registered / computed | 30 / 30 |
| Data quality score | 98.18 / 100 (Excellent) |
| Tests | **574 passing** |

**Compatibility question resolved.** pandas **3.0.6** works with streamlit 1.64.0,
plotly 7.1.0, scikit-learn 1.9.1, scipy 1.17.1 and pyarrow 25.0.1. The full stack
was exercised with `FutureWarning` escalated to an error and passed, so no
downgrade was needed. Exact tested versions are pinned in `requirements.txt`.

**Modules delivered**

| Module | Purpose |
|---|---|
| `src/config.py` | Immutable settings from env; invalid values degrade with a warning rather than crashing; API keys read on demand, never stored |
| `src/logging_setup.py` | Namespaced logging with a `SecretRedactingFilter` safety net (verified to mask `sk-…` and `AIza…` patterns) |
| `src/schema.py` | `Role` enum + `ColumnRegistry`; raw→canonical mapping; code label maps; column meanings; unavailable-capability explanations |
| `src/data_loader.py` | Discovery, `.xls/.xlsx/.csv` reading with evidence-based header detection and encoding fallback, 16-check validation, structural profiling |
| `src/data_quality.py` | Four scored dimensions + outliers reported unscored; documented project-defined score |
| `src/data_processing.py` | Six cleaning steps, each with a recorded rationale; wide→long panel reshape; parquet persistence |
| `src/feature_engineering.py` | 37 features with `safe_divide` guarding every ratio; per-feature documentation objects |
| `src/kpi_engine.py` | 30-KPI registry with adaptive omission and explained unavailability |
| `src/analytics.py` | Univariate, bivariate, multivariate and outlier analysis with insufficient-data guards |
| `src/trends.py` | Period-over-period analysis across the 6-month panel; forecasting explicitly refused |
| `src/drivers.py` | Four triangulated methods; association-only language enforced centrally |
| `src/findings.py` | FACT→INSIGHT→IMPLICATION→ACTION structure; evidence mandatory at construction |
| `src/visualization.py` | 13 themed Plotly builders, each with an empty-state placeholder |
| `src/pipeline.py` | Orchestration with optional Streamlit caching |

**Decisions taken during implementation, beyond the plan**

1. **Quality scoring pivoted to semantic defects.** The source has 0 nulls and 0
   exact duplicates, so a completeness-only score would read 100/100 and be
   useless. Consistency (0.883) is now the genuinely weakest dimension, driven by
   2,037 clients delinquent with no balance and 1,854 marked "paid in full" against
   a zero payment.
2. **Outliers are reported but not scored.** Penalising a naturally skewed credit
   distribution would mislabel real premium customers as dirty data.
3. **Bounded companions added for unbounded ratios.** `repayment_ratio_mean`
   legitimately reaches 3,333 when a prior statement was near zero. The raw value is
   kept, a clipped `repayment_ratio_capped_mean` was added for averaging and
   modelling, and both `repayment_ratio_mean` and `balance_growth_6m` are excluded
   from the ML feature list. 20 of 37 features are ML-safe.
4. **Correlation ranking filters structurally redundant pairs.** Without it the top
   associations were restatements of formulas (`utilisation_latest` vs
   `utilisation_m1` at r = 1.000, `bill_amt` vs `utilisation` because utilisation
   *is* balance ÷ limit). Derivation families and dependencies are now excluded, and
   the top results are behavioural: repayment status vs repayment ratio
   (r = −0.771) and utilisation vs repayment ratio (r = −0.756).
5. **Anomaly detection uses a modified z-score (median/MAD).** With only five
   period-to-period changes, a single spike inflates the standard deviation enough
   to hide itself. MAD resists that masking.
6. **Outlier detection gained a MAD fallback.** When the IQR collapses to zero
   (most values identical) the fence would flag nothing, missing a genuine extreme.
7. **Protected attributes excluded from driver analysis by default.** `sex` and
   `marriage` are reserved for the fairness audit and are verified absent from
   driver output.

**Measured findings from the real data** (all reproducible, none hand-written)

- Observed default rate **22.12%** (6,636 of 30,000). Majority-class baseline
  accuracy is 77.88%, which is why accuracy is not the headline metric.
- **Portfolio deterioration across the window:** mean balance rose 31.8%
  (NT$38,872 → NT$51,223), utilisation rose 33.0% (31.9% → 42.4%), and the share
  of clients behind on payment rose **121.4%** (10.26% → 22.73%).
- Default rate falls monotonically as credit limit rises: **31.79%** in the
  NT$10k–50k band versus **13.98%** in the NT$240k–1,000k band.
- Payment history dominates the driver consensus: `max_delinquency` is flagged by
  all three applicable methods with Cohen's d = 0.961 (large).
- 62.4% of clients are revolving borrowers; 26.8% typically clear their statement.

**Known limitations after the data foundation**

- No predictive model, segmentation, explainability, AI layer or recommendation
  rules yet — phases 7–11.
- SHAP is deliberately absent until phase 9, where it will be optional with a
  permutation-importance fallback.
- Trend findings are historical (2005, Taiwan) and region-specific.

---

### Implementation status — dashboard (phase 6)

Built directly in the premium dark aesthetic rather than as a plain build followed
by a restyle. **805 tests passing** (574 from the data foundation plus 231 new).

| Item | Result |
|---|---|
| Pages fully implemented | 5 — Executive Overview, Financial Analytics, Data Explorer, Data Quality, Methodology |
| Pages with honest placeholders | 4 — Segmentation, Credit Risk, AI Insights, Recommendations |
| UI modules created | 9 under `src/ui/` plus 6 page modules |
| Charts | 7 dashboard compositions over 13 generic primitives, one shared Plotly template |
| Filters | 6 dimensions, applied once per rerun and shared by every visual |
| New analytics modules | `src/filtering.py`, `src/signals.py` |
| Tests added | 231 (`test_filtering`, `test_signals`, `test_ui`, `test_app`) |

**Architecture decisions**

1. **One context per rerun.** `src/ui/context.py` assembles the pipeline result,
   filtered selection, KPIs, findings and signals once, and every visual reads
   from it. This structurally prevents the failure mode where a filter moves some
   charts and silently misses others.
2. **Filter state read from widget keys, not a stored copy.** Streamlit assembles
   the page context before the filter bar renders, so reading a separately stored
   spec would lag one rerun behind. Reading widget state directly makes a filter
   change take effect immediately.
3. **Analytics stayed out of the UI.** Filtering and signal derivation went into
   `src/filtering.py` and `src/signals.py` as pure modules, so both are unit
   tested without a browser. `src/ui/` renders only.
4. **Change indicators only where a real comparison exists.** This dataset is a
   single extract with no prior period, so no "vs last month" is fabricated. The
   one genuine comparison — filtered selection versus full portfolio — is used,
   and the Executive Overview states why no indicator appears when unfiltered.
5. **Per-section error isolation.** Each analytical section is wrapped
   individually, so one failure degrades to a single notice instead of blanking
   the page below it.

**Problems found and fixed during the build**

1. **Utilisation finding was silently suppressed.** It compared the first and last
   band in category order, but the "Negative / zero" band (accounts in credit)
   sits first with a high 24.75% rate, masking a real 10.3-point gradient
   (16.93% → 27.25%). Now the negative band is excluded from the gradient and
   reported separately.
2. **Severity ranked by reach alone.** A 58-point delinquency separation ranked
   below a wider but weaker pattern. A large measured gap now escalates severity,
   so the sharpest signal surfaces first.
3. **`background_gradient` required matplotlib**, which is deliberately not a
   dependency. Caught by `AppTest`, not by the health check. Replaced with a
   themed Plotly heatmap — no new dependency and a better fit for the dark theme.
4. **Whole-page error wrapping** meant one broken section blanked everything
   after it. Replaced with per-section wrapping.
5. **Fragile monkeypatch of the page header** in the first routing draft was
   replaced with an explicit `show_filters` parameter.

**Measured findings now surfaced in the UI** (all derived at runtime, none written
by hand)

- Payment delay is the sharpest split: **13.83%** default with no delay against
  **71.92%** at 3+ months behind, roughly 5.2×.
- Utilisation gradient **16.93% → 27.25%**, then a plateau — reported honestly as
  directional but not strictly monotonic.
- Credit-limit gradient **31.79% → 13.98%**, monotonic across quartiles.
- Portfolio deterioration across the six periods: delinquency **+121.4%**,
  utilisation **+33.0%**, balances **+31.8%**.
- Transactor opportunity: **26.8%** of customers clear their statement and default
  at **14.69%** against 22.12% overall.

**Known limitations of the dashboard build**

- Four pages remain placeholders by design; they state their scope and show no
  mock content.
- The AI status indicator reports "Analytics Mode" because `AI_PROVIDER=none`. No
  AI text is generated anywhere.
- Screenshots for the report are not yet captured; the pages are built to be
  captured cleanly at 1920×1080 and 1440×900.
- The Data Quality page deliberately ignores filters, since it describes the raw
  source as it arrived.

On §24 of the brief: the clean `src/` architecture is the real deliverable. For the internship's single-file requirement, Phase 14 generates `submission/finrisk_ai_complete.py` as an additional artefact assembled from the modules — the modular codebase is not flattened or damaged to satisfy the format.

---

## 21. Risk Register for the Build

| Risk | Likelihood | Mitigation |
|---|---|---|
| pandas 3.x incompatible with Streamlit/Plotly/SHAP | Medium | Resolved empirically in Phase 2; pin to 2.2.x if needed |
| SHAP install or runtime failure on Windows/Py3.11 | Medium | Optional by design, permutation-importance fallback |
| `pay_0` dominates the model, looking like leakage | High | Documented as legitimate prior information; models also reported without it for comparison |
| Reviewer expects income-based KPIs from the brief | High | §4.5 explains the absence prominently, in plan, README, report, and UI |
| Class imbalance produces flattering accuracy | High | Accuracy demoted; ROC-AUC/PR-AUC/recall lead; dummy baseline shown |
| KMeans clusters are not meaningfully distinct | Medium | Silhouette reported honestly; if weak, say so rather than over-claiming |
| Dashboard slow on 30,000 rows | Low | `st.cache_data`/`cache_resource`; model trained once and persisted |
| Scope creep across 7 pages | Medium | Phase gates; every chart must answer a stated question |

---

## 22. Success Criteria

The project is complete when:

1. `streamlit run app.py` works from a clean clone after the documented steps.
2. All 7 pages render, with filters applied consistently and no traceback surfacing.
3. Every displayed number traces to a function call on the real dataset.
4. The model beats the dummy baseline on ROC-AUC, with metrics reported as measured.
5. Predicted risk is never conflated with observed default.
6. Every recommendation shows its supporting evidence.
7. The app is fully functional with no AI key configured.
8. No secret is present anywhere in the repository.
9. Tests pass, including the edge cases in §17.
10. README, data dictionary, and report are complete, honest, and cite the dataset source.
11. Missing-column limitations are stated openly rather than papered over.

---

## Awaiting approval

Phase 1 is complete. On approval I will begin **Phase 2** (environment, structure, and the pandas-version compatibility test) and continue through the phases, stopping at any point you want to redirect.

---

### Implementation status — customer segmentation (phase 7)

Behaviour-based K-Means segmentation, with every methodological choice resolved by
measurement. **984 tests passing** (805 prior plus 179 new).

| Item | Result |
|---|---|
| Method | K-Means on 7 standardised behavioural features |
| Scaler | **Robust** — chosen by measurement, not convention |
| Segments (K) | **3** — the measured silhouette optimum |
| Silhouette | **0.4035** (stratified sample of ~5,000, seed 42) |
| Davies-Bouldin | **1.0001** (lower is better) |
| Calinski-Harabasz | 12,276 |
| Customers segmented | 30,000 — none dropped |
| PCA (display only) | 44.56% + 27.37% = **71.9%** of variance |
| Fit time | ~6 s including the scaler comparison, cached per session |

**Features used.** `credit_limit`, `utilisation_mean_6m`,
`repayment_ratio_capped_mean`, `delinquent_months_count`, `bill_trend_slope`,
`utilisation_volatility`, `months_zero_payment`.

**Features excluded by policy.** The outcome (`default_next_month`) and every
demographic attribute (`sex`, `marriage`, `education`, `age`), plus identifiers.
Requesting any of them raises rather than silently succeeding, and this is asserted
from four directions in the tests — including a permutation test that shuffles the
target and confirms the cluster assignments do not move.

#### Decisions resolved by measurement

**1. Scaler: robust, by a clear margin.**

| Scaler | Best K | Silhouette | Davies-Bouldin | Smallest segment |
|---|---|---|---|---|
| Standard | 6 | 0.2879 | 1.2033 | 4.74% |
| **Robust** | **3** | **0.4035** | **1.0001** | **12.07%** |

Robust scaling won on both metrics and produced healthier segment sizes. The
reason is in the data: the clustering features are strongly skewed (utilisation
volatility 3.43, balance trend 2.35, delinquency count 2.13). Standard scaling
divides by a standard deviation that a long tail inflates, which compresses the
bulk of a skewed feature into a narrow band; the median and IQR are far less
affected.

**2. K = 3, on silhouette.**

| K | Silhouette | Davies-Bouldin | Smallest segment |
|---|---|---|---|
| 2 | 0.3525 | 1.2222 | 17.14% |
| **3** | **0.4035** | **1.0001** | **12.07%** |
| 4 | 0.3350 | 1.0388 | 2.85% |
| 5 | 0.3508 | 0.9805 | 1.83% |
| 6 | 0.3014 | 1.0373 | 1.81% |
| 7 | 0.2698 | 1.0755 | 1.76% |
| 8 | 0.2832 | 1.0882 | 1.48% |

Silhouette peaks at K=3. The inertia elbow sits at K=4 and Davies-Bouldin is best
at K=5, so the three criteria disagree — which is normal for continuous
behavioural data with no naturally separated groups. Silhouette is treated as
primary and the full curve is displayed, so the disagreement is visible rather
than hidden. K=4 and beyond also produce thin segments (under 3%), which describe
outliers rather than usable populations.

**3. Feature set: all 7 retained, and deliberately *not* chosen on silhouette.**
Dropping the two trajectory features (`bill_trend_slope`,
`utilisation_volatility`) leaves five, which scores a *higher* silhouette — 0.5672
at its own recommended K=2, against 0.4035 for seven at K=3 — but
**silhouette rises mechanically as dimensionality falls**, so it is only
comparable across K within a fixed feature space — never across feature spaces of
different size. The higher score also describes a visibly coarser partition: the
five-feature set puts 87.5% of the book in one segment against 72.6% for seven.
The set was therefore chosen on redundancy and coverage: the
strongest correlation between any two candidates is 0.756 (utilisation against
repayment ratio), below the 0.80 redundancy threshold, and the two measure
genuinely different things. The reduced set used for this comparison is pinned in
code as `COMPARISON_FEATURE_SUBSET` so the figures are reproducible.

**4. Missing values: median-imputed, never dropped.**
`repayment_ratio_capped_mean` is undefined for **1,406 customers (4.69%)** whose
prior statements were all zero or negative. Dropping them would bias the
segmentation toward active accounts, so the median is imputed, recorded in the
result, and surfaced on the page with the caveat that an undefined ratio usually
signals an inactive or in-credit account rather than poor repayment, and that an
imputed value is not an observed customer value.

#### Segments discovered

| Segment | Customers | Share | Observed default rate |
|---|---|---|---|
| Mid-Range / Typical-Behaviour | 21,775 | 72.6% | 16.51% |
| Growing-Balance / Volatile-Usage | 4,604 | 15.3% | 18.68% |
| Delinquent / Low-Repayment | 3,621 | 12.1% | **60.20%** |

Names are composed at runtime from each centroid's standardised distance from the
portfolio mean, so every word traces to a measured value. Nothing is pre-assigned,
and the vocabulary contains no judgemental terms — no "good", "bad" or "safe"
customer labels.

The 43.7-point spread in observed default rate is notable precisely **because the
outcome was excluded from clustering**. Behaviour alone separates the portfolio
that sharply; the rate was measured afterwards and is descriptive only.

#### The inconvenient result, stated plainly

**72.6% of customers fall into one undifferentiated segment** that sits near the
portfolio average on every clustering feature. This is a real limitation of the
measured outcome, not a display problem: credit behaviour in this portfolio is
largely continuous rather than naturally clustered, so a clean three-way split of
the whole book does not exist. Higher K values subdivide that majority but score
worse and produce thin segments. The page says so directly and provides a K
selector so alternatives can be compared.

#### Architecture

* `src/segmentation.py` — all logic, no Streamlit. Dataclasses:
  `SegmentationConfig`, `ImputationInfo`, `FeatureMatrix`, `KScore`,
  `KSelectionResult`, `ScalerComparison`, `SegmentProfile`, `SegmentationResult`,
  `SegmentationUnavailable`.
* `src/ui/pages/segmentation.py` — presentation only.
* `src/ui/charts.py` — five new builders: K-selection diagnostics, segment
  distribution, observed risk, PCA map, profile comparison, plus a scaler
  comparison chart.
* **Fitted on the whole portfolio, not the filtered selection.** Segments are a
  property of the book; refitting per filter change would make them shift
  underfoot and prevent comparison between views. When filters are active the page
  keeps the fixed segments and shows how the selection distributes across them.
* Cached with `st.cache_resource` keyed on K, so exploring an alternative K reuses
  the scaler comparison and the K sweep.

#### Problems found and fixed

1. **Sampled silhouette could crash.** Silhouette on a random subset raises
   `ValueError: Number of labels is 1` if a small cluster is absent from the
   sample. First fixed reactively by catching the error and recomputing on all
   rows, then replaced with a correct fix: `_stratified_sample_index` draws a
   seeded sample that allocates each cluster a share proportional to its real
   size, floored at two rows, so at least two labels and every cluster are present
   **by construction**. The reactive version also hid a performance trap — the
   fallback computed silhouette on all 30,000 rows, a quadratic operation.
   `_safe_silhouette` now returns NaN only when a partition genuinely cannot be
   scored (fewer than two clusters), and the caller skips that K and says so.
2. **`DataFrame.to_numpy()` is read-only in pandas 3**, so `np.fill_diagonal` on a
   correlation matrix failed. Replaced with a pandas-native diagonal mask.
3. **Degenerate partitions were adopted silently.** If silhouette's best K leaves
   a cluster holding almost nobody, the next best non-degenerate K is now
   recommended and the substitution is explained in the rationale.
4. **pytest 9 deprecates class-scoped fixtures written as instance methods**, which
   surfaced as a fixture error because warnings are errors in this suite.

#### Limitations

* Unsupervised and descriptive. It assigns groups; it does not score, rank or
  predict any individual.
* Observed default rates are historical measurements attached to segments after
  clustering. No causal link between membership and outcome is claimed.
* The PCA map retains 71.9% of the variance, so roughly 28% of the structure
  K-Means used is not visible. Points that appear to overlap may be separated on
  the discarded dimensions; the silhouette and Davies-Bouldin scores are the
  evidence about cluster quality, not the plot.
* K-Means assumes roughly spherical, similarly sized clusters. Credit behaviour
  does not fully satisfy that, which is part of why one segment dominates.
* Segments describe this 2005 Taiwan portfolio and would need refitting for any
  other book.

---

### Implementation status — dashboard launch defect (phase 7a)

**Status: fixed and verified.**

The dashboard would not open for the user. Two independent causes, both in launch
configuration rather than in application code. No page, chart or analysis was at
fault — every page already rendered correctly under AppTest.

#### Cause 1 — `server.headless = true` in `.streamlit/config.toml`

With `headless` set in the config file, Streamlit never opens a browser tab on
`streamlit run app.py`. The server starts normally, prints its URLs and answers
`/_stcore/health` with `200 ok`, so nothing in the terminal indicates a problem —
but no window appears. That presents exactly as "the dashboard is not opening".

The setting was originally added so automated health probes would not spawn
browsers. It has been removed from the config file; automated runs pass
`--server.headless true` on the command line instead, which keeps the interactive
default correct. The dark theme block was left untouched.

#### Cause 2 — `streamlit` is not on the system PATH

`streamlit.exe` lives in `.venv\Scripts`. From a shell where the environment is not
active, the documented command fails outright:

```text
streamlit : The term 'streamlit' is not recognized as the name of a cmdlet ...
```

Added `run_dashboard.ps1`, which resolves the interpreter inside `.venv`
explicitly, warns when `data\raw` is empty, and accepts `-Port` and `-Headless`.
The README now documents the launcher, the activation route and the direct
`python -m streamlit` route.

#### Why HTTP 200 was not accepted as proof

Streamlit serves a static SPA shell on `/` and only executes `app.py` when a client
opens the `/_stcore/stream` websocket. A 200 on `/` therefore proves the server is
listening, not that the application runs. Verification used three layers instead:

| Layer | Evidence |
|---|---|
| AppTest | 82 tests; all 9 pages render, 0 exceptions, navigation, filters and the empty-selection path all pass |
| Real websocket client | Connecting forced a genuine script run: pipeline completed in 4.92 s, 30 KPIs, 7 findings, 4 risk signals, full driver analysis |
| Log inspection | 78 log lines across the live session, **0 ERROR / CRITICAL / Traceback entries** |

#### Verified end state

| Check | Result |
|---|---|
| Full test suite | **997 passed**, 0 failed, 0 errors |
| Segmentation tests | **173 passed** (160 existing + 13 new stratified-sampling tests) |
| AppTest pages | 9/9 render, 0 exceptions |
| Segmentation contract | `is_available=True`, `k=3`, `scaler=robust`, 3 profiles |
| Target leakage | Permuting the outcome leaves every cluster label byte-identical, while observed rates collapse to the ~22% base rate |
| Demographic leakage | `sex`, `marriage`, `education`, `age` all absent from the 30,000 × 7 matrix |
| Dependencies | `pip check`: no broken requirements; nothing added |
