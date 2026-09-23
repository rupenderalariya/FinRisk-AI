# Data Dictionary

# Default of Credit Card Clients

> **This file is generated.** Run `python scripts/generate_data_dictionary.py`
> to regenerate it. Every dtype, missing percentage and example value below was
> read from an actual pipeline run, so this document cannot drift away from the code.

---

## 1. Dataset provenance

| Property | Value |
|---|---|
| Dataset | Default of Credit Card Clients |
| Source | UCI Machine Learning Repository (dataset 350) |
| URL | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| Licence | CC BY 4.0 (UCI ML Repository) |
| Period covered | April 2005 - September 2005 |
| Currency | NT$ (New Taiwan Dollar) |
| Source rows | 30,000 |
| Source columns | 25 |
| Columns after processing | 68 |
| Missing cells in source | 0 |
| Exact duplicate rows | 0 |
| Duplicates excluding `client_id` | 35 |
| Data quality score | 98.18 / 100 (Excellent) |

**Citation.** Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.

---

## 2. Panel structure

Month index 1 is the most recent observation (September 2005) and index 6 is the oldest (April 2005). A payment recorded in month m settles the statement issued in month m+1, so repayment ratios divide pay_amt_m<m> by bill_amt_m<m+1>. The source file's repayment-status columns are named PAY_0, PAY_2..PAY_6 with no PAY_1; PAY_0 is the September status and is mapped to month index 1.

| Month index | Calendar month | Role |
|---|---|---|
| 1 | September 2005 | Most recent observation |
| 2 | August 2005 | |
| 3 | July 2005 | |
| 4 | June 2005 | |
| 5 | May 2005 | |
| 6 | April 2005 | Oldest observation |

---

## 3. Source columns

Columns exactly as they arrive from the source file, after header
normalisation to snake_case.

| Column | Data type | Meaning | Missing % | Example | Used for analytics | Used for ML |
|---|---|---|---|---|---|---|
| `client_id` | int64 | Surrogate identifier for the credit-card client. An auto-increment key carrying no information about the client, used only to join and count. | 0.00% | 1 | No (join key only) | No (identifier) |
| `credit_limit` | int64 | Amount of revolving credit granted to the client, in NT$. This is a CREDIT LIMIT, not income: the dataset has no income column, so this is used as a coarse proxy for credit capacity and is always labelled as a limit. | 0.00% | 20,000 | Yes | Yes (continuous capacity measure) |
| `sex_code` | int64 | Biological sex as an integer code (1 = male, 2 = female). Retained for the fairness audit only; excluded from risk modelling because pricing credit on sex is discriminatory and unlawful in many jurisdictions. | 0.00% | 2 | Yes (fairness audit only) | **No - excluded on fairness grounds** |
| `education_code` | int64 | Highest education level as an integer code. The source documents 1 = graduate school, 2 = university, 3 = high school, 4 = others. Codes 0, 5 and 6 also occur in the data but are undocumented. | 0.00% | 2 | Yes | Yes (reviewed in the fairness audit) |
| `marriage_code` | int64 | Marital status as an integer code. The source documents 1 = married, 2 = single, 3 = others. Code 0 also occurs but is undocumented. Retained for the fairness audit only; excluded from risk modelling. | 0.00% | 1 | Yes (fairness audit only) | **No - excluded on fairness grounds** |
| `age` | int64 | Client age in years at the time of the extract. | 0.00% | 24 | Yes | Yes |
| `pay_status_m1` | int64 | Repayment status for Sep 2005. Documented codes: -1 = paid in full, and 1 to 9 = that many months of payment delay. Codes -2 and 0 occur frequently but are not defined in the source paper; they are commonly read as 'no transaction' and 'revolving credit used'. Only codes of 1 or more are treated as delinquency, so this ambiguity does not affect the delinquency measures. | 0.00% | 2 | Yes | Yes (documented as legitimate prior information) |
| `pay_status_m2` | int64 | Repayment status for Aug 2005. Documented codes: -1 = paid in full, and 1 to 9 = that many months of payment delay. Codes -2 and 0 occur frequently but are not defined in the source paper; they are commonly read as 'no transaction' and 'revolving credit used'. Only codes of 1 or more are treated as delinquency, so this ambiguity does not affect the delinquency measures. | 0.00% | 2 | Yes | Yes (documented as legitimate prior information) |
| `pay_status_m3` | int64 | Repayment status for Jul 2005. Documented codes: -1 = paid in full, and 1 to 9 = that many months of payment delay. Codes -2 and 0 occur frequently but are not defined in the source paper; they are commonly read as 'no transaction' and 'revolving credit used'. Only codes of 1 or more are treated as delinquency, so this ambiguity does not affect the delinquency measures. | 0.00% | -1 | Yes | Yes (documented as legitimate prior information) |
| `pay_status_m4` | int64 | Repayment status for Jun 2005. Documented codes: -1 = paid in full, and 1 to 9 = that many months of payment delay. Codes -2 and 0 occur frequently but are not defined in the source paper; they are commonly read as 'no transaction' and 'revolving credit used'. Only codes of 1 or more are treated as delinquency, so this ambiguity does not affect the delinquency measures. | 0.00% | -1 | Yes | Yes (documented as legitimate prior information) |
| `pay_status_m5` | int64 | Repayment status for May 2005. Documented codes: -1 = paid in full, and 1 to 9 = that many months of payment delay. Codes -2 and 0 occur frequently but are not defined in the source paper; they are commonly read as 'no transaction' and 'revolving credit used'. Only codes of 1 or more are treated as delinquency, so this ambiguity does not affect the delinquency measures. | 0.00% | -2 | Yes | Yes (documented as legitimate prior information) |
| `pay_status_m6` | int64 | Repayment status for Apr 2005. Documented codes: -1 = paid in full, and 1 to 9 = that many months of payment delay. Codes -2 and 0 occur frequently but are not defined in the source paper; they are commonly read as 'no transaction' and 'revolving credit used'. Only codes of 1 or more are treated as delinquency, so this ambiguity does not affect the delinquency measures. | 0.00% | -2 | Yes | Yes (documented as legitimate prior information) |
| `bill_amt_m1` | int64 | Statement balance for Sep 2005 in NT$. Negative values are legitimate and mean the account is in credit, usually after an overpayment or refund. | 0.00% | 3,913 | Yes | No (superseded by engineered summary features) |
| `bill_amt_m2` | int64 | Statement balance for Aug 2005 in NT$. Negative values are legitimate and mean the account is in credit, usually after an overpayment or refund. | 0.00% | 3,102 | Yes | No (superseded by engineered summary features) |
| `bill_amt_m3` | int64 | Statement balance for Jul 2005 in NT$. Negative values are legitimate and mean the account is in credit, usually after an overpayment or refund. | 0.00% | 689 | Yes | No (superseded by engineered summary features) |
| `bill_amt_m4` | int64 | Statement balance for Jun 2005 in NT$. Negative values are legitimate and mean the account is in credit, usually after an overpayment or refund. | 0.00% | 0 | Yes | No (superseded by engineered summary features) |
| `bill_amt_m5` | int64 | Statement balance for May 2005 in NT$. Negative values are legitimate and mean the account is in credit, usually after an overpayment or refund. | 0.00% | 0 | Yes | No (superseded by engineered summary features) |
| `bill_amt_m6` | int64 | Statement balance for Apr 2005 in NT$. Negative values are legitimate and mean the account is in credit, usually after an overpayment or refund. | 0.00% | 0 | Yes | No (superseded by engineered summary features) |
| `pay_amt_m1` | int64 | Amount paid by the client during Sep 2005, in NT$. This payment settles the statement issued in the PREVIOUS month, which is why repayment ratios divide by the month+1 balance. | 0.00% | 0 | Yes | No (superseded by engineered summary features) |
| `pay_amt_m2` | int64 | Amount paid by the client during Aug 2005, in NT$. This payment settles the statement issued in the PREVIOUS month, which is why repayment ratios divide by the month+1 balance. | 0.00% | 689 | Yes | No (superseded by engineered summary features) |
| `pay_amt_m3` | int64 | Amount paid by the client during Jul 2005, in NT$. This payment settles the statement issued in the PREVIOUS month, which is why repayment ratios divide by the month+1 balance. | 0.00% | 0 | Yes | No (superseded by engineered summary features) |
| `pay_amt_m4` | int64 | Amount paid by the client during Jun 2005, in NT$. This payment settles the statement issued in the PREVIOUS month, which is why repayment ratios divide by the month+1 balance. | 0.00% | 0 | Yes | No (superseded by engineered summary features) |
| `pay_amt_m5` | int64 | Amount paid by the client during May 2005, in NT$. This payment settles the statement issued in the PREVIOUS month, which is why repayment ratios divide by the month+1 balance. | 0.00% | 0 | Yes | No (superseded by engineered summary features) |
| `pay_amt_m6` | int64 | Amount paid by the client during Apr 2005, in NT$. This payment settles the statement issued in the PREVIOUS month, which is why repayment ratios divide by the month+1 balance. | 0.00% | 0 | Yes | No (superseded by engineered summary features) |
| `default_next_month` | int64 | TARGET. Whether the client defaulted on the following month's payment (1 = default, 0 = no default). This is the recorded historical outcome, never a model prediction. | 0.00% | 1 | Yes (as the outcome) | Yes (target) |

---

## 4. Columns added during cleaning

Flags and preserved originals. Nothing here alters a source value; these
columns exist so suspect records can be isolated rather than deleted.

| Column | Data type | Meaning | Missing % | Example | Used for analytics | Used for ML |
|---|---|---|---|---|---|---|
| `is_duplicate_excluding_id` | bool | Flag: this row is identical to another once client_id is excluded. Such rows are retained by default, because a coincidental match between two genuinely different clients is plausible with integer-coded fields. | 0.00% | False | Audit trail | No |
| `sex` | category | Readable label derived from sex_code. | 0.00% | Female | Yes (fairness audit only) | **No - excluded on fairness grounds** |
| `education` | category | Readable label derived from education_code. Undocumented codes (0, 5, 6) are grouped as 'Unknown' rather than guessed at. | 0.00% | University | Yes | Yes (reviewed in the fairness audit) |
| `marriage` | category | Readable label derived from marriage_code. Undocumented code 0 is grouped as 'Unknown'. | 0.00% | Married | Yes (fairness audit only) | **No - excluded on fairness grounds** |
| `has_negative_balance` | bool | Flag: the client had a negative statement balance in at least one month, meaning the account was in credit. Valid data, retained unchanged. | 0.00% | False | Yes | No |
| `has_extreme_value` | bool | Flag: at least one monetary value for this client lies beyond a 3x IQR fence. Nothing is capped or removed; the flag exists so these clients can be isolated on request. | 0.00% | False | Audit trail | No |

---

## 5. Engineered features

37 features were created. A feature is only produced when
its source columns are verified present at runtime; otherwise it is skipped
and the reason recorded in section 6.

**On the `Used for ML` column.** Some features are excellent for reading and
charting but unsuitable as model inputs. An unbounded ratio such as
`repayment_ratio_mean` can reach into the thousands when a prior statement was
near zero, which would dominate any distance-based model, so a bounded
companion is used instead.

### Credit utilisation

| Feature | Formula | Source columns | Interpretation | Missing % | Example | Used for ML |
|---|---|---|---|---|---|---|
| `utilisation_m1` | `bill_amt_m1 / credit_limit` | `bill_amt_m1`, `credit_limit` | Proportion of the credit limit used in month 1. Values above 1.0 mean the account is over its limit; negative values mean the account is in credit. | 0.00% | 0.1956 | No |
| `utilisation_m2` | `bill_amt_m2 / credit_limit` | `bill_amt_m2`, `credit_limit` | Proportion of the credit limit used in month 2. Values above 1.0 mean the account is over its limit; negative values mean the account is in credit. | 0.00% | 0.1551 | No |
| `utilisation_m3` | `bill_amt_m3 / credit_limit` | `bill_amt_m3`, `credit_limit` | Proportion of the credit limit used in month 3. Values above 1.0 mean the account is over its limit; negative values mean the account is in credit. | 0.00% | 0.0345 | No |
| `utilisation_m4` | `bill_amt_m4 / credit_limit` | `bill_amt_m4`, `credit_limit` | Proportion of the credit limit used in month 4. Values above 1.0 mean the account is over its limit; negative values mean the account is in credit. | 0.00% | 0 | No |
| `utilisation_m5` | `bill_amt_m5 / credit_limit` | `bill_amt_m5`, `credit_limit` | Proportion of the credit limit used in month 5. Values above 1.0 mean the account is over its limit; negative values mean the account is in credit. | 0.00% | 0 | No |
| `utilisation_m6` | `bill_amt_m6 / credit_limit` | `bill_amt_m6`, `credit_limit` | Proportion of the credit limit used in month 6. Values above 1.0 mean the account is over its limit; negative values mean the account is in credit. | 0.00% | 0 | No |
| `utilisation_latest` | `bill_amt_m1 / credit_limit` | `bill_amt_m1`, `credit_limit` | Utilisation in the most recent month. The primary point-in-time measure of how much of the available credit is committed. | 0.00% | 0.1956 | Yes |
| `utilisation_mean_6m` | `mean(utilisation_m1..m6)` | `utilisation_m1`, `utilisation_m2`, `utilisation_m3`, `utilisation_m4`, `utilisation_m5`, `utilisation_m6` | Average utilisation across the observed months. Smooths one-off spikes and describes sustained reliance on credit. | 0.00% | 0.0642 | Yes |
| `utilisation_max_6m` | `max(utilisation_m1..m6)` | `utilisation_m1`, `utilisation_m2`, `utilisation_m3`, `utilisation_m4`, `utilisation_m5`, `utilisation_m6` | Peak utilisation observed, i.e. the worst-case credit stretch. | 0.00% | 0.1956 | Yes |
| `utilisation_volatility` | `std(utilisation_m1..m6)` | `utilisation_m1`, `utilisation_m2`, `utilisation_m3`, `utilisation_m4`, `utilisation_m5`, `utilisation_m6` | Variability of utilisation. High values indicate erratic borrowing rather than a steady balance. | 0.00% | 0.0804 | Yes |
| `is_high_utilisation` | `utilisation_latest > 0.8` | `utilisation_latest` | True when the most recent utilisation exceeds 80% of the credit limit. Configurable via HIGH_UTILISATION_THRESHOLD. | 0.00% | False | Yes |
| `is_over_limit` | `utilisation_max_6m > 1.0` | `utilisation_max_6m` | True when the statement balance exceeded the credit limit in at least one month. | 0.00% | False | Yes |

### Repayment behaviour

| Feature | Formula | Source columns | Interpretation | Missing % | Example | Used for ML |
|---|---|---|---|---|---|---|
| `repayment_ratio_m1` | `pay_amt_m1 / bill_amt_m2` | `pay_amt_m1`, `bill_amt_m2` | Fraction of the month-2 statement repaid by the month-1 payment. Undefined (null) when the prior statement was zero or negative. | 10.58% | 0 | No |
| `repayment_ratio_m2` | `pay_amt_m2 / bill_amt_m3` | `pay_amt_m2`, `bill_amt_m3` | Fraction of the month-3 statement repaid by the month-2 payment. Undefined (null) when the prior statement was zero or negative. | 11.75% | 1 | No |
| `repayment_ratio_m3` | `pay_amt_m3 / bill_amt_m4` | `pay_amt_m3`, `bill_amt_m4` | Fraction of the month-4 statement repaid by the month-3 payment. Undefined (null) when the prior statement was zero or negative. | 12.90% | 0.3056 | No |
| `repayment_ratio_m4` | `pay_amt_m4 / bill_amt_m5` | `pay_amt_m4`, `bill_amt_m5` | Fraction of the month-5 statement repaid by the month-4 payment. Undefined (null) when the prior statement was zero or negative. | 13.87% | 0.2894 | No |
| `repayment_ratio_m5` | `pay_amt_m5 / bill_amt_m6` | `pay_amt_m5`, `bill_amt_m6` | Fraction of the month-6 statement repaid by the month-5 payment. Undefined (null) when the prior statement was zero or negative. | 15.69% | 0 | No |
| `repayment_ratio_mean` | `mean(repayment_ratio_m1, repayment_ratio_m2, repayment_ratio_m3, repayment_ratio_m4, repayment_ratio_m5)` | `repayment_ratio_m1`, `repayment_ratio_m2`, `repayment_ratio_m3`, `repayment_ratio_m4`, `repayment_ratio_m5` | Average share of each statement repaid. Near 1.0 indicates a client who clears the balance; near 0 indicates minimal payment. CAVEAT: when a prior statement was very small but positive (say NT$5), an ordinary payment produces an enormous ratio, so this unbounded mean is heavily right-skewed. Use repayment_ratio_median for a typical value and repayment_ratio_capped_mean for averaging. | 4.69% | 0.5 | No |
| `repayment_ratio_median` | `median(repayment_ratio_m1, repayment_ratio_m2, repayment_ratio_m3, repayment_ratio_m4, repayment_ratio_m5)` | `repayment_ratio_m1`, `repayment_ratio_m2`, `repayment_ratio_m3`, `repayment_ratio_m4`, `repayment_ratio_m5` | Typical repayment share, resistant to a single unusual month. Still unbounded above, so a client with several tiny prior statements can score very high. | 4.69% | 0.5 | No |
| `repayment_ratio_capped_mean` | `mean(clip(repayment_ratio_m1, repayment_ratio_m2, repayment_ratio_m3, repayment_ratio_m4, repayment_ratio_m5, 0, 2.0))` | `repayment_ratio_m1`, `repayment_ratio_m2`, `repayment_ratio_m3`, `repayment_ratio_m4`, `repayment_ratio_m5` | Average repayment share with each month clipped at 200% of the statement. Clipping is applied because a near-zero prior balance can inflate the raw ratio into the thousands, which would distort both the average and any distance-based model. A value of 1.0 means the statement was typically cleared; 2.0 means it was repaid at least twice over. The uncapped values remain available in repayment_ratio_mean. | 4.69% | 0.5 | Yes |
| `is_full_payer` | `repayment_ratio_median >= 0.95` | `repayment_ratio_median` | Client typically clears the statement in full - a transactor rather than a borrower. | 0.00% | False | Yes |
| `is_revolver` | `repayment_ratio_median < 0.3` | `repayment_ratio_median` | Client typically repays less than 30% of the statement, carrying a revolving balance forward. | 0.00% | False | Yes |
| `months_zero_payment` | `count(pay_amt_m1..m6 == 0)` | `pay_amt_m1`, `pay_amt_m2`, `pay_amt_m3`, `pay_amt_m4`, `pay_amt_m5`, `pay_amt_m6` | Number of observed months with no payment at all. A direct count of non-payment events. | 0.00% | 5 | Yes |
| `total_paid_6m` | `sum(pay_amt_m1..m6)` | `pay_amt_m1`, `pay_amt_m2`, `pay_amt_m3`, `pay_amt_m4`, `pay_amt_m5`, `pay_amt_m6` | Total amount repaid across the observed months (NT$). | 0.00% | 689 | Yes |
| `avg_payment_6m` | `mean(pay_amt_m1..m6)` | `pay_amt_m1`, `pay_amt_m2`, `pay_amt_m3`, `pay_amt_m4`, `pay_amt_m5`, `pay_amt_m6` | Average monthly payment (NT$). | 0.00% | 114.8333 | Yes |

### Delinquency

| Feature | Formula | Source columns | Interpretation | Missing % | Example | Used for ML |
|---|---|---|---|---|---|---|
| `delinquent_months_count` | `count(pay_status_m1..m6 >= 1)` | `pay_status_m1`, `pay_status_m2`, `pay_status_m3`, `pay_status_m4`, `pay_status_m5`, `pay_status_m6` | Number of observed months in which the client was behind on payment. Measures how often, not how badly. | 0.00% | 2 | Yes |
| `max_delinquency` | `max(pay_status_m1..m6), floored at 0` | `pay_status_m1`, `pay_status_m2`, `pay_status_m3`, `pay_status_m4`, `pay_status_m5`, `pay_status_m6` | Worst payment delay observed, in months. Measures severity. Negative status codes (paid in full / no transaction) are floored to 0 so they do not read as negative severity. | 0.00% | 2 | Yes |
| `current_delinquency` | `max(pay_status_m1, 0)` | `pay_status_m1` | Payment delay in the most recent month, in months. The closest available signal to present-day status. | 0.00% | 2 | Yes |
| `ever_delinquent` | `any(pay_status_m1..m6 >= 1)` | `pay_status_m1`, `pay_status_m2`, `pay_status_m3`, `pay_status_m4`, `pay_status_m5`, `pay_status_m6` | True when the client was behind on payment in any observed month. | 0.00% | True | Yes |
| `is_currently_delinquent` | `current_delinquency >= 1` | `current_delinquency` | True when the client was behind on payment in the most recent month. | 0.00% | True | Yes |
| `delinquency_trend` | `compare count of delinquent months in m1-m3 (recent) against m4-m6 (older): more recent = Worsening, fewer = Improving, equal = Stable` | `pay_status_m1`, `pay_status_m2`, `pay_status_m3`, `pay_status_m4`, `pay_status_m5`, `pay_status_m6` | Direction of delinquency over the observed window. 'Worsening' means the client was late more often in the recent three months than in the earlier three. | 0.00% | Worsening | No |

### Balance trajectory

| Feature | Formula | Source columns | Interpretation | Missing % | Example | Used for ML |
|---|---|---|---|---|---|---|
| `avg_bill_6m` | `mean(bill_amt_m1..m6)` | `bill_amt_m1`, `bill_amt_m2`, `bill_amt_m3`, `bill_amt_m4`, `bill_amt_m5`, `bill_amt_m6` | Average statement balance across the observed months (NT$). | 0.00% | 1,284 | Yes |
| `bill_volatility` | `std(bill_amt_m1..m6)` | `bill_amt_m1`, `bill_amt_m2`, `bill_amt_m3`, `bill_amt_m4`, `bill_amt_m5`, `bill_amt_m6` | Variability of the statement balance. High values indicate erratic spending rather than a steady carried balance. | 0.00% | 1,608.1438 | Yes |
| `bill_trend_slope` | `least-squares slope of statement balance against month position, ordered oldest to newest (bill_amt_m6 -> bill_amt_m5 -> bill_amt_m4 -> bill_amt_m3 -> bill_amt_m2 -> bill_amt_m1)` | `bill_amt_m6`, `bill_amt_m5`, `bill_amt_m4`, `bill_amt_m3`, `bill_amt_m2`, `bill_amt_m1` | Average change in statement balance per month (NT$/month). Positive means the balance is growing over time; negative means it is being paid down. | 0.00% | 844.5714 | Yes |
| `balance_growth_6m` | `(bill_amt_m1 - bill_amt_m6) / abs(bill_amt_m6)` | `bill_amt_m1`, `bill_amt_m6` | Proportional change in balance from the oldest to the most recent month. Undefined (null) when the starting balance was zero. CAVEAT: a very small starting balance produces an extremely large growth figure, so this measure is unbounded and strongly right-skewed. Report its median rather than its mean, and prefer bill_trend_slope (denominated in NT$/month) for modelling. | 13.40% | -0.1776 | No |

### Bands

| Feature | Formula | Source columns | Interpretation | Missing % | Example | Used for ML |
|---|---|---|---|---|---|---|
| `limit_band` | `quantile band of credit_limit into 4 groups` | `credit_limit` | Credit-limit band, used as a proxy for credit capacity. This is a CREDIT LIMIT, not income - the dataset contains no income column, so no income banding is possible. Bands are quantiles, so each holds a similar number of clients. | 0.00% | NT$10k-50k | No |
| `age_band` | `fixed bins on age: Under 30, 30-39, 40-49, 50-59, 60+` | `age` | Age group. Fixed bands are used rather than quantiles so the groups stay comparable across any filtered subset. | 0.00% | Under 30 | No |

---

## 5b. Derived analytical bands (computed on demand)

These groupings are computed when a view needs them rather than stored as
columns, so they always reflect the current filtered selection. They are
documented here because they appear as axes and groupings in the dashboard.

| Band | Definition | Where it is used |
|---|---|---|
| `utilisation_band` | Fixed bands on `utilisation_latest`: negative/zero, low (<30%), moderate (30-60%), high (60-threshold), very high (threshold+). The top boundary is the configured `HIGH_UTILISATION_THRESHOLD`. | Financial Analytics (default rate by utilisation), Executive Overview (utilisation finding) |
| Payment-delay group | `current_delinquency` bucketed into no delay, 1 month, 2 months, 3+ months, using documented delay codes only. | Executive Overview (delinquency snapshot), Financial Analytics |
| Payment segment | Derived from `is_full_payer` and `is_revolver`: full payers, partial payers, revolvers. Defined purely by repayment behaviour, never by outcome, so outcome comparisons are not circular. | Financial Analytics (behaviour comparison) |
| `segment` | K-Means cluster assignment over 7 scaled behavioural features. Fitted on the whole portfolio, so the assignment is stable regardless of dashboard filters. See section 5c. | Customer Segmentation |

Full banding methodology is shown in the dashboard's Methodology page and is
generated from `src/filtering.py`, so the two cannot diverge.

---

## 5c. Segmentation assignment

The Customer Segmentation page assigns every customer to a behavioural segment. The assignment is computed on demand rather than stored as a column, because it depends on the fitted model rather than on the row itself.

| Property | Value |
|---|---|
| Method | K-Means clustering |
| Clustering features | `credit_limit`, `utilisation_mean_6m`, `repayment_ratio_capped_mean`, `delinquent_months_count`, `bill_trend_slope`, `utilisation_volatility`, `months_zero_payment` |
| Scaler | Robust (median / IQR), selected by measured silhouette: 0.4035 against 0.2879 for standard scaling |
| Number of segments | 3, the measured silhouette optimum across K = 2-8 |
| Silhouette sampling | Scored on a fixed-seed sample of ~5,000 rows, stratified by segment so every segment is represented in proportion to its real size. A plain random subset can omit a small segment and leave the metric undefined; stratification prevents that by construction. Inertia and Davies-Bouldin use all 30,000 rows. |
| Excluded by policy | `default_next_month` (outcome leakage), `sex`, `marriage`, `education`, `age` (fairness), `client_id` and bookkeeping flags |
| Missing-value handling | `repayment_ratio_capped_mean` is undefined for 1,406 customers (4.69%); the median is imputed so no customer is dropped. An imputed value is not an observed customer value. |
| Naming | Segment names are composed at runtime from each centroid's standardised distance from the portfolio mean, so every word traces to a measured value. |
| Observed default rate | Calculated per segment **after** clustering, for profiling only. It played no part in forming the segments and is not a prediction. |

**Leakage guarantee.** The outcome column is blocked from the feature matrix by policy, and a permutation test in the suite shuffles the target and confirms the cluster assignments do not change.

---

## 6. Features NOT created, and why

This section is deliberate. The project brief lists several standard financial
ratios as examples. This dataset cannot support them, and they have **not** been
approximated from unrelated fields.

| Feature | Would require | Status |
|---|---|---|
| `debt_to_income_ratio` | `income` | Required column(s) ['income'] do not exist in this dataset. This feature is deliberately not computed rather than approximated from unrelated fields. |
| `savings_rate` | `savings`, `income` | Required column(s) ['savings', 'income'] do not exist in this dataset. This feature is deliberately not computed rather than approximated from unrelated fields. |
| `expense_ratio` | `expenses`, `income` | Required column(s) ['expenses', 'income'] do not exist in this dataset. This feature is deliberately not computed rather than approximated from unrelated fields. |
| `loan_to_income_ratio` | `loan_amount`, `income` | Required column(s) ['loan_amount', 'income'] do not exist in this dataset. This feature is deliberately not computed rather than approximated from unrelated fields. |
| `income_band` | `income` | Required column(s) ['income'] do not exist in this dataset. This feature is deliberately not computed rather than approximated from unrelated fields. |

### Analytical capabilities unavailable in this dataset

| Capability | Explanation |
|---|---|
| income | No income column exists in this dataset. Debt-to-income and income-band analyses are therefore not computed. `credit_limit` is used as a credit-capacity proxy and is always labelled as a credit limit, never as income. |
| savings | No savings column exists, so savings rate is not computed. |
| expenses | No expense column exists, so expense ratio is not computed. |
| credit_score | No bureau credit score exists. The risk model's predicted probability serves as an internal risk score and is always labelled as model output. |
| employment | No employment-status column exists. |
| loan_amount | This is a revolving credit-card portfolio, not an instalment loan book, so there is no loan amount or loan term. |
| date | There is no calendar date column. The six monthly columns form a panel, which supports period-over-period comparison but not date-indexed forecasting. |

### What is used instead

| Unavailable measure | Substitute actually used | Why the substitute is valid |
|---|---|---|
| Debt-to-income ratio | Credit utilisation (`bill_amt_m1 / credit_limit`) | Both express debt against capacity to carry it. Utilisation uses the granted credit limit as the capacity measure instead of income. |
| Income band | Credit-limit band (`limit_band`) | A lender sets the limit partly from assessed capacity, so it is a coarse proxy. It is named and described as a credit limit throughout, never as income. |
| Bureau credit score | Model-predicted default probability | Derived from this dataset's own behavioural features and always labelled as model output, never presented as an external score. |
| Transaction volume | Share of accounts with a positive balance | An activity proxy. The dataset holds no transaction counts. |

---

## 7. Notes on data quality

The source file has **zero missing values and zero exact duplicate rows**, so a
quality check that only counted blanks would report a perfect score and tell an
analyst nothing. The defects in this data are semantic:

- **delinquent_without_balance** (medium): 2,068 client-month record(s), spanning 2,037 distinct client(s), show a payment delay of one month or more while the statement balance is zero or negative. Affected rows: 2,037 (6.79%).
- **status_payment_contradiction** (medium): 2,062 client-month record(s), spanning 1,854 distinct client(s), are marked 'paid in full' (-1) yet record a zero payment against a positive prior statement balance. Affected rows: 1,854 (6.18%).
- **undocumented_category_code** (medium): 'education_code' (education level) contains code(s) [0, 5, 6] that appear in the data but are not defined in the source documentation (documented codes: [1, 2, 3, 4]). Affected rows: 345 (1.15%).
- **undocumented_category_code** (medium): 'marriage_code' (marital status) contains code(s) [0] that appear in the data but are not defined in the source documentation (documented codes: [1, 2, 3]). Affected rows: 54 (0.18%).
- **duplicate_rows_excluding_id** (medium): 35 row(s) are identical once 'client_id' is excluded. These may be genuinely distinct clients with coincidentally identical records, or repeated data entry. Affected rows: 35 (0.12%).
- **undocumented_category_code** (low): Repayment-status codes [-2, 0] occur in 120,334 client-month cell(s) across 6 monthly column(s), affecting 25,939 distinct client(s). These codes are not defined in the source paper; the common reading is -2 = no transaction and 0 = revolving credit used. Affected rows: 25,939 (86.46%).
- **over_limit_balance** (info): 3,931 client(s) had a statement balance above their credit limit in at least one month, i.e. utilisation above 100%. Affected rows: 3,931 (13.10%).
- **negative_statement_balance** (info): Negative statement balances occur in 6 monthly column(s), up to 688 rows in a single month. A negative balance means the account is in credit, usually from an overpayment or a refund. Affected rows: 688 (2.29%).

---

## 8. Target variable

| Property | Value |
|---|---|
| Column | `default_next_month` |
| Definition | Client defaulted on the following month's payment |
| Type | Binary (0 = no default, 1 = default) |
| Positive class (default) | 6,636 |
| Negative class (no default) | 23,364 |
| Observed default rate | 22.12% |
| Majority-class baseline accuracy | 77.88% |

The majority-class baseline is the reason accuracy is not used as the headline
metric: always predicting 'no default' scores 77.88% while identifying no
defaulters at all. ROC-AUC, PR-AUC and recall on the default class are
reported instead.

**Leakage note.** `pay_status_m1` records the September repayment status and
the target is the October default. That is legitimately prior information, not
leakage, but it is expected to dominate feature importance and is documented
so a reader is not surprised by it.

---

_Generated from a live pipeline run over 30,000 rows._
