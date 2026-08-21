# Empirical Evaluation of Equity Statistical Alpha Signals on the Historical S&P 100 Universe (2014–2024)

**A Formal Academic Research Report & Reproducibility Audit**  
**Specification Reference:** Quantitative Research Protocol v1.2  
**Date:** August 2026  
**Point-in-Time Universe:** S&P 100 Historical Constituents as of 2014-01-01  
**Sample Period:** 2014-01-02 to 2024-12-31 (11 Calendar Years, 2,768 NYSE Trading Sessions)  

---

## 1. Abstract

This study presents a comprehensive, point-in-time empirical evaluation of four canonical cross-sectional equity statistical alpha signals—Price Momentum ($H_1$), Short-Term Mean Reversion ($H_2$), Low Volatility ($H_3$), and Abnormal Volume ($H_4$)—as well as a composite baseline and non-linear machine learning models on the historical S&P 100 constituent universe from January 2014 through December 2024. The research design enforces strict survivorship bias mitigation, causal feature engineering, point-in-time corporate successor identity tracking, Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) statistical inference, multiple-testing controls (Bonferroni and Benjamini-Hochberg FDR), single-pass basis-point transaction cost accounting, 7-window expanding walk-forward validation, and an isolated 2024 holdout.

Under the pre-specified confirmatory inference protocol, **no statistically significant evidence of predictive alpha was detected** for any of the four individual hypotheses ($H_1\text{ HAC }p=0.9107$, $H_2\text{ HAC }p=0.9357$, $H_3\text{ HAC }p=0.8856$, $H_4\text{ HAC }p=0.3114$) or the Combined Baseline (Mean $\text{Rank IC} = -0.0005$, $\text{HAC }p=0.9728$). In realistic event-driven portfolio simulations with 15 basis points of one-way transaction friction, the combined strategy generated an annualized compound return (CAGR) of $-27.45\%$, an annualized volatility of $9.12\%$, and a Sharpe ratio of $-3.472$, driven by high portfolio turnover ($\approx 205\times$ annually) that incurred $\$9.90\text{M}$ in cumulative transaction friction. Non-linear machine learning architectures (OLS, Ridge, Lasso, XGBoost) similarly failed to produce statistically significant alpha improvements over the baseline ($\Delta\text{IC}\le +0.0117$, $\text{HAC }p\ge 0.7540$). These empirical findings robustly illustrate the high informational efficiency of U.S. mega-cap equities and highlight the critical role of transaction cost drag in quantitative portfolio management.

---

## 2. Research Question & Objectives

The primary research objective is to test whether canonical equity anomalies documented in the academic literature provide statistically significant and economically exploitable predictive power within the large-cap U.S. equity universe after rigorous adjustments for:
1. Point-in-time survivorship bias and corporate identity transitions.
2. Serial correlation and overlapping forecast horizons via HAC inference.
3. Family-wise error rate (FWER) and false discovery rate (FDR) inflation.
4. Realistic execution timing, hard portfolio constraints, and market frictions.
5. Out-of-sample temporal stability across walk-forward splits and machine learning benchmarks.

---

## 3. Data and Point-in-Time Universe

### 3.1 Universe Specification
The study defines a frozen point-in-time universe of the **100 constituents of the S&P 100 index as of January 1, 2014** (`data/universe/sp100_20140101.csv`). To prevent survivorship bias, securities that subsequently merged, delisted, or underwent restructuring remain in the universe definition for their active trading spans.

### 3.2 Corporate Identity & Delisting Audit
A complete forensic identity audit established the following empirical status for the 100 historical constituents:
- **Continuous Historical Identities ($N=92$)**: Valid, continuous trading histories from 2014-01-02 through 2024-12-31.
- **Genuinely Terminated Identity ($N=1$)**: Time Warner Inc. (`TWX`), which actively traded until its acquisition on 2018-06-14. All holdings and trades for `TWX` cease permanently after 2018-06-15.
- **Unavailable Legacy Series ($N=7$)**: `APC`, `DOW` (legacy Dow Chemical), `EMC` (legacy EMC Corp), `FOXA` (legacy 21st Century Fox), `MON`, `RTN` (legacy Raytheon Co.), and `WAG`. These historical series are not available from the public Yahoo Finance endpoint. In adherence to strict research integrity protocols, **no synthetic data were fabricated** and **no recycled successor identities were substituted**.
- **Provider Lookup Mappings**:
  - `BRK.B` $\to$ `BRK-B` (standard Yahoo provider hyphenation).
  - `FB` $\to$ `META` (corporate rebranding).
  - `UTX` $\to$ `RTX` (legal entity corporate survivor following the 2020 United Technologies / Raytheon merger).
  - `BK` (retained as research identity `BK`; mapped to provider lookup key `BNY`).

The final cleaned research panel contains **255,777 rows** across **93 usable historical constituents** over 2,768 trading sessions.

---

## 4. Data Provenance and Quality

### 4.1 Split and Dividend Adjustments
- **Split Adjustments**: Discrete split ratios are applied strictly to historical share volume ($V_{\text{adj}} = V \times \text{split\_factor}$).
- **Cash Distributions**: Dividends and cash distributions adjust prices (`adj_close`) but do not alter raw trade share volumes, preventing distortion of liquidity metrics.

### 4.2 Data Quality Screening
Automated data quality checks confirmed:
- Zero non-positive prices or zero trade volumes on active sessions.
- Zero high-low inversions ($\text{High} \ge \text{Low}$, $\text{High} \ge \text{Close}$, $\text{Low} \le \text{Open}$).
- Zero synthetic backfilling or forward imputation.

---

## 5. Trading Calendar

All observations align to the authoritative **New York Stock Exchange (NYSE)** trading calendar across the 11-year span:
- Total authoritative sessions: **2,768 trading days**.
- Zero weekend bars and zero exchange-holiday bars.
- Earliest session: `2014-01-02`.
- Latest session: `2024-12-31`.

---

## 6. Feature Engineering

All features are calculated causally using strictly trailing information available at market close $t$:
1. **Jegadeesh-Titman 12–1 Momentum (`ret_12_1_mom`)**: Trailing 252-day return skipping the most recent 21 trading days:
   $$\text{MOM}_{i,t} = \frac{P_{i, t-21} - P_{i, t-252}}{P_{i, t-252}}$$
2. **Short-Term Price Reversal Z-Score (`zscore_price_20`)**: Inverted distance of current close from the 20-day simple moving average, normalized by trailing 20-day standard deviation:
   $$Z_{i,t} = \frac{P_{i,t} - \text{SMA}_{20}(P_i)_t}{\sigma_{20}(P_i)_t}$$
3. **Realized Volatility (`realized_vol_20`)**: Trailing 20-day sample standard deviation of daily close-to-close log returns, annualized:
   $$\sigma_{i,t} = \sqrt{252} \times \sqrt{\frac{1}{19} \sum_{k=0}^{19} (r_{i, t-k} - \bar{r}_i)^2}$$
4. **Abnormal Relative Volume (`volume_relative_20`)**: Log-ratio of current split-adjusted volume to its trailing 20-day moving average:
   $$\text{VolRel}_{i,t} = \log\left(\frac{V_{i,t}}{\text{SMA}_{20}(V_i)_t}\right)$$

---

## 7. Four Confirmatory Alpha Hypotheses

The confirmatory research registry pre-specifies four primary hypotheses, cryptographically frozen under SHA-256 hash `3caabc3550691880f4d66cc24a7a4ad14b76f623c169fd065cd2470fb3025840`:

| Hypothesis ID | Anomaly Name | Signal Mathematical Formulation | Forward Return Target | Theoretical Rationale | Expected Sign |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **$H_1$** | **Price Momentum** | $\text{Signal}_{i,t} = \text{MOM}_{i,t}$ | $h=20\text{d}$ (`fwd_ret_20d`) | Underreaction to persistent fundamental news | Positive ($\text{IC} > 0$) |
| **$H_2$** | **Mean Reversion** | $\text{Signal}_{i,t} = -Z_{i,t}$ | $h=5\text{d}$ (`fwd_ret_5d`) | Liquidity provision / overreaction correction | Positive ($\text{IC} > 0$) |
| **$H_3$** | **Low Volatility** | $\text{Signal}_{i,t} = -\sigma_{i,t}$ | $h=20\text{d}$ (`fwd_ret_20d`) | Leverage constraints / lottery preference | Positive ($\text{IC} > 0$) |
| **$H_4$** | **Abnormal Volume** | $\text{Signal}_{i,t} = \text{sign}(r_{i,t}) \times \text{VolRel}_{i,t}$ | $h=5\text{d}$ (`fwd_ret_5d`) | Volume-confirmed institutional price discovery | Positive ($\text{IC} > 0$) |

---

## 8. Statistical Methodology

### 8.1 Daily Cross-Sectional Information Coefficient (IC)
For each trading date $t$, the cross-sectional Spearman rank correlation between the signal vector and the realized forward return vector is calculated:
$$\text{IC}_t = \rho_s(\text{Signal}_{\cdot, t}, \text{ForwardReturn}_{\cdot, t \to t+h})$$

### 8.2 Primary Statistical Inference: Newey-West HAC Test
Because overlapping forward returns induce moving-average serial correlation of order $h-1$, standard errors on the mean IC series $\overline{\text{IC}} = \frac{1}{T}\sum_{t=1}^T \text{IC}_t$ are estimated using the Newey-West HAC estimator with Bartlett kernel lag $L = \max(1, h)$:
$$\widehat{\text{Var}}(\overline{\text{IC}}) = \frac{1}{T} \left( \hat{\gamma}_0 + 2 \sum_{l=1}^L \left(1 - \frac{l}{L+1}\right) \hat{\gamma}_l \right), \quad t_{\text{HAC}} = \frac{\overline{\text{IC}}}{\sqrt{\widehat{\text{Var}}(\overline{\text{IC}})}}$$

### 8.3 Non-Parametric Resampling
- **Stationary Block Bootstrap**: 1,000 resamples via Politis & Romano (1994) with mean block length $\bar{B}=10$ days.
- **Within-Date Permutation Test**: 1,000 replications cross-sectionally shuffling ticker signals on each date while keeping forward returns fixed.

---

## 9. Multiple-Testing Control

To guard against data snooping across the confirmatory family ($m=4$), two formal adjustments are applied to the primary HAC $p$-values:
1. **Bonferroni FWER Adjustment**: $\alpha_{\text{adjusted}} = \frac{\alpha}{m} = \frac{0.05}{4} = 0.0125$.
2. **Benjamini-Hochberg FDR Control**: Controls the expected proportion of false discoveries at $q = 0.05$.

---

## 10. Confirmatory Empirical Results

The empirical evaluation was executed across all 2,768 trading sessions. The results are summarized below:

| Hypothesis | Target Horizon | Mean Rank IC | IC Std Dev | ICIR | HAC $t$-stat ($L=h$) | HAC $p$-value | Bootstrap 95% CI | Permutation $p$-value | Bonferroni $p$ | BH-FDR $p$ | Confirmatory Discovery? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H_1$: Momentum** | $20\text{d}$ | $+0.0021$ | $0.2618$ | $+0.008$ | $+0.112$ | **$0.9107$** | $[-0.0350, +0.0384]$ | $0.3516$ | $1.0000$ | $0.9357$ | **NO** |
| **$H_2$: Mean Reversion** | $5\text{d}$ | $+0.0005$ | $0.2034$ | $+0.003$ | $+0.081$ | **$0.9357$** | $[-0.0122, +0.0134]$ | $0.7852$ | $1.0000$ | $0.9357$ | **NO** |
| **$H_3$: Low Volatility** | $20\text{d}$ | $-0.0027$ | $0.2721$ | $-0.010$ | $-0.144$ | **$0.8856$** | $[-0.0396, +0.0340]$ | $0.1698$ | $1.0000$ | $0.9357$ | **NO** |
| **$H_4$: Abnormal Volume** | $5\text{d}$ | $-0.0026$ | $0.1343$ | $-0.019$ | $-1.012$ | **$0.3114$** | $[-0.0076, +0.0024]$ | $0.2098$ | $1.0000$ | $0.9357$ | **NO** |

**Statistical Conclusion**: All four hypotheses fail to reach statistical significance under the primary HAC inference ($p > 0.30$). After Bonferroni and Benjamini-Hochberg adjustments, the number of confirmed discoveries is **zero**.

*Figure 2: Primary Confirmatory Information Coefficient Forest Plot with Newey-West HAC 95% Confidence Intervals (`reports/figures/fig2_confirmatory_ic_forest_plot.png`).*

---

## 11. Combined Baseline Signal

The composite baseline signal combines the standardized ranks of $H_1$ through $H_4$ with equal cross-sectional weighting, strictly returning NaN if any component is missing, followed by re-ranking to $[-1, 1]$:
$$\text{Signal}_{\text{Combined}, i, t} = \text{Rank}\left( \frac{1}{4}\sum_{k=1}^4 \text{Rank}(S_{k, i, t}) \right)$$

- **Target Horizon**: $20\text{d}$ (`fwd_ret_20d`)
- **Mean Rank IC**: **$-0.0005$** (IC Std: $0.2224$, $\text{ICIR} = -0.002$)
- **HAC $t$-statistic**: **$-0.034$** ($\text{HAC }p = \mathbf{0.9728}$)
- **Bootstrap 95% CI**: $[-0.0298, +0.0269]$
- **Permutation $p$-value**: $0.7982$

The estimated cross-sectional rank correlation for the combined signal is statistically indistinguishable from zero out-of-sample (Mean Rank IC = -0.0005, HAC p = 0.9728).

---

## 12. Portfolio Construction & Hard Risk Constraints

Target weights $w_{i,t}$ are computed at the close of date $t$ and projected onto the convex constraint polytope:
$$\min_w \frac{1}{2} \|w - w_{\text{raw}}\|_2^2 \quad \text{s.t.} \quad |w_i| \le 0.10, \quad \sum_i |w_i| \le 1.00, \quad \left|\sum_i w_i\right| \le 0.20$$

### Target vs. Realized Mark-to-Market Gross Exposure
- **Target Constraint Enforcement**: Every target weight vector generated across all 2,768 trading sessions strictly satisfied target gross $\le 1.0000$, target net $= 0.0000$, and single-position weight $\le 0.1000$.
- **Realized Mark-to-Market Gross Exposure**: Between rebalance dates, price movements and transaction cost deductions from cash cause the realized mark-to-market ratio $\frac{\sum |S_i P_i|}{\text{Equity}}$ to fluctuate. Under severe drawdown conditions, intra-day price moves briefly pushed this ratio to a maximum of $1.1754$, which is the arithmetic result of price drift on a long/short ledger rather than a target leverage violation.

---

## 13. Transaction-Cost Model

Execution economics follow the single-pass basis-point specification defined in Protocol v1.2:
$$\text{Fill Price} = P_t \times \left(1 + \text{sign}(\text{direction}) \times \frac{\text{Spread\_bps} + \text{Slippage\_bps}}{10\,000}\right)$$
$$\text{Commission (USD)} = \max\left(\text{Min\_Commission}, \text{Notional} \times \frac{\text{Commission\_bps}}{10\,000}\right)$$

### Cost Regimes:
1. **Zero-Cost (`zero`)**: $0.0\text{ bps}$ commission, $0.0\text{ bps}$ spread, $0.0\text{ bps}$ slippage (**$0.0\text{ bps}$ total one-way friction**).
2. **Low-Cost (`low`)**: $2.0\text{ bps}$ commission, $3.0\text{ bps}$ spread, $2.0\text{ bps}$ slippage (**$7.0\text{ bps}$ total one-way friction**).
3. **Medium-Cost (Base Case, `medium` / `base_case`)**: $5.0\text{ bps}$ commission, $5.0\text{ bps}$ spread, $5.0\text{ bps}$ slippage (**$15.0\text{ bps}$ total one-way friction**).
4. **High-Cost (`high`)**: $10.0\text{ bps}$ commission, $10.0\text{ bps}$ spread, $10.0\text{ bps}$ slippage (**$30.0\text{ bps}$ total one-way friction**).
5. **Very-High-Cost (`very_high`)**: $15.0\text{ bps}$ commission, $15.0\text{ bps}$ spread, $20.0\text{ bps}$ slippage (**$50.0\text{ bps}$ total one-way friction**).

---

## 14. Full-Sample Backtest Results (2014–2024)

Simulated with $\$10,000,000$ initial cash across the full 11-year sample period:

| Strategy | Cost Regime | CAGR | Ann. Vol | Sharpe | Sortino | Max DD | Hit Rate | Profit Factor | Turnover | Total Costs (USD) | Terminal Equity (USD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EqualWeightLongShort** | **Zero-Cost** | $-1.40\%$ | $9.07\%$ | $-0.110$ | $-0.099$ | $-39.02\%$ | $49.4\%$ | $0.985$ | $204.7\times$ | $\$0$ | $\$8,567,846$ |
| | **Base-Case ($15\text{ bps}$)** | **$-27.45\%$** | **$9.12\%$** | **$-3.472$** | **$-3.122$** | **$-97.06\%$** | $45.6\%$ | $0.627$ | $204.7\times$ | **$\$9,903,113$** | **$\$294,390$** |
| **SignalWeightedLongShort** | **Zero-Cost** | $-1.28\%$ | $7.01\%$ | $-0.149$ | $-0.134$ | $-32.28\%$ | $49.5\%$ | $0.980$ | $160.8\times$ | $\$0$ | $\$8,679,702$ |
| | **Base-Case ($15\text{ bps}$)** | **$-22.43\%$** | **$7.06\%$** | **$-3.562$** | **$-3.220$** | **$-93.88\%$** | $45.4\%$ | $0.612$ | $160.8\times$ | **$\$9,298,337$** | **$\$614,296$** |
| **InverseVolatilityWeighted** | **Zero-Cost** | $-0.88\%$ | $6.36\%$ | $-0.107$ | $-0.097$ | $-25.86\%$ | $49.6\%$ | $0.985$ | $161.3\times$ | $\$0$ | $\$9,075,709$ |
| | **Base-Case ($15\text{ bps}$)** | **$-22.17\%$** | **$6.40\%$** | **$-3.882$** | **$-3.502$** | **$-93.64\%$** | $45.0\%$ | $0.589$ | $161.3\times$ | **$\$9,336,448$** | **$\$637,390$** |

### Reconciliation of Cost Drag:
- For Equal-Weight: Annualized cost drag $= \text{CAGR}_{\text{Zero}} - \text{CAGR}_{\text{Base}} = \mathbf{+26.06\%}$ (Sharpe drag $= +3.362$).
- The strategy trades $\approx 205\times$ its capital per year, causing transaction friction to consume virtually all invested capital.

---

## 15. Cost Sensitivity Analysis

Evaluating the EqualWeight Long-Short strategy across the 5 predefined cost regimes demonstrates strict monotonic performance degradation:

| Regime | Total One-Way Friction | CAGR | Ann. Vol | Sharpe | Max DD | Total Friction Paid (USD) | Terminal Equity (USD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero** | $0\text{ bps}$ | $-1.40\%$ | $9.07\%$ | $-0.110$ | $-39.02\%$ | $\$0$ | $\$8,567,846$ |
| **Low** | $7\text{ bps}$ | $-14.54\%$ | $9.08\%$ | $-1.684$ | $-82.35\%$ | $\$7,936,759$ | $\$1,779,240$ |
| **Medium (Base)** | $15\text{ bps}$ | $-27.45\%$ | $9.12\%$ | $-3.472$ | $-97.06\%$ | $\$9,903,113$ | $\$294,390$ |
| **High** | $30\text{ bps}$ | $-46.66\%$ | $9.23\%$ | $-6.753$ | $-99.90\%$ | $\$10,388,886$ | $\$10,044$ |
| **Very High** | $50\text{ bps}$ | $-60.69\%$ | $9.51\%$ | $-9.751$ | $-100.00\%$ | $\$10,400,683$ | $\$352$ |

*Figure 3: Transaction Cost Sensitivity Curves across Five Cost Regimes (`reports/figures/fig3_cost_sensitivity_curves.png`).*

---

## 16. Walk-Forward Expanding Windows Validation

The strategy was evaluated across 7 expanding windows. The 6 development windows (2018–2023) were executed as **independent one-year simulations**, each initialized with fresh $\$10,000,000$ capital.

| Window | Classification | Historical In-Sample Warmup | Out-of-Sample Test Window | OOS CAGR | OOS Vol | OOS Sharpe | OOS Max DD | PSR ($SR > 0$) | Total Costs (USD) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **W1** | Dev OOS | 2014-01-01 to 2017-12-31 | 2018-01-01 to 2018-12-31 | $-29.72\%$ | $4.83\%$ | $-7.277$ | $-29.63\%$ | $0.0\%$ | $\$2,889,458$ |
| **W2** | Dev OOS | 2014-01-01 to 2018-12-31 | 2019-01-01 to 2019-12-31 | $-26.52\%$ | $7.58\%$ | $-4.025$ | $-26.52\%$ | $0.0\%$ | $\$2,717,907$ |
| **W3** | Dev OOS | 2014-01-01 to 2019-12-31 | 2020-01-01 to 2020-12-31 | $-30.12\%$ | $20.32\%$ | $-1.661$ | $-40.62\%$ | $4.5\%$ | $\$2,989,049$ |
| **W4** | Dev OOS | 2014-01-01 to 2020-12-31 | 2021-01-01 to 2021-12-31 | $-41.67\%$ | $9.06\%$ | $-5.898$ | $-41.75\%$ | $0.0\%$ | $\$2,790,059$ |
| **W5** | Dev OOS | 2014-01-01 to 2021-12-31 | 2022-01-01 to 2022-12-31 | $-27.19\%$ | $9.54\%$ | $-3.277$ | $-27.09\%$ | $0.1\%$ | $\$2,895,643$ |
| **W6** | Dev OOS | 2014-01-01 to 2022-12-31 | 2023-01-01 to 2023-12-31 | $-33.05\%$ | $7.33\%$ | $-5.431$ | $-32.97\%$ | $0.0\%$ | $\$2,571,953$ |

### Aggregated Development OOS (2018–2023):
- **Concatenated OOS CAGR**: **$-31.58\%$**
- **Annualized Volatility**: **$10.97\%$**
- **Sharpe Ratio**: **$-3.403$**
- **Maximum Drawdown**: **$-89.71\%$**
- **Cumulative Trading Costs**: $\$16.85\text{M}$ paid across the 6 separate $\$10\text{M}$ initializations ($\approx \$2.8\text{M}$/year).

*Figure 4: Annual Out-of-Sample Sharpe Performance across Expanding Development Windows (2018–2023) and 2024 Final Holdout (`reports/figures/fig4_walk_forward_oos_performance.png`).*

---

## 17. 2024 Final Holdout (Window 7)

Window 7 was maintained as an untouched final holdout, evaluated strictly once with zero parameter tuning:
- **Test Period**: `2024-01-01` to `2024-12-31` (252 trading sessions)
- **OOS CAGR**: **$-22.83\%$**
- **Annualized Volatility**: **$5.48\%$**
- **OOS Sharpe Ratio**: **$-4.696$**
- **OOS Sortino Ratio**: **$-4.332$**
- **Maximum Drawdown**: **$-23.73\%$**
- **Probabilistic Sharpe Ratio**: **$0.0\%$**
- **Trading Costs Paid**: **$\$3,060,870$**
- **Terminal Holdout Equity**: **$\$7,716,634$**

---

## 18. Machine-Learning Benchmark

Four ML models were fitted strictly on historical in-sample training splits and evaluated out-of-sample on the common 20-day target (`fwd_ret_20d`) across all 7 expanding windows ($N=1,760$ OOS days):

| Model | OOS Target Horizon | ML Mean Rank IC | ML IC Std Dev | ML ICIR | ML HAC $t$-stat ($L=20$) | ML HAC $p$-value | Baseline Mean IC | $\Delta\text{IC}$ ($\text{ML} - \text{Base}$) | $\Delta\text{IC}$ HAC $t$-stat | $\Delta\text{IC}$ HAC $p$-value | Significant Outperformance? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OLS** | $20\text{d}$ | $+0.0108$ | $0.2606$ | $+0.042$ | $+0.50$ | $0.6177$ | $-0.0009$ | $+0.0117$ | $+0.31$ | $0.7540$ | **NO** |
| **Ridge** | $20\text{d}$ | $+0.0108$ | $0.2606$ | $+0.042$ | $+0.50$ | $0.6177$ | $-0.0009$ | $+0.0117$ | $+0.31$ | $0.7540$ | **NO** |
| **Lasso** | $20\text{d}$ | $+0.0105$ | $0.2626$ | $+0.040$ | $+0.48$ | $0.6303$ | $-0.0009$ | $+0.0115$ | $+0.30$ | $0.7620$ | **NO** |
| **XGBoost** | $20\text{d}$ | $+0.0066$ | $0.1926$ | $+0.034$ | $+0.46$ | $0.6451$ | $-0.0009$ | $+0.0075$ | $+0.26$ | $0.7981$ | **NO** |

*Finding*: No machine learning model achieved statistically significant alpha or significantly outperformed the Combined Baseline under Newey-West HAC inference ($\Delta\text{IC}\text{ HAC }p \ge 0.7540$).

---

## 19. Robustness Analysis

### 19.1 Pre- vs. Post-2020 Subperiod Stability
- **Pre-2020 (2014–2019, $N=1,510\text{d}$)**: $\text{CAGR} = -23.81\%$, $\text{Vol} = 6.35\%$, $\text{Sharpe} = -4.249$, $\text{MaxDD} = -80.46\%$.
- **Post-2020 (2020–2024, $N=1,258\text{d}$)**: $\text{CAGR} = -31.55\%$, $\text{Vol} = 11.59\%$, $\text{Sharpe} = -3.208$, $\text{MaxDD} = -86.95\%$.

### 19.2 Asset Exclusion Jackknife (10% Random Drop, 5 Iterations)
- **Iteration 1**: $\text{CAGR} = -27.79\%$, $\text{Sharpe} = -3.497$.
- **Iteration 2**: $\text{CAGR} = -27.53\%$, $\text{Sharpe} = -3.316$.
- **Iteration 3**: $\text{CAGR} = -27.43\%$, $\text{Sharpe} = -3.489$.
- **Iteration 4**: $\text{CAGR} = -27.21\%$, $\text{Sharpe} = -3.378$.
- **Iteration 5**: $\text{CAGR} = -26.95\%$, $\text{Sharpe} = -3.560$.  
*Finding*: Narrow Sharpe range $[-3.560, -3.316]$ confirms that negative returns are universe-wide and not driven by idiosyncratic outlier stocks.

### 19.3 Extreme-Day Trimming
Trimming the 5 best and 5 worst PnL days ($N=2,758\text{d}$) yields $\text{CAGR} = -27.15\%$ and $\text{Sharpe} = -3.802$, confirming that negative economics are structural rather than outlier-driven.

### 19.4 Stationary Block Bootstrap PnL Confidence Intervals
- **Sharpe Ratio 95% Bootstrap CI**: **$[-4.204, -2.729]$**
- **CAGR 95% Bootstrap CI**: **$[-31.53\%, -23.13\%]$**  
The entire 95% empirical confidence interval lies deeply in negative territory.

### 19.5 Market-Regime Slicing
- **Volatility Regimes**: Low Vol ($\text{Sharpe} = -5.472$), Normal Vol ($\text{Sharpe} = -3.531$), High Vol ($\text{Sharpe} = -2.967$).
- **Trend Regimes**: Bull Trend ($\text{Sharpe} = -3.836$), Bear Trend ($\text{Sharpe} = -3.140$).

### 19.6 Exploratory Extension: Signal Horizon Profile & Half-Life Decay
> **Disclaimer**: This exploratory analysis investigates horizon decay dynamics and is explicitly **not part of the pre-specified confirmatory hypothesis family**. It does not alter the primary confirmatory conclusions or hypothesis registry.

To evaluate whether any signal contains short-lived predictive structure despite failing at its pre-specified horizon, daily cross-sectional Spearman IC was computed across $h \in \{1\text{d}, 5\text{d}, 10\text{d}, 20\text{d}, 40\text{d}, 60\text{d}\}$ using Newey-West HAC standard errors:

| Signal Anomaly | $1\text{d}$ Mean IC | $5\text{d}$ Mean IC | $10\text{d}$ Mean IC | $20\text{d}$ Mean IC | $40\text{d}$ Mean IC | $60\text{d}$ Mean IC | Decay Dynamic |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$H_1$: Momentum** | $+0.0161$ | $+0.0134$ | $+0.0086$ | $+0.0021$ | $-0.0009$ | $-0.0028$ | Monotonic decay toward zero and negative by $40\text{d}$ |
| **$H_2$: Mean Reversion** | $+0.0020$ | $+0.0005$ | $-0.0025$ | $+0.0004$ | $+0.0062$ | $+0.0061$ | Flat near zero across all horizons ($p > 0.60$) |
| **$H_3$: Low Volatility** | $+0.0076$ | $+0.0050$ | $+0.0034$ | $-0.0027$ | $-0.0156$ | $-0.0193$ | Mild positive short-term drifting into negative long-term |
| **$H_4$: Abnormal Volume** | $-0.0048$ | $-0.0026$ | $-0.0033$ | $-0.0015$ | $-0.0009$ | $-0.0018$ | Weak negative correlation across all horizons |

*Figure 5: Exploratory Signal Half-Life & Horizon Decay Profile across 1d–60d Horizons (`reports/figures/fig5_exploratory_ic_horizon_profile.png`).*

---

## 20. Discussion

### 20.1 Market Efficiency in U.S. Mega-Caps
The absence of statistically significant linear or non-linear alpha in the S&P 100 universe is consistent with modern market microstructure and asset pricing literature. Mega-cap equities are the most heavily researched and efficiently priced financial instruments globally. The empirical evidence fails to detect persistent, statistically significant momentum or reversal premiums within this large-cap universe over the 2014–2024 period.

### 20.2 The Primacy of Transaction Cost Drag
The empirical results dramatically demonstrate the destructive impact of portfolio turnover on high-turnover quantitative strategies. With annual turnover exceeding $200\times$, a modest $15\text{ bps}$ one-way transaction friction generates $\approx 26\%$ in annual performance drag, rendering even neutral signals economically disastrous.

---

## 21. Limitations

1. **Constituent Data Availability**: Seven frozen 2014 S&P 100 constituents (`APC`, `DOW`, `EMC`, `FOXA`, `MON`, `RTN`, `WAG`) lack usable historical OHLCV data from the public Yahoo Finance endpoint and were excluded without synthetic fabrication or successor substitution.
2. **Universe Breadth**: The study is restricted to the 93 usable historical constituents of the S&P 100 index. Results should not be generalized to small-cap, international, or less liquid asset classes.
3. **Execution Modeling**: The backtester assumes next-day market-on-close fills with fixed basis-point friction and does not model intra-day order routing, hidden limit order books, or borrow rebate fees.

---

## 22. Conclusion

This research conducted an exhaustive empirical evaluation of equity statistical alpha signals on the S&P 100 universe over 2014–2024. Under rigorous Newey-West HAC inference and multiple-testing corrections, **no statistically significant predictive alpha was detected** for Momentum ($H_1$), Mean Reversion ($H_2$), Low Volatility ($H_3$), Abnormal Volume ($H_4$), or their Composite Baseline. Furthermore, non-linear machine learning models failed to generate significant improvements. Under event-driven portfolio simulations, transaction cost drag dominated strategy economics across all market regimes. The empirical evidence underscores the necessity of strict out-of-sample validation, realistic transaction cost modeling, and robust multiple-testing controls in quantitative finance.

---

## 23. Reproducibility & Cryptographic Artifact Hashes

All empirical results reported herein are fully reproducible using the specified repository code, frozen configurations, and persisted checkpoint artifacts.

| Artifact Description | File Path | SHA-256 Hash Representation |
| :--- | :--- | :--- |
| **Cleaned Market Data Panel** | `data/processed/cleaned_ohlcv.parquet` | `c3d67525d09fc052` (16-character SHA-256 hash prefix) |
| **Engineered Feature Cache** | `data/cache/features_c3d67525d09fc052.parquet` | `18b4358995ae9881` (16-character SHA-256 hash prefix) |
| **Alpha Signal Cache** | `data/cache/signals_c3d67525d09fc052.parquet` | `a8e62995b6a4950f` (16-character SHA-256 hash prefix) |
| **Production Experiment Config** | `configs/default.yaml` | `3e3b02ce22d1afd3af30aa60520eac232c8d2a765f66d170aed514e0406e638c` (full 64-character SHA-256 hash) |
| **Confirmatory Hypothesis Registry** | `src/statistics/multiple_testing.py` | `3caabc3550691880f4d66cc24a7a4ad14b76f623c169fd065cd2470fb3025840` (full 64-character SHA-256 hash) |

### Primary Checkpoint Files:
- **Confirmatory Statistical Evaluation**: [`results/checkpoints/confirmatory_results_c3d67525d09fc052_3caabc3550691880f4d66cc24a7a4ad14b76f623c169fd065cd2470fb3025840.json`](file:///e:/Quant-p1/results/checkpoints/confirmatory_results_c3d67525d09fc052_3caabc3550691880f4d66cc24a7a4ad14b76f623c169fd065cd2470fb3025840.json)
- **Full-Sample Portfolio Backtests**: [`results/checkpoints/portfolio_backtests_c3d67525d09fc052.json`](file:///e:/Quant-p1/results/checkpoints/portfolio_backtests_c3d67525d09fc052.json)
- **Walk-Forward Expanding Windows**: [`results/checkpoints/walk_forward_c3d67525d09fc052.json`](file:///e:/Quant-p1/results/checkpoints/walk_forward_c3d67525d09fc052.json)
- **2024 Final Holdout**: [`results/checkpoints/final_holdout_2024_c3d67525d09fc052.json`](file:///e:/Quant-p1/results/checkpoints/final_holdout_2024_c3d67525d09fc052.json)
- **Machine Learning Benchmark**: [`results/checkpoints/ml_benchmark_c3d67525d09fc052.json`](file:///e:/Quant-p1/results/checkpoints/ml_benchmark_c3d67525d09fc052.json)
- **Robustness Analysis Suite**: [`results/checkpoints/robustness_analysis_c3d67525d09fc052.json`](file:///e:/Quant-p1/results/checkpoints/robustness_analysis_c3d67525d09fc052.json)

**Test Suite Verification**: All **109 unit, integration, and statistical regression tests** pass (`pytest tests/ -q`).
