# Empirical Evaluation of Equity Statistical Alpha Signals on the S&P 100 (2014–2024)

An institutional-grade, reproducible quantitative equity alpha research and event-driven backtesting framework implementing the authoritative **Specification v1.2** for the historical S&P 100 universe over 2,768 NYSE trading sessions.

---

## 1. Research Question

> **Do canonical cross-sectional equity signals produce statistically significant and economically viable alpha in a point-in-time large-cap U.S. equity universe after rigorous multiple-testing control and realistic transaction costs?**

Specifically, this study evaluates four foundational quantitative anomaly categories—Price Momentum, Short-Term Mean Reversion, Low Volatility, and Abnormal Volume Interaction—alongside multi-factor composites and machine-learning models within the historical S&P 100 constituent universe from January 2, 2014 to December 31, 2024.

---

## 2. Executive Result

**Under the pre-specified confirmatory inference protocol, no statistically significant evidence of predictive alpha was detected for any of the four primary hypotheses ($H_1–H_4$) or their Composite Baseline.**

- **Confirmatory Family Discoveries**: **0 confirmed discoveries** after Family-Wise Error Rate (Bonferroni) and False Discovery Rate (Benjamini-Hochberg) adjustments ($p_{\text{adjusted}} > 0.90$).
- **Combined Baseline Strategy ($15\text{ bps}$ Base-Case Friction)**: Annualized Compound Return (CAGR) of **$-27.45\%$**, Sharpe ratio of **$-3.472$**, Maximum Drawdown of **$-97.06\%$**, and cumulative execution friction of **$\$9.90\text{M}$** on $\$10.0\text{M}$ initial capital.
- **Economic Viability**: High portfolio turnover ($\approx 205\times$ annually) combined with neutral underlying signal predictive power generated **$26.06\%$ in annual transaction cost drag**, rendering the strategy economically unviable across all market regimes.
- **Machine Learning Benchmark**: OLS, Ridge, Lasso, and XGBoost models failed to demonstrate statistically significant predictive outperformance over the baseline ($\Delta\text{IC} \le +0.0117$, Newey-West $\text{HAC }p \ge 0.7540$).

---

## 3. Why the Result Matters

1. **Combating Publication and Snooping Bias**: The academic literature is heavily biased toward reporting positive anomalies ($t > 2.0$), frequently derived from broad CRSP universes without point-in-time constituent handling, multiple-testing controls, or realistic execution friction.
2. **Microstructure Efficiency of Mega-Caps**: Large-cap equities are among the most liquid, scrutinized, and efficiently priced financial instruments globally. This study demonstrates that linear and simple non-linear price/volume anomalies fail to yield exploitable out-of-sample edge in mega-caps over the 2014–2024 decade.
3. **The Primacy of Transaction Costs**: High-turnover daily rebalanced portfolios face severe performance destruction from execution frictions. Even if gross alpha is near zero, transaction costs compound into total capital destruction.

---

## 4. Methodology & Pipeline Architecture

```
                          ┌──────────────────────────┐
                          │   Historical Data & QC   │
                          │ (Point-in-Time Universe) │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │   Feature Engineering    │
                          │ (No Look-Ahead Guarantee)│
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │ Core Alpha Signals H1-H4 │
                          │  (Frozen SHA-256 Hash)   │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │   Statistical Testing    │
                          │(HAC, Bootstrap, Multiple)│
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │  Portfolio Construction  │
                          │ (Quadratic Projection)   │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │   Event-Driven Engine    │
                          │(Single-Pass Cost Model)  │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │ Walk-Forward Validation  │
                          │(7 Windows + 2024 Holdout)│
                          └──────────────────────────┘
```

---

## 5. Point-in-Time Universe & Data Provenance

- **Universe Definition**: 100 historical constituents of the S&P 100 index as of January 1, 2014 (`data/universe/sp100_20140101.csv`).
- **Forensic Identity Audit**:
  - **92 Continuous Identities**: Unbroken trading histories from 2014-01-02 to 2024-12-31.
  - **1 Terminated Identity**: Time Warner Inc. (`TWX`), actively traded until acquisition on 2018-06-14.
  - **7 Unavailable Legacy Series**: `APC`, `DOW`, `EMC`, `FOXA`, `MON`, `RTN`, `WAG` (excluded without synthetic fabrication or successor backfilling).
- **Calendar Alignment**: Authoritative NYSE exchange calendar comprising **2,768 trading sessions** (0 weekend/holiday bars).
- **Split & Dividend Accounting**: Volume is adjusted strictly by discrete stock splits; cash dividends adjust prices but preserve raw trade volume.

---

## 6. Four Confirmatory Alpha Hypotheses

All hypotheses are pre-specified in the confirmatory hypothesis registry (SHA-256 hash `3caabc3550691880f4d66cc24a7a4ad14b76f623c169fd065cd2470fb3025840`):

| ID | Anomaly Name | Signal Mathematical Formulation | Horizon ($h$) | Target Column | Expected Sign |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **$H_1$** | **Price Momentum** | $\text{MOM}_{i,t} = \frac{P_{i, t-21} - P_{i, t-252}}{P_{i, t-252}}$ | $20\text{d}$ | `fwd_ret_20d` | Positive ($\text{IC} > 0$) |
| **$H_2$** | **Mean Reversion** | $-Z_{i,t} = -\frac{P_{i,t} - \text{SMA}_{20}(P_i)_t}{\sigma_{20}(P_i)_t}$ | $5\text{d}$ | `fwd_ret_5d` | Positive ($\text{IC} > 0$) |
| **$H_3$** | **Low Volatility** | $-\sigma_{i,t} = -\sqrt{252 \times \text{Var}_{60}(r_i)}$ | $20\text{d}$ | `fwd_ret_20d` | Positive ($\text{IC} > 0$) |
| **$H_4$** | **Abnormal Volume** | $\text{sign}(r_{i,t}) \times \log\left(\frac{V_{i,t}}{\text{SMA}_{20}(V_i)_t}\right)$ | $5\text{d}$ | `fwd_ret_5d` | Positive ($\text{IC} > 0$) |

---

## 7. Statistical Testing Framework

1. **Daily Cross-Sectional Information Coefficient (IC)**:
   $$\text{IC}_t = \rho_s(\text{Signal}_{\cdot, t}, \text{ForwardReturn}_{\cdot, t \to t+h})$$
2. **Primary Inference — Newey-West HAC Test**:
   Estimates standard errors using Bartlett kernel truncation lag $L = \max(1, h)$ to account for overlapping return autocorrelation.
3. **Non-Parametric Validation**:
   - **Stationary Block Bootstrap**: 1,000 resamples (Politis & Romano, 1994) with random geometric block lengths ($\bar{B}=10$).
   - **Within-Date Permutation Test**: 1,000 cross-sectional ticker shuffles per date ($p = \frac{1 + \text{count}}{B + 1}$).
4. **Multiple-Testing Controls**:
   - **Bonferroni FWER Control**: $\alpha_{\text{adj}} = 0.05 / 4 = 0.0125$.
   - **Benjamini-Hochberg FDR Control**: False discovery rate capped at $q = 0.05$.

### Confirmatory Evaluation Results:

| Hypothesis | Horizon | Mean Rank IC | IC Std | ICIR | HAC $t$-stat | HAC $p$-value | Bootstrap 95% CI | Permutation $p$ | Bonferroni $p$ | BH-FDR $p$ | Discovery? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H_1$: Momentum** | $20\text{d}$ | $+0.0021$ | $0.2618$ | $+0.008$ | $+0.112$ | **$0.9107$** | $[-0.0350, +0.0384]$ | $0.3516$ | $1.0000$ | $0.9357$ | **NO** |
| **$H_2$: Mean Reversion** | $5\text{d}$ | $+0.0005$ | $0.2034$ | $+0.003$ | $+0.081$ | **$0.9357$** | $[-0.0122, +0.0134]$ | $0.7852$ | $1.0000$ | $0.9357$ | **NO** |
| **$H_3$: Low Volatility** | $20\text{d}$ | $-0.0027$ | $0.2721$ | $-0.010$ | $-0.144$ | **$0.8856$** | $[-0.0396, +0.0340]$ | $0.1698$ | $1.0000$ | $0.9357$ | **NO** |
| **$H_4$: Abnormal Volume** | $5\text{d}$ | $-0.0026$ | $0.1343$ | $-0.019$ | $-1.012$ | **$0.3114$** | $[-0.0076, +0.0024]$ | $0.2098$ | $1.0000$ | $0.9357$ | **NO** |
| **Combined Baseline** | $20\text{d}$ | $-0.0005$ | $0.2224$ | $-0.002$ | $-0.034$ | **$0.9728$** | $[-0.0298, +0.0269]$ | $0.7982$ | — | — | **NO** |

---

## 8. Portfolio Construction & Hard Risk Constraints

Target weights $w_{i,t}$ are computed via cross-sectional signal ranking and projected onto a convex constraint polytope:
$$\min_w \frac{1}{2} \|w - w_{\text{raw}}\|_2^2 \quad \text{s.t.} \quad |w_i| \le 0.10, \quad \sum_i |w_i| \le 1.00, \quad \left|\sum_i w_i\right| \le 0.20$$
- **Target Gross Exposure**: Strictly $\le 1.0000$ at all rebalances (**Zero Leverage Guarantee**).
- **Maximum Realized Mark-to-Market Gross**: $1.1754$ (driven by intra-day asset price drift during market drawdowns).

---

## 9. Event-Driven Backtesting & Cost Sensitivity

Execution economics utilize a single-pass basis-point specification absorbing commission, spread, and market impact slippage:

| Cost Regime | Commission | Spread | Slippage | Total One-Way Friction | CAGR | Sharpe | Max DD | Total Cost Paid |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Cost (`zero`)** | $0.0\text{ bps}$ | $0.0\text{ bps}$ | $0.0\text{ bps}$ | **$0.0\text{ bps}$** | $-1.40\%$ | $-0.110$ | $-39.02\%$ | $\$0$ |
| **Low-Cost (`low`)** | $2.0\text{ bps}$ | $3.0\text{ bps}$ | $2.0\text{ bps}$ | **$7.0\text{ bps}$** | $-14.54\%$ | $-1.684$ | $-82.35\%$ | $\$7,936,759$ |
| **Medium / Base (`medium`)** | $5.0\text{ bps}$ | $5.0\text{ bps}$ | $5.0\text{ bps}$ | **$15.0\text{ bps}$** | **$-27.45\%$** | **$-3.472$** | **$-97.06\%$** | **$\$9,903,113$** |
| **High-Cost (`high`)** | $10.0\text{ bps}$ | $10.0\text{ bps}$ | $10.0\text{ bps}$ | **$30.0\text{ bps}$** | $-46.66\%$ | $-6.753$ | $-99.90\%$ | $\$10,388,886$ |
| **Very-High-Cost (`very_high`)** | $15.0\text{ bps}$ | $15.0\text{ bps}$ | $20.0\text{ bps}$ | **$50.0\text{ bps}$** | $-60.69\%$ | $-9.751$ | $-100.00\%$ | $\$10,400,683$ |

---

## 10. Walk-Forward Expanding Windows & 2024 Final Holdout

- **Development OOS (Windows 1–6, 2018–2023)**: Concatenated OOS CAGR = **$-31.58\%$**, Sharpe = **$-3.403$**, MaxDD = **$-89.71\%$**.
- **Untouched Final Holdout (Window 7, 2024)**: Evaluated strictly once without parameter tuning:
  - **OOS Period**: 2024-01-01 to 2024-12-31 (252 trading sessions)
  - **OOS CAGR**: **$-22.83\%$** (Sharpe: **$-4.696$**, MaxDD: **$-23.73\%$**)
  - **Probabilistic Sharpe Ratio (PSR)**: **$0.0\%$**

---

## 11. Machine Learning Benchmark

Four machine learning architectures were trained on expanding historical windows to forecast `fwd_ret_20d` and evaluated against the Combined Baseline ($N=1,760$ OOS sessions):

| Model | OOS Mean Rank IC | IC Std | ICIR | HAC $p$-value | Baseline Mean IC | $\Delta\text{IC}$ ($\text{ML} - \text{Base}$) | $\Delta\text{IC}$ HAC $p$ | Significant Outperformance? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OLS Regression** | $+0.0108$ | $0.2606$ | $+0.042$ | $0.6177$ | $-0.0009$ | $+0.0117$ | $0.7540$ | **NO** |
| **Ridge Regression** | $+0.0108$ | $0.2606$ | $+0.042$ | $0.6177$ | $-0.0009$ | $+0.0117$ | $0.7540$ | **NO** |
| **Lasso Regression** | $+0.0105$ | $0.2626$ | $+0.040$ | $0.6303$ | $-0.0009$ | $+0.0115$ | $0.7620$ | **NO** |
| **XGBoost Regressor** | $+0.0066$ | $0.1926$ | $+0.034$ | $0.6451$ | $-0.0009$ | $+0.0075$ | $0.7981$ | **NO** |

---

## 12. Robustness Analysis Suite

- **Pre- vs. Post-2020 Stability**: Pre-2020 CAGR = $-23.81\%$ (Sharpe $-4.249$); Post-2020 CAGR = $-31.55\%$ (Sharpe $-3.208$).
- **Asset Exclusion Jackknife (10% Drop, 5 Iterations)**: Sharpe range $[-3.560, -3.316]$ confirms universe-wide consistency.
- **Extreme-Day Trimming**: Excluding 5 best and 5 worst PnL days yields CAGR = $-27.15\%$ (Sharpe $-3.802$).
- **Stationary Block Bootstrap CI**: 95% Sharpe CI $[-4.204, -2.729]$; 95% CAGR CI $[-31.53\%, -23.13\%]$.
- **Market-Regime Slicing**: Strategy returns remained negative across Low/Normal/High Volatility and Bull/Bear Trend environments.

---

## 13. Key Findings

1. **Zero Confirmatory Discoveries**: None of the 4 canonical anomalies achieved statistical significance under Newey-West HAC inference.
2. **Turnover Destroys Capital**: With annualized turnover exceeding $200\times$, a realistic $15\text{ bps}$ friction generates a $26\%$ annual drag.
3. **ML Fails to Add Exploitable Edge**: Linear and non-linear models did not produce statistically significant outperformance over the simple linear baseline.

---

## 14. Research Contributions

1. **Frozen Pre-Specified Alpha Hypotheses**: Mitigates $p$-hacking and researcher degrees of freedom via cryptographic SHA-256 registry hashing.
2. **Survivorship-Controlled Point-in-Time Universe**: Audits 100 historical constituents, correctly retaining terminated entities (`TWX`) without synthetic backfilling.
3. **Exchange-Aware Calendar**: Strict alignment to 2,768 NYSE trading sessions, preventing non-trading day distortion.
4. **Recycled-Ticker Protection**: Prevents look-ahead and survivorship leaks from corporate reorganizations (`BK`, `FB`, `UTX`).
5. **HAC Inference for Overlapping Horizons**: Implements Newey-West variance estimation with Bartlett kernel lag $L = \max(1, h)$.
6. **Multiple-Testing Control**: Rigorous Bonferroni FWER and Benjamini-Hochberg FDR adjustments applied to primary confirmatory tests.
7. **Event-Driven Execution with Single-Pass Costs**: Realistic order fill modeling incorporating commissions, bid-ask spreads, and slippage.
8. **Expanding Walk-Forward Validation & Untouched Holdout**: 7-window protocol ensuring temporal out-of-sample discipline.
9. **ML Benchmarking vs. Fixed Baseline**: Formal hypothesis testing on the paired daily IC difference series $\Delta\text{IC}_t$.
10. **Reproducible Checkpointed Artifacts**: Full persistence of cryptographically hashed JSON results checkpoints.

---

## 15. Resume / Portfolio Summary

> **Quantitative Alpha Research & Event-Driven Backtesting Framework (S&P 100, 2014–2024)**  
> Built an institutional-grade, point-in-time quantitative alpha research and event-driven backtesting engine for the historical S&P 100 universe ($N=93$ securities, 2,768 trading days). Engineered causal feature pipelines, implemented Newey-West HAC statistical inference for overlapping forecast horizons, and enforced Bonferroni FWER and Benjamini-Hochberg FDR multiple-testing controls. Designed a convex quadratic portfolio optimizer with hard gross/net leverage bounds and a single-pass transaction cost model (commissions, spreads, slippage). Evaluated four canonical anomalies ($H_1–H_4$), a multi-factor baseline, and machine learning models (OLS, Ridge, Lasso, XGBoost) across 7 expanding walk-forward windows and an isolated 2024 holdout. The empirical study found no statistically significant out-of-sample evidence of predictive alpha, with negative strategy economics robustly verified across transaction-cost regimes, temporal splits, asset jackknife resampling, and market volatility regimes.

---

## 16. Reproducibility & Cryptographic Hashes

All empirical results are 100% reproducible from repository code and verified checkpoints:

| Artifact Description | File Path | SHA-256 Hash Representation |
| :--- | :--- | :--- |
| **Cleaned Market Data Panel** | `data/processed/cleaned_ohlcv.parquet` | `c3d67525d09fc052` (16-char prefix) |
| **Engineered Feature Cache** | `data/cache/features_c3d67525d09fc052.parquet` | `18b4358995ae9881` (16-char prefix) |
| **Alpha Signal Cache** | `data/cache/signals_c3d67525d09fc052.parquet` | `a8e62995b6a4950f` (16-char prefix) |
| **Production Experiment Config** | `configs/default.yaml` | `3e3b02ce22d1afd3af30aa60520eac232c8d2a765f66d170aed514e0406e638c` (full 64-char hash) |
| **Confirmatory Hypothesis Registry** | `src/statistics/multiple_testing.py` | `3caabc3550691880f4d66cc24a7a4ad14b76f623c169fd065cd2470fb3025840` (full 64-char hash) |

---

## 17. Repository Structure

```
quant-alpha-research-p1/
├── configs/                  # Immutable experiment and system YAML configs
├── data/
│   ├── universe/             # 2014 S&P 100 point-in-time constituent CSV & metadata
│   ├── raw/                  # Raw cached price/volume/split parquet (.gitignore)
│   ├── cache/                # Engineered feature and signal parquet caches (.gitignore)
│   └── processed/            # Validated cleaned OHLCV parquet panel (.gitignore)
├── reports/
│   ├── final_report.md       # Comprehensive formal academic research report
│   └── figures/              # Publication-quality research visualizations
├── results/
│   └── checkpoints/          # Verified cryptographic JSON results checkpoints
├── scripts/
│   ├── run_pipeline.py       # End-to-end research pipeline runner
│   ├── run_backtest.py       # Isolated backtest simulation CLI
│   ├── run_phase_2d2.py      # Portfolio & walk-forward evaluation runner
│   ├── run_phase_2e.py       # ML benchmark and robustness suite runner
│   ├── generate_figures.py   # Publication figures generator
│   └── exploratory_horizon_analysis.py  # Isolated signal half-life profile runner
├── src/
│   ├── backtest/             # Event-driven engine, broker, portfolio tracker
│   ├── config/               # Pydantic schema and YAML loaders
│   ├── data/                 # Ingestion, cleaning, calendar, universe
│   ├── evaluation/           # 14 performance/risk metrics & tearsheet
│   ├── execution/            # Single-pass cost model (commissions, spreads, slippage)
│   ├── features/             # Causal feature engineering (returns, indicators, vol)
│   ├── ml/                   # OLS, Ridge, Lasso, XGBoost & HAC comparison
│   ├── portfolio/            # Equal-weight, signal-weighted, inv-vol, projection
│   ├── robustness/           # Subperiods, jackknife, trimming, bootstrap
│   ├── signals/              # H1-H4 confirmatory signals & composite signal
│   ├── statistics/           # IC, Newey-West HAC, bootstrap, permutation, multiple testing
│   └── validation/           # 7 walk-forward windows, PSR, DSR diagnostics
└── tests/                    # 109 passing unit, property, and regression tests
```

---

## 18. How to Run

```bash
# 1. Clone repository
git clone https://github.com/shouryasarkar713/quant-alpha-research-p1.git
cd quant-alpha-research-p1

# 2. Set up virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies in editable mode
pip install -e .

# 4. Run test suite (109 tests)
pytest tests/ -v

# 5. Generate publication figures
python scripts/generate_figures.py

# 6. Run exploratory horizon decay analysis
python scripts/exploratory_horizon_analysis.py
```

---

## 19. Limitations & Future Research

1. **Constituent Breadth**: Restricted to 93 usable historical constituents of the S&P 100. Findings should not be extrapolated to small-cap or international equities.
2. **Alternative Data Sources**: Analysis is restricted to daily OHLCV price/volume signals; alternative datasets (order flow, earnings transcripts, sentiment) were not evaluated.
3. **Execution Modeling**: Assumes next-day market-on-close fills with fixed basis-point friction without modeling intra-day order routing or borrow rebate fees.
