# Statistical Alpha Research & Event-Driven Backtesting Framework

An institutional-grade, fully reproducible quantitative equity alpha research framework implementing the authoritative **Specification v1.2** for the historical S&P 100 universe (2014-01-02 to 2024-12-31).

---

## 🏛️ Architecture Overview

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

## 🚀 Key Framework Features

1. **Point-in-Time Historical S&P 100 Universe**:
   - S&P 100 constituent panel as of 2014-01-01 (100 securities).
   - Full survivorship bias control: historical constituents are tracked through merger/delisting termination dates without retroactive backfilling.
   - Discrete split adjustment: historical volume adjusted purely by stock splits, preventing dividend distribution distortion.

2. **Core Confirmatory Hypotheses ($m=4$)**:
   - **H1: Momentum** (Jegadeesh-Titman 12–1 skip return, $h=20\text{d}$)
   - **H2: Mean Reversion** (Inverted 20d price $z$-score, $h=5\text{d}$)
   - **H3: Low Volatility** (Inverted 60d realized return volatility, $h=20\text{d}$)
   - **H4: Abnormal Volume** ($\text{sign}(r_{1\text{d}}) \cdot \log(\text{vol} / \text{sma}_{20}), h=5\text{d}$)
   - **Combined Alpha**: Multi-factor equal-weighted benchmark with strict NaN propagation.

3. **Rigorous Statistical Testing Subsystem**:
   - Daily cross-sectional Spearman IC series: $\text{IC}_t = \text{Corr}(\text{signal}_{i,t}, \text{return}_{i,t+1 \to t+h})$.
   - Newey-West HAC standard errors with Bartlett kernel truncation lag $L = \max(1, h)$.
   - Stationary Block Bootstrap (Politis & Romano, 1994) with random geometric block lengths.
   - Finite-sample-safe within-date cross-sectional permutation test ($p = \frac{1 + \text{count}}{B + 1}$).
   - Family-Wise Error Rate (Bonferroni) and False Discovery Rate (Benjamini-Hochberg) applied strictly to the 4 primary HAC IC $p$-values.

4. **Portfolio Construction & Hard Risk Constraints**:
   - Methodologies: `EqualWeightLongShort`, `SignalWeightedLongShort`, `InverseVolatilitySignalWeighted`.
   - Hard Risk Constraints: Single position $|w_i| \le 0.10$, Net dollar exposure $|\sum w_i| \le 0.20$, Gross exposure $\sum |w_i| \le 1.00$ (**Zero Leverage Guarantee**).
   - Solved via exact quadratic Euclidean projection onto the convex polytope.

5. **Event-Driven Backtest & Single-Pass Cost Model**:
   - Close $t$ signal generation $\to$ Rebalance order emission $\to$ Synthetic close $t+1$ execution.
   - Absorbs bid-ask spread and market impact slippage into fill prices in a single pass.
   - Cash and share holdings tracked in strict mark-to-market accounting ledger.

6. **Expanding-Window Walk-Forward Validation & Holdout Audit**:
   - 7 expanding windows: Windows 1–6 (2018–2023) Development OOS; Window 7 (2024) untouched final holdout.
   - Evaluated using Probabilistic Sharpe Ratio (PSR) against $SR^* = 0$.

7. **Machine Learning Comparative Benchmarking**:
   - OLS, Ridge, Lasso, and XGBoost forecasting common 20-day forward return target.
   - Statistical comparison against baseline via Newey-West HAC inference on the daily IC difference series $\Delta \text{IC}_t$.

---

## 🛠️ Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/shouryasarkar713/quant-alpha-research-p1.git
cd quant-alpha-research-p1

# 2. Set up virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -e .
```

---

## 🧪 Running Unit & Property Tests

The repository contains 109 comprehensive unit, property, statistical, and integration tests:

```bash
pytest tests/ -v
```

---

## 💻 CLI Usage

### Run End-to-End Research Pipeline
```bash
python scripts/run_pipeline.py --data-path data/processed/cleaned_ohlcv.parquet --output-dir reports
```

### Run Custom Strategy Backtest
```bash
python scripts/run_backtest.py --config configs/default.yaml
```

---

## 📁 Repository Structure

```
quant-alpha-research-p1/
├── configs/                  # Immutable experiment and system YAML configs
├── data/
│   ├── universe/             # Documented 2014 S&P 100 point-in-time universe
│   ├── raw/                  # Raw cached price/volume/split data (.gitignore)
│   ├── cache/                # Engineered feature and signal caches (.gitignore)
│   └── processed/            # Validated cleaned OHLCV parquet data (.gitignore)
├── reports/
│   └── final_report.md       # Comprehensive formal quantitative research paper
├── results/
│   └── checkpoints/          # Verified cryptographic JSON results checkpoints
├── scripts/
│   ├── run_pipeline.py       # End-to-end research pipeline CLI runner
│   ├── run_backtest.py       # Isolated backtest simulation CLI runner
│   ├── run_phase_2d2.py      # Portfolio & walk-forward evaluation runner
│   └── run_phase_2e.py       # ML benchmark and robustness suite runner
├── src/
│   ├── backtest/             # Event-driven engine, broker, portfolio tracker, events
│   ├── config/               # Pydantic/dataclass config schema and YAML loaders
│   ├── data/                 # Ingestion, cleaning, validation, anomalies, splits
│   ├── evaluation/           # 14 performance/risk metrics, drawdowns, tearsheet
│   ├── execution/            # Single-pass cost model, spread, slippage, regimes
│   ├── features/             # Returns, technical indicators, volume, regimes
│   ├── ml/                   # OLS, Ridge, Lasso, XGBoost & HAC comparison
│   ├── portfolio/            # Equal-weight, signal-weighted, inverse-vol, projection
│   ├── signals/              # H1-H4 confirmatory signals & composite signal
│   ├── statistics/           # IC, Newey-West HAC, bootstrap, permutation, multiple testing
│   └── validation/           # 7 walk-forward windows, PSR, DSR diagnostics
└── tests/                    # 109 passing unit, integration, and regression tests
```

---

## 📜 Key Empirical Findings

- **Confirmatory Family Discoveries ($m=4$)**:
  - H1 Momentum: Mean Rank IC = $+0.0021$, HAC $p = 0.9107$ (**Fail to reject**)
  - H2 Mean Reversion: Mean Rank IC = $+0.0005$, HAC $p = 0.9357$ (**Fail to reject**)
  - H3 Low Volatility: Mean Rank IC = $-0.0027$, HAC $p = 0.8856$ (**Fail to reject**)
  - H4 Abnormal Volume: Mean Rank IC = $-0.0026$, HAC $p = 0.3114$ (**Fail to reject**)
  - **Confirmed Discoveries after Bonferroni / BH-FDR**: **0**
- **Combined Baseline Strategy ($15\text{ bps}$ base-case cost)**: CAGR = $-27.45\%$, Sharpe = $-3.472$, MaxDD = $-97.06\%$, Annual Turnover = $204.7\times$, Annual Cost Drag = $+26.06\%$.
- **Machine Learning Benchmark**: OLS, Ridge, Lasso, and XGBoost yielded $\Delta\text{IC} \le +0.0117$ vs. baseline ($\text{HAC }p \ge 0.7540$), failing to demonstrate statistically significant predictive outperformance.
- **Detailed Research Paper**: See [`reports/final_report.md`](reports/final_report.md) for full methodology, proofs, and audit tables.
