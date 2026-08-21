# STATISTICAL ALPHA RESEARCH & EVENT-DRIVEN BACKTESTING FRAMEWORK
## Complete Project Specification v1.2

> **Revision note**: This is v1.2, revised from v1.1 following a second statistical/backtesting audit. The changes are targeted corrections to point-in-time universe eligibility, portfolio constraint enforcement, removal of contradictory portfolio-level volatility targeting, final-holdout reporting hierarchy, split-aware volume handling, and risk-metric definitions. The project scope and research architecture are intentionally unchanged.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Goals](#2-project-goals)
3. [Research Questions](#3-research-questions)
4. [Research Philosophy](#4-research-philosophy)
5. [Scope and Believability Constraints](#5-scope-and-believability-constraints)
6. [MVP / Strong / Research-Grade Versions](#6-mvp--strong--research-grade-versions)
7. [System Architecture](#7-system-architecture)
8. [Repository Structure](#8-repository-structure)
9. [Data Strategy](#9-data-strategy)
10. [Data Schema](#10-data-schema)
11. [Data Pipeline](#11-data-pipeline)
12. [Feature Engineering](#12-feature-engineering)
13. [Signal Specifications](#13-signal-specifications)
14. [Statistical Testing](#14-statistical-testing)
15. [Multiple-Hypothesis Testing](#15-multiple-hypothesis-testing)
16. [Portfolio Construction](#16-portfolio-construction)
17. [Event-Driven Backtester](#17-event-driven-backtester)
18. [Execution and Transaction Costs](#18-execution-and-transaction-costs)
19. [Risk Management](#19-risk-management)
20. [Walk-Forward Validation](#20-walk-forward-validation)
21. [Robustness Testing](#21-robustness-testing)
22. [Regime Analysis](#22-regime-analysis)
23. [ML Extension](#23-ml-extension)
24. [Experiment Matrix](#24-experiment-matrix)
25. [Notebook Specifications](#25-notebook-specifications)
26. [Python Module Specifications](#26-python-module-specifications)
27. [Configuration System](#27-configuration-system)
28. [Testing Strategy](#28-testing-strategy)
29. [Reproducibility](#29-reproducibility)
30. [Results Storage](#30-results-storage)
31. [Visualization Plan](#31-visualization-plan)
32. [Research Report](#32-research-report)
33. [GitHub README](#33-github-readme)
34. [Development Roadmap](#34-development-roadmap)
35. [Priority Classification](#35-priority-classification)
36. [Technical Tradeoffs](#36-technical-tradeoffs)
37. [Quant Interview Relevance](#37-quant-interview-relevance)
38. [Candidate Defensibility](#38-candidate-defensibility)
39. [Common Failure Modes](#39-common-failure-modes)
40. [Resume Bullet Templates](#40-resume-bullet-templates)
41. [Final Acceptance Criteria](#41-final-acceptance-criteria)
42. [Future Extensions](#42-future-extensions)

---

## 1. Executive Summary

This project is a self-contained quantitative research study investigating whether simple statistical trading signals — momentum, short-term mean reversion, volatility, and abnormal volume — contain robust, out-of-sample predictive association with future equity returns after realistic trading costs.

The project is **not** a production trading system. It is a **research framework**: a disciplined pipeline that moves from hypothesis → data → signal construction → statistical testing → event-driven backtesting → walk-forward validation → robustness analysis → conclusion, with an optional machine-learning comparison layer.

The core deliverable is a **research conclusion**, not a profitable trading strategy. A negative result — signals that fail to survive transaction costs or out-of-sample testing — is a fully valid outcome and is treated as such throughout the design.

**What makes this project distinctive:**

| Dimension | This project | Typical student project |
|---|---|---|
| Evaluation rigor | Walk-forward, robustness, multiple-testing correction | Single train/test split |
| Transaction costs | Configurable, sensitivity-analyzed, single-pass accounting | Ignored or fixed at zero |
| Backtester | Event-driven with order/fill lifecycle | Vectorized signal × return |
| Statistical discipline | IC with HAC inference, permutation tests, FDR correction | p-values without context |
| Research honesty | Negative results acceptable; confirmatory vs. exploratory distinguished | Cherry-picked winning strategy |
| Universe integrity | Start-of-sample membership; no future look-ahead in index composition | Current index members backtested historically |
| Missing data | Missing bars preserved as NaN; no fabricated prices | Forward-filled or ignored |

**Scope:** Designed for a strong undergraduate to complete in 6–8 weeks, working part-time on implementation after receiving this specification.

---

## 2. Project Goals

### Primary Goals

| ID | Goal | Measurable criterion |
|----|------|---------------------|
| G1 | Determine whether simple statistical signals show predictive association with future returns out-of-sample | Walk-forward IC analysis with HAC-adjusted statistical significance tests |
| G2 | Quantify the impact of transaction costs on signal-based strategies | Performance degradation curves across cost regimes; economic break-even cost |
| G3 | Build an event-driven backtester that correctly models order lifecycle | Unit tests proving no look-ahead bias, no cost double-counting, correct timing |
| G4 | Apply rigorous multiple-testing correction to avoid false discoveries | Bonferroni and BH-FDR applied; confirmatory vs. exploratory hypotheses distinguished |
| G5 | Demonstrate robustness or fragility of results across perturbations | Parameter sensitivity, time-period stability, regime analysis |

### Secondary Goals

| ID | Goal |
|----|------|
| G6 | Compare statistical baselines against a tree-based ML model |
| G7 | Produce a reproducible research report with honest conclusions |
| G8 | Demonstrate quantitative research methodology suitable for quant interviews |

### Non-Goals

- Building a live-trading system
- Maximizing backtest Sharpe ratio
- Using deep learning, reinforcement learning, or transformers
- Processing tick-level or intraday data
- Handling alternative data (satellite, NLP, etc.)
- Implementing portfolio optimization (Markowitz, Black-Litterman, etc.)

---

## 3. Research Questions

### Central Research Question

> **Do simple statistical trading signals show robust out-of-sample predictive association with future equity returns, and does that association remain economically meaningful after transaction costs, walk-forward validation, and robustness testing?**

### Sub-Questions

| ID | Sub-question | Method |
|----|-------------|--------|
| RQ1 | Does cross-sectional momentum (past 12-month return, skipping the most recent month) predict next-month returns? | IC analysis with HAC inference, walk-forward |
| RQ2 | Does short-term mean reversion (price z-score relative to 20-day moving average) predict next-week returns? | IC analysis with HAC inference, walk-forward |
| RQ3 | Does recent realized volatility contain cross-sectional information about future returns? | Quintile spread analysis, IC |
| RQ4 | Does abnormal trading volume predict short-term future returns? | IC analysis, event-study-style analysis |
| RQ5 | Do any signals that show statistically significant in-sample predictive association survive realistic transaction costs? | Cost-sensitivity analysis |
| RQ6 | Does XGBoost improve out-of-sample prediction relative to the pre-specified combined statistical baseline? | Walk-forward comparison, HAC test on IC difference |
| RQ7 | Are results robust to parameter perturbation, time-period changes, and regime conditioning? | Robustness suite |

### Research Stance

The project **does not assume** the answer to any sub-question is "yes." The architecture is designed so that a "no" answer is as informative as a "yes" answer. The research conclusion should state what was found, not what the researcher hoped to find.

---

## 4. Research Philosophy

### Workflow

```
Hypothesis formulation (confirmatory hypotheses pre-specified)
       ↓
Data acquisition & audit
       ↓
Feature construction (with lag discipline)
       ↓
Signal construction
       ↓
Statistical testing (IC with HAC inference, permutation, bootstrap)
       ↓
Multiple-testing correction (confirmatory family defined)
       ↓
Portfolio construction
       ↓
Event-driven backtest (with single-pass transaction costs)
       ↓
Walk-forward out-of-sample evaluation (development phase)
       ↓
Final holdout evaluation (methodology frozen)
       ↓
Robustness analysis
       ↓
ML comparison (late-stage)
       ↓
Research conclusion
```

### Anti-Patterns Explicitly Prohibited

| Anti-pattern | Why it's harmful | How this project prevents it |
|-------------|-----------------|------------------------------|
| Cherry-picking | Selectively showing winning results | All experiments logged; multiple-testing correction required; confirmatory vs. exploratory distinguished |
| Sharpe hacking | Optimizing parameters to maximize Sharpe | Walk-forward; parameters chosen on training set only; primary configs pre-specified |
| Test-set optimization | Iterating on the test set | Strict train/validate/test split; final holdout used exactly once |
| Look-ahead bias | Using future data in current decisions | Lag discipline in features; event-driven backtest; unit tests; start-of-sample universe |
| Survivorship bias | Using only stocks that survived | Start-of-sample universe; documented limitation; no forward-fill of missing bars |
| Unrealistic costs | Ignoring or underestimating trading frictions | Configurable cost model; sensitivity analysis mandatory; single-pass accounting |
| p-hacking | Running many tests and reporting only significant ones | FDR correction; all tests reported; exploratory results labeled as such |
| Missing-data fabrication | Forward-filling OHLC bars to create artificial observations | Missing bars preserved as NaN; features propagate NaN; signals skip missing data |

### Information Timing Rule

> **A signal computed at market close on day $t$ may only influence orders placed after the close of day $t$. Those orders are assumed to execute at the close of day $t+1$.** This is a simplified daily-bar execution convention chosen for reproducibility, consistency, and prevention of same-bar look-ahead. It is NOT a claim that the trader can literally obtain the official closing price in a real market. The project does not model intraday execution; the execution price is synthetic.

This rule must be enforced in the data pipeline, signal construction, and backtester. Unit tests must verify it.

---

## 5. Scope and Believability Constraints

### Believability Criteria

The finished project should be credible as the work of a strong undergraduate who:

- Took courses in probability, statistics, linear algebra, and ML
- Has Python experience with NumPy/Pandas/scikit-learn
- Spent 6–8 weeks on implementation
- Used coding assistants (GitHub Copilot, ChatGPT, etc.) for scaffolding
- Actually understands the math, statistics, and design decisions
- Did not merely copy an institutional trading platform

### Red Flags to Avoid

| Red flag | Why it's suspicious | Mitigation |
|----------|-------------------|------------|
| > 8 signal families | Unrealistic scope | Limit to 4 core + 1 optional |
| HMM, GARCH, stochastic vol as core | Overly sophisticated for scope | List as future extensions only |
| Deep learning for alpha | Overkill for daily data with < 50 features | XGBoost/LightGBM is the ML ceiling |
| Custom portfolio optimization (Markowitz, risk parity) | Graduate-level quant finance | Use simple weighting schemes |
| Tick-level market microstructure | Institutional HFT territory | Use daily OHLCV only |
| Real-time execution engine | Production system, not research | Event-driven backtest only |
| > 30 performance metrics | Metric bloat | ~14 well-understood metrics |
| "State-of-the-art" / "production-grade" language | AI-generated marketing | Use research-oriented language |

### Scope Anchoring

- **Universe**: 50–100 liquid US equities (start-of-sample S&P 100 constituents, defined as of 2014-01-01)
- **Frequency**: Daily bars (OHLCV)
- **History**: ~10 years (2014–2024)
- **Signals**: 4 core families
- **ML models**: Linear + 1 tree-based (XGBoost or LightGBM, not both)
- **Notebooks**: 10
- **Python modules**: ~15–20 files across ~10 packages
- **Total Python LOC (excluding tests)**: ~2,000–3,500
- **Total test LOC**: ~800–1,500

---

## 6. MVP / Strong / Research-Grade Versions

### MVP (Minimum Viable Research System)

> A working pipeline from data → features → one signal → vectorized backtest → basic metrics.

| Component | MVP requirement |
|-----------|----------------|
| Data | Single data source loaded and cleaned (no forward-fill of OHLC) |
| Features | Returns, rolling mean, rolling std (of prices and returns, distinguished) |
| Signals | Momentum only |
| Statistics | IC, naive t-test (diagnostic only) |
| Backtest | Vectorized signal × next-period return |
| Costs | Fixed proportional cost |
| Validation | Single train/test split |
| Metrics | Sharpe, max drawdown, total return |
| Tests | Return calculation, IC calculation |
| Notebooks | 3 (data, signal, backtest) |

**Estimated effort**: 2 weeks

### Strong Version (TARGET)

> Full statistical research project with event-driven backtester, walk-forward validation, multiple signals, robustness testing, and honest conclusions.

| Component | Strong requirement |
|-----------|-------------------|
| Data | Cleaned, validated, multi-asset OHLCV; start-of-sample universe; no forward-filled bars |
| Features | Full feature library (returns, momentum, mean-reversion, volatility, volume) with dimensionally correct formulas |
| Signals | 4 core signal families with pre-specified primary configurations |
| Statistics | IC with HAC inference (primary), naive t-stat (diagnostic), bootstrap CI, within-date permutation test |
| Multiple testing | Bonferroni + BH-FDR; confirmatory vs. exploratory distinguished |
| Backtest | Event-driven with order lifecycle; single-pass cost accounting |
| Costs | Configurable: commission + spread + slippage (no double-counting) |
| Portfolio | Equal-weight, signal-weight, volatility-scaled; gross exposure ≤ 1.0 |
| Validation | Walk-forward (expanding window) with final holdout |
| Robustness | Parameter perturbation, cost sensitivity, time-period splits |
| Regime | Simple volatility-based regime analysis (descriptive); regime-conditioned trading as optional new strategy |
| Metrics | Full suite (~14 metrics) with precise definitions |
| Tests | Comprehensive including look-ahead bias tests, cost accounting tests, data-quality tests |
| Notebooks | 10 |
| Report | Complete research report with cautious language |
| Config | YAML-driven experiments |

**Estimated effort**: 6–8 weeks

### Research-Grade (Optional Extensions)

| Extension | Effort | Value |
|-----------|--------|-------|
| ML comparison (XGBoost/LightGBM) | 1 week | High — demonstrates ML vs. statistical baseline |
| Cross-sectional signal (industry-neutral) | 3–4 days | Moderate |
| Pairs/cointegration for 2–3 pairs | 3–4 days | Moderate |
| Market-impact model | 2 days | Moderate |
| Monte Carlo simulation of strategy returns | 2 days | Moderate |
| Comprehensive regime analysis | 3 days | Moderate |
| Portfolio-level volatility targeting (future extension; must remain capped so gross exposure never exceeds 1.0) | 2–3 days | Moderate |

---

## 7. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     CONFIGURATION                        │
│                  (YAML experiment files)                  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                    DATA LAYER                             │
│  raw data → validation → cleaning (no OHLC forward-fill) │
│  → start-of-sample universe                              │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  FEATURE LAYER                            │
│  returns, rolling stats (price & return std separated),  │
│  z-scores, volume metrics — NaN propagation              │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   SIGNAL LAYER                            │
│  momentum, mean-reversion, volatility, volume signals    │
└─────────────────────┬───────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
┌─────────▼─────────┐  ┌─────────▼─────────────┐
│  STATISTICAL       │  │  PORTFOLIO              │
│  TESTING LAYER     │  │  CONSTRUCTION LAYER     │
│  IC (HAC primary), │  │  equal-wt, signal-wt,   │
│  permutation,      │  │  vol-scaled              │
│  bootstrap, FDR    │  │  (gross ≤ 1.0)           │
└────────────────────┘  └─────────┬───────────────┘
                                  │
                      ┌───────────▼───────────────┐
                      │  EVENT-DRIVEN BACKTESTER   │
                      │  events → strategy →       │
                      │  orders → execution →      │
                      │  fills (single-pass costs)  │
                      │  → positions → PnL          │
                      └───────────┬───────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│  WALK-FORWARD      │  │  ROBUSTNESS        │  │  EVALUATION        │
│  VALIDATION        │  │  TESTING           │  │  LAYER             │
│  expanding window  │  │  perturbation,     │  │  metrics, plots,   │
│  + final holdout   │  │  cost sensitivity  │  │  reports           │
└────────────────────┘  └────────────────────┘  └────────────────────┘
```

### Design Principles

1. **Separation of concerns**: Data, features, signals, portfolio, execution, and evaluation are independent modules with clear interfaces.
2. **No circular dependencies**: Data flows downward; no module imports from a layer above it.
3. **Configuration over code**: Experiment parameters live in YAML files, not hard-coded.
4. **Testability**: Every module has pure functions that can be unit-tested with known inputs/outputs.
5. **Notebook ≠ library**: Notebooks call library functions; they do not define reusable logic.

### Dependency Graph

```
configs (YAML)
    │
    ├── data.loader ← no internal deps
    │       │
    │       ▼
    ├── data.cleaning ← data.loader
    │       │
    │       ▼
    ├── data.universe ← data.cleaning
    │       │
    │       ▼
    ├── features.engine ← data.cleaning
    │       │
    │       ▼
    ├── signals.generator ← features.engine
    │       │
    │       ├──────────────────┐
    │       ▼                  ▼
    ├── statistics.testing     portfolio.construction
    │       │                      │
    │       │                      ▼
    │       │              backtest.engine ← execution.costs
    │       │                      │
    │       ▼                      ▼
    ├── statistics.multiple_testing  evaluation.metrics
    │                              │
    │                              ▼
    └── evaluation.reports     evaluation.visualization
```

---

## 8. Repository Structure

```
alpha-research/
│
├── README.md                        # Project overview, research question, setup
├── pyproject.toml                   # Project metadata and dependencies
├── requirements.txt                 # Pinned dependencies (generated from pyproject.toml)
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # Download / load raw OHLCV data
│   │   ├── cleaning.py             # Validate, clean, handle missing data (no OHLC forward-fill)
│   │   └── universe.py             # Define and filter tradeable universe (start-of-sample)
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── returns.py              # Return calculations (simple, log, multi-horizon)
│   │   ├── technical.py            # Rolling mean, rolling std (price & return), z-score, volatility
│   │   ├── volume.py               # Volume-based features
│   │   ├── cross_sectional.py      # Rank, percentile, cross-sectional z-score
│   │   └── engine.py               # Feature pipeline orchestrator
│   │
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── momentum.py             # Momentum signal family
│   │   ├── mean_reversion.py       # Mean-reversion signal family
│   │   ├── volatility.py           # Volatility signal family
│   │   ├── volume_signal.py        # Abnormal-volume signal family
│   │   └── base.py                 # Abstract signal interface
│   │
│   ├── statistics/
│   │   ├── __init__.py
│   │   ├── information_coefficient.py  # IC, rolling IC, IC t-test (naive + HAC)
│   │   ├── hypothesis_tests.py     # Newey-West, permutation test, bootstrap CI
│   │   ├── stationarity.py         # ADF, KPSS tests
│   │   └── multiple_testing.py     # Bonferroni, BH-FDR, hypothesis registry
│   │
│   ├── portfolio/
│   │   ├── __init__.py
│   │   └── construction.py         # Equal-weight, signal-weight, vol-scaled
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── costs.py                # Transaction cost model (single-pass accounting)
│   │   └── fill_model.py           # Fill simulation (next-day close, single-pass slippage)
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── events.py               # Event definitions (MarketEvent, SignalEvent, etc.)
│   │   ├── engine.py               # Main backtest loop
│   │   ├── broker.py               # Order management, fill processing
│   │   └── portfolio_tracker.py    # Position tracking, cash, PnL
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Performance metrics (Sharpe, drawdown, etc.)
│   │   ├── visualization.py        # Plotting functions
│   │   └── reports.py              # Report generation
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   └── walk_forward.py         # Walk-forward splitting and orchestration
│   │
│   ├── robustness/
│   │   ├── __init__.py
│   │   └── tests.py                # Parameter perturbation, stability tests
│   │
│   └── config/
│       ├── __init__.py
│       └── schema.py               # Configuration dataclasses / validation
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_feature_exploration.ipynb
│   ├── 03_signal_research.ipynb
│   ├── 04_statistical_testing.ipynb
│   ├── 05_backtesting.ipynb
│   ├── 06_walk_forward.ipynb
│   ├── 07_robustness.ipynb
│   ├── 08_regime_analysis.ipynb
│   ├── 09_ml_comparison.ipynb
│   └── 10_final_results.ipynb
│
├── configs/
│   ├── default.yaml                # Default parameters
│   ├── momentum.yaml               # Momentum signal experiment
│   ├── mean_reversion.yaml         # Mean-reversion signal experiment
│   ├── volatility_signal.yaml      # Volatility signal experiment
│   ├── volume_signal.yaml          # Volume signal experiment
│   ├── combined.yaml               # Multi-signal combination experiment
│   ├── ml_experiment.yaml          # ML comparison experiment
│   └── cost_sensitivity.yaml       # Transaction-cost sensitivity experiment
│
├── tests/
│   ├── __init__.py
│   ├── test_returns.py
│   ├── test_features.py
│   ├── test_signals.py
│   ├── test_statistics.py
│   ├── test_portfolio.py
│   ├── test_costs.py
│   ├── test_backtest.py
│   ├── test_metrics.py
│   ├── test_walk_forward.py
│   ├── test_no_lookahead.py        # Dedicated look-ahead bias tests
│   └── test_data_quality.py        # Missing-data, universe, timing integrity tests
│
├── scripts/
│   ├── run_experiment.py           # CLI entry point: python run_experiment.py --config ...
│   ├── download_data.py            # Data download script
│   └── generate_report.py          # Generate final research report
│
├── data/
│   ├── raw/                        # Raw downloaded data (gitignored)
│   ├── processed/                  # Cleaned, aligned data (gitignored)
│   └── universe/                   # Universe definition files
│
├── results/
│   ├── experiments/                # Per-experiment outputs (gitignored)
│   └── reports/                    # Generated reports
│
├── reports/
│   └── final_report.md             # Research report
│
└── .gitignore
```

### Directory Purposes

| Directory | Purpose | Does NOT contain |
|-----------|---------|-----------------| 
| `src/data/` | Data loading, cleaning, universe definition | Feature computation, signal logic |
| `src/features/` | All feature calculations (returns, rolling stats, volume) | Signal logic, trading logic |
| `src/signals/` | Signal construction from features | Feature computation, backtesting |
| `src/statistics/` | Statistical tests, HAC inference, and multiple-testing correction | Trading logic, portfolio logic |
| `src/portfolio/` | Portfolio weight computation from signals | Execution, backtesting |
| `src/execution/` | Transaction cost and fill models (single-pass) | Portfolio logic, signal logic |
| `src/backtest/` | Event-driven backtest engine | Signal computation, statistical tests |
| `src/evaluation/` | Performance metrics, plots, reports | Trading logic |
| `src/validation/` | Walk-forward splitting | Statistical tests |
| `src/robustness/` | Robustness test orchestration | Core signal logic |
| `notebooks/` | Research experiments, visualization | Reusable library functions |
| `configs/` | YAML experiment configurations | Code |
| `tests/` | Unit and integration tests | Research experiments |
| `data/` | Raw and processed data files | Code, notebooks |
| `results/` | Experiment outputs | Source code |

---

## 9. Data Strategy

### Primary Data Source

**Yahoo Finance via `yfinance`** (Python library)

| Property | Detail |
|----------|--------|
| Cost | Free |
| Coverage | US equities, ETFs, indices |
| Frequency | Daily OHLCV |
| Adjusted prices | Split- and dividend-adjusted close provided |
| History depth | Typically 20+ years for large caps |
| API limits | Informal rate limits; no API key required |
| Reliability | Adequate for research; not institutional-grade |

### Acceptable Alternatives

| Source | Pros | Cons |
|--------|------|------|
| Alpha Vantage (free tier) | API key, JSON format | 5 calls/min, 500/day |
| Kaggle datasets (e.g., "Huge Stock Market Dataset") | Pre-downloaded, stable | May be stale, survivorship bias |
| FRED (for macro/benchmark data) | Free, reliable | Not individual equities |

### Universe Definition

**Primary universe**: S&P 100 constituents as of the start of the research sample (approximately 2014-01-01), frozen for the duration of the backtest.

**How to construct the start-of-sample universe**: Obtain the S&P 100 constituent list as of January 2014 (or the closest reliably documented date) from a public historical source such as a dated Wikipedia revision, Wayback Machine snapshot, or another historical reference. Store the resulting list as `data/universe/sp100_20140101.csv` and commit that universe file to the repository. If a reliable historical 2014 constituent list cannot be established, **do not substitute the current S&P 100 list or heuristically remove later IPOs/additions**. Instead, use only a separately documented historical source/list that establishes start-of-sample membership. If no defensible historical list can be obtained, document the limitation rather than constructing a pseudo-point-in-time universe. The key principle is that universe membership is determined only from information available at the start of the research sample.

| Parameter | Value |
|-----------|-------|
| Number of securities | Historical S&P 100 start-of-sample membership (typically ~80–100 tickers) |
| Type | Large-cap US equities |
| Liquidity | Large-cap universe; any liquidity screen must be point-in-time using only trailing data available on the date |
| Data period | 2014-01-01 to 2024-12-31 (≈10 years) |
| In-sample period | 2014-01-01 to 2019-12-31 |
| Out-of-sample period | 2020-01-01 to 2024-12-31 |
| Membership | Frozen at start-of-sample; no reconstitution |

### Known Limitations and Biases

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **Survivorship bias** | Start-of-sample universe avoids future-membership look-ahead but still suffers from survivorship bias: stocks that delisted or were acquired between 2014 and 2024 may have incomplete data, and the project does not model delisting returns. Results may overstate signal performance. | Document as a known limitation. Acknowledge that survivorship-free data requires paid databases (e.g., CRSP, Compustat) with delisting returns. The project uses survivorship-biased data knowingly and discusses the impact. A fully survivorship-free universe is outside student scope. |
| **Incomplete delistings** | Some 2014 constituents may have been delisted or acquired during the sample period. Their data will end early and their last observation must be recorded. | Document last valid observation for any terminated security. Do not forward-fill beyond termination. See Missing Data policy. |
| **Adjusted vs. unadjusted prices** | Yahoo's adjusted close accounts for splits and dividends; raw close does not. | Use adjusted close throughout as a total-return approximation for research, features, signals, and synthetic backtest reference prices. Use split-aware volume for volume-based features where reliable split factors are available; otherwise exclude split-affected dates from volume-anomaly features. |
| **Corporate actions** | Splits, dividends, mergers. | Adjusted close handles most. For mergers/acquisitions that cause a ticker to disappear, document the last valid observation. |
| **Missing data** | Some tickers may have missing trading days. | Missing observations preserved as NaN. Never forward-fill OHLC bars. No full-sample missing-data percentage is used to remove start-of-sample securities because that would use future information. A security is usable on date t only when the required historical observations are available up to t. |
| **Timezone** | Yahoo Finance returns data in exchange timezone (US/Eastern for NYSE/NASDAQ). | All timestamps converted to `US/Eastern` and stored as timezone-aware dates. |
| **Data provenance** | Yahoo Finance data is scraped and may differ slightly from exchange records. | Not suitable for institutional research; adequate for a learning project. |

### Professional vs. Student Data

| Aspect | Student project (this) | Professional setting |
|--------|----------------------|---------------------|
| Data source | Yahoo Finance (free) | CRSP, TAQ, Bloomberg, Refinitiv |
| Survivorship | Start-of-sample membership; delisting bias remains | Survivorship-free with delisting returns |
| Corporate actions | Adjusted close only | Full corporate action database |
| Intraday data | Not used | Tick-level or minute bars |
| Point-in-time | Not available | Point-in-time constituents, fundamentals |

---

## 10. Data Schema

### 10.1 Raw Market Data

**Storage**: Parquet files, one per ticker or one combined file.

**DataFrame schema** (`raw_ohlcv`):

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `date` | `datetime64[ns]` (index) | Trading day (US/Eastern) | Trading date |
| `ticker` | `str` | — | Security identifier |
| `open` | `float64` | USD | Unadjusted open price |
| `high` | `float64` | USD | Unadjusted high price |
| `low` | `float64` | USD | Unadjusted low price |
| `close` | `float64` | USD | Unadjusted close price |
| `adj_close` | `float64` | USD | Split- and dividend-adjusted close |
| `volume` | `int64` | shares | Raw daily trading volume as provided by the source |
| `volume_split_adjusted` | `float64` | split-adjusted shares | Split-adjusted volume used for volume-anomaly features when reliable split factors are available |

**Indexing**: MultiIndex `(date, ticker)` or separate date index with ticker as column level in a wide-format panel.

**Missing-value policy**: NaN for missing prices; NaN for missing volume (not 0). The start-of-sample universe is not filtered using a full-sample missing-data percentage. A ticker remains a member of the research universe but is eligible for a signal on date t only if the required historical observations up to t are available.

### 10.2 Cleaned Market Data

Same schema as raw, after:
- Removal of non-trading days (using NYSE calendar)
- **No forward-fill of OHLC prices** — missing bars remain NaN
- Removal of tickers failing quality checks
- Validation: `low ≤ open, close ≤ high`; `volume ≥ 0`; `adj_close > 0`
- Documentation of last valid observation for any security that terminates during the sample period
- Reconstruction of `volume_split_adjusted` using split factors when reliably available
- If split factors cannot be reconstructed reliably, flag split-affected dates and exclude them from volume-anomaly feature calculations rather than treating the mechanical volume jump as an information signal

Stored as Parquet with metadata (cleaning date, universe version, parameters).

An additional boolean column `is_trading_day` (or equivalent mask) distinguishes exchange non-trading days from genuinely missing observations. Exchange non-trading days (weekends, holidays) are removed from the dataset entirely. Remaining NaN values represent genuinely missing data for a ticker on a day when the exchange was open.

**Point-in-time eligibility rule**: Data-quality eligibility is evaluated using only observations available on or before the current date. Do not exclude a ticker from the historical universe because of missingness, future delisting, or future liquidity information that becomes known later in the sample. For a given signal date t, require the minimum lookback needed by that signal (plus any required lag) to be valid through t; otherwise emit NaN and take no new position in that ticker.

### 10.3 Feature Data

**DataFrame schema** (`features`):

| Column | Type | Description |
|--------|------|-------------|
| `date` | `datetime64[ns]` (index level 0) | Feature computation date |
| `ticker` | `str` (index level 1) | Security |
| `ret_1d` | `float64` | 1-day simple return |
| `log_ret_1d` | `float64` | 1-day log return |
| `ret_5d` | `float64` | 5-day cumulative return |
| `ret_20d` | `float64` | 20-day cumulative return |
| `ret_60d` | `float64` | 60-day cumulative return |
| `ret_252d` | `float64` | 252-day (≈1 year) cumulative return |
| `ret_252d_skip21d` | `float64` | 252-day return skipping most recent 21 days |
| `sma_20` | `float64` | 20-day simple moving average of adj_close |
| `sma_60` | `float64` | 60-day simple moving average of adj_close |
| `rolling_std_ret_20` | `float64` | 20-day rolling std of daily returns |
| `rolling_std_ret_60` | `float64` | 60-day rolling std of daily returns |
| `rolling_std_price_20` | `float64` | 20-day rolling std of adj_close prices |
| `zscore_price_20` | `float64` | z-score: (price − SMA_20) / rolling_std_price_20 |
| `realized_vol_20` | `float64` | 20-day annualized realized volatility (√252 × rolling_std_ret_20) |
| `realized_vol_60` | `float64` | 60-day annualized realized volatility (√252 × rolling_std_ret_60) |
| `volume_sma_20` | `float64` | 20-day moving average of split-adjusted volume |
| `relative_volume` | `float64` | split-adjusted volume / volume_sma_20 |
| `volume_zscore_20` | `float64` | z-score of split-adjusted volume relative to 20-day rolling distribution |
| `rank_ret_20d` | `float64` | Cross-sectional percentile rank of 20-day return (0 to 1) |
| `rank_ret_252d_skip21d` | `float64` | Cross-sectional percentile rank of momentum |
| `rank_vol_20` | `float64` | Cross-sectional percentile rank of 20-day volatility |

**Critical distinction**: `rolling_std_ret_20` is the rolling standard deviation of daily **returns** (dimensionless). `rolling_std_price_20` is the rolling standard deviation of **adj_close prices** (in USD). These are different quantities serving different purposes. The z-score uses `rolling_std_price_20` as its denominator, because the numerator is a price deviation.

**Missing-value policy**: Features with insufficient lookback are NaN. Features computed on a day where the underlying OHLCV observation is NaN (missing bar) are also NaN. These rows are excluded from signal computation. NaN propagation is explicit (no silent fill). Features must never be computed on forward-filled or imputed prices.

### 10.4 Forward Returns

**Definition**: Forward returns are computed using adjusted close prices for signal evaluation (IC analysis, quintile analysis). They must use the same convention everywhere.

| Forward return | Formula | Used for |
|---------------|---------|----------|
| `fwd_ret_1d` | $\text{adj\_close}[t+1] / \text{adj\_close}[t] - 1$ | Short-horizon IC |
| `fwd_ret_5d` | $\text{adj\_close}[t+5] / \text{adj\_close}[t] - 1$ | Mean-reversion evaluation |
| `fwd_ret_20d` | $\text{adj\_close}[t+20] / \text{adj\_close}[t] - 1$ | Momentum evaluation |

**Important**: For overlapping horizons (5-day, 20-day), consecutive daily observations share most of the same return period. For example, `fwd_ret_20d` on day $t$ and day $t+1$ share 19 of 20 return days. This creates serial correlation in IC series, which must be accounted for in statistical inference (see Section 14).

### 10.5 Signal Data

**DataFrame schema** (`signals`):

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `date` | `datetime64[ns]` (index level 0) | — | Signal computation date |
| `ticker` | `str` (index level 1) | — | Security |
| `momentum_signal` | `float64` | [−1, 1] or ranked | Momentum signal value |
| `mean_reversion_signal` | `float64` | [−1, 1] or ranked | Mean-reversion signal value |
| `volatility_signal` | `float64` | [−1, 1] or ranked | Volatility signal value |
| `volume_signal` | `float64` | [−1, 1] or ranked | Abnormal-volume signal value |
| `combined_signal` | `float64` | [−1, 1] or ranked | Equal-weight combination |

**Convention**: Positive signal = expected positive future return (long bias). Negative signal = expected negative future return (short bias).

**Critical timing rule**: A signal computed using data up to and including close of day $t$ is stamped with date $t$. It may only be acted upon after the close of day $t$. An order generated from signal[t] is assumed to execute at close[t+1].

### 10.6 Portfolio Weights

**DataFrame schema** (`weights`):

| Column | Type | Description |
|--------|------|-------------|
| `date` | `datetime64[ns]` (index level 0) | Target date (position effective at close of this date) |
| `ticker` | `str` (index level 1) | Security |
| `target_weight` | `float64` | Target portfolio weight; gross exposure constrained ≤ 1.0 (or as configured) |
| `signal_name` | `str` | Which signal generated this weight |
| `portfolio_method` | `str` | Construction method (equal_weight, signal_weight, vol_scaled) |

### 10.7 Orders

**Dataclass / named tuple**:

```python
@dataclass
class Order:
    order_id: str           # Unique order identifier (UUID or sequential)
    timestamp: datetime     # When the order was generated
    ticker: str             # Security
    direction: str          # "BUY" or "SELL"
    quantity: float         # Number of shares (can be fractional for sizing)
    order_type: str         # "MARKET" (only type for this project)
    signal_name: str        # Originating signal
```

### 10.8 Fills

```python
@dataclass
class Fill:
    fill_id: str            # Unique fill identifier
    order_id: str           # Reference to originating order
    timestamp: datetime     # Execution timestamp
    ticker: str             # Security
    direction: str          # "BUY" or "SELL"
    quantity: float         # Filled quantity
    reference_price: float  # Reference execution price (adj_close[t+1])
    fill_price: float       # Effective execution price (after spread + slippage adjustment)
    commission: float       # Commission paid (USD) — separate cash deduction
```

**Accounting note**: `fill_price` already incorporates spread and slippage adjustments. Commission is a separate cash cost. The portfolio tracker uses `fill_price` for position entry and `commission` for cash adjustment. Spread and slippage are NOT counted again as separate line items. See Section 18 for full accounting formulas.

### 10.9 Positions

```python
@dataclass
class Position:
    ticker: str
    quantity: float         # Current shares held (negative = short)
    avg_entry_price: float  # Volume-weighted average entry price (using fill_price)
    current_price: float    # Most recent adj_close price
    market_value: float     # quantity × current_price
    unrealized_pnl: float   # market_value − (quantity × avg_entry_price)
    realized_pnl: float     # Cumulative realized PnL from closed trades
    termination_liquidation: bool  # True if position was force-closed under the simplified termination rule
```

### 10.10 Portfolio Snapshot

```python
@dataclass
class PortfolioSnapshot:
    date: datetime
    cash: float                     # Available cash
    positions: dict[str, Position]  # ticker → Position
    total_market_value: float       # Sum of all position market values
    total_equity: float             # cash + total_market_value
    total_unrealized_pnl: float
    total_realized_pnl: float
    gross_exposure: float           # Sum of |market_value| for each position / total_equity
    net_exposure: float             # Sum of market_value / total_equity (long − short)
```

### 10.11 Performance Metrics

```python
@dataclass
class PerformanceMetrics:
    total_return: float             # (final_equity / initial_equity) − 1
    cagr: float                     # Compound Annual Growth Rate
    annualized_volatility: float    # Std of daily returns × √252
    sharpe_ratio: float             # Zero-risk-free-rate Sharpe: (mean_daily_return / std_daily_return) × √252
    sortino_ratio: float            # (mean_daily_return / downside_std) × √252
    max_drawdown: float             # Maximum peak-to-trough decline
    calmar_ratio: float             # CAGR / |max_drawdown|
    daily_hit_rate: float           # Fraction of trading days with positive portfolio return
    avg_daily_return: float         # Mean daily portfolio return
    avg_daily_turnover: float       # Average daily portfolio turnover
    gross_exposure_avg: float       # Average gross exposure
    net_exposure_avg: float         # Average net exposure
    return_skewness: float          # Skewness of daily returns
    return_kurtosis: float          # Excess kurtosis of daily returns
    num_rebalances: int             # Total number of rebalancing days with trades
```

**Note**: `daily_hit_rate` is precisely defined as the fraction of trading days on which the portfolio's daily return was positive. `avg_daily_return` is the arithmetic mean of daily portfolio returns. No ambiguous "average trade return" metric is included — in a daily-rebalancing context, "trade" is not well-defined as a unit.

### 10.12 Experiment Metadata

**Stored as JSON**:

```json
{
    "experiment_id": "exp_20240315_momentum_v1",
    "config_file": "configs/momentum.yaml",
    "config_hash": "sha256:abc123...",
    "timestamp_start": "2024-03-15T10:30:00Z",
    "timestamp_end": "2024-03-15T10:35:42Z",
    "random_seed": 42,
    "data_version": "v1_20240301",
    "universe": "sp100_20140101",
    "train_start": "2014-01-01",
    "train_end": "2017-12-31",
    "val_start": "2018-01-01",
    "val_end": "2019-12-31",
    "test_start": "2020-01-01",
    "test_end": "2023-12-31",
    "holdout_start": "2024-01-01",
    "holdout_end": "2024-12-31",
    "signal_name": "momentum",
    "signal_params": {"lookback": 252, "skip": 21},
    "hypothesis_type": "confirmatory",
    "portfolio_method": "vol_scaled",
    "cost_model": {"commission_bps": 5, "spread_bps": 5, "slippage_bps": 5},
    "metrics": { "...": "..." },
    "all_tested_configurations": [ "..." ],
    "python_version": "3.11.x",
    "git_commit": "abc1234"
}
```

---

## 11. Data Pipeline

### Processing Order

```
1. Download raw data (scripts/download_data.py)
       ↓
2. Validate raw data (src/data/cleaning.py)
   - Check for missing dates, negative prices, zero-volume days
   - Log anomalies
       ↓
3. Clean data (src/data/cleaning.py)
   - Remove exchange non-trading days (using NYSE calendar)
   - Do NOT forward-fill missing OHLC bars — preserve NaN
   - Do not drop start-of-sample securities using full-sample missingness; use point-in-time eligibility when generating features/signals
   - Validate: low ≤ open, close ≤ high
   - Document last valid observation for terminated securities
       ↓
4. Define universe (src/data/universe.py)
   - Load start-of-sample S&P 100 constituents (as of ~2014-01-01)
   - Freeze membership permanently
   - Do NOT filter membership using full-sample missingness, future delistings, or future liquidity
   - Apply only point-in-time signal eligibility when computing features/signals
       ↓
5. Compute features (src/features/engine.py)
   - Compute all features from cleaned price/volume data
   - Apply lag discipline: features use data up to and including day t
   - NaN propagation: features on missing-bar days are NaN
       ↓
6. Compute signals (src/signals/*.py)
   - Transform features into trading signals
   - Cross-sectional ranking where specified (only over non-NaN values)
   - Signal date = feature date (day t)
       ↓
7. Store processed data (data/processed/)
   - Parquet files with metadata
```

### Lag Discipline (Critical)

```
Day t:    Market closes → adj_close[t], volume[t] become available
          Features computed using data [t-lookback, ..., t]
          Signal computed from features
          Signal stamped with date t

Day t+1:  At market close, signal from day t is available for action
          Orders generated based on signal[t]
          Orders assumed to execute at adj_close[t+1]
          (This is a synthetic daily-bar execution convention, not a
          claim of obtainable fill price. See Section 17.5.)
          Fill recorded at t+1

Day t+1 close: Position updated, PnL computed
```

### Implementation rule

> **Every feature function must accept a `lag` parameter (default 0). When lag > 0, the output is shifted forward by `lag` periods, meaning today's value uses data from `lag` days ago. For signal construction, the minimum lag is 0 (use today's close to generate today's signal, executable tomorrow). The backtester enforces the timing: signals generated on day $t$ produce orders filled on day $t+1$.**

### Missing Data Handling

| Situation | Action |
|-----------|--------|
| Exchange non-trading day (weekend/holiday) | Removed from dataset entirely (not a missing observation) |
| Missing OHLC bar for a single day (exchange was open) | Preserved as NaN — **not** forward-filled |
| Missing OHLC bar for multiple consecutive days | All remain NaN. The ticker remains a member of the frozen universe; signal eligibility is evaluated point-in-time based on the required lookback. |
| Missing volume for a single day | Set to NaN; exclude from volume features for that day |
| Feature with insufficient lookback | NaN; row excluded from signal computation |
| Feature on a day with NaN underlying data | NaN (features must not be computed on missing observations) |
| NaN signal | No position taken in that ticker on that day |
| Security termination (delisting/acquisition) | Document last valid observation date; all subsequent dates are NaN; do not forward-fill. If a position is open at termination and no delisting-return data is available, force-close the position at the last valid adjusted close under an explicit `termination_liquidation` flag. No new position may be opened after termination. |
| Portfolio valuation when a position's current price is unavailable | Do not leave a position indefinitely marked at a stale price. Handle termination through the explicit simplified termination-liquidation rule above; flag the event in logs and treat the missing delisting-return effect as a known limitation. |

### Alignment Rules

- All DataFrames are indexed by `(date, ticker)`.
- All operations use `pandas.DataFrame.align()` to ensure matching indices before arithmetic.
- No implicit broadcasting across misaligned dates or tickers.

---

## 12. Feature Engineering

### Feature Library

All features are computed in `src/features/` and must be deterministic (no randomness).

#### 12.1 Return Features (`src/features/returns.py`)

| Feature | Formula | Lookback | Lag required? | Leakage risk |
|---------|---------|----------|--------------|-------------|
| `ret_1d` | $(P_t - P_{t-1}) / P_{t-1}$ | 1 day | No (uses past prices) | None |
| `log_ret_1d` | $\ln(P_t / P_{t-1})$ | 1 day | No | None |
| `ret_5d` | $(P_t - P_{t-5}) / P_{t-5}$ | 5 days | No | None |
| `ret_20d` | $(P_t - P_{t-20}) / P_{t-20}$ | 20 days | No | None |
| `ret_60d` | $(P_t - P_{t-60}) / P_{t-60}$ | 60 days | No | None |
| `ret_252d` | $(P_t - P_{t-252}) / P_{t-252}$ | 252 days | No | None |
| `ret_252d_skip21d` | $(P_{t-21} - P_{t-252}) / P_{t-252}$ | 252 days, skip 21 | No | None |

$P_t$ = `adj_close` on day $t$. If any price used in the formula is NaN (due to a missing bar), the return is NaN.

**Why these exist**: Different return horizons capture different time-scale effects. The 252-day return with 21-day skip is the classic Jegadeesh-Titman momentum formulation (excludes the short-term reversal effect of the most recent month).

#### 12.2 Momentum Features (defined in returns, used in `src/signals/momentum.py`)

| Feature | Formula | Predictive intuition |
|---------|---------|---------------------|
| `momentum_12_1` | `ret_252d_skip21d` | Past winners continue to outperform (Jegadeesh & Titman, 1993) |
| `momentum_6_1` | $(P_{t-21} - P_{t-126}) / P_{t-126}$ | Shorter-horizon variant |

**Expected failure modes**: Momentum crashes (sudden reversals in trending markets); poor performance in high-volatility regimes.

#### 12.3 Mean-Reversion Features (`src/features/technical.py`)

| Feature | Formula | Predictive intuition |
|---------|---------|---------------------|
| `sma_20` | $\frac{1}{20}\sum_{i=0}^{19} P_{t-i}$ | Smoothed price level |
| `sma_60` | $\frac{1}{60}\sum_{i=0}^{59} P_{t-i}$ | Longer smoothed price |
| `rolling_std_ret_20` | $\text{std}(r_{t}, r_{t-1}, \ldots, r_{t-19})$ | Recent return dispersion (used for realized volatility) |
| `rolling_std_price_20` | $\text{std}(P_{t}, P_{t-1}, \ldots, P_{t-19})$ | Recent price dispersion (used for z-score denominator) |
| `zscore_price_20` | $\frac{P_t - \text{SMA}_{20}(P)_t}{\text{rolling\_std\_price}_{20,t}}$ | How far price deviates from recent mean, in price-std units |

**Dimensional consistency**: The numerator `(P_t − SMA_20(P))` is in price units (e.g., USD). The denominator `rolling_std_price_20` is also in price units. The resulting z-score is dimensionless — a valid standardized measure.

**This is different from `rolling_std_ret_20`**, which measures the standard deviation of daily returns (dimensionless). Rolling return std is used for realized volatility. Rolling price std is used for the z-score. These two quantities must not be confused or swapped.

**Predictive intuition**: Extreme deviations from a moving average may revert. A z-score of −2 suggests the stock has fallen roughly 2σ (in price terms) below its 20-day mean, which may (or may not) predict a bounce.

**Expected failure modes**: Trending markets where "cheap" keeps getting cheaper; structural breaks.

#### 12.4 Volatility Features (`src/features/technical.py`)

| Feature | Formula | Predictive intuition |
|---------|---------|---------------------|
| `realized_vol_20` | $\sqrt{252} \times \text{rolling\_std\_ret}_{20}$ | Annualized 20-day realized volatility |
| `realized_vol_60` | $\sqrt{252} \times \text{rolling\_std\_ret}_{60}$ | Annualized 60-day realized volatility |
| `vol_ratio` | `realized_vol_20 / realized_vol_60` | Short-term vs. long-term vol; > 1 suggests expanding vol |

**Note**: Realized volatility is based on rolling std of **returns**, not prices. This is the standard financial definition.

**Predictive intuition**: Low-volatility stocks have historically delivered higher returns in some periods and universes (the "low-volatility anomaly"). Volatility clustering means high recent vol predicts high near-term vol.

**Expected failure modes**: Low-vol anomaly may not hold in all periods or universes.

#### 12.5 Volume Features (`src/features/volume.py`)

| Feature | Formula | Predictive intuition |
|---------|---------|---------------------|
| `volume_sma_20` | $\frac{1}{20}\sum_{i=0}^{19} V^{adj}_{t-i}$ | Baseline split-adjusted volume level |
| `relative_volume` | $V^{adj}_t / \text{volume\_sma\_20}_t$ | How unusual today's split-adjusted volume is vs. recent average |
| `volume_zscore_20` | $(V^{adj}_t - \text{volume\_sma\_20}_t) / \text{std}_{20}(V^{adj})$ | Standardized volume anomaly |

If volume is NaN on any day (missing bar), volume features for that day are NaN. Split-affected dates are also excluded when a reliable split adjustment cannot be constructed.

**Predictive intuition**: Abnormally high volume may signal informed trading, news, or institutional activity. Volume spikes combined with price moves may predict continuation or reversal.

**Expected failure modes**: Volume is noisy; expiration days, rebalancing days create false signals; mechanical volume changes around stock splits can create spurious anomalies if not adjusted.

**Corporate-action rule**: Stock splits can mechanically change raw share volume without representing abnormal trading activity. The volume signal must use `volume_split_adjusted` when reliable split factors are available. If reliable adjustment is unavailable, split-affected dates are excluded from the volume-anomaly calculation and logged.

### 12.6 Cross-Sectional Features (`src/features/cross_sectional.py`)

| Feature | Formula | Purpose |
|---------|---------|---------| 
| `rank_*` | Percentile rank (0 to 1) across all tickers with valid data on day $t$ | Normalize signals to be comparable across time and across tickers |
| `cs_zscore_*` | $(x_{i,t} - \bar{x}_t) / \sigma_{x,t}$ | Cross-sectional z-score on day $t$ |

**Why**: Raw feature values change scale over time (e.g., volatility was higher in 2020 than in 2017). Cross-sectional ranking removes this time-varying scale, making signals more stationary.

**Leakage risk**: Cross-sectional operations use data from all tickers on day $t$. This is **not** look-ahead because all tickers' day-$t$ closes are available simultaneously. Tickers with NaN values on day $t$ are excluded from the cross-section for that day.

---

## 13. Signal Specifications

### Signal Interface

All signals implement a common interface:

```python
# src/signals/base.py

from abc import ABC, abstractmethod
import pandas as pd

class BaseSignal(ABC):
    """Base class for all trading signals."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique signal name."""
        ...
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the signal."""
        ...
    
    @abstractmethod
    def compute(self, features: pd.DataFrame, params: dict) -> pd.Series:
        """
        Compute signal values.
        
        Args:
            features: DataFrame with MultiIndex (date, ticker), containing 
                      pre-computed features.
            params: Signal-specific parameters (lookback, thresholds, etc.)
        
        Returns:
            Series with MultiIndex (date, ticker), values in [-1, 1] after
            cross-sectional ranking/normalization.
            
        Signal convention:
            Positive → expected positive future return (go long)
            Negative → expected negative future return (go short)
        """
        ...
```

### Signal 1: Momentum (`src/signals/momentum.py`)

| Property | Value |
|----------|-------|
| **Name** | `momentum_12_1` |
| **Intuition** | Stocks that have risen over the past year (excluding the last month) tend to continue rising. This is one of the most documented anomalies in academic finance (Jegadeesh & Titman, 1993). |
| **Formula** | $\text{signal}_{i,t} = \text{rank}_{cs}\left(\frac{P_{i,t-21} - P_{i,t-252}}{P_{i,t-252}}\right)$ |
| | Cross-sectional rank mapped to [−1, 1]: $2 \times \text{percentile} - 1$ |
| **Lookback** | 252 trading days (≈12 months), skipping 21 days (≈1 month) |
| **Data requirements** | `adj_close` for 252+ trading days per ticker |
| **Assumptions** | (1) Past winners continue to outperform; (2) momentum is a cross-sectional phenomenon (relative, not absolute) |
| **Expected failure modes** | Momentum crashes (sharp reversals); poor in high-vol regimes; reduced after 2000 in some studies |
| **Potential biases** | Survivorship bias inflates momentum returns (dead stocks were losers); transaction costs are high for momentum (high turnover) |
| **Statistical evaluation** | Cross-sectional IC with forward 20-day return; quintile spread; HAC t-test on IC |

**Primary configuration** (confirmatory):

| Parameter | Value |
|-----------|-------|
| `lookback_days` | 252 |
| `skip_days` | 21 |
| `ranking_method` | percentile |
| `forward_horizon` | 20 days |

**Exploratory parameter grid**:

| Parameter | Values |
|-----------|--------|
| `lookback_days` | {126, 189, 252} |
| `skip_days` | {0, 5, 10, 21} |
| `ranking_method` | {percentile, z-score} |

### Signal 2: Mean Reversion (`src/signals/mean_reversion.py`)

| Property | Value |
|----------|-------|
| **Name** | `mean_reversion_zscore` |
| **Intuition** | Stocks that have moved far below their recent average price (in standard-deviation units) tend to revert toward that average over the next few days. |
| **Formula** | $z_{i,t} = \frac{P_{i,t} - \text{SMA}_{k}(P_{i})_t}{\text{rolling\_std\_price}_{k}(P_{i})_t}$ |
| | $\text{signal}_{i,t} = -\text{rank}_{cs}(z_{i,t}) \times 2 + 1$ |
| | (Negative z-score → positive signal, because we expect reversion upward) |
| **Dimensional note** | Both numerator and denominator are in price units (e.g., USD). The z-score is dimensionless. |
| **Lookback** | $k = 20$ trading days |
| **Data requirements** | `adj_close` for 20+ trading days |
| **Assumptions** | (1) Prices are mean-reverting at short horizons; (2) deviations are temporary, not structural |
| **Expected failure modes** | Trending markets; stocks in fundamental decline; distressed stocks ("falling knives") |
| **Potential biases** | Lookahead if z-score is computed using same-day data that includes fill price; survivorship bias (delisted stocks often showed extreme negative z-scores) |
| **Statistical evaluation** | IC with forward 5-day return; auto-correlation of signal; quintile analysis |

**Primary configuration** (confirmatory):

| Parameter | Value |
|-----------|-------|
| `lookback_days` | 20 |
| `forward_horizon` | 5 days |
| `z_threshold` | None (continuous signal) |

**Exploratory parameter grid**:

| Parameter | Values |
|-----------|--------|
| `lookback_days` | {10, 20, 40, 60} |
| `forward_horizon` | {1, 5, 10, 20} |
| `z_threshold` | {None, 1.5, 2.0} |

### Signal 3: Volatility (`src/signals/volatility.py`)

| Property | Value |
|----------|-------|
| **Name** | `low_vol` |
| **Intuition** | Stocks with lower recent realized volatility have historically delivered higher returns in some periods and universes (the "low-volatility anomaly"). |
| **Formula** | $\text{vol}_{i,t} = \sqrt{252} \cdot \text{std}(r_{i,t-k+1}, \ldots, r_{i,t})$ |
| | $\text{signal}_{i,t} = -\text{rank}_{cs}(\text{vol}_{i,t}) \times 2 + 1$ |
| | (Negative rank → positive signal: lower volatility → higher signal) |
| **Lookback** | $k = 60$ trading days |
| **Data requirements** | Daily returns for 60+ days |
| **Assumptions** | Low-vol anomaly holds in this universe and period |
| **Expected failure modes** | Sector concentration (low-vol stocks cluster in utilities, staples); the anomaly may not hold in recent periods |
| **Primary target** | Forward cumulative return (20-day). "Risk-adjusted return" is NOT used as the primary target — it would require an explicit definition and separate analysis. |
| **Statistical evaluation** | IC with forward 20-day return; quintile returns |

**Primary configuration** (confirmatory):

| Parameter | Value |
|-----------|-------|
| `lookback_days` | 60 |
| `forward_horizon` | 20 days |

**Exploratory parameter grid**:

| Parameter | Values |
|-----------|--------|
| `lookback_days` | {20, 40, 60} |
| `vol_measure` | {realized, parkinson} |

### Signal 4: Abnormal Volume (`src/signals/volume_signal.py`)

| Property | Value |
|----------|-------|
| **Name** | `abnormal_volume` |
| **Intuition** | Unusually high trading volume may indicate informed trading, institutional activity, or news. Combined with the direction of the price move, it can predict short-term continuation. |
| **Formula** | $\text{rvol}_{i,t} = \frac{V^{adj}_{i,t}}{\text{SMA}_{20}(V^{adj}_{i})_t}$ |
| | $\text{signal}_{i,t} = \text{rank}_{cs}(\text{sign}(r_{i,t}) \times \log(\text{rvol}_{i,t}))$ mapped to [−1, 1] |
| | Positive return + high volume → positive signal (continuation) |
| | Negative return + high volume → negative signal (continuation of decline) |
| **Lookback** | 20 days for volume baseline; 1 day for return direction |
| **Data requirements** | Daily volume, daily return |
| **Assumptions** | Volume is informative; high volume confirms the direction of the move |
| **Expected failure modes** | Index rebalancing days, options expiration, earnings clustering; volume can be high for non-informative reasons |
| **Statistical evaluation** | IC with forward 5-day return; conditional analysis (high volume days only vs. all days) |

**Primary configuration** (confirmatory):

| Parameter | Value |
|-----------|-------|
| `volume_lookback` | 20 |
| `volume_threshold` | None (continuous) |
| `direction_horizon` | 1 day |
| `forward_horizon` | 5 days |

**Exploratory parameter grid**:

| Parameter | Values |
|-----------|--------|
| `volume_lookback` | {10, 20, 40} |
| `volume_threshold` | {None, 1.5, 2.0} |
| `direction_horizon` | {1, 5} |

### Signal Combination

```python
# Simple equal-weight combination (pre-specified baseline)
combined_signal = (momentum + mean_reversion + volatility + volume) / 4.0

# Re-rank cross-sectionally after combination
combined_signal = cross_sectional_rank(combined_signal) * 2 - 1
```

The combined signal is the **pre-specified statistical baseline for the ML comparison** (see Section 23). Because the four individual signals use different primary forecast horizons, the combined signal is evaluated on a **common 20-day forward-return target** for the ML comparison only. The individual 5-day and 20-day hypotheses remain distinct. The research should evaluate:
1. Each signal at its own pre-specified horizon
2. The equal-weight combined signal against the common 20-day target
3. Whether individual signals add marginal information beyond the combined baseline, evaluated on that same 20-day target

---

## 14. Statistical Testing

### 14.1 Information Coefficient (IC)

**Definition**: The rank correlation between the signal on day $t$ and the forward return over the horizon $[t+1, t+h]$.

$$\text{IC}_t = \text{Spearman}\left(\text{signal}_{t}, r_{t+1:t+h}\right)$$

where $r_{t+1:t+h}$ is the cross-sectional vector of forward returns for all tickers with valid data.

**Implementation** (`src/statistics/information_coefficient.py`):

```python
def compute_ic(
    signal: pd.DataFrame,       # MultiIndex (date, ticker), column: signal value
    forward_returns: pd.DataFrame,  # MultiIndex (date, ticker), column: forward return
    method: str = "spearman"    # "spearman" or "pearson"
) -> pd.Series:
    """
    Compute daily cross-sectional IC.
    
    Returns:
        Series indexed by date, containing IC for each day.
        Days with fewer than 5 valid (signal, forward_return) pairs are set to NaN.
    """
```

**Derived metrics**:

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Mean IC | $\bar{\text{IC}} = \frac{1}{T}\sum_{t=1}^{T} \text{IC}_t$ | Average predictive association |
| IC naive t-statistic | $t_{naive} = \frac{\bar{\text{IC}}}{\text{std}(\text{IC}) / \sqrt{T}}$ | **Diagnostic only** — valid only under i.i.d. assumption |
| IC HAC t-statistic | $t_{HAC} = \frac{\bar{\text{IC}}}{\text{se}_{HAC}(\text{IC})}$ | **Primary inference** — robust to serial correlation |
| IC Information Ratio | $\text{ICIR} = \frac{\bar{\text{IC}}}{\text{std}(\text{IC})}$ | Consistency of signal quality (analogous to Sharpe) |
| Rolling IC | IC computed over rolling 60-day windows | Time variation in signal quality |
| IC hit rate | Fraction of days with $\text{IC}_t > 0$ | Consistency |

**Interpretation guidance** (contextual, not hard thresholds):

IC magnitude depends on the forward-return horizon, the universe, turnover, costs, and methodology. No single IC value determines whether a signal is useful. Statistical significance, stability, and economic significance should be considered jointly. As rough context:
- Mean IC of 0.02–0.05 is commonly seen for meaningful daily equity signals, but these numbers are not acceptance criteria.
- IC consistently above 0.10 should prompt investigation for data leakage or universe artifacts.
- The combination of IC magnitude, stability (ICIR), statistical significance (HAC t-stat), and economic viability (after costs) determines signal quality.

**Null hypothesis**: $H_0$: Mean IC = 0 (signal has no cross-sectional predictive association)

**Alternative**: $H_1$: Mean IC ≠ 0

### 14.2 HAC/Newey-West Inference on IC (Primary)

Daily ICs are likely serially correlated, especially when forward returns overlap. For a 20-day forward return, IC on day $t$ and day $t+1$ share 19 of 20 return observations, creating strong mechanical overlap.

**Newey-West/HAC standard error**:

$$t_{HAC} = \frac{\bar{\text{IC}}}{\text{se}_{HAC}(\text{IC})}$$

where $\text{se}_{HAC}$ is the Newey-West heteroskedasticity- and autocorrelation-consistent standard error, computed with lag parameter ≈ the forward-return horizon (e.g., lag = 20 for 20-day forward returns).

**Implementation**: Use `statsmodels.stats.sandwich_covariance.cov_hac` or equivalent.

The research report must show:
1. Naive t-stat (diagnostic only — assumes i.i.d.)
2. **HAC/Newey-West t-stat (primary inference)**
3. Block bootstrap 95% CI (robustness check)

The naive t-stat must NOT be treated as the definitive significance claim.

### 14.3 Bootstrap Confidence Interval for Mean IC

**Method**: Stationary block bootstrap (block length ≈ forward-return horizon or chosen by autocorrelation).

```
For b = 1 to B (e.g., B = 10,000):
    Resample daily IC series with block bootstrap
    Compute mean of resampled IC
Store B means → compute 2.5th and 97.5th percentiles → 95% CI
```

**Why block bootstrap**: Daily ICs are serially correlated (especially with overlapping forward returns). A standard i.i.d. bootstrap would understate the width of the CI.

**Implementation** (`src/statistics/hypothesis_tests.py`):

```python
def bootstrap_mean_ci(
    series: pd.Series,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    block_size: int | None = None,  # None → auto-select (≈ forward horizon)
    seed: int = 42
) -> tuple[float, float, float]:
    """
    Block bootstrap CI for the mean.
    
    Returns:
        (mean, ci_lower, ci_upper)
    """
```

### 14.4 Permutation Test

**Purpose**: Non-parametric test of whether the observed IC could have arisen by chance.

**Method — within-date cross-sectional permutation**:

```
1. Compute observed mean IC from actual (signal, forward_return) pairs
2. For p = 1 to P (e.g., P = 10,000):
   a. For each date t:
      - Keep forward returns fixed for that date
      - Randomly permute the signal values across securities within that date
      (This breaks the signal-return association while preserving:
       - the time-series structure of returns
       - the cross-sectional structure of returns
       - the daily cross-section size)
   b. Compute daily cross-sectional IC for each date under permutation
   c. Compute mean IC under permutation
3. p-value = fraction of permutation mean ICs ≥ observed mean IC (one-sided)
   or fraction of |permutation mean IC| ≥ |observed mean IC| (two-sided)
```

**Why within-date permutation**: The hypothesis being tested is about cross-sectional association: "does the signal rank predict forward return rank within a cross-section?" The time-series structure of returns (volatility clustering, momentum, etc.) should not be destroyed. The null should preserve the daily cross-sectional structure.

**Why NOT shuffling across dates**: Shuffling signals across dates would destroy the cross-sectional structure and test a different (less appropriate) null hypothesis.

**Implementation** (`src/statistics/hypothesis_tests.py`):

```python
def permutation_test_ic(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_permutations: int = 10_000,
    seed: int = 42
) -> tuple[float, float, pd.Series]:
    """
    Within-date cross-sectional permutation test for mean IC.
    
    For each permutation: on each date, shuffle signal values across tickers,
    recompute cross-sectional IC, then compute mean IC across dates.
    
    Returns:
        (observed_mean_ic, p_value, permutation_distribution)
    """
```

### 14.5 Stationarity Tests (for data exploration)

| Test | Null hypothesis | When to use |
|------|----------------|-------------|
| ADF (Augmented Dickey-Fuller) | Series has a unit root (non-stationary) | Check if return series, features, or signals are stationary |
| KPSS | Series is stationary | Complementary to ADF (tests the opposite null) |

**Use `statsmodels.tsa.stattools.adfuller` and `kpss`.**

**Why**: Non-stationary features may create spurious correlations. If a feature is non-stationary, cross-sectional ranking (which removes the time-series level) is preferred.

### 14.6 Quintile Spread Analysis

**Method**:
1. On each date $t$, sort stocks by signal value (only stocks with valid signal and forward return)
2. Split into quintiles (5 approximately equal-count groups)
3. Compute average forward return of each quintile
4. "Long-short spread" = Q5 (top quintile) return − Q1 (bottom quintile) return
5. Test whether the spread is significantly different from zero

**Edge cases**:
- If fewer than 5 valid securities exist on a date, skip quintile analysis for that date and log a warning.
- Ties are handled using `method='average'` in pandas rank.
- Securities with NaN signal or NaN forward return are excluded before sorting.
- Quintile membership is determined using only information available at time $t$.

**Visualization**: Bar chart of average return by quintile, with error bars.

---

## 15. Multiple-Hypothesis Testing

### The Problem

This project tests multiple signals (at least 4), each with multiple parameter configurations, across multiple assets and time periods. If each test uses a significance level of $\alpha = 0.05$, the probability of at least one false positive increases rapidly:

$$P(\text{at least one false positive}) = 1 - (1 - \alpha)^m$$

For $m = 20$ independent tests: $P = 1 - 0.95^{20} \approx 0.64$.

### 15.1 Confirmatory vs. Exploratory Hypotheses

The project distinguishes two categories of hypotheses:

**Confirmatory hypotheses** — pre-specified before looking at results:
- Momentum (12-1) → forward 20-day return (primary config)
- Mean-reversion (20-day price z-score) → forward 5-day return (primary config)
- Low-volatility (60-day) → forward 20-day return (primary config)
- Abnormal volume (20-day baseline) → forward 5-day return (primary config)

These 4 hypotheses form the **confirmatory family**. Bonferroni and BH-FDR are applied to this family as the primary multiple-testing correction.

**Exploratory research** — parameter sweeps, alternative configurations:
- Multiple lookbacks, skip periods, horizons, thresholds
- Different ranking methods
- Combined signals

Exploratory results must be clearly labeled as exploratory and must NOT automatically be presented as confirmatory evidence. All tested configurations must be logged in the experiment metadata (`all_tested_configurations` field).

**Important limitation**: Applying Bonferroni/FDR at the end does not erase research-selection bias accumulated during the research process. The researcher's choices about which signals to try, which features to construct, and which parameters to explore all constitute implicit hypothesis selection that cannot be fully corrected by post-hoc multiple-testing adjustment. The report must acknowledge this.

### 15.2 Bonferroni Correction

**Adjusted significance level**: $\alpha_{\text{adj}} = \alpha / m$

where $m$ = number of hypotheses in the confirmatory family (= 4 for the primary signals).

**Example**: $\alpha_{\text{adj}} = 0.05 / 4 = 0.0125$

A confirmatory signal is significant only if its HAC p-value < 0.0125.

**Pros**: Simple, controls family-wise error rate (FWER).
**Cons**: Very conservative, especially when tests are correlated (which they will be, since signals share the same returns data).

**Implementation** (`src/statistics/multiple_testing.py`):

```python
def bonferroni_correction(
    p_values: dict[str, float],  # signal_name → p-value (from HAC test)
    alpha: float = 0.05
) -> dict[str, dict]:
    """
    Apply Bonferroni correction.
    
    Returns:
        {signal_name: {"p_value": float, "adjusted_alpha": float, "significant": bool}}
    """
```

### 15.3 Benjamini-Hochberg FDR

**Method**:
1. Sort $m$ p-values in ascending order: $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}$
2. Find the largest $k$ such that $p_{(k)} \leq \frac{k}{m} \alpha$
3. Reject all hypotheses $H_{(1)}, \ldots, H_{(k)}$

**Pros**: Less conservative than Bonferroni; controls False Discovery Rate rather than FWER.
**Cons**: Assumes tests are independent or positively dependent (PRDS condition).

**Implementation** (`src/statistics/multiple_testing.py`):

```python
def benjamini_hochberg(
    p_values: dict[str, float],
    alpha: float = 0.05
) -> dict[str, dict]:
    """
    Apply Benjamini-Hochberg FDR correction.
    
    Returns:
        {signal_name: {"p_value": float, "bh_threshold": float, "significant": bool, "rank": int}}
    """
```

### 15.4 Hypothesis Registry

The project must maintain a log of all tested configurations:

```python
def log_hypothesis(
    registry: list[dict],
    signal_name: str,
    params: dict,
    hypothesis_type: str,  # "confirmatory" or "exploratory"
    forward_horizon: int,
    result: dict
) -> None:
    """Append a tested configuration to the registry."""
```

The final report must include the total number of configurations explored and explain that the confirmatory family was pre-specified.

### 15.5 Demonstration Experiment

**Experiment 9 (see Experiment Matrix)** is specifically designed to demonstrate the multiple-testing problem:

1. Generate $N = 100$ random "signals" with no real predictive content (random cross-sectional ranks)
2. Compute IC and HAC p-value for each
3. Repeat the null experiment across several random seeds/runs and show that the number of nominally significant results varies stochastically but is approximately 5 on average at $\alpha = 0.05$
4. Apply Bonferroni and BH-FDR
5. Demonstrate that correction substantially reduces false discoveries; do not require every null simulation to produce zero rejections

This experiment is pedagogically valuable and demonstrates understanding of the multiple-testing problem. The observed number of false positives in any one simulation is random and should not be forced to equal the theoretical expectation.

---

## 16. Portfolio Construction

### 16.1 Exposure Constraints (Default)

For the student project, the following constraints are **hard constraints** on every generated target-weight vector. The portfolio-construction layer must return weights that satisfy them, and unit tests must verify them.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_gross_exposure` | 1.0 (100%) | Sum of absolute weights may not exceed this value |
| `max_net_exposure` | 0.20 (20%) | Absolute sum of weights may not exceed this value |
| `max_position_weight` | 0.10 (10%) | Absolute weight of any single stock may not exceed this value |
| `leverage` | None above 1x gross | No leverage beyond the gross-exposure cap |
| `borrow_fees` | Not modeled | Simplified short assumption |

**Hard-constraint rule**: If raw signal weights violate a constraint, the portfolio constructor must project/normalize them to satisfy all configured limits before returning the weights. A final validation step must assert:

```text
sum(abs(w_i)) <= max_gross_exposure
abs(sum(w_i)) <= max_net_exposure
max(abs(w_i)) <= max_position_weight
```

For long-short portfolios, the default construction should allocate approximately 50% of gross exposure to the long side and 50% to the short side, yielding net exposure near zero. Position caps must be enforced on each side without introducing leverage. If there are too few valid securities to satisfy the configured caps, the portfolio should scale gross exposure down rather than violate the constraints.

**Point-in-time rule**: Constraints are applied using only the signal/volatility information available at the rebalancing date. They must never use future returns or future volatility estimates.

### 16.2 Equal-Weight Signal Portfolio

**Method**: On each rebalancing date, rank stocks by signal. Go long the top quintile (Q5) with equal weight. Go short the bottom quintile (Q1) with equal weight. Neutral on Q2–Q4.

$$w_{i,t} = \begin{cases} +1/(2 \cdot N_{long}) & \text{if } \text{signal}_{i,t} \in Q5 \\ -1/(2 \cdot N_{short}) & \text{if } \text{signal}_{i,t} \in Q1 \\ 0 & \text{otherwise} \end{cases}$$

where $N_{long}$, $N_{short}$ are the number of stocks in Q5 and Q1.

**Gross exposure**: $\sum|w_i| = 1.0$ (50% long + 50% short).

**Net exposure**: $\sum w_i \approx 0$ (approximately market-neutral).

**Why this exists**: Simplest possible portfolio from a cross-sectional signal. Easy to understand and debug.

### 16.3 Signal-Weighted Portfolio

**Method**: Weights are proportional to signal strength, with separate long/short normalization and hard position caps.

For a cross-sectional signal with both positive and negative values, allocate 50% of gross exposure to positive signals and 50% to negative signals. Within each side, weights are proportional to absolute signal magnitude. Apply `max_position_weight` using an iterative capped-normalization procedure that redistributes excess weight only across uncapped positions. If a side cannot be fully allocated without violating the cap, reduce gross exposure rather than violate the constraint.

The final weights must satisfy:

$$\sum_i |w_{i,t}| \leq \text{gross\_exposure}$$

$$|\sum_i w_{i,t}| \leq \text{max\_net\_exposure}$$

$$|w_{i,t}| \leq \text{max\_position\_weight}$$

**Why**: Allows stronger signals to receive more weight while maintaining explicit portfolio-risk constraints.

### 16.4 Volatility-Scaled Portfolio

**Method**: Scale signal-weighted positions inversely by recent volatility, then apply the same hard portfolio constraints defined in Section 16.1.

$$w_{i,t}^{raw} = \frac{\text{signal}_{i,t}}{\text{vol}_{i,t}}$$

where $\text{vol}_{i,t}$ is the 60-day realized volatility of stock $i$ computed using information available through date $t$. The raw weights are then passed through the hard-constraint normalization/capping procedure.

**Why**: Prevents high-volatility stocks from dominating the raw signal allocation while maintaining a simple, explainable student-level portfolio construction method.

**Portfolio-level volatility targeting is NOT part of the Strong version** because scaling a 1x-gross portfolio by a factor above 1 would contradict the project's no-leverage constraint. It is listed only as a future extension.

**Implementation** (`src/portfolio/construction.py`):

```python
def equal_weight_long_short(
    signal: pd.DataFrame,
    n_quantiles: int = 5,
    long_quantile: int = 5,
    short_quantile: int = 1,
    gross_exposure: float = 1.0,
    max_position_weight: float = 0.10,
    max_net_exposure: float = 0.20
) -> pd.DataFrame:
    """Returns constrained target weights."""

def signal_weighted(
    signal: pd.DataFrame,
    gross_exposure: float = 1.0,
    max_position_weight: float = 0.10,
    max_net_exposure: float = 0.20
) -> pd.DataFrame:
    """Returns constrained signal-proportional target weights."""

def volatility_scaled(
    signal: pd.DataFrame,
    volatility: pd.DataFrame,
    gross_exposure: float = 1.0,
    max_position_weight: float = 0.10,
    max_net_exposure: float = 0.20
) -> pd.DataFrame:
    """Returns constrained inverse-volatility-scaled target weights."""


def validate_weight_constraints(
    weights: pd.DataFrame,
    max_gross_exposure: float = 1.0,
    max_net_exposure: float = 0.20,
    max_position_weight: float = 0.10
) -> None:
    """Raise an error if any hard portfolio constraint is violated."""
```

---

## 17. Event-Driven Backtester

### 17.1 Why Event-Driven?

| Approach | Pros | Cons |
|----------|------|------|
| **Vectorized** (`signal × return`) | Fast, simple | Cannot model order lifecycle, timing, slippage, partial fills, or position tracking accurately |
| **Event-driven** | Models order flow realistically; catches timing bugs; tracks cash and positions | Slower, more complex |

**Decision**: The project uses an event-driven backtester because:
1. It forces correct signal/order/fill timing
2. It enables realistic transaction-cost modeling
3. It produces realistic position and PnL tracking
4. It's a valuable software-engineering artifact for interviews
5. It catches look-ahead bugs that vectorized approaches hide

The vectorized approach is still used in notebooks for quick signal analysis (IC, quintile spreads). The event-driven backtester is used for final performance evaluation.

### 17.2 Event Types

```python
# src/backtest/events.py
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"

@dataclass
class MarketEvent:
    """New market data bar available."""
    event_type: EventType = EventType.MARKET
    date: datetime = None
    data: dict = None  # {ticker: {open, high, low, close, adj_close, volume}}

@dataclass
class SignalEvent:
    """Signal computed from market data."""
    event_type: EventType = EventType.SIGNAL
    date: datetime = None
    signal_name: str = ""
    weights: dict = None  # {ticker: target_weight}

@dataclass
class OrderEvent:
    """Order generated from signal."""
    event_type: EventType = EventType.ORDER
    date: datetime = None
    orders: list = None  # List of Order objects

@dataclass
class FillEvent:
    """Order has been filled."""
    event_type: EventType = EventType.FILL
    date: datetime = None
    fills: list = None  # List of Fill objects
```

### 17.3 Backtest Engine

```python
# src/backtest/engine.py

class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Processes one trading day at a time:
    
    Day t:
        1. Receive MarketEvent (adj_close prices for day t)
        2. Update position mark-to-market using adj_close[t]
        3. Execute pending orders at adj_close[t] price (orders generated yesterday)
           → Apply single-pass cost model to compute effective fill price
        4. Compute signal using data up to and including close[t]
        5. Generate new orders (to be executed tomorrow)
        6. Record portfolio snapshot
    
    Day t+1:
        1. Receive MarketEvent (adj_close prices for day t+1)
        2. Update position mark-to-market
        3. Execute orders from day t at adj_close[t+1] price
        4. ... repeat
    
    Price convention: adj_close (adjusted close) is used throughout as a
    daily-bar total-return approximation. See Section 9 for details and
    limitations.
    """
    
    def __init__(
        self,
        market_data: pd.DataFrame,
        strategy: BaseStrategy,
        broker: Broker,
        portfolio_tracker: PortfolioTracker,
        cost_model: CostModel,
        initial_cash: float = 1_000_000.0
    ):
        ...
    
    def run(self) -> BacktestResult:
        """
        Run the backtest over all trading days.
        
        Returns:
            BacktestResult containing daily equity curve, trades, metrics.
        """
        ...
```

### 17.4 Component Responsibilities

| Component | Responsibility | Does NOT do |
|-----------|---------------|-------------|
| `BacktestEngine` | Orchestrate the event loop; iterate over dates | Compute signals, execute fills |
| `Strategy` (wraps signal + portfolio construction) | Receive MarketEvent; emit SignalEvent with target weights | Execute orders, track positions |
| `Broker` | Receive target weights; compute trades needed; generate Orders; simulate Fills using single-pass cost model | Compute signals, decide weights |
| `PortfolioTracker` | Track cash, positions, PnL; record daily snapshots | Generate orders |
| `CostModel` | Compute effective fill price and commission for a given order | Anything else |

### 17.5 Order Timing Model

```
                Day t                              Day t+1
    ┌──────────────────────────┐      ┌──────────────────────────┐
    │ Market data available    │      │ Market data available    │
    │ at close of day t        │      │ at close of day t+1      │
    │                          │      │                          │
    │ Signal computed using    │      │ Orders from day t        │
    │ data up to close[t]      │      │ assumed executed at      │
    │                          │      │ adj_close[t+1]           │
    │ Orders generated         │      │                          │
    │ (pending until t+1)      │      │ Fills recorded with      │
    │                          │      │ single-pass cost applied │
    │                          │      │ Positions updated        │
    │                          │      │ PnL computed             │
    └──────────────────────────┘      └──────────────────────────┘
```

**Execution reference price**: Adjusted close price of day $t+1$.

**Execution convention**: This is a deliberately simplified daily-bar execution convention. It is chosen because:
- It is simple and reproducible.
- It is consistent with daily data (only close prices are reliable from Yahoo Finance).
- It prevents same-bar look-ahead (signal at t cannot use t+1 data).
- It is easy to test and verify.

**What this convention is NOT**: It is NOT a claim that the trader can literally obtain the official closing price in a real market. The execution price is synthetic — a research approximation. The project does not model intraday execution, market-on-close order mechanics, or auction dynamics.

### 17.6 Position and PnL Tracking

The `PortfolioTracker` maintains:

```python
class PortfolioTracker:
    def __init__(self, initial_cash: float):
        self.cash: float = initial_cash
        self.positions: dict[str, Position] = {}
        self.history: list[PortfolioSnapshot] = []
    
    def update_market_prices(self, date: datetime, prices: dict[str, float]):
        """Mark all positions to market using new adj_close prices."""
        ...
    
    def process_fill(self, fill: Fill):
        """
        Update cash and positions based on a fill.
        
        SINGLE-PASS COST ACCOUNTING:
        
        BUY: cash -= (fill_price × quantity + commission)
             position.quantity += quantity
             (fill_price already includes spread/slippage adjustment)
        
        SELL: cash += (fill_price × quantity - commission)
              position.quantity -= quantity
              (fill_price already includes spread/slippage adjustment)
        
        Spread and slippage are embedded in fill_price.
        Commission is the only separate cash deduction.
        There is NO additional spread/slippage line item.
        """
        ...
    
    def take_snapshot(self, date: datetime) -> PortfolioSnapshot:
        """Record current portfolio state."""
        ...
    
    def get_equity_curve(self) -> pd.Series:
        """Return daily total_equity as a time series."""
        ...
```

### 17.7 Unit Test Requirements for the Backtester

These tests are **critical**:

| Test | Purpose |
|------|---------|
| `test_no_orders_on_first_day` | Signal on day 1 should not generate fills on day 1 |
| `test_fill_uses_next_day_price` | Orders generated from signal[t] use adj_close[t+1] as reference price |
| `test_cash_decreases_on_buy` | Cash decreases by fill_price × quantity + commission |
| `test_cash_increases_on_sell` | Cash increases by fill_price × quantity − commission |
| `test_position_tracking` | After buying 100 shares, position.quantity = 100 |
| `test_pnl_calculation` | Known buy/sell prices → verify realized PnL |
| `test_equity_curve` | Sum of cash + position values matches total_equity |
| `test_costs_counted_once` | Single-pass: spread/slippage not double-counted (see Section 28) |
| `test_known_result` | 3-day toy backtest with hand-calculated result |

---

## 18. Execution and Transaction Costs

### 18.1 Cost Model (Single-Pass Accounting)

```python
# src/execution/costs.py

@dataclass
class CostConfig:
    """Transaction cost configuration."""
    commission_bps: float = 5.0      # Commission in basis points of trade value
    spread_bps: float = 5.0          # Half bid-ask spread in basis points
    slippage_bps: float = 5.0        # Market impact / slippage in basis points
    min_commission: float = 0.0      # Minimum commission per trade (USD)

class CostModel:
    def __init__(self, config: CostConfig):
        self.config = config
    
    def compute_fill(self, reference_price: float, quantity: float, direction: str) -> dict:
        """
        Compute effective fill price and commission using SINGLE-PASS accounting.
        
        The effective fill price absorbs spread and slippage.
        Commission is a separate cash cost.
        
        Args:
            reference_price: Reference execution price (adj_close[t+1])
            quantity: Number of shares (absolute value)
            direction: "BUY" or "SELL"
        
        Returns:
            {"fill_price": float, "commission": float}
        
        Formulas:
            execution_adjustment = (spread_bps + slippage_bps) / 10000
            
            For BUY:
                fill_price = reference_price × (1 + execution_adjustment)
            For SELL:
                fill_price = reference_price × (1 - execution_adjustment)
            
            commission = max(min_commission, |reference_price × quantity| × commission_bps / 10000)
        
        IMPORTANT: fill_price already includes spread and slippage.
        The PortfolioTracker uses fill_price for position accounting and
        deducts commission separately. Spread and slippage are NOT
        counted a second time.
        
        Hand-verifiable example:
            reference_price = 100.00, quantity = 100, direction = "BUY"
            spread_bps = 5, slippage_bps = 5, commission_bps = 5
            
            execution_adjustment = (5 + 5) / 10000 = 0.001
            fill_price = 100.00 × 1.001 = 100.10
            commission = |100.00 × 100| × 5 / 10000 = 0.50
            
            Cash change = -(100.10 × 100 + 0.50) = -$10,010.50
            Position change = +100 shares at avg_entry_price = 100.10
        """
        ...
```

### 18.2 Cost Regimes for Sensitivity Analysis

| Regime | Commission (bps) | Spread (bps) | Slippage (bps) | Total one-way (bps) | Scenario |
|--------|-----------------|-------------|----------------|--------------------| ---------|
| Zero | 0 | 0 | 0 | 0 | Theoretical upper bound |
| Low | 2 | 3 | 2 | 7 | Very liquid large caps, institutional execution |
| Medium | 5 | 5 | 5 | 15 | Baseline for this project |
| High | 10 | 10 | 10 | 30 | Retail execution, less liquid stocks |
| Very high | 15 | 15 | 20 | 50 | Stress test / worst case |

### 18.3 Fill Model

```python
# src/execution/fill_model.py

class FillModel:
    """
    Simulates order fills using single-pass cost accounting.
    
    Current model: All orders are fully filled at the adjusted close 
    price of the execution day, with spread and slippage absorbed into 
    the fill price, and commission applied separately.
    
    No partial fills in MVP or Strong version.
    """
    
    def simulate_fill(
        self,
        order: Order,
        market_data: dict,  # {open, high, low, close, adj_close, volume}
        cost_model: CostModel
    ) -> Fill:
        """
        Simulate filling an order.
        
        reference_price = adj_close
        fill_price = reference_price adjusted for spread + slippage
        commission = separate cash cost
        """
        ...
```

### 18.4 Transaction Cost Sensitivity Experiment

The research must include an experiment (Experiment 7) that:

1. Runs each signal's backtest across all five cost regimes
2. Reports:
   - Net CAGR at each cost level
   - Net Sharpe ratio at each cost level
   - Turnover
   - Average daily return (net)
   - Cost elasticity (change in Sharpe per unit change in cost)
3. Defines **economic break-even cost**: the cost level at which the strategy's net expected return (average daily return) becomes approximately zero
4. Also reports Sharpe-zero crossing as a supplementary diagnostic (the cost level at which net Sharpe ≤ 0)
5. Plots strategy Sharpe ratio vs. total one-way cost (bps)

**Expected insight**: Mean-reversion strategies (short-horizon, high turnover) are more cost-sensitive than momentum strategies (longer horizon, lower turnover).

---

## 19. Risk Management

### Scope

Risk management in this project is **analytical, not real-time**. The project does not implement dynamic risk limits or real-time position sizing. Instead, it computes and reports risk metrics to evaluate strategy quality.

### Risk Metrics

All risk metrics are computed in `src/evaluation/metrics.py`.

**Risk-free-rate convention**: The project sets the daily risk-free rate to zero for all reported Sharpe/Sortino calculations. This keeps the student implementation simple and avoids introducing an additional market-data dependency. This convention must be stated in the report and README.

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Annualized volatility | $\sigma_{ann} = \sigma_{daily} \times \sqrt{252}$ | Total portfolio risk |
| Maximum drawdown | $\text{MDD} = \max_{t}\left(\frac{\text{peak}_t - \text{equity}_t}{\text{peak}_t}\right)$ | Worst cumulative loss from a peak |
| Maximum drawdown duration | Number of days from peak to recovery | How long the drawdown lasted |
| Calmar ratio | $\text{CAGR} / |\text{MDD}|$ | Return per unit of tail risk |
| Sortino ratio | $\frac{\bar{r}}{\sigma_{downside}} \times \sqrt{252}$ under the project's zero-risk-free-rate convention, where $\sigma_{downside} = \text{std}(r \mid r < 0)$ | Return per unit of downside risk |
| Skewness | $\frac{1}{T}\sum\left(\frac{r_t - \bar{r}}{\sigma}\right)^3$ | Asymmetry of return distribution (negative = left tail) |
| Kurtosis (excess) | $\frac{1}{T}\sum\left(\frac{r_t - \bar{r}}{\sigma}\right)^4 - 3$ | Fat tails (> 0 = fatter than normal) |
| Gross exposure | $\sum_i |w_i|$ | Total capital deployed (long + short) |
| Net exposure | $\sum_i w_i$ | Directional bias (0 = market-neutral) |

### Position Limits (Configuration)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_position_weight` | 0.10 (10%) | No single stock > 10% of portfolio |
| `max_gross_exposure` | 1.0 (100%) | No leverage above 1x |
| `max_net_exposure` | 0.20 (20%) | Limit directional bias |

These are hard portfolio-construction constraints. The constructor must enforce them before returning target weights, and unit tests must verify that every weight vector satisfies them.

---

## 20. Walk-Forward Validation

### 20.1 Why Not k-Fold Cross-Validation?

Standard random k-fold CV is inappropriate for time-series data because:

1. **Temporal dependence**: Financial returns are serially correlated (volatility clustering, momentum). Random shuffling breaks this structure.
2. **Look-ahead bias**: Training on future data and testing on past data leaks future information.
3. **Non-stationarity**: The data-generating process changes over time. A model trained on 2014 data and tested on 2020 data faces a different regime.

Walk-forward validation respects temporal ordering and simulates the actual research/trading process: learn from the past, trade into the future.

### 20.2 Expanding Window Walk-Forward (Development Phase)

The following 7 walk-forward windows are used during the development/research phase. The last window's test period (2024) is reserved as the **final holdout** — it is not used during research iteration.

```
Window 1:
    Train:  2014-01-01 to 2016-12-31 (3 years)
    Val:    2017-01-01 to 2017-12-31 (1 year)
    Test:   2018-01-01 to 2018-12-31 (1 year)

Window 2:
    Train:  2014-01-01 to 2017-12-31 (4 years)
    Val:    2018-01-01 to 2018-12-31 (1 year)
    Test:   2019-01-01 to 2019-12-31 (1 year)

Window 3:
    Train:  2014-01-01 to 2018-12-31 (5 years)
    Val:    2019-01-01 to 2019-12-31 (1 year)
    Test:   2020-01-01 to 2020-12-31 (1 year)

Window 4:
    Train:  2014-01-01 to 2019-12-31 (6 years)
    Val:    2020-01-01 to 2020-12-31 (1 year)
    Test:   2021-01-01 to 2021-12-31 (1 year)

Window 5:
    Train:  2014-01-01 to 2020-12-31 (7 years)
    Val:    2021-01-01 to 2021-12-31 (1 year)
    Test:   2022-01-01 to 2022-12-31 (1 year)

Window 6:
    Train:  2014-01-01 to 2021-12-31 (8 years)
    Val:    2022-01-01 to 2022-12-31 (1 year)
    Test:   2023-01-01 to 2023-12-31 (1 year)

Window 7 (FINAL HOLDOUT):
    Train:  2014-01-01 to 2022-12-31 (9 years)
    Val:    2023-01-01 to 2023-12-31 (1 year)
    Test:   2024-01-01 to 2024-12-31 (1 year)  ← USED ONLY ONCE
```

**Expanding** (not rolling): Training set grows over time. This is appropriate because we want to use all available history, and the universe is small enough that more data helps.

### 20.3 Development vs. Final Evaluation

The project distinguishes two stages:

**Development / research process** (Windows 1–6): Used for exploring hypotheses, discovering promising configurations, designing methods. The researcher iteratively inspects aggregate OOS results from windows 1–6 and may modify the strategy.

**Final evaluation** (Window 7): Used only after methodology is frozen, signal definitions are fixed, and primary parameters are committed. The final holdout (2024) is evaluated exactly once and reported as the final result.

**Important caveat**: Even with this structure, the aggregate OOS results from windows 1–6 are "out-of-sample with respect to individual walk-forward windows, but potentially exploratory with respect to research-design iteration." Repeatedly inspecting OOS results and changing the strategy creates research-selection bias layered on top of the walk-forward design. The report must acknowledge this. Walk-forward validation does NOT automatically make every research decision unbiased.

### 20.4 How Parameters Are Selected

1. **On the training set**: Compute signal with the primary (confirmatory) configuration and any exploratory configurations
2. **On the validation set**: If exploring multiple configurations, select the one with the best IC or Sharpe on the validation set
3. **On the test set**: Evaluate performance with the selected parameters — **no further tuning**

**The test set is used exactly once per walk-forward window.**

### 20.5 Aggregating Walk-Forward Results

The out-of-sample test-set results from windows 1–6 (2018–2023) are concatenated to form a development-phase out-of-sample equity curve and IC series. Window 7 (2024) is reported separately as the final holdout evaluation.

Metrics computed on the concatenated development-phase OOS series are supporting/development evidence because the researcher may inspect those results while iterating on methodology. The 2024 final holdout is the **primary confirmatory out-of-sample result** and must be evaluated exactly once after the methodology is frozen. The report should show both, but should not pool the development-phase OOS results with the final holdout when making the primary confirmatory claim.

### 20.6 Implementation

```python
# src/validation/walk_forward.py

@dataclass
class WalkForwardWindow:
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime
    is_holdout: bool = False  # True for the final holdout window

def generate_walk_forward_windows(
    start_date: datetime,
    end_date: datetime,
    min_train_years: int = 3,
    val_years: int = 1,
    test_years: int = 1,
    step_years: int = 1,
    method: str = "expanding"  # "expanding" or "rolling"
) -> list[WalkForwardWindow]:
    """
    Generate walk-forward windows.
    The last window is marked as is_holdout=True.
    
    Returns:
        List of WalkForwardWindow objects.
    """
    ...

class WalkForwardRunner:
    """
    Orchestrates walk-forward evaluation.
    
    For each window:
        1. Train: compute signal on training data; select parameters on validation data
        2. Test: evaluate on test data with selected parameters
        3. Record test-set results
    
    After development windows:
        Concatenate test-set results → compute aggregate metrics
    
    Final holdout window:
        Evaluate with frozen methodology → report separately
    """
    
    def __init__(
        self,
        windows: list[WalkForwardWindow],
        strategy_factory: callable,
        param_grid: dict,
        backtest_engine_factory: callable
    ):
        ...
    
    def run(self) -> WalkForwardResult:
        ...
```

---

## 21. Robustness Testing

### 21.1 Philosophy

A result is "robust" if it does not collapse under reasonable perturbations. A result that depends on a specific parameter value, a specific time period, or a specific subset of assets is fragile and should not be relied upon.

**Holdout isolation rule**: All parameter perturbation, asset-subset, cost-sensitivity, extreme-day removal, and regime-conditional robustness experiments are performed using the development period/windows only. The 2024 final holdout is not used to choose robustness settings, interpretively select a strategy variant, or tune any parameter. After the methodology is frozen, the final holdout is evaluated once using the frozen specification only.

### 21.2 Robustness Tests

| Test | Method | What it reveals |
|------|--------|-----------------|
| **Parameter perturbation** | Run each signal with 3–5 different parameter values (e.g., lookback ∈ {10, 20, 40, 60}). Check if results are consistent. | Sensitivity to parameter choice |
| **Time-period stability** | Split the concatenated development OOS period (2018–2023) into halves. Evaluate metrics on each half. Do not use the 2024 holdout for this test. | Whether development performance is concentrated in one sub-period |
| **Asset subset stability** | Remove 10% of stocks at random (5 trials). Re-evaluate. | Whether performance depends on a few specific stocks |
| **Transaction-cost sensitivity** | Evaluate across 5 cost regimes. Plot Sharpe vs. cost. | Economic break-even cost level |
| **Removal of best 5 days** | Remove the 5 best return days from the equity curve and re-compute Sharpe. | Whether returns are concentrated in a few outlier days |
| **Removal of worst 5 days** | Remove the 5 worst return days. | Whether drawdowns are concentrated in a few events |
| **Regime conditioning** | Evaluate separately in high-vol and low-vol sub-periods. | Regime dependency |
| **Bootstrap PnL** | Block-bootstrap the daily return series; compute CI for Sharpe. | Statistical uncertainty of performance |

### 21.3 Implementation

```python
# src/robustness/tests.py

def parameter_sensitivity(
    signal_class: type,
    param_grid: dict[str, list],
    features: pd.DataFrame,
    forward_returns: pd.DataFrame
) -> pd.DataFrame:
    """
    Evaluate signal IC across parameter grid.
    
    Returns:
        DataFrame with columns: param_name, param_value, mean_ic, hac_t_stat, sharpe
    """
    ...

def time_period_stability(
    equity_curve: pd.Series,
    n_splits: int = 2
) -> list[PerformanceMetrics]:
    """Split equity curve into n sub-periods and compute metrics for each."""
    ...

def asset_subset_stability(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_trials: int = 5,
    drop_fraction: float = 0.10,
    seed: int = 42
) -> pd.DataFrame:
    """Randomly drop assets and re-evaluate. Returns IC distribution."""
    ...

def remove_extreme_days(
    equity_curve: pd.Series,
    n_days: int = 5,
    extreme: str = "best"  # "best" or "worst"
) -> PerformanceMetrics:
    """Remove N best/worst days and recompute metrics."""
    ...
```

---

## 22. Regime Analysis

### 22.1 Regime Definition (Simple, Interpretable)

Regimes are defined using observable, lagged market characteristics. No hidden-state models (HMM) in the core version.

| Regime | Definition | Intuition |
|--------|-----------|-----------|
| **Low volatility** | VIX proxy (rolling 20-day vol of SPY/index) < 25th percentile of its own 252-day rolling distribution | Calm, low-risk market |
| **High volatility** | VIX proxy > 75th percentile | Turbulent market |
| **Trending** | 60-day return of SPY/index > 0 | Market in an uptrend |
| **Non-trending** | 60-day return ≤ 0 | Market flat or declining |

Since the project uses free data and may not have VIX, the "VIX proxy" is the 20-day realized volatility of the equal-weighted universe portfolio or a benchmark ETF (SPY).

### 22.2 Regime Analysis Process (Descriptive)

Descriptive regime analysis can be performed freely without changing strategy rules:

1. Assign each trading day to a regime (or combination: e.g., "high-vol + trending")
2. For each signal, compute performance metrics separately within each regime
3. Compare: does the signal's IC / Sharpe differ significantly between regimes?
4. Visualize: conditional equity curves, IC by regime bar charts

### 22.3 Regime-Conditioned Trading (Optional Strategy Extension)

**Regime-conditioned trading** — e.g., "only trade momentum in trending regimes" — is a **new strategy**, not a free observation. It introduces an additional decision parameter (which regime is favorable).

If implemented:
- It must be treated as a new strategy variant.
- It must be evaluated using walk-forward validation (the regime rule is learned on training/validation data, not on the full sample).
- The regime rule becomes another research choice and is therefore part of the hypothesis-selection process.
- It must NOT be applied post-hoc as if it were always part of the methodology.

This prevents accidental post-hoc "regime optimization."

### 22.4 Implementation

```python
# Regime computation can be part of src/features/technical.py or a dedicated function

def classify_regime(
    benchmark_returns: pd.Series,
    vol_lookback: int = 20,
    trend_lookback: int = 60,
    vol_percentile_window: int = 252
) -> pd.DataFrame:
    """
    Classify each day into volatility and trend regimes.
    
    Returns:
        DataFrame with columns: date, vol_regime ('low', 'medium', 'high'),
        trend_regime ('trending', 'non_trending')
    """
    ...
```

---

## 23. ML Extension

### 23.1 Scope

ML is a **late-stage extension**, not the center of the project. The research question is:

> "Does ML provide incremental out-of-sample value over the pre-specified combined statistical baseline?"

### 23.2 Models to Compare

| Model | Type | Why included | Library |
|-------|------|-------------|---------|
| Linear regression (OLS) | Baseline | Simplest model; reveals if features are linearly predictive | `sklearn.linear_model.LinearRegression` |
| Ridge regression | Regularized linear | Handles collinear features; L2 regularization | `sklearn.linear_model.Ridge` |
| Lasso regression | Regularized linear | Feature selection via L1; identifies most important features | `sklearn.linear_model.Lasso` |
| XGBoost | Tree-based ensemble | Captures nonlinear interactions; industry-standard | `xgboost.XGBRegressor` |

**Choose ONE of XGBoost or LightGBM for the core project, not both.** Recommendation: **XGBoost**, because the candidate lists it on their resume.

### 23.3 ML Baseline Definition

**The ML baseline (benchmark) is the pre-specified equal-weight combined statistical signal** (see Section 13, Signal Combination). This is defined before looking at any results.

Do NOT define the benchmark as "the best statistical signal found after looking at the results." That would create winner-selection bias — the benchmark would be chosen to look good, making it harder (or easier) for ML to beat, depending on which direction the selection went.

### 23.4 ML Pipeline

```
For each walk-forward window:
    1. Training set:
       Features: all features from Section 12, properly lagged
       Target: forward 20-day return (the fixed common target for the ML comparison)
       
    2. Feature lagging rule:
       All features use data available at close of day t
       Target return = adj_close[t+20] / adj_close[t] - 1
       THIS MUST BE STRICTLY ENFORCED TO PREVENT LEAKAGE
       
    3. Preprocessing:
       StandardScaler fit ONLY on training set
       Transform applied to validation and test sets
       
    4. Train model on training set
    5. Select hyperparameters on validation set
    6. Generate predictions on test set
    7. Convert predictions to signals → portfolio weights → backtest
    8. Compare to the pre-specified combined statistical baseline
```

### 23.5 Hyperparameter Tuning

| Model | Hyperparameters to tune | Range |
|-------|------------------------|-------|
| Ridge | alpha | {0.01, 0.1, 1.0, 10.0, 100.0} |
| Lasso | alpha | {0.0001, 0.001, 0.01, 0.1, 1.0} |
| XGBoost | max_depth | {3, 5, 7} |
| | n_estimators | {50, 100, 200} |
| | learning_rate | {0.01, 0.05, 0.1} |
| | subsample | {0.7, 0.8, 1.0} |

**Tuning method**: Grid search on validation set IC or validation set Sharpe. NOT on test set.

### 23.6 ML-Specific Leakage Prevention

| Risk | Mitigation |
|------|------------|
| Target leakage (using future return as a feature) | Feature matrix does not include any column computed from future data |
| Feature leakage (using close[t+1] in a feature for day t) | All features explicitly lag by at least 0 days (use data up to close[t] only) |
| Cross-validation leakage | Only walk-forward temporal splits; NO random k-fold |
| Scaler leakage | StandardScaler fit on training set only; transform applied to val/test |
| Universe leakage | Same universe for train, val, test |

### 23.7 ML Evaluation

| Metric | Purpose |
|--------|---------|
| IC (ML predictions vs. forward returns) | Direct comparability with statistical signals |
| Rolling IC | Time variation in ML predictive power |
| Walk-forward Sharpe | Trading performance |
| Feature importance (XGBoost) | Which features drive predictions |
| Comparison: ML Sharpe vs. combined-signal Sharpe | The key research question |
| HAC test on daily IC difference: ML vs. combined baseline | Is the difference statistically significant? |

**Significance test for ML improvement**: Use HAC/Newey-West standard errors on the daily IC difference series (ML IC − baseline IC), NOT an ordinary paired t-test. Daily IC differences are serially correlated (especially with overlapping forward returns), so an ordinary t-test would overstate significance.

### 23.8 Expected Outcome

The project should **not assume** ML will outperform. In practice, for daily equity data with standard features:
- Linear models often capture most of the predictive content
- XGBoost may provide marginal improvement, zero improvement, or even worse performance (overfitting)
- The value of ML is more likely at higher frequencies or with richer feature sets

An honest conclusion like "XGBoost provided marginal improvement in IC but the improvement was not statistically significant after HAC adjustment" is a perfectly valid and interesting result.

---

## 24. Experiment Matrix

### Experiment 1: Data Quality and Market Characteristics

| Property | Value |
|----------|-------|
| **Hypothesis** | The dataset is clean, complete, and representative of the intended universe |
| **Independent variables** | Data source, universe definition |
| **Dependent variables** | Missing data rate, price anomalies, return distribution statistics |
| **Dataset** | Full dataset (2014–2024) |
| **Methodology** | Compute summary statistics; check for anomalies; visualize distributions; verify start-of-sample universe membership; confirm no forward-filled bars |
| **Control/baseline** | Expected properties of liquid US equities |
| **Metrics** | Missing data %, autocorrelation of returns, stationarity tests, return distribution moments |
| **Statistical tests** | ADF on return series; Jarque-Bera for normality; Ljung-Box for autocorrelation |
| **Expected plots** | Return distribution histogram; QQ-plot; missing data heatmap; autocorrelation plot |
| **Interpretation** | Identify any data quality issues before proceeding. Establish baseline market characteristics. Confirm that the universe was defined as of the start of the sample period. |

### Experiment 2: Momentum Signal

| Property | Value |
|----------|-------|
| **Hypothesis** | Cross-sectional momentum (12-1 month return) shows predictive association with next-month cross-sectional returns |
| **Hypothesis type** | Confirmatory (primary config); exploratory (parameter sweep) |
| **Primary config** | lookback=252, skip=21, forward=20d |
| **Exploratory configs** | Lookback {126, 189, 252}; skip {0, 5, 10, 21} |
| **Dependent variables** | IC, HAC t-stat, quintile spread return |
| **Dataset** | Full in-sample period (2014–2019) |
| **Methodology** | Compute momentum signal; compute daily cross-sectional IC vs. forward 20-day return |
| **Control/baseline** | Random signal (IC ≈ 0) |
| **Metrics** | Mean IC, naive t-stat (diagnostic), HAC t-stat (primary), ICIR, quintile spread, daily hit rate |
| **Statistical tests** | HAC t-test on mean IC (primary); permutation test; bootstrap CI |
| **Expected plots** | Rolling IC time series; quintile return bar chart; IC distribution histogram |
| **Interpretation** | Does the evidence suggest momentum IC is significantly positive (HAC)? How stable is it over time? Exploratory results clearly labeled. |

### Experiment 3: Mean-Reversion Signal

| Property | Value |
|----------|-------|
| **Hypothesis** | Short-term price deviation (20-day price z-score) shows predictive association with next-week return reversal |
| **Hypothesis type** | Confirmatory (primary config); exploratory (parameter sweep) |
| **Primary config** | lookback=20, forward=5d, threshold=None |
| **Exploratory configs** | Lookback {10, 20, 40, 60}; forward {1, 5, 10, 20}; threshold {None, 1.5, 2.0} |
| **Dependent variables** | IC, HAC t-stat |
| **Dataset** | In-sample period |
| **Methodology** | Compute dimensionally correct price z-score signal (negated: low z → long); compute IC vs. forward return |
| **Control/baseline** | Random signal |
| **Metrics** | Mean IC, HAC t-stat, ICIR |
| **Statistical tests** | HAC t-test on IC; permutation test |
| **Expected plots** | Rolling IC; quintile returns; heatmap of IC across lookback × forward horizon (exploratory) |
| **Interpretation** | Does the evidence suggest mean reversion at short horizons? Primary result from primary config; heatmap shows exploration. |

### Experiment 4: Volatility Signal

| Property | Value |
|----------|-------|
| **Hypothesis** | Low realized volatility shows predictive association with higher future returns (low-vol anomaly) |
| **Hypothesis type** | Confirmatory (primary config) |
| **Primary config** | lookback=60, forward=20d |
| **Primary target** | Forward 20-day cumulative return (NOT "risk-adjusted return") |
| **Dependent variables** | IC vs. forward 20-day return; quintile returns |
| **Dataset** | In-sample period |
| **Methodology** | Rank stocks by volatility (low vol → high signal); compute IC |
| **Control/baseline** | Random signal |
| **Metrics** | Mean IC, HAC t-stat, quintile returns, sector composition of quintiles |
| **Statistical tests** | HAC t-test on IC |
| **Expected plots** | Quintile return bar chart; sector breakdown per quintile |
| **Interpretation** | Does the evidence suggest the low-vol anomaly is present in this universe and period? Is sector concentration a confound? |

### Experiment 5: Abnormal Volume Signal

| Property | Value |
|----------|-------|
| **Hypothesis** | Abnormally high volume combined with return direction shows predictive association with short-term continuation |
| **Hypothesis type** | Confirmatory (primary config) |
| **Primary config** | volume_lookback=20, threshold=None, direction=1d, forward=5d |
| **Dependent variables** | IC vs. forward 5-day return |
| **Dataset** | In-sample period |
| **Methodology** | Compute volume z-score × sign(return); compute IC |
| **Control/baseline** | Volume signal without directional component |
| **Metrics** | Mean IC, HAC t-stat, conditional IC (high-volume days only vs. all days) |
| **Statistical tests** | HAC t-test on IC; comparison of conditional vs. unconditional IC |
| **Expected plots** | IC by volume tercile; scatter of volume anomaly vs. forward return |
| **Interpretation** | Does volume add information beyond price? Is the directional component important? |

### Experiment 6: Signal Comparison

| Property | Value |
|----------|-------|
| **Hypothesis** | Signals show predictive association at their pre-specified horizons, and the pre-specified combined signal may add information for the common 20-day ML evaluation target |

| **Independent variables** | Signal identity; combination method |
| **Dependent variables** | Within-horizon IC comparisons, Sharpe, turnover, cost-adjusted Sharpe |

| **Dataset** | In-sample period |
| **Methodology** | Compare momentum vs. low-volatility at the common 20-day horizon; compare mean-reversion vs. volume at the common 5-day horizon. Do NOT directly compare raw IC magnitudes across different horizons. Evaluate the pre-specified combined signal against 20-day forward returns for the ML baseline. |

| **Control/baseline** | Each signal individually |
| **Metrics** | IC, ICIR, Sharpe (zero-cost), Sharpe (medium-cost), correlation matrix between signals |
| **Statistical tests** | HAC test on daily IC differences only within a common forward-return horizon; signal correlation; no cross-horizon IC significance test |

| **Expected plots** | IC comparison bar chart; signal correlation heatmap; equity curves overlaid |
| **Interpretation** | Which signals show the strongest evidence within their own forecast horizon? Are signals complementary? Does the common 20-day combined baseline add value? |


### Experiment 7: Transaction-Cost Sensitivity

| Property | Value |
|----------|-------|
| **Hypothesis** | Strategy performance degrades with increasing transaction costs; high-turnover strategies are more affected |
| **Independent variables** | Cost regime (zero, low, medium, high, very high) |
| **Dependent variables** | Net Sharpe, net CAGR, net average daily return, turnover |
| **Dataset** | In-sample period |
| **Methodology** | Run event-driven backtest for each signal at each cost level using single-pass cost accounting |
| **Control/baseline** | Zero-cost benchmark |
| **Metrics** | Net Sharpe at each cost level; economic break-even cost (net expected return ≈ 0); Sharpe-zero crossing (supplementary); turnover; cost elasticity |
| **Statistical tests** | None (descriptive analysis) |
| **Expected plots** | Sharpe vs. cost (line chart per signal); turnover bar chart per signal |
| **Interpretation** | Which signals survive realistic costs? Is high turnover the primary driver of cost sensitivity? |

### Experiment 8: Walk-Forward Evaluation

| Property | Value |
|----------|-------|
| **Hypothesis** | In-sample evidence of predictive association generalizes to out-of-sample data |
| **Independent variables** | Walk-forward window; signal identity |
| **Dependent variables** | Out-of-sample IC, Sharpe |
| **Dataset** | Development period (2014–2023) for selection/robustness, with 2024 reserved as untouched holdout |
| **Methodology** | Run walk-forward with expanding windows; concatenate OOS results from development windows (1–6); evaluate final holdout (window 7) separately |
| **Control/baseline** | In-sample performance (expect degradation) |
| **Metrics** | OOS IC per window; OOS Sharpe; IS vs. OOS degradation ratio |
| **Statistical tests** | HAC t-test on OOS IC; comparison of IS vs. OOS Sharpe |
| **Expected plots** | OOS equity curve; IS vs. OOS IC per window; degradation chart |
| **Interpretation** | How much performance degrades out-of-sample? Which signals are most robust? Does the final holdout confirm development-phase findings? |

### Experiment 9: Multiple-Testing Correction

| Property | Value |
|----------|-------|
| **Hypothesis** | Testing many signals/parameters without correction leads to false discoveries |
| **Independent variables** | Number of hypotheses tested; correction method |
| **Dependent variables** | Number of "significant" signals before/after correction |
| **Dataset** | In-sample period |
| **Methodology** | (a) Apply correction to the 4 confirmatory signal tests (HAC p-values). (b) Apply correction to all exploratory tests when they are summarized inferentially. (c) Generate 100 random signals, compute HAC p-values, demonstrate the expected false-positive rate across repeated null simulations, then apply correction and show the reduction in false discoveries. (d) Log total number of configurations explored in the hypothesis registry. |
| **Control/baseline** | 100 random signals (null distribution) |
| **Metrics** | Number of discoveries before/after Bonferroni; before/after BH-FDR; total configurations explored |
| **Statistical tests** | Bonferroni, BH-FDR |
| **Expected plots** | p-value histogram (real signals + random); significance threshold visualization |
| **Interpretation** | How many confirmatory signals survive correction? Does the random-signal experiment work as expected? How many total configurations were explored? |

### Experiment 10: Robustness Testing

| Property | Value |
|----------|-------|
| **Hypothesis** | Results are robust to parameter perturbation, time-period changes, and asset subset changes |
| **Independent variables** | Perturbation type (parameter, time, assets) |
| **Dependent variables** | Sharpe, IC stability |
| **Dataset** | Development period/windows only; the 2024 final holdout is excluded from robustness selection |
| **Methodology** | Run robustness suite from Section 21 on development data only |
| **Control/baseline** | Baseline (primary parameters, development period, full start-of-sample universe) |
| **Metrics** | Range of Sharpe across perturbations; coefficient of variation of IC |
| **Statistical tests** | Bootstrap CI for Sharpe |
| **Expected plots** | Sensitivity tornado chart; sub-period performance bars; asset-drop stability |
| **Interpretation** | Are the results fragile or robust? What is the main source of instability? |

### Experiment 11: Regime Analysis

| Property | Value |
|----------|-------|
| **Hypothesis** | Signal predictive association varies across market regimes |
| **Independent variables** | Regime (low-vol, high-vol, trending, non-trending) |
| **Dependent variables** | IC, Sharpe per regime |
| **Dataset** | Development period/windows only; 2024 final holdout remains untouched for regime selection |
| **Methodology** | Assign days to regimes using only information available by each date; compute metrics conditionally on development data (descriptive analysis). |
| **Control/baseline** | Unconditional (all-regime) performance |
| **Metrics** | IC per regime; Sharpe per regime; regime frequency |
| **Statistical tests** | HAC t-test for IC difference between regimes |
| **Expected plots** | Bar charts of IC/Sharpe by regime; regime timeline overlay on equity curve |
| **Interpretation** | Which signals show stronger/weaker evidence in which regimes? NOTE: Regime-conditioned trading (if attempted) is a separate strategy extension that must be walk-forward validated. |

### Experiment 12: Statistical Model vs. XGBoost

| Property | Value |
|----------|-------|
| **Hypothesis** | XGBoost captures nonlinear feature interactions that improve out-of-sample prediction |
| **Independent variables** | Model type (linear, Ridge, Lasso, XGBoost) |
| **Dependent variables** | OOS IC, OOS Sharpe |
| **Dataset** | Development period plus separate 2024 final holdout; same walk-forward structure for all models |
| **Methodology** | Walk-forward evaluation for each model using the fixed 20-day target; identical features and target. Development results may be inspected during research iteration; the 2024 holdout is evaluated once after methodology freeze. |
| **Control/baseline** | Pre-specified equal-weight combined statistical signal (NOT the best individual signal selected after seeing results) |
| **Metrics** | OOS IC, OOS Sharpe, feature importance (XGBoost) |
| **Statistical tests** | HAC test on daily IC difference series (ML − baseline), NOT ordinary paired t-test |
| **Expected plots** | OOS IC comparison bar chart; feature importance chart; OOS equity curves overlaid |
| **Interpretation** | Does ML add value? Which features matter? Is the improvement (if any) statistically significant after accounting for serial correlation? |

---

## 25. Notebook Specifications

### `01_data_audit.ipynb`

| Property | Value |
|----------|-------|
| **Purpose** | Load, inspect, and validate raw market data. Establish data quality baseline. Verify start-of-sample universe and missing-data policy. |
| **Sections** | 1. Data loading & shape inspection 2. Universe verification (start-of-sample) 3. Missing data analysis (confirm no forward-filled bars) 4. Price validation (low ≤ close ≤ high) 5. Return distribution analysis 6. Stationarity tests (ADF on a few representative tickers) 7. Autocorrelation analysis 8. Summary statistics table 9. Terminated securities identification 10. Data quality conclusion |
| **Inputs** | Raw OHLCV Parquet files, universe definition |
| **Outputs** | Cleaned Parquet files; data quality summary JSON |
| **Key plots** | Missing data heatmap; return histogram + QQ-plot; autocorrelation function; price chart for a few tickers |
| **Key tables** | Summary stats per ticker (mean return, vol, missing %); universe composition; terminated securities |
| **Conclusions** | "The dataset contains N tickers over M years. Missing observations are preserved as NaN (no forward-fill). K securities terminated during the sample period. Returns are [approximately / not] normally distributed with [positive / negative] excess kurtosis..." |

### `02_feature_exploration.ipynb`

| Property | Value |
|----------|-------|
| **Purpose** | Compute and explore all features. Verify dimensional consistency (price std vs. return std). Check distributions, stationarity, and cross-correlations. |
| **Sections** | 1. Feature computation 2. Verify rolling_std_price vs. rolling_std_ret distinction 3. Feature distributions (histograms) 4. Feature stationarity check 5. Cross-correlation matrix 6. Time-series plots of selected features 7. Cross-sectional feature distributions 8. NaN propagation verification |
| **Inputs** | Cleaned market data |
| **Outputs** | Feature DataFrame (Parquet) |
| **Key plots** | Feature correlation heatmap; feature distribution histograms (grid); time-series of rolling vol for a few tickers |
| **Key tables** | Feature summary stats; stationarity test results |
| **Conclusions** | "Features X, Y, Z are approximately stationary. The z-score uses rolling_std_price_20 (denominator in USD) and is dimensionally consistent. Features computed on days with missing bars are correctly NaN..." |

### `03_signal_research.ipynb`

| Property | Value |
|----------|-------|
| **Purpose** | Construct and visualize all four core signals using development-period data only. Preliminary IC analysis with HAC inference; final holdout is not inspected here. |
| **Sections** | 1. Momentum signal construction + IC + HAC t-stat 2. Mean-reversion signal construction + IC + HAC t-stat 3. Volatility signal construction + IC + HAC t-stat 4. Volume signal construction + IC + HAC t-stat 5. Signal correlation analysis 6. Combined signal |
| **Inputs** | Feature DataFrame |
| **Outputs** | Signal DataFrame (Parquet) |
| **Key plots** | Signal distribution per day; rolling IC per signal; signal correlation heatmap |
| **Key tables** | Mean IC, naive t-stat (diagnostic), HAC t-stat (primary), ICIR per signal |
| **Conclusions** | "Preliminary analysis suggests evidence of predictive association for signals X and Y (HAC t-stat > 2). Signal Z appears weak..." |

### `04_statistical_testing.ipynb`

| Property | Value |
|----------|-------|
| **Purpose** | Rigorous statistical evaluation of signal predictive power using development/in-sample data only. The 2024 holdout is isolated from this notebook. |
| **Sections** | 1. IC analysis (full detail: naive + HAC + bootstrap) 2. Overlapping-return dependence discussion 3. Within-date permutation test 4. Quintile spread analysis (with edge-case handling) 5. Multiple-testing correction (confirmatory family) 6. Exploratory results log 7. Random-signal demonstration 8. Hypothesis registry summary |
| **Inputs** | Signal DataFrame, forward returns |
| **Outputs** | Statistical test results (JSON); hypothesis registry |
| **Key plots** | IC distribution; permutation null distribution with observed IC; quintile return bar charts; p-value histogram (real + random) |
| **Key tables** | IC summary per signal (naive t, HAC t, bootstrap CI); p-values before/after correction; quintile returns; total configurations explored |
| **Conclusions** | "After Bonferroni correction of the 4 confirmatory hypotheses, N remain significant at α=0.05 (using HAC p-values). The random-signal experiment demonstrates the expected stochastic false-positive rate under the null (approximately 5% on average before correction at α=0.05, with variability across runs). A total of M configurations were explored; exploratory results are labeled as such." |

### `05_backtesting.ipynb`

| Property | Value |
|----------|-------|
| **Purpose** | Run event-driven backtests for all signals and portfolio construction methods. Verify single-pass cost accounting. |
| **Sections** | 1. Backtest setup (config, cost model) 2. Individual signal backtests 3. Combined signal backtest 4. Portfolio construction comparison (equal-wt, signal-wt, vol-scaled) 5. Transaction cost impact 6. Exposure tracking (gross ≤ 1.0, net ≈ 0) |
| **Inputs** | Signal DataFrame, cleaned market data, config |
| **Outputs** | Equity curves, trade logs, performance metrics (Parquet/JSON) |
| **Key plots** | Equity curves (overlaid); drawdown chart; rolling Sharpe; turnover over time; exposure over time |
| **Key tables** | Performance metrics comparison table (all signals × portfolio methods) |
| **Conclusions** | "The signal with the strongest evidence of predictive association achieves a Sharpe of [X] before costs and [Y] after costs (medium regime, 15 bps one-way)..." |

### `06_walk_forward.ipynb`

| Property | Value |
|----------|-------|
| **Purpose** | Walk-forward out-of-sample evaluation with development windows and a separately executed final holdout. No results from the final holdout feed back into research choices. |
| **Sections** | 1. Walk-forward window definition 2. Per-window results (development phase, windows 1–6) 3. Concatenated development-phase OOS equity curve 4. IS vs. OOS comparison 5. OOS statistical significance (HAC) 6. Final holdout evaluation (window 7, 2024) 7. Research-iteration bias caveat |
| **Inputs** | Features, signals, market data, config |
| **Outputs** | Walk-forward results (JSON/Parquet) |
| **Key plots** | OOS equity curve; IS vs. OOS IC bar chart per window; degradation chart; holdout equity curve |
| **Key tables** | Per-window metrics; aggregate OOS metrics; holdout metrics |
| **Conclusions** | "Expanding-window walk-forward evaluation over the 2018–2023 development OOS period produces supporting evidence with a concatenated Sharpe of [X], representing a [Y]% degradation from in-sample. The final holdout (2024), evaluated once after methodology freeze, shows [Z] and is treated as the primary confirmatory OOS result. Note: development OOS results may reflect research-design iteration." |

### `07_robustness.ipynb`

| Property | Value |
|----------|-------|
| **Purpose** | Robustness testing across perturbations using development data/windows only; the final holdout remains untouched. |
| **Sections** | 1. Parameter perturbation (primary vs. exploratory clearly labeled) 2. Time-period stability 3. Asset-subset stability 4. Transaction-cost sensitivity (economic break-even, Sharpe-zero crossing) 5. Extreme day removal 6. Bootstrap PnL CI |
| **Inputs** | Signals, market data, config |
| **Outputs** | Robustness results (JSON) |
| **Key plots** | Parameter sensitivity heatmap; sub-period bar charts; cost sensitivity line chart; bootstrap Sharpe histogram |
| **Key tables** | Sharpe across perturbations; economic break-even costs; Sharpe-zero crossing costs |
| **Conclusions** | "Signal [X] is robust to parameter changes (Sharpe range: [A]–[B]). Signal [Y] is fragile..." |

### `08_regime_analysis.ipynb`

| Property | Value |
|----------|-------|
| **Purpose** | Analyze signal performance across market regimes (descriptive). |
| **Sections** | 1. Regime classification 2. Regime timeline 3. Per-regime signal metrics 4. Conditional equity curves 5. Statistical test of regime dependence 6. Note: regime-conditioned trading is a separate strategy (if attempted) |
| **Inputs** | Signals, market data, benchmark returns |
| **Outputs** | Regime analysis results (JSON) |
| **Key plots** | Regime timeline; IC by regime bar chart; conditional equity curves |
| **Key tables** | Metrics per signal per regime |
| **Conclusions** | "The evidence suggests that momentum shows stronger predictive association in trending regimes (IC = [X]) than non-trending (IC = [Y]). NOTE: regime-conditioned trading, if implemented, must be walk-forward validated as a separate strategy." |

### `09_ml_comparison.ipynb`

| Property | Value |
|----------|-------|
| **Purpose** | Compare pre-specified combined statistical baseline against ML models. |
| **Sections** | 1. ML pipeline setup 2. Feature preparation (lagging, scaling — scaler on train only) 3. Walk-forward training + evaluation 4. Model comparison (IC, Sharpe) 5. Feature importance 6. Statistical significance of ML improvement (HAC on IC difference series) |
| **Inputs** | Features, forward returns |
| **Outputs** | ML comparison results (JSON/Parquet) |
| **Key plots** | OOS IC comparison (combined stat baseline vs. ML); feature importance chart; OOS equity curves |
| **Key tables** | Model comparison (IC, Sharpe, cost-adjusted Sharpe); HAC t-stat on IC difference |
| **Conclusions** | "XGBoost provides [marginal / no / significant] improvement over the pre-specified combined statistical baseline. The HAC t-stat on the IC difference is [X] (p = [Y])..." |

### `10_final_results.ipynb`

| Property | Value |
|----------|-------|
| **Purpose** | Compile and present final research results. |
| **Sections** | 1. Executive summary 2. Signal identification (primary configs, confirmatory results) 3. Final OOS and holdout performance 4. Robustness summary 5. Limitations (including survivorship bias, research-iteration bias, adjusted-price convention, missing-data handling) 6. Key findings 7. Research conclusions |
| **Inputs** | All previous experiment outputs |
| **Outputs** | Final results summary (JSON); key plots for report |
| **Key plots** | Summary dashboard; final equity curve; key comparison charts |
| **Key tables** | Master comparison table (all signals × all metrics) |
| **Conclusions** | Complete research conclusion answering the central research question with appropriately cautious language. |

---

## 26. Python Module Specifications

### 26.1 `src/data/loader.py`

| Property | Value |
|----------|-------|
| **Purpose** | Download and load OHLCV data from Yahoo Finance |
| **Dependencies** | `yfinance`, `pandas`, `pathlib` |

```python
def download_ohlcv(
    tickers: list[str],
    start_date: str,           # "YYYY-MM-DD"
    end_date: str,             # "YYYY-MM-DD"
    output_dir: str | Path
) -> pd.DataFrame:
    """
    Download daily OHLCV data for a list of tickers.
    
    Returns:
        DataFrame with MultiIndex (date, ticker) and columns:
        [open, high, low, close, adj_close, volume, volume_split_adjusted]
    
    Side effects:
        Saves raw data to output_dir as Parquet.
    
    Raises:
        DataDownloadError: If download fails for > 20% of tickers.
    """

def load_ohlcv(data_path: str | Path) -> pd.DataFrame:
    """
    Load previously saved OHLCV data from Parquet.
    
    Returns:
        DataFrame with MultiIndex (date, ticker).
    """
```

### 26.2 `src/data/cleaning.py`

| Property | Value |
|----------|-------|
| **Purpose** | Validate and clean raw OHLCV data (no forward-fill of OHLC bars) |
| **Dependencies** | `pandas`, `numpy`, `logging` |

```python
@dataclass
class DataQualityReport:
    total_rows: int
    missing_pct_by_ticker: dict[str, float]
    terminated_tickers: dict[str, str]  # ticker → last valid date
    anomalies: list[dict]  # {ticker, date, issue}


def validate_ohlcv(df: pd.DataFrame) -> DataQualityReport:
    """
    Check for data anomalies without making universe-membership decisions:
    - Negative prices
    - low > high
    - Zero or negative adj_close
    - Missing dates on trading days
    - Extreme returns (> 50% daily) flagged for review, not silently removed

    Returns:
        DataQualityReport
    """


def clean_ohlcv(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Clean raw data without using future information to filter securities:
    1. Remove exchange non-trading days (based on NYSE calendar)
    2. Do NOT forward-fill missing OHLC bars — preserve NaN
    3. Preserve every security that belongs to the frozen start-of-sample universe
    4. Document last valid observation for terminated securities
    5. Reconstruct split-adjusted volume when reliable split factors are available;
       otherwise flag split-affected dates for exclusion from volume features
    6. Log all operations

    Returns:
        (cleaned DataFrame, DataQualityReport)
    """
```
### 26.3 `src/data/universe.py`

```python
def load_universe(
    universe_name: str,        # e.g., "sp100_20140101"
    universe_dir: str | Path
) -> list[str]:
    """Load the historically documented start-of-sample universe definition."""
```

**Important**: This module loads and validates the frozen membership file. It must not remove securities because of missingness, future delisting, or future liquidity information. Those issues affect point-in-time signal eligibility, not historical membership.
### 26.4 `src/features/returns.py`

```python
def simple_return(prices: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """
    Compute simple return: (P_t - P_{t-period}) / P_{t-period}
    
    Args:
        prices: DataFrame with (date, ticker) MultiIndex, 'adj_close' column
        period: Number of days
    
    Returns:
        DataFrame with same index, column f'ret_{period}d'
    
    If any price in the computation is NaN (missing bar), the return is NaN.
    """

def log_return(prices: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """Compute log return: ln(P_t / P_{t-period})"""

def skip_return(
    prices: pd.DataFrame,
    total_period: int = 252,
    skip_period: int = 21
) -> pd.DataFrame:
    """
    Compute return from t-total_period to t-skip_period.
    
    Classic momentum: skip_return(252, 21) = 12-month return skipping last month.
    """

def forward_return(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Compute forward return: adj_close[t+horizon] / adj_close[t] - 1
    
    Used for IC computation and signal evaluation.
    Same formula used everywhere in the project.
    """
```

### 26.5 `src/features/technical.py`

```python
def rolling_mean(
    series: pd.DataFrame,
    window: int,
    column: str = "adj_close"
) -> pd.DataFrame:
    """Rolling simple moving average, computed per ticker."""

def rolling_std(
    series: pd.DataFrame,
    window: int,
    column: str = "ret_1d"
) -> pd.DataFrame:
    """
    Rolling standard deviation, computed per ticker.
    
    IMPORTANT: The caller specifies the column.
    - column="ret_1d" → std of daily returns (for realized volatility)
    - column="adj_close" → std of prices (for z-score denominator)
    These are different quantities. See Section 12.3.
    """

def zscore_price(
    prices: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """
    Price-based z-score: (P_t - SMA_window(P)) / rolling_std_price_window(P)
    
    Both numerator and denominator are in price units.
    The result is a dimensionless z-score.
    
    Computed per ticker.
    """

def realized_volatility(
    returns: pd.DataFrame,
    window: int = 20,
    annualize: bool = True
) -> pd.DataFrame:
    """
    Realized volatility: rolling std of RETURNS, optionally annualized (× √252).
    Uses rolling_std of returns (not prices).
    """
```

### 26.6 `src/features/volume.py`

```python
def relative_volume(
    volume: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """volume / rolling_mean(volume, window). NaN if volume is NaN."""

def volume_zscore(
    volume: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """(volume - rolling_mean) / rolling_std. NaN if volume is NaN."""
```

### 26.7–26.30

(Remaining module specifications are identical to v1.0, with the following changes applied where relevant: HAC t-stat added to `ic_summary()`, within-date permutation in `permutation_test_ic()`, `DataConfig.universe` defaults to `"sp100_20140101"`, `CostModel.compute_fill()` replaces `compute_cost()` with single-pass semantics, and `PortfolioConfig.gross_exposure` defaults to `1.0`. All other module specs from v1.0 are preserved.)

---

## 27. Configuration System

### YAML Configuration Files

Each YAML file defines one experiment configuration. All parameters that affect results must be in the config file (not hard-coded).

### Example: `configs/momentum.yaml`

```yaml
experiment_id: "momentum_baseline_v1"
random_seed: 42

data:
  universe: "sp100_20140101"
  start_date: "2014-01-01"
  end_date: "2024-12-31"
  data_source: "yfinance"

signal:
  name: "momentum_12_1"
  params:
    lookback: 252
    skip: 21
  forward_horizon: 20
  hypothesis_type: "confirmatory"

costs:
  commission_bps: 5.0
  spread_bps: 5.0
  slippage_bps: 5.0
  min_commission: 0.0

portfolio:
  method: "vol_scaled"
  gross_exposure: 1.0
  max_position_weight: 0.10
  max_net_exposure: 0.20

backtest:
  initial_cash: 1000000.0
  execution_price: "next_adj_close"

walk_forward:
  min_train_years: 3
  val_years: 1
  test_years: 1
  step_years: 1
  method: "expanding"
  final_holdout: true

output_dir: "results/experiments/momentum_baseline_v1"
```

---

## 28. Testing Strategy

### 28.1 Test Organization

```
tests/
├── test_returns.py             # Return calculation tests
├── test_features.py            # Feature computation tests (including dimensional checks)
├── test_signals.py             # Signal computation tests
├── test_statistics.py          # Statistical test correctness (HAC, permutation, bootstrap)
├── test_portfolio.py           # Portfolio weight computation
├── test_costs.py               # Transaction cost model (single-pass verification)
├── test_backtest.py            # Backtest engine
├── test_metrics.py             # Performance metrics
├── test_walk_forward.py        # Walk-forward window generation
├── test_no_lookahead.py        # Dedicated look-ahead bias tests
└── test_data_quality.py        # Missing-data, universe, timing integrity tests
```

### 28.2 Key Unit Tests

#### Return Calculations (`test_returns.py`)

```python
def test_simple_return_basic():
    """
    Prices: [100, 110, 105]
    Expected 1-day returns: [NaN, 0.10, -0.04545...]
    """

def test_log_return_basic():
    """
    Prices: [100, 110, 105]
    Expected 1-day log returns: [NaN, ln(1.1), ln(105/110)]
    """

def test_skip_return():
    """
    Known 10-day sequence. 
    skip_return(total=5, skip=2) should use prices from t-5 to t-2.
    """

def test_return_nan_propagation():
    """If a price is NaN (missing bar), the return should be NaN (not 0 or interpolated)."""

def test_forward_return_alignment():
    """
    Verify forward return for horizon h:
    fwd_ret[t] = adj_close[t+h] / adj_close[t] - 1
    
    Hand-calculated for a 5-day sequence.
    """
```

#### Feature Calculations (`test_features.py`)

```python
def test_rolling_mean():
    """
    Prices: [10, 12, 14, 16, 18]
    rolling_mean(window=3): [NaN, NaN, 12.0, 14.0, 16.0]
    """

def test_zscore_price_known_values():
    """
    Hand-calculated z-score using rolling_std of PRICES (not returns).
    Verify denominator is in price units.
    
    Prices: [100, 102, 98, 104, 100]
    SMA_3: [NaN, NaN, 100.0, 101.33, 100.67]
    std_3 of prices: [NaN, NaN, 2.0, 3.06, 3.06]
    zscore[4] = (100 - 100.67) / 3.06 ≈ -0.218
    """

def test_rolling_std_price_vs_return_different():
    """
    rolling_std(column='adj_close') produces different values from 
    rolling_std(column='ret_1d'). They are fundamentally different quantities.
    """

def test_realized_vol_annualization():
    """
    If daily std = 0.01, annualized vol should be 0.01 × √252 ≈ 0.1587
    """

def test_relative_volume():
    """
    Volume: [100, 100, 100, 200]
    With window=3: relative_volume[3] = 200 / 100 = 2.0
    """

def test_split_adjusted_volume():
    """
    A synthetic 2-for-1 split should not create a false volume spike when
    split-adjusted volume is used. If split adjustment is unavailable, the
    split-affected date must be flagged/excluded from volume features.
    """

def test_feature_nan_on_missing_bar():
    """
    If adj_close is NaN on day t (missing bar), all features derived
    from that price should be NaN on day t. No imputation.
    """
```

#### Signal Tests (`test_signals.py`)

```python
def test_momentum_ranking():
    """
    3 stocks with 252-day returns: [0.20, -0.05, 0.10]
    Expected rank signal: [+1.0, -1.0, 0.0] (or proportional)
    """

def test_mean_reversion_direction():
    """
    Stock with z-score = +2.0 should have negative signal (expect reversion down).
    Stock with z-score = -2.0 should have positive signal (expect reversion up).
    """

def test_signal_values_in_range():
    """All signal values should be in [-1, 1] after ranking."""
```

#### Statistical Tests (`test_statistics.py`)

```python
def test_ic_perfect_signal():
    """
    Signal perfectly predicts forward returns (rank-identical).
    IC should be 1.0.
    """

def test_ic_random_signal():
    """
    Random signal. IC should be approximately 0 (within tolerance).
    Use fixed seed for reproducibility.
    """

def test_hac_tstat_differs_from_naive():
    """
    With overlapping 20-day forward returns, HAC t-stat should be
    smaller than naive t-stat (standard errors are larger).
    """

def test_permutation_within_date():
    """
    Verify that permutation shuffles signal values across tickers
    within each date, NOT across dates.
    """

def test_bonferroni_basic():
    """
    p_values = {"a": 0.01, "b": 0.04, "c": 0.06}
    With alpha=0.05 and m=3: adjusted_alpha = 0.0167
    Only "a" should be significant.
    """

def test_benjamini_hochberg_basic():
    """
    p_values = {"a": 0.005, "b": 0.01, "c": 0.04, "d": 0.5}
    alpha=0.05, m=4
    BH thresholds: 0.0125, 0.025, 0.0375, 0.05
    Significant: "a" (0.005 ≤ 0.0125), "b" (0.01 ≤ 0.025), "c" (0.04 > 0.0375 → NO)
    """
```

#### Portfolio Tests (`test_portfolio.py`)

```python
def test_equal_weight_long_short():
    """
    5 stocks, signal rank Q1=[A], Q2=[B], Q3=[C], Q4=[D], Q5=[E]
    Long E, short A, zero for B,C,D.
    Gross exposure should be 1.0 (50% long + 50% short).
    """

def test_weights_sum_constraint():
    """
    Verify hard constraints on every generated weight vector:
    gross exposure <= 1.0, abs(net exposure) <= 0.20,
    and max(abs(weight)) <= 0.10.
    """

def test_position_and_net_exposure_caps():
    """Verify position and net exposure caps are enforced after normalization."""

def test_vol_scaled_weights():
    """
    Two stocks: same signal, but vol_A = 2 × vol_B.
    Weight of B should be approximately 2 × weight of A.
    """
```

#### Transaction Cost Tests (`test_costs.py`)

```python
def test_zero_cost():
    """With all costs = 0, fill_price = reference_price and commission = 0."""

def test_single_pass_accounting():
    """
    HAND-CALCULATED END-TO-END COST ACCOUNTING TEST:
    
    Reference price = 100.00
    Quantity = 100 shares
    Direction = BUY
    spread_bps = 5, slippage_bps = 5, commission_bps = 5
    
    Step 1: execution_adjustment = (5 + 5) / 10000 = 0.001
    Step 2: fill_price = 100.00 × (1 + 0.001) = 100.10
    Step 3: commission = |100.00 × 100| × 5 / 10000 = 0.50
    Step 4: cash_change = -(100.10 × 100 + 0.50) = -$10,010.50
    Step 5: position_change = +100 shares at avg_entry = $100.10
    Step 6: total_equity = initial_cash - 10010.50 + 100 × current_price
    
    This test PROVES:
    - Spread is counted ONCE (embedded in fill_price)
    - Slippage is counted ONCE (embedded in fill_price)
    - Commission is counted ONCE (separate cash deduction)
    - No cost is double-counted
    """

def test_sell_single_pass():
    """
    SELL side verification:
    reference_price = 100.00, quantity = 100, direction = SELL
    spread_bps = 5, slippage_bps = 5, commission_bps = 5
    
    fill_price = 100.00 × (1 - 0.001) = 99.90
    commission = 0.50
    cash_change = +(99.90 × 100 - 0.50) = +$9,989.50
    """

def test_effective_price_buy():
    """
    Buy at price=100, spread_bps=5, slippage_bps=5
    Effective price = 100 × (1 + 10/10000) = 100.10
    """

def test_effective_price_sell():
    """
    Sell at price=100, spread_bps=5, slippage_bps=5
    Effective price = 100 × (1 - 10/10000) = 99.90
    """
```

#### Backtest Tests (`test_backtest.py`)

```python
def test_three_day_backtest():
    """
    Hand-calculated 3-day backtest:
    Day 1: Market data arrives. Signal computed. Orders generated.
    Day 2: Orders filled at adj_close[2] (single-pass costs). New signal. New orders.
    Day 3: Orders filled at adj_close[3]. Final snapshot.
    
    All values (cash, position, equity) verified against manual calculation.
    """

def test_no_orders_on_first_day():
    """Backtest should not generate fills on the first day (no prior signal)."""

def test_cash_accounting():
    """After buy + sell sequence, cash should match expected value."""
```

### 28.3 Look-Ahead Bias Tests (`test_no_lookahead.py`) — P0

These are the most important tests in the project.

```python
def test_signal_does_not_use_future_prices():
    """
    Given a signal calculated at t:
    Modifying OHLCV data after t must NOT change signal[t].
    
    Method:
    1. Compute signal on full dataset
    2. Corrupt all prices after day t (set to 999999)
    3. Recompute signal on corrupted dataset
    4. Assert signal[t] is identical in both cases
    """

def test_signal_cannot_use_close_t_plus_1():
    """
    Specifically verify that signal[t] does not use close[t+1].
    Corrupt close[t+1] only. Signal[t] must be unchanged.
    """

def test_fill_price_is_after_signal_date():
    """
    For every fill in a backtest, assert:
    fill.date > signal.date
    """

def test_signal_cannot_affect_execution_at_t():
    """
    Signal computed at t must not produce a fill at t.
    The earliest fill from signal[t] must be at t+1.
    """

def test_portfolio_return_begins_after_execution():
    """
    Portfolio return attributable to a new position from signal[t]
    cannot begin before the execution point (close[t+1]).
    """

def test_feature_uses_only_past_data():
    """
    Compute feature on day t.
    Remove all data after day t.
    Recompute feature on day t.
    Assert identical.
    """

def test_no_future_in_cross_sectional_rank():
    """
    Cross-sectional rank on day t should only use data from day t.
    Verify by adding/removing tickers from future dates and checking
    that today's ranks don't change.
    """
```

**Synthetic dataset for timing tests**: Create a small (3 tickers × 10 days) synthetic dataset with hand-calculated expected behavior at each step. The dataset should have known prices and signals so that exact cash, position, and equity values can be verified at each time step.

### 28.4 Data Quality Tests (`test_data_quality.py`) — P0

```python
def test_no_forward_filled_bars_in_features():
    """
    Insert a NaN gap in OHLC data for a ticker on a specific date.
    Verify that the feature for that ticker on that date is NaN,
    not computed from a forward-filled price.
    """

def test_missing_bars_identifiable():
    """
    After cleaning, genuinely missing observations (exchange was open,
    but data is absent) are NaN, not zero or forward-filled.
    """

def test_security_termination_handled():
    """
    For a ticker that terminates mid-sample:
    - Data after termination is NaN
    - Features after termination are NaN
    - Signal after termination is NaN
    - Any open position is handled by the explicit termination-liquidation rule
    - No new position is taken after termination
    """

def test_signal_no_future_timestamps():
    """
    For every signal value stamped at date t, verify that the signal
    was computed using only data available at or before close of day t.
    """

def test_universe_no_future_information():
    """
    Verify that the universe file contains only tickers that were
    plausible S&P 100 members as of the start of the sample (2014-01-01).
    """

def test_corporate_action_consistency():
    """
    Verify that adj_close is used consistently for features, signals,
    and backtest reference prices. No mixing of raw close for some
    calculations and adj_close for others (except split-adjusted volume for
    volume features).
    """
```

### 28.5 Testing Tools

- **Framework**: `pytest`
- **Tolerance**: `numpy.testing.assert_allclose` with `atol=1e-8` for floating-point comparisons
- **Fixtures**: Common test data (toy price series, known signals) as pytest fixtures
- **Markers**: `@pytest.mark.slow` for expensive tests (bootstrap, permutation)
- **Coverage target**: ≥ 80% for `src/` modules

---

## 29. Reproducibility

(Identical to v1.0 except `universe` references updated to `sp100_20140101`)

### 29.1 Environment

| Item | Specification |
|------|---------------|
| Python version | 3.11.x (specified in `pyproject.toml`) |
| Dependency management | `pyproject.toml` with `pip-compile` → `requirements.txt` |
| Virtual environment | `venv` (standard library) |

### 29.2 `pyproject.toml` Dependencies

```toml
[project]
name = "alpha-research"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "numpy>=1.24,<2.0",
    "pandas>=2.0,<3.0",
    "scipy>=1.11,<2.0",
    "scikit-learn>=1.3,<2.0",
    "statsmodels>=0.14,<1.0",
    "xgboost>=2.0,<3.0",
    "yfinance>=0.2,<1.0",
    "matplotlib>=3.7,<4.0",
    "seaborn>=0.12,<1.0",
    "pyyaml>=6.0,<7.0",
    "pyarrow>=14.0,<18.0",
    "jupyter>=1.0,<2.0",
    "tqdm>=4.65,<5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4,<9.0",
    "pytest-cov>=4.1,<6.0",
]
```

### 29.3 Random Seeds

- All random operations use a seed from the config file (default: 42)
- NumPy: `np.random.default_rng(seed)`
- scikit-learn / XGBoost: `random_state=seed`
- Permutation tests: seed is explicitly passed
- Bootstrap: seed is explicitly passed

### 29.4 Experiment Tracking

Every experiment run saves:
1. **Config file** (copied to output directory)
2. **Config hash** (SHA-256 of the YAML content)
3. **Git commit** (`git rev-parse HEAD`)
4. **Timestamp** (start and end)
5. **Random seed**
6. **Data version** (hash of the input Parquet file)
7. **Results** (metrics, equity curve, trade log)
8. **Hypothesis registry** (all tested configurations)

---

## 30–31. Results Storage & Visualization Plan

(Identical to v1.0, with updated cost sensitivity plots to include economic break-even cost, and all IC-related plots showing HAC t-stats as primary.)

---

## 32. Research Report

### Report Structure

`reports/final_report.md`

```markdown
# Statistical Alpha Research: Do Simple Trading Signals Show Robust Predictive Association?

## Abstract
[2-3 paragraphs summarizing the research question, methodology, and key findings.
Use language like "evidence suggests" not "we proved."]

## 1. Research Question
[Central question and sub-questions]

## 2. Motivation
[Why this research matters; gap in typical student projects]

## 3. Dataset
[Source, start-of-sample universe (2014), time period, 
survivorship bias discussion (not solved — documented),
missing-data policy (no forward-fill), adjusted-price convention]

## 4. Methodology
[Research workflow diagram; anti-patterns avoided;
confirmatory vs. exploratory distinction]

## 5. Feature Engineering
[Feature definitions, lag discipline, dimensional consistency
(price std vs. return std), NaN propagation, leakage prevention]

## 6. Signal Construction
[Each signal: formula, intuition, primary parameters.
Mean-reversion z-score: price-based, dimensionally correct.]

## 7. Statistical Testing
[IC analysis with HAC inference (primary), naive t-stat (diagnostic),
bootstrap CI, within-date permutation tests.
Overlapping forward-return dependence discussed.]
[Multiple-testing correction: confirmatory family of 4.
Total exploratory configurations logged.]

## 8. Portfolio Construction
[Methods used, exposure constraints (gross ≤ 1.0), rationale]

## 9. Event-Driven Backtesting
[Architecture, timing model (synthetic daily-bar convention),
single-pass cost accounting, validation]

## 10. Transaction Costs
[Single-pass cost model, sensitivity analysis,
economic break-even cost, Sharpe-zero crossing]

## 11. Walk-Forward Validation
[Window design, development OOS, final holdout (2024),
research-iteration bias caveat]

## 12. Robustness Analysis
[Parameter stability, time-period stability, asset-subset stability]

## 13. Regime Analysis
[Regime definitions, conditional performance (descriptive).
Regime-conditioned trading: if attempted, walk-forward validated.]

## 14. ML Comparison
[Pre-specified combined baseline (not winner-selected).
HAC significance test on IC difference.]

## 15. Results
[Summary table of all signals × all metrics.
Key findings stated with appropriate caution.]

## 16. Limitations
- Survivorship bias: start-of-sample universe removes future-membership 
  look-ahead but does not eliminate survivorship/delisting bias (would 
  require CRSP)
- Missing data: bars not forward-filled, but some securities may have 
  gaps that affect feature coverage
- Adjusted-price convention: total-return approximation, not actual 
  tradable execution
- Synthetic execution price: close[t+1] is a research convention, not 
  obtainable in practice
- Free data source (not institutional-grade)
- Small universe (~80-100 stocks)
- Historical start-of-sample membership is frozen from a documented 2014 list; no full point-in-time constituent database is available
- Simple transaction cost model (proportional, no market-impact function)
- Research-selection bias: walk-forward does not eliminate bias from 
  iterative research-design decisions
- Applying Bonferroni/FDR does not erase implicit hypothesis selection 
  during the research process
- [Other limitations discovered during research]

## 17. Conclusions
[Answer to the central research question.
What was demonstrated and what was not.
Honest assessment using cautious language.]

## 18. Future Work
[Extensions that would strengthen the research]
```

### Report Principles

1. **Negative results are reported**: If momentum fails after costs, say so.
2. **Limitations are explicit**: Don't hide known weaknesses.
3. **No overclaiming**: "The signal shows statistically significant predictive association (HAC p < 0.01)" ≠ "We found alpha."
4. **Plots support claims**: Every claim has a corresponding figure or table.
5. **Cautious language**: Use "evidence suggests," "predictive association," "statistically significant under the stated test," "economically meaningful after costs," "robust across the tested perturbations." Do NOT use "proved alpha" or "discovered alpha" without explicit qualification.

---

## 33. GitHub README

### Template

```markdown
# Statistical Alpha Research & Event-Driven Backtesting Framework

## Research Question

Do simple statistical trading signals (momentum, mean reversion, volatility, 
abnormal volume) show robust out-of-sample predictive association with future 
equity returns after realistic transaction costs?

## Methodology

This project follows a disciplined quantitative research workflow:

1. **Data**: Daily OHLCV for ~80-100 liquid US equities (S&P 100 constituents 
   as of 2014), 2014–2024. Start-of-sample universe; no forward-fill of 
   missing bars.
2. **Features**: Returns (multi-horizon), momentum, mean-reversion price 
   z-scores, realized volatility, abnormal volume — all dimensionally 
   consistent.
3. **Signals**: 4 statistical signal families, cross-sectionally ranked. 
   Pre-specified primary configurations (confirmatory); additional 
   exploratory parameter sweeps.
4. **Statistical Testing**: Information Coefficient with HAC/Newey-West 
   inference (primary), within-date permutation tests, block bootstrap 
   confidence intervals, Bonferroni and BH-FDR correction.
5. **Backtesting**: Event-driven backtester with single-pass transaction 
   cost accounting (no double-counting).
6. **Validation**: Expanding-window walk-forward evaluation with six 
   development test windows (2018–2023) and a separate final holdout 
   (2024) evaluated exactly once after methodology freeze. The development 
   OOS results are supporting evidence; the 2024 holdout is the primary 
   confirmatory OOS result.
7. **Robustness**: Parameter perturbation, time-period stability, cost 
   sensitivity, regime analysis (descriptive), all performed without using the 
   final 2024 holdout for strategy selection. ML comparison uses a fixed common 
   20-day forward-return target.

A negative result (signals that fail out-of-sample) is a valid research outcome.

## Key Findings

[To be filled after experiments are completed]

## Architecture

[Simplified architecture diagram]

## Directory Structure

[Project tree]

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py --universe sp100_20140101
```

## Running Experiments

```bash
python scripts/run_experiment.py --config configs/momentum.yaml
```

## Running Tests

```bash
pytest tests/ -v
```

## Experiment Matrix

| # | Experiment | Notebook |
|---|-----------|----------|
| 1 | Data quality audit | 01_data_audit.ipynb |
| 2 | Momentum signal | 03_signal_research.ipynb |
| ... | ... | ... |

## Limitations

- Start-of-sample universe removes future-membership look-ahead but 
  survivorship/delisting bias remains (would require CRSP)
- Daily frequency only (no intraday)
- Free data source (Yahoo Finance)
- Simple proportional transaction cost model
- Historical start-of-sample membership is frozen from a documented 2014 list; no full point-in-time constituent database is available
- Adjusted-price total-return approximation
- Simplified termination liquidation with no delisting-return model
- Synthetic execution price convention
- Zero risk-free rate assumed for reported Sharpe/Sortino metrics

## Author

[Name] — B.Tech + MS, ECE, IIIT Hyderabad
```

---

## 34–42. Development Roadmap, Priority Classification, Technical Tradeoffs, Interview Relevance, Candidate Defensibility, Common Failure Modes, Resume Bullets, Acceptance Criteria, Future Extensions

These sections are substantially identical to v1.0 with the following specific updates applied throughout:

- All references to `sp100_20240101` → `sp100_20140101`
- All references to "forward-fill missing prices" → removed; replaced with "preserve NaN"
- All references to `rolling_std_20` as z-score denominator → `rolling_std_price_20` (price-based)
- All references to "IC t-test" as primary → "HAC t-test (primary), naive t-stat (diagnostic)"
- Historical universe is frozen at start-of-sample; no current-index fallback or full-sample eligibility filter
- Missing OHLC bars remain NaN; signal eligibility is point-in-time
- Volume anomaly features are split-aware; split-affected dates are adjusted or excluded
- Development OOS is 2018–2023 supporting evidence; 2024 is a separate final holdout
- Final holdout (2024) is the primary confirmatory OOS result; development OOS is supporting evidence subject to research-iteration bias
- Portfolio exposure constraints are hard constraints and are unit-tested
- Portfolio-level volatility targeting removed from the Strong version because it can conflict with the no-leverage rule; retained only as a future extension
- Sharpe convention explicitly assumes zero risk-free rate
- All references to "best statistical signal" as ML baseline → "pre-specified combined statistical signal"
- All references to "break-even cost" as Sharpe ≤ 0 → includes "economic break-even cost" as primary
- Regime conditioning: descriptive analysis is free; regime-conditioned trading is a new strategy
- `avg_trade_return` → `avg_daily_return`; `hit_rate` → `daily_hit_rate`
- Gross exposure default: 1.0 (not 2.0)
- Cost accounting: "single-pass" everywhere
- Resume bullets use cautious language ("predictive association" not "alpha")

### Final Acceptance Criteria (Section 41, updated)

The project is complete when **all** of the following are satisfied:

#### Data & Pipeline
- [ ] Data downloads and loads correctly for the start-of-sample universe
- [ ] Data quality report generated with documented anomalies, termination dates, missingness, and point-in-time eligibility decisions
- [ ] No OHLC forward-fill — missing bars are NaN
- [ ] Terminated securities documented with last valid observation
- [ ] No full-sample missingness or future liquidity filter is used to remove start-of-sample securities
- [ ] Split-affected volume dates are adjusted or explicitly excluded from the volume signal
- [ ] Cleaned data passes validation checks (no negative prices, low ≤ high, etc.)
- [ ] All features compute without errors; NaN where expected (insufficient lookback, missing bars)

#### Signals & Statistics
- [ ] All 4 core signals implemented with documented formulas and primary configurations
- [ ] Mean-reversion z-score is dimensionally correct (price std denominator)
- [ ] IC computed for all signals vs. forward returns
- [ ] HAC t-test on IC (primary inference)
- [ ] Naive t-stat computed as diagnostic only
- [ ] Bootstrap CI computed for mean IC
- [ ] Within-date permutation test conducted
- [ ] Bonferroni and BH-FDR correction applied to confirmatory family
- [ ] Exploratory configurations logged in hypothesis registry
- [ ] Random-signal demonstration experiment completed

#### Backtesting
- [ ] Event-driven backtester produces correct results on 3-day toy example (hand-verified)
- [ ] Look-ahead bias tests pass (including synthetic dataset timing tests)
- [ ] Transaction costs correctly applied — single-pass, no double-counting (verified by hand-calculated test)
- [ ] At least 3 portfolio construction methods implemented (gross exposure ≤ 1.0)
- [ ] Full backtest runs for all signals with medium-cost model

#### Validation & Robustness
- [ ] Walk-forward evaluation with 7 windows completed (6 development + 1 final holdout)
- [ ] IS vs. OOS comparison available for all signals
- [ ] Transaction-cost sensitivity analysis across ≥ 3 cost regimes with economic break-even cost
- [ ] Parameter perturbation analysis for at least 2 signals (primary vs. exploratory clearly labeled)
- [ ] Time-period stability analysis (split test period in half)

#### Evaluation & Reporting
- [ ] ~14 performance metrics computed with precise definitions
- [ ] Key visualizations generated (equity curve, drawdown, IC, quintile, cost sensitivity)
- [ ] Research report completed with abstract, methodology, results, limitations, conclusions
- [ ] Report uses cautious language throughout
- [ ] Report discusses negative results honestly
- [ ] Report distinguishes confirmatory from exploratory findings
- [ ] README contains research question, setup instructions, and experiment matrix

#### Engineering
- [ ] ≥ 30 unit tests passing (including data-quality, timing, and cost-accounting tests)
- [ ] Configuration-driven experiments (YAML)
- [ ] All experiments reproducible from config + seed
- [ ] Code organized with clean separation between library and notebooks
- [ ] No business logic only inside notebooks

#### Honesty
- [ ] The project does not claim "alpha" without proper OOS evidence and cautious qualification
- [ ] Limitations section discusses survivorship bias, missing-data policy, adjusted-price convention, execution convention, research-selection bias
- [ ] Failed signals are reported, not hidden
- [ ] Multiple-testing analysis demonstrates awareness of false discovery risk, including the limitation of post-hoc correction

---

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **Alpha** | Returns in excess of a benchmark, after adjusting for risk |
| **IC (Information Coefficient)** | Cross-sectional rank correlation between signal and forward return |
| **ICIR (IC Information Ratio)** | Mean IC / Std IC; measures consistency of signal quality |
| **HAC (Heteroskedasticity and Autocorrelation Consistent)** | Standard error estimation robust to serial correlation |
| **Walk-forward** | Time-respecting validation: train on past, test on future, advance window |
| **Drawdown** | Peak-to-trough decline in portfolio value |
| **Sharpe ratio** | Mean daily portfolio return / standard deviation of daily returns (annualized), assuming a zero risk-free rate for this student project |
| **Sortino ratio** | Mean daily portfolio return / downside standard deviation (annualized), assuming a zero risk-free rate |
| **Calmar ratio** | CAGR / |maximum drawdown| |
| **Turnover** | Sum of absolute weight changes; higher = more trading |
| **Slippage** | Difference between expected and actual execution price |
| **BH-FDR** | Benjamini-Hochberg False Discovery Rate correction |
| **FWER** | Family-Wise Error Rate — probability of at least one false rejection |
| **Cross-sectional** | Across securities at a single point in time |
| **Quintile** | One-fifth; stocks sorted into 5 groups by signal |
| **OHLCV** | Open, High, Low, Close, Volume — standard daily bar data |
| **Look-ahead bias** | Using future information to make past decisions |
| **Survivorship bias** | Analyzing only entities that survived, ignoring those that didn't |
| **Confirmatory hypothesis** | Pre-specified before examining results |
| **Exploratory hypothesis** | Discovered through data exploration or parameter sweeps |

---

## Appendix B: Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| $P_{i,t}$ | Adjusted close price of stock $i$ on day $t$ |
| $r_{i,t}$ | Simple return of stock $i$ on day $t$: $(P_{i,t} - P_{i,t-1}) / P_{i,t-1}$ |
| $V_{i,t}$ | Trading volume of stock $i$ on day $t$ |
| $\text{SMA}_k(x)_t$ | Simple moving average of $x$ over the past $k$ days ending at $t$ |
| $\sigma_{i,t}$ | Realized volatility of stock $i$ at time $t$ (based on return std) |
| $\text{std}_k(P)_t$ | Rolling standard deviation of prices (for z-score denominator) |
| $\text{std}_k(r)_t$ | Rolling standard deviation of returns (for realized volatility) |
| $w_{i,t}$ | Portfolio weight of stock $i$ on day $t$ |
| $\text{IC}_t$ | Cross-sectional Spearman correlation on day $t$ |
| $t_{HAC}$ | HAC/Newey-West t-statistic (primary inference) |
| $t_{naive}$ | Ordinary t-statistic (diagnostic only) |
| $N$ | Number of securities in the universe |
| $T$ | Number of trading days |
| $\alpha$ | Significance level (default 0.05) |
| $m$ | Number of hypotheses in the confirmatory family |

---

*Specification version: 1.2*
*Author: Research Lead / System Architect*
*Date: August 2025*
*Revision: Audit corrections integrated — see amendment_report.md*
*Target implementation time: 6–8 weeks*
