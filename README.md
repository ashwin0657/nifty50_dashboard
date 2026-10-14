# 📈 NIFTY 50 Two-Decade Financial Analytics & ML Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An institutional-grade quantitative finance analytics suite and interactive web dashboard built with **Streamlit**, **Pandas**, and **Scikit-Learn**. The application performs deep quantitative risk-return profiling, algorithmic candidate stock screening, portfolio backtesting, and machine learning volatility regime classification across two decades of National Stock Exchange of India (NSE) NIFTY 50 historical data.

---

## 📑 Table of Contents

- [Overview & Architecture](#overview--architecture)
- [Key Features](#key-features)
- [Quantitative Methodology](#quantitative-methodology)
  - [Performance & Risk Metrics](#performance--risk-metrics)
  - [Algorithmic Screening Score](#algorithmic-screening-score)
  - [Correlation-Constrained Diversification](#correlation-constrained-diversification)
- [Machine Learning Pipeline](#machine-learning-pipeline)
  - [Feature Engineering](#feature-engineering)
  - [Volatility Regime Classifier](#volatility-regime-classifier)
- [Dashboard Walkthrough](#dashboard-walkthrough)
- [Project Structure](#project-structure)
- [Installation & Quickstart](#installation--quickstart)
- [Dataset Specifications](#dataset-specifications)
- [Disclaimer](#disclaimer)

---

## 🏛 Overview & Architecture

Modern algorithmic portfolio management requires combining robust historical risk analysis with forward-looking regime modeling. This dashboard provides financial analysts, data scientists, and quantitative researchers with a high-performance, interactive toolkit to dissect and model Indian equity performance.

```
┌─────────────────────────────────────────────────────────────┐
│                   NIFTY 50 Dataset Engine                   │
│         (Excel / CSV Market History + Metadata CSV)         │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌───────────────────────────┐         ┌───────────────────────────┐
│ Quantitative Risk Engine  │         │ Feature Engineering & ML  │
│ • CAGR & Volatility       │         │ • Momentum & Rolling Vol  │
│ • Sharpe & Sortino Ratios │         │ • Moving Avg Distances    │
│ • Max Drawdowns & Win Rate│         │ • Intraday Range & RSI    │
│ • Multi-Factor Screening  │         │ • Random Forest Vol Reg.  │
└───────────┬───────────────┘         └─────────────┬─────────────┘
            │                                       │
            └──────────────────┬────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  Streamlit Web Application                  │
│   [1. Overview]   [2. Stock Deep-Dive]   [3. Portfolio]     │
│                     [4. ML Regimes]                         │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

1. **Comprehensive Executive Overview**:
   - High-level KPI summary cards: valid rows count, unique constituent tickers, starting and ending dates.
   - Interactive, styled sortable data table containing quantitative risk-return metrics for every constituent over the recent decade (2011–present).

2. **Single-Stock Deep Dive**:
   - Interactive stock selector with immediate metadata lookup (Company Name, Industry classification).
   - Core metrics: CAGR, Annualized Volatility, Maximum Drawdown.
   - Clean, high-resolution historical closing price chart.
   - Live inspection table for 25 technical & engineered indicators.

3. **Algorithmic Candidate Portfolio Screening**:
   - Automated multi-factor screening model combining return rate, risk-adjusted ratios, and drawdown safety.
   - Greedy low-correlation diversification filter ensures the candidate basket does not overconcentrate in correlated instruments ($|\rho| < 0.75$).
   - Top candidate basket identified from historical data: `HINDUNILVR`, `SHREECEM`, `ULTRACEMCO`, `BAJAJFINSV`, `NESTLEIND`, `BRITANNIA`.

4. **Equal-Weight Portfolio Backtesting Engine**:
   - Cumulative wealth backtest curve modeling the growth of ₹1 over time.
   - Portfolio-level performance figures: Backtest CAGR, Portfolio Volatility, Portfolio Sharpe Ratio, and Peak-to-Trough Drawdown.

5. **Machine Learning Volatility Regime Classifier**:
   - Supervised classification leveraging `RandomForestClassifier` (150 estimators, balanced class weights).
   - Predicts 5-day forward high-volatility market states.
   - Visual feature importance ranking and ROC-AUC / Accuracy performance metrics.

---

## 📐 Quantitative Methodology

### Performance & Risk Metrics

For each stock $i$ with daily closing prices $P_t$ and daily returns $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$:

1. **Compound Annual Growth Rate (CAGR)**:
   $$\text{CAGR} = \left(\frac{P_T}{P_0}\right)^{\frac{1}{\text{Years}}} - 1$$

2. **Annualized Volatility ($\sigma_{\text{ann}}$)**:
   $$\sigma_{\text{ann}} = \text{Std}(R_t) \times \sqrt{252}$$

3. **Sharpe Ratio** (assuming zero risk-free benchmark):
   $$\text{Sharpe} = \frac{\text{Mean}(R_t) \times 252}{\sigma_{\text{ann}}}$$

4. **Sortino Ratio** (downside deviation penalization):
   $$\text{Sortino} = \frac{\text{Mean}(R_t) \times 252}{\text{Std}(R_t \mid R_t < 0) \times \sqrt{252}}$$

5. **Maximum Drawdown (MDD)**:
   $$\text{MDD} = \min_{t} \left(\frac{W_t}{\max_{\tau \le t} W_\tau} - 1\right), \quad \text{where } W_t = \prod_{s=1}^t (1 + R_s)$$

6. **Daily Win Rate**:
   $$\text{Win Rate} = \frac{1}{N}\sum_{t=1}^N \mathbf{1}_{\{R_t > 0\}}$$

---

### Algorithmic Screening Score

To score stocks holistically, metrics are winsorized and normalized between their 5th ($Q_{0.05}$) and 95th ($Q_{0.95}$) percentiles:

$$z(X) = \text{clip}\left(\frac{X - Q_{0.05}}{Q_{0.95} - Q_{0.05}}, 0, 1\right)$$

The composite score balances performance, efficiency, and risk preservation:

$$\text{Screening Score} = 0.25 \cdot z(\text{CAGR}) + 0.25 \cdot z(\text{Sharpe}) + 0.15 \cdot z(\text{Sortino}) + 0.15 \cdot (1 - z(\sigma)) + 0.15 \cdot (1 - z(|\text{MDD}|)) + 0.05 \cdot z(\text{Win Rate})$$

---

### Correlation-Constrained Diversification

To ensure the portfolio is not exposed to collinear industry shocks, candidates are selected sequentially according to their screening score. A new candidate $t$ is added to the basket $\mathcal{C}$ only if:

$$\max_{s \in \mathcal{C}} \left|\rho(R^{(t)}, R^{(s)})\right| < 0.75$$

This process continues until the desired basket size ($N=6$) is filled.

---

## 🤖 Machine Learning Pipeline

### Feature Engineering

The feature generation pipeline computes 12 technical signals capturing momentum, mean reversion, microstructure, and volatility:

| Feature | Description | Formula / Logic |
|:---|:---|:---|
| `ret1` | 1-Day Return | $\frac{P_t - P_{t-1}}{P_{t-1}}$ |
| `ret5` | 5-Day Cumulative Return | $\frac{P_t - P_{t-5}}{P_{t-5}}$ |
| `ret20` | 20-Day Cumulative Return | $\frac{P_t - P_{t-20}}{P_{t-20}}$ |
| `vol20` | 20-Day Rolling Volatility | $\text{Std}(R_{t-19 \dots t})$ |
| `vol60` | 60-Day Rolling Volatility | $\text{Std}(R_{t-59 \dots t})$ |
| `sma20` | Distance from 20-Day SMA | $\frac{P_t}{\text{SMA}_{20}(P)} - 1$ |
| `sma50` | Distance from 50-Day SMA | $\frac{P_t}{\text{SMA}_{50}(P)} - 1$ |
| `range` | Normalized Intraday Range | $\frac{\text{High}_t - \text{Low}_t}{\text{Close}_t}$ |
| `gap` | Overnight Gap | $\frac{\text{Open}_t}{\text{Prev Close}_t} - 1$ |
| `volchg` | Volume Percentage Change | $\frac{V_t - V_{t-1}}{V_{t-1}}$ |
| `delivery` | Percentage Deliverable Volume | Supplied NSE `%Deliverble` |
| `rsi` | 14-Day Relative Strength Index | $100 - \frac{100}{1 + \frac{\text{AvgGain}_{14}}{\text{AvgLoss}_{14}}}$ |

### Volatility Regime Classifier

- **Target Variable**: 5-day forward absolute return exceeding the sample median:
  $$Y_t = \mathbf{1}_{\left\{\left|\frac{P_{t+5} - P_t}{P_t}\right| > \text{median}\left(\left|\frac{P_{t+5} - P_t}{P_t}\right|\right)\right\}}$$
- **Model**: `RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=20, class_weight='balanced')`
- **Evaluation**: Stratified 80/20 train/test split evaluated using **ROC-AUC** and **Classification Accuracy**.

---

## 🖥 Dashboard Walkthrough

The Streamlit user interface is structured into four primary modules:

1. **Overview Tab**:
   Displays dataset metrics and a formatted table of all stocks sorted by historical screening score.
2. **Stock Analysis Tab**:
   Allows user to choose any constituent stock to view company name, sector/industry, annualized metrics, interactive closing price chart, and recent feature table.
3. **Portfolio Tab**:
   Displays the screening basket candidates, equal weight allocations, and backtest cumulative growth curve with disclaimer.
4. **ML Tab**:
   Runs the volatility regime model, displays test-set ROC-AUC and Accuracy metrics, and visualizes feature importances via a horizontal bar chart.

---

## 📂 Project Structure

```
.
├── Combined_P3.xlsx            # Primary historical NIFTY 50 dataset (~2000–2021)
├── stock_metadata.csv          # Metadata file mapping symbols to Company Name & Industry
├── nifty50_dashboard.py        # Core Streamlit analytics dashboard & modeling application
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Git exclusions for python, caches, and environment artifacts
└── README.md                   # Project documentation and user manual
```

---

## 🚀 Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12
- `git`

### 1. Clone the Repository
```bash
git clone https://github.com/ashwin0657/nifty50_dashboard.git
cd nifty50_dashboard
```

### 2. Set Up a Virtual Environment (Recommended)
On Windows (Command Prompt / PowerShell):
```powershell
python -m venv venv
venv\Scripts\activate
```

On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard
```bash
streamlit run nifty50_dashboard.py
```

The application will open in your default browser at `http://localhost:8501`.

> **Note on Data Loading**: The dashboard automatically detects and loads `Combined_P3.xlsx` and `stock_metadata.csv` if they are present in the project directory. You can also upload any custom or updated dataset file using the sidebar uploader.

---

## 📊 Dataset Specifications

The dataset includes historical trading records containing the following fields:
- `File Name` / `Symbol`: NSE ticker identifier (e.g., `RELIANCE`, `TCS`, `INFY`, `HDFCBANK`)
- `Date`: Trading session timestamp
- `Prev Close`: Preceding trading day's closing price
- `Open`: Opening market price
- `High`: Highest traded price during the session
- `Low`: Lowest traded price during the session
- `Close`: Final traded closing price
- `VWAP`: Volume Weighted Average Price
- `Volume`: Total number of shares traded
- `Turnover`: Total monetary turnover
- `Trades`: Number of trade orders executed
- `%Deliverble`: Ratio of deliverable quantity to total volume

---

## ⚠️ Disclaimer

This software is for **educational, academic, and research purposes only**.
- Historical screening results, metrics, and backtests are **not indicative of future performance**.
- The backtesting model does **not** simulate transaction costs, taxes, slippage, liquidity constraints, short-selling restrictions, or dividend adjustments.
- This application does not constitute personalized investment, financial, legal, or tax advice. Always conduct independent due diligence before making investment decisions.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
