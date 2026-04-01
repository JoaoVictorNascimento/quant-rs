# Quantitative Finance Fundamentals

This document explains the core concepts behind the library.

The goal is to provide intuition, mathematical background, and context for the implemented modules.

---

# 📈 Returns

Returns measure how much an asset changes in value over time.

## Simple Return

R = (P_t / P_{t-1}) - 1

Represents the percentage gain or loss between two periods.

---

## Log Return

r = ln(P_t / P_{t-1})

Log returns are additive and useful for statistical modeling.

---

## Why Returns Matter

Returns are the foundation for:

- Volatility
- Sharpe Ratio
- Portfolio analysis
- Backtesting

---

# 📊 Volatility

Volatility measures how much returns fluctuate.

## Variance

σ² = (1 / (n - 1)) Σ (r_i - mean)^2

---

## Standard Deviation

σ = sqrt(variance)

---

## Interpretation

- High volatility → high uncertainty
- Low volatility → stable asset

---

# ⚖️ Risk vs Return

In finance, return alone is not enough.

We need to evaluate:

- how much we gain
- how much risk we take

---

# 📉 Drawdown

Drawdown measures the decline from a peak.

## Formula

DD = (P_t / peak) - 1

---

## Maximum Drawdown

Largest drop from peak to trough.

---

## Why It Matters

- Measures real losses
- Important for risk management
- Used by funds and traders

---

# 📊 Sharpe Ratio

Measures return adjusted for risk.

## Formula

S = (mean(r) - r_f) / σ

---

## Interpretation

- Higher is better
- Penalizes volatility

---

# 📉 Sortino Ratio

Variant of Sharpe that considers only downside risk.

## Formula

S = (mean(r) - r_f) / downside_deviation

---

## Why It Matters

- Ignores upside volatility
- Focuses on harmful risk

---

# 📦 Portfolio

A portfolio combines multiple assets.

## Portfolio Return

R_t = Σ w_i * r_i

---

## Key Concepts

- Diversification reduces risk
- Weights define exposure

---

# 🔄 Backtesting

Backtesting evaluates a strategy using historical data.

## Key Steps

1. Generate signals
2. Simulate trades
3. Track performance

---

## Common Pitfalls

- Look-ahead bias
- Overfitting
- Ignoring transaction costs

---

# 🧠 Final Insight

This library follows a pipeline:

prices → returns → metrics → strategy → backtest

Each module builds on top of the previous one.

Understanding this flow is essential to use the library correctly.