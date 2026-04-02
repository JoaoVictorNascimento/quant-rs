# Sortino Ratio

This module provides the **Sortino ratio** — a risk-adjusted return metric that refines the Sharpe ratio by penalising only **downside** volatility instead of total volatility.  
It is particularly useful when a return distribution is positively skewed, since upside variability should not be treated as risk.

The function requires at least two finite return values and at least one return strictly below the risk-free rate; invalid inputs return `QuantError` (`InsufficientData`, `InvalidValue`, or `DivisionByZero`).

---

# 📉 1. Sortino Ratio (`sortino_ratio`)

## Definition

The Sortino ratio divides the mean excess return (return minus risk-free rate) by the **downside deviation** — the standard deviation computed only over returns that fall below the risk-free rate.

## Mathematical Formula

**Downside deviation:**

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}%5Csigma_d%3D%5Csqrt%7B%5Cfrac%7B1%7D%7Bk%7D%5Csum_%7Br_i%3Cr_f%7D%5Cleft(r_i-r_f%5Cright)%5E2%7D" alt="σ_d = sqrt( (1/k) * Σ_{r_i < r_f} (r_i − r_f)² )" /></p>

**Sortino ratio:**

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}SR_d%3D%5Cfrac%7B%5Cbar%7Br%7D-r_f%7D%7B%5Csigma_d%7D" alt="SR_d = (r̄ − r_f) / σ_d" /></p>

Where:

- `r̄` is the arithmetic mean of all returns in the series  
- `r_f` is the period risk-free rate (same frequency as the returns)  
- `k` is the number of returns strictly below `r_f`  
- `σ_d` is the downside deviation, computed exclusively over the `k` underperforming periods  

## Intuition

Unlike the Sharpe ratio, which divides by the standard deviation of all returns, the Sortino ratio divides by a measure of risk that only considers periods where the return fell short of the target (risk-free rate). This means that a strategy with high upside variability is not penalised, which better reflects investor preferences: variance on the upside is desirable.

When no return falls below the risk-free rate, the denominator is undefined and the function returns `QuantError::DivisionByZero`.

## Example

Returns:

[0.1, 0.2, −0.1], risk-free = 0.0

Mean = (0.1 + 0.2 − 0.1) / 3 = **0.0667**

Downside returns (< 0.0): [−0.1]

Downside deviation = √((−0.1 − 0.0)² / 1) = √0.01 = **0.1**

Sortino = 0.0667 / 0.1 = **2/3 ≈ 0.6667**

## Code Example

```rust
use quantix::metrics::sortino::sortino_ratio;

let returns = vec![0.1, 0.2, -0.1];

let s = sortino_ratio(&returns, 0.0).unwrap();
let expected = 2.0_f64 / 3.0;
assert!((s - expected).abs() < 1e-12);
```

### With non-zero risk-free rate

Returns [0.1, 0.2, 0.05], risk-free = 0.08:

- Mean = 0.35 / 3 ≈ 0.1167
- Downside returns (< 0.08): [0.05]
- Downside deviation = √((0.05 − 0.08)² / 1) = √0.0009 = 0.03
- Sortino = (0.35/3 − 0.08) / 0.03 = **11/9 ≈ 1.2222**

```rust
use quantix::metrics::sortino::sortino_ratio;

let returns = vec![0.1, 0.2, 0.05];
let s = sortino_ratio(&returns, 0.08).unwrap();
let expected = 11.0_f64 / 9.0;
assert!((s - expected).abs() < 1e-12);
```
