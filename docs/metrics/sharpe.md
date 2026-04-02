# Sharpe Ratio

This module provides functions to compute the **Sharpe ratio** — the most widely used risk-adjusted return metric in quantitative finance.  
It measures how much excess return (above a risk-free rate) a strategy earns per unit of volatility, allowing fair comparison between strategies with different return and risk profiles.

All functions require at least two finite return values and non-zero volatility; invalid inputs return `QuantError` (`InsufficientData`, `InvalidValue`, or `DivisionByZero`).

---

# 📐 1. Sharpe Ratio (`sharpe_ratio`)

## Definition

The Sharpe ratio divides the mean excess return (return minus risk-free rate) by the period volatility of the return series. A higher value indicates better risk-adjusted performance.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}S%3D%5Cfrac%7B%5Cbar%7Br%7D-r_f%7D%7B%5Csigma%7D" alt="S = (r̄ − r_f) / σ" /></p>

Where:

- `r̄` is the arithmetic mean of the return series  
- `r_f` is the period risk-free rate (same frequency as the returns)  
- `σ` is the sample standard deviation of the returns  

## Intuition

The Sharpe ratio answers: "For each unit of risk taken, how much return did the strategy earn above the risk-free alternative?" A ratio above 1.0 is generally considered acceptable; above 2.0 is strong. A negative ratio means the strategy underperformed the risk-free rate. When volatility is exactly zero (all returns identical), the ratio is undefined and the function returns `QuantError::DivisionByZero`.

## Example

Returns:

[0.1, 0.2, 0.3]

Mean = 0.2, volatility = √((0.01 + 0 + 0.01) / 2) = √0.01 = 0.1

Sharpe (risk-free = 0.0): (0.2 − 0.0) / 0.1 = **2.0**  
Sharpe (risk-free = 0.1): (0.2 − 0.1) / 0.1 = **1.0**

## Code Example

```rust
use quantix::metrics::sharpe::sharpe_ratio;

let returns = vec![0.1, 0.2, 0.3];

let s = sharpe_ratio(&returns, 0.0).unwrap();
assert!((s - 2.0).abs() < 1e-12);

let s_rf = sharpe_ratio(&returns, 0.1).unwrap();
assert!((s_rf - 1.0).abs() < 1e-12);
```

---

# 📈 2. Annualized Sharpe Ratio (`annualized_sharpe_ratio`)

## Definition

Scales the period Sharpe ratio to an annual basis by multiplying by the square root of the number of periods per year. Requires `periods_per_year > 0`; otherwise returns `QuantError::InvalidValue`.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}S_%7B%5Ctext%7Bann%7D%7D%3DS%5Ccdot%5Csqrt%7BT%7D" alt="S_ann = S · √T" /></p>

Where:

- `S` is the period Sharpe ratio  
- `T` is the number of periods per year (e.g. 252 for daily, 12 for monthly, 52 for weekly)  

## Intuition

Because volatility scales with `√T` and mean return scales with `T`, the Sharpe ratio itself scales with `√T` when returns are independent and identically distributed. Multiplying by `√T` converts a period Sharpe into an annualized figure, making it comparable across strategies sampled at different frequencies.

## Example

Returns (daily):

[0.1, 0.2, 0.3]

Period Sharpe (risk-free = 0.0) = 2.0

Annualized (252 trading days): 2.0 × √252 ≈ **31.75**

## Code Example

```rust
use quantix::metrics::sharpe::{sharpe_ratio, annualized_sharpe_ratio};

let returns = vec![0.1, 0.2, 0.3];
let sharpe = sharpe_ratio(&returns, 0.0).unwrap();
let ann = annualized_sharpe_ratio(&returns, 0.0, 252.0).unwrap();
let expected = sharpe * 252.0_f64.sqrt();
assert!((ann - expected).abs() < 1e-12);
```
