# Backtest Results

This module provides convenience analytics over a completed backtest path through the `BacktestResult` type.  
It centralizes strategy-return statistics and equity-curve risk metrics in one place.

`BacktestResult` stores:

- `strategy_returns`: realized returns per period
- `equity_curve`: cumulative wealth path
- `final_equity`: terminal wealth

Most analytics delegate to existing metric modules (`returns`, `volatility`, `sharpe`, `drawdown`) and therefore inherit their validation/error behavior.

---

# 🧾 1. Result Shape Helpers (`len`, `is_empty`)

## Definition

- `len()` returns the number of strategy return observations.
- `is_empty()` checks whether `strategy_returns` is empty.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}len%20%3D%20%7Cstrategy\_returns%7C,\quad%20is\_empty%20%3D%20(len%20%3D%3D%200)" alt="len = |strategy_returns|, is_empty = (len == 0)" /></p>

## Intuition

These methods are quick structural checks used before computing statistics that require minimum sample size.

## Example

If `strategy_returns = [0.1, -0.05, 0.02]`, then:

- `len() = 3`
- `is_empty() = false`

## Code Example

```rust
use quantix::backtest::results::BacktestResult;

let result = BacktestResult {
    strategy_returns: vec![0.1, -0.05, 0.02],
    equity_curve: vec![1.0, 1.1, 1.045, 1.0659],
    final_equity: 1.0659,
};

assert_eq!(result.len(), 3);
assert!(!result.is_empty());
```

---

# 📈 2. Cumulative Return (`cumulative_return`)

## Definition

Computes total compounded return from the stored `strategy_returns` by chaining gross returns `(1 + r_t)` and subtracting 1.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}R_{cum}%3D%5Cprod_t(1%2Br_t)-1" alt="R_cum = Π(1+r_t) - 1" /></p>

Where:

- `r_t` is the strategy return at period *t*  

## Intuition

This is path-consistent performance aggregation: returns compound multiplicatively, not additively.

## Example

`strategy_returns = [0.1, -0.05, 0.02]`

`R_cum = (1.1 * 0.95 * 1.02) - 1 = 0.0659` (6.59%)

## Code Example

```rust
use quantix::backtest::results::BacktestResult;

let result = BacktestResult {
    strategy_returns: vec![0.1, -0.05, 0.02],
    equity_curve: vec![1.0, 1.1, 1.045, 1.0659],
    final_equity: 1.0659,
};

let cum = result.cumulative_return().unwrap();
assert!((cum - 0.0659).abs() < 1e-12);
```

---

# 📊 3. Volatility (`volatility`)

## Definition

Computes sample volatility (standard deviation) of `strategy_returns`.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}\sigma%3D%5Csqrt%7B%5Cfrac%7B1%7D%7Bn-1%7D%5Csum_t(r_t-%5Cbar%7Br%7D)^2%7D" alt="σ = sqrt((1/(n-1)) Σ(r_t-r̄)^2)" /></p>

Where:

- `n` is the number of strategy returns  
- `r̄` is the mean return  

## Intuition

Volatility measures dispersion of realized strategy returns around their mean; higher values imply higher realized variability/risk.

## Example

For constant returns `[1.0, 1.0, 1.0]`, volatility is exactly `0.0`.

## Code Example

```rust
use quantix::backtest::results::BacktestResult;

let result = BacktestResult {
    strategy_returns: vec![1.0, 1.0, 1.0],
    equity_curve: vec![1.0, 2.0, 4.0, 8.0],
    final_equity: 8.0,
};

assert!((result.volatility().unwrap() - 0.0).abs() < 1e-12);
```

---

# ⚖️ 4. Sharpe Ratio (`sharpe_ratio`)

## Definition

Computes Sharpe ratio of `strategy_returns` for a given period risk-free rate.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}S%3D%5Cfrac{%5Cbar%7Br%7D-r_f}{%5Csigma}" alt="S = (r̄-r_f)/σ" /></p>

Where:

- `r̄` is mean strategy return  
- `r_f` is risk-free rate per period  
- `σ` is strategy return volatility  

## Intuition

Sharpe normalizes excess return by total variability, enabling risk-adjusted comparison across strategies.

## Example

With non-constant returns, the ratio is finite; with constant returns, `σ = 0` and the method returns `QuantError::DivisionByZero`.

## Code Example

```rust
use quantix::backtest::results::BacktestResult;

let result = BacktestResult {
    strategy_returns: vec![0.1, -0.05, 0.02],
    equity_curve: vec![1.0, 1.1, 1.045, 1.0659],
    final_equity: 1.0659,
};

let s = result.sharpe_ratio(0.0).unwrap();
assert!(s.is_finite());
```

---

# 📉 5. Maximum Drawdown (`max_drawdown`)

## Definition

Computes maximum drawdown directly from `equity_curve`, i.e., the worst peak-to-trough decline over the path.

## Mathematical Formula

Drawdown at time *t*:

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}DD_t%3D%5Cfrac%7BE_t%7D%7B%5Cmax_%7Bs%5Cle%20t%7DE_s%7D-1" alt="DD_t = E_t / max_{s<=t}E_s - 1" /></p>

Maximum drawdown:

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}MDD%3D%5Cmin_t%20DD_t" alt="MDD = min_t DD_t" /></p>

Where:

- `E_t` is equity at time *t*  

## Intuition

This metric captures the deepest loss from any historical equity peak, a key downside-risk measure often more intuitive than volatility for practitioners.

## Example

`equity_curve = [1.0, 1.2, 1.1, 1.3]`

Worst drawdown occurs at `1.1` relative to prior peak `1.2`:

`MDD = 1.1/1.2 - 1 = -0.08333...`

## Code Example

```rust
use quantix::backtest::results::BacktestResult;

let result = BacktestResult {
    strategy_returns: vec![0.0, 0.0, 0.0],
    equity_curve: vec![1.0, 1.2, 1.1, 1.3],
    final_equity: 1.3,
};

let mdd = result.max_drawdown().unwrap();
let expected = 1.1_f64 / 1.2 - 1.0;
assert!((mdd - expected).abs() < 1e-12);
```
