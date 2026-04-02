# Backtest Engine

This module provides a simple event-driven backtest engine for long-only exposure toggled by trading signals.  
Given a price series and one signal per timestamp, it computes strategy period returns, equity curve, and final equity.

The core output type is `BacktestResult { strategy_returns, equity_curve, final_equity }`.

Inputs must satisfy:

- `prices.len() >= 2`
- `signals.len() == prices.len()`
- all prices are finite and strictly positive

Otherwise, the function returns `QuantError` (`InsufficientData`, `InvalidValue`, or `NonPositivePrice`).

---

# 🧾 1. Backtest Result (`BacktestResult`)

## Definition

`BacktestResult` stores the full backtest path and terminal wealth:

- `strategy_returns`: one realized return per interval (`prices.len() - 1`)
- `equity_curve`: cumulative wealth path starting at 1.0 (`prices.len()`)
- `final_equity`: last value of `equity_curve`

## Mathematical Relationship

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}E_0%3D1,\quad%20E_t%3DE_{t-1}(1%2Br_t),\quad%20final\_equity%3DE_{T}" alt="E_0 = 1, E_t = E_{t-1}(1+r_t), final_equity = E_T" /></p>

Where:

- `r_t` is the strategy return in interval *t*  
- `E_t` is cumulative equity after interval *t*  

## Intuition

`strategy_returns` explains *what happened each period* while `equity_curve` shows *path-dependent compounding*. `final_equity` is the concise terminal performance number.

## Example

If strategy returns are `[0.10, -0.10, 0.10]`:

- `E0 = 1.0`
- `E1 = 1.0 * 1.10 = 1.10`
- `E2 = 1.10 * 0.90 = 0.99`
- `E3 = 0.99 * 1.10 = 1.089`

So `equity_curve = [1.0, 1.10, 0.99, 1.089]` and `final_equity = 1.089`.

---

# ▶️ 2. Run Backtest (`BacktestEngine::run`)

## Definition

Runs the backtest over adjacent price intervals, applying signals with one-step alignment:

- At interval `i` (from `prices[i-1]` to `prices[i]`), the engine first reads `signals[i-1]`.
- `Buy` sets `in_position = true`, `Sell` sets `in_position = false`, `Hold` keeps previous state.
- If in position, strategy captures the asset return for that interval; otherwise return is `0.0`.

## Mathematical Formula

Period return:

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}r_i%3D%5Cbegin%7Bcases%7D%5Cfrac%7BP_i%7D%7BP_%7Bi-1%7D%7D-1,&%20if%20in\_position_i%20%3D%20true%5C%5C0,&%20if%20in\_position_i%20%3D%20false%5Cend%7Bcases%7D" alt="r_i = P_i/P_{i-1}-1 if in_position else 0" /></p>

Signal state transition before interval *i*:

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}in\_position_i%3Df(signals_%7Bi-1%7D,\%20in\_position_%7Bi-1%7D)" alt="in_position_i = f(signal_{i-1}, in_position_{i-1})" /></p>

with:

- `Buy` → true
- `Sell` → false
- `Hold` → unchanged

And equity compounding:

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}E_i%3DE_%7Bi-1%7D(1%2Br_i)" alt="E_i = E_{i-1}(1+r_i)" /></p>

## Intuition

The engine models a binary exposure strategy (in/out of market). Signals do not create same-bar returns; they decide whether the *next* interval is invested. This avoids look-ahead bias and matches realistic execution timing in bar-based systems.

## Example

Prices: `[100, 90, 99, 108.9]`  
Signals: `[Buy, Sell, Buy, Hold]`

Interval logic:

- `i=1` uses `Buy` → in position → return `90/100 - 1 = -0.10`
- `i=2` uses `Sell` → out of market → return `0.00`
- `i=3` uses `Buy` → in position → return `108.9/99 - 1 = 0.10`

Strategy returns: `[-0.10, 0.00, 0.10]`  
Equity path: `[1.0, 0.9, 0.9, 0.99]`  
Final equity: `0.99`

## Code Example

```rust
use quantix::backtest::engine::BacktestEngine;
use quantix::strategy::signal::Signal;

let prices = [100.0, 90.0, 99.0, 108.9];
let signals = [Signal::Buy, Signal::Sell, Signal::Buy, Signal::Hold];

let result = BacktestEngine::run(&prices, &signals).unwrap();

assert_eq!(result.strategy_returns.len(), 3);
assert!((result.strategy_returns[0] - (-0.1)).abs() < 1e-12);
assert!((result.strategy_returns[1] - 0.0).abs() < 1e-12);
assert!((result.strategy_returns[2] - 0.1).abs() < 1e-12);

assert_eq!(result.equity_curve.len(), 4);
assert!((result.equity_curve[0] - 1.0).abs() < 1e-12);
assert!((result.equity_curve[1] - 0.9).abs() < 1e-12);
assert!((result.equity_curve[2] - 0.9).abs() < 1e-12);
assert!((result.equity_curve[3] - 0.99).abs() < 1e-12);
assert!((result.final_equity - 0.99).abs() < 1e-12);
```
