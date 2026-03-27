# Returns

This module provides fundamental functions to compute financial returns from price series.  
Returns are the foundation of most quantitative finance metrics, such as volatility, Sharpe ratio, drawdown, and portfolio performance.

Price-based functions (`simple_returns`, `log_returns`, `cumulative_return`, `cumulative_log_return`) require at least two strictly positive finite prices; invalid inputs return `QuantError` (see `validate_prices` in the core module).

---

# 📈 1. Simple Returns (`simple_returns`)

## Definition

Simple return measures the percentage change between two consecutive prices. The function returns one value per adjacent pair in order.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}R_t%3D%5Cfrac%7BP_t%7D%7BP_%7Bt-1%7D%7D-1" alt="R_t = P_t / P_{t-1} - 1" /></p>

Where:

- `P_t` is the current price  
- `P_{t-1}` is the previous price  

## Intuition

It represents the direct percentage gain or loss between two time periods. Compounding several simple returns is done by converting to gross returns (1 + R) and multiplying, not by summing the simple returns.

## Example

Prices:

[100, 110, 121]

Returns:

[0.10, 0.10]

## Code Example

```rust
use quant_rs::metrics::returns::simple_returns;

let prices = vec![100.0, 110.0, 121.0];
let r = simple_returns(&prices).unwrap();
assert!((r[0] - 0.1).abs() < 1e-12);
assert!((r[1] - 0.1).abs() < 1e-12);
```

---

# 📉 2. Log Returns (`log_returns`)

## Definition

Log return (continuously compounded return) is the natural logarithm of the price ratio between consecutive observations. Like simple returns, the output has length `prices.len() - 1`.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}r_t%3D%5Cln%5Cfrac%7BP_t%7D%7BP_%7Bt-1%7D%7D" alt="r_t = ln(P_t / P_{t-1})" /></p>

Equivalently: **ln(P<sub>t</sub>) − ln(P<sub>t−1</sub>)** (same expression).

Where:

- `P_t` and `P_{t-1}` are consecutive positive prices  

## Intuition

Log returns are symmetric in log space and add over time: the log return over multiple steps equals the sum of single-period log returns (when using the same compounding interpretation). They are often used in statistics because sums of returns are easier to model than products.

## Example

Prices:

[100, 110]

Log returns:

[ln(1.1)] ≈ [0.0953]

## Code Example

```rust
use quant_rs::metrics::returns::log_returns;

let prices = vec![100.0, 110.0];
let r = log_returns(&prices).unwrap();
let expected = (110.0_f64 / 100.0).ln();
assert!((r[0] - expected).abs() < 1e-12);
```

---

# 📊 3. Cumulative Return (`cumulative_return`)

## Definition

Cumulative simple return from the first price in the series to the last, using only the endpoints (not the path).

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}R%3D%5Cfrac%7BP_n%7D%7BP_0%7D-1" alt="Cumulative simple return" /></p>

Where:

- `P_0` is the first price  
- `P_n` is the last price  

## Intuition

It answers: “If I bought at the first observation and sold at the last, what was my total percentage gain or loss?” Intermediate prices do not affect this number.

## Example

Prices:

[100, 110, 121]

Cumulative return:

(121 / 100) − 1 = 0.21

## Code Example

```rust
use quant_rs::metrics::returns::cumulative_return;

let prices = vec![100.0, 110.0, 121.0];
let r = cumulative_return(&prices).unwrap();
assert!((r - 0.21).abs() < 1e-12);
```

---

# 📐 4. Cumulative Log Return (`cumulative_log_return`)

## Definition

Natural logarithm of the total growth factor from the first to the last price—equivalent to the sum of one-period log returns along the path (for the same price series).

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}r%3D%5Cln%5Cfrac%7BP_n%7D%7BP_0%7D" alt="Cumulative log return" /></p>

Where:

- `P_0` is the first price  
- `P_n` is the last price  

## Intuition

This is the log counterpart of cumulative simple return: ln(1 + R) when R is the cumulative simple return over the whole window (exactly: ln(Pₙ/P₀)).

## Example

Prices:

[100, 121]

Cumulative log return:

ln(121 / 100) = ln(1.21)

## Code Example

```rust
use quant_rs::metrics::returns::cumulative_log_return;

let prices = vec![100.0, 121.0];
let r = cumulative_log_return(&prices).unwrap();
let expected = (121.0_f64 / 100.0).ln();
assert!((r - expected).abs() < 1e-12);
```

---

# 🔗 5. Cumulative Return from Period Returns (`cumulative_from_returns`)

## Definition

Chains period simple returns by compounding: multiply gross returns (1 + Rᵢ) for each period, then subtract 1 to get total simple return. Input is a slice of already-computed simple returns, not prices.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}R%3D%5Cprod_%7Bi%7D%5Cleft(1%2BR_i%5Cright)-1" alt="Compounded return from period returns" /></p>

Where:

- `R_i` is the simple return in period *i*  

## Intuition

This is how you aggregate sequential percentage returns: each period you grow wealth by (1 + Rᵢ). The function rejects empty slices and non-finite values.

## Example

Period returns:

[0.10, 0.10]

Cumulative:

(1.10 × 1.10) − 1 = 0.21

## Code Example

```rust
use quant_rs::metrics::returns::cumulative_from_returns;

let per_period = [0.1, 0.1];
let total = cumulative_from_returns(&per_period).unwrap();
assert!((total - 0.21).abs() < 1e-12);
```
