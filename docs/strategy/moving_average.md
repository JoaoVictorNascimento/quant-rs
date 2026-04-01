# Moving Average

This module provides moving-average based utilities for price series:  
`simple_moving_average` computes rolling arithmetic means, and `moving_average_crossover_signals` converts short-vs-long average relationships into trading signals (`Buy`, `Sell`, `Hold`).

All price-based functions require non-empty series with strictly positive finite prices; invalid inputs return `QuantError`.

---

# 📊 1. Simple Moving Average (`simple_moving_average`)

## Definition

Computes the Simple Moving Average (SMA) over a fixed window and returns a vector of `Option<f64>` with the same length as input prices.

- `None` for indices where there is not enough history (`i + 1 < window`)
- `Some(sma)` once the window is fully available

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}SMA_t^{(w)}%3D%5Cfrac%7B1%7D%7Bw%7D%5Csum_%7Bj%3D0%7D%5E%7Bw-1%7DP_%7Bt-j%7D,\quad%20t%20%5Cge%20w-1" alt="SMA_t^(w) = (1/w) Σ_{j=0}^{w-1} P_{t-j}, t >= w-1" /></p>

Where:

- `P_t` is the price at time *t*  
- `w` is the window size (`w > 0`)  

## Intuition

SMA smooths short-term noise by averaging recent prices. Smaller windows react faster to new movements; larger windows are smoother but slower.

## Example

Prices: `[100, 110, 120]`, window = 2

- t0: insufficient history → `None`
- t1: `(100 + 110)/2 = 105`
- t2: `(110 + 120)/2 = 115`

Result: `[None, Some(105), Some(115)]`

## Code Example

```rust
use quant_rs::strategy::moving_average::simple_moving_average;

let prices = [100.0, 110.0, 120.0];
let sma = simple_moving_average(&prices, 2).unwrap();

assert_eq!(sma[0], None);
assert!((sma[1].unwrap() - 105.0).abs() < 1e-12);
assert!((sma[2].unwrap() - 115.0).abs() < 1e-12);
```

---

# 🔀 2. Moving Average Crossover Signals (`moving_average_crossover_signals`)

## Definition

Generates one trading signal per price index from a short/long SMA crossover rule.

Rules at each time *t*:

- if both averages are available and `SMA_short > SMA_long` → `Signal::Buy`
- if both averages are available and `SMA_short < SMA_long` → `Signal::Sell`
- otherwise (including unavailable averages) → `Signal::Hold`

Constraints:

- `short_window > 0`
- `long_window > 0`
- `short_window < long_window`

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}s_t%3D%5Cbegin%7Bcases%7DBuy,%20&%20SMA_t^{(s)}%20%3E%20SMA_t^{(l)}%5C%5CSell,%20&%20SMA_t^{(s)}%20%3C%20SMA_t^{(l)}%5C%5CHold,%20&%20%5Ctext%7Botherwise%7D%5Cend%7Bcases%7D" alt="s_t = Buy if SMA_short > SMA_long, Sell if SMA_short < SMA_long, else Hold" /></p>

Where:

- `s` is the short window  
- `l` is the long window (`s < l`)  

## Intuition

When the short average rises above the long average, recent momentum is stronger than the broader trend (`Buy`). When it falls below, momentum weakens relative to trend (`Sell`). Before the long window is ready, the strategy remains neutral (`Hold`).

## Example

Prices: `[100, 101, 102, 103]`, short = 2, long = 3

- t0: long SMA unavailable → `Hold`
- t1: long SMA unavailable → `Hold`
- t2: short=101.5, long=101.0 → `Buy`
- t3: short=102.5, long=102.0 → `Buy`

Result: `[Hold, Hold, Buy, Buy]`

## Code Example

```rust
use quant_rs::strategy::moving_average::moving_average_crossover_signals;
use quant_rs::strategy::signal::Signal;

let prices = [100.0, 101.0, 102.0, 103.0];
let signals = moving_average_crossover_signals(&prices, 2, 3).unwrap();

assert_eq!(signals, vec![Signal::Hold, Signal::Hold, Signal::Buy, Signal::Buy]);
```
