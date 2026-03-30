# Drawdown

This module provides functions to measure **drawdown** — the decline from a historical peak in a price series.  
Drawdown metrics quantify downside risk and are widely used to evaluate the resilience of a strategy, define risk limits, and compare portfolio behaviour under adverse conditions.

All functions operate on **price series** (not returns) and require at least one strictly positive finite price; invalid inputs return `QuantError` (`InsufficientData`, `InvalidValue`, or `NonPositivePrice`).

---

# 📉 1. Drawdowns (`drawdowns`)

## Definition

Computes the drawdown at every point in the price series relative to the highest price observed up to and including that point (the running peak). The output has the same length as the input.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}DD_t%3D%5Cfrac%7BP_t%7D%7BP_t%5E%7B*%7D%7D-1%2C%5Cquad%20P_t%5E%7B*%7D%3D%5Cmax_%7Bs%5Cle%20t%7DP_s" alt="DD_t = P_t / P_t* − 1, P_t* = max{s ≤ t} P_s" /></p>

Where:

- `P_t` is the price at time *t*  
- `P_t*` is the running peak — the maximum price seen from the start up to time *t*  

## Intuition

A drawdown of −0.20 at time *t* means the price is 20 % below its highest point so far. The first element is always 0.0 because the first price is, by definition, both the current price and the running peak. Whenever a new all-time high is reached, the drawdown resets to 0.0.

## Example

Prices:

[100, 80, 120, 90]

| t | Price | Peak | Drawdown |
|---|-------|------|----------|
| 0 | 100   | 100  | 0.0      |
| 1 | 80    | 100  | −0.20    |
| 2 | 120   | 120  | 0.0      |
| 3 | 90    | 120  | −0.25    |

## Code Example

```rust
use quant_rs::metrics::drawdown::drawdowns;

let prices = vec![100.0, 80.0, 120.0, 90.0];
let dd = drawdowns(&prices).unwrap();

assert!((dd[0] - 0.0).abs() < 1e-12);
assert!((dd[1] - (80.0 / 100.0 - 1.0)).abs() < 1e-12); // -0.20
assert!((dd[2] - 0.0).abs() < 1e-12);
assert!((dd[3] - (90.0 / 120.0 - 1.0)).abs() < 1e-12); // -0.25
```

---

# 📊 2. Maximum Drawdown (`max_drawdown`)

## Definition

The single worst (most negative) drawdown value over the entire price series — the largest percentage decline from any peak to any subsequent trough.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}%5Ctext%7BMDD%7D%3D%5Cmin_t%5C%2CDD_t" alt="MDD = min_t DD_t" /></p>

Where:

- `DD_t` is the drawdown at time *t* as defined above  

## Intuition

Maximum drawdown answers: "What is the worst peak-to-trough loss an investor would have experienced holding this asset throughout the series?" It is always ≤ 0.0; a value of 0.0 means prices never fell below their previous peak. The function is equivalent to taking the minimum of the full drawdown series returned by `drawdowns`.

## Example

Prices:

[100, 80, 90, 60]

Running peak stays at 100. Drawdowns: [0.0, −0.20, −0.10, −0.40]

MDD = min([0.0, −0.20, −0.10, −0.40]) = **−0.40**

## Code Example

```rust
use quant_rs::metrics::drawdown::max_drawdown;

let prices = vec![100.0, 80.0, 90.0, 60.0];
let mdd = max_drawdown(&prices).unwrap();
let expected = 60.0_f64 / 100.0 - 1.0; // -0.40
assert!((mdd - expected).abs() < 1e-12);
```

---

# ⏱️ 3. Maximum Drawdown Duration (`max_drawdown_duration`)

## Definition

The longest consecutive run of periods in which the price remains strictly below its running peak. Returns 0 when the price never falls below its peak. The counter resets as soon as the price reaches or exceeds the current peak.

## Mathematical Formula

The duration is a counting quantity rather than a ratio. Informally:

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}%5Ctext%7BMDD-dur%7D%3D%5Cmax%5C%2C%5Cleft%7C%5C%7Bt%5Cin%5B a%2Cb%5D%3AP_t%3CP_t%5E%7B*%7D%5C%7D%5Cright%7C" alt="MDD-dur = max length of a consecutive run where P_t < P_t*" /></p>

Where each consecutive run is a maximal interval [a, b] such that `P_t < P_t*` for all *t* in that interval.

## Intuition

Duration measures how long a drawdown episode lasts before a full recovery. A strategy with a large maximum drawdown but short duration recovers quickly; the same drawdown with a long duration implies the investor was underwater for an extended period — often considered a more painful form of risk.

## Example

Prices:

[100, 90, 80, 70]

The price never recovers above the initial peak of 100. Consecutive periods below the peak: 3

MDD Duration = **3**

For a series with two separate drawdowns:

[100, 90, 100, 80, 70] → first drawdown lasts 1 period; second lasts 2 periods → MDD Duration = **2**

## Code Example

```rust
use quant_rs::metrics::drawdown::max_drawdown_duration;

// Monotonically decreasing: 3 consecutive periods below peak
let prices = vec![100.0, 90.0, 80.0, 70.0];
assert_eq!(max_drawdown_duration(&prices).unwrap(), 3);

// Two separate drawdowns; longest is 2 periods
let prices2 = vec![100.0, 90.0, 100.0, 80.0, 70.0];
assert_eq!(max_drawdown_duration(&prices2).unwrap(), 2);
```
