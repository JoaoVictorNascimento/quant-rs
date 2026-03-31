# Portfolio

This module provides a simple portfolio model based on weighted positions and period return series.  
It allows you to aggregate asset-level returns into portfolio returns, inspect total allocation weight, and normalize weights to sum to 1.0.

The main types are:

- `Position { weight, returns }` for each asset/strategy sleeve  
- `Portfolio { positions }` for the full allocation set  

For `returns()`, the portfolio must contain at least one position, each position must have the same non-empty return series length, and all weights/returns must be finite values; otherwise it returns `QuantError`.

---

# 📊 1. Portfolio Returns (`Portfolio::returns`)

## Definition

Computes the portfolio return series as the weighted sum of each position return at every time index.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}R_{p,t}%3D%5Csum_{i%3D1}^{N}w_i%20r_{i,t}" alt="R_{p,t} = Σ w_i r_{i,t}" /></p>

Where:

- `R_{p,t}` is the portfolio return at period *t*  
- `w_i` is the weight of position *i*  
- `r_{i,t}` is the return of position *i* at period *t*  
- `N` is the number of positions  

## Intuition

Each period, the portfolio return is just a linear combination of component returns. Long positions contribute positively with positive weights, while short positions (negative weights) can increase or decrease portfolio return depending on asset direction.

## Example

Positions:

- `w1 = 0.6`, returns = [0.01, 0.02, -0.01]
- `w2 = 0.4`, returns = [0.03, -0.01, 0.02]

Portfolio returns:

- t0: `0.6*0.01 + 0.4*0.03 = 0.018`
- t1: `0.6*0.02 + 0.4*(-0.01) = 0.008`
- t2: `0.6*(-0.01) + 0.4*0.02 = 0.002`

Result: **[0.018, 0.008, 0.002]**

## Code Example

```rust
use quant_rs::portfolio::portfolio::{Portfolio, Position};

let portfolio = Portfolio {
    positions: vec![
        Position {
            weight: 0.6,
            returns: vec![0.01, 0.02, -0.01],
        },
        Position {
            weight: 0.4,
            returns: vec![0.03, -0.01, 0.02],
        },
    ],
};

let r = portfolio.returns().unwrap();
assert!((r[0] - 0.018).abs() < 1e-12);
assert!((r[1] - 0.008).abs() < 1e-12);
assert!((r[2] - 0.002).abs() < 1e-12);
```

---

# ⚖️ 2. Sum of Weights (`Portfolio::weights_sum`)

## Definition

Returns the arithmetic sum of all position weights in the portfolio.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}W%3D%5Csum_{i%3D1}^{N}w_i" alt="W = Σ w_i" /></p>

Where:

- `w_i` is the weight of position *i*  
- `N` is the number of positions  

## Intuition

This value indicates gross net allocation in signed terms. A fully invested long-only portfolio typically has sum near 1.0; long-short portfolios may have sum different from 1.0 or even 0.0.

## Example

Weights: `[0.5, -0.5]`  

Sum: `0.5 + (-0.5) = 0.0`

## Code Example

```rust
use quant_rs::portfolio::portfolio::{Portfolio, Position};

let portfolio = Portfolio {
    positions: vec![
        Position { weight: 0.5, returns: vec![0.0] },
        Position { weight: -0.5, returns: vec![0.0] },
    ],
};

assert!((portfolio.weights_sum() - 0.0).abs() < 1e-12);
```

---

# 🔁 3. Normalize Weights (`Portfolio::normalize_weights`)

## Definition

Scales all position weights by their current sum so the new total becomes exactly 1.0.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}w_i'%3D%5Cfrac{w_i}{%5Csum_j%20w_j}" alt="w_i' = w_i / Σ w_j" /></p>

Where:

- `w_i` is the original weight of position *i*  
- `w_i'` is the normalized weight  

## Intuition

Normalization preserves relative proportions between positions while re-scaling the portfolio to a target net weight sum of 1.0. If the current sum is zero, normalization is impossible and returns `QuantError::DivisionByZero`.

## Example

Weights before: `[2.0, 3.0]` (sum = 5.0)  
Weights after: `[0.4, 0.6]` (sum = 1.0)

## Code Example

```rust
use quant_rs::portfolio::portfolio::{Portfolio, Position};

let mut portfolio = Portfolio {
    positions: vec![
        Position { weight: 2.0, returns: vec![0.01, 0.02] },
        Position { weight: 3.0, returns: vec![0.03, 0.04] },
    ],
};

portfolio.normalize_weights().unwrap();
assert!((portfolio.positions[0].weight - 0.4).abs() < 1e-12);
assert!((portfolio.positions[1].weight - 0.6).abs() < 1e-12);
assert!((portfolio.weights_sum() - 1.0).abs() < 1e-12);
```
