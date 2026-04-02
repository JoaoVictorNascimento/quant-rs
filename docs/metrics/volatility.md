# Volatility

This module provides functions to measure the dispersion of financial returns — commonly referred to as **volatility** in quantitative finance.  
Volatility quantifies how much returns fluctuate around their mean and is a core building block for risk metrics such as Sharpe ratio, Value-at-Risk, and position sizing.

All functions require at least two finite return values; invalid inputs return `QuantError` (`InsufficientData` or `InvalidValue`).

---

# 📊 1. Variance (`variance`)

## Definition

Sample variance of a return series, using Bessel's correction (denominator `n − 1`) to produce an unbiased estimate of the population variance.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}s%5E2%3D%5Cfrac%7B1%7D%7Bn-1%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28r_i-%5Cbar%7Br%7D%29%5E2" alt="s² = (1 / (n−1)) * Σ(rᵢ − r̄)²" /></p>

Where:

- `rᵢ` is the return at period *i*  
- `r̄` is the arithmetic mean of all returns  
- `n` is the number of observations  

## Intuition

Variance measures the average squared deviation from the mean. The `n − 1` denominator corrects for the fact that the sample mean is estimated from the same data, slightly underestimating the true spread if `n` were used.

## Example

Returns:

[1.0, 2.0, 3.0]

Mean = 2.0, sum of squared deviations = (1−2)² + (2−2)² + (3−2)² = 2.0

Variance = 2.0 / (3 − 1) = **1.0**

## Code Example

```rust
use quantix::metrics::volatility::variance;

let returns = vec![1.0, 2.0, 3.0];
let var = variance(&returns).unwrap();
assert!((var - 1.0).abs() < 1e-12);
```

---

# 📉 2. Volatility (`volatility`)

## Definition

Volatility is the standard deviation of a return series — the square root of the sample variance. It expresses dispersion in the same unit as the returns themselves.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}%5Csigma%3D%5Csqrt%7Bs%5E2%7D" alt="σ = √s²" /></p>

Where:

- `s²` is the sample variance (computed with `n − 1`)  

## Intuition

While variance penalises large deviations quadratically, volatility brings the measure back to the scale of the original returns. A daily volatility of 0.01, for example, means returns typically deviate about 1 % from their mean each day.

## Example

Returns:

[1.0, 2.0, 3.0]

Variance = 1.0 → Volatility = √1.0 = **1.0**

## Code Example

```rust
use quantix::metrics::volatility::volatility;

let returns = vec![1.0, 2.0, 3.0];
let vol = volatility(&returns).unwrap();
assert!((vol - 1.0).abs() < 1e-12);
```

---

# 📈 3. Annualized Volatility (`annualized_volatility`)

## Definition

Scales period volatility to an annual basis by multiplying by the square root of the number of periods per year. Requires `periods_per_year > 0`; otherwise returns `QuantError::InvalidValue`.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}%5Csigma_%7B%5Ctext%7Bann%7D%7D%3D%5Csigma%5Ccdot%5Csqrt%7BT%7D" alt="σ_ann = σ · √T" /></p>

Where:

- `σ` is the period volatility  
- `T` is the number of periods per year (e.g. 252 for daily, 12 for monthly, 52 for weekly)  

## Intuition

Under the assumption that returns are independent and identically distributed, variance scales linearly with time and standard deviation scales with the square root of time. Multiplying by `√T` converts period volatility to the equivalent annual figure, making it comparable across strategies with different sampling frequencies.

## Example

Returns (daily):

[0.1, −0.1]

Mean = 0.0, variance = 0.02, volatility = √0.02 ≈ 0.14142

Annualized (252 trading days): 0.14142 × √252 ≈ **2.245**

## Code Example

```rust
use quantix::metrics::volatility::{volatility, annualized_volatility};

let returns = vec![0.1, -0.1];
let vol = volatility(&returns).unwrap();
let ann = annualized_volatility(&returns, 252.0).unwrap();
let expected = vol * 252.0_f64.sqrt();
assert!((ann - expected).abs() < 1e-12);
```
