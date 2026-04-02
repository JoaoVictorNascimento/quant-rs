# Position

This module provides a `Position` type to represent a single allocation sleeve inside a portfolio: one weight and one time series of returns.  
It centralizes input validation and basic helpers used by portfolio construction and analytics.

The core struct is:

- `Position { weight, returns }`

`Position::new` validates that the weight is finite and the return series is non-empty with finite values only; invalid inputs return `QuantError`.

---

# 🧱 1. Position Constructor (`Position::new`)

## Definition

Creates a validated `Position` from a signed weight and a return series.

## Mathematical Formula

There is no transformation formula in the constructor; it enforces validity constraints:

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}w%20%5Cin%20%5Cmathbb{R}_{finite},%20%5Cquad%20n%20%3D%20%7Creturns%7C%20%3E%200,%20%5Cquad%20r_i%20%5Cin%20%5Cmathbb{R}_{finite}" alt="w finite, n = |returns| > 0, and every r_i finite" /></p>

Where:

- `w` is the position weight  
- `returns = [r_1, ..., r_n]` is the period return series  
- `n` is the series length, which must be strictly positive  

## Intuition

The constructor guarantees that any `Position` instance is numerically valid before it enters portfolio calculations. This prevents silent propagation of invalid values (`NaN`, `±∞`) and avoids undefined operations on empty series.

Weights can be:

- positive (long exposure)
- zero (inactive sleeve)
- negative (short exposure)

## Example

Valid:

- `weight = 0.5`, `returns = [0.01, -0.02, 0.03]` → `Ok(Position)`
- `weight = -1.0`, `returns = [0.01]` → `Ok(Position)` (short allowed)

Invalid:

- `weight = NaN` → `Err(QuantError::InvalidValue(_))`
- `returns = []` → `Err(QuantError::InsufficientData)`
- any non-finite return → `Err(QuantError::InvalidValue(_))`

## Code Example

```rust
use quantix::portfolio::position::Position;

let p = Position::new(0.5, vec![0.01, -0.02, 0.03]).unwrap();
assert!((p.weight - 0.5).abs() < 1e-12);
assert_eq!(p.returns.len(), 3);

let short = Position::new(-1.0, vec![0.01]).unwrap();
assert!((short.weight - (-1.0)).abs() < 1e-12);
```

---

# 📏 2. Returns Length (`Position::len`)

## Definition

Returns the number of observations in the position return series.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}len%20%3D%20%7Creturns%7C" alt="len = |returns|" /></p>

Where:

- `returns` is the stored return vector for the position  

## Intuition

`len()` is a lightweight helper for shape checks and alignment with other positions in the same portfolio.

## Example

If `returns = [0.01, 0.02, 0.03, 0.04]`, then `len() = 4`.

## Code Example

```rust
use quantix::portfolio::position::Position;

let p = Position::new(1.0, vec![0.01, 0.02, 0.03, 0.04]).unwrap();
assert_eq!(p.len(), 4);
```

---

# ✅ 3. Empty Check (`Position::is_empty`)

## Definition

Returns whether the position return series is empty.

## Mathematical Formula

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}is\\_empty%20%3D%20(len%20%3D%3D%200)" alt="is_empty = (len == 0)" /></p>

## Intuition

For positions created via `Position::new`, this always returns `false` because empty series are rejected at construction time. The method still exists as a standard collection-style convenience.

## Example

For `Position::new(1.0, vec![0.01])`, `is_empty()` returns `false`.

## Code Example

```rust
use quantix::portfolio::position::Position;

let p = Position::new(1.0, vec![0.01]).unwrap();
assert!(!p.is_empty());
```
