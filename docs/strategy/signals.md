# Signals

This module defines the discrete trading actions used by strategy outputs.  
`Signal` is a compact enum for decision layers such as execution engines, backtests, and portfolio overlays.

---

# 🚦 1. Trading Signal Enum (`Signal`)

## Definition

`Signal` represents three mutually exclusive actions:

- `Buy` → bullish/long action  
- `Sell` → bearish/short or reduce exposure action  
- `Hold` → no trade / keep current exposure  

## Mathematical Mapping

Many quantitative workflows map categorical signals to signed actions:

<p align="center"><img src="https://latex.codecogs.com/svg.image?\color{white}a_t%20%5Cin%20%5C%7B%2B1,%200,%20-1%5C%7D,\quad%20Buy%20%5Cmapsto%20%2B1,\quad%20Hold%20%5Cmapsto%200,\quad%20Sell%20%5Cmapsto%20-1" alt="a_t ∈ {+1,0,-1}, Buy→+1, Hold→0, Sell→-1" /></p>

This numeric mapping is conceptual; the enum itself stores symbolic variants, not numbers.

## Intuition

Using a typed enum avoids ambiguous string/integer conventions and makes strategy intent explicit at compile time. It also improves testability through direct equality checks between expected and generated signals.

## Example

For a trend-following model:

- short MA above long MA → `Buy`
- short MA below long MA → `Sell`
- otherwise → `Hold`

## Code Example

```rust
use quantix::strategy::signal::Signal;

let enter = Signal::Buy;
let exit = Signal::Sell;
let wait = Signal::Hold;

assert_eq!(enter, Signal::Buy);
assert_ne!(enter, exit);
assert_eq!(wait, Signal::Hold);
```
