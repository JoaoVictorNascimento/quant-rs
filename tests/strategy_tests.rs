use quantix::core::QuantError;
use quantix::strategy::moving_average::{moving_average_crossover_signals, simple_moving_average};
use quantix::strategy::signal::Signal;

fn assert_approx_eq(a: f64, b: f64) {
    let eps = 1e-12_f64.max(a.abs().max(b.abs()) * 1e-12);
    assert!(
        (a - b).abs() < eps,
        "expected {a} ≈ {b}, diff {}",
        (a - b).abs()
    );
}

// ── signal_enum ───────────────────────────────────────────────────────────────

#[test]
fn signal_variants_are_comparable() {
    assert_eq!(Signal::Buy, Signal::Buy);
    assert_eq!(Signal::Sell, Signal::Sell);
    assert_eq!(Signal::Hold, Signal::Hold);
    assert_ne!(Signal::Buy, Signal::Sell);
    assert_ne!(Signal::Buy, Signal::Hold);
    assert_ne!(Signal::Sell, Signal::Hold);
}

// ── simple_moving_average ─────────────────────────────────────────────────────

#[test]
fn simple_moving_average_window_one_equals_prices() {
    let prices = [100.0, 110.0, 120.0];
    let sma = simple_moving_average(&prices, 1).unwrap();

    assert_eq!(sma.len(), 3);
    assert_approx_eq(sma[0].unwrap(), 100.0);
    assert_approx_eq(sma[1].unwrap(), 110.0);
    assert_approx_eq(sma[2].unwrap(), 120.0);
}

#[test]
fn simple_moving_average_basic_window_two() {
    let prices = [100.0, 110.0, 120.0];
    let sma = simple_moving_average(&prices, 2).unwrap();

    assert_eq!(sma.len(), 3);
    assert_eq!(sma[0], None);
    assert_approx_eq(sma[1].unwrap(), 105.0);
    assert_approx_eq(sma[2].unwrap(), 115.0);
}

#[test]
fn simple_moving_average_window_equals_series_length() {
    let prices = [100.0, 110.0, 120.0];
    let sma = simple_moving_average(&prices, 3).unwrap();

    assert_eq!(sma[0], None);
    assert_eq!(sma[1], None);
    assert_approx_eq(sma[2].unwrap(), 110.0);
}

#[test]
fn simple_moving_average_window_larger_than_series() {
    let prices = [100.0, 110.0, 120.0];
    let sma = simple_moving_average(&prices, 5).unwrap();
    assert_eq!(sma, vec![None, None, None]);
}

#[test]
fn simple_moving_average_constant_series() {
    let prices = [100.0, 100.0, 100.0, 100.0];
    let sma = simple_moving_average(&prices, 3).unwrap();

    assert_eq!(sma[0], None);
    assert_eq!(sma[1], None);
    assert_approx_eq(sma[2].unwrap(), 100.0);
    assert_approx_eq(sma[3].unwrap(), 100.0);
}

#[test]
fn simple_moving_average_empty_prices_is_error() {
    assert!(matches!(
        simple_moving_average(&[], 2),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn simple_moving_average_zero_window_is_error() {
    assert!(matches!(
        simple_moving_average(&[100.0, 110.0], 0),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn simple_moving_average_non_positive_price_is_error() {
    assert!(matches!(
        simple_moving_average(&[100.0, 0.0], 2),
        Err(QuantError::NonPositivePrice(_))
    ));
    assert!(matches!(
        simple_moving_average(&[100.0, -1.0], 2),
        Err(QuantError::NonPositivePrice(_))
    ));
}

#[test]
fn simple_moving_average_invalid_value_is_error() {
    assert!(matches!(
        simple_moving_average(&[100.0, f64::NAN], 2),
        Err(QuantError::InvalidValue(_))
    ));
    assert!(matches!(
        simple_moving_average(&[100.0, f64::INFINITY], 2),
        Err(QuantError::InvalidValue(_))
    ));
}

// ── moving_average_crossover_signals ─────────────────────────────────────────

#[test]
fn moving_average_crossover_buy_signal_when_short_above_long() {
    let prices = [100.0, 101.0, 102.0, 103.0];
    let signals = moving_average_crossover_signals(&prices, 2, 3).unwrap();
    assert_eq!(signals.len(), 4);
    assert_eq!(signals[0], Signal::Hold);
    assert_eq!(signals[1], Signal::Hold);
    assert_eq!(signals[2], Signal::Buy);
    assert_eq!(signals[3], Signal::Buy);
}

#[test]
fn moving_average_crossover_sell_signal_when_short_below_long() {
    let prices = [103.0, 102.0, 101.0, 100.0];
    let signals = moving_average_crossover_signals(&prices, 2, 3).unwrap();
    assert_eq!(signals.len(), 4);
    assert_eq!(signals[0], Signal::Hold);
    assert_eq!(signals[1], Signal::Hold);
    assert_eq!(signals[2], Signal::Sell);
    assert_eq!(signals[3], Signal::Sell);
}

#[test]
fn moving_average_crossover_hold_when_equal_averages() {
    let prices = [100.0, 100.0, 100.0, 100.0];
    let signals = moving_average_crossover_signals(&prices, 2, 3).unwrap();
    assert_eq!(
        signals,
        vec![Signal::Hold, Signal::Hold, Signal::Hold, Signal::Hold]
    );
}

#[test]
fn moving_average_crossover_hold_until_long_window_is_ready() {
    let prices = [100.0, 101.0, 102.0, 103.0, 104.0];
    let signals = moving_average_crossover_signals(&prices, 2, 4).unwrap();
    assert_eq!(signals[0], Signal::Hold);
    assert_eq!(signals[1], Signal::Hold);
    assert_eq!(signals[2], Signal::Hold);
    assert_eq!(signals[3], Signal::Buy);
    assert_eq!(signals[4], Signal::Buy);
}

#[test]
fn moving_average_crossover_empty_prices_is_error() {
    assert!(matches!(
        moving_average_crossover_signals(&[], 2, 3),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn moving_average_crossover_zero_short_window_is_error() {
    assert!(matches!(
        moving_average_crossover_signals(&[100.0, 101.0], 0, 2),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn moving_average_crossover_zero_long_window_is_error() {
    assert!(matches!(
        moving_average_crossover_signals(&[100.0, 101.0], 1, 0),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn moving_average_crossover_short_greater_or_equal_long_is_error() {
    assert!(matches!(
        moving_average_crossover_signals(&[100.0, 101.0, 102.0], 3, 3),
        Err(QuantError::InvalidValue(_))
    ));
    assert!(matches!(
        moving_average_crossover_signals(&[100.0, 101.0, 102.0], 4, 3),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn moving_average_crossover_propagates_price_validation_errors() {
    assert!(matches!(
        moving_average_crossover_signals(&[100.0, 0.0, 102.0], 1, 2),
        Err(QuantError::NonPositivePrice(_))
    ));
    assert!(matches!(
        moving_average_crossover_signals(&[100.0, f64::NAN, 102.0], 1, 2),
        Err(QuantError::InvalidValue(_))
    ));
}
