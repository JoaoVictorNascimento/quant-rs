use quant_rs::core::QuantError;
use quant_rs::metrics::returns::{
    cumulative_from_returns, cumulative_log_return, cumulative_return, log_returns, simple_returns,
};
use quant_rs::metrics::volatility::{annualized_volatility, variance, volatility};

fn assert_approx_eq(a: f64, b: f64) {
    let eps = 1e-12_f64.max(a.abs().max(b.abs()) * 1e-12);
    assert!(
        (a - b).abs() < eps,
        "expected {a} ≈ {b}, diff {}",
        (a - b).abs()
    );
}

// ── simple_returns ────────────────────────────────────────────────────────────

#[test]
fn simple_returns_two_prices() {
    let r = simple_returns(&[100.0, 110.0]).unwrap();
    assert_eq!(r.len(), 1);
    assert_approx_eq(r[0], 0.1);
}

#[test]
fn simple_returns_chain() {
    let r = simple_returns(&[100.0, 105.0, 110.25]).unwrap();
    assert_eq!(r.len(), 2);
    assert_approx_eq(r[0], 0.05);
    assert_approx_eq(r[1], 110.25 / 105.0 - 1.0);
}

#[test]
fn simple_returns_insufficient_data() {
    assert!(matches!(
        simple_returns(&[100.0]),
        Err(QuantError::InsufficientData)
    ));
    assert!(matches!(simple_returns(&[]), Err(QuantError::InsufficientData)));
}

#[test]
fn simple_returns_validation_errors() {
    assert!(matches!(
        simple_returns(&[100.0, 0.0]),
        Err(QuantError::ZeroPrice)
    ));
    assert!(matches!(
        simple_returns(&[100.0, -1.0]),
        Err(QuantError::NonPositivePrice(_))
    ));
    assert!(matches!(
        simple_returns(&[f64::NAN, 100.0]),
        Err(QuantError::InvalidValue(_))
    ));
}

// ── log_returns ───────────────────────────────────────────────────────────────

#[test]
fn log_returns_two_prices() {
    let r = log_returns(&[100.0, 110.0]).unwrap();
    assert_eq!(r.len(), 1);
    assert_approx_eq(r[0], (110.0_f64 / 100.0).ln());
}

#[test]
fn log_returns_matches_simple_identity() {
    let prices = [50.0, 75.0];
    let s = simple_returns(&prices).unwrap();
    let l = log_returns(&prices).unwrap();
    assert_approx_eq(l[0], (1.0 + s[0]).ln());
}

// ── cumulative_return ─────────────────────────────────────────────────────────

#[test]
fn cumulative_return_basic() {
    assert_approx_eq(cumulative_return(&[100.0, 120.0]).unwrap(), 0.2);
}

#[test]
fn cumulative_return_single_step_equals_simple() {
    let p = [10.0, 12.0];
    let cr = cumulative_return(&p).unwrap();
    let sr = simple_returns(&p).unwrap();
    assert_approx_eq(cr, sr[0]);
}

// ── cumulative_log_return ─────────────────────────────────────────────────────

#[test]
fn cumulative_log_return_basic() {
    assert_approx_eq(
        cumulative_log_return(&[100.0, 120.0]).unwrap(),
        (120.0_f64 / 100.0).ln(),
    );
}

#[test]
fn cumulative_log_equals_sum_log_returns() {
    let prices = [100.0, 110.0, 121.0];

    let clr = cumulative_log_return(&prices).unwrap();
    let lr = log_returns(&prices).unwrap();
    let sum: f64 = lr.iter().sum();

    assert_approx_eq(clr, sum);
}

// ── cumulative_from_returns ───────────────────────────────────────────────────

#[test]
fn cumulative_from_returns_basic() {
    assert_approx_eq(cumulative_from_returns(&[0.1, 0.1]).unwrap(), 0.21);
}

#[test]
fn cumulative_from_returns_single() {
    assert_approx_eq(cumulative_from_returns(&[0.05]).unwrap(), 0.05);
}

#[test]
fn cumulative_from_returns_full_loss() {
    let returns = [0.5, -1.0];
    let result = cumulative_from_returns(&returns).unwrap();

    assert_approx_eq(result, -1.0);
}

#[test]
fn cumulative_from_returns_negative() {
    let returns = [-0.2, 0.1];
    let result = cumulative_from_returns(&returns).unwrap();

    assert_approx_eq(result, -0.12);
}

#[test]
fn cumulative_from_returns_empty() {
    assert!(matches!(
        cumulative_from_returns(&[]),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn cumulative_from_returns_invalid() {
    assert!(matches!(
        cumulative_from_returns(&[0.1, f64::NAN]),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn cumulative_matches_returns_composition() {
    let prices = [100.0, 110.0, 121.0];

    let cr = cumulative_return(&prices).unwrap();
    let returns = simple_returns(&prices).unwrap();
    let composed = cumulative_from_returns(&returns).unwrap();

    assert_approx_eq(cr, composed);
}

// ── variance ─────────────────────────────────────────────────────────────────

#[test]
fn variance_two_equal_returns() {
    assert_approx_eq(variance(&[0.0, 0.0]).unwrap(), 0.0);
}

#[test]
fn variance_basic() {
    // mean = 2.0, sum_sq = 2.0, var = 2.0 / 2 = 1.0
    assert_approx_eq(variance(&[1.0, 2.0, 3.0]).unwrap(), 1.0);
}

#[test]
fn variance_two_symmetric() {
    // mean = 0.0, sum_sq = 0.02, var = 0.02 / 1 = 0.02
    assert_approx_eq(variance(&[0.1, -0.1]).unwrap(), 0.02);
}

#[test]
fn variance_insufficient_data() {
    assert!(matches!(variance(&[0.1]), Err(QuantError::InsufficientData)));
    assert!(matches!(variance(&[]), Err(QuantError::InsufficientData)));
}

#[test]
fn variance_invalid_values() {
    assert!(matches!(
        variance(&[0.1, f64::NAN]),
        Err(QuantError::InvalidValue(_))
    ));
    assert!(matches!(
        variance(&[f64::INFINITY, 0.1]),
        Err(QuantError::InvalidValue(_))
    ));
    assert!(matches!(
        variance(&[f64::NEG_INFINITY, 0.1]),
        Err(QuantError::InvalidValue(_))
    ));
}

// ── volatility ───────────────────────────────────────────────────────────────

#[test]
fn volatility_equals_sqrt_variance() {
    let returns = &[0.1, -0.1, 0.05, -0.05, 0.02];
    let var = variance(returns).unwrap();
    let vol = volatility(returns).unwrap();
    assert_approx_eq(vol, var.sqrt());
}

#[test]
fn volatility_constant_returns_is_zero() {
    assert_approx_eq(volatility(&[0.05, 0.05, 0.05]).unwrap(), 0.0);
}

#[test]
fn volatility_basic() {
    // var = 1.0 → vol = 1.0
    assert_approx_eq(volatility(&[1.0, 2.0, 3.0]).unwrap(), 1.0);
}

#[test]
fn volatility_insufficient_data() {
    assert!(matches!(
        volatility(&[0.1]),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn volatility_invalid_values() {
    assert!(matches!(
        volatility(&[0.1, f64::NAN]),
        Err(QuantError::InvalidValue(_))
    ));
}

// ── annualized_volatility ─────────────────────────────────────────────────────

#[test]
fn annualized_volatility_daily_to_annual() {
    let returns = &[0.01, -0.01, 0.02, -0.02, 0.005];
    let vol = volatility(returns).unwrap();
    let ann = annualized_volatility(returns, 252.0).unwrap();
    assert_approx_eq(ann, vol * 252.0_f64.sqrt());
}

#[test]
fn annualized_volatility_monthly_to_annual() {
    let returns = &[0.03, -0.02, 0.05, -0.01];
    let vol = volatility(returns).unwrap();
    let ann = annualized_volatility(returns, 12.0).unwrap();
    assert_approx_eq(ann, vol * 12.0_f64.sqrt());
}

#[test]
fn annualized_volatility_one_period_equals_volatility() {
    let returns = &[0.1, -0.1, 0.05];
    let vol = volatility(returns).unwrap();
    let ann = annualized_volatility(returns, 1.0).unwrap();
    assert_approx_eq(ann, vol);
}

#[test]
fn annualized_volatility_zero_periods_is_error() {
    assert!(matches!(
        annualized_volatility(&[0.1, -0.1], 0.0),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn annualized_volatility_negative_periods_is_error() {
    assert!(matches!(
        annualized_volatility(&[0.1, -0.1], -252.0),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn annualized_volatility_propagates_insufficient_data() {
    assert!(matches!(
        annualized_volatility(&[0.1], 252.0),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn annualized_volatility_propagates_invalid_values() {
    assert!(matches!(
        annualized_volatility(&[0.1, f64::NAN], 252.0),
        Err(QuantError::InvalidValue(_))
    ));
}