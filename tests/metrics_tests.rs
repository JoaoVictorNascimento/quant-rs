use quant_rs::core::QuantError;
use quant_rs::metrics::returns::{
    cumulative_from_returns, cumulative_log_return, cumulative_return, log_returns, simple_returns,
};

fn assert_approx_eq(a: f64, b: f64) {
    let eps = 1e-12_f64.max(a.abs().max(b.abs()) * 1e-12);
    assert!(
        (a - b).abs() < eps,
        "expected {a} ≈ {b}, diff {}",
        (a - b).abs()
    );
}

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

#[test]
fn cumulative_return_basic() {
    assert_approx_eq(cumulative_return(&[100.0, 120.0]).unwrap(), 0.2);
}

#[test]
fn cumulative_log_return_basic() {
    assert_approx_eq(
        cumulative_log_return(&[100.0, 120.0]).unwrap(),
        (120.0_f64 / 100.0).ln(),
    );
}

#[test]
fn cumulative_return_single_step_equals_simple() {
    let p = [10.0, 12.0];
    let cr = cumulative_return(&p).unwrap();
    let sr = simple_returns(&p).unwrap();
    assert_approx_eq(cr, sr[0]);
}

#[test]
fn cumulative_from_returns_basic() {
    assert_approx_eq(cumulative_from_returns(&[0.1, 0.1]).unwrap(), 0.21);
}

#[test]
fn cumulative_from_returns_single() {
    assert_approx_eq(cumulative_from_returns(&[0.05]).unwrap(), 0.05);
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
fn cumulative_log_equals_sum_log_returns() {
    let prices = [100.0, 110.0, 121.0];

    let clr = cumulative_log_return(&prices).unwrap();
    let lr = log_returns(&prices).unwrap();
    let sum: f64 = lr.iter().sum();

    assert_approx_eq(clr, sum);
}