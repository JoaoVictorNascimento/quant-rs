use quant_rs::core::QuantError;
use quant_rs::metrics::returns::{
    cumulative_from_returns, cumulative_log_return, cumulative_return, log_returns, simple_returns,
};
use quant_rs::metrics::drawdown::{drawdowns, max_drawdown, max_drawdown_duration};
use quant_rs::metrics::sharpe::{annualized_sharpe_ratio, sharpe_ratio};
use quant_rs::metrics::sortino::sortino_ratio;
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
    assert_approx_eq(volatility(&[1.0, 1.0, 1.0]).unwrap(), 0.0);
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

// ── sharpe_ratio ──────────────────────────────────────────────────────────────

#[test]
fn sharpe_ratio_zero_risk_free() {
    assert_approx_eq(sharpe_ratio(&[0.1, 0.2, 0.3], 0.0).unwrap(), 2.0);
}

#[test]
fn sharpe_ratio_nonzero_risk_free() {
    assert_approx_eq(sharpe_ratio(&[0.1, 0.2, 0.3], 0.1).unwrap(), 1.0);
}

#[test]
fn sharpe_ratio_negative() {
    let r = &[0.01, 0.02, 0.03];
    let mean = 0.02_f64;
    let vol = volatility(r).unwrap();
    let expected = (mean - 0.1) / vol;
    assert_approx_eq(sharpe_ratio(r, 0.1).unwrap(), expected);
}

#[test]
fn sharpe_ratio_consistent_with_formula() {
    let returns = &[0.05, -0.02, 0.08, 0.01, -0.03];
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let vol = volatility(returns).unwrap();
    let risk_free = 0.02;
    assert_approx_eq(
        sharpe_ratio(returns, risk_free).unwrap(),
        (mean - risk_free) / vol,
    );
}

#[test]
fn sharpe_ratio_constant_returns_is_division_by_zero() {
    assert!(matches!(
        sharpe_ratio(&[0.0, 0.0, 0.0], 0.0),
        Err(QuantError::DivisionByZero)
    ));
    assert!(matches!(
        sharpe_ratio(&[1.0, 1.0, 1.0], 0.0),
        Err(QuantError::DivisionByZero)
    ));
}

#[test]
fn sharpe_ratio_insufficient_data() {
    assert!(matches!(
        sharpe_ratio(&[0.1], 0.0),
        Err(QuantError::InsufficientData)
    ));
    assert!(matches!(
        sharpe_ratio(&[], 0.0),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn sharpe_ratio_invalid_values() {
    assert!(matches!(
        sharpe_ratio(&[0.1, f64::NAN], 0.0),
        Err(QuantError::InvalidValue(_))
    ));
    assert!(matches!(
        sharpe_ratio(&[f64::INFINITY, 0.1], 0.0),
        Err(QuantError::InvalidValue(_))
    ));
}

// ── annualized_sharpe_ratio ───────────────────────────────────────────────────

#[test]
fn annualized_sharpe_ratio_daily_to_annual() {
    let returns = &[0.1, 0.2, 0.3];
    let sharpe = sharpe_ratio(returns, 0.0).unwrap();
    let ann = annualized_sharpe_ratio(returns, 0.0, 252.0).unwrap();
    assert_approx_eq(ann, sharpe * 252.0_f64.sqrt());
}

#[test]
fn annualized_sharpe_ratio_monthly_to_annual() {
    let returns = &[0.03, -0.01, 0.05, -0.02, 0.04];
    let sharpe = sharpe_ratio(returns, 0.01).unwrap();
    let ann = annualized_sharpe_ratio(returns, 0.01, 12.0).unwrap();
    assert_approx_eq(ann, sharpe * 12.0_f64.sqrt());
}

#[test]
fn annualized_sharpe_ratio_one_period_equals_sharpe() {
    let returns = &[0.1, 0.2, 0.3];
    let sharpe = sharpe_ratio(returns, 0.0).unwrap();
    let ann = annualized_sharpe_ratio(returns, 0.0, 1.0).unwrap();
    assert_approx_eq(ann, sharpe);
}

#[test]
fn annualized_sharpe_ratio_zero_periods_is_error() {
    assert!(matches!(
        annualized_sharpe_ratio(&[0.1, 0.2], 0.0, 0.0),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn annualized_sharpe_ratio_negative_periods_is_error() {
    assert!(matches!(
        annualized_sharpe_ratio(&[0.1, 0.2], 0.0, -252.0),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn annualized_sharpe_ratio_propagates_insufficient_data() {
    assert!(matches!(
        annualized_sharpe_ratio(&[0.1], 0.0, 252.0),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn annualized_sharpe_ratio_propagates_division_by_zero() {
    assert!(matches!(
        annualized_sharpe_ratio(&[1.0, 1.0, 1.0], 0.0, 252.0),
        Err(QuantError::DivisionByZero)
    ));
}

#[test]
fn annualized_sharpe_ratio_propagates_invalid_values() {
    assert!(matches!(
        annualized_sharpe_ratio(&[0.1, f64::NAN], 0.0, 252.0),
        Err(QuantError::InvalidValue(_))
    ));
}

// ── drawdowns ─────────────────────────────────────────────────────────────────

#[test]
fn drawdowns_single_price() {
    let dd = drawdowns(&[100.0]).unwrap();
    assert_eq!(dd.len(), 1);
    assert_approx_eq(dd[0], 0.0);
}

#[test]
fn drawdowns_first_element_is_always_zero() {
    let dd = drawdowns(&[50.0, 30.0, 80.0]).unwrap();
    assert_approx_eq(dd[0], 0.0);
}

#[test]
fn drawdowns_length_matches_input() {
    let prices = &[100.0, 90.0, 110.0, 80.0, 120.0];
    assert_eq!(drawdowns(prices).unwrap().len(), prices.len());
}

#[test]
fn drawdowns_all_increasing() {
    let dd = drawdowns(&[100.0, 110.0, 120.0]).unwrap();
    for d in &dd {
        assert_approx_eq(*d, 0.0);
    }
}

#[test]
fn drawdowns_simple_drop() {
    let dd = drawdowns(&[100.0, 80.0]).unwrap();
    assert_approx_eq(dd[0], 0.0);
    assert_approx_eq(dd[1], 80.0 / 100.0 - 1.0);
}

#[test]
fn drawdowns_new_peak_resets_reference() {
    let dd = drawdowns(&[100.0, 120.0, 90.0]).unwrap();
    assert_approx_eq(dd[0], 0.0);
    assert_approx_eq(dd[1], 0.0);
    assert_approx_eq(dd[2], 90.0 / 120.0 - 1.0);
}

#[test]
fn drawdowns_full_recovery() {
    let dd = drawdowns(&[100.0, 80.0, 100.0]).unwrap();
    assert_approx_eq(dd[0], 0.0);
    assert_approx_eq(dd[1], 80.0 / 100.0 - 1.0);
    assert_approx_eq(dd[2], 0.0);
}

#[test]
fn drawdowns_empty() {
    assert!(matches!(drawdowns(&[]), Err(QuantError::InsufficientData)));
}

#[test]
fn drawdowns_invalid_values() {
    assert!(matches!(
        drawdowns(&[100.0, f64::NAN]),
        Err(QuantError::InvalidValue(_))
    ));
    assert!(matches!(
        drawdowns(&[100.0, f64::INFINITY]),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn drawdowns_non_positive_price() {
    assert!(matches!(
        drawdowns(&[100.0, 0.0]),
        Err(QuantError::NonPositivePrice(_))
    ));
    assert!(matches!(
        drawdowns(&[100.0, -50.0]),
        Err(QuantError::NonPositivePrice(_))
    ));
}

// ── max_drawdown ──────────────────────────────────────────────────────────────

#[test]
fn max_drawdown_single_price() {
    assert_approx_eq(max_drawdown(&[100.0]).unwrap(), 0.0);
}

#[test]
fn max_drawdown_no_drawdown() {
    assert_approx_eq(max_drawdown(&[100.0, 110.0, 120.0]).unwrap(), 0.0);
}

#[test]
fn max_drawdown_simple() {
    assert_approx_eq(max_drawdown(&[100.0, 80.0]).unwrap(), 80.0 / 100.0 - 1.0);
}

#[test]
fn max_drawdown_picks_worst_drop() {
    assert_approx_eq(
        max_drawdown(&[100.0, 80.0, 90.0, 60.0]).unwrap(),
        60.0 / 100.0 - 1.0,
    );
}

#[test]
fn max_drawdown_new_peak_then_drop() {
    assert_approx_eq(
        max_drawdown(&[100.0, 150.0, 120.0]).unwrap(),
        120.0 / 150.0 - 1.0,
    );
}

#[test]
fn max_drawdown_consistent_with_drawdowns() {
    let prices = &[100.0, 90.0, 120.0, 80.0, 110.0];
    let mdd = max_drawdown(prices).unwrap();
    let all_dd = drawdowns(prices).unwrap();
    let min_dd = all_dd.iter().cloned().fold(f64::INFINITY, f64::min);
    assert_approx_eq(mdd, min_dd);
}

#[test]
fn max_drawdown_empty() {
    assert!(matches!(
        max_drawdown(&[]),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn max_drawdown_invalid_values() {
    assert!(matches!(
        max_drawdown(&[100.0, f64::NAN]),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn max_drawdown_non_positive_price() {
    assert!(matches!(
        max_drawdown(&[100.0, 0.0]),
        Err(QuantError::NonPositivePrice(_))
    ));
    assert!(matches!(
        max_drawdown(&[100.0, -10.0]),
        Err(QuantError::NonPositivePrice(_))
    ));
}

// ── max_drawdown_duration ─────────────────────────────────────────────────────

#[test]
fn max_drawdown_duration_single_price() {
    assert_eq!(max_drawdown_duration(&[100.0]).unwrap(), 0);
}

#[test]
fn max_drawdown_duration_no_drawdown() {
    assert_eq!(max_drawdown_duration(&[100.0, 110.0, 120.0]).unwrap(), 0);
}

#[test]
fn max_drawdown_duration_one_period() {
    assert_eq!(max_drawdown_duration(&[100.0, 90.0]).unwrap(), 1);
}

#[test]
fn max_drawdown_duration_consecutive_drops() {
    assert_eq!(
        max_drawdown_duration(&[100.0, 90.0, 80.0, 70.0]).unwrap(),
        3
    );
}

#[test]
fn max_drawdown_duration_recovery_resets_counter() {
    assert_eq!(
        max_drawdown_duration(&[100.0, 90.0, 100.0, 90.0]).unwrap(),
        1
    );
}

#[test]
fn max_drawdown_duration_longer_second_drawdown_wins() {
    assert_eq!(
        max_drawdown_duration(&[100.0, 90.0, 100.0, 80.0, 70.0]).unwrap(),
        2
    );
}

#[test]
fn max_drawdown_duration_new_peak_resets_reference() {
    assert_eq!(
        max_drawdown_duration(&[100.0, 80.0, 120.0, 100.0]).unwrap(),
        1
    );
}

#[test]
fn max_drawdown_duration_exact_peak_price_resets() {
    assert_eq!(
        max_drawdown_duration(&[100.0, 80.0, 100.0]).unwrap(),
        1
    );
}

#[test]
fn max_drawdown_duration_empty() {
    assert!(matches!(
        max_drawdown_duration(&[]),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn max_drawdown_duration_invalid_values() {
    assert!(matches!(
        max_drawdown_duration(&[100.0, f64::NAN]),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn max_drawdown_duration_non_positive_price() {
    assert!(matches!(
        max_drawdown_duration(&[100.0, 0.0]),
        Err(QuantError::NonPositivePrice(_))
    ));
    assert!(matches!(
        max_drawdown_duration(&[100.0, -10.0]),
        Err(QuantError::NonPositivePrice(_))
    ));
}

// ── sortino_ratio ─────────────────────────────────────────────────────────────

#[test]
fn sortino_ratio_zero_risk_free() {
    assert_approx_eq(sortino_ratio(&[0.1, 0.2, -0.1], 0.0).unwrap(), 2.0 / 3.0);
}

#[test]
fn sortino_ratio_nonzero_risk_free() {
    assert_approx_eq(
        sortino_ratio(&[0.1, 0.2, 0.05], 0.08).unwrap(),
        11.0 / 9.0,
    );
}

#[test]
fn sortino_ratio_consistent_with_formula() {
    let returns = [0.08, -0.04, 0.03, -0.01, 0.12];
    let risk_free = 0.02;

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let downside: Vec<f64> = returns
        .iter()
        .copied()
        .filter(|&r| r < risk_free)
        .map(|r| {
            let d = r - risk_free;
            d * d
        })
        .collect();
    let downside_dev = (downside.iter().sum::<f64>() / downside.len() as f64).sqrt();
    let expected = (mean - risk_free) / downside_dev;

    assert_approx_eq(sortino_ratio(&returns, risk_free).unwrap(), expected);
}

#[test]
fn sortino_ratio_negative_value() {
    let ratio = sortino_ratio(&[-0.02, -0.01, 0.01], 0.02).unwrap();
    assert!(ratio < 0.0);
}

#[test]
fn sortino_ratio_insufficient_data() {
    assert!(matches!(
        sortino_ratio(&[0.1], 0.0),
        Err(QuantError::InsufficientData)
    ));
    assert!(matches!(
        sortino_ratio(&[], 0.0),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn sortino_ratio_invalid_values() {
    assert!(matches!(
        sortino_ratio(&[0.1, f64::NAN], 0.0),
        Err(QuantError::InvalidValue(_))
    ));
    assert!(matches!(
        sortino_ratio(&[f64::INFINITY, 0.1], 0.0),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn sortino_ratio_no_downside_is_division_by_zero() {
    assert!(matches!(
        sortino_ratio(&[0.02, 0.03, 0.04], 0.02),
        Err(QuantError::DivisionByZero)
        ));
}