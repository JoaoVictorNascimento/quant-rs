use quantix::backtest::engine::BacktestEngine;
use quantix::backtest::results::BacktestResult;
use quantix::core::QuantError;
use quantix::strategy::signal::Signal;

fn assert_approx_eq(a: f64, b: f64) {
    let eps = 1e-12_f64.max(a.abs().max(b.abs()) * 1e-12);
    assert!(
        (a - b).abs() < eps,
        "expected {a} ≈ {b}, diff {}",
        (a - b).abs()
    );
}

// ── backtest_engine_run ───────────────────────────────────────────────────────

#[test]
fn run_hold_only_keeps_flat_equity() {
    let prices = [100.0, 110.0, 90.0, 120.0];
    let signals = [Signal::Hold, Signal::Hold, Signal::Hold, Signal::Hold];

    let result = BacktestEngine::run(&prices, &signals).unwrap();

    assert_eq!(result.strategy_returns.len(), 3);
    assert_approx_eq(result.strategy_returns[0], 0.0);
    assert_approx_eq(result.strategy_returns[1], 0.0);
    assert_approx_eq(result.strategy_returns[2], 0.0);

    assert_eq!(result.equity_curve.len(), 4);
    assert_approx_eq(result.equity_curve[0], 1.0);
    assert_approx_eq(result.equity_curve[1], 1.0);
    assert_approx_eq(result.equity_curve[2], 1.0);
    assert_approx_eq(result.equity_curve[3], 1.0);
    assert_approx_eq(result.final_equity, 1.0);
}

#[test]
fn run_buy_then_hold_captures_returns() {
    let prices = [100.0, 110.0, 121.0];
    let signals = [Signal::Buy, Signal::Hold, Signal::Hold];

    let result = BacktestEngine::run(&prices, &signals).unwrap();

    assert_eq!(result.strategy_returns.len(), 2);
    assert_approx_eq(result.strategy_returns[0], 0.1);
    assert_approx_eq(result.strategy_returns[1], 0.1);

    assert_eq!(result.equity_curve.len(), 3);
    assert_approx_eq(result.equity_curve[0], 1.0);
    assert_approx_eq(result.equity_curve[1], 1.1);
    assert_approx_eq(result.equity_curve[2], 1.21);
    assert_approx_eq(result.final_equity, 1.21);
}

#[test]
fn run_sell_before_period_stays_out_of_market() {
    let prices = [100.0, 110.0, 120.0];
    let signals = [Signal::Sell, Signal::Buy, Signal::Hold];

    let result = BacktestEngine::run(&prices, &signals).unwrap();

    assert_eq!(result.strategy_returns.len(), 2);
    assert_approx_eq(result.strategy_returns[0], 0.0);
    assert_approx_eq(result.strategy_returns[1], 120.0 / 110.0 - 1.0);
}

#[test]
fn run_signal_applies_to_next_period() {
    let prices = [100.0, 90.0, 99.0, 108.9];
    let signals = [Signal::Buy, Signal::Sell, Signal::Buy, Signal::Hold];

    let result = BacktestEngine::run(&prices, &signals).unwrap();

    // i=1 uses signals[0]=Buy   -> in position -> -10%
    // i=2 uses signals[1]=Sell  -> out         -> 0%
    // i=3 uses signals[2]=Buy   -> in position -> +10%
    assert_eq!(result.strategy_returns.len(), 3);
    assert_approx_eq(result.strategy_returns[0], -0.1);
    assert_approx_eq(result.strategy_returns[1], 0.0);
    assert_approx_eq(result.strategy_returns[2], 0.1);

    assert_eq!(result.equity_curve.len(), 4);
    assert_approx_eq(result.equity_curve[0], 1.0);
    assert_approx_eq(result.equity_curve[1], 0.9);
    assert_approx_eq(result.equity_curve[2], 0.9);
    assert_approx_eq(result.equity_curve[3], 0.99);
    assert_approx_eq(result.final_equity, 0.99);
}

#[test]
fn run_lengths_are_consistent() {
    let prices = [100.0, 105.0, 110.0, 100.0, 95.0];
    let signals = [
        Signal::Buy,
        Signal::Hold,
        Signal::Hold,
        Signal::Sell,
        Signal::Hold,
    ];

    let result = BacktestEngine::run(&prices, &signals).unwrap();
    assert_eq!(result.strategy_returns.len(), prices.len() - 1);
    assert_eq!(result.equity_curve.len(), prices.len());
}

#[test]
fn run_compounds_equity_from_strategy_returns() {
    let prices = [100.0, 110.0, 99.0, 108.9];
    let signals = [Signal::Buy, Signal::Hold, Signal::Hold, Signal::Hold];

    let result = BacktestEngine::run(&prices, &signals).unwrap();
    let compounded = result
        .strategy_returns
        .iter()
        .fold(1.0, |acc, r| acc * (1.0 + r));
    assert_approx_eq(result.final_equity, compounded);
}

#[test]
fn run_insufficient_data_is_error() {
    let signals = [Signal::Hold];
    assert!(matches!(
        BacktestEngine::run(&[100.0], &signals),
        Err(QuantError::InsufficientData)
    ));
    assert!(matches!(
        BacktestEngine::run(&[], &[]),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn run_signals_length_mismatch_is_error() {
    let prices = [100.0, 110.0, 120.0];
    let signals = [Signal::Buy, Signal::Hold];
    assert!(matches!(
        BacktestEngine::run(&prices, &signals),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn run_non_positive_price_is_error() {
    let signals = [Signal::Hold, Signal::Hold];
    assert!(matches!(
        BacktestEngine::run(&[100.0, 0.0], &signals),
        Err(QuantError::NonPositivePrice(_))
    ));
    assert!(matches!(
        BacktestEngine::run(&[100.0, -1.0], &signals),
        Err(QuantError::NonPositivePrice(_))
    ));
}

#[test]
fn run_invalid_price_is_error() {
    let signals = [Signal::Hold, Signal::Hold];
    assert!(matches!(
        BacktestEngine::run(&[100.0, f64::NAN], &signals),
        Err(QuantError::InvalidValue(_))
    ));
    assert!(matches!(
        BacktestEngine::run(&[100.0, f64::INFINITY], &signals),
        Err(QuantError::InvalidValue(_))
    ));
}

fn sample_backtest_result() -> BacktestResult {
    BacktestResult {
        strategy_returns: vec![0.1, -0.05, 0.02],
        equity_curve: vec![1.0, 1.1, 1.045, 1.0659],
        final_equity: 1.0659,
    }
}

// ── backtest_result_len_is_empty ─────────────────────────────────────────────

#[test]
fn backtest_result_len_matches_strategy_returns() {
    let result = sample_backtest_result();
    assert_eq!(result.len(), 3);
}

#[test]
fn backtest_result_is_empty_false_when_has_returns() {
    let result = sample_backtest_result();
    assert!(!result.is_empty());
}

#[test]
fn backtest_result_is_empty_true_when_no_returns() {
    let result = BacktestResult {
        strategy_returns: vec![],
        equity_curve: vec![1.0],
        final_equity: 1.0,
    };
    assert!(result.is_empty());
    assert_eq!(result.len(), 0);
}

// ── backtest_result_cumulative_return ────────────────────────────────────────

#[test]
fn backtest_result_cumulative_return_basic() {
    let result = sample_backtest_result();
    let expected = (1.1_f64 * 0.95 * 1.02) - 1.0;
    assert_approx_eq(result.cumulative_return().unwrap(), expected);
}

#[test]
fn backtest_result_cumulative_return_empty_is_error() {
    let result = BacktestResult {
        strategy_returns: vec![],
        equity_curve: vec![1.0],
        final_equity: 1.0,
    };
    assert!(matches!(
        result.cumulative_return(),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn backtest_result_cumulative_return_invalid_value_is_error() {
    let result = BacktestResult {
        strategy_returns: vec![0.1, f64::NAN],
        equity_curve: vec![1.0, 1.1, 1.0],
        final_equity: 1.0,
    };
    assert!(matches!(
        result.cumulative_return(),
        Err(QuantError::InvalidValue(_))
    ));
}

// ── backtest_result_volatility ───────────────────────────────────────────────

#[test]
fn backtest_result_volatility_basic() {
    let result = sample_backtest_result();
    let vol = result.volatility().unwrap();
    assert!(vol > 0.0);
}

#[test]
fn backtest_result_volatility_constant_returns_is_zero() {
    let result = BacktestResult {
        strategy_returns: vec![1.0, 1.0, 1.0],
        equity_curve: vec![1.0, 2.0, 4.0, 8.0],
        final_equity: 8.0,
    };
    assert_approx_eq(result.volatility().unwrap(), 0.0);
}

#[test]
fn backtest_result_volatility_insufficient_data_is_error() {
    let result = BacktestResult {
        strategy_returns: vec![0.1],
        equity_curve: vec![1.0, 1.1],
        final_equity: 1.1,
    };
    assert!(matches!(result.volatility(), Err(QuantError::InsufficientData)));
}

#[test]
fn backtest_result_volatility_invalid_value_is_error() {
    let result = BacktestResult {
        strategy_returns: vec![0.1, f64::INFINITY],
        equity_curve: vec![1.0, 1.1, 1.1],
        final_equity: 1.1,
    };
    assert!(matches!(
        result.volatility(),
        Err(QuantError::InvalidValue(_))
    ));
}

// ── backtest_result_sharpe_ratio ─────────────────────────────────────────────

#[test]
fn backtest_result_sharpe_ratio_basic() {
    let result = sample_backtest_result();
    let ratio = result.sharpe_ratio(0.0).unwrap();
    assert!(ratio.is_finite());
}

#[test]
fn backtest_result_sharpe_ratio_constant_returns_is_division_by_zero() {
    let result = BacktestResult {
        strategy_returns: vec![1.0, 1.0, 1.0],
        equity_curve: vec![1.0, 2.0, 4.0, 8.0],
        final_equity: 8.0,
    };
    assert!(matches!(
        result.sharpe_ratio(0.0),
        Err(QuantError::DivisionByZero)
    ));
}

#[test]
fn backtest_result_sharpe_ratio_insufficient_data_is_error() {
    let result = BacktestResult {
        strategy_returns: vec![0.1],
        equity_curve: vec![1.0, 1.1],
        final_equity: 1.1,
    };
    assert!(matches!(
        result.sharpe_ratio(0.0),
        Err(QuantError::InsufficientData)
    ));
}

// ── backtest_result_max_drawdown ─────────────────────────────────────────────

#[test]
fn backtest_result_max_drawdown_basic() {
    let result = BacktestResult {
        strategy_returns: vec![0.0, 0.0, 0.0],
        equity_curve: vec![1.0, 1.2, 1.1, 1.3],
        final_equity: 1.3,
    };
    assert_approx_eq(result.max_drawdown().unwrap(), 1.1 / 1.2 - 1.0);
}

#[test]
fn backtest_result_max_drawdown_no_drawdown_is_zero() {
    let result = BacktestResult {
        strategy_returns: vec![0.1, 0.1],
        equity_curve: vec![1.0, 1.1, 1.21],
        final_equity: 1.21,
    };
    assert_approx_eq(result.max_drawdown().unwrap(), 0.0);
}

#[test]
fn backtest_result_max_drawdown_empty_equity_curve_is_error() {
    let result = BacktestResult {
        strategy_returns: vec![0.1, -0.1],
        equity_curve: vec![],
        final_equity: 1.0,
    };
    assert!(matches!(
        result.max_drawdown(),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn backtest_result_max_drawdown_invalid_value_is_error() {
    let result = BacktestResult {
        strategy_returns: vec![0.1, -0.1],
        equity_curve: vec![1.0, f64::NAN, 0.9],
        final_equity: 0.9,
    };
    assert!(matches!(
        result.max_drawdown(),
        Err(QuantError::InvalidValue(_))
    ));
}
