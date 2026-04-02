use quantix::backtest::engine::{BacktestEngine, BacktestResult as EngineBacktestResult};
use quantix::backtest::results::BacktestResult as AnalyticsBacktestResult;
use quantix::core::QuantError;
use quantix::strategy::moving_average::moving_average_crossover_signals;
use quantix::strategy::signal::Signal;

fn to_analytics_result(result: EngineBacktestResult) -> AnalyticsBacktestResult {
    AnalyticsBacktestResult {
        strategy_returns: result.strategy_returns,
        equity_curve: result.equity_curve,
        final_equity: result.final_equity,
    }
}

fn run_manual_signal_backtest() -> Result<(), QuantError> {
    let prices = vec![100.0, 102.0, 101.0, 104.0, 106.0, 103.0, 108.0];
    let signals = vec![
        Signal::Buy,
        Signal::Hold,
        Signal::Hold,
        Signal::Sell,
        Signal::Hold,
        Signal::Buy,
        Signal::Hold,
    ];

    let result = BacktestEngine::run(&prices, &signals)?;

    println!("=== Backtest with manual signals ===");
    println!("Final equity: {:.6}", result.final_equity);
    println!("Period returns: {:?}", result.strategy_returns);
    println!("Equity curve: {:?}", result.equity_curve);
    println!();

    Ok(())
}

fn run_moving_average_backtest() -> Result<(), QuantError> {
    let prices = vec![100.0, 99.0, 101.0, 103.0, 105.0, 102.0, 106.0, 108.0];
    let signals = moving_average_crossover_signals(&prices, 2, 4)?;

    let engine_result = BacktestEngine::run(&prices, &signals)?;
    let analytics_result = to_analytics_result(engine_result);

    println!("=== Backtest with moving average crossover ===");
    println!("Signals: {:?}", signals);
    println!("Final equity: {:.6}", analytics_result.final_equity);
    println!(
        "Cumulative return: {:.6}",
        analytics_result.cumulative_return()?
    );
    println!("Volatility: {:.6}", analytics_result.volatility()?);
    println!(
        "Sharpe (rf=0): {:.6}",
        analytics_result.sharpe_ratio(0.0)?
    );
    println!("Max drawdown: {:.6}", analytics_result.max_drawdown()?);
    println!();

    Ok(())
}

fn main() -> Result<(), QuantError> {
    run_manual_signal_backtest()?;
    run_moving_average_backtest()?;
    Ok(())
}