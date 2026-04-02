use quantix::core::QuantError;
use quantix::metrics::drawdown::{drawdowns, max_drawdown, max_drawdown_duration};
use quantix::metrics::returns::{
    cumulative_from_returns, cumulative_log_return, cumulative_return, log_returns, simple_returns,
};
use quantix::metrics::sharpe::{annualized_sharpe_ratio, sharpe_ratio};
use quantix::metrics::sortino::sortino_ratio;
use quantix::metrics::volatility::{annualized_volatility, variance, volatility};

fn run_returns_examples(prices: &[f64]) -> Result<(), QuantError> {
    let simple = simple_returns(prices)?;
    let log = log_returns(prices)?;
    let cum_price = cumulative_return(prices)?;
    let cum_log = cumulative_log_return(prices)?;
    let cum_from_simple = cumulative_from_returns(&simple)?;

    println!("=== Returns metrics ===");
    println!("Prices: {:?}", prices);
    println!("Simple returns: {:?}", simple);
    println!("Log returns: {:?}", log);
    println!("Cumulative return (from prices): {:.6}", cum_price);
    println!("Cumulative log return: {:.6}", cum_log);
    println!("Cumulative return (from simple returns): {:.6}", cum_from_simple);
    println!();

    Ok(())
}

fn run_volatility_examples(returns: &[f64]) -> Result<(), QuantError> {
    let var = variance(returns)?;
    let vol = volatility(returns)?;
    let ann_vol = annualized_volatility(returns, 252.0)?;

    println!("=== Volatility metrics ===");
    println!("Returns: {:?}", returns);
    println!("Variance: {:.6}", var);
    println!("Volatility: {:.6}", vol);
    println!("Annualized volatility (252): {:.6}", ann_vol);
    println!();

    Ok(())
}

fn run_risk_adjusted_examples(returns: &[f64]) -> Result<(), QuantError> {
    let risk_free = 0.0;
    let sharpe = sharpe_ratio(returns, risk_free)?;
    let ann_sharpe = annualized_sharpe_ratio(returns, risk_free, 252.0)?;
    let sortino = sortino_ratio(returns, risk_free)?;

    println!("=== Risk-adjusted metrics ===");
    println!("Returns: {:?}", returns);
    println!("Sharpe ratio (rf=0): {:.6}", sharpe);
    println!("Annualized Sharpe ratio (rf=0, 252): {:.6}", ann_sharpe);
    println!("Sortino ratio (rf=0): {:.6}", sortino);
    println!();

    Ok(())
}

fn run_drawdown_examples(equity_curve: &[f64]) -> Result<(), QuantError> {
    let dds = drawdowns(equity_curve)?;
    let mdd = max_drawdown(equity_curve)?;
    let mdd_duration = max_drawdown_duration(equity_curve)?;

    println!("=== Drawdown metrics ===");
    println!("Equity curve: {:?}", equity_curve);
    println!("Drawdowns: {:?}", dds);
    println!("Max drawdown: {:.6}", mdd);
    println!("Max drawdown duration: {}", mdd_duration);
    println!();

    Ok(())
}

fn main() -> Result<(), QuantError> {
    let prices = vec![100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 110.0];
    let returns = simple_returns(&prices)?;
    let equity_curve = vec![1.0, 1.02, 1.01, 1.05, 1.03, 1.08, 1.10];

    run_returns_examples(&prices)?;
    run_volatility_examples(&returns)?;
    run_risk_adjusted_examples(&returns)?;
    run_drawdown_examples(&equity_curve)?;

    Ok(())
}