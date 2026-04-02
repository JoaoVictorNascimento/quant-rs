use crate::core::QuantError;
use crate::strategy::signal::Signal;

#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub strategy_returns: Vec<f64>,
    pub equity_curve: Vec<f64>,
    pub final_equity: f64,
}

pub struct BacktestEngine;

impl BacktestEngine {
    pub fn run(prices: &[f64], signals: &[Signal]) -> Result<BacktestResult, QuantError> {
        if prices.len() < 2 {
            return Err(QuantError::InsufficientData);
        }

        if signals.len() != prices.len() {
            return Err(QuantError::InvalidValue(signals.len() as f64));
        }

        for &price in prices {
            if !price.is_finite() {
                return Err(QuantError::InvalidValue(price));
            }

            if price <= 0.0 {
                return Err(QuantError::NonPositivePrice(price));
            }
        }

        let mut strategy_returns = Vec::with_capacity(prices.len() - 1);
        let mut equity_curve = Vec::with_capacity(prices.len());
        let mut equity = 1.0;
        let mut in_position = false;

        equity_curve.push(equity);

        for i in 1..prices.len() {
            match signals[i - 1] {
                Signal::Buy => in_position = true,
                Signal::Sell => in_position = false,
                Signal::Hold => {}
            }

            let period_return = if in_position {
                (prices[i] / prices[i - 1]) - 1.0
            } else {
                0.0
            };

            strategy_returns.push(period_return);
            equity *= 1.0 + period_return;
            equity_curve.push(equity);
        }

        Ok(BacktestResult {
            strategy_returns,
            equity_curve,
            final_equity: equity,
        })
    }
}
