use crate::core::QuantError;
use crate::metrics::drawdown::max_drawdown;
use crate::metrics::returns::cumulative_from_returns;
use crate::metrics::sharpe::sharpe_ratio;
use crate::metrics::volatility::volatility;

#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub strategy_returns: Vec<f64>,
    pub equity_curve: Vec<f64>,
    pub final_equity: f64,
}

impl BacktestResult {
    pub fn len(&self) -> usize {
        self.strategy_returns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strategy_returns.is_empty()
    }

    pub fn cumulative_return(&self) -> Result<f64, QuantError> {
        cumulative_from_returns(&self.strategy_returns)
    }

    pub fn volatility(&self) -> Result<f64, QuantError> {
        volatility(&self.strategy_returns)
    }

    pub fn sharpe_ratio(&self, risk_free: f64) -> Result<f64, QuantError> {
        sharpe_ratio(&self.strategy_returns, risk_free)
    }

    pub fn max_drawdown(&self) -> Result<f64, QuantError> {
        max_drawdown(&self.equity_curve)
    }
}
