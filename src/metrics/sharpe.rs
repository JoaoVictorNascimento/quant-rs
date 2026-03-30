use crate::core::QuantError;
use crate::metrics::volatility::volatility;


pub fn sharpe_ratio(
    returns: &[f64],
    risk_free: f64,
) -> Result<f64, QuantError> {
    let n = returns.len();

    if n < 2 {
        return Err(QuantError::InsufficientData);
    }

    for &r in returns {
        if !r.is_finite() {
            return Err(QuantError::InvalidValue(r));
        }
    }

    let mean = returns.iter().sum::<f64>() / n as f64;

    let vol = volatility(returns)?;

    if vol == 0.0 {
        return Err(QuantError::DivisionByZero);
    }

    Ok((mean - risk_free) / vol)
}

pub fn annualized_sharpe_ratio(
    returns: &[f64],
    risk_free: f64,
    periods_per_year: f64,
) -> Result<f64, QuantError> {
    if periods_per_year <= 0.0 {
        return Err(QuantError::InvalidValue(periods_per_year));
    }

    let sharpe = sharpe_ratio(returns, risk_free)?;
    Ok(sharpe * periods_per_year.sqrt())
}