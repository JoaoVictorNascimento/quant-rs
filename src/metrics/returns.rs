use crate::core::validation::validate_prices;
use crate::core::QuantError;

pub fn simple_returns(prices: &[f64]) -> Result<Vec<f64>, QuantError> {
    validate_prices(prices)?;

    let mut returns = Vec::with_capacity(prices.len() - 1);

    for i in 1..prices.len() {
        let p0 = prices[i - 1];
        let p1 = prices[i];

        returns.push((p1 / p0) - 1.0);
    }

    Ok(returns)
}


pub fn log_returns(prices: &[f64]) -> Result<Vec<f64>, QuantError> {
    validate_prices(prices)?;

    let mut returns = Vec::with_capacity(prices.len() - 1);

    for window in prices.windows(2) {
        let (p0, p1) = (window[0], window[1]);
        returns.push((p1 / p0).ln());
    }

    Ok(returns)
}

pub fn cumulative_return(prices: &[f64]) -> Result<f64, QuantError> {
    validate_prices(prices)?;

    let first = prices.first().unwrap();
    let last = prices.last().unwrap();

    Ok((last / first) - 1.0)
}

pub fn cumulative_log_return(prices: &[f64]) -> Result<f64, QuantError> {
    validate_prices(prices)?;

    let first = prices.first().unwrap();
    let last = prices.last().unwrap();

    Ok((last / first).ln())
}

pub fn cumulative_from_returns(returns: &[f64]) -> Result<f64, QuantError> {
    if returns.is_empty() {
        return Err(QuantError::InsufficientData);
    }

    let mut acc = 1.0;

    for &r in returns {
        if !r.is_finite() {
            return Err(QuantError::InvalidValue(r));
        }

        acc *= 1.0 + r;
    }

    Ok(acc - 1.0)
}