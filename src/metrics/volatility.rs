use crate::core::QuantError;

pub fn variance(returns: &[f64]) -> Result<f64, QuantError> {
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

    let mut sum_sq = 0.0;

    for &r in returns {
        let diff = r - mean;
        sum_sq += diff * diff;
    }

    Ok(sum_sq / (n as f64 - 1.0))
}

pub fn volatility(returns: &[f64]) -> Result<f64, QuantError> {
    let var = variance(returns)?;
    Ok(var.sqrt())
}

pub fn annualized_volatility(
    returns: &[f64],
    periods_per_year: f64,
) -> Result<f64, QuantError> {
    if periods_per_year <= 0.0 {
        return Err(QuantError::InvalidValue(periods_per_year));
    }

    let vol = volatility(returns)?;
    Ok(vol * periods_per_year.sqrt())
}