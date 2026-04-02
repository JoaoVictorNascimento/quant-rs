use crate::core::QuantError;

pub fn sortino_ratio(returns: &[f64], risk_free: f64) -> Result<f64, QuantError> {
    let n = returns.len();

    if n < 2 {
        return Err(QuantError::InsufficientData);
    }

    let mut sum = 0.0;
    let mut downside_sum = 0.0;
    let mut downside_count = 0;

    for &r in returns {
        if !r.is_finite() {
            return Err(QuantError::InvalidValue(r));
        }

        sum += r;

        if r < risk_free {
            let diff = r - risk_free;
            downside_sum += diff * diff;
            downside_count += 1;
        }
    }

    let mean = sum / n as f64;

    if downside_count == 0 {
        return Err(QuantError::DivisionByZero);
    }

    let downside_variance = downside_sum / downside_count as f64;
    let downside_dev = downside_variance.sqrt();

    if downside_dev == 0.0 {
        return Err(QuantError::DivisionByZero);
    }

    Ok((mean - risk_free) / downside_dev)
}
