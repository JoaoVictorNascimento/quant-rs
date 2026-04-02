use crate::core::QuantError;

pub fn drawdowns(prices: &[f64]) -> Result<Vec<f64>, QuantError> {
    if prices.is_empty() {
        return Err(QuantError::InsufficientData);
    }

    let mut result = Vec::with_capacity(prices.len());
    let mut peak = prices[0];

    for &price in prices {
        if !price.is_finite() {
            return Err(QuantError::InvalidValue(price));
        }

        if price <= 0.0 {
            return Err(QuantError::NonPositivePrice(price));
        }

        if price > peak {
            peak = price;
        }

        let dd = (price / peak) - 1.0;
        result.push(dd);
    }

    Ok(result)
}

pub fn max_drawdown(prices: &[f64]) -> Result<f64, QuantError> {
    if prices.is_empty() {
        return Err(QuantError::InsufficientData);
    }

    let mut peak = prices[0];
    let mut max_dd = 0.0;

    for &price in prices {
        if !price.is_finite() {
            return Err(QuantError::InvalidValue(price));
        }

        if price <= 0.0 {
            return Err(QuantError::NonPositivePrice(price));
        }

        if price > peak {
            peak = price;
        }

        let dd = (price / peak) - 1.0;

        if dd < max_dd {
            max_dd = dd;
        }
    }

    Ok(max_dd)
}

pub fn max_drawdown_duration(prices: &[f64]) -> Result<usize, QuantError> {
    if prices.is_empty() {
        return Err(QuantError::InsufficientData);
    }

    let mut peak = prices[0];
    let mut current_duration = 0;
    let mut max_duration = 0;

    for &price in prices {
        if !price.is_finite() {
            return Err(QuantError::InvalidValue(price));
        }

        if price <= 0.0 {
            return Err(QuantError::NonPositivePrice(price));
        }

        if price >= peak {
            peak = price;
            current_duration = 0;
        } else {
            current_duration += 1;
            if current_duration > max_duration {
                max_duration = current_duration;
            }
        }
    }

    Ok(max_duration)
}
