use super::errors::QuantError;

pub fn validate_prices(prices: &[f64]) -> Result<(), QuantError> {
    if prices.len() < 2 {
        return Err(QuantError::InsufficientData);
    }

    for &p in prices {
        if !p.is_finite() {
            return Err(QuantError::InvalidValue(p));
        }

        if p == 0.0 {
            return Err(QuantError::ZeroPrice);
        }

        if p < 0.0 {
            return Err(QuantError::NonPositivePrice(p));
        }
    }

    Ok(())
}
