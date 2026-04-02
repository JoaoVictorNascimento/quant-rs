use super::signal::Signal;
use crate::core::QuantError;

pub fn simple_moving_average(
    prices: &[f64],
    window: usize,
) -> Result<Vec<Option<f64>>, QuantError> {
    if prices.is_empty() {
        return Err(QuantError::InsufficientData);
    }

    if window == 0 {
        return Err(QuantError::InvalidValue(window as f64));
    }

    for &price in prices {
        if !price.is_finite() {
            return Err(QuantError::InvalidValue(price));
        }

        if price <= 0.0 {
            return Err(QuantError::NonPositivePrice(price));
        }
    }

    let mut result = vec![None; prices.len()];
    let mut sum = 0.0;

    for i in 0..prices.len() {
        sum += prices[i];

        if i >= window {
            sum -= prices[i - window];
        }

        if i + 1 >= window {
            result[i] = Some(sum / window as f64);
        }
    }

    Ok(result)
}

pub fn moving_average_crossover_signals(
    prices: &[f64],
    short_window: usize,
    long_window: usize,
) -> Result<Vec<Signal>, QuantError> {
    if short_window == 0 {
        return Err(QuantError::InvalidValue(short_window as f64));
    }

    if long_window == 0 {
        return Err(QuantError::InvalidValue(long_window as f64));
    }

    if short_window >= long_window {
        return Err(QuantError::InvalidValue(short_window as f64));
    }

    let short_sma = simple_moving_average(prices, short_window)?;
    let long_sma = simple_moving_average(prices, long_window)?;

    let mut signals = Vec::with_capacity(prices.len());

    for i in 0..prices.len() {
        match (short_sma[i], long_sma[i]) {
            (Some(short), Some(long)) if short > long => signals.push(Signal::Buy),
            (Some(short), Some(long)) if short < long => signals.push(Signal::Sell),
            (Some(_), Some(_)) => signals.push(Signal::Hold),
            _ => signals.push(Signal::Hold),
        }
    }

    Ok(signals)
}
