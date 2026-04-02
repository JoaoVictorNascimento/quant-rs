use crate::core::QuantError;

#[derive(Debug, Clone)]
pub struct Position {
    pub weight: f64,
    pub returns: Vec<f64>,
}

impl Position {
    pub fn new(weight: f64, returns: Vec<f64>) -> Result<Self, QuantError> {
        if !weight.is_finite() {
            return Err(QuantError::InvalidValue(weight));
        }

        if returns.is_empty() {
            return Err(QuantError::InsufficientData);
        }

        for &r in &returns {
            if !r.is_finite() {
                return Err(QuantError::InvalidValue(r));
            }
        }

        Ok(Self { weight, returns })
    }

    pub fn len(&self) -> usize {
        self.returns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.returns.is_empty()
    }
}
