use crate::core::QuantError;

#[derive(Debug, Clone)]
pub struct Position {
    pub weight: f64,
    pub returns: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Portfolio {
    pub positions: Vec<Position>,
}

impl Portfolio {
    fn validate(&self) -> Result<usize, QuantError> {
        if self.positions.is_empty() {
            return Err(QuantError::InsufficientData);
        }

        let len = self.positions[0].returns.len();

        if len == 0 {
            return Err(QuantError::InsufficientData);
        }

        for p in &self.positions {
            if p.returns.len() != len {
                return Err(QuantError::InvalidValue(p.returns.len() as f64));
            }

            if !p.weight.is_finite() {
                return Err(QuantError::InvalidValue(p.weight));
            }

            for &r in &p.returns {
                if !r.is_finite() {
                    return Err(QuantError::InvalidValue(r));
                }
            }
        }

        Ok(len)
    }

    pub fn returns(&self) -> Result<Vec<f64>, QuantError> {
        let len = self.validate()?;

        let mut result = vec![0.0; len];

        for p in &self.positions {
            for (i, &r) in p.returns.iter().enumerate() {
                result[i] += p.weight * r;
            }
        }

        Ok(result)
    }

    pub fn weights_sum(&self) -> f64 {
        self.positions.iter().map(|p| p.weight).sum()
    }

    pub fn normalize_weights(&mut self) -> Result<(), QuantError> {
        let sum = self.weights_sum();

        if sum == 0.0 {
            return Err(QuantError::DivisionByZero);
        }

        for p in &mut self.positions {
            p.weight /= sum;
        }

        Ok(())
    }
}
