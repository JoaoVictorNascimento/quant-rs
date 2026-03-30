#[derive(Debug)]
pub enum QuantError {
    InsufficientData,
    NonPositivePrice(f64),
    ZeroPrice,
    InvalidValue(f64),
    DivisionByZero,
}