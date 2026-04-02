use quantix::core::QuantError;
use quantix::portfolio::portfolio::{Portfolio, Position as PortfolioPosition};
use quantix::portfolio::position::Position as ValidatedPosition;

fn run_portfolio_returns_example() -> Result<(), QuantError> {
    let mut portfolio = Portfolio {
        positions: vec![
            PortfolioPosition {
                weight: 0.60,
                returns: vec![0.010, -0.005, 0.012, 0.004],
            },
            PortfolioPosition {
                weight: 0.40,
                returns: vec![0.004, 0.002, -0.003, 0.006],
            },
        ],
    };

    println!("=== Portfolio returns ===");
    println!("Initial weights sum: {:.6}", portfolio.weights_sum());
    println!("Portfolio period returns: {:?}", portfolio.returns()?);

    portfolio.normalize_weights()?;
    println!(
        "Weights sum after normalization: {:.6}",
        portfolio.weights_sum()
    );
    println!(
        "Portfolio period returns (normalized): {:?}",
        portfolio.returns()?
    );
    println!();

    Ok(())
}

fn run_validated_position_example() -> Result<(), QuantError> {
    let position = ValidatedPosition::new(0.25, vec![0.01, -0.02, 0.015, 0.0])?;

    println!("=== Validated position ===");
    println!("Weight: {:.6}", position.weight);
    println!("Returns: {:?}", position.returns);
    println!("Length: {}", position.len());
    println!("Is empty: {}", position.is_empty());
    println!();

    Ok(())
}

fn main() -> Result<(), QuantError> {
    run_portfolio_returns_example()?;
    run_validated_position_example()?;
    Ok(())
}
