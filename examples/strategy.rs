use quantix::core::QuantError;
use quantix::strategy::moving_average::{moving_average_crossover_signals, simple_moving_average};

fn run_simple_moving_average_example() -> Result<(), QuantError> {
    let prices = vec![100.0, 101.0, 102.0, 104.0, 103.0, 105.0, 106.0];
    let short_sma = simple_moving_average(&prices, 2)?;
    let long_sma = simple_moving_average(&prices, 4)?;

    println!("=== Simple moving average ===");
    println!("Prices: {:?}", prices);
    println!("SMA (window=2): {:?}", short_sma);
    println!("SMA (window=4): {:?}", long_sma);
    println!();

    Ok(())
}

fn run_crossover_signals_example() -> Result<(), QuantError> {
    let prices = vec![100.0, 99.0, 101.0, 103.0, 105.0, 102.0, 106.0, 108.0];
    let signals = moving_average_crossover_signals(&prices, 2, 4)?;

    println!("=== Moving average crossover signals ===");
    println!("Prices: {:?}", prices);
    println!("Signals (short=2, long=4): {:?}", signals);
    println!();

    Ok(())
}

fn main() -> Result<(), QuantError> {
    run_simple_moving_average_example()?;
    run_crossover_signals_example()?;
    Ok(())
}
