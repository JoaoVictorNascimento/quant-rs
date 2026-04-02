use quantix::core::QuantError;
use quantix::portfolio::portfolio::{Portfolio, Position};
use quantix::portfolio::position::Position as PortfolioPosition;

fn assert_approx_eq(a: f64, b: f64) {
    let eps = 1e-12_f64.max(a.abs().max(b.abs()) * 1e-12);
    assert!(
        (a - b).abs() < eps,
        "expected {a} ≈ {b}, diff {}",
        (a - b).abs()
    );
}

fn sample_portfolio() -> Portfolio {
    Portfolio {
        positions: vec![
            Position {
                weight: 0.6,
                returns: vec![0.01, 0.02, -0.01],
            },
            Position {
                weight: 0.4,
                returns: vec![0.03, -0.01, 0.02],
            },
        ],
    }
}

// ── returns ───────────────────────────────────────────────────────────────────

#[test]
fn returns_single_position_equals_weighted_series() {
    let portfolio = Portfolio {
        positions: vec![Position {
            weight: 0.5,
            returns: vec![0.02, -0.01, 0.03],
        }],
    };

    let r = portfolio.returns().unwrap();
    assert_eq!(r.len(), 3);
    assert_approx_eq(r[0], 0.5 * 0.02);
    assert_approx_eq(r[1], 0.5 * -0.01);
    assert_approx_eq(r[2], 0.5 * 0.03);
}

#[test]
fn returns_multiple_positions_weighted_sum() {
    let portfolio = sample_portfolio();
    let r = portfolio.returns().unwrap();

    assert_eq!(r.len(), 3);
    assert_approx_eq(r[0], 0.018);
    assert_approx_eq(r[1], 0.008);
    assert_approx_eq(r[2], 0.002);
}

#[test]
fn returns_supports_short_weight() {
    let portfolio = Portfolio {
        positions: vec![
            Position {
                weight: 1.0,
                returns: vec![0.01, 0.01],
            },
            Position {
                weight: -0.5,
                returns: vec![0.02, -0.02],
            },
        ],
    };

    let r = portfolio.returns().unwrap();
    assert_approx_eq(r[0], 1.0 * 0.01 + (-0.5) * 0.02);
    assert_approx_eq(r[1], 1.0 * 0.01 + (-0.5) * -0.02);
}

#[test]
fn returns_empty_positions_is_error() {
    let portfolio = Portfolio { positions: vec![] };
    assert!(matches!(
        portfolio.returns(),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn returns_empty_position_series_is_error() {
    let portfolio = Portfolio {
        positions: vec![Position {
            weight: 1.0,
            returns: vec![],
        }],
    };
    assert!(matches!(
        portfolio.returns(),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn returns_mismatched_series_length_is_error() {
    let portfolio = Portfolio {
        positions: vec![
            Position {
                weight: 0.5,
                returns: vec![0.01, 0.02],
            },
            Position {
                weight: 0.5,
                returns: vec![0.03],
            },
        ],
    };
    assert!(matches!(
        portfolio.returns(),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn returns_invalid_weight_is_error() {
    let portfolio = Portfolio {
        positions: vec![
            Position {
                weight: f64::NAN,
                returns: vec![0.01, 0.02],
            },
            Position {
                weight: 1.0,
                returns: vec![0.01, 0.02],
            },
        ],
    };
    assert!(matches!(
        portfolio.returns(),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn returns_invalid_return_value_is_error() {
    let portfolio = Portfolio {
        positions: vec![Position {
            weight: 1.0,
            returns: vec![0.01, f64::INFINITY],
        }],
    };
    assert!(matches!(
        portfolio.returns(),
        Err(QuantError::InvalidValue(_))
    ));
}

// ── weights_sum ───────────────────────────────────────────────────────────────

#[test]
fn weights_sum_basic() {
    let portfolio = sample_portfolio();
    assert_approx_eq(portfolio.weights_sum(), 1.0);
}

#[test]
fn weights_sum_can_be_negative_or_zero() {
    let portfolio = Portfolio {
        positions: vec![
            Position {
                weight: 0.5,
                returns: vec![0.0],
            },
            Position {
                weight: -0.5,
                returns: vec![0.0],
            },
        ],
    };
    assert_approx_eq(portfolio.weights_sum(), 0.0);
}

#[test]
fn weights_sum_empty_is_zero() {
    let portfolio = Portfolio { positions: vec![] };
    assert_approx_eq(portfolio.weights_sum(), 0.0);
}

// ── normalize_weights ─────────────────────────────────────────────────────────

#[test]
fn normalize_weights_scales_to_one() {
    let mut portfolio = Portfolio {
        positions: vec![
            Position {
                weight: 2.0,
                returns: vec![0.01, 0.02],
            },
            Position {
                weight: 3.0,
                returns: vec![0.03, 0.04],
            },
        ],
    };

    portfolio.normalize_weights().unwrap();
    assert_approx_eq(portfolio.positions[0].weight, 0.4);
    assert_approx_eq(portfolio.positions[1].weight, 0.6);
    assert_approx_eq(portfolio.weights_sum(), 1.0);
}

#[test]
fn normalize_weights_preserves_relative_proportions() {
    let mut portfolio = Portfolio {
        positions: vec![
            Position {
                weight: 1.5,
                returns: vec![0.0],
            },
            Position {
                weight: 0.5,
                returns: vec![0.0],
            },
        ],
    };

    portfolio.normalize_weights().unwrap();
    let ratio = portfolio.positions[0].weight / portfolio.positions[1].weight;
    assert_approx_eq(ratio, 3.0);
}

#[test]
fn normalize_weights_negative_total_also_normalizes() {
    let mut portfolio = Portfolio {
        positions: vec![
            Position {
                weight: -2.0,
                returns: vec![0.0],
            },
            Position {
                weight: -3.0,
                returns: vec![0.0],
            },
        ],
    };

    portfolio.normalize_weights().unwrap();
    assert_approx_eq(portfolio.positions[0].weight, 0.4);
    assert_approx_eq(portfolio.positions[1].weight, 0.6);
    assert_approx_eq(portfolio.weights_sum(), 1.0);
}

#[test]
fn normalize_weights_zero_sum_is_division_by_zero() {
    let mut portfolio = Portfolio {
        positions: vec![
            Position {
                weight: 1.0,
                returns: vec![0.0],
            },
            Position {
                weight: -1.0,
                returns: vec![0.0],
            },
        ],
    };
    assert!(matches!(
        portfolio.normalize_weights(),
        Err(QuantError::DivisionByZero)
    ));
}

#[test]
fn normalize_weights_empty_positions_is_division_by_zero() {
    let mut portfolio = Portfolio { positions: vec![] };
    assert!(matches!(
        portfolio.normalize_weights(),
        Err(QuantError::DivisionByZero)
    ));
}

// ── position_new ──────────────────────────────────────────────────────────────

#[test]
fn position_new_basic() {
    let p = PortfolioPosition::new(0.5, vec![0.01, -0.02, 0.03]).unwrap();
    assert_approx_eq(p.weight, 0.5);
    assert_eq!(p.returns.len(), 3);
    assert_approx_eq(p.returns[0], 0.01);
    assert_approx_eq(p.returns[1], -0.02);
    assert_approx_eq(p.returns[2], 0.03);
}

#[test]
fn position_new_allows_zero_and_negative_weight() {
    let p_zero = PortfolioPosition::new(0.0, vec![0.01]).unwrap();
    let p_negative = PortfolioPosition::new(-1.0, vec![0.01]).unwrap();
    assert_approx_eq(p_zero.weight, 0.0);
    assert_approx_eq(p_negative.weight, -1.0);
}

#[test]
fn position_new_invalid_weight_is_error() {
    assert!(matches!(
        PortfolioPosition::new(f64::NAN, vec![0.01, 0.02]),
        Err(QuantError::InvalidValue(_))
    ));
    assert!(matches!(
        PortfolioPosition::new(f64::INFINITY, vec![0.01, 0.02]),
        Err(QuantError::InvalidValue(_))
    ));
}

#[test]
fn position_new_empty_returns_is_error() {
    assert!(matches!(
        PortfolioPosition::new(1.0, vec![]),
        Err(QuantError::InsufficientData)
    ));
}

#[test]
fn position_new_invalid_return_is_error() {
    assert!(matches!(
        PortfolioPosition::new(1.0, vec![0.01, f64::NAN]),
        Err(QuantError::InvalidValue(_))
    ));
    assert!(matches!(
        PortfolioPosition::new(1.0, vec![0.01, f64::NEG_INFINITY]),
        Err(QuantError::InvalidValue(_))
    ));
}

// ── position_len ──────────────────────────────────────────────────────────────

#[test]
fn position_len_matches_returns_size() {
    let p = PortfolioPosition::new(1.0, vec![0.01, 0.02, 0.03, 0.04]).unwrap();
    assert_eq!(p.len(), 4);
}

#[test]
fn position_len_single_value() {
    let p = PortfolioPosition::new(1.0, vec![0.01]).unwrap();
    assert_eq!(p.len(), 1);
}

// ── position_is_empty ─────────────────────────────────────────────────────────

#[test]
fn position_is_empty_is_false_for_valid_position() {
    let p = PortfolioPosition::new(1.0, vec![0.01]).unwrap();
    assert!(!p.is_empty());
}
