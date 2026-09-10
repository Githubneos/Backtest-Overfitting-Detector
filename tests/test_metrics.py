import numpy as np
import pandas as pd
import pytest

from overfit_detector.metrics import (
    annualize_sharpe,
    annualized_sharpe_ratio,
    returns_moments,
    sharpe_ratio,
)


def test_sharpe_matches_hand_computation():
    r = np.array([0.01, -0.02, 0.03, 0.00, 0.02])
    expected = (r.mean() - 0.0) / r.std(ddof=1)
    assert sharpe_ratio(r) == pytest.approx(expected)


def test_sharpe_uses_sample_std_not_population():
    r = np.array([0.01, -0.01, 0.02, 0.03])
    # ddof=1 gives a larger denominator, hence a smaller Sharpe than ddof=0.
    assert sharpe_ratio(r) < (r.mean() / r.std(ddof=0))


def test_risk_free_shifts_numerator():
    r = np.array([0.01, 0.02, 0.03, 0.015])
    assert sharpe_ratio(r, risk_free=0.01) == pytest.approx(
        (r.mean() - 0.01) / r.std(ddof=1)
    )


def test_accepts_pandas_series():
    r = pd.Series([0.01, -0.02, 0.03, 0.00, 0.02])
    assert sharpe_ratio(r) == pytest.approx(sharpe_ratio(r.to_numpy()))


def test_annualization_is_sqrt_scaling():
    assert annualize_sharpe(0.1, 252) == pytest.approx(0.1 * np.sqrt(252))
    r = np.array([0.01, -0.02, 0.03, 0.00, 0.02])
    assert annualized_sharpe_ratio(r, 252) == pytest.approx(
        sharpe_ratio(r) * np.sqrt(252)
    )


def test_zero_volatility_raises():
    with pytest.raises(ValueError, match="zero volatility"):
        sharpe_ratio(np.ones(10) * 0.01)


def test_too_few_observations_raises():
    with pytest.raises(ValueError, match="at least 2"):
        sharpe_ratio([0.01])


def test_nan_input_raises():
    with pytest.raises(ValueError, match="NaN"):
        sharpe_ratio([0.01, np.nan, 0.02])


def test_moments_of_normal_sample_are_near_zero_and_three():
    rng = np.random.default_rng(0)
    r = rng.normal(size=200_000)
    n, skew, kurt = returns_moments(r)
    assert n == 200_000
    assert skew == pytest.approx(0.0, abs=0.02)
    # Non-excess (Pearson) kurtosis: normal == 3.0, not 0.0.
    assert kurt == pytest.approx(3.0, abs=0.05)
