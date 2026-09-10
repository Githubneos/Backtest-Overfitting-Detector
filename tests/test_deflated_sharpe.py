import math

import numpy as np
import pytest
from scipy import stats

from overfit_detector.deflated_sharpe import (
    EULER_MASCHERONI,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


# --------------------------------------------------------------------------
# expected_max_sharpe
# --------------------------------------------------------------------------

def test_expected_max_sharpe_matches_formula_by_hand():
    n, var = 100, 1.0
    g = EULER_MASCHERONI
    expected = math.sqrt(var) * (
        (1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * math.e))
    )
    got = expected_max_sharpe(n, var)
    assert got == pytest.approx(expected)
    # With unit variance this sits in the usual ~2.7-sigma region for N=100.
    assert 2.5 < got < 3.0


def test_expected_max_sharpe_increases_with_trials():
    vals = [expected_max_sharpe(n, 1.0) for n in (2, 10, 100, 1000, 10_000)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_expected_max_sharpe_scales_with_sqrt_variance():
    assert expected_max_sharpe(50, 4.0) == pytest.approx(2 * expected_max_sharpe(50, 1.0))


def test_expected_max_sharpe_degenerate_cases():
    assert expected_max_sharpe(1, 1.0) == 0.0  # nothing was selected
    assert expected_max_sharpe(100, 0.0) == 0.0  # trials are indistinguishable


def test_expected_max_sharpe_rejects_bad_input():
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 1.0)
    with pytest.raises(ValueError):
        expected_max_sharpe(10, -1.0)


# --------------------------------------------------------------------------
# probabilistic_sharpe_ratio
# --------------------------------------------------------------------------

def test_psr_matches_closed_form_under_gaussian_moments():
    # With skew=0 and kurtosis=3 the estimator variance is 1 + SR^2/2 --
    # not 1: the SR^2 term survives even for normal returns.
    sr, t = 0.1, 250
    denom = math.sqrt(1.0 + 0.5 * sr**2)
    assert probabilistic_sharpe_ratio(sr, t, skew=0.0, kurtosis=3.0) == pytest.approx(
        stats.norm.cdf(sr * math.sqrt(t - 1) / denom)
    )


def test_psr_reduces_to_normal_cdf_at_zero_sharpe():
    assert probabilistic_sharpe_ratio(0.0, 250) == pytest.approx(0.5)


def test_psr_is_one_half_when_sharpe_equals_benchmark():
    assert probabilistic_sharpe_ratio(0.2, 500, benchmark_sharpe=0.2) == pytest.approx(0.5)


def test_psr_monotonic_in_sharpe_and_in_sample_length():
    assert probabilistic_sharpe_ratio(0.05, 500) < probabilistic_sharpe_ratio(0.10, 500)
    assert probabilistic_sharpe_ratio(0.05, 100) < probabilistic_sharpe_ratio(0.05, 1000)


def test_psr_penalises_fat_tails_and_negative_skew():
    base = probabilistic_sharpe_ratio(0.1, 500, skew=0.0, kurtosis=3.0)
    fat = probabilistic_sharpe_ratio(0.1, 500, skew=0.0, kurtosis=9.0)
    left = probabilistic_sharpe_ratio(0.1, 500, skew=-1.5, kurtosis=3.0)
    right = probabilistic_sharpe_ratio(0.1, 500, skew=1.5, kurtosis=3.0)
    assert fat < base  # fat tails widen the estimator's standard error
    assert left < base < right  # negative skew is punished, positive skew rewarded


def test_psr_rejects_bad_input():
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(0.1, 1)
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(0.1, 100, kurtosis=0.0)
    with pytest.raises(ValueError, match="non-positive"):
        # Enormous positive skew with a large Sharpe drives the variance negative.
        probabilistic_sharpe_ratio(2.0, 100, skew=5.0, kurtosis=3.0)


# --------------------------------------------------------------------------
# deflated_sharpe_ratio
# --------------------------------------------------------------------------

def test_dsr_never_exceeds_undeflated_psr():
    res = deflated_sharpe_ratio(sharpe=0.15, n_trials=50, n_obs=500, variance_of_trials=0.01)
    assert res.expected_max_sharpe > 0
    assert res.dsr < res.psr_zero


def test_dsr_equals_psr_when_single_trial():
    res = deflated_sharpe_ratio(sharpe=0.15, n_trials=1, n_obs=500, variance_of_trials=0.01)
    assert res.expected_max_sharpe == 0.0
    assert res.dsr == pytest.approx(res.psr_zero)


def test_dsr_falls_as_more_variants_are_tried():
    vals = [
        deflated_sharpe_ratio(0.12, n, 500, variance_of_trials=0.005).dsr
        for n in (1, 10, 100, 5000)
    ]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert vals[-1] < 0.5  # a Sharpe this ordinary is unremarkable after 5000 tries


def test_dsr_result_exposes_all_intermediates():
    res = deflated_sharpe_ratio(0.15, 40, 500, variance_of_trials=0.01, skew=-0.5, kurtosis=6.0)
    assert res.n_trials == 40 and res.n_obs == 500
    assert res.skew == -0.5 and res.kurtosis == 6.0
    assert res.variance_of_trials == 0.01
    assert 0.0 <= res.dsr <= 1.0
    assert "DSR=" in str(res)


def test_from_returns_estimates_trial_variance_from_observed_sharpes():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.01, size=600)
    sharpes = [0.02, 0.05, -0.01, 0.09, 0.03]
    res = deflated_sharpe_ratio_from_returns(returns, n_trials=5, all_trial_sharpes=sharpes)
    assert res.variance_of_trials == pytest.approx(np.var(sharpes, ddof=1))
    assert res.n_obs == 600


def test_from_returns_requires_a_variance_source():
    with pytest.raises(ValueError, match="variance_of_trials"):
        deflated_sharpe_ratio_from_returns(np.random.default_rng(0).normal(size=100), n_trials=5)
