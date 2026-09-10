"""Monte Carlo validation of the statistics against simulated ground truth.

These are the tests that would catch a wrong formula.  Everything else in the
suite checks that the code computes what I *think* the formula is; this file
checks that the formula matches what actually happens when you simulate the
process it claims to describe.

All simulations use fixed seeds, so the assertions are deterministic.  Each
threshold carries a comment naming the theoretical value it brackets.
"""

import numpy as np
import pytest
from scipy import stats

from overfit_detector.data import make_synthetic_strategies
from overfit_detector.deflated_sharpe import (
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from overfit_detector.metrics import sharpe_ratio
from overfit_detector.pbo import cscv_pbo

pytestmark = pytest.mark.slow


# --------------------------------------------------------------------------
# SR*: does the Gumbel formula match a simulated maximum?
# --------------------------------------------------------------------------

def _simulate_expected_max_sharpe(n_trials, n_obs, n_reps, rng):
    """Empirical E[max Sharpe] over n_trials skill-free strategies."""
    maxima = np.empty(n_reps)
    for i in range(n_reps):
        r = rng.normal(0.0, 0.01, size=(n_obs, n_trials))
        maxima[i] = (r.mean(axis=0) / r.std(axis=0, ddof=1)).max()
    return float(maxima.mean())


@pytest.mark.parametrize("n_trials,tolerance", [(10, 0.06), (50, 0.04), (200, 0.03), (1000, 0.03)])
def test_expected_max_sharpe_matches_simulation(n_trials, tolerance):
    """SR* should reproduce the simulated expected maximum.

    Under the null the per-period Sharpe estimator has variance ~1/T, which is
    what we hand the formula as V[SR]."""
    n_obs, rng = 1000, np.random.default_rng(1234 + n_trials)
    simulated = _simulate_expected_max_sharpe(n_trials, n_obs, n_reps=300, rng=rng)
    predicted = expected_max_sharpe(n_trials, 1.0 / n_obs)
    assert predicted == pytest.approx(simulated, rel=tolerance)


def test_gumbel_approximation_improves_with_more_trials():
    """The formula is an extreme-value *limit*, so its error must shrink as N
    grows.  Measured: ~2.4% at N=10 down to ~0.1% at N=1000."""
    n_obs, rng = 1000, np.random.default_rng(99)
    errors = {}
    for n_trials in (10, 1000):
        simulated = _simulate_expected_max_sharpe(n_trials, n_obs, n_reps=400, rng=rng)
        predicted = expected_max_sharpe(n_trials, 1.0 / n_obs)
        errors[n_trials] = abs(predicted / simulated - 1.0)
    assert errors[1000] < errors[10]
    assert errors[1000] < 0.01


def test_expected_max_sharpe_beats_the_naive_zero_hurdle():
    """Sanity: the whole point is that the hurdle is well above zero and grows
    with the number of trials tried."""
    data = make_synthetic_strategies(n_strategies=100, n_periods=1000, mu=0.0, seed=5)
    observed_max = max(sharpe_ratio(data[c]) for c in data.columns)
    variance = float(np.var([sharpe_ratio(data[c]) for c in data.columns], ddof=1))
    hurdle = expected_max_sharpe(100, variance)
    # A skill-free search of 100 variants produces a best Sharpe of roughly the
    # size of the hurdle -- that is what "no evidence" looks like numerically.
    assert 0.5 * hurdle < observed_max < 2.0 * hurdle


# --------------------------------------------------------------------------
# PSR calibration: under the null it must be Uniform(0, 1)
# --------------------------------------------------------------------------

def _psr_sample(n_obs, n_reps, generator, rng):
    out = np.empty(n_reps)
    for i in range(n_reps):
        r = generator(n_obs, rng)
        skew = float(stats.skew(r, bias=False))
        kurt = float(stats.kurtosis(r, fisher=False, bias=False))
        out[i] = probabilistic_sharpe_ratio(sharpe_ratio(r), n_obs, skew, kurt, 0.0)
    return out


@pytest.mark.parametrize("n_obs", [100, 500, 2000])
def test_psr_is_uniform_under_the_null(n_obs):
    """A calibrated probability must be uniform when the null is true.

    If PSR were systematically optimistic this test fails, and no amount of
    checking the formula against itself would have caught it."""
    rng = np.random.default_rng(7 + n_obs)
    ps = _psr_sample(n_obs, 2000, lambda t, g: g.normal(0.0, 0.01, t), rng)
    assert ps.mean() == pytest.approx(0.5, abs=0.03)  # uniform mean
    assert stats.kstest(ps, "uniform").pvalue > 0.01  # uniform shape
    assert np.mean(ps < 0.05) == pytest.approx(0.05, abs=0.02)  # nominal tail rate


@pytest.mark.parametrize(
    "label,generator",
    [
        ("student_t4", lambda t, g: g.standard_t(4, t) * 0.005),
        ("skewed_gamma", lambda t, g: (g.standard_gamma(1.0, t) - 1.0) * 0.01),
        ("negatively_skewed", lambda t, g: -(g.standard_gamma(1.0, t) - 1.0) * 0.01),
    ],
)
def test_psr_stays_calibrated_under_non_normal_returns(label, generator):
    """This is what the skew and kurtosis terms are *for*.  Fat tails (t with 4
    degrees of freedom, kurtosis is infinite in the limit) and strong skew must
    not blow up the nominal 5% tail rates."""
    rng = np.random.default_rng(hash(label) % 2**32)
    ps = _psr_sample(500, 2000, generator, rng)
    assert np.mean(ps < 0.05) == pytest.approx(0.05, abs=0.025)
    assert np.mean(ps > 0.95) == pytest.approx(0.05, abs=0.025)


def test_psr_has_power_against_a_real_edge():
    """Calibration is worthless without power: a genuinely good strategy must
    be detected nearly always."""
    rng = np.random.default_rng(11)
    detected = 0
    for _ in range(300):
        r = rng.normal(0.002, 0.01, size=500)  # per-period SR 0.2, ann. ~3.2
        detected += probabilistic_sharpe_ratio(sharpe_ratio(r), 500) > 0.95
    assert detected / 300 > 0.95


# --------------------------------------------------------------------------
# PBO calibration
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_strategies,expected", [(5, 0.55), (20, 0.51), (100, 0.45)])
def test_pbo_is_near_one_half_for_skill_free_variants(n_strategies, expected):
    """Averaged over independent datasets, PBO for pure noise sits near 0.5:
    which variant leads in-sample carries no information about out-of-sample.

    Note the mild *downward* drift as the number of variants grows (measured:
    0.55 / 0.51 / 0.45 for N = 5 / 20 / 100).  Any single dataset is very noisy
    -- individual runs range from 0.06 to 1.00 -- which is exactly why this is
    asserted on an average and not on one seed."""
    rng = np.random.default_rng(2024 + n_strategies)
    pbos = [
        cscv_pbo(rng.normal(0.0, 0.01, size=(600, n_strategies)), n_blocks=8).pbo
        for _ in range(30)
    ]
    assert np.mean(pbos) == pytest.approx(expected, abs=0.12)


def test_pbo_single_dataset_variance_is_large():
    """Pins the caveat above: one dataset is not an estimate of PBO's null.
    A reader who tries to conclude something from a single noise run should see
    this test and understand why they cannot."""
    rng = np.random.default_rng(31)
    pbos = [
        cscv_pbo(rng.normal(0.0, 0.01, size=(600, 20)), n_blocks=8).pbo
        for _ in range(30)
    ]
    assert np.std(pbos) > 0.08  # measured ~0.17
    assert max(pbos) - min(pbos) > 0.3


def test_pbo_power_against_a_real_edge():
    """With genuine signal, PBO must be low essentially every time -- not just
    on the one seed the unit tests happen to use."""
    pbos = []
    for seed in range(10):
        data = make_synthetic_strategies(
            n_strategies=30, n_periods=1000, mu=0.0, sigma=0.01,
            n_signal=1, signal_mu=0.002, seed=seed,
        )
        pbos.append(cscv_pbo(data, n_blocks=8).pbo)
    assert max(pbos) < 0.2
    assert np.mean(pbos) < 0.05
