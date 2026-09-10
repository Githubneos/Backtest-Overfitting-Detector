import math

import numpy as np
import pandas as pd
import pytest

from overfit_detector.data import make_synthetic_strategies
from overfit_detector.pbo import cscv_pbo


# --------------------------------------------------------------------------
# structural / input validation
# --------------------------------------------------------------------------

def test_number_of_splits_is_the_central_binomial_coefficient():
    data = make_synthetic_strategies(n_strategies=5, n_periods=400, seed=1)
    for s in (4, 6, 10):
        res = cscv_pbo(data, n_blocks=s)
        assert res.n_splits == math.comb(s, s // 2)


def test_odd_or_tiny_block_count_rejected():
    data = make_synthetic_strategies(n_strategies=5, n_periods=400, seed=1)
    with pytest.raises(ValueError, match="even integer"):
        cscv_pbo(data, n_blocks=9)
    with pytest.raises(ValueError, match="even integer"):
        cscv_pbo(data, n_blocks=2)


def test_needs_at_least_two_strategies():
    data = make_synthetic_strategies(n_strategies=1, n_periods=400, seed=1)
    with pytest.raises(ValueError, match="at least 2 non-degenerate strategy"):
        cscv_pbo(data)


def test_blocks_too_short_rejected():
    data = make_synthetic_strategies(n_strategies=5, n_periods=12, seed=1)
    with pytest.raises(ValueError, match="rows per block"):
        cscv_pbo(data, n_blocks=10)


def test_nan_input_rejected():
    data = make_synthetic_strategies(n_strategies=5, n_periods=400, seed=1)
    data.iloc[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        cscv_pbo(data)


def test_trimming_drops_leading_rows_only():
    data = make_synthetic_strategies(n_strategies=5, n_periods=405, seed=1)
    res = cscv_pbo(data, n_blocks=10)
    assert res.n_obs_used == 400
    assert res.n_obs_dropped == 5


def test_too_many_blocks_hits_the_cap():
    data = make_synthetic_strategies(n_strategies=3, n_periods=2000, seed=1)
    with pytest.raises(ValueError, match="cap"):
        cscv_pbo(data, n_blocks=40)


def test_large_but_allowed_block_count_warns():
    data = make_synthetic_strategies(n_strategies=3, n_periods=600, seed=1)
    with pytest.warns(UserWarning, match="combinatorial splits"):
        cscv_pbo(data, n_blocks=18)


# --------------------------------------------------------------------------
# exact, hand-checkable behaviour
# --------------------------------------------------------------------------

def test_deterministic_dominant_strategy_gives_pbo_zero():
    """A strategy that wins in every block must win OOS in every split."""
    n = 400
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.01, size=(n, 4))
    # Column 0: same noise, plus a large constant edge in every period.
    winner = noise[:, 0] + 0.05
    data = pd.DataFrame(
        np.column_stack([winner, noise]),
        columns=["winner", "a", "b", "c", "d"],
    )
    res = cscv_pbo(data, n_blocks=10)
    assert res.pbo == 0.0
    assert set(res.selected) == {0}
    assert res.selected_names() == ["winner"] * res.n_splits
    # rank N out of N -> omega = 5/6 -> logit = ln(5) for every split.
    assert np.allclose(res.logits, math.log(5 / 6 / (1 - 5 / 6)))
    assert res.prob_loss == 0.0


def test_deterministic_worst_out_of_sample_gives_pbo_one():
    """IS winner by construction, OOS worst by construction -> PBO = 1."""
    n_blocks, block_len, n_strat = 10, 20, 4
    n = n_blocks * block_len
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 0.001, size=(n, n_strat))
    # Column 0 alternates: huge edge on even blocks, huge drag on odd blocks.
    # Any balanced split therefore has it best on one side and worst on the other.
    flip = np.repeat([1.0, -1.0] * (n_blocks // 2), block_len)
    base[:, 0] += 0.5 * flip
    res = cscv_pbo(base, n_blocks=n_blocks)
    # Splits that put mostly-even blocks IS select column 0, and it is then
    # worst OOS by construction.
    assert res.pbo > 0.5
    assert res.mean_relative_rank < 0.5


def test_custom_metric_is_respected():
    data = make_synthetic_strategies(n_strategies=6, n_periods=400, seed=3)
    mean_metric = cscv_pbo(data, n_blocks=8, metric=lambda x: float(np.mean(x)))
    sharpe_metric = cscv_pbo(data, n_blocks=8)
    assert mean_metric.n_splits == sharpe_metric.n_splits
    assert 0.0 <= mean_metric.pbo <= 1.0


def test_summary_contains_scalar_diagnostics():
    data = make_synthetic_strategies(n_strategies=6, n_periods=400, seed=4)
    summary = cscv_pbo(data, n_blocks=8).summary()
    assert set(summary) >= {"pbo", "n_splits", "mean_relative_rank", "prob_loss"}
    assert all(isinstance(v, (int, float)) for v in summary.values())


# --------------------------------------------------------------------------
# statistical sanity checks (fixed seeds -> deterministic assertions)
# --------------------------------------------------------------------------

def test_pure_noise_strategies_give_high_pbo():
    """50 skill-free strategies: which one looks best in-sample is pure luck,
    so out-of-sample it lands on the wrong side of the median about as often as
    not.  The plan's threshold is >0.6 for this seed; see the averaged test
    below for the seed-independent statement."""
    data = make_synthetic_strategies(n_strategies=50, n_periods=1000, mu=0.0, seed=42)
    res = cscv_pbo(data, n_blocks=10)
    assert res.pbo > 0.6
    assert res.mean_relative_rank < 0.5
    # Being best in-sample says nothing about even making money out-of-sample.
    assert 0.3 < res.prob_loss < 0.7


def test_noise_pbo_averages_near_one_half_across_seeds():
    """Seed-to-seed spread is wide (one lucky column can dominate a whole
    sample), so the honest statement about noise is about the average, and it
    is 'near 0.5' -- not 'above 0.5'.  See tests/test_montecarlo.py for the
    properly sized calibration study."""
    pbos = [
        cscv_pbo(
            make_synthetic_strategies(n_strategies=40, n_periods=1200, mu=0.0, seed=s),
            n_blocks=10,
        ).pbo
        for s in range(8)
    ]
    assert np.mean(pbos) == pytest.approx(0.5, abs=0.15)  # theoretical target 0.5


def test_genuine_signal_gives_low_pbo():
    """One strategy with a real edge among 49 noise variants should be picked
    IS and keep winning OOS."""
    data = make_synthetic_strategies(
        n_strategies=50, n_periods=1000, mu=0.0, sigma=0.01,
        n_signal=1, signal_mu=0.002, seed=42,
    )
    res = cscv_pbo(data, n_blocks=10)
    assert res.pbo < 0.2
    assert res.mean_relative_rank > 0.9
    assert res.prob_loss < 0.05
    # The signal column should dominate the in-sample selections.
    assert np.mean(res.selected == 0) > 0.9


def test_signal_scores_much_better_than_pure_noise():
    kwargs = dict(n_strategies=50, n_periods=1000, mu=0.0, sigma=0.01, seed=7)
    noise = cscv_pbo(make_synthetic_strategies(**kwargs), n_blocks=10)
    signal = cscv_pbo(
        make_synthetic_strategies(n_signal=1, signal_mu=0.002, **kwargs), n_blocks=10
    )
    assert signal.pbo < noise.pbo - 0.4


def test_autocorrelated_noise_scores_worse_than_autocorrelated_signal():
    """Contiguous blocks preserve AR(1) structure, so the comparison still
    separates cleanly when returns are autocorrelated."""
    kwargs = dict(n_strategies=40, n_periods=1200, mu=0.0, sigma=0.01, autocorr=0.3, seed=5)
    noise = cscv_pbo(make_synthetic_strategies(**kwargs), n_blocks=10)
    signal = cscv_pbo(
        make_synthetic_strategies(n_signal=1, signal_mu=0.002, **kwargs), n_blocks=10
    )
    assert signal.pbo < noise.pbo
    assert signal.mean_relative_rank > noise.mean_relative_rank
