"""Hostile input: everything a real CSV throws at this that a clean simulation
never will.

Every test here corresponds to something that either did break the code or
plausibly could.  The rule for this file is that the tool must respond to bad
input with either a correct answer or a clear error -- never with a confident
wrong number.
"""

import numpy as np
import pandas as pd
import pytest

from overfit_detector.data import make_synthetic_strategies
from overfit_detector.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from overfit_detector.metrics import is_degenerate, sharpe_ratio
from overfit_detector.pbo import cscv_pbo
from overfit_detector.report import build_report


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _frame(rng, n_periods=400, n_strategies=4, scale=0.01):
    return pd.DataFrame(
        rng.normal(0.0, scale, size=(n_periods, n_strategies)),
        columns=[f"s{i}" for i in range(n_strategies)],
    )


# --------------------------------------------------------------------------
# Degenerate columns and matrices
# --------------------------------------------------------------------------

def test_flat_cash_column_is_dropped_with_a_warning_not_a_crash(rng):
    """Regression: build_report used to die on an all-zero 'hold cash' column
    while cscv_pbo silently scored it 0.0 -- two paths, two behaviours."""
    data = _frame(rng, n_strategies=3)
    data["cash"] = 0.0
    with pytest.warns(UserWarning, match="zero-volatility"):
        report = build_report(data, n_blocks=10)
    assert "cash" not in report.table.index
    assert len(report.table) == 3
    # Dropped, but it was still a trial, so it still counts in the deflation.
    assert report.n_trials == 4


def test_cscv_drops_flat_columns_too(rng):
    data = _frame(rng, n_strategies=3)
    data["cash"] = 0.0
    with pytest.warns(UserWarning, match="zero-volatility"):
        result = cscv_pbo(data, n_blocks=10)
    assert result.n_strategies == 3
    assert "cash" not in (result.strategy_names or [])


def test_constant_nonzero_column_counts_as_degenerate(rng):
    data = _frame(rng, n_strategies=3)
    data["fixed"] = 0.01  # a constant return every period: no volatility
    assert is_degenerate(data["fixed"])
    with pytest.warns(UserWarning, match="zero-volatility"):
        assert cscv_pbo(data, n_blocks=10).n_strategies == 3


def test_all_degenerate_matrix_refuses_to_answer():
    """Regression: an all-zero matrix used to return a confident PBO of 1.0."""
    for matrix in (np.zeros((400, 4)), np.full((400, 4), 0.01)):
        with pytest.warns(UserWarning, match="zero-volatility"):
            with pytest.raises(ValueError, match="at least 2 non-degenerate"):
                cscv_pbo(pd.DataFrame(matrix), n_blocks=10)


def test_only_one_usable_column_left_is_an_error(rng):
    data = _frame(rng, n_strategies=1)
    data["cash"] = 0.0
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="at least 2 non-degenerate"):
            cscv_pbo(data, n_blocks=10)


# --------------------------------------------------------------------------
# Extreme trial counts
# --------------------------------------------------------------------------

def test_astronomical_trial_count_errors_instead_of_returning_inf():
    """Regression: N >= ~1e18 made ppf(1 - 1/N) collapse to ppf(1.0) = inf, and
    the DSR came back as a confident 0.0.  isf(1/N) keeps the tail."""
    assert np.isfinite(expected_max_sharpe(10**18, 1.0))
    assert np.isfinite(expected_max_sharpe(10**300, 1.0))
    # Still finite and ordered where the old implementation was already inf.
    assert expected_max_sharpe(10**300, 1.0) > expected_max_sharpe(10**18, 1.0)


def test_expected_max_sharpe_rejects_non_integral_trials():
    with pytest.raises(ValueError, match="whole number"):
        expected_max_sharpe(10.5, 1.0)


def test_expected_max_sharpe_rejects_non_finite_variance():
    with pytest.raises(ValueError, match="finite"):
        expected_max_sharpe(10, float("inf"))
    with pytest.raises(ValueError, match="finite"):
        expected_max_sharpe(10, float("nan"))


def test_huge_trial_count_drives_dsr_to_zero_not_to_nonsense():
    res = deflated_sharpe_ratio(0.1, 10**12, 1000, variance_of_trials=0.001)
    assert 0.0 <= res.dsr <= 1.0
    assert res.dsr < 0.01  # no Sharpe survives a trillion trials
    assert np.isfinite(res.expected_max_sharpe)


# --------------------------------------------------------------------------
# Extreme return series
# --------------------------------------------------------------------------

def test_total_wipeout_day_is_handled(rng):
    """A -100% return is a real thing that happens to real strategies."""
    data = _frame(rng)
    data.iloc[10, 0] = -1.0
    report = build_report(data, n_blocks=10)
    assert report.table.loc["s0", "sharpe"] < 0
    assert report.table.loc["s0", "skew"] < -5  # one huge negative outlier
    assert 0.0 <= report.table.loc["s0", "deflated_sharpe"] <= 1.0


def test_massive_outlier_produces_extreme_kurtosis_but_valid_output(rng):
    data = _frame(rng)
    data.iloc[0, 0] = 5.0
    report = build_report(data, n_blocks=10)
    assert report.table.loc["s0", "kurtosis"] > 100
    assert 0.0 <= report.table.loc["s0", "deflated_sharpe"] <= 1.0


@pytest.mark.parametrize("scale", [1e-12, 1e-6, 1.0, 1e6])
def test_wildly_different_return_scales_all_work(rng, scale):
    """Sharpe is scale-invariant, so results must not depend on whether returns
    are quoted as fractions, percents, or basis points."""
    data = _frame(rng, scale=scale)
    result = cscv_pbo(data, n_blocks=10)
    assert 0.0 <= result.pbo <= 1.0
    assert np.all(np.isfinite(result.logits))


def test_psr_rejects_moment_combinations_that_break_the_variance():
    """The estimator variance 1 - g3*SR + (g4-1)/4*SR^2 can go negative for
    impossible moment combinations; that must raise, not return a NaN."""
    with pytest.raises(ValueError, match="non-positive"):
        probabilistic_sharpe_ratio(2.0, 100, skew=5.0, kurtosis=3.0)


def test_nan_and_inf_are_rejected_everywhere(rng):
    data = _frame(rng)
    for bad in (np.nan, np.inf, -np.inf):
        broken = data.copy()
        broken.iloc[5, 1] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            cscv_pbo(broken, n_blocks=10)
        with pytest.raises(ValueError, match="NaN or infinite"):
            build_report(broken, n_blocks=10)


def test_non_numeric_input_gives_a_readable_error(rng):
    data = _frame(rng)
    data["s0"] = "not a number"
    with pytest.raises(ValueError, match="not numeric"):
        cscv_pbo(data, n_blocks=10)


# --------------------------------------------------------------------------
# Shape and block-count edges
# --------------------------------------------------------------------------

def test_minimum_viable_shape_works(rng):
    """Exactly 2 rows per block and exactly 2 strategies: the smallest input
    CSCV accepts at all."""
    data = _frame(rng, n_periods=8, n_strategies=2)
    result = cscv_pbo(data, n_blocks=4)
    assert result.n_splits == 6
    assert result.n_obs_used == 8


@pytest.mark.parametrize("n_blocks", [-4, 0, 2, 3, 7, 9, 11])
def test_bad_block_counts_rejected(rng, n_blocks):
    data = _frame(rng)
    with pytest.raises(ValueError, match="even integer"):
        cscv_pbo(data, n_blocks=n_blocks)


def test_more_blocks_than_rows_rejected(rng):
    data = _frame(rng, n_periods=12)
    with pytest.raises(ValueError, match="rows per block"):
        cscv_pbo(data, n_blocks=10)


@pytest.mark.parametrize("n_periods", [401, 405, 409])
def test_ragged_lengths_trim_from_the_start(rng, n_periods):
    """Trimming keeps the most recent data, which is the half a practitioner
    cares about."""
    data = _frame(rng, n_periods=n_periods)
    result = cscv_pbo(data, n_blocks=10)
    assert result.n_obs_used == (n_periods // 10) * 10
    assert result.n_obs_used + result.n_obs_dropped == n_periods


def test_split_cap_is_enforced_and_overridable(rng):
    data = _frame(rng, n_periods=2000, n_strategies=3)
    with pytest.raises(ValueError, match="cap"):
        cscv_pbo(data, n_blocks=40)
    with pytest.warns(UserWarning, match="combinatorial splits"):
        assert cscv_pbo(data, n_blocks=22, force=True).n_splits == 705432


def test_1d_input_rejected(rng):
    with pytest.raises(ValueError, match="2-D"):
        cscv_pbo(rng.normal(size=400), n_blocks=10)


# --------------------------------------------------------------------------
# Naming and ordering traps
# --------------------------------------------------------------------------

def test_duplicate_column_names_rejected(rng):
    """Regression: results were keyed by name, so two columns called 'a'
    collapsed and best_dsr could describe a different column than
    best_strategy named."""
    data = _frame(rng, n_strategies=3)
    data.columns = ["a", "a", "b"]
    with pytest.raises(ValueError, match="duplicate strategy column name"):
        build_report(data, n_blocks=10)


def test_best_dsr_matches_the_named_best_strategy(rng):
    """The positional keying must survive lookalike names."""
    data = _frame(rng, n_strategies=5)
    data.columns = ["s", "s ", "s_", "S", "s0"]  # confusable but distinct
    report = build_report(data, n_blocks=10)
    assert report.best_dsr.sharpe == pytest.approx(
        report.table.loc[report.best_strategy, "sharpe"]
    )


def test_unsorted_datetime_index_warns(rng):
    """Regression: a newest-first CSV export silently produced different blocks
    and a different PBO with no warning at all."""
    data = _frame(rng, n_periods=500)
    data.index = pd.bdate_range("2020-01-01", periods=500)
    with pytest.warns(UserWarning, match="not in increasing order"):
        cscv_pbo(data.iloc[::-1], n_blocks=10)
    with pytest.warns(UserWarning, match="not in increasing order"):
        cscv_pbo(data.sample(frac=1.0, random_state=1), n_blocks=10)


def test_sorted_or_indexless_frames_do_not_warn(rng):
    """The warning must not cry wolf on ordinary input."""
    import warnings

    data = _frame(rng, n_periods=500)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cscv_pbo(data, n_blocks=10)  # RangeIndex
        dated = data.copy()
        dated.index = pd.bdate_range("2020-01-01", periods=500)
        cscv_pbo(dated, n_blocks=10)  # sorted DatetimeIndex


# --------------------------------------------------------------------------
# Ties
# --------------------------------------------------------------------------

def test_identical_strategies_land_exactly_on_the_median(rng):
    """With every column identical there is nothing to select.  Midrank puts
    the 'winner' exactly at the median: omega = 0.5, logit = 0.  The paper's
    Pr[lambda <= 0] convention then counts that as overfit, which is the
    honest reading -- the selection carried no information whatsoever."""
    column = rng.normal(0.0, 0.01, size=400)
    data = pd.DataFrame({f"s{i}": column for i in range(5)})
    result = cscv_pbo(data, n_blocks=10)
    assert np.allclose(result.logits, 0.0)
    assert result.mean_relative_rank == pytest.approx(0.5)
    assert result.pbo == 1.0


def test_partial_ties_use_midrank(rng):
    """Three tied columns and one clear loser: the tied winner sits at the
    average of the ranks the tie spans, not at the bottom of them."""
    shared = rng.normal(0.0, 0.01, size=400)
    loser = shared - 0.05  # same shape, far worse mean => always ranks last
    data = pd.DataFrame({"a": shared, "b": shared, "c": shared, "loser": loser})
    result = cscv_pbo(data, n_blocks=10)
    # Ranks are 2, 3, 4 for the tied trio (loser takes rank 1); midrank = 3.
    assert result.mean_relative_rank == pytest.approx(3.0 / 5.0)
    assert result.pbo == 0.0


def test_duplicate_columns_do_not_inflate_the_deflation(rng):
    """Correlated trials are effectively fewer trials.  Identical columns give
    V[SR] = 0, which correctly collapses the hurdle to zero -- documented here
    because it is a trap: N alone does not measure how much searching you did."""
    column = rng.normal(0.001, 0.01, size=400)
    data = pd.DataFrame({f"s{i}": column for i in range(5)})
    report = build_report(data, n_blocks=10)
    assert report.variance_of_trials == pytest.approx(0.0)
    assert report.best_dsr.expected_max_sharpe == 0.0
    assert report.best_dsr.dsr == pytest.approx(report.best_dsr.psr_zero)


# --------------------------------------------------------------------------
# Single-series entry points
# --------------------------------------------------------------------------

def test_sharpe_rejects_a_multi_column_frame(rng):
    """Regression: a DataFrame used to be ravelled into one long series and
    scored as if it were a single strategy."""
    with pytest.raises(ValueError, match="single return series"):
        sharpe_ratio(_frame(rng, n_strategies=3))


def test_sharpe_accepts_a_one_column_frame(rng):
    data = _frame(rng, n_strategies=1)
    assert sharpe_ratio(data) == pytest.approx(sharpe_ratio(data["s0"]))


def test_report_rejects_a_single_variant():
    data = make_synthetic_strategies(n_strategies=1, n_periods=400, seed=1)
    with pytest.raises(ValueError, match="at least 2 non-degenerate"):
        build_report(data, n_blocks=10)
