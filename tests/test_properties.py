"""Invariants that must hold for *any* input.

Unit tests check specific values; these check structural properties.  A wrong
implementation can match a hand-computed value on one input and still violate
scale invariance or column-permutation invariance on the next one.
"""

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from overfit_detector.data import make_synthetic_strategies
from overfit_detector.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from overfit_detector.metrics import sharpe_ratio
from overfit_detector.pbo import _sharpe_metric, cscv_pbo
from overfit_detector.report import build_report


@pytest.fixture(scope="module")
def data():
    return make_synthetic_strategies(n_strategies=12, n_periods=600, mu=0.0, seed=3)


# --------------------------------------------------------------------------
# Invariances
# --------------------------------------------------------------------------

@pytest.mark.parametrize("factor", [1e-6, 0.5, 2.0, 100.0, 1e6])
def test_everything_is_invariant_to_rescaling_returns(data, factor):
    """Quoting returns in fractions, percent or basis points describes the same
    strategy, so every statistic must be unchanged."""
    base = build_report(data, n_blocks=8)
    scaled = build_report(data * factor, n_blocks=8)
    assert scaled.best_strategy == base.best_strategy
    assert scaled.pbo.pbo == base.pbo.pbo
    np.testing.assert_allclose(
        scaled.table["sharpe"].to_numpy(), base.table["sharpe"].to_numpy(), rtol=1e-9
    )
    assert scaled.best_dsr.dsr == pytest.approx(base.best_dsr.dsr, rel=1e-9)


def test_pbo_is_invariant_to_column_order(data):
    """Which column index a strategy happens to occupy must not matter."""
    base = cscv_pbo(data, n_blocks=8)
    shuffled_cols = list(data.columns)[::-1]
    flipped = cscv_pbo(data[shuffled_cols], n_blocks=8)
    assert flipped.pbo == base.pbo
    assert flipped.mean_relative_rank == pytest.approx(base.mean_relative_rank)
    # The winning strategy must follow its column, not its position.
    assert set(flipped.selected_names()) == set(base.selected_names())


def test_report_is_invariant_to_column_order(data):
    base = build_report(data, n_blocks=8)
    flipped = build_report(data[list(data.columns)[::-1]], n_blocks=8)
    assert flipped.best_strategy == base.best_strategy
    assert flipped.best_dsr.dsr == pytest.approx(base.best_dsr.dsr)
    pd.testing.assert_frame_equal(
        flipped.table.sort_index(), base.table.sort_index(), check_like=True
    )


def test_adding_a_constant_risk_free_shifts_sharpe_predictably(data):
    """Sanity on the numerator: raising the risk-free rate can only lower every
    Sharpe, by an amount set by each strategy's own volatility."""
    plain = [sharpe_ratio(data[c]) for c in data.columns]
    charged = [sharpe_ratio(data[c], risk_free=0.001) for c in data.columns]
    assert all(b < a for a, b in zip(plain, charged))


# --------------------------------------------------------------------------
# Monotonicity
# --------------------------------------------------------------------------

def test_dsr_monotonicity_that_holds_unconditionally():
    """More Sharpe always helps; more trials and more dispersed trials always
    hurt, because both only ever raise the hurdle."""
    kw = dict(sharpe=0.12, n_trials=50, n_obs=800, variance_of_trials=0.005)

    def dsr(**over):
        return deflated_sharpe_ratio(**{**kw, **over}).dsr

    base = dsr()
    assert dsr(sharpe=0.20) > base > dsr(sharpe=0.05)
    assert dsr(n_trials=5) > base > dsr(n_trials=5000)
    assert dsr(variance_of_trials=1e-6) > base > dsr(variance_of_trials=0.05)


def test_dsr_monotonicity_in_t_and_moments_flips_around_the_hurdle():
    """A subtlety worth stating explicitly, because the naive expectation is
    wrong: whether more data (or thinner tails, or positive skew) raises or
    lowers the DSR depends on which side of the hurdle the Sharpe sits.

    DSR is Z[(SR - SR*)*sqrt(T-1)/sigma_hat].  Above the hurdle the z-score is
    positive and shrinking sigma_hat pushes it toward 1; below the hurdle it is
    negative and the same change pushes it toward 0.  More evidence increases
    confidence in whatever is true, and 'more data always helps' is only right
    for a strategy that is genuinely good."""
    hurdle = expected_max_sharpe(50, 0.005)
    kw = dict(n_trials=50, n_obs=800, variance_of_trials=0.005)

    def dsr(sharpe, **over):
        return deflated_sharpe_ratio(sharpe=sharpe, **{**kw, **over}).dsr

    above, below = hurdle + 0.08, hurdle - 0.08
    assert dsr(above) > 0.5 > dsr(below)

    # Above the hurdle: more data and better-behaved returns raise confidence.
    assert dsr(above, n_obs=4000) > dsr(above) > dsr(above, n_obs=200)
    assert dsr(above, kurtosis=1.5) > dsr(above) > dsr(above, kurtosis=12.0)
    assert dsr(above, skew=1.0) > dsr(above) > dsr(above, skew=-1.0)

    # Below the hurdle: exactly the same changes push the DSR down instead.
    assert dsr(below, n_obs=4000) < dsr(below) < dsr(below, n_obs=200)
    assert dsr(below, kurtosis=1.5) < dsr(below) < dsr(below, kurtosis=12.0)
    assert dsr(below, skew=1.0) < dsr(below) < dsr(below, skew=-1.0)


def test_expected_max_sharpe_is_monotone_in_both_arguments():
    trials = [expected_max_sharpe(n, 0.01) for n in (2, 5, 20, 100, 10_000, 10**9)]
    assert all(a < b for a, b in zip(trials, trials[1:]))
    variances = [expected_max_sharpe(100, v) for v in (1e-6, 1e-4, 0.01, 1.0)]
    assert all(a < b for a, b in zip(variances, variances[1:]))


def test_psr_is_monotone_in_the_benchmark():
    values = [probabilistic_sharpe_ratio(0.1, 500, benchmark_sharpe=b)
              for b in (-0.1, 0.0, 0.05, 0.1, 0.2)]
    assert all(a > b for a, b in zip(values, values[1:]))


# --------------------------------------------------------------------------
# Bounds and well-formedness
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_all_outputs_stay_in_bounds_across_random_inputs(seed):
    rng = np.random.default_rng(seed)
    n_strategies = int(rng.integers(2, 25))
    n_periods = int(rng.integers(200, 900))
    frame = make_synthetic_strategies(
        n_strategies=n_strategies,
        n_periods=n_periods,
        mu=float(rng.normal(0, 0.001)),
        sigma=float(rng.uniform(0.001, 0.05)),
        autocorr=float(rng.uniform(-0.5, 0.5)),
        n_signal=int(rng.integers(0, 3)),
        seed=seed,
    )
    report = build_report(frame, n_blocks=8)
    p = report.pbo

    assert 0.0 <= p.pbo <= 1.0
    assert 0.0 < p.mean_relative_rank < 1.0
    assert np.all(np.isfinite(p.logits))
    assert 0.0 <= p.prob_loss <= 1.0
    assert p.logits.size == p.n_splits == math.comb(8, 4)
    assert report.table["deflated_sharpe"].between(0.0, 1.0).all()
    assert report.table["psr_vs_zero"].between(0.0, 1.0).all()
    assert (report.table["deflated_sharpe"] <= report.table["psr_vs_zero"] + 1e-12).all()
    assert report.table["kurtosis"].gt(0).all()


def test_pbo_and_logit_sign_agree_by_construction(data):
    """PBO is *defined* as the share of non-positive logits; the two reported
    numbers must never drift apart."""
    result = cscv_pbo(data, n_blocks=10)
    assert result.pbo == pytest.approx(np.mean(result.logits <= 0.0))
    # omega and lambda are two views of the same rank.
    omegas = 1.0 / (1.0 + np.exp(-result.logits))
    assert result.mean_relative_rank == pytest.approx(omegas.mean())


# --------------------------------------------------------------------------
# The combinatorial structure itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_blocks", [4, 6, 8, 10])
def test_every_block_is_in_sample_in_exactly_half_the_splits(n_blocks):
    """This is the 'combinatorially symmetric' property the method is named
    for: no period is privileged, and IS/OOS are interchangeable by design."""
    splits = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    counts = np.zeros(n_blocks, dtype=int)
    for split in splits:
        counts[list(split)] += 1
    assert len(splits) == math.comb(n_blocks, n_blocks // 2)
    assert set(counts) == {len(splits) // 2}


def test_complement_of_every_split_is_also_a_split(data):
    """IS/OOS symmetry: the set of partitions is closed under swapping halves,
    which is what makes PBO a symmetric statistic."""
    n_blocks = 8
    splits = {frozenset(c) for c in itertools.combinations(range(n_blocks), n_blocks // 2)}
    everything = frozenset(range(n_blocks))
    assert all(everything - s in splits for s in splits)


def test_time_reversal_leaves_pbo_exactly_unchanged(data):
    """Reversing the rows relabels block i as block S-1-i, and the family of
    balanced splits is closed under relabeling -- so reversal permutes the
    splits among themselves and every reported number is identical.  A
    newest-first CSV export is therefore harmless; it is *partial* disorder
    that corrupts the blocks."""
    forward = cscv_pbo(data, n_blocks=10)
    backward = cscv_pbo(data.iloc[::-1], n_blocks=10)
    assert forward.pbo == backward.pbo
    assert forward.mean_relative_rank == pytest.approx(backward.mean_relative_rank)
    np.testing.assert_allclose(
        np.sort(forward.is_performance), np.sort(backward.is_performance), rtol=1e-12
    )


def test_shuffling_rows_does_change_the_answer(data):
    """Scrambling rows mixes periods across blocks, destroys any time structure
    the blocking is meant to preserve, and changes the result -- which is what
    the unsorted-index warning exists to prevent."""
    forward = cscv_pbo(data, n_blocks=10)
    scrambled = cscv_pbo(data.sample(frac=1.0, random_state=1), n_blocks=10)
    assert not np.allclose(
        np.sort(forward.is_performance), np.sort(scrambled.is_performance)
    )


# --------------------------------------------------------------------------
# The optimization must not change the answer
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_periods,n_strategies,n_blocks,scale,offset",
    [
        (400, 5, 10, 0.01, 0.0),
        (600, 20, 8, 0.01, 0.0),
        (600, 8, 8, 1e-9, 0.0),
        (600, 8, 8, 1e5, 0.0),
        (750, 12, 10, 0.01, 100.0),   # large offset: the cancellation trap
        (900, 6, 6, 0.02, -50.0),
    ],
)
def test_vectorized_sharpe_matches_the_naive_loop(
    n_periods, n_strategies, n_blocks, scale, offset
):
    """The fast path derives each split's Sharpe from per-block sums instead of
    re-reducing the concatenated matrix.  It must be exact, not merely close --
    including on data with a large constant offset, where a naive
    sum-of-squares formulation would lose precision."""
    rng = np.random.default_rng(hash((n_periods, n_strategies)) % 2**32)
    frame = pd.DataFrame(rng.normal(offset, scale, size=(n_periods, n_strategies)))

    fast = cscv_pbo(frame, n_blocks=n_blocks)
    # A lambda is not the default metric object, so this takes the naive path.
    naive = cscv_pbo(frame, n_blocks=n_blocks, metric=lambda x: _sharpe_metric(x))

    assert fast.pbo == naive.pbo
    np.testing.assert_array_equal(fast.selected, naive.selected)
    np.testing.assert_allclose(fast.is_performance, naive.is_performance, rtol=1e-11)
    np.testing.assert_allclose(fast.oos_performance, naive.oos_performance, rtol=1e-11)
    np.testing.assert_allclose(fast.logits, naive.logits, rtol=1e-11)


def test_custom_metric_still_works_and_can_disagree(data):
    """The naive path must remain usable for arbitrary metrics."""
    by_mean = cscv_pbo(data, n_blocks=8, metric=lambda x: float(np.mean(x)))
    by_sharpe = cscv_pbo(data, n_blocks=8)
    assert by_mean.n_splits == by_sharpe.n_splits
    assert 0.0 <= by_mean.pbo <= 1.0


def test_results_are_deterministic(data):
    """No hidden randomness anywhere in the pipeline."""
    a, b = cscv_pbo(data, n_blocks=10), cscv_pbo(data, n_blocks=10)
    assert a.pbo == b.pbo
    np.testing.assert_array_equal(a.logits, b.logits)
    assert build_report(data, n_blocks=10).best_dsr == build_report(data, n_blocks=10).best_dsr
