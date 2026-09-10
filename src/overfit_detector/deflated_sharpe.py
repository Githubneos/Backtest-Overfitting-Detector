r"""Deflated Sharpe Ratio (DSR) and its building blocks.

Reference
---------
Bailey, D. H. and Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality.*
Journal of Portfolio Management, 40(5), 94-107.

The idea in one paragraph: an observed Sharpe ratio is an *estimate*, and when
it is the maximum over :math:`N` tried variants it is the maximum of :math:`N`
noisy estimates -- which is biased upward even when every variant is worthless.
The DSR asks a sharper question than "is the Sharpe positive?": it asks for the
probability that the true Sharpe exceeds the Sharpe you would *expect* to see
from the luckiest of :math:`N` skill-free trials.

All Sharpe ratios in this module are **per-period** (not annualized).  See
:mod:`overfit_detector.metrics`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .metrics import returns_moments, sharpe_ratio

__all__ = [
    "EULER_MASCHERONI",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_returns",
    "DeflatedSharpeResult",
]

#: Euler-Mascheroni constant, used in the expected-maximum order statistic.
EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials: int, variance_of_trials: float) -> float:
    r"""Expected maximum Sharpe ratio under the null of **no skill**.

    .. math::

        SR^{*} = \sqrt{V[\widehat{SR}]}\,
                 \left[(1-\gamma)\,Z^{-1}\!\left(1-\tfrac{1}{N}\right)
                       + \gamma\, Z^{-1}\!\left(1-\tfrac{1}{N e}\right)\right]

    where :math:`\gamma` is the Euler-Mascheroni constant, :math:`Z^{-1}` the
    standard-normal quantile function, :math:`N` the number of independent
    trials and :math:`V[\widehat{SR}]` the variance of the Sharpe estimates
    *across those trials*.

    This is the expected value of the maximum of :math:`N` i.i.d. Gaussian
    draws (a Gumbel-limit approximation), scaled to the dispersion of the
    trial Sharpes.  It is the threshold the observed Sharpe must clear before
    it counts as evidence of anything.

    Parameters
    ----------
    n_trials : int
        Number of strategy variants tried, :math:`N \ge 1`.
    variance_of_trials : float
        :math:`V[\widehat{SR}]`, the cross-sectional variance of the trial
        Sharpe ratios (per-period units, squared).

    Returns
    -------
    float
        :math:`SR^{*}`.  Exactly ``0.0`` when ``n_trials == 1`` (with a single
        trial there is no selection to correct for) or when the trial variance
        is zero.
    """
    if int(n_trials) != n_trials:
        raise ValueError(f"n_trials must be a whole number, got {n_trials!r}")
    n_trials = int(n_trials)
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if not math.isfinite(variance_of_trials):
        raise ValueError("variance_of_trials must be finite")
    if variance_of_trials < 0:
        raise ValueError("variance_of_trials must be non-negative")
    if n_trials == 1 or variance_of_trials == 0.0:
        return 0.0

    n = float(n_trials)
    # isf(1/n), not ppf(1 - 1/n): for large n the latter loses the whole tail to
    # rounding (1 - 1/n evaluates to exactly 1.0 by n ~ 1e17, giving inf), while
    # isf works directly with the small upper-tail probability.
    gumbel = (1.0 - EULER_MASCHERONI) * stats.norm.isf(1.0 / n) + (
        EULER_MASCHERONI * stats.norm.isf(1.0 / (n * math.e))
    )
    result = float(math.sqrt(variance_of_trials) * gumbel)
    if not math.isfinite(result):
        raise ValueError(
            f"expected maximum Sharpe is not finite for n_trials={n_trials} and "
            f"variance_of_trials={variance_of_trials}; the trial count is too "
            "large to represent"
        )
    return result


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sharpe: float = 0.0,
) -> float:
    r"""Probabilistic Sharpe Ratio: :math:`\Pr[SR_{\text{true}} > SR_0]`.

    .. math::

        \widehat{PSR}(SR_0) = Z\!\left[
            \frac{(\widehat{SR} - SR_0)\sqrt{T-1}}
                 {\sqrt{1 - \gamma_3 \widehat{SR}
                        + \frac{\gamma_4 - 1}{4}\widehat{SR}^2}}\right]

    The denominator is the (non-normality adjusted) standard error of the
    Sharpe estimator: negative skew and fat tails both *inflate* it, which
    correctly lowers your confidence in a Sharpe estimated from such returns.

    Parameters
    ----------
    sharpe : float
        Observed per-period Sharpe :math:`\widehat{SR}`.
    n_obs : int
        Number of return observations :math:`T` (must be >= 2).
    skew : float
        :math:`\gamma_3`, skewness of the returns.
    kurtosis : float
        :math:`\gamma_4`, **non-excess** kurtosis (3.0 for a normal).
    benchmark_sharpe : float
        :math:`SR_0`, the threshold to beat.  For the DSR this is
        :math:`SR^{*}` from :func:`expected_max_sharpe`.

    Returns
    -------
    float
        A probability in ``[0, 1]``.
    """
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    if kurtosis <= 0:
        raise ValueError("kurtosis must be positive (non-excess convention: normal = 3)")
    variance = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2
    if variance <= 0:
        raise ValueError(
            "non-normality adjusted variance of the Sharpe estimator is "
            f"non-positive ({variance:.6g}); check skew/kurtosis inputs"
        )
    z = (sharpe - benchmark_sharpe) * math.sqrt(n_obs - 1) / math.sqrt(variance)
    return float(stats.norm.cdf(z))


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Every input and intermediate of a DSR computation, kept inspectable."""

    sharpe: float
    """Observed per-period Sharpe ratio."""
    dsr: float
    """Deflated Sharpe Ratio: Pr[true Sharpe > expected max under the null]."""
    expected_max_sharpe: float
    """:math:`SR^{*}`, the per-period hurdle implied by the number of trials."""
    psr_zero: float
    """PSR against a zero benchmark, i.e. the *undeflated* confidence."""
    n_trials: int
    variance_of_trials: float
    n_obs: int
    skew: float
    kurtosis: float
    """Non-excess kurtosis (normal = 3.0)."""

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (
            f"DSR={self.dsr:.4f} (SR={self.sharpe:.4f}, SR*={self.expected_max_sharpe:.4f}, "
            f"N={self.n_trials}, T={self.n_obs})"
        )


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    variance_of_trials: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> DeflatedSharpeResult:
    r"""Deflate an observed Sharpe for selection bias and non-normality.

    :math:`DSR = \widehat{PSR}(SR_0 = SR^{*})`, i.e. the probability that the
    strategy's true Sharpe beats the best a skill-free search over
    ``n_trials`` variants would be expected to produce.

    Interpretation: values near 1 mean the result survives the multiple-testing
    correction; values near 0 mean the observed Sharpe is within the range luck
    alone would deliver given how many variants were tried.

    One subtlety worth knowing before you quote this number: the DSR is
    monotone in ``sharpe``, ``n_trials`` and ``variance_of_trials`` in the
    obvious directions, but its response to ``n_obs``, ``skew`` and
    ``kurtosis`` **flips sign at the hurdle**.  Those three enter only through
    the standard error, so they scale a z-score whose sign is set by
    ``sharpe - SR*``.  A longer backtest raises the DSR of a strategy above the
    hurdle and *lowers* the DSR of one below it: more evidence increases
    confidence in whatever happens to be true.

    Parameters
    ----------
    sharpe : float
        Observed **per-period** Sharpe ratio.
    n_trials : int
        Number of variants tried (this is the multiple-testing count -- be
        honest about it; it includes variants you discarded).
    n_obs : int
        Length of the backtest in periods.
    variance_of_trials : float
        Variance of the Sharpe ratios across the trials.
    skew, kurtosis : float
        Moments of the strategy's returns; kurtosis is non-excess.
    """
    sr_star = expected_max_sharpe(n_trials, variance_of_trials)
    return DeflatedSharpeResult(
        sharpe=float(sharpe),
        dsr=probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurtosis, benchmark_sharpe=sr_star),
        expected_max_sharpe=sr_star,
        psr_zero=probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurtosis, benchmark_sharpe=0.0),
        n_trials=int(n_trials),
        variance_of_trials=float(variance_of_trials),
        n_obs=int(n_obs),
        skew=float(skew),
        kurtosis=float(kurtosis),
    )


def deflated_sharpe_ratio_from_returns(
    returns,
    n_trials: int,
    variance_of_trials: float | None = None,
    all_trial_sharpes=None,
    risk_free: float = 0.0,
) -> DeflatedSharpeResult:
    """DSR straight from a return series, computing T, skew and kurtosis for you.

    ``variance_of_trials`` may be given explicitly; otherwise it is estimated
    as the cross-sectional sample variance of ``all_trial_sharpes`` (the
    practical estimator suggested in the paper).  One of the two must be
    supplied -- there is no safe default, because the deflation hurdle scales
    with the square root of this number.
    """
    if variance_of_trials is None:
        if all_trial_sharpes is None:
            raise ValueError(
                "supply either variance_of_trials or all_trial_sharpes "
                "(the variance of the trial Sharpes drives the deflation)"
            )
        arr = np.asarray(all_trial_sharpes, dtype=float).ravel()
        if arr.size < 2:
            raise ValueError("all_trial_sharpes needs at least 2 values to have a variance")
        variance_of_trials = float(np.var(arr, ddof=1))

    n_obs, skew, kurt = returns_moments(returns)
    return deflated_sharpe_ratio(
        sharpe=sharpe_ratio(returns, risk_free=risk_free),
        n_trials=n_trials,
        n_obs=n_obs,
        variance_of_trials=variance_of_trials,
        skew=skew,
        kurtosis=kurt,
    )
