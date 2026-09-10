"""Basic performance statistics.

Everything in this module works in **per-period** units unless a function name
says otherwise.  This matters: the Deflated Sharpe Ratio machinery in
:mod:`overfit_detector.deflated_sharpe` is derived for the per-period Sharpe
estimator, and feeding it an annualized Sharpe silently corrupts the result.
Annualization is a presentation-layer concern only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "sharpe_ratio",
    "is_degenerate",
    "annualize_sharpe",
    "annualized_sharpe_ratio",
    "returns_moments",
]


#: Sharpe is undefined below this volatility, measured **relative to the scale
#: of the data**.  A constant series does not always give exactly 0.0 standard
#: deviation -- catastrophic cancellation leaves ~1e-18 -- so the comparison
#: cannot be against zero.  It must also be purely relative: returns quoted in
#: fractions, percent or basis points describe the same strategy, and a fixed
#: absolute floor would declare the small-unit version flat.
_VOL_TOLERANCE = 1e-12


def _as_1d_array(returns) -> np.ndarray:
    """Coerce a Series/array/list of returns to a finite 1-D float array.

    A 2-D input is rejected rather than flattened: silently ravelling a
    DataFrame of several strategies into one series produces a number that
    looks like a Sharpe ratio and means nothing.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            raise ValueError(
                f"expected a single return series, got a DataFrame with "
                f"{returns.shape[1]} columns; pass one column at a time"
            )
        returns = returns.iloc[:, 0]

    try:
        arr = np.asarray(
            returns.to_numpy() if isinstance(returns, pd.Series) else returns,
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"returns are not numeric: {exc}") from exc

    if arr.ndim > 1:
        if sum(d > 1 for d in arr.shape) > 1:
            raise ValueError(
                f"expected a single return series, got an array of shape {arr.shape}"
            )
        arr = arr.ravel()

    if arr.size < 2:
        raise ValueError("need at least 2 return observations")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns contain NaN or infinite values")
    return arr


def is_degenerate(returns) -> bool:
    """True when a return series is flat enough that its Sharpe is undefined.

    Catches an all-zero "hold cash" variant, a strategy that was never in the
    market, and any other constant series.  Both :func:`sharpe_ratio` and the
    CSCV routine use this so they agree on what counts as unusable.
    """
    try:
        arr = _as_1d_array(returns)
    except ValueError:
        # Non-finite or non-numeric data is a different problem with a
        # different error message; let the caller's own validation report it.
        return False
    return _is_flat(arr)


def _is_flat(arr: np.ndarray) -> bool:
    """Scale-relative flatness test shared by every code path."""
    sigma = float(np.std(arr, ddof=1))
    scale = float(np.max(np.abs(arr)))
    return sigma <= _VOL_TOLERANCE * scale


def sharpe_ratio(returns, risk_free: float = 0.0) -> float:
    r"""Per-period Sharpe ratio.

    .. math::

        SR = \frac{\operatorname{mean}(r) - r_f}{\operatorname{std}(r)}

    The standard deviation uses ``ddof=1`` (sample estimator), which is the
    convention assumed by the Probabilistic/Deflated Sharpe derivations.

    Parameters
    ----------
    returns : array-like
        Per-period returns.
    risk_free : float
        Risk-free rate expressed in the **same period units** as ``returns``.

    Returns
    -------
    float
        The per-period Sharpe ratio.  Not annualized -- see
        :func:`annualize_sharpe`.
    """
    arr = _as_1d_array(returns)
    if _is_flat(arr):
        raise ValueError("returns have zero volatility; Sharpe ratio is undefined")
    sigma = float(np.std(arr, ddof=1))
    return float((np.mean(arr) - risk_free) / sigma)


def annualize_sharpe(sharpe: float, periods_per_year: float) -> float:
    r"""Scale a per-period Sharpe to annual units: :math:`SR \sqrt{p}`.

    Valid under i.i.d. returns; with autocorrelation the square-root rule
    over- or under-states the annual figure, so treat it as a display
    convention rather than an estimate.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    return float(sharpe * np.sqrt(periods_per_year))


def annualized_sharpe_ratio(returns, periods_per_year: float, risk_free: float = 0.0) -> float:
    """Convenience wrapper: :func:`sharpe_ratio` then :func:`annualize_sharpe`."""
    return annualize_sharpe(sharpe_ratio(returns, risk_free=risk_free), periods_per_year)


def returns_moments(returns) -> tuple[int, float, float]:
    r"""Return ``(n_obs, skewness, kurtosis)`` for a return series.

    Kurtosis is **non-excess** (Pearson): a normal distribution gives 3.0.
    The Probabilistic Sharpe Ratio formula is written in terms of
    :math:`\gamma_4` in that convention, so ``fisher=False`` is required here.
    """
    arr = _as_1d_array(returns)
    return (
        int(arr.size),
        float(stats.skew(arr, bias=False)),
        float(stats.kurtosis(arr, fisher=False, bias=False)),
    )
