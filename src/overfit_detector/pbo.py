r"""Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
Cross-Validation (CSCV).

Reference
---------
Bailey, D. H., Borwein, J. M., Lopez de Prado, M. and Zhu, Q. J. (2017).
*The Probability of Backtest Overfitting.* Journal of Computational Finance,
20(4), 39-69.

The idea: if the variant that looked best in-sample is genuinely good, it
should keep ranking near the top out-of-sample.  CSCV chops the backtest into
:math:`S` contiguous blocks, forms **every** balanced split of those blocks into
an in-sample half and an out-of-sample half, and records where the IS winner
lands in the OOS ranking.  PBO is the fraction of splits where the winner
lands at or below the OOS median -- i.e. the frequency with which "best" was
an artefact of the sample you happened to fit on.

Why *combinatorially symmetric*: every block appears in-sample in exactly half
the splits and out-of-sample in the other half, so no period is privileged and
the procedure is symmetric under swapping IS and OOS.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .metrics import _VOL_TOLERANCE, is_degenerate, sharpe_ratio

__all__ = ["cscv_pbo", "PBOResult", "DEFAULT_BLOCKS"]

#: Default number of blocks S.  C(10, 5) = 252 splits: enough resolution for a
#: stable estimate, cheap enough to run interactively.
DEFAULT_BLOCKS = 10

#: Emit a warning above this many splits, and refuse above the hard cap.
_WARN_SPLITS = 10_000
_MAX_SPLITS = 200_000


def _sharpe_metric(block: np.ndarray) -> float:
    """Per-period Sharpe, returning 0.0 for a degenerate (zero-vol) column."""
    try:
        return sharpe_ratio(block)
    except ValueError:
        return 0.0


def _performance(matrix: np.ndarray, metric) -> np.ndarray:
    """Apply ``metric`` column-wise to a (time x strategy) matrix."""
    return np.array([metric(matrix[:, j]) for j in range(matrix.shape[1])], dtype=float)


def _block_aggregates(trimmed: np.ndarray, n_blocks: int):
    """Per-block sufficient statistics for the vectorized Sharpe fast path.

    Returns ``(shift, sums, sumsq, maxabs)`` where ``sums`` and ``sumsq`` are
    computed on **column-centred** data.  Variance is shift-invariant, so
    centring changes nothing algebraically while keeping
    ``sumsq - n*mean^2`` far away from catastrophic cancellation -- which is
    what would otherwise make this fast path disagree with the naive one on
    returns carrying a large constant offset.
    """
    n_used, n_strategies = trimmed.shape
    block_len = n_used // n_blocks
    shift = trimmed.mean(axis=0)
    centred = (trimmed - shift).reshape(n_blocks, block_len, n_strategies)
    return (
        shift,
        centred.sum(axis=1),
        np.einsum("sij,sij->sj", centred, centred),
        np.abs(trimmed).reshape(n_blocks, block_len, n_strategies).max(axis=1),
    )


def _fast_sharpe(idx, shift, sums, sumsq, maxabs, block_len) -> np.ndarray:
    """Per-period Sharpe for every column over the blocks in ``idx``.

    Algebraically identical to calling :func:`sharpe_ratio` on the concatenated
    blocks, including the ddof=1 denominator and the zero-volatility rule (a
    degenerate column scores 0.0, matching ``_sharpe_metric``).
    """
    n = block_len * len(idx)
    csum = sums[idx].sum(axis=0)
    csq = sumsq[idx].sum(axis=0)
    cmean = csum / n
    var = (csq - n * cmean**2) / (n - 1)
    sigma = np.sqrt(np.maximum(var, 0.0))
    scale = maxabs[idx].max(axis=0)
    usable = sigma > _VOL_TOLERANCE * scale
    out = np.zeros_like(sigma)
    np.divide(shift + cmean, sigma, out=out, where=usable)
    return out


@dataclass(frozen=True)
class PBOResult:
    """Result of a CSCV run, with the diagnostics the paper reports alongside PBO."""

    pbo: float
    """Fraction of splits where the IS-best strategy ranked at or below the OOS median."""
    logits: np.ndarray = field(repr=False)
    """Per-split logit :math:`\\lambda_c` of the winner's relative OOS rank."""
    is_performance: np.ndarray = field(repr=False)
    """Per-split IS performance of the selected strategy."""
    oos_performance: np.ndarray = field(repr=False)
    """Per-split OOS performance of that same strategy.

    Plotted against :attr:`is_performance` this gives the paper's performance
    degradation scatter.  Read it qualitatively, not as a regression slope: IS
    and OOS are complementary halves of one fixed sample, so their sum is
    pinned and the fitted slope is dragged toward -1 whenever the same strategy
    is selected across splits -- which is exactly what a *good* strategy does.
    """
    selected: np.ndarray = field(repr=False)
    """Per-split index of the IS-best strategy."""
    n_splits: int
    n_blocks: int
    n_strategies: int
    n_obs_used: int
    """Rows actually used, after trimming to a multiple of ``n_blocks``."""
    n_obs_dropped: int
    """Rows dropped from the *start* of the sample by that trim."""
    mean_relative_rank: float
    r"""Mean OOS relative rank :math:`\omega_c` of the IS winner, in ``(0, 1)``.

    0.5 is what pure chance delivers; values well above 0.5 mean the winner
    keeps winning out-of-sample.  This is the same quantity PBO thresholds at
    the median, but without discarding the magnitude.
    """
    prob_loss: float
    """Fraction of splits where the selected strategy's OOS performance was negative.

    For skill-free variants this sits near 0.5: having been best in-sample tells
    you nothing about whether the strategy even makes money out-of-sample.
    """
    strategy_names: list[str] | None = None

    def selected_names(self) -> list[str] | None:
        """The IS-winning strategy per split, by name (``None`` if unnamed)."""
        if self.strategy_names is None:
            return None
        return [self.strategy_names[i] for i in self.selected]

    def summary(self) -> dict:
        """Scalar diagnostics as a plain dict (handy for ``--json`` output)."""
        return {
            "pbo": self.pbo,
            "n_splits": self.n_splits,
            "n_blocks": self.n_blocks,
            "n_strategies": self.n_strategies,
            "n_obs_used": self.n_obs_used,
            "n_obs_dropped": self.n_obs_dropped,
            "mean_relative_rank": self.mean_relative_rank,
            "prob_loss": self.prob_loss,
            "median_logit": float(np.median(self.logits)),
        }


def cscv_pbo(
    returns,
    n_blocks: int = DEFAULT_BLOCKS,
    metric=_sharpe_metric,
    force: bool = False,
) -> PBOResult:
    r"""Estimate the Probability of Backtest Overfitting by CSCV.

    Algorithm (following the paper):

    1. Trim the sample to a multiple of :math:`S` and split it into :math:`S`
       contiguous, equal-length blocks.  Blocks are contiguous so that any
       autocorrelation structure survives the partitioning.
    2. For each of the :math:`\binom{S}{S/2}` ways of choosing :math:`S/2`
       blocks as in-sample, recombine the chosen blocks **in original time
       order** to form :math:`J` (IS) and the complement to form
       :math:`\bar{J}` (OOS).
    3. Compute the performance metric for every strategy on :math:`J`; let
       :math:`n^{*}` be the argmax.
    4. Rank :math:`n^{*}` among all strategies on :math:`\bar{J}`
       (1 = worst, :math:`N` = best), form the relative rank
       :math:`\omega_c = \frac{\text{rank}}{N+1}` and the logit
       :math:`\lambda_c = \ln\frac{\omega_c}{1-\omega_c}`.
       The :math:`N+1` denominator keeps :math:`\omega_c` strictly inside
       :math:`(0,1)` so the logit is always finite.
    5. :math:`PBO = \Pr[\lambda_c \le 0]`, estimated as the fraction of splits
       with a non-positive logit.

    Parameters
    ----------
    returns : DataFrame or 2-D array
        Rows = time, columns = strategy variants.  At least 2 columns.
    n_blocks : int
        :math:`S`, must be even and >= 4.
    metric : callable
        ``f(1-D array) -> float``, higher is better.  Defaults to the
        per-period Sharpe ratio.
    force : bool
        Allow split counts above the internal hard cap.

    Returns
    -------
    PBOResult
    """
    names = None
    if isinstance(returns, pd.DataFrame):
        names = [str(c) for c in returns.columns]
        # CSCV blocks are contiguous slices of the row order, so rows that are
        # not in time order silently produce different blocks and a different
        # answer.  A newest-first export is the common way to hit this.
        if isinstance(returns.index, pd.DatetimeIndex) and not (
            returns.index.is_monotonic_increasing
        ):
            warnings.warn(
                "the DatetimeIndex is not in increasing order; CSCV blocks are "
                "contiguous in row order, so sort the rows by date first. "
                "(An exact reversal happens to be harmless -- it just relabels "
                "the blocks -- but any partial disorder changes the result.)",
                stacklevel=2,
            )
        try:
            matrix = returns.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"returns are not numeric: {exc}") from exc
    else:
        try:
            matrix = np.asarray(returns, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"returns are not numeric: {exc}") from exc

    if matrix.ndim != 2:
        raise ValueError("returns must be 2-D (rows = time, columns = strategies)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("returns contain NaN or infinite values")

    # Flat columns (an all-zero "hold cash" variant, a strategy never in the
    # market) have no Sharpe ratio and cannot be ranked.  Drop them loudly
    # rather than scoring them 0.0 alongside real strategies.
    keep = [j for j in range(matrix.shape[1]) if not is_degenerate(matrix[:, j])]
    if len(keep) < matrix.shape[1]:
        dropped = [
            names[j] if names else str(j)
            for j in range(matrix.shape[1])
            if j not in set(keep)
        ]
        warnings.warn(
            f"dropped {len(dropped)} zero-volatility column(s) from the CSCV "
            f"ranking: {', '.join(dropped[:5])}"
            + (" ..." if len(dropped) > 5 else ""),
            stacklevel=2,
        )
        matrix = matrix[:, keep]
        if names is not None:
            names = [names[j] for j in keep]

    n_obs, n_strategies = matrix.shape
    if n_strategies < 2:
        raise ValueError(
            "CSCV needs at least 2 non-degenerate strategy variants to rank"
        )
    if n_blocks < 4 or n_blocks % 2 != 0:
        raise ValueError("n_blocks must be an even integer >= 4")

    block_len = n_obs // n_blocks
    if block_len < 2:
        raise ValueError(
            f"{n_obs} observations split into {n_blocks} blocks leaves "
            f"{block_len} rows per block; need at least 2"
        )

    n_splits = math.comb(n_blocks, n_blocks // 2)
    if n_splits > _MAX_SPLITS and not force:
        raise ValueError(
            f"n_blocks={n_blocks} implies {n_splits:,} splits, above the "
            f"{_MAX_SPLITS:,} cap; lower n_blocks or pass force=True"
        )
    if n_splits > _WARN_SPLITS:
        warnings.warn(
            f"n_blocks={n_blocks} implies {n_splits:,} combinatorial splits "
            f"x {n_strategies} strategies; this may take a while",
            stacklevel=2,
        )

    # Trim from the START so the most recent data is always retained.
    n_used = block_len * n_blocks
    dropped = n_obs - n_used
    trimmed = matrix[dropped:]
    blocks = [trimmed[i * block_len : (i + 1) * block_len] for i in range(n_blocks)]
    all_blocks = set(range(n_blocks))

    logits = np.empty(n_splits)
    omegas = np.empty(n_splits)
    is_perf = np.empty(n_splits)
    oos_perf = np.empty(n_splits)
    selected = np.empty(n_splits, dtype=int)

    # The default Sharpe metric is computed from per-block sufficient statistics
    # instead of re-reducing the concatenated matrix on every split: same
    # numbers, O(S*N) per split instead of O(T*N).  A custom metric is an
    # arbitrary callable, so it keeps the straightforward path.
    fast = metric is _sharpe_metric
    if fast:
        shift, sums, sumsq, maxabs = _block_aggregates(trimmed, n_blocks)

    for c, is_idx in enumerate(combinations(range(n_blocks), n_blocks // 2)):
        oos_idx = sorted(all_blocks - set(is_idx))
        is_idx = sorted(is_idx)
        if fast:
            r_is = _fast_sharpe(is_idx, shift, sums, sumsq, maxabs, block_len)
            r_oos = _fast_sharpe(oos_idx, shift, sums, sumsq, maxabs, block_len)
        else:
            # sorted() on both sides keeps the blocks in original time order.
            r_is = _performance(np.concatenate([blocks[i] for i in is_idx]), metric)
            r_oos = _performance(np.concatenate([blocks[i] for i in oos_idx]), metric)

        best = int(np.argmax(r_is))
        # Midrank: 1 = worst, N = best, ties share the average of the ranks they
        # span.  Exact ties therefore put omega at 0.5 and the logit at 0, which
        # counts toward PBO under the paper's Pr[lambda <= 0] convention.
        below = int((r_oos < r_oos[best]).sum())
        equal = int((r_oos == r_oos[best]).sum())
        rank = below + (equal + 1) / 2.0
        omega = rank / (n_strategies + 1.0)

        omegas[c] = omega
        logits[c] = math.log(omega / (1.0 - omega))
        selected[c] = best
        is_perf[c] = r_is[best]
        oos_perf[c] = r_oos[best]

    pbo = float(np.mean(logits <= 0.0))

    return PBOResult(
        pbo=pbo,
        logits=logits,
        is_performance=is_perf,
        oos_performance=oos_perf,
        selected=selected,
        n_splits=n_splits,
        n_blocks=n_blocks,
        n_strategies=n_strategies,
        n_obs_used=n_used,
        n_obs_dropped=dropped,
        mean_relative_rank=float(np.mean(omegas)),
        prob_loss=float(np.mean(oos_perf < 0.0)),
        strategy_names=names,
    )
