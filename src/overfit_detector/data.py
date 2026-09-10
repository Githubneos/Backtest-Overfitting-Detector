"""Sources of strategy return streams to analyse.

Two entry points:

* :func:`make_synthetic_strategies` -- controlled data with known ground truth,
  used by the test suite and by anyone who wants to see what PBO looks like
  when the answer is known in advance.
* :func:`load_yfinance_strategies` -- a grid of moving-average crossover
  variants on a real ticker, so the tool can be pointed at something real.
  Requires the optional ``[data]`` extra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["make_synthetic_strategies", "load_yfinance_strategies"]


def make_synthetic_strategies(
    n_strategies: int = 50,
    n_periods: int = 1000,
    mu: float = 0.0,
    sigma: float = 0.01,
    autocorr: float = 0.0,
    n_signal: int = 0,
    signal_mu: float = 0.002,
    seed: int | None = None,
) -> pd.DataFrame:
    r"""Generate strategy-like return streams with configurable properties.

    Each column is an AR(1) process

    .. math:: r_t = \mu + \phi (r_{t-1} - \mu) + \varepsilon_t

    with :math:`\varepsilon_t \sim N(0, \sigma^2 (1 - \phi^2))` so that the
    unconditional volatility is :math:`\sigma` regardless of :math:`\phi`.

    Parameters
    ----------
    n_strategies : int
        Number of columns (strategy variants).
    n_periods : int
        Number of rows (time steps).
    mu, sigma : float
        Per-period mean and unconditional volatility of the *noise* strategies.
        ``mu=0`` gives strategies with genuinely no edge.
    autocorr : float
        AR(1) coefficient :math:`\phi`, in ``(-1, 1)``.
    n_signal : int
        Number of leading columns given a real edge: their per-period mean is
        ``signal_mu`` instead of ``mu``.  These are named ``signal_0``, ... ;
        the rest are ``noise_0``, ... .
    signal_mu : float
        Per-period mean for the signal columns.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    DataFrame
        Shape ``(n_periods, n_strategies)``.
    """
    if n_strategies < 1 or n_periods < 2:
        raise ValueError("need n_strategies >= 1 and n_periods >= 2")
    if not -1.0 < autocorr < 1.0:
        raise ValueError("autocorr must be strictly between -1 and 1")
    if not 0 <= n_signal <= n_strategies:
        raise ValueError("n_signal must be between 0 and n_strategies")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    rng = np.random.default_rng(seed)
    means = np.full(n_strategies, float(mu))
    means[:n_signal] = float(signal_mu)

    shock_sd = sigma * np.sqrt(1.0 - autocorr**2)
    shocks = rng.normal(0.0, shock_sd, size=(n_periods, n_strategies))

    out = np.empty((n_periods, n_strategies))
    # Start each series from its stationary distribution so there is no burn-in.
    out[0] = rng.normal(0.0, sigma, size=n_strategies)
    for t in range(1, n_periods):
        out[t] = autocorr * out[t - 1] + shocks[t]
    out += means

    names = [f"signal_{i}" for i in range(n_signal)] + [
        f"noise_{i}" for i in range(n_strategies - n_signal)
    ]
    return pd.DataFrame(out, columns=names)


def load_yfinance_strategies(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: str | None = None,
    fast_windows: tuple[int, ...] = (5, 10, 15, 20, 25, 30),
    slow_windows: tuple[int, ...] = (50, 100, 150, 200),
    interval: str = "1d",
) -> pd.DataFrame:
    """Build a grid of moving-average crossover variants on a real ticker.

    Every ``(fast, slow)`` pair with ``fast < slow`` becomes one column of
    strategy returns: long when the fast MA is above the slow MA, flat
    otherwise, with the signal lagged one period to avoid look-ahead.

    This is deliberately a *naive* parameter sweep -- that is exactly the
    situation DSR and PBO are designed to diagnose, and a grid like this over a
    single index usually scores badly, which is the point.

    Requires the optional dependency: ``pip install 'overfit-detector[data]'``.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "load_yfinance_strategies needs yfinance; install the optional "
            "extra with: pip install 'overfit-detector[data]'"
        ) from exc

    raw = yf.download(
        ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False
    )
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker!r}")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance returns a MultiIndex for some calls
        close = close.iloc[:, 0]
    close = close.astype(float).dropna()

    asset_returns = close.pct_change()
    columns: dict[str, pd.Series] = {}
    for fast in fast_windows:
        for slow in slow_windows:
            if fast >= slow:
                continue
            position = (
                close.rolling(fast).mean() > close.rolling(slow).mean()
            ).astype(float).shift(1)
            columns[f"ma_{fast}_{slow}"] = asset_returns * position

    if not columns:
        raise ValueError("no valid (fast, slow) window pairs were produced")
    return pd.DataFrame(columns).dropna()
