"""Backtest Overfitting Detector.

Two complementary corrections for selection bias in backtests:

* :func:`~overfit_detector.deflated_sharpe.deflated_sharpe_ratio` -- how much
  of an observed Sharpe survives the fact that you tried N variants.
* :func:`~overfit_detector.pbo.cscv_pbo` -- how often the variant that looked
  best in-sample fails to beat the median out-of-sample.
"""

from .data import load_yfinance_strategies, make_synthetic_strategies
from .deflated_sharpe import (
    DeflatedSharpeResult,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from .metrics import annualize_sharpe, annualized_sharpe_ratio, returns_moments, sharpe_ratio
from .pbo import PBOResult, cscv_pbo
from .report import ReportResult, build_report, format_report, plot_report

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "sharpe_ratio",
    "annualize_sharpe",
    "annualized_sharpe_ratio",
    "returns_moments",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_returns",
    "DeflatedSharpeResult",
    "cscv_pbo",
    "PBOResult",
    "build_report",
    "format_report",
    "plot_report",
    "ReportResult",
    "make_synthetic_strategies",
    "load_yfinance_strategies",
]
