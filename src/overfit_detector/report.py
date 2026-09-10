"""Pull DSR and PBO together into a single human-readable verdict."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .deflated_sharpe import DeflatedSharpeResult, deflated_sharpe_ratio
from .metrics import annualize_sharpe, is_degenerate, returns_moments, sharpe_ratio
from .pbo import DEFAULT_BLOCKS, PBOResult, cscv_pbo

__all__ = ["ReportResult", "build_report", "format_report", "plot_report"]


@dataclass(frozen=True)
class ReportResult:
    """Everything the report shows, in structured form."""

    table: pd.DataFrame
    """One row per strategy: sharpe, annualized_sharpe, deflated_sharpe, ..."""
    pbo: PBOResult
    best_strategy: str
    best_dsr: DeflatedSharpeResult
    n_trials: int
    periods_per_year: float
    variance_of_trials: float

    def interpretation(self) -> str:
        """A plain-English verdict, safe to paste into a research note."""
        d = self.best_dsr
        raw = annualize_sharpe(d.sharpe, self.periods_per_year)
        hurdle = annualize_sharpe(d.expected_max_sharpe, self.periods_per_year)
        pbo_pct = 100 * self.pbo.pbo

        if self.pbo.pbo >= 0.5 or d.dsr < 0.5:
            verdict = "treat it with caution"
        elif self.pbo.pbo >= 0.25 or d.dsr < 0.9:
            verdict = "the evidence is suggestive but not conclusive"
        else:
            verdict = "this one survives both corrections"

        return (
            f"You tested {self.n_trials} variants over {d.n_obs} periods. "
            f"The top strategy ({self.best_strategy}) had a raw annualized Sharpe of "
            f"{raw:.2f}, but {self.n_trials} skill-free trials would be expected to "
            f"produce {hurdle:.2f} by luck alone, leaving a deflated Sharpe "
            f"(probability the true Sharpe clears that hurdle) of {d.dsr:.2f}. "
            f"CSCV puts the probability that this result is overfit at "
            f"{pbo_pct:.0f}% -- {verdict}."
        )

    def to_dict(self) -> dict:
        """JSON-serialisable summary (used by the CLI's ``--json``)."""
        return {
            "n_trials": self.n_trials,
            "periods_per_year": self.periods_per_year,
            "variance_of_trials": self.variance_of_trials,
            "best_strategy": self.best_strategy,
            "deflated_sharpe": {
                "sharpe": self.best_dsr.sharpe,
                "annualized_sharpe": annualize_sharpe(
                    self.best_dsr.sharpe, self.periods_per_year
                ),
                "expected_max_sharpe": self.best_dsr.expected_max_sharpe,
                "dsr": self.best_dsr.dsr,
                "psr_zero": self.best_dsr.psr_zero,
                "skew": self.best_dsr.skew,
                "kurtosis": self.best_dsr.kurtosis,
                "n_obs": self.best_dsr.n_obs,
            },
            "pbo": self.pbo.summary(),
            "interpretation": self.interpretation(),
            "strategies": self.table.reset_index().to_dict(orient="records"),
        }


def build_report(
    returns: pd.DataFrame,
    n_blocks: int = DEFAULT_BLOCKS,
    n_trials: int | None = None,
    periods_per_year: float = 252.0,
    risk_free: float = 0.0,
    force: bool = False,
) -> ReportResult:
    """Run both diagnostics over a set of strategy variants.

    Parameters
    ----------
    returns : DataFrame
        Rows = time, columns = strategy variants.
    n_blocks : int
        CSCV block count :math:`S` (even, >= 4).
    n_trials : int, optional
        Multiple-testing count for the deflation.  Defaults to the number of
        columns.  Override it upward if you discarded variants before saving
        this file -- **the deflation is only as honest as this number**.
    periods_per_year : float
        Used for display-level annualization only; the DSR math stays
        per-period throughout.
    risk_free : float
        Per-period risk-free rate.

    Returns
    -------
    ReportResult
    """
    if not isinstance(returns, pd.DataFrame):
        returns = pd.DataFrame(returns)

    duplicates = returns.columns[returns.columns.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(
            f"duplicate strategy column name(s): {duplicates}; rename them so "
            "each variant can be identified in the report"
        )

    n_supplied = returns.shape[1]
    # Count degenerate variants in n_trials before dropping them: they were
    # still trials, so they still contributed to the selection bias.
    n_trials = int(n_trials if n_trials is not None else n_supplied)
    if n_trials < n_supplied:
        raise ValueError(
            f"n_trials={n_trials} is below the {n_supplied} variants supplied; "
            "the trial count cannot be smaller than what you are analysing"
        )

    numeric = returns.select_dtypes("number")
    if numeric.shape[1] != returns.shape[1]:
        raise ValueError(
            "returns are not numeric: "
            f"{[str(c) for c in returns.columns if c not in numeric.columns][:5]}"
        )
    if not np.all(np.isfinite(returns.to_numpy(dtype=float))):
        raise ValueError("returns contain NaN or infinite values")

    flat = [c for c in returns.columns if is_degenerate(returns[c])]
    if flat:
        warnings.warn(
            f"dropped {len(flat)} zero-volatility variant(s) with no Sharpe ratio: "
            f"{', '.join(str(c) for c in flat[:5])}"
            + (" ..." if len(flat) > 5 else "")
            + f" (still counted in n_trials={n_trials})",
            stacklevel=2,
        )
        returns = returns.drop(columns=flat)

    if returns.shape[1] < 2:
        raise ValueError(
            "need at least 2 non-degenerate strategy variants to reason about selection"
        )

    sharpes = [sharpe_ratio(returns[c], risk_free=risk_free) for c in returns.columns]
    # V[SR] estimated cross-sectionally across the trials, per Bailey & Lopez
    # de Prado's practical suggestion.
    variance_of_trials = float(np.var(sharpes, ddof=1))

    rows = []
    # Keyed by position, never by name: two columns sharing a label would
    # otherwise collapse and mislabel which strategy the headline DSR describes.
    dsr_by_position: list[DeflatedSharpeResult] = []
    for position, name in enumerate(returns.columns):
        n_obs, skew, kurt = returns_moments(returns[name])
        res = deflated_sharpe_ratio(
            sharpe=sharpes[position],
            n_trials=n_trials,
            n_obs=n_obs,
            variance_of_trials=variance_of_trials,
            skew=skew,
            kurtosis=kurt,
        )
        dsr_by_position.append(res)
        rows.append(
            {
                "strategy": str(name),
                "position": position,
                "sharpe": res.sharpe,
                "annualized_sharpe": annualize_sharpe(res.sharpe, periods_per_year),
                "deflated_sharpe": res.dsr,
                "psr_vs_zero": res.psr_zero,
                "expected_max_sharpe": res.expected_max_sharpe,
                "skew": res.skew,
                "kurtosis": res.kurtosis,
                "n_obs": res.n_obs,
                "n_trials": res.n_trials,
            }
        )

    table = (
        pd.DataFrame(rows).set_index("strategy").sort_values("sharpe", ascending=False)
    )
    best_position = int(table["position"].iloc[0])
    best = str(table.index[0])
    table = table.drop(columns="position")

    return ReportResult(
        table=table,
        pbo=cscv_pbo(returns, n_blocks=n_blocks, force=force),
        best_strategy=best,
        best_dsr=dsr_by_position[best_position],
        n_trials=n_trials,
        periods_per_year=periods_per_year,
        variance_of_trials=variance_of_trials,
    )


def format_report(report: ReportResult, top: int = 10) -> str:
    """Render the report as a printable text block."""
    view = report.table.head(top)[
        ["annualized_sharpe", "deflated_sharpe", "expected_max_sharpe", "skew", "kurtosis"]
    ].rename(
        columns={
            "annualized_sharpe": "ann.Sharpe",
            "deflated_sharpe": "DSR",
            "expected_max_sharpe": "SR* (per-period)",
        }
    )
    p = report.pbo

    lines = [
        "=" * 78,
        "BACKTEST OVERFITTING REPORT",
        "=" * 78,
        f"Strategies analysed : {report.table.shape[0]}",
        f"Trials assumed (N)  : {report.n_trials}",
        f"Observations (T)    : {int(report.table['n_obs'].iloc[0])}",
        f"V[SR] across trials : {report.variance_of_trials:.6g}",
        "",
        f"Top {min(top, report.table.shape[0])} strategies by raw Sharpe:",
        view.to_string(float_format=lambda x: f"{x: .4f}"),
        "",
        "-" * 78,
        "PROBABILITY OF BACKTEST OVERFITTING (CSCV)",
        f"  PBO                  : {p.pbo:.1%}",
        f"  Blocks (S) / splits  : {p.n_blocks} / {p.n_splits}",
        f"  Mean OOS rank of IS winner : {p.mean_relative_rank:.3f}  (0.5 = pure chance)",
        f"  Prob. of OOS loss    : {p.prob_loss:.1%}",
        f"  Observations used    : {p.n_obs_used} ({p.n_obs_dropped} trimmed)",
        "-" * 78,
        "",
        "INTERPRETATION",
        report.interpretation(),
        "=" * 78,
    ]
    return "\n".join(lines)


def plot_report(report: ReportResult, path: str | None = None):
    """Three-panel diagnostic chart; returns the matplotlib Figure.

    Panels: raw vs deflated Sharpe for the top variants, the CSCV logit
    distribution with the PBO region shaded, and the IS-vs-OOS performance
    scatter for the selected strategy.
    """
    import matplotlib

    if path is not None:  # headless rendering when writing to a file
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    p = report.pbo
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    top = report.table.head(10)
    idx = np.arange(len(top))
    # Bar length is the raw Sharpe; colour is the DSR.  Plotting a Sharpe and a
    # probability as side-by-side bars would put two different units on one
    # scale, so the probability is encoded as colour instead.
    cmap = plt.get_cmap("RdYlGn")
    axes[0].barh(idx, top["annualized_sharpe"], color=[cmap(v) for v in top["deflated_sharpe"]])
    for i, (sr, dsr) in enumerate(zip(top["annualized_sharpe"], top["deflated_sharpe"])):
        axes[0].text(
            sr + 0.01 * max(abs(top["annualized_sharpe"]).max(), 1e-9),
            i,
            f"DSR {dsr:.2f}",
            va="center",
            fontsize=7,
        )
    axes[0].set_yticks(idx, top.index, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].margins(x=0.22)
    axes[0].set_xlabel("raw annualized Sharpe")
    axes[0].set_title("Raw Sharpe, coloured by Deflated Sharpe")
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=cmap),
        ax=axes[0],
        label="DSR (probability)",
    )

    axes[1].hist(p.logits, bins=25, color="#4C72B0")
    axes[1].axvline(0, color="crimson", lw=1.5)
    axes[1].set_xlabel(r"logit of OOS relative rank $\lambda_c$")
    axes[1].set_ylabel("splits")
    axes[1].set_title(f"CSCV logits -- PBO = {p.pbo:.1%} (mass left of 0)")

    axes[2].scatter(p.is_performance, p.oos_performance, s=14, alpha=0.6)
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_xlabel("IS performance of selected strategy")
    axes[2].set_ylabel("OOS performance")
    axes[2].set_title("Performance degradation")

    fig.suptitle(
        f"{report.best_strategy}: ann. Sharpe "
        f"{annualize_sharpe(report.best_dsr.sharpe, report.periods_per_year):.2f}, "
        f"DSR {report.best_dsr.dsr:.2f}, PBO {p.pbo:.1%}"
    )
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig
