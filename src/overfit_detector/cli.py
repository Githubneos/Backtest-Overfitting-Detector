"""Command-line entry point: ``python -m overfit_detector``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .data import NotEnoughStrategies, make_synthetic_strategies, prepare_returns
from .pbo import DEFAULT_BLOCKS
from .report import build_report, format_report, plot_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m overfit_detector",
        description=(
            "Quantify how likely a backtest is overfit, via the Deflated Sharpe "
            "Ratio and the Probability of Backtest Overfitting (CSCV)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--strategies-csv",
        help="CSV of per-period returns: one column per strategy variant, rows = time.",
    )
    source.add_argument(
        "--demo",
        choices=["noise", "signal"],
        help="Run on generated data instead of a file: 50 skill-free variants, "
        "or 49 plus one with a real edge.",
    )
    parser.add_argument(
        "--index-col",
        default=None,
        help="Column to use as the time index (e.g. 'date'); dropped from the analysis.",
    )
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS,
                        help="CSCV block count S (even, >= 4).")
    parser.add_argument("--periods-per-year", type=float, default=252.0,
                        help="Annualization factor for display only.")
    parser.add_argument("--risk-free", type=float, default=0.0,
                        help="Per-period risk-free rate.")
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Variants actually tried, if more than the columns supplied. "
             "Defaults to the number of columns.",
    )
    parser.add_argument("--top", type=int, default=10,
                        help="How many strategies to show in the table (>= 1).")
    parser.add_argument("--plot", metavar="PATH", default=None,
                        help="Write the diagnostic chart to this path.")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON instead of the text report.")
    parser.add_argument("--force", action="store_true",
                        help="Allow split counts above the internal safety cap.")
    return parser


def _load(args: argparse.Namespace) -> pd.DataFrame:
    if args.demo:
        return make_synthetic_strategies(
            n_strategies=50,
            n_periods=1000,
            mu=0.0,
            sigma=0.01,
            n_signal=1 if args.demo == "signal" else 0,
            signal_mu=0.002,
            seed=42,
        )

    path = Path(args.strategies_csv)
    if path.is_dir():
        raise SystemExit(f"error: {path} is a directory, not a CSV file")
    try:
        # utf-8-sig transparently strips a BOM, which Excel exports carry.
        frame = pd.read_csv(path, index_col=args.index_col, encoding="utf-8-sig")
    except FileNotFoundError:
        raise SystemExit(f"error: no such file: {path}") from None
    except PermissionError:
        raise SystemExit(f"error: cannot read {path}: permission denied") from None
    except UnicodeDecodeError:
        raise SystemExit(
            f"error: {path} is not valid UTF-8 text; re-export it as a UTF-8 CSV"
        ) from None
    except pd.errors.EmptyDataError:
        raise SystemExit(f"error: {path} is empty") from None
    except pd.errors.ParserError as exc:
        raise SystemExit(
            f"error: could not parse {path} as CSV ({exc}); if it is "
            "semicolon-delimited or uses commas as decimal separators, convert "
            "it to a standard comma-delimited CSV first"
        ) from None

    if args.index_col is not None:
        # read_csv leaves the index as strings, which would make the
        # out-of-order check in cscv_pbo dead code for the CLI's own input.
        try:
            frame.index = pd.to_datetime(frame.index)
        except (ValueError, TypeError):
            pass  # a non-date index is fine; it just cannot be order-checked

    try:
        frame, notes = prepare_returns(frame)
    except NotEnoughStrategies as exc:
        for note in exc.notes:
            print(f"note: {note}", file=sys.stderr)
        raise SystemExit(f"error: {args.strategies_csv} {exc}") from None
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    return frame


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.top < 1:
        raise SystemExit("error: --top must be at least 1")
    frame = _load(args)

    try:
        report = build_report(
            frame,
            n_blocks=args.blocks,
            n_trials=args.n_trials,
            periods_per_year=args.periods_per_year,
            risk_free=args.risk_free,
            force=args.force,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    if args.plot:
        plot_report(report, path=args.plot)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=float))
    else:
        print(format_report(report, top=args.top))
        if args.plot:
            print(f"\nChart written to {args.plot}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
