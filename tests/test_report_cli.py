import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from overfit_detector.data import make_synthetic_strategies
from overfit_detector.report import build_report, format_report, plot_report

EXAMPLE_CSV = Path(__file__).resolve().parents[1] / "examples" / "sample_returns.csv"


@pytest.fixture(scope="module")
def noise_report():
    data = make_synthetic_strategies(n_strategies=20, n_periods=800, mu=0.0, seed=42)
    return build_report(data, n_blocks=10)


@pytest.fixture(scope="module")
def signal_report():
    data = make_synthetic_strategies(
        n_strategies=20, n_periods=800, mu=0.0, n_signal=1, signal_mu=0.002, seed=42
    )
    return build_report(data, n_blocks=10)


# --------------------------------------------------------------------------
# build_report
# --------------------------------------------------------------------------

def test_table_has_one_row_per_strategy_sorted_by_sharpe(noise_report):
    table = noise_report.table
    assert len(table) == 20
    assert table["sharpe"].is_monotonic_decreasing
    assert {"sharpe", "deflated_sharpe", "n_trials", "expected_max_sharpe"} <= set(table.columns)
    assert (table["n_trials"] == 20).all()


def test_annualization_is_display_only(noise_report):
    row = noise_report.table.iloc[0]
    assert row["annualized_sharpe"] == pytest.approx(row["sharpe"] * 252**0.5)
    # The DSR hurdle is reported in per-period units, matching the DSR math.
    assert noise_report.best_dsr.expected_max_sharpe == row["expected_max_sharpe"]


def test_variance_of_trials_is_cross_sectional(noise_report):
    import numpy as np

    assert noise_report.variance_of_trials == pytest.approx(
        np.var(noise_report.table["sharpe"], ddof=1)
    )


def test_n_trials_override_raises_the_hurdle():
    data = make_synthetic_strategies(n_strategies=20, n_periods=800, mu=0.0, seed=42)
    few = build_report(data, n_blocks=10)
    many = build_report(data, n_blocks=10, n_trials=2000)
    assert many.best_dsr.expected_max_sharpe > few.best_dsr.expected_max_sharpe
    assert many.best_dsr.dsr < few.best_dsr.dsr


def test_n_trials_below_column_count_rejected():
    data = make_synthetic_strategies(n_strategies=20, n_periods=800, seed=1)
    with pytest.raises(ValueError, match="below the 20 variants"):
        build_report(data, n_blocks=10, n_trials=5)


def test_single_strategy_rejected():
    data = make_synthetic_strategies(n_strategies=1, n_periods=800, seed=1)
    with pytest.raises(ValueError, match="at least 2 non-degenerate strategy"):
        build_report(data, n_blocks=10)


def test_noise_and_signal_reports_disagree_as_expected(noise_report, signal_report):
    assert signal_report.best_strategy == "signal_0"
    assert signal_report.pbo.pbo < noise_report.pbo.pbo
    assert signal_report.best_dsr.dsr > noise_report.best_dsr.dsr


# --------------------------------------------------------------------------
# formatting / interpretation
# --------------------------------------------------------------------------

def test_interpretation_reports_the_actual_numbers(noise_report):
    text = noise_report.interpretation()
    assert "You tested 20 variants" in text
    assert f"{100 * noise_report.pbo.pbo:.0f}%" in text
    assert noise_report.best_strategy in text
    assert "caution" in text  # 20 skill-free variants must not get a clean bill


def test_interpretation_clears_a_genuine_edge(signal_report):
    assert "survives both corrections" in signal_report.interpretation()


def test_format_report_contains_table_and_verdict(noise_report):
    text = format_report(noise_report, top=5)
    assert "PROBABILITY OF BACKTEST OVERFITTING" in text
    assert "INTERPRETATION" in text
    assert noise_report.table.index[0] in text
    assert text.count("\n") > 15


def test_to_dict_is_json_serialisable(noise_report):
    payload = json.loads(json.dumps(noise_report.to_dict(), default=float))
    assert payload["pbo"]["pbo"] == pytest.approx(noise_report.pbo.pbo)
    assert len(payload["strategies"]) == 20


def test_plot_writes_a_file_headlessly(noise_report, tmp_path):
    out = tmp_path / "report.png"
    fig = plot_report(noise_report, path=str(out))
    assert out.exists() and out.stat().st_size > 1000
    assert len(fig.axes) == 4  # three panels plus the DSR colourbar


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "overfit_detector", *args],
        capture_output=True,
        text=True,
    )


def test_cli_runs_on_the_example_csv():
    proc = _run("--strategies-csv", str(EXAMPLE_CSV), "--blocks", "10", "--index-col", "date")
    assert proc.returncode == 0, proc.stderr
    assert "PBO" in proc.stdout
    assert "INTERPRETATION" in proc.stdout


def test_cli_json_output_parses():
    proc = _run("--demo", "signal", "--blocks", "8", "--json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["best_strategy"] == "signal_0"
    assert 0.0 <= payload["pbo"]["pbo"] <= 1.0
    assert payload["deflated_sharpe"]["dsr"] > 0.9


def test_cli_plot_flag_writes_chart(tmp_path):
    out = tmp_path / "cli.png"
    proc = _run("--demo", "noise", "--blocks", "8", "--plot", str(out))
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert str(out) in proc.stdout


def test_cli_rejects_odd_block_count():
    proc = _run("--demo", "noise", "--blocks", "7")
    assert proc.returncode != 0
    assert "even integer" in proc.stderr


def test_cli_requires_a_data_source():
    proc = _run()
    assert proc.returncode != 0
    assert "one of the arguments" in proc.stderr


def test_cli_errors_helpfully_on_a_single_column_csv(tmp_path):
    csv = tmp_path / "one.csv"
    csv.write_text("date,only\n2024-01-01,0.01\n2024-01-02,-0.02\n")
    proc = _run("--strategies-csv", str(csv), "--index-col", "date")
    assert proc.returncode != 0
    assert "at least 2 strategy variants" in proc.stderr
