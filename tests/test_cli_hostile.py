"""The CLI is the surface a stranger touches first, so it gets its own gauntlet.

The governing rule: a bad invocation must exit non-zero with a readable
``error:`` line and **no traceback**.  A Python traceback in front of a user is
a bug, not an error message.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLE_CSV = REPO / "examples" / "sample_returns.csv"


def run(*args):
    return subprocess.run(
        [sys.executable, "-m", "overfit_detector", *args],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def assert_clean_failure(proc, expected_fragment=None):
    """Non-zero exit, an 'error:' line, and no leaked traceback."""
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"expected failure, got success:\n{combined}"
    assert "Traceback" not in combined, f"leaked a traceback:\n{combined}"
    assert "error" in combined.lower(), f"no readable error line:\n{combined}"
    if expected_fragment:
        assert expected_fragment in combined, combined


@pytest.fixture
def good_csv(tmp_path):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(0, 0.01, size=(400, 5)),
                         columns=[f"s{i}" for i in range(5)])
    frame.index = pd.bdate_range("2022-01-03", periods=400, name="date")
    path = tmp_path / "good.csv"
    frame.to_csv(path)
    return path


# --------------------------------------------------------------------------
# Malformed files
# --------------------------------------------------------------------------

def test_missing_file(tmp_path):
    assert_clean_failure(run("--strategies-csv", str(tmp_path / "nope.csv")), "no such file")


def test_directory_instead_of_file(tmp_path):
    assert_clean_failure(run("--strategies-csv", str(tmp_path)), "is a directory")


def test_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    assert_clean_failure(run("--strategies-csv", str(path)), "is empty")


def test_header_only(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text("date,a,b\n")
    assert_clean_failure(run("--strategies-csv", str(path), "--index-col", "date"))


def test_single_row(tmp_path):
    path = tmp_path / "one.csv"
    path.write_text("date,a,b\n2024-01-01,0.01,0.02\n")
    assert_clean_failure(run("--strategies-csv", str(path), "--index-col", "date"))


def test_semicolon_delimited_gets_actionable_advice(tmp_path):
    """A European CSV export used to surface a raw pandas ParserError."""
    path = tmp_path / "semi.csv"
    path.write_text("date;a;b\n2024-01-01;0,01;0,02\n2024-01-02;0,02;0,01\n")
    proc = run("--strategies-csv", str(path), "--index-col", "date")
    assert_clean_failure(proc)
    assert "semicolon" in (proc.stdout + proc.stderr)


def test_bom_prefixed_file_is_read(good_csv, tmp_path):
    """Excel writes a UTF-8 BOM; it must not break the header parsing."""
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbf" + good_csv.read_bytes())
    proc = run("--strategies-csv", str(path), "--index-col", "date", "--blocks", "8")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PBO" in proc.stdout


def test_non_utf8_file(tmp_path):
    path = tmp_path / "latin.csv"
    path.write_bytes(b"date,a,b\n2024-01-01,0.01,\xff\xfe\n")
    assert_clean_failure(run("--strategies-csv", str(path), "--index-col", "date"))


def test_only_one_numeric_column(tmp_path):
    path = tmp_path / "one_col.csv"
    path.write_text("date,only\n2024-01-01,0.01\n2024-01-02,-0.02\n")
    assert_clean_failure(run("--strategies-csv", str(path), "--index-col", "date"))


def test_forgetting_index_col_is_survivable(good_csv):
    """Without --index-col the date column is non-numeric; it should be
    reported and ignored, not crash."""
    proc = run("--strategies-csv", str(good_csv), "--blocks", "8")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "non-numeric" in proc.stderr
    assert "PBO" in proc.stdout


def test_extra_text_column_is_ignored(good_csv, tmp_path):
    frame = pd.read_csv(good_csv, index_col="date")
    frame["note"] = "hello"
    path = tmp_path / "with_text.csv"
    frame.to_csv(path)
    proc = run("--strategies-csv", str(path), "--index-col", "date", "--blocks", "8")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "note" in proc.stderr


def test_nan_rows_are_dropped_with_a_note(good_csv, tmp_path):
    frame = pd.read_csv(good_csv, index_col="date")
    frame.iloc[5, 0] = np.nan
    path = tmp_path / "nan.csv"
    frame.to_csv(path)
    proc = run("--strategies-csv", str(path), "--index-col", "date", "--blocks", "8")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "dropped 1 rows" in proc.stderr


def test_flat_column_warns_but_still_reports(good_csv, tmp_path):
    frame = pd.read_csv(good_csv, index_col="date")
    frame["cash"] = 0.0
    path = tmp_path / "cash.csv"
    frame.to_csv(path)
    proc = run("--strategies-csv", str(path), "--index-col", "date", "--blocks", "8")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "zero-volatility" in proc.stderr
    assert "PBO" in proc.stdout


def test_unsorted_dates_warn(good_csv, tmp_path):
    frame = pd.read_csv(good_csv, index_col="date").sample(frac=1.0, random_state=1)
    path = tmp_path / "unsorted.csv"
    frame.to_csv(path)
    proc = run("--strategies-csv", str(path), "--index-col", "date", "--blocks", "8")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "increasing order" in proc.stderr


# --------------------------------------------------------------------------
# Bad flags
# --------------------------------------------------------------------------

@pytest.mark.parametrize("blocks", ["0", "-4", "3", "7", "1000"])
def test_bad_block_counts(blocks):
    assert_clean_failure(run("--demo", "noise", "--blocks", blocks))


@pytest.mark.parametrize("top", ["0", "-5"])
def test_bad_top_counts(top):
    assert_clean_failure(run("--demo", "noise", "--top", top), "--top must be at least 1")


@pytest.mark.parametrize("ppy", ["0", "-252"])
def test_bad_periods_per_year(ppy):
    assert_clean_failure(run("--demo", "noise", "--periods-per-year", ppy))


def test_n_trials_below_column_count():
    assert_clean_failure(run("--demo", "noise", "--n-trials", "3"))


def test_absurd_trial_count_is_handled_not_silently_infinite():
    """Regression: this used to exit 0 with a confident DSR of 0.0 computed
    from an infinite hurdle."""
    proc = run("--demo", "noise", "--n-trials", "1" + "0" * 18, "--blocks", "8")
    combined = proc.stdout + proc.stderr
    assert "Traceback" not in combined
    if proc.returncode == 0:
        # If it succeeds, the hurdle must be a real number, not an infinity.
        assert "inf" not in proc.stdout.lower()


def test_no_data_source_given():
    assert_clean_failure(run())


def test_both_data_sources_given(good_csv):
    assert_clean_failure(run("--demo", "noise", "--strategies-csv", str(good_csv)))


def test_unknown_demo():
    assert_clean_failure(run("--demo", "wishful"))


# --------------------------------------------------------------------------
# Successful paths still work
# --------------------------------------------------------------------------

def test_example_csv_still_runs():
    proc = run("--strategies-csv", str(EXAMPLE_CSV), "--index-col", "date", "--blocks", "10")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "INTERPRETATION" in proc.stdout


@pytest.mark.parametrize("demo", ["noise", "signal"])
def test_json_round_trips(demo):
    proc = run("--demo", demo, "--blocks", "8", "--json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert set(payload) >= {"pbo", "deflated_sharpe", "interpretation", "strategies"}
    assert 0.0 <= payload["pbo"]["pbo"] <= 1.0
    assert 0.0 <= payload["deflated_sharpe"]["dsr"] <= 1.0
    assert len(payload["strategies"]) == 50
    # Every value must survive a JSON round trip as a real number.
    assert all(np.isfinite(v) for v in payload["pbo"].values())


def test_help_works():
    proc = run("--help")
    assert proc.returncode == 0
    assert "--strategies-csv" in proc.stdout
