# Backtest Overfitting Detector

**[Try it in your browser](https://githubneos.github.io/Backtest-Overfitting-Detector/)**
— upload your own returns or generate synthetic variants. It runs this package
via Pyodide, so nothing you analyse is uploaded anywhere.

A Sharpe ratio is not evidence of skill when it is the **maximum over many tried
variants**. This package quantifies how much of a reported backtest result is
selection bias, using two established techniques:

| Method | Question it answers | Output |
| --- | --- | --- |
| **Deflated Sharpe Ratio (DSR)** | Given that I tried *N* variants over *T* periods, what is the probability this strategy's *true* Sharpe beats what the luckiest skill-free trial would have produced? | probability in [0, 1] |
| **Probability of Backtest Overfitting (PBO)** | How often does the variant that looked best in-sample fail to beat the median out-of-sample? | probability in [0, 1] |

They are complementary, and neither subsumes the other — see
[Limitations](#limitations).

---

## Why this matters

Strategy research is a multiple-testing problem that nobody reports as one. A
researcher sweeps 40 parameter combinations, keeps the one with the best Sharpe,
and writes it up. The reported Sharpe is then the **maximum of 40 noisy
estimates**, and the maximum of 40 draws from a zero-mean distribution is
comfortably positive — with 20 skill-free variants over three years of daily
data, the expected best annualized Sharpe is roughly **1.1 purely by luck**. A
1.6 in that setting is not a 1.6; it is a 1.6 that had 20 chances to happen.

Two things follow, and both are implemented here:

1. **The in-sample Sharpe is a biased estimator when it is selected.** The
   correction is not "shrink it a bit" — it is a specific hurdle that grows with
   the number of trials and the dispersion of trial performance, and shrinks with
   backtest length. That is the DSR.
2. **Bias-corrected point estimates are not enough**; you also want to know
   whether the *selection procedure itself* generalizes. That is PBO: rerun the
   selection on every balanced split of the sample and count how often the winner
   fails out-of-sample.

This is the difference between reporting a number and knowing what the number is
worth. A strategy with a 1.8 Sharpe and a 62% PBO is not a strategy; it is a
search artifact.

---

## The two methods

### Deflated Sharpe Ratio

Bailey & López de Prado (2014), *The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality*, **Journal of Portfolio
Management** 40(5), 94–107.

The DSR is a Probabilistic Sharpe Ratio evaluated against a benchmark that
encodes the multiple testing. First, the expected maximum Sharpe under the null
of no skill, from the Gumbel limit of the maximum of *N* i.i.d. Gaussian draws:

```
SR* = sqrt(V[SR]) · [ (1 − γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]
```

with γ the Euler–Mascheroni constant (≈ 0.5772), `Z⁻¹` the standard-normal
quantile function, *N* the number of trials, and `V[SR]` the variance of the
Sharpe estimates across those trials. Then:

```
DSR = PSR(SR*) = Z[ (SR − SR*)·sqrt(T − 1) / sqrt(1 − γ₃·SR + ((γ₄ − 1)/4)·SR²) ]
```

where γ₃ is the skewness and γ₄ the **non-excess** kurtosis of the returns
(normal ⇒ 3.0), and *T* the number of observations. The denominator is the
standard error of the Sharpe estimator under non-normality: negative skew and
fat tails inflate it, correctly lowering confidence in a Sharpe estimated from
such returns.

`SR*` is exposed as its own function, `expected_max_sharpe(n_trials,
variance_of_trials)`, so the deflation is inspectable rather than a black box —
and every intermediate is returned on the result object.

### Probability of Backtest Overfitting (CSCV)

Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of Backtest
Overfitting*, **Journal of Computational Finance** 20(4), 39–69.

Combinatorially Symmetric Cross-Validation:

1. Split the returns matrix (rows = time, columns = variants) into *S* contiguous
   equal-length blocks. Contiguous, so autocorrelation survives the partition.
2. For each of the `C(S, S/2)` ways to choose half the blocks as in-sample,
   recombine the chosen blocks **in original time order** as *J* (IS) and the
   complement as *J̄* (OOS). Every block is IS in exactly half the splits — this
   is the symmetry the name refers to.
3. Find the best strategy `n*` in-sample; find its rank among all strategies
   out-of-sample (1 = worst, *N* = best).
4. Form the relative rank `ω_c = rank / (N + 1)` and its logit
   `λ_c = ln(ω_c / (1 − ω_c))`. The `N + 1` denominator keeps ω strictly inside
   (0, 1) so the logit is always finite.
5. **`PBO = Pr[λ_c ≤ 0]`** — the fraction of splits where the in-sample winner
   landed at or below the out-of-sample median.

PBO ≈ 0.5 means selection is a coin flip: in-sample rank carries no information
about out-of-sample rank. PBO near 0 means the winner keeps winning.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'          # core + pytest
pip install -e '.[dev,data]'     # also installs yfinance for the real-data loader
```

Dependencies are numpy, pandas, scipy and matplotlib. No ML frameworks.

CSCV computes each split's Sharpe from per-block sufficient statistics rather than
re-reducing the returns matrix, so the expensive settings stay interactive:
`S=16` over 200 variants and 2000 periods (12,870 splits) takes ~0.6s.

## Usage

### CLI

```bash
python -m overfit_detector --strategies-csv examples/sample_returns.csv \
    --index-col date --blocks 10 --top 5
```

Other flags: `--n-trials` (if you tried more variants than you kept — **the
deflation is only as honest as this number**), `--periods-per-year`,
`--risk-free`, `--plot report.png`, `--json`, and `--demo noise|signal` to run on
generated data with known ground truth.

### Example output

Real output from the command above. `examples/sample_returns.csv` is 20
**pure-noise** variants over 756 days, generated with a true mean of exactly zero:

```
==============================================================================
BACKTEST OVERFITTING REPORT
==============================================================================
Strategies analysed : 20
Trials assumed (N)  : 20
Observations (T)    : 756
V[SR] across trials : 0.00137013

Top 5 strategies by raw Sharpe:
            ann.Sharpe     DSR  SR* (per-period)    skew  kurtosis
strategy
variant_11      1.5999  0.7966            0.0704 -0.1070    3.0836
variant_06      0.8366  0.3144            0.0704 -0.0952    2.8156
variant_09      0.6861  0.2278            0.0704  0.0666    3.1356
variant_01      0.5973  0.1849            0.0704 -0.1291    3.1128
variant_05      0.5715  0.1727            0.0704 -0.0043    2.7727

------------------------------------------------------------------------------
PROBABILITY OF BACKTEST OVERFITTING (CSCV)
  PBO                  : 34.5%
  Blocks (S) / splits  : 10 / 252
  Mean OOS rank of IS winner : 0.651  (0.5 = pure chance)
  Prob. of OOS loss    : 28.2%
  Observations used    : 750 (6 trimmed)
------------------------------------------------------------------------------

INTERPRETATION
You tested 20 variants over 756 periods. The top strategy (variant_11) had a raw
annualized Sharpe of 1.60, but 20 skill-free trials would be expected to produce
1.12 by luck alone, leaving a deflated Sharpe (probability the true Sharpe clears
that hurdle) of 0.80. CSCV puts the probability that this result is overfit at
35% -- the evidence is suggestive but not conclusive.
==============================================================================
```

Note what happened: a 1.60 annualized Sharpe, out of returns with a true mean of
**zero**. The hurdle `SR*` alone accounts for 1.12 of it. This is what a lucky
draw looks like from the inside, and it is why "suggestive but not conclusive" is
the right verdict on data where we know the answer is "nothing here".

Contrast `--demo signal` (one genuine edge among 49 noise variants): DSR ≈ 1.00,
PBO = 0.0%, mean OOS rank 0.98.

### On real data

`load_yfinance_strategies("SPY", start="2018-01-01", end="2024-01-01")` builds 24
moving-average crossover variants and hands them to the same report:

```
Top 3 strategies by raw Sharpe:
           ann.Sharpe     DSR  SR* (per-period)    skew  kurtosis
strategy
ma_10_100      0.8109  0.9118            0.0155 -0.7270    9.9391
ma_10_50       0.7381  0.8811            0.0155 -0.6962   11.0459
ma_5_150       0.7346  0.8788            0.0155 -0.8381    9.4222

  PBO                        : 86.9%
  Mean OOS rank of IS winner : 0.199  (0.5 = pure chance)
```

This is the most instructive output in the repo, because **the two methods
disagree**. DSR says 0.91: the 24 variants all perform similarly, so `V[SR]` is
tiny, the luck hurdle `SR*` is only 0.25 annualized, and a 0.81 clears it easily.
PBO says 87%: whichever variant happens to lead in-sample lands near the *bottom*
of the out-of-sample ranking almost every time. Both are correct about different
questions. The MA grid as a family may be mildly positive on a rising index —
but *picking the best member of it* is worthless, and picking is what a parameter
sweep does. Note also the returns: skew ≈ −0.8 and kurtosis ≈ 10, precisely the
non-normality the PSR denominator exists to penalize.

### Library

```python
from overfit_detector import build_report, format_report, cscv_pbo, deflated_sharpe_ratio

report = build_report(returns_df, n_blocks=10, n_trials=200, periods_per_year=252)
print(format_report(report))

report.pbo.pbo                       # 0.345
report.best_dsr.dsr                  # 0.797
report.best_dsr.expected_max_sharpe  # 0.0704 (per-period hurdle)
report.to_dict()                     # JSON-ready
```

Lower-level entry points — `sharpe_ratio`, `expected_max_sharpe`,
`probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `cscv_pbo` — are all
importable directly and carry their formulas in their docstrings.

Data helpers: `make_synthetic_strategies(...)` builds AR(1) return streams with
configurable mean, volatility, autocorrelation and an optional embedded signal;
`load_yfinance_strategies("SPY")` builds a moving-average crossover parameter
grid on a real ticker (needs the `[data]` extra).

---

## Limitations

Worth stating plainly, because they are the interesting part:

- **PBO measures selection overfitting *relative to the sample you have*.** If one
  variant got lucky across the *entire* backtest, it looks good in every split
  and PBO stays low. That kind of luck is exactly what DSR catches, via the
  trial count. Run both; they fail in different directions.
- **`N` is your responsibility.** The deflation defaults to the number of columns
  supplied, which undercounts if you discarded variants before saving the file.
  Pass `--n-trials` honestly. Trials that are highly correlated with each other
  are also effectively fewer than *N* independent trials.
- **The IS-vs-OOS scatter is not a regression to read a slope off.** IS and OOS
  are complementary halves of one fixed sample, so their sum is pinned; a fitted
  slope is dragged toward −1 whenever the same strategy is consistently selected
  — which is what a *good* strategy does. The panel is a qualitative view; the
  headline numbers are PBO, mean OOS rank, and probability of loss.
- All DSR math runs on **per-period** Sharpes; annualization is display-only.
  Mixing the two is the most common bug in DSR implementations found in the wild.
- **Row order matters, and the tool warns you.** CSCV blocks are contiguous slices of
  the row order. An exact reversal (a newest-first export) happens to be harmless —
  it just relabels the blocks, and the family of balanced splits is closed under
  relabeling — but any partial disorder silently changes the answer. Unsorted dates
  raise a warning.
- **DSR's response to backtest length flips sign at the hurdle.** `T`, skew and
  kurtosis enter only through the standard error, scaling a z-score whose sign is set
  by `SR − SR*`. So a longer backtest *raises* the DSR of a strategy above the hurdle
  and *lowers* it for one below: more evidence increases confidence in whatever is
  true. "More data always helps" is only right if the strategy is genuinely good.
- **Ties use midranks.** Tied out-of-sample performances share the average of the
  ranks they span, so identical variants land at exactly `ω = 0.5`, `λ = 0` — which
  counts toward PBO under the paper's `Pr[λ ≤ 0]` convention. That is the honest
  reading: if every variant is the same, the selection carried no information.

## Tests

```bash
pytest -q                  # 198 tests
pytest -q -m "not slow"    # skip the Monte Carlo validation
```

The suite is in two halves. The unit tests check each formula against closed forms
and hand-computed values. The **gauntlet** — `test_montecarlo.py`,
`test_adversarial.py`, `test_properties.py`, `test_cli_hostile.py` — checks the
things unit tests structurally cannot:

- **Monte Carlo validation against simulated ground truth.** `SR*` reproduces the
  simulated `E[max]` of N skill-free Sharpes (within 2.4% at N=10, 0.1% at N=1000,
  with the error shrinking as N grows — the signature of a correct extreme-value
  limit). PSR is verified *calibrated*: under the null its distribution is uniform
  (KS test), and its 5% tail rates stay nominal under t(4) and skewed-gamma returns,
  which is precisely what the skew and kurtosis terms exist to deliver. A formula
  error would survive every other test in this repo and die here.
- **Invariants**: scale invariance (fractions vs percent vs basis points),
  column-permutation invariance, monotonicity in every DSR argument, bounds on every
  probability, and the combinatorial symmetry the method is named for (each block is
  in-sample in exactly half the splits).
- **Hostile input**: flat "hold cash" columns, all-zero matrices, `-100%` days,
  kurtosis ≈ 400 outliers, returns quoted at `1e-12` and `1e6`, NaN/inf, duplicate
  column names, non-numeric data, trial counts up to `10**300`, and every bad block
  count. Each must give a correct answer or a clear error — never a confident wrong
  number.
- **CLI**: malformed CSVs (empty, header-only, semicolon-delimited, BOM-prefixed,
  non-UTF-8, single-column), bad flags, and an assertion that no failure ever leaks
  a Python traceback.

Alongside that, the statistical behaviour that matters:

- 50 pure-noise variants → PBO > 0.6 on the pinned seed, and averaged over 30
  independent datasets PBO sits near 0.5 (measured: 0.55 / 0.51 / 0.45 for 5 / 20 /
  100 variants — a mild *downward* drift as variant count grows). A single noise
  dataset is a terrible estimate of this: individual runs range from 0.06 to 1.00,
  which is itself pinned by a test.
- One genuine-signal variant among 49 noise → PBO < 0.2, mean OOS rank > 0.9,
  probability of OOS loss < 5%, and the signal column selected in > 90% of splits.
- Deterministic fixtures where the IS winner and its OOS rank are known by
  construction, so the rank/logit path is verified exactly rather than
  statistically.
- The same separation holds with autocorrelated returns (AR(1), φ = 0.3).

## Project layout

```
src/overfit_detector/
  metrics.py          Sharpe, annualization, skew/kurtosis (per-period units)
  deflated_sharpe.py  expected_max_sharpe, probabilistic_sharpe_ratio, DSR
  pbo.py              CSCV -> PBOResult
  data.py             synthetic generator + yfinance loader (optional extra)
  report.py           combined report, text table, matplotlib chart
  cli.py              python -m overfit_detector
tests/                198 tests (5 unit files + 4 gauntlet files)
examples/             sample_returns.csv used by the README example
```

## References

- Bailey, D. H. and López de Prado, M. (2014). *The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality.*
  Journal of Portfolio Management, 40(5), 94–107.
- Bailey, D. H., Borwein, J. M., López de Prado, M. and Zhu, Q. J. (2017).
  *The Probability of Backtest Overfitting.* Journal of Computational Finance,
  20(4), 39–69.
- Bailey, D. H. and López de Prado, M. (2012). *The Sharpe Ratio Efficient
  Frontier.* Journal of Risk, 15(2), 3–44. (Origin of the Probabilistic Sharpe
  Ratio used as the DSR's engine.)

## License

MIT
