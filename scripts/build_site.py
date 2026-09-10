"""Generate the static site published to GitHub Pages.

Runs the detector on both bundled demos and writes an HTML page showing the
report text and diagnostic chart for each, so the deployed link is a live
example of what the tool produces rather than a copy of the README.
"""

from __future__ import annotations

import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from overfit_detector.data import make_synthetic_strategies
from overfit_detector.report import build_report, format_report, plot_report

DEMOS = {
    "noise": (
        "50 skill-free variants",
        "Every variant is pure noise. A high PBO and a Deflated Sharpe near "
        "zero are the correct answer here — the best-looking backtest is "
        "just the luckiest one.",
    ),
    "signal": (
        "49 skill-free variants plus one real edge",
        "One variant has a genuine positive drift. The Deflated Sharpe Ratio "
        "should survive the multiple-testing correction that sinks the noise "
        "demo.",
    ),
}


def _run(kind: str, out_dir: Path) -> tuple[str, dict]:
    frame = make_synthetic_strategies(
        n_strategies=50,
        n_periods=1000,
        mu=0.0,
        sigma=0.01,
        n_signal=1 if kind == "signal" else 0,
        signal_mu=0.002,
        seed=42,
    )
    report = build_report(frame)
    plot_report(report, path=str(out_dir / f"{kind}.png"))
    payload = report.to_dict()
    (out_dir / f"{kind}.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )
    return format_report(report), payload


def _section(kind: str, text: str) -> str:
    subtitle, blurb = DEMOS[kind]
    return f"""
    <section id="{kind}">
      <h2>Demo: <code>--demo {kind}</code></h2>
      <p class="subtitle">{html.escape(subtitle)}</p>
      <p>{blurb}</p>
      <img src="{kind}.png" alt="Diagnostic chart for the {kind} demo">
      <pre>{html.escape(text)}</pre>
      <p class="links"><a href="{kind}.json">Raw JSON</a></p>
    </section>
    """


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "site")
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = []
    for kind in DEMOS:
        text, _ = _run(kind, out_dir)
        sections.append(_section(kind, text))

    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backtest Overfitting Detector</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ max-width: 52rem; margin: 0 auto; padding: 2rem 1rem;
         font: 16px/1.6 system-ui, -apple-system, sans-serif; }}
  h1 {{ margin-bottom: .25rem; }}
  .subtitle {{ color: #666; margin-top: 0; }}
  pre {{ overflow-x: auto; padding: 1rem; border-radius: 6px;
        background: rgba(127,127,127,.12); font-size: 13px; }}
  img {{ max-width: 100%; height: auto; border-radius: 6px; }}
  section {{ margin-top: 3rem; }}
  footer {{ margin-top: 4rem; color: #666; font-size: 14px; }}
</style>
</head>
<body>
<h1>Backtest Overfitting Detector</h1>
<p class="subtitle">Deflated Sharpe Ratio and Probability of Backtest
Overfitting, run on the two bundled demos.</p>
<p>Source: <a
href="https://github.com/Githubneos/Backtest-Overfitting-Detector">GitHub</a>.
Reproduce either panel locally with
<code>python -m overfit_detector --demo noise</code>.</p>
{"".join(sections)}
<footer>Built {built} from the latest commit on <code>main</code>.</footer>
</body>
</html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    print(f"wrote site to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
