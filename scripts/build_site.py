"""Assemble the static site published to GitHub Pages.

The site is a client-side application: ``web/`` holds the hand-written assets,
and the analysis runs in the browser via Pyodide against a wheel of this very
package.  This script only copies those assets and pre-computes one example
result, so the page has something on screen during the runtime download rather
than sitting blank.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from overfit_detector.data import make_synthetic_strategies
from overfit_detector.report import build_report, format_report, plot_report
from overfit_detector import __version__

WEB = Path(__file__).resolve().parent.parent / "web"


def _preview() -> dict:
    """The noise demo, in the same payload shape worker.js returns."""
    frame = make_synthetic_strategies(
        n_strategies=50, n_periods=1000, mu=0.0, sigma=0.01, seed=42
    )
    report = build_report(frame)

    fig = plot_report(report)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")

    return {
        "summary": report.to_dict(),
        "text": format_report(report, top=10),
        "png": base64.b64encode(buf.getvalue()).decode(),
        "notes": [],
        "shape": list(frame.shape),
    }


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "site")
    out_dir.mkdir(parents=True, exist_ok=True)

    for asset in ("index.html", "app.js", "worker.js", "style.css"):
        shutil.copy2(WEB / asset, out_dir / asset)

    payload = json.dumps(_preview(), default=float)
    (out_dir / "demo.json").write_text(payload, encoding="utf-8")

    # The worker reads the wheel's name from here rather than hard-coding it,
    # so bumping the version in pyproject.toml cannot silently break the page.
    wheels = sorted(w.name for w in out_dir.glob("*.whl"))
    (out_dir / "manifest.json").write_text(
        json.dumps({"wheel": wheels[-1] if wheels else None}), encoding="utf-8"
    )

    # Pages would otherwise run Jekyll, which skips files it does not recognise.
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    wheel = out_dir / f"overfit_detector-{__version__}-py3-none-any.whl"
    if not wheel.is_file():
        raise SystemExit(
            f"missing {wheel.name}; build it first with "
            f"python -m pip wheel --no-deps --wheel-dir {out_dir} ."
        )

    print(f"wrote site to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
