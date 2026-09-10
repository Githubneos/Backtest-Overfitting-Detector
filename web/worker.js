/* Pyodide host. Runs the real overfit_detector package off the main thread so a
   long CSCV sweep never freezes the page. */

importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

let pyodide = null;
let runner = null;

function progress(stage, detail) {
  self.postMessage({ type: "progress", stage, detail });
}

// Mirrors cli.py: the analysis itself is untouched library code.
const GLUE = `
import base64, io, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from overfit_detector.data import (
    NotEnoughStrategies,
    make_synthetic_strategies,
    prepare_returns,
)
from overfit_detector.report import build_report, format_report, plot_report


def _frame_from(cfg):
    notes = []
    if cfg["source"] == "generate":
        g = cfg["generate"]
        seed = g.get("seed")
        frame = make_synthetic_strategies(
            n_strategies=int(g["n_strategies"]),
            n_periods=int(g["n_periods"]),
            mu=float(g["mu"]),
            sigma=float(g["sigma"]),
            autocorr=float(g["autocorr"]),
            n_signal=int(g["n_signal"]),
            signal_mu=float(g["signal_mu"]),
            seed=None if seed in (None, "") else int(seed),
        )
        return frame, notes

    text = cfg["csv"].lstrip("\\ufeff")  # Excel exports carry a BOM
    index_col = cfg.get("index_col") or None
    frame = pd.read_csv(io.StringIO(text), index_col=index_col)
    if index_col is not None:
        # Match the CLI: a date index lets cscv_pbo run its ordering check.
        try:
            frame.index = pd.to_datetime(frame.index)
        except (ValueError, TypeError):
            pass
    return prepare_returns(frame)


def run(cfg):
    # cfg arrives already converted by pyodide.toPy(), so it is a plain dict.
    try:
        frame, notes = _frame_from(cfg)
    except NotEnoughStrategies as exc:
        return json.dumps({"error": str(exc), "notes": exc.notes})
    except ValueError as exc:
        return json.dumps({"error": str(exc), "notes": []})
    except Exception as exc:
        return json.dumps({"error": f"could not read that CSV ({exc})", "notes": []})

    p = cfg["params"]
    n_trials = p.get("n_trials")
    try:
        report = build_report(
            frame,
            n_blocks=int(p["blocks"]),
            n_trials=None if n_trials in (None, "") else int(n_trials),
            periods_per_year=float(p["periods_per_year"]),
            risk_free=float(p["risk_free"]),
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc), "notes": notes})

    fig = plot_report(report)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    return json.dumps({
        "summary": report.to_dict(),
        "text": format_report(report, top=int(p["top"])),
        "png": base64.b64encode(buf.getvalue()).decode(),
        "notes": notes,
        "shape": list(frame.shape),
    }, default=float)
`;

async function boot() {
  progress("runtime", "Downloading Python runtime");
  pyodide = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/",
  });

  progress("packages", "Downloading numpy, pandas, scipy, matplotlib");
  await pyodide.loadPackage(["numpy", "pandas", "scipy", "matplotlib", "micropip"]);

  progress("package", "Installing overfit_detector");
  const manifest = await fetch("manifest.json").then((r) => r.json());
  if (!manifest.wheel) {
    throw new Error(
      "No wheel was published alongside this page, so the analysis engine " +
        "cannot be installed. The deploy step that builds it did not run."
    );
  }
  const micropip = pyodide.pyimport("micropip");
  // deps:false -- numpy/pandas/scipy/matplotlib came from loadPackage above.
  // Left on, micropip re-resolves them against PyPI, where they have no
  // pure-Python wheel, and the install fails.
  // callKwargs, not a plain second argument: Pyodide maps a trailing JS object
  // to Python keyword arguments only through this call form.
  await micropip.install.callKwargs(new URL(manifest.wheel, self.location.href).href, {
    deps: false,
  });

  progress("warmup", "Starting up");
  pyodide.runPython(GLUE);
  runner = pyodide.globals.get("run");

  const version = pyodide.runPython("import overfit_detector; overfit_detector.__version__");
  self.postMessage({ type: "ready", version });
}

const booted = boot().catch((err) => {
  self.postMessage({ type: "fatal", message: String(err) });
});

self.onmessage = async (event) => {
  const msg = event.data;
  if (msg.type !== "run") return;

  await booted;
  if (!runner) return;

  try {
    const cfg = pyodide.toPy(msg.config);
    const raw = runner(cfg);
    cfg.destroy();
    self.postMessage({ type: "result", id: msg.id, payload: JSON.parse(raw) });
  } catch (err) {
    self.postMessage({ type: "result", id: msg.id, payload: { error: String(err), notes: [] } });
  }
};
