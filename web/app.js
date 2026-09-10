/* UI for the detector. All statistics happen in worker.js; this file only
   collects inputs, guards the expensive knobs, and renders what comes back. */

const $ = (id) => document.getElementById(id);

const worker = new Worker("worker.js");
let ready = false;
let csvText = null;
let lastPayload = null;
let runId = 0;

/* ---------- boot progress ---------- */

const STAGES = { runtime: 20, packages: 65, package: 85, warmup: 95 };

worker.onmessage = (event) => {
  const msg = event.data;

  if (msg.type === "progress") {
    $("boot-fill").style.width = (STAGES[msg.stage] ?? 10) + "%";
    $("boot-text").textContent = msg.detail + "…";
    return;
  }

  if (msg.type === "ready") {
    ready = true;
    $("boot-fill").style.width = "100%";
    $("boot").classList.add("done");
    $("version").textContent = "v" + msg.version;
    setEnabled(true);
    $("preview-badge").hidden = true;
    return;
  }

  if (msg.type === "fatal") {
    $("boot-text").textContent = "Could not start the Python runtime.";
    showError(
      msg.message +
        "\n\nThis usually means the CDN was blocked or the browser lacks " +
        "WebAssembly support. The example results below still stand."
    );
    return;
  }

  if (msg.type === "result") {
    if (msg.id !== runId) return; // a newer run superseded this one
    $("run").disabled = false;
    $("run-status").textContent = "";
    render(msg.payload, false);
  }
};

function setEnabled(on) {
  // Inputs, tabs, and CSV selection should remain usable while Pyodide starts.
  // A cold cache can take long enough that a page which disables everything
  // feels broken.  Only analysis itself needs the runtime.
  $("run").disabled = !on;
}

/* ---------- tabs ---------- */

let source = "generate";
for (const name of ["generate", "upload"]) {
  $("tab-" + name).addEventListener("click", () => {
    source = name;
    for (const other of ["generate", "upload"]) {
      $("tab-" + other).classList.toggle("active", other === name);
      $("pane-" + other).hidden = other !== name;
    }
  });
}

/* ---------- file input ---------- */

const drop = $("drop");
$("browse").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (e) => takeFile(e.target.files[0]));

drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  takeFile(e.dataTransfer.files[0]);
});

function takeFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    csvText = reader.result;
    $("file-name").textContent = `${file.name} — ${(file.size / 1024).toFixed(0)} KB`;
  };
  reader.readAsText(file);
}

/* ---------- blocks guard ---------- */
/* Splits are C(S, S/2), which explodes fast: S=20 is already 184,756. The
   slider stops at 20 precisely because C(22,11) = 705,432 clears pbo.py's
   200,000 hard cap, so every reachable setting is one the library accepts and
   the only thing left to warn about is how long it will take. */

function comb(n, k) {
  let out = 1;
  for (let i = 1; i <= k; i++) out = (out * (n - k + i)) / i;
  return Math.round(out);
}

function updateBlocks() {
  const s = Number($("p-blocks").value);
  const splits = comb(s, s / 2);
  $("blocks-out").textContent = `S = ${s} — ${splits.toLocaleString()} splits`;

  const warn = $("blocks-warn");
  if (splits > 10000) {  // pbo.py's own _WARN_SPLITS threshold
    warn.hidden = false;
    warn.textContent = `${splits.toLocaleString()} splits runs slowly in the browser — expect to wait.`;
  } else {
    warn.hidden = true;
  }
}
$("p-blocks").addEventListener("input", updateBlocks);
updateBlocks();

/* ---------- run ---------- */

$("run").addEventListener("click", () => {
  if (!ready) return;
  if (source === "upload" && !csvText) {
    showError("Choose a CSV first.");
    return;
  }

  $("error").hidden = true;
  $("run").disabled = true;
  $("run-status").textContent = "Running…";

  const num = (id) => Number($(id).value);
  worker.postMessage({
    type: "run",
    id: ++runId,
    config: {
      source,
      csv: csvText,
      index_col: $("index-col").value.trim(),
      generate: {
        n_strategies: num("g-n_strategies"),
        n_periods: num("g-n_periods"),
        mu: num("g-mu"),
        sigma: num("g-sigma"),
        autocorr: num("g-autocorr"),
        n_signal: num("g-n_signal"),
        signal_mu: num("g-signal_mu"),
        seed: $("g-seed").value.trim(),
      },
      params: {
        blocks: num("p-blocks"),
        periods_per_year: num("p-periods_per_year"),
        risk_free: num("p-risk_free"),
        n_trials: $("p-n_trials").value.trim(),
        top: num("p-top"),
      },
    },
  });
});

/* ---------- rendering ---------- */

function showError(text) {
  $("error").hidden = false;
  $("error").textContent = text;
}

const pct = (x) => (x * 100).toFixed(1) + "%";

function render(payload, isPreview) {
  if (payload.error) {
    showError(payload.error);
    if (payload.notes && payload.notes.length) {
      $("notes").hidden = false;
      $("notes").textContent = payload.notes.join("\n");
    }
    return;
  }

  lastPayload = payload;
  const s = payload.summary;
  $("results").hidden = false;
  $("preview-badge").hidden = !isPreview;

  const notes = $("notes");
  notes.hidden = !(payload.notes && payload.notes.length);
  if (!notes.hidden) notes.textContent = payload.notes.join("\n");

  const verdict = $("verdict");
  verdict.textContent = s.interpretation;
  // Mirrors ReportResult.interpretation()'s own tiers, so colour and words agree.
  const dsr = s.deflated_sharpe.dsr;
  const bad = s.pbo.pbo >= 0.5 || dsr < 0.5;
  const good = !bad && s.pbo.pbo < 0.25 && dsr >= 0.9;
  verdict.classList.toggle("good", good);
  verdict.classList.toggle("bad", bad);

  $("fig-dsr").textContent = s.deflated_sharpe.dsr.toFixed(3);
  $("fig-pbo").textContent = pct(s.pbo.pbo);
  $("fig-best").textContent = s.best_strategy;
  $("fig-sharpe").textContent =
    "annualised Sharpe " + s.deflated_sharpe.annualized_sharpe.toFixed(2);

  $("chart").src = "data:image/png;base64," + payload.png;
  $("report").textContent = payload.text;
}

$("download").addEventListener("click", () => {
  if (!lastPayload) return;
  const blob = new Blob([JSON.stringify(lastPayload.summary, null, 2)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "overfit-report.json";
  a.click();
  URL.revokeObjectURL(a.href);
});

/* ---------- precomputed preview, so the page is never blank ---------- */

setEnabled(false);
fetch("demo.json")
  .then((r) => (r.ok ? r.json() : null))
  .then((payload) => {
    if (payload && !ready) render(payload, true);
  })
  .catch(() => {});
