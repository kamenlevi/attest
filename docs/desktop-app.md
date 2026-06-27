# The Attest desktop app

A local web app wired straight to the engine, shown in a native window. Sleek
black-and-white, light/dark mode, and every feature from the engine: any model
(local or cloud), document ingestion (incl. math-aware vision extraction),
incremental indexing, the full retrieval pipeline (hybrid + HyDE + rerank), and
grounded answers that cite their source or abstain — never guess.

## Run it (from source)

```bash
pip install -e '.[ingest,embed-local,vision,ui]'
attest ui                 # opens a native window (pywebview)
attest ui --no-window     # serve only; opens your browser instead
```

First launch seeds its settings from your `.env` / `ATTEST_*` vars, so if the CLI
already works, the app does too. Settings persist in `~/.attest/config.json`.

## What's in the window

- **Ask** — type a question; get a grounded answer with a trust badge:
  - 🟢 **Grounded · cited [n]** — answered from your sources, with the passage ids
  - 🟡 **Not in your sources** — the honest abstention (it won't guess)
  - 🔴 **Unverifiable** — only if you enable "allow uncited"; the model's own answer, clearly flagged
  Expand *retrieved passages* to see exactly what it read.
- **Library** — add a PDF/text file (embedded once; re-adding is instant). Tick
  **Math-aware extraction** for equation-dense PDFs (renders pages → clean LaTeX
  via the vision model).
- **Settings** — provider (base URL + key), models (generator / judge / vision /
  embedder), the retrieval pipeline toggles, passages `k`, allow-uncited, theme.

## Package a downloadable app (Linux & macOS)

The app is plain Python + a webview, so [PyInstaller](https://pyinstaller.org)
produces a self-contained binary per OS:

```bash
pip install pyinstaller
pyinstaller --name Attest --windowed \
  --collect-all attest --add-data "attest/ui/static:attest/ui/static" \
  -c "import attest.ui.launch as l; l.run()"   # entry shim
```

Notes:
- Build on each target OS (PyInstaller is not a cross-compiler): a Linux box for
  the Linux build, a Mac for the `.app` / `.dmg`.
- The embedding model and `torch` make the bundle large; for a lean cloud-only
  build, ship without `embed-local` and use the `mock` embedder or a remote one.
- On the Mac, the same UI will drive a local **MLX** generator/vision model once
  that backend lands — no UI changes needed (it's just another "any model").
