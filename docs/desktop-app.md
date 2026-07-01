# The Attest desktop app

A local web app wired straight to the engine, shown in a native window. Sleek
black-and-white, light/dark mode, and every feature from the engine: any model
(cloud or local), document ingestion (incl. math-aware vision extraction),
incremental indexing, the full retrieval pipeline (hybrid + HyDE + rerank), and
**verified** grounded answers that cite their source or abstain — never guess.

## Run it (from source)

```bash
pip install -e '.[ingest,embed-local,vision,ui]'
attest ui                 # opens a native window (pywebview)
attest ui --no-window     # serve only; opens your browser instead
```

First launch seeds its settings from your `.env` / `ATTEST_*` vars, so if the CLI
already works, the app does too. Settings persist in `~/.attest/config.json`.

## What's in the window

- **Ask** — type a question; get a grounded answer whose badge reflects the
  **verification ladder**, not the model's own claim:
  - 🟢 **Verified** — the citations point at passages the model was really shown,
    AND an independent judge model confirmed those passages support every claim.
  - 🔵 **Cited · not verified** — real citations, but no judge configured to confirm.
  - 🔴 **Citation doesn't support this** — the judge checked and the cited passage
    does NOT back the answer (the model likely answered from its own memory).
  - 🔴 **Fabricated citation** — the model cited passage numbers it was never shown.
  - 🔴 **No citation** — answered without citing; treat as unverifiable.
  - 🟡 **Not in your sources** — the honest abstention (it won't guess).
  Citations show file and **page number** ("qb.pdf · p. 112") so you can open the
  book and check. Expand *retrieved passages* to see exactly what it read.
- **Library** — add a PDF/text file (embedded once; re-adding is instant). Tick
  **Math-aware extraction** for equation-dense PDFs (renders pages → clean LaTeX
  via the vision model).
- **Convert** — extract a PDF to clean text (fixes ligatures, broken ℏ,
  hyphenation), or vision-transcribe selected pages to Markdown + LaTeX.
- **Measure** — point it at a questions file
  (`[{"question": "…", "answerable": true}, …]`; traps use `false`) and get the
  trust report: bluff rate, coverage, citation rate, judge-graded correctness,
  plus a per-question breakdown.
- **Compare** — two model names, one questions file: both models run the same
  pipeline over the same documents and their trust reports land side by side.
  Quantized / fine-tuned variants (roadmap P2/P3) will appear here as just
  another model name.
- **Settings** — provider presets (OpenRouter / OpenAI / Ollama / LM Studio),
  base URL + key, models (generator / judge / vision / embedder), the retrieval
  pipeline toggles, the **verify** toggle, passages `k`, allow-uncited, theme.

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
