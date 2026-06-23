# Attest

**A local-model workbench for Apple Silicon where every operation comes with a trust signal you can verify.**

Status: 🟢 Phase 1 in progress. The headless engine + trust-measurement runs today (no GPU
needed, via a mock backend); the real MLX backend slots in on Apple Silicon.

## Quickstart (developers)

```bash
pip install numpy pytest        # minimal deps to run the engine + tests
python -m pytest -q             # 9 tests, no model required
python -m attest.demo           # runs the full pipeline on a mock backend, prints a trust report
```

The engine is a plain Python package (`attest/`) with no GUI and no hard dependency on a model —
that's deliberate (see [Architecture](#architecture)).

---

## The problem

Running open models locally on a Mac is now easy. *Trusting* them is not.

- You quantize a model to make it smaller — but you have no idea how much quality you lost.
- You point a small model at your documents — but it confidently makes things up, and you can't tell when.
- You fine-tune on your own data — but you can't prove it actually learned anything.
- Existing tools either ignore this (just run the model and hope) or *claim* trust without letting you check (closed source, unreproducible benchmarks).

The result: people either don't trust local models, or trust them when they shouldn't.

## The thesis

Every useful thing you do to a local model should come with **a number that tells you whether to trust it** — and you should be able to reproduce that number yourself.

| Operation | The trust signal Attest gives you |
|---|---|
| **Grounded Q&A over your documents** (Phase 1) | every answer cites its source passage; the model says *"not in my sources"* instead of bluffing; a measured **bluff rate** |
| **Quantization** (later) | an automatic before/after eval showing exactly how much quality changed |
| **Fine-tuning** (later) | an auto-generated quiz from your source, scored before vs. after, proving it learned |
| **Format conversion** (later) | verification that the converted model gives identical output to the original |

The common thread is a single **verification engine**, built first in Phase 1 and reused by every later feature. That is what makes Attest one coherent tool instead of a pile of scripts — and what makes "trustworthy" a thing you can check, not a thing we claim.

## Why build in the open

The closest existing tool to this vision is closed-source with benchmark claims nobody can reproduce. That's the exact failure mode we're avoiding. Everything here — code, evals, and the numbers they produce — is public and reproducible. If we say "the bluff rate is 2%," you can run it and get 2%.

## Architecture

The **engine** and the **UI** are separate, on purpose:

- **Engine** (`attest/`) — a plain Python package + CLI holding all the real logic (ingest,
  retrieve, ground, measure). It talks to models only through two interfaces (`Embedder`,
  `Generator`), so a **mock backend** lets the whole thing run and be tested with no GPU. On Apple
  Silicon, the real **MLX backend** drops in behind the same interfaces — nothing else changes.
- **App** (Phase 4) — a thin Mac desktop shell over the engine, shipped as a **Homebrew cask** +
  a `.dmg` on GitHub Releases (LM-Studio-style). Because the engine is clean, the UI is just a client.

## Scope

- **Platform:** Apple Silicon (M-series), built on Apple's [MLX](https://github.com/ml-explore/mlx) framework.
- **Models:** small, local, open models (the Phase 1 target is Gemma-class models a teacher would actually run).
- **License:** MIT — open *with* the source, not just the license file.

## Roadmap

See [ROADMAP.md](./ROADMAP.md). We build one phase at a time, and a phase ships only when its trust signal actually works.

Phase 1 — the grounded-knowledge core — is fully specified in [docs/phase-1-design.md](./docs/phase-1-design.md).
