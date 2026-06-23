# Attest

**A local-model workbench for Apple Silicon where every operation comes with a trust signal you can verify.**

Status: 🟡 Designing Phase 1 in public. No code yet — this repo currently holds the design.

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

## Scope

- **Platform:** Apple Silicon (M-series), built on Apple's [MLX](https://github.com/ml-explore/mlx) framework.
- **Models:** small, local, open models (the Phase 1 target is Gemma-class models a teacher would actually run).
- **License:** MIT — open *with* the source, not just the license file.

## Roadmap

See [ROADMAP.md](./ROADMAP.md). We build one phase at a time, and a phase ships only when its trust signal actually works.

Phase 1 — the grounded-knowledge core — is fully specified in [docs/phase-1-design.md](./docs/phase-1-design.md).
