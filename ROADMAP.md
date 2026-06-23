# Roadmap

One rule: **a phase ships only when its trust signal actually works.** We never start the next
phase to escape a hard problem in the current one. Each phase reuses the verification engine built
in Phase 1, so the all-in-one tool is genuinely integrated rather than four separate scripts.

---

## Phase 1 — Grounded knowledge core  ← we are here (designing)

Make a small local model answer questions about a specific document set such that:
- every answer is **grounded** in and **cites** a retrieved source passage,
- the model **abstains** ("not in my sources") when the answer isn't supported,
- we **measure the bluff rate** (how often it fabricates instead of abstaining).

This is the heart of the whole project: the first real user is a physics teacher whose small model
doesn't know what's in his textbooks. It's also where the reusable verification engine is born.

Full spec: [docs/phase-1-design.md](./docs/phase-1-design.md).

**Done when:** on a real physics chapter, bluff rate on unanswerable questions is driven near zero
while keeping reasonable accuracy on answerable ones — and anyone can reproduce that number.

## Phase 2 — Quantization with a quality score

Wrap existing MLX quantization, then **reuse the Phase 1 eval** to report how much capability the
model lost. The differentiator vs. existing tools: the loss is *measured and reproducible*, not claimed.

## Phase 3 — Fine-tuning with proof + conversion glue

- Fine-tune / continued-pretraining on a corpus, with an **auto-generated quiz** scored before vs.
  after to prove the model learned.
- Format conversion (HF / GGUF → MLX) with an **identical-output check** after converting.

## Phase 4 — One Mac GUI

Wrap everything in a single native Mac app once each piece is proven on its own.

---

## What we are deliberately NOT doing

- Not reimplementing quantization math — that's an active specialist's domain; we wrap solid
  existing libraries and add the *verification* they lack.
- Not building structured/grammar decoding — already solved (XGrammar runs on Apple Silicon); we
  use it as a dependency when needed.
- Not shipping a GUI before the engine is proven.
