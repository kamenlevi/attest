# Phase 1 design — the grounded-knowledge core

This is the "on paper" design. No code yet. The goal of this document is that you (and anyone
else) understand *exactly* what we're going to build and why, before a line is written.

---

## 1. What we're building, in one sentence

A command-line tool where you give a small local model a set of documents (e.g. a physics
textbook chapter), ask it questions, and it either **answers with a citation to the exact source
passage** or **says "that's not in my sources"** — and we can **measure how often it cheats.**

## 2. Why retrieval (RAG), not fine-tuning, for this phase

Two ways to make a model "know" a document:

- **Fine-tuning** bakes the text into the model's weights. Problem: it's opaque (you can't see
  *where* an answer came from), it's easy to do badly, and it can't cite a source. Good for teaching
  *style*, bad for *facts you must trust*.
- **Retrieval (RAG)** keeps the documents outside the model. At question time we find the relevant
  passages and hand them to the model as context. It's inherently **citable** (we know which passage
  we gave it), updatable, and far more reliable for factual recall on small models.

For a teacher who needs to *trust and learn from* answers, citability is everything. So Phase 1 is
retrieval-grounded. (Fine-tuning returns later, in Phase 3, with its own proof mechanism.)

## 3. How it works — the pipeline

Each step below is a thing we'll build and that you'll learn. Plain-language explanation included.

1. **Ingest** — read the PDF, pull out its text.
   - *Concept:* PDFs store text awkwardly; we extract it into plain text.
   - *Known risk (important):* physics textbooks are full of **equations and notation** that extract
     badly from PDFs. This is the single biggest technical risk in Phase 1, and we test it early
     (see §6). If math extraction is too poor, we adjust (e.g. better extractor, or start with a
     text-heavy chapter).

2. **Chunk** — split the text into small passages (say a few hundred words each, with a little
   overlap so we don't cut a sentence in half).
   - *Concept:* we retrieve *passages*, not whole books. Chunk size is a knob: too big = noisy
     context; too small = lost meaning. We'll make it adjustable and measure.

3. **Embed** — turn each chunk into a list of numbers (a "vector") that captures its meaning.
   - *Concept:* an **embedding** maps text to a point in space so that passages about the same idea
     sit close together. This is how we search by *meaning*, not keywords.
   - *Choice:* a small embedding model that runs in MLX on Apple Silicon (from the `mlx-community`
     models). Embedding is cheap and local.

4. **Store + search** — keep the vectors so we can, given a question, find the closest chunks.
   - *Concept:* "closest" = highest cosine similarity. For Phase 1 the corpus is small (one chapter),
     so a plain NumPy similarity search is enough — no heavy vector database yet. Simplicity first.

5. **Retrieve** — embed the user's question, return the top-k most similar chunks (start k=4).

6. **Generate, grounded** — give the small model a strict prompt: *"Answer using ONLY the passages
   below. Cite the passage number(s) you used. If the answer isn't in them, reply exactly: NOT IN
   SOURCES."* plus the retrieved chunks and the question.
   - *Choice of model:* **Gemma-class small model via `mlx-lm`** — the same kind the teacher
     actually uses. The whole point is to make a *small* model trustworthy, not to cheat with a
     huge one.

7. **Verify the citation** — check that the passage the model cited actually exists and was one we
   retrieved. A model that cites a passage we never gave it is caught here.

8. **Abstention** — detect the `NOT IN SOURCES` response and surface it honestly instead of an answer.

## 4. The measurement — the part that matters most

Anyone can build steps 1–8. The reason Attest exists is **step 9: measuring whether to trust it.**

We build a small **evaluation set** for a chosen chapter:
- **Answerable questions** (~15): the answer is genuinely in the chapter.
- **Trap questions** (~10): plausible physics questions whose answer is **not** in this chapter.

Then we compute three numbers:

| Metric | What it asks | Why it matters |
|---|---|---|
| **Answer accuracy** | On answerable questions, is the answer correct and grounded? | Is it actually useful? |
| **Bluff rate** | On trap questions, how often does it fabricate instead of saying NOT IN SOURCES? | **The core trust number.** A model that bluffs can't be learned from. |
| **Citation validity** | When it answers, does the cited passage really support the claim? | Catches "right answer, wrong/fake source." |

**Phase 1 is "done" when:** bluff rate is driven near zero on the trap questions while answer
accuracy stays reasonable on the answerable ones — and the numbers are reproducible by anyone who
clones the repo.

## 5. The stack (kept deliberately small)

- **Language:** Python.
- **Generation:** `mlx-lm` (runs the small model on Apple Silicon).
- **Embeddings:** a small MLX-compatible embedding model.
- **Search:** NumPy (corpus is small in Phase 1).
- **Interface:** command line. No GUI yet — we earn that in Phase 4.
- **Runs on:** Apple Silicon only (MLX requirement). Code is written on Linux; MLX steps run on the Mac.

## 6. First experiment (the very first thing we'll actually run)

Smallest possible test that proves the core idea, before building the full pipeline:

> One small model + one physics chapter PDF + ~25 questions (15 answerable, 10 traps).
> Force citations and the NOT IN SOURCES rule. **Measure the bluff rate.**

Two early checks decide everything:
1. **Does the PDF's math survive extraction** well enough to answer questions? (the big risk, §3.1)
2. **Does the small model obey the grounding + abstention rules**, or does it bluff anyway?

If both look workable, we build out the pipeline. If not, we learn it cheaply and adjust the plan —
which is the whole point of measuring instead of vibing.

## 7. Open questions to resolve before coding

- Which exact chapter/PDF do we test on? (Ideally one the teacher actually uses.)
- Which small model + which embedding model, by name? (Pick during scoping on the Mac.)
- How do we grade "answer accuracy" — by hand for 25 questions (fine for v1), or with a checker?

## 8. What success teaches you

By the end of Phase 1 you'll understand, hands-on: embeddings, retrieval, prompting for grounding,
and — most importantly — how to *evaluate* an AI system instead of trusting how it feels. That
evaluation skill is the reusable core of every later phase.
