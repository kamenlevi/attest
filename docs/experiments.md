# Experiments log

Reproducible results from running Attest. The point of the project is that these
numbers can be re-derived by anyone, so we record them as we go.

---

## Exp 1 — First real-model run on a full book (2026-06)

**Setup**
- Document: Einstein, *Relativity: The Special and General Theory* (Project
  Gutenberg #30155), 34k words → **214 chunks** (chunk_size=200, overlap=40).
- Model (generator): `meta-llama/llama-3.1-8b-instruct` via OpenRouter.
- Eval: 10 questions — 5 answerable, 5 traps. Labels **verified against the
  source by grep** before running (e.g. "quantum"/"entanglement" = 0 hits → safe
  traps; "Lorentz" = 47 hits → answerable). See `examples/relativity_eval.json`.

**Results**

| Retriever          | k  | bluff rate | coverage | citation rate |
|--------------------|----|-----------|----------|---------------|
| Lexical (mock)     | 4  | 0%        | 80%      | 75%           |
| Semantic (MiniLM)  | 4  | 0%        | 60%      | 100%          |
| Semantic (MiniLM)  | 10 | 0%        | 80%      | 100%          |
| Lexical (mock)     | 10 | 0%        | 80%      | 75%           |

**Findings**
1. **Bluff rate held at 0% everywhere.** With the strict grounding prompt, the
   8B model abstained on every trap — including the near-miss numeric one ("exact
   speed of light in m/s") — and even suppressed facts it surely knows from
   training when they weren't in the source.
2. **`k` matters.** Semantic at k=4 looked worse (60%) purely because the right
   chunk ranked below the top 4; raising k to 10 recovered it to 80%. A tuning
   knob, not a real regression — caught in seconds by re-measuring.
3. **Semantic's real advantage is grounding, not coverage.** At equal k both
   answer the same questions, but semantic reaches **100% citation vs 75%**: it
   retrieves the actual source passage, so the model grounds its answer instead
   of relying on parametric memory (notably the Mercury-perihelion answer, which
   was *uncited* under lexical retrieval).
4. **One open hard case:** "What is the principle of equivalence?" abstains under
   both retrievers even at k=10. "Equivalence" appears only twice and Einstein
   phrases the idea differently → a retrieval/phrasing-mismatch case.

**Conclusions / next leads**
- The "lexical vs semantic" question is not either/or — this is direct evidence
  for **hybrid retrieval** (combine keyword + semantic) plus **reranking**, which
  is the planned next improvement. The project's "measure both, let evidence
  decide" thesis held up on real data.
- Investigate the equivalence-principle miss (chunking? phrasing? needs hybrid?).
- Add genuine answer-*correctness* grading (currently we measure abstain vs
  answer + citations, not whether an answer is factually right). [DONE — Exp 2]

---

## Exp 2 — Correctness grading + hybrid retrieval (2026-06)

Same book/model/eval as Exp 1. Added an LLM-as-judge correctness grade and a
hybrid (lexical + semantic, RRF-fused) retriever.

**Results**

| Retriever        | k  | coverage | citation | correctness        |
|------------------|----|----------|----------|--------------------|
| Semantic         | 10 | 80%      | 100%     | 100% (of 4 graded) |
| Hybrid (run A)   | 10 | 60%      | 100%     | 100% (of 3 graded) |
| Hybrid (run B)   | 10 | 80%      | 100%     | —                  |
| Hybrid           | 20 | 60%      | 100%     | —                  |
| Hybrid           | 30 | 60%      | 100%     | —                  |

**Findings**
1. **Correctness grading works:** every answered-and-graded question was judged
   correct, so the answers aren't just present — they're right.
2. **Measurement noise is real.** Identical config (hybrid, k=10) gave 60% and
   80% on two runs: the model is not deterministic even at temperature 0
   (provider-side routing/batching). On a 10-question eval, ±20% is noise.
3. **Naive hybrid did NOT beat semantic here.** RRF rewards *agreement*, so a
   chunk both retrievers rank mid-list can bury a chunk one retriever ranks #1
   (e.g. lexical's strong "Lorentz" hit). Bigger k didn't help.

**Conclusions / next leads**
- **Do not draw retrieval conclusions from a 10-question eval.** The top priority
  is a *larger* eval set (30-50+ questions) and averaging over multiple runs,
  so differences exceed the noise floor. Trust metrics need enough samples to be
  trustworthy themselves — fitting, for this project.
- Semantic remains the safe default (best citations, no fusion pitfalls).
- Hybrid is implemented and correct; revisit with a bigger eval and a tuned
  fusion constant `c` / per-retriever top-k guarantees.

---

## Exp 3 — Real PDF ingestion (2026-06)

**Setup**
- Document: a real, math-dense quantum-mechanics / circuit-QED lecture-notes PDF
  from arXiv (1 MB, 238 chunks after extraction).
- Pipeline: pypdf ingestion → semantic retrieval (k=10) → grounded answer →
  judge. Generator/judge: Llama 3.1 8B via OpenRouter.
- Eval: 6 questions (4 conceptual answerable, 2 traps), labels grep-verified
  against the extracted text. See `examples/qm_pdf_eval.json`.

**Extraction quality (the key finding)**
- **Prose extracts cleanly.** Table of contents, explanations, definitions are
  all readable.
- **Math gets mangled.** Layout-dependent notation breaks: `ẍ` → `¨x`; fractions
  split across lines (`k/m` → `k` / `mx`); square roots detach (`√` on its own
  line); occasional fused/split words (`xand`, `oscillatio ns`).
- This confirms the risk flagged in `phase-1-design.md` §3.1 before any code.

**Eval result**

| bluff | coverage | citation | correctness        |
|-------|----------|----------|--------------------|
| 0%    | 100%     | 100%     | 100% (of 4 graded) |

**Findings**
1. The full pipeline works end-to-end on a real PDF: conceptual questions were
   answered, cited, and judged correct; both traps abstained.
2. **Conceptual Q&A survives bad math extraction** — answers about the harmonic
   oscillator, qubits, the Hamiltonian, and cavities were all correct despite the
   garbled equations, because the surrounding prose carries the meaning.
3. Ingestion **fails loudly** on non-PDFs (an HTML error page raised a clear
   PdfStreamError rather than silently returning garbage).

**Conclusions / next leads**
- For physics specifically: RAG-over-PDF is viable for *conceptual* learning now;
  *equation-exact* questions need math-aware extraction (e.g. an OCR/LaTeX-aware
  ingester) — a real, scoped future sub-problem, not a blocker.
- The text path (`.txt`) remains the cleanest; offer it when source text exists.

---

> Exp 4 (hierarchical chapter-routed retrieval — **rejected**) lives on the
> `hierarchical-retrieval` branch; it isn't merged, so it isn't repeated here.

---

## Exp 5 — Query understanding (HyDE) for recall — measured on the QM problem bank (2026-06)

Branch `query-understanding`. Goal: fix the questions that abstained in our QM
problem-bank test (P9 commutator `[x,T]`, P11 uncertainty relation, P12 probability
current). Index: Binney & Skinner, *The Physics of Quantum Mechanics*, the real PDF
→ **970 chunks**, semantic (MiniLM, 384-d). Generator/expander: the configured
OpenRouter model. 6 questions, all answerable.

**Step 0 — is it a ranking or a recall problem? (k-sweep, k=8 vs k=40)**
Going from k=8→k=40 *recovered* some abstentions but **broke others** (P8 answered
at k=8, abstained at k=40; P10 lost its citation). So naively widening k is a wash:
more passages dilute the prompt ("lost in the middle"). The lesson: *retrieve wide,
then trim precisely* — not "use a big k".

**The decisive measurement — rank of the true gold passage (deterministic, no
generation noise).** For each question we embedded the plain question, a HyDE
"hypothetical answer", and a bag-of-terms list, and recorded where the verified
gold chunk lands (0 = top hit; <8 = fed to the model):

| question        | plain Q | **HyDE** | terms | naive-fuse(all 3) |
|-----------------|---------|----------|-------|-------------------|
| P9  `[x,T]`     | 7       | **0**    | 5     | 3                 |
| P10 Heisenberg  | 0       | **0**    | 0     | 0                 |
| P12 current `j` | 3       | **0**    | 21    | 11                |

**Findings**
1. **HyDE is a bullseye.** Writing a fake textbook-style answer and embedding *that*
   ranks the true passage **#1 for every question**. This is the query-understanding
   win: a plausible answer lives in "document space", close to the real one.
2. **A synonym/term list is noise, and naive RRF fusion HURTS.** P12: gold was rank
   3 on the plain question and rank 0 on HyDE, but fusing in the junk terms list
   (gold at rank 21) dragged the *fused* result to **11 — worse than doing nothing**.
   Fix: drop terms from semantic search; fuse only `[question, HyDE]` (question kept
   as a safety net against a bad HyDE). `terms` is still stored for a future keyword
   retriever.
3. **Reframe — at single-book scale, retrieval was not the main bottleneck.** With
   the corrected expander the gold chunk is in the top-8 for P9/P10/P12 — but so it
   often was for the *plain* question (rank 7/0/3, all <8). The remaining abstentions
   are the **generator** refusing even with the right passage present (strict
   grounding prompt + garbled math from PDF extraction, cf. Exp 3), not a missing
   chunk. Query understanding's payoff *grows with corpus size*: at 100 books the
   plain query won't put gold in the top-8, but HyDE still ranks it #1.
4. **A 6-question eval can't measure answer-rate.** End-to-end coverage swung
   50%→33%→50% across identical-config runs at temperature 0 (provider nondeterminism
   + so few items). This re-confirms Exp 2: trust metrics need a bigger sample.

**Conclusions / next leads**
- Ship query understanding (`--expand`): proven to improve *ranking* (gold → #1) with
  no downside (the literal question is always retained). It's the right investment for
  the multi-book scale the project targets.
- The single-book answer-quality bottleneck is now **generation + math extraction**,
  not retrieval: math-aware ingestion, the abstain/`--allow-uncited` setting, and a
  larger eval set are the levers there.
- A reranker (retrieve wide → re-score → keep top few) remains the natural precision
  step, especially once the corpus is large enough that gold no longer fits in a small k.

---

## Exp 6 — Base model vs RAG on generic textbook facts — RAG LOSES (and that's the lesson) (2026-06)

Branch `query-understanding`. 20 standard QM questions (P13–P32: `[x,p]=iℏ`, `S²=¾ℏ²`,
infinite-well energies, the TDSE, the Born rule, …), each graded against a gold answer
by an LLM judge. We compared the **base model alone** (no sources) against **RAG +
`--expand`** (k=8). Index: the Binney–Skinner QM PDF. See `examples/qb_eval2.json`.

**Result**

| | correct |
|---|---|
| Base model alone | **20 / 20** |
| RAG + `--expand` | **14 / 20** (abstained on P17, P22, P24, P25, P26, P28) |

**The content was not missing.** All six abstained topics are demonstrably in the
book (e.g. "square well" 13 chunks, "harmonic oscillator" 58, "stationary state" 63,
"probability density" 16, "spin" 160). RAG abstained ("NOT IN SOURCES") on content
that was present — a strict-prompt + retrieval + garbled-math failure, not a gap.

**Findings**
1. **For facts the model already knows, RAG can only tie or lose.** These are famous,
   generic results any competent model has from training (base: 20/20). RAG adds
   nothing on the wins and, via strict grounding, *subtracts* on the misses: when the
   exact passage doesn't surface cleanly in the top-k, abstention turns a known-correct
   answer into a non-answer. Net: −6.
2. **We were measuring RAG on the one question class where it cannot help.** This is a
   measurement-design lesson, and exactly the kind of thing Attest exists to catch with
   numbers rather than vibes. It's consistent with the RAG-vs-parametric literature:
   RAG is for knowledge the model *lacks*, not knowledge it has.
3. **Where RAG should win (untested here):** questions whose answers are *specific to
   this book* and not generic knowledge — a worked example the text uses, this book's
   notation, the content of a particular Box or equation. There the base model should
   bluff or fail and grounded RAG should win + cite. That is the physics-teacher case
   ("the model doesn't know what's in *this* textbook") and the right next eval.
4. **The 6 abstentions-on-present-content are a real, separable bug.** Levers: a less
   trigger-happy abstention setting (answer from sources if present, else say so), a
   reranker to tighten top-k precision, and math-aware extraction (cf. Exp 3).

**Conclusions / next leads**
- Build a *book-specific* eval (answers not in the model's parametric memory) to
  measure RAG's actual value, plus a trap set to measure bluff-rate (the trust half).
- Treat "abstained despite the passage being present" as the headline bug to fix next.
