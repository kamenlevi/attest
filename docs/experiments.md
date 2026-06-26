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

---

## Exp 7 — Reranker (cross-encoder) + a better grounding prompt (2026-06)

Branch `query-understanding`. Two changes since Exp 6:
- **Grounding prompt** rewritten to extract when the answer is present (even reworded,
  split, or buried in OCR/math noise) and abstain only on true absence.
- **Reranker**: retrieve a wide pool with the bi-encoder (recall) → re-score each
  candidate with a cross-encoder that reads question+passage together (precision) →
  keep the top-k. Opt-in via `--rerank`.

**Results on the 20-fact QM set (all answerable) + 6 out-of-domain traps:**

| config | RAG correct | abstained | bluff (traps) |
|--------|-------------|-----------|---------------|
| expand only (new prompt) | 16/20 | 3 (P22,P24,P25) | 0/6 |
| expand + rerank          | 16/20 | 2 (P25,P28)     | 0/6 |

**Findings**
1. **The reranker did exactly what it was built to do.** It recovered P22 (the clean
   Schrödinger-equation passage) and P24 (the clean Born-rule passage) — the two cases
   Exp 6 identified as "clean passage exists but ranks too low". Mechanistic win,
   predicted in advance.
2. **Aggregate didn't move (16→16) because the eval is noise-bound.** An 8B generator
   + 8B judge over 20 questions flips individual items run-to-run (P28 OK→abstain,
   P29/P31 OK→"wrong"). Several "wrong"/"abstain" verdicts are suspect: P27 in Exp 6
   was algebraically correct but judged wrong. **We cannot distinguish 16 from 19 from
   20 with this judge.** Measurement quality is now the binding constraint, not retrieval.
3. **Trust held through every change — bluff rate 0/6 throughout.** The "never guesses"
   guarantee is robust to the looser prompt and the reranker.
4. **Speed (measured on the ThinkPad CPU):**
   - bi-encoder search over 971 chunks, k=40: **~66 ms** (and this is the part ANN makes
     scale to millions).
   - cross-encoder rerank of 40 candidates: **~8.4 s** — the expensive part, and it's a
     weak-CPU artifact (a 6-layer transformer over 40 long passages, unbatched-ish, no
     Metal/MLX). On the target Mac GPU this is ~100 ms; mitigations meanwhile: a smaller
     reranker (ms-marco-MiniLM-L-2), a smaller pool, or ONNX.
   - end-to-end ~15 s/question is dominated by **cloud-model round trips** (expand call +
     answer call + judge call), NOT our retrieval.

**Conclusions / next leads**
- **Stronger judge is the top priority** — until grading is trustworthy we're flying
  blind on whether quality work is helping.
- Reranker is a clear quality lever but CPU-costly here; make it light or lean on the Mac.
- Clean ingestion (format conversion to clean text + math-aware extraction) is the fix
  for the truly-garbled cases like P25 — a separate branch.

---

## Exp 8 — A stronger judge: the measurement tool was lying (2026-06)

Branch `query-understanding`. We suspected the 8B judge (same model as the generator)
was misgrading, so we graded the SAME RAG answers with two judges: Llama-3.1-8B (old)
and gpt-4o-mini (new), against the gold answers. Added `--judge-model` to the CLI so
grading can use a separate, stronger model.

**Result (expand+rerank answers, 8B generator):**

| judge | "correct" |
|-------|-----------|
| Llama-3.1-8B | 16/20 |
| gpt-4o-mini  | **12/20** |

The strong judge graded *lower*, not higher — the 8B judge was too **lenient**, not too
harsh. Reading the four disagreements settled who's right:
- **P21**: answer `iLz` dropped the ℏ (gold `iℏLz`) — wrong; 8B judge missed it. ✓ strong
- **P32**: answer `j=3/2, 1/2` (a specific case) when the general `j=l±½` was asked — wrong. ✓ strong
- **P27**: answer copied the garbled PDF math `mωx + ip√ 2mℏω` — ambiguous/wrong as written. ✓ strong
- **P16**: answer `iℏ` (terse but essentially correct) — here gpt-4o-mini was slightly harsh.

**Findings**
1. **We were over-counting.** Honest correctness with the 8B generator is ~12–13/20, not
   16. A small model is an unreliable judge of itself; trustworthy metrics need a
   stronger grader. (This is the project's whole thesis, turned on our own tooling.)
2. **The failures are generation + source, not retrieval.** The right passages arrive;
   the 8B generator then drops terms / over-specializes, or copies garbled equations.
3. **The decisive point: base 8B alone scored 20/20 (Exp 6); RAG over the garbled PDF
   scores ~13.** The mangled source is *dirtier than the model's own memory*, so grounding
   currently COSTS accuracy. RAG can only beat base when the source text is at least as
   clean as parametric memory — which it is not, for a math-dense PDF.

**Conclusions / next leads**
- Use a strong judge (`--judge-model openai/gpt-4o-mini`) for all future measurement.
- **Clean ingestion is now the critical path**, not a nicety: convert sources to clean
  text (and math-aware extraction) so grounding stops corrupting answers. Until the
  source is clean, no retrieval/grounding work can make RAG beat the base model on facts
  the model already knows.

---

## Exp 9 — Clean ingestion (text normalization): necessary, not sufficient (2026-06)

Branch `clean-ingestion`. Added `clean_text` (ligatures ﬁﬂﬀ→fi/fl/ff, `¯h`→`ℏ`,
split accents, line-break hyphenation) applied at ingest, and an `attest convert`
command (PDF → clean .txt). Rebuilt the index from cleaned extraction and re-ran
expand+rerank with the 8B generator and the gpt-4o-mini judge.

**Result (vs the dirty index from Exp 8):**

| | dirty | clean |
|---|-------|-------|
| correct (gpt-4o-mini judge) | 12/20 | 13/20 |
| abstentions | 3 | **1** |
| P22 Schrödinger eq | abstained | **answered, correct** |

The aggregate (+1) is within run-to-run noise, but the qualitative change is real:
cleaning removed the high-volume noise (~1100 `¯h`, ~2500 ligatures), so prose and
simple-formula answers returned and abstentions dropped 3→1.

**Findings**
1. **Cleaning fixes the high-volume noise but not structure.** P27's annihilation
   operator is still `mωx + ip√ 2mℏω` — `ℏ` is fixed but the `/√(…)` fraction is still
   collapsed. Inline-math layout needs a better extractor / math-OCR, not string fixes.
2. **The residual gap is now a MIX, not one bug:** (a) collapsed fractions (P27) →
   math-aware extraction; (b) the 8B generator's own slips (P14 garbled, P25 answered
   the oscillator not the well, P29 mislabeled m as l) → a small-model ceiling; (c)
   judge strictness on terse-but-correct answers (P16 `iℏ`).
3. **Strategic truth:** 20 generic, memorized facts is a test where the *base* model
   has the home advantage (flawless memory) and RAG fights uphill (imperfect source +
   extraction + a small generator). Clean ingestion narrows the gap; closing it fully
   needs math-aware extraction AND bumps into the small-model ceiling. RAG's real edge
   is *book-specific* questions the base model cannot answer — still the eval to build.

**Conclusions / next leads**
- Keep `clean_text` (safe, strictly removes noise) — merge-worthy on its own.
- Math-aware extraction (try PyMuPDF; consider Nougat on the Mac) is the next ingestion
  step for collapsed equations like P27.
- Build the book-specific eval to measure where RAG actually beats base.

---

## Exp 10 — Book-specific questions: RAG decisively beats base (the thesis, proven) (2026-06)

Branch `clean-ingestion`. The mirror of Exp 6: instead of generic facts the model has
memorized, 12 questions whose answers live ONLY in *this* book — its dedication, its
publisher, its equation numbers (2.87, 2.88), its chosen examples (the ammonia maser),
its asides (quaternions), its framings. Base model (no sources) vs RAG (clean index +
expand + rerank), both graded by gpt-4o-mini against gold. See `examples/qb_bookspecific.json`.

**Result**

| | correct |
|---|---|
| Base model alone | **2/12** |
| RAG | **7/12** (6 correct+cited, 4 honest abstentions, 0 bluffs) |

The two base "wins" (B4 parity violation, B10 Landau levels) were questions generic
enough to be in the model's training. Restricting to the **10 genuinely book-specific
questions**:

| | correct | wrong | honest abstain |
|---|---------|-------|----------------|
| Base | **0/10** | 10 | 0 |
| RAG  | **6/10** | 1 | 3 |

**Findings**
1. **RAG knows what base cannot.** Base scored 0/10 on truly book-specific facts; RAG
   answered 6 correctly, each with a citation to the exact passage (B1→dedication [0],
   B9→quaternions [73], B8→ammonia maser [244], B12→virial [117], B6→eq 2.88 [116],
   B2→authors/publisher). This is the product's value, measured.
2. **RAG never bluffs.** Its 4 misses were honest "NOT IN SOURCES" abstentions (B3/B7/B11
   retrieval misses) — the trust guarantee holding. Base, by contrast, confidently made
   things up on all 10.
3. **The two evals together fully characterize RAG's value.** Exp 6: on memorized generic
   facts, base wins (20 vs 13) — RAG can't beat the model's own clean memory, and a garbled
   source even hurts. Exp 10: on document-specific facts, RAG wins decisively (base 0/10) —
   and always with citations, never guessing. So: **RAG is not for what the model already
   knows; it is for what is specific to YOUR documents — answered with a citation and
   without ever lying.** That is exactly the physics-teacher use case.
4. **RAG's ceiling here is higher than 7/12.** The 4 abstentions are retrieval misses
   (the passage exists but didn't reach the top-8); better k / retrieval would convert
   several to correct answers. The one true error (B5) had the right chunk [116] but the
   8B mis-extracted the still-imperfect J equation.

**Conclusions / next leads**
- The core thesis is demonstrated and reproducible. Headline framing for the project:
  "make a small local model answer questions about YOUR documents it otherwise can't —
  with citations, and without making things up."
- Remaining upside: retrieval recall on the abstained cases; math-aware extraction for
  equation-valued answers (B5); a bigger eval to tighten the numbers.

---

## Exp 11 — Hybrid (semantic + BM25 lexical) retrieval: recall solved (2026-06)

Branch `clean-ingestion`. The Exp 10 abstentions (B3/B7/B11) were semantic search
missing *exact* terms — proper nouns ("Walter of Merton"), labels ("Box 2.1"),
section numbers. Added `attest/lexical.py` BM25Retriever + `FusedRetriever` (RRF over
semantic + lexical), a `--lexical` flag, and a tokenizer that keeps dotted labels
("2.1", "2.87") as single tokens. Best-recall stack: **hybrid → HyDE expand → rerank**.

**Deterministic recall probe (gold passage in top-8?):**

| | semantic+expand+rerank | hybrid+expand+rerank |
|---|---|---|
| B7 "Box 2.1" (chunk 75) | NO | **YES** |
| B5 eq 2.87 (chunk 116)  | YES | YES |
| B11 "merely" (chunk 7)  | YES | YES |
| B1 "Walter of Merton" (chunk 0) | semantic alone: NO | BM25 alone: **YES** |

**End-to-end on the 12 book-specific questions (8B gen, gpt-4o-mini judge):**

| | base | RAG (Exp 10) | RAG BEST (hybrid) |
|---|------|--------------|-------------------|
| correct | 2/12 | 7/12 | **10/12** |
| abstentions | — | 4 | 1 |
| bluffs | (bluffs all) | 0 | 0 |

**Findings**
1. **Lexical fusion is the recall fix.** B3/B4/B7/B11 converted from abstention to
   correct+cited. BM25 catches the exact tokens embeddings miss (it alone found the
   "Walter of Merton" dedication that semantic ranked nowhere); keeping "2.1"/"2.87"
   as tokens rescued the label questions; HyDE supplies content words the question
   omits ("Hermitian" for Box 2.1).
2. **11 of 12 gold passages are now reachable in the top-8.** The lone real failure is
   B5: the chunk IS retrieved, but the 8B mis-states the still-garbled J equation —
   a generation/math-extraction problem, NOT recall.
3. **Still zero bluffs.** The one miss (B8) was an honest abstention; the trust
   guarantee holds while recall climbs.

**Conclusions / next leads**
- Recall is essentially solved for findable content: hybrid (semantic+BM25) + HyDE +
  rerank. RAG now answers 10/12 document-specific questions the base model gets 2/12 on.
- The residual is the math-aware-extraction problem (B5) — equations need a better
  extractor before grounding can quote them exactly.
- B8's single-run abstention is noise; a slightly larger k/pool or a bigger eval would
  smooth it.
