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
