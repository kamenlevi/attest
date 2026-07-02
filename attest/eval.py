"""Step 9 — the part that matters most: measure whether to trust the system.

We run an evaluation set of questions through the pipeline and compute the trust
numbers. The headline one is the **bluff rate**: on questions whose answer is
NOT in the documents, how often does the model make something up instead of
abstaining? A tool you can learn from must keep that near zero.

Note: this module is deliberately model-agnostic. It scores *parsed responses*,
so it works identically whether the answers came from RAG, fine-tuning, or the
mock backend — which is exactly why one engine can later compare all of them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .grounding import GroundedAnswer, build_prompt, parse_response
from .interfaces import Generator
from .judge import Judge
from .retrieval import Retriever


@dataclass(frozen=True)
class EvalItem:
    """One test question.

    `answerable` is the ground truth: True if the answer really is in the
    documents, False if it's a 'trap' whose answer is absent.
    """

    question: str
    answerable: bool


@dataclass(frozen=True)
class EvalResult:
    item: EvalItem
    answer: GroundedAnswer
    context: str = ""  # the retrieved passages shown to the model (for grading)
    correct: bool | None = None  # set by grade_results(); None = not graded


def run_eval(
    items: list[EvalItem], retriever: Retriever, generator: Generator, k: int = 4,
    progress=None,
) -> list[EvalResult]:
    """Run every question through retrieve -> ground -> generate -> parse.

    `progress(done, total)` is an optional callback for long runs (the app's
    job system uses it to show "question 12/30").
    """
    results: list[EvalResult] = []
    for i, item in enumerate(items, 1):
        if progress:
            progress(i, len(items))
        retrieved = retriever.search(item.question, k=k)
        prompt = build_prompt(item.question, retrieved)
        response = generator.generate(prompt)
        context = "\n\n".join(r.chunk.text for r in retrieved)
        results.append(
            EvalResult(item=item, answer=parse_response(response), context=context)
        )
    return results


def grade_results(results: list[EvalResult], judge: Judge, progress=None) -> list[EvalResult]:
    """Add a correctness verdict to each *answered answerable* result.

    We only grade answerable questions that were actually answered: correctness
    of a trap is already captured by the bluff rate (correct = abstain), and an
    abstention has no answer to grade.
    """
    graded: list[EvalResult] = []
    for i, r in enumerate(results, 1):
        if progress:
            progress(i, len(results))
        if r.item.answerable and not r.answer.abstained:
            verdict = judge.grade(r.item.question, r.answer.text, r.context)
            graded.append(EvalResult(r.item, r.answer, r.context, correct=verdict))
        else:
            graded.append(r)
    return graded


def compute_metrics(results: list[EvalResult]) -> dict[str, float | int]:
    """Turn raw results into the trust numbers.

    - bluff_rate: of the trap questions, the fraction the model answered instead
      of abstaining. THE core trust signal. Lower is better; 0.0 is ideal.
    - answer_coverage: of the answerable questions, the fraction it actually
      answered (didn't wrongly abstain). Higher is better.
    - citation_rate: of all answered questions, the fraction that included at
      least one [n] citation.
    (Genuine answer *correctness* needs a grader; that's a later addition. These
    three are computable with no human judge and already expose bluffing.)
    """
    traps = [r for r in results if not r.item.answerable]
    answerable = [r for r in results if r.item.answerable]
    answered = [r for r in results if not r.answer.abstained]

    bluffs = sum(1 for r in traps if not r.answer.abstained)
    covered = sum(1 for r in answerable if not r.answer.abstained)
    cited = sum(1 for r in answered if r.answer.citations)
    graded = [r for r in results if r.correct is not None]

    metrics: dict[str, float | int] = {
        "n_total": len(results),
        "n_answerable": len(answerable),
        "n_traps": len(traps),
        "bluff_rate": (bluffs / len(traps)) if traps else 0.0,
        "answer_coverage": (covered / len(answerable)) if answerable else 0.0,
        "citation_rate": (cited / len(answered)) if answered else 0.0,
    }
    if graded:  # only present when answers were graded by a judge
        metrics["n_graded"] = len(graded)
        metrics["correctness_rate"] = sum(1 for r in graded if r.correct) / len(graded)
    return metrics


def format_report(metrics: dict[str, float | int]) -> str:
    """A small human-readable summary for the CLI."""
    return (
        "Trust report\n"
        f"  questions:        {metrics['n_total']} "
        f"({metrics['n_answerable']} answerable, {metrics['n_traps']} traps)\n"
        f"  bluff rate:       {metrics['bluff_rate']:.0%}  (lower is better; 0% ideal)\n"
        f"  answer coverage:  {metrics['answer_coverage']:.0%}  (higher is better)\n"
        f"  citation rate:    {metrics['citation_rate']:.0%}\n"
    ) + (
        f"  correctness:      {metrics['correctness_rate']:.0%}  "
        f"(of {metrics['n_graded']} graded answers)\n"
        if "correctness_rate" in metrics
        else ""
    )
