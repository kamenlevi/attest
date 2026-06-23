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


def run_eval(
    items: list[EvalItem], retriever: Retriever, generator: Generator, k: int = 4
) -> list[EvalResult]:
    """Run every question through retrieve -> ground -> generate -> parse."""
    results: list[EvalResult] = []
    for item in items:
        retrieved = retriever.search(item.question, k=k)
        prompt = build_prompt(item.question, retrieved)
        response = generator.generate(prompt)
        results.append(EvalResult(item=item, answer=parse_response(response)))
    return results


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

    return {
        "n_total": len(results),
        "n_answerable": len(answerable),
        "n_traps": len(traps),
        "bluff_rate": (bluffs / len(traps)) if traps else 0.0,
        "answer_coverage": (covered / len(answerable)) if answerable else 0.0,
        "citation_rate": (cited / len(answered)) if answered else 0.0,
    }


def format_report(metrics: dict[str, float | int]) -> str:
    """A small human-readable summary for the CLI."""
    return (
        "Trust report\n"
        f"  questions:        {metrics['n_total']} "
        f"({metrics['n_answerable']} answerable, {metrics['n_traps']} traps)\n"
        f"  bluff rate:       {metrics['bluff_rate']:.0%}  (lower is better; 0% ideal)\n"
        f"  answer coverage:  {metrics['answer_coverage']:.0%}  (higher is better)\n"
        f"  citation rate:    {metrics['citation_rate']:.0%}\n"
    )
