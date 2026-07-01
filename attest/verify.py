"""The verification layer: turn "the model says it's grounded" into "we checked".

A citation regex tells us the model *claims* passage [3] backs its answer. That
claim can be wrong in two ways, and this module catches both:

  1. INVALID CITATION — the cited number isn't one of the passages the model was
     shown at all. Caught for free, no model call (`check_citations`).

  2. UNSUPPORTED ANSWER — the citation is real, but the passage doesn't actually
     say what the answer says (the model answered from its own memory and
     decorated it with a plausible citation — small models do this). Caught by
     the `SupportChecker`: a second model (the judge) reads ONLY the cited
     passages and the answer, and decides whether every claim is supported.

The result is a `Verification` with one of these statuses, in trust order:

    verified     — citations valid AND the judge confirmed the passages support it
    unverified   — citations valid, but no support check was run (no judge configured)
    unsupported  — citations valid, but the judge says the passages do NOT back it
    invalid      — the model cited passage numbers it was never shown
    uncited      — the model answered without any citation
    abstained    — the model said the answer isn't in the sources

Only "verified" earns the green badge. That's the product promise made real:
answers it can prove — because we proved them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .grounding import GroundedAnswer
from .interfaces import Generator
from .retrieval import Retrieved

_SUPPORT_PROMPT = (
    "You are a strict fact-checker. Below are PASSAGES from a document, a QUESTION, "
    "and an ANSWER that claims to be based only on those passages.\n"
    "Decide: is EVERY factual claim in the ANSWER stated in, or directly inferable "
    "from, the passages? Reworded is fine; invented, imported from outside "
    "knowledge, or contradicted is not.\n"
    "Reply with exactly one word: SUPPORTED or UNSUPPORTED.\n\n"
    "PASSAGES:\n{passages}\n\n"
    "QUESTION: {question}\n"
    "ANSWER: {answer}\n"
    "VERDICT:"
)


@dataclass(frozen=True)
class Verification:
    status: str                                # see module docstring
    valid_citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    note: str = ""                             # one human-readable sentence for the UI


def check_citations(
    answer: GroundedAnswer, retrieved: list[Retrieved]
) -> tuple[list[int], list[int]]:
    """Split the answer's citations into (valid, invalid) against what was shown."""
    shown = {r.chunk.index for r in retrieved}
    valid = [c for c in answer.citations if c in shown]
    invalid = [c for c in answer.citations if c not in shown]
    return valid, invalid


class SupportChecker:
    """Asks a (judge) model whether the cited passages actually support the answer."""

    def __init__(self, generator: Generator) -> None:
        self._gen = generator

    def check(self, question: str, answer_text: str, cited: list[Retrieved]) -> bool:
        passages = "\n\n".join(f"[{r.chunk.index}] {r.chunk.text}" for r in cited)
        verdict = self._gen.generate(
            _SUPPORT_PROMPT.format(passages=passages, question=question,
                                   answer=answer_text)
        ).upper()
        # "UNSUPPORTED" contains "SUPPORTED", so check the negative first.
        if "UNSUPPORTED" in verdict:
            return False
        return "SUPPORTED" in verdict


def verify_answer(
    question: str,
    answer: GroundedAnswer,
    retrieved: list[Retrieved],
    checker: SupportChecker | None = None,
) -> Verification:
    """Run the full verification ladder over one answer.

    The citation validity check always runs (free). The support check runs only
    when a `checker` is provided — it costs one judge-model call.
    """
    if answer.abstained:
        return Verification("abstained", note="The model declared the answer absent "
                            "from your sources — it didn't guess.")
    valid, invalid = check_citations(answer, retrieved)
    if not answer.citations:
        return Verification("uncited", note="Answered without citing any passage — "
                            "treat as unverifiable.")
    if not valid:
        return Verification("invalid", valid_citations=valid, invalid_citations=invalid,
                            note="Cited passage number(s) that were never shown to "
                            "the model — the citation is fabricated.")
    if checker is None:
        return Verification("unverified", valid_citations=valid,
                            invalid_citations=invalid,
                            note="Citations point at real passages, but no judge is "
                            "configured to confirm they support the answer.")
    cited = [r for r in retrieved if r.chunk.index in set(valid)]
    supported = checker.check(question, answer.text, cited)
    if supported:
        return Verification("verified", valid_citations=valid,
                            invalid_citations=invalid,
                            note="An independent judge confirmed the cited passages "
                            "support every claim in the answer.")
    return Verification("unsupported", valid_citations=valid, invalid_citations=invalid,
                        note="The cited passages do NOT support the answer — the "
                        "model likely answered from its own memory.")
