"""LLM-as-judge: grade whether an answer is actually correct and grounded.

The trust report measures whether the model answered, abstained, and cited. It
does NOT by itself measure whether an answer is *factually right*. The Judge adds
that: it shows the source passages, the question, and the model's answer to a
model, and asks for a CORRECT / INCORRECT verdict.

Using a (ideally stronger or separate) model to grade is a standard evaluation
technique. It's not infallible, so we keep it as one signal among several.
"""

from __future__ import annotations

from .interfaces import Generator

_PROMPT = (
    "You are a strict grader. Decide whether the ANSWER correctly answers the "
    "QUESTION and is supported by the CONTEXT. Reply with exactly one word: "
    "CORRECT or INCORRECT.\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION: {question}\n"
    "ANSWER: {answer}\n"
    "VERDICT:"
)


class Judge:
    def __init__(self, generator: Generator) -> None:
        self._gen = generator

    def grade(self, question: str, answer: str, context: str) -> bool:
        """Return True if the answer is judged correct and grounded."""
        verdict = self._gen.generate(
            _PROMPT.format(context=context, question=question, answer=answer)
        ).upper()
        # "INCORRECT" contains "CORRECT", so check the negative first.
        if "INCORRECT" in verdict:
            return False
        return "CORRECT" in verdict
