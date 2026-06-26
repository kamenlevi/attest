"""Step 6-8: build the strict 'grounded' prompt, and parse the model's reply.

The prompt forces the model to (a) use only the supplied passages, (b) cite the
passage number(s) it used, and (c) say exactly NOT IN SOURCES when the answer
isn't there. Parsing then tells us whether it answered or abstained, and which
passages it cited — the raw material the trust measurement needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .retrieval import Retrieved

ABSTAIN = "NOT IN SOURCES"

_INSTRUCTIONS = (
    "You answer questions using ONLY the passages below, which are quoted from the "
    "user's documents. They come from real books, so expect formatting noise: "
    "garbled math symbols, split or fused words, equation numbers. Read through the "
    "noise for the meaning.\n"
    "\n"
    "Follow these rules exactly:\n"
    "1. If the passages contain the answer, ANSWER it — even when it is worded "
    "differently from the question, spread across several passages, or must be "
    "assembled or lightly inferred from what they state. Then cite the passage "
    "number(s) you used, like [3].\n"
    f"2. Only if the passages genuinely do not contain the answer, reply with "
    f"exactly: {ABSTAIN}\n"
    "3. Never use outside knowledge to fill a gap. If a claim is not supported by "
    "the passages, it must not appear in your answer. Do not guess.\n"
    "\n"
    "The passages really do contain the answer to most questions, so look carefully "
    "before deciding it is absent.\n"
)


def build_prompt(question: str, retrieved: list[Retrieved]) -> str:
    """Assemble the grounded prompt from the retrieved passages and the question."""
    blocks = [_INSTRUCTIONS]
    for r in retrieved:
        blocks.append(f"PASSAGE [{r.chunk.index}]:\n{r.chunk.text}")
    body = "\n\n".join(blocks)
    return f"{body}\n\nQUESTION: {question}\nANSWER:"


@dataclass(frozen=True)
class GroundedAnswer:
    abstained: bool
    text: str
    citations: list[int]


_CITATION = re.compile(r"\[(\d+)\]")


def parse_response(response: str) -> GroundedAnswer:
    """Turn the model's raw text into a structured result."""
    text = response.strip()
    abstained = ABSTAIN in text.upper()
    citations = [int(n) for n in _CITATION.findall(text)]
    return GroundedAnswer(abstained=abstained, text=text, citations=citations)
