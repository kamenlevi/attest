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
    "You are a careful assistant. Answer the question using ONLY the passages "
    "below. Cite the passage number(s) you use, like [1]. If the answer is not "
    f"contained in the passages, reply with exactly: {ABSTAIN}\n"
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
