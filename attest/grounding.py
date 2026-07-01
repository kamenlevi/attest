"""Step 6-8: build the strict 'grounded' prompt, and parse the model's reply.

The prompt forces the model to (a) use only the supplied passages, (b) cite the
passage number(s) it used, and (c) declare NOT IN SOURCES when the answer isn't
there. The model is asked to reply as a small JSON object — an exact protocol we
can parse without guessing. Because small models sometimes break JSON, parsing
falls back to the legacy plain-text protocol (the ABSTAIN phrase + [n] citation
regex), so a malformed reply degrades gracefully instead of erroring.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

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
    "assembled or lightly inferred from what they state. Cite the passage "
    "number(s) you actually used.\n"
    f"2. Only if the passages genuinely do not contain the answer, abstain: set "
    f'"found" to false and "answer" to "{ABSTAIN}".\n'
    "3. Never use outside knowledge to fill a gap. If a claim is not supported by "
    "the passages, it must not appear in your answer. Do not guess.\n"
    "\n"
    "The passages really do contain the answer to most questions, so look carefully "
    "before deciding it is absent.\n"
    "\n"
    "Reply with ONLY a JSON object, nothing else, in exactly this shape:\n"
    '{"found": true, "answer": "your answer here", "citations": [3, 7]}\n'
    '"citations" lists the passage numbers your answer is drawn from (empty when '
    "abstaining).\n"
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
    structured: bool = field(default=False, compare=False)  # parsed via the JSON protocol?


_CITATION = re.compile(r"\[(\d+)\]")
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(response: str) -> GroundedAnswer:
    """Turn the model's raw text into a structured result.

    Tries the JSON protocol first (exact), then falls back to the legacy text
    protocol (ABSTAIN phrase + [n] regex) for models that ignore the format.
    """
    text = response.strip()
    parsed = _parse_json(text)
    if parsed is not None:
        return parsed
    abstained = ABSTAIN in text.upper()
    citations = [int(n) for n in _CITATION.findall(text)]
    return GroundedAnswer(abstained=abstained, text=text, citations=citations)


def _parse_json(text: str) -> GroundedAnswer | None:
    """Parse the JSON reply shape, tolerating code fences and surrounding prose."""
    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "answer" not in data:
        return None
    answer = str(data.get("answer") or "").strip()
    found = bool(data.get("found", True)) and ABSTAIN not in answer.upper()
    raw_cites = data.get("citations") or []
    citations = [int(c) for c in raw_cites
                 if isinstance(c, (int, str)) and str(c).lstrip("-").isdigit()]
    if not found:
        citations = []
    return GroundedAnswer(abstained=not found, text=answer or ABSTAIN,
                          citations=citations, structured=True)
