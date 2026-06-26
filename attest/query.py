"""Query understanding: turn a short user question into a rich search intent.

A one-line question ("what is the uncertainty relation?") often embeds *far* from
the passage that actually answers it, because the source states it with different
words and notation (the derivation, the symbols ΔA ΔB, a commutator). We saw this
miss real questions in Exp 1 and Exp 5.

So before searching, we spend ONE cheap LLM call to expand the question into:

  1. a HYPOTHETICAL ANSWER (the "HyDE" trick) — a short passage written the way
     the *source* would write it. Embedding a plausible answer lands much closer
     to the real answer than embedding the bare question does.
  2. KEY TERMS / NOTATION — synonyms and symbols likely to appear in the text.

We then search the index with the original question *and* these expansions, and
fuse the result lists (RRF). The point is recall: surface the right passage even
when the question and the source don't share wording. It costs one extra model
call per query — the user opts in with --expand.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .interfaces import Generator
from .retrieval import Retrieved, reciprocal_rank_fusion

_PROMPT = """You help search a document/textbook corpus. A user asked the question \
below. Produce JSON that will improve retrieval of the passage that answers it.

Question: {question}

Return ONLY a JSON object with exactly these keys:
{{
  "hypothetical": "A short factual passage (2-4 sentences) that would plausibly \
appear in the source and directly answer the question. Write it the way a \
textbook would: use the standard technical terms, notation, and symbols. Do not \
hedge; state the answer as fact.",
  "terms": ["key technical term, synonym, or notation likely to appear in the \
source", "..."]
}}
Give 4-8 terms. Output the JSON and nothing else."""


def _extract_json(text: str) -> dict:
    """Pull the first {...} object out of a model response (which may add prose)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


@dataclass(frozen=True)
class ExpandedQuery:
    original: str
    hypothetical: str = ""
    terms: list[str] = field(default_factory=list)

    def search_texts(self) -> list[str]:
        """The distinct texts to embed and search with.

        Just the original question + the HyDE hypothetical answer. We measured
        (Exp 5) that the hypothetical answer ranks the true passage #1, while a
        bag-of-synonyms "terms" list ranks it poorly and, fused in via RRF,
        actively drags good results down. So `terms` is kept on the object (handy
        for display or a future keyword retriever) but deliberately NOT searched
        semantically. The original stays as a safety net if HyDE goes astray.
        """
        texts = [self.original]
        if self.hypothetical.strip():
            texts.append(self.hypothetical.strip())
        return texts


class QueryExpander:
    """Expands a question into an ExpandedQuery via one LLM call.

    Fails safe: if the model errors or returns unparseable output, we fall back
    to an expansion containing just the original question, so retrieval degrades
    to ordinary search rather than breaking.
    """

    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    def expand(self, question: str) -> ExpandedQuery:
        try:
            raw = self._generator.generate(_PROMPT.format(question=question))
        except Exception:  # noqa: BLE001 - expansion is best-effort, never fatal
            return ExpandedQuery(question)
        data = _extract_json(raw)
        hypothetical = data.get("hypothetical", "")
        terms = data.get("terms", [])
        if not isinstance(hypothetical, str):
            hypothetical = ""
        if not isinstance(terms, list):
            terms = []
        terms = [str(t) for t in terms if str(t).strip()]
        return ExpandedQuery(question, hypothetical, terms)


class ExpandingRetriever:
    """Wraps any base retriever/index (anything with .search(text, k)).

    On search(): expand the query, retrieve a pool of candidates with EACH
    expansion text, then RRF-fuse the ranked lists into one. Duck-typed like
    Retriever (has search()), so the CLI and eval use it interchangeably.
    """

    def __init__(self, base, expander: QueryExpander, pool: int = 30) -> None:
        self._base = base
        self._expander = expander
        self._pool = pool  # candidates per expansion text before fusing
        self.last_expansion: ExpandedQuery | None = None  # for --verbose / debugging

    def search(self, query: str, k: int = 4) -> list[Retrieved]:
        expanded = self._expander.expand(query)
        self.last_expansion = expanded
        ranked_lists: list[list[int]] = []
        by_index: dict[int, Retrieved] = {}
        for text in expanded.search_texts():
            hits = self._base.search(text, k=self._pool)
            ranked_lists.append([h.chunk.index for h in hits])
            for h in hits:
                by_index.setdefault(h.chunk.index, h)  # keep first-seen Retrieved
        fused = reciprocal_rank_fusion(ranked_lists)[:k]
        return [by_index[idx] for idx in fused]
