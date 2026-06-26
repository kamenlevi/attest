"""Query-understanding tests: expansion parsing + RRF fusion, no network."""

from attest.chunking import Chunk
from attest.interfaces import Generator
from attest.query import (
    ExpandedQuery,
    ExpandingRetriever,
    QueryExpander,
    _extract_json,
)
from attest.retrieval import Retrieved


class _ScriptedGenerator(Generator):
    """Returns a fixed string — stands in for the LLM expansion call."""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    def generate(self, prompt: str) -> str:
        return self._reply


def test_extract_json_tolerates_surrounding_prose():
    raw = 'Sure! Here is the JSON:\n{"hypothetical": "x", "terms": ["a", "b"]}\nHope it helps.'
    assert _extract_json(raw) == {"hypothetical": "x", "terms": ["a", "b"]}


def test_expand_parses_model_output():
    gen = _ScriptedGenerator('{"hypothetical": "The answer is 0.", "terms": ["momentum", "p"]}')
    eq = QueryExpander(gen).expand("what is <p>?")
    assert eq.hypothetical == "The answer is 0."
    assert eq.terms == ["momentum", "p"]
    # original is always first so expansion can only add recall, never lose it.
    assert eq.search_texts()[0] == "what is <p>?"
    # original + HyDE answer; terms are kept on the object but NOT searched.
    assert eq.search_texts() == ["what is <p>?", "The answer is 0."]


def test_expand_fails_safe_on_garbage():
    eq = QueryExpander(_ScriptedGenerator("I cannot help with that.")).expand("q")
    assert eq == ExpandedQuery("q")
    assert eq.search_texts() == ["q"]  # degrades to plain search


class _FakeIndex:
    """A tiny base retriever: maps each query text to a fixed ranked list."""

    def __init__(self, routes: dict[str, list[int]]) -> None:
        self._routes = routes

    def search(self, query: str, k: int = 4):
        ids = self._routes.get(query, [])[:k]
        return [Retrieved(chunk=Chunk(index=i, text=f"chunk {i}"), score=1.0) for i in ids]


def test_expanding_retriever_fuses_across_expansions():
    # The original question misses chunk 99; the HyDE answer finds it. Fusion
    # must surface 99 even though the literal question never retrieved it.
    gen = _ScriptedGenerator('{"hypothetical": "HYDE", "terms": []}')
    base = _FakeIndex({"Q": [1, 2, 3], "HYDE": [99, 2, 5]})
    retr = ExpandingRetriever(base, QueryExpander(gen), pool=3)
    got = [r.chunk.index for r in retr.search("Q", k=4)]
    assert 99 in got  # recovered purely via the hypothetical answer
    assert 2 in got   # agreement across both lists ranks it well
    assert retr.last_expansion.hypothetical == "HYDE"
