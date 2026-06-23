from attest.chunking import Chunk
from attest.grounding import ABSTAIN, build_prompt, parse_response
from attest.retrieval import Retrieved


def test_prompt_includes_passages_and_question():
    retrieved = [Retrieved(Chunk(0, "alpha text"), 0.9), Retrieved(Chunk(1, "beta text"), 0.5)]
    prompt = build_prompt("what is alpha?", retrieved)
    assert "PASSAGE [0]:" in prompt
    assert "PASSAGE [1]:" in prompt
    assert "QUESTION: what is alpha?" in prompt
    assert ABSTAIN in prompt  # the abstain instruction is present


def test_parse_detects_abstention():
    ans = parse_response("NOT IN SOURCES")
    assert ans.abstained is True
    assert ans.citations == []


def test_parse_extracts_citations():
    ans = parse_response("The answer is X, see [2] and [5].")
    assert ans.abstained is False
    assert ans.citations == [2, 5]
