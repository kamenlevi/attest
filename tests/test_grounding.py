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


def test_parse_json_protocol():
    ans = parse_response('{"found": true, "answer": "It is X.", "citations": [3, 7]}')
    assert ans.abstained is False
    assert ans.text == "It is X."
    assert ans.citations == [3, 7]
    assert ans.structured is True


def test_parse_json_abstention():
    ans = parse_response('{"found": false, "answer": "NOT IN SOURCES", "citations": []}')
    assert ans.abstained is True
    assert ans.citations == []


def test_parse_json_tolerates_code_fences_and_prose():
    ans = parse_response('Sure! ```json\n{"found": true, "answer": "X", "citations": [1]}\n```')
    assert ans.abstained is False
    assert ans.citations == [1]


def test_parse_broken_json_falls_back_to_text_protocol():
    # Malformed JSON must degrade to the legacy [n]-regex parse, not error.
    ans = parse_response('{"found": true, "answer": "X" -- see [4]')
    assert ans.abstained is False
    assert ans.citations == [4]
    assert ans.structured is False


def test_parse_json_answer_containing_bracket_numbers_not_double_counted():
    # A book's own "[12]" reference inside the answer text must not become a
    # citation when the model already told us its citations exactly.
    ans = parse_response('{"found": true, "answer": "See ref [12] in the text.", "citations": [3]}')
    assert ans.citations == [3]
