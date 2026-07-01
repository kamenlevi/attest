"""The verification layer: the green badge must mean 'we checked', not 'it said so'."""

from attest.chunking import Chunk
from attest.grounding import GroundedAnswer
from attest.retrieval import Retrieved
from attest.verify import SupportChecker, check_citations, verify_answer


def _retrieved():
    return [
        Retrieved(Chunk(3, "the dedication is to walter of merton", "book.pdf", page=5), 0.9),
        Retrieved(Chunk(7, "probability current J = (hbar/m) S^2 grad phi", "book.pdf", page=112), 0.8),
    ]


def _answer(text="It is dedicated to Walter of Merton.", citations=None, abstained=False):
    return GroundedAnswer(abstained=abstained, text=text, citations=citations or [])


class _FixedVerdict:
    """A fake judge that always returns the given verdict text."""

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.verdict


def test_check_citations_splits_valid_and_invalid():
    valid, invalid = check_citations(_answer(citations=[3, 99]), _retrieved())
    assert valid == [3]
    assert invalid == [99]


def test_abstained_answer_verifies_as_abstained():
    v = verify_answer("q", _answer(abstained=True), _retrieved())
    assert v.status == "abstained"


def test_uncited_answer_flagged():
    v = verify_answer("q", _answer(citations=[]), _retrieved())
    assert v.status == "uncited"


def test_fabricated_citation_flagged_as_invalid():
    # Cites only passages the model was never shown -> the citation is fabricated.
    v = verify_answer("q", _answer(citations=[42]), _retrieved())
    assert v.status == "invalid"
    assert v.invalid_citations == [42]


def test_valid_citation_without_checker_is_unverified_not_verified():
    # No judge configured: we can't confirm support, so no green badge.
    v = verify_answer("q", _answer(citations=[3]), _retrieved())
    assert v.status == "unverified"
    assert v.valid_citations == [3]


def test_judge_confirms_support_gives_verified():
    checker = SupportChecker(_FixedVerdict("SUPPORTED"))
    v = verify_answer("who is it dedicated to?", _answer(citations=[3]), _retrieved(), checker)
    assert v.status == "verified"


def test_judge_rejects_support_gives_unsupported():
    checker = SupportChecker(_FixedVerdict("UNSUPPORTED"))
    v = verify_answer("q", _answer(citations=[3]), _retrieved(), checker)
    assert v.status == "unsupported"


def test_unsupported_verdict_not_misread_as_supported():
    # "UNSUPPORTED" contains "SUPPORTED" — the negative must win.
    assert SupportChecker(_FixedVerdict("UNSUPPORTED")).check("q", "a", _retrieved()) is False
    assert SupportChecker(_FixedVerdict("SUPPORTED")).check("q", "a", _retrieved()) is True


def test_checker_only_sees_the_cited_passages():
    fake = _FixedVerdict("SUPPORTED")
    verify_answer("q", _answer(citations=[3]), _retrieved(), SupportChecker(fake))
    prompt = fake.prompts[0]
    assert "walter of merton" in prompt
    assert "probability current" not in prompt  # [7] wasn't cited -> not shown
