from attest.eval import EvalItem, EvalResult, compute_metrics
from attest.grounding import GroundedAnswer


def _result(answerable: bool, abstained: bool, citations=None) -> EvalResult:
    return EvalResult(
        item=EvalItem("q", answerable=answerable),
        answer=GroundedAnswer(abstained=abstained, text="", citations=citations or []),
    )


def test_bluff_rate_counts_traps_that_did_not_abstain():
    results = [
        _result(answerable=False, abstained=True),          # good: abstained on trap
        _result(answerable=False, abstained=False),         # BLUFF
        _result(answerable=False, abstained=False),         # BLUFF
        _result(answerable=True, abstained=False, citations=[1]),  # answered correctly-ish
    ]
    m = compute_metrics(results)
    assert m["n_traps"] == 3
    assert m["bluff_rate"] == 2 / 3
    assert m["answer_coverage"] == 1.0  # the single answerable was answered
    # 3 responses were "answered" (1 real + 2 bluffs); only the real one cited.
    # Bluffs answering without citations is expected — low citation rate is a smell.
    assert m["citation_rate"] == 1 / 3


def test_perfect_behaviour_is_zero_bluff_full_coverage():
    results = [
        _result(answerable=True, abstained=False, citations=[0]),
        _result(answerable=False, abstained=True),
    ]
    m = compute_metrics(results)
    assert m["bluff_rate"] == 0.0
    assert m["answer_coverage"] == 1.0
