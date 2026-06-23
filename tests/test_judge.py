from attest.eval import EvalItem, EvalResult, compute_metrics, grade_results
from attest.grounding import GroundedAnswer
from attest.interfaces import Generator
from attest.judge import Judge


class ScriptedGenerator(Generator):
    """Returns canned verdicts in order — lets us test grading with no model."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)

    def generate(self, prompt: str) -> str:
        return self._verdicts.pop(0)


def _answered(answerable: bool, abstained: bool) -> EvalResult:
    return EvalResult(
        item=EvalItem("q", answerable=answerable),
        answer=GroundedAnswer(abstained=abstained, text="some answer", citations=[1]),
        context="ctx",
    )


def test_judge_parses_correct_and_incorrect():
    # "INCORRECT" contains "CORRECT" — make sure it isn't misread as correct.
    judge = Judge(ScriptedGenerator(["CORRECT", "INCORRECT"]))
    assert judge.grade("q", "a", "c") is True
    assert judge.grade("q", "a", "c") is False


def test_grade_results_only_grades_answered_answerable():
    results = [
        _answered(answerable=True, abstained=False),   # graded
        _answered(answerable=True, abstained=True),    # abstained -> not graded
        _answered(answerable=False, abstained=False),  # trap -> not graded
    ]
    judge = Judge(ScriptedGenerator(["CORRECT"]))  # only ONE grade should be requested
    graded = grade_results(results, judge)

    assert graded[0].correct is True
    assert graded[1].correct is None
    assert graded[2].correct is None

    m = compute_metrics(graded)
    assert m["n_graded"] == 1
    assert m["correctness_rate"] == 1.0
