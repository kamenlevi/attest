"""Normalizer tests: fixes real PDF garble, and is conservative (no meaning change)."""

from attest.normalize import clean_text


def test_fixes_ligatures():
    assert clean_text("the diﬀerence in eﬀective ﬂux deﬁnes the coeﬃcient") == \
        "the difference in effective flux defines the coefficient"


def test_fixes_reduced_planck_constant():
    assert clean_text("i¯h∂|ψ⟩/∂t = H|ψ⟩") == "iℏ∂|ψ⟩/∂t = H|ψ⟩"


def test_fixes_split_accent_in_schrodinger():
    assert "Schrödinger" in clean_text("the Schr¨ odinger equation")


def test_dehyphenates_line_break_splits():
    assert clean_text("the proba- bility of ﬁnding") == "the probability of finding"


def test_leaves_clean_text_unchanged():
    clean = "The commutator [x, p] = iℏ defines the uncertainty principle."
    assert clean_text(clean) == clean


def test_does_not_join_spaced_math_dashes():
    # "x - y" is a subtraction, not a hyphenated word: must stay put.
    assert clean_text("we compute x - y exactly") == "we compute x - y exactly"
