"""Clean up text extracted from PDFs so the model reads meaning, not noise.

We measured (Exp 8) that grounding over a math-dense PDF scored *worse* than the
model's own memory, because PDF extraction mangles the text. The single biggest
offenders in a real physics book were deterministic and safe to fix:

  * ligatures:  ﬁ ﬂ ﬀ ﬃ ﬄ  ->  fi fl ff ffi ffl   (~2500 occurrences)
  * reduced Planck constant:  ¯h  ->  ℏ              (~1100 occurrences)
  * split accents:  Schr¨ odinger  ->  Schrödinger
  * line-break hyphenation:  "proba- bility"  ->  "probability"

This pass handles exactly those — the safe, high-volume cases. It deliberately
does NOT try to repair spurious mid-word spaces ("operat or", "p article") or
collapsed fractions: guessing word boundaries needs a dictionary and risks
merging legitimate words, and fractions need layout-aware extraction. Those are a
separate, harder step (a better extractor / math-OCR). Cleaning is conservative
on purpose — it should only ever remove noise, never change meaning.
"""

from __future__ import annotations

import re

_LIGATURES = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}
_DIAERESIS = {"a": "ä", "o": "ö", "u": "ü", "A": "Ä", "O": "Ö", "U": "Ü", "e": "ë"}
_ACUTE = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú"}

# A spacing diaeresis/acute that got detached from its vowel: "¨ o" -> "ö".
_SPLIT_DIAERESIS = re.compile(r"¨ ?([aouAOUe])")
_SPLIT_ACUTE = re.compile(r"´ ?([aeiou])")
# Line-break hyphenation: a word split as "proba- bility". Restricted to
# lowercase-hyphen-space-lowercase so it won't touch "x - y" or real dashes.
_HYPHEN_BREAK = re.compile(r"([a-z])- ([a-z])")


def clean_text(text: str) -> str:
    """Return `text` with safe, high-volume extraction noise removed."""
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    text = text.replace("¯h", "ℏ").replace("h¯", "ℏ")
    text = _SPLIT_DIAERESIS.sub(lambda m: _DIAERESIS[m.group(1)], text)
    text = _SPLIT_ACUTE.sub(lambda m: _ACUTE[m.group(1)], text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = text.replace("­", "")  # soft hyphen
    return text
