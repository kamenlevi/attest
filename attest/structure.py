"""Split a document into sections for hierarchical retrieval — robustly.

Detecting real chapters in arbitrary PDFs is unreliable, so the design is
defensive:

  1. Detect headings with STRICT patterns (real "Chapter N" / numbered / genuine
     ALL-CAPS lines) — never with IGNORECASE, which previously matched almost
     every short line and exploded one book into ~944 "sections".
  2. VALIDATE the result: keep heading-based sections only if there are at least
     a couple and their median size is substantial. A problem bank whose every
     "3.14"-style label looks like a heading fails this check...
  3. ...and falls back to fixed-size word blocks, so the hierarchy is always
     sane (a 700-page book -> ~100 evenly-sized sections to route over).

The fallback is the point: useful structure on clean books, graceful coarse
blocks on messy ones — never a degenerate one-section-per-line index.
"""

from __future__ import annotations

import re
import statistics

# "Chapter 9", "Part 2", "Lecture 3" — case-insensitive on the keyword only.
_CHAPTER = re.compile(r"^(?:chapter|part|lecture|section)\s+\d+\b", re.IGNORECASE)
# "9.2 Einstein A and B", "3 Introduction" — a number then a Capitalised title.
_NUMBERED = re.compile(r"^\d+(?:\.\d+){0,2}\.?\s+[A-Z][A-Za-z].{0,70}$")


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80:
        return False
    if _CHAPTER.match(s) or _NUMBERED.match(s):
        return True
    # A genuine ALL-CAPS heading: has real letters, is fully upper-case, short.
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 4 and s == s.upper() and not s.endswith("."):
        return True
    return False


def _sections_from_headings(lines: list[str], heads: list[int]) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    if heads[0] > 0:
        pre = "\n".join(lines[: heads[0]]).strip()
        if pre:
            sections.append(("(front matter)", pre))
    bounds = heads + [len(lines)]
    for a, b in zip(heads, bounds[1:]):
        sections.append((lines[a].strip()[:80], "\n".join(lines[a:b])))
    return sections


def _fixed_blocks(text: str, target_words: int) -> list[tuple[str, str]]:
    words = text.split()
    if not words:
        return [("(empty)", "")]
    blocks = []
    for i in range(0, len(words), target_words):
        blocks.append((f"Section {i // target_words + 1}", " ".join(words[i : i + target_words])))
    return blocks


def _looks_sane(sections: list[tuple[str, str]], min_section_words: int) -> bool:
    """Heading-based sections are trusted only if there are a few and they're big
    enough — otherwise detection mistook body text/labels for headings."""
    if len(sections) < 2:
        return False
    sizes = [len(t.split()) for _, t in sections]
    return statistics.median(sizes) >= min_section_words


def split_into_sections(
    text: str, target_words: int = 2000, min_section_words: int = 300
) -> list[tuple[str, str]]:
    """Return [(title, section_text), ...] — detected chapters or fixed blocks."""
    lines = text.split("\n")
    heads = [i for i, ln in enumerate(lines) if _is_heading(ln)]
    if heads:
        sections = _sections_from_headings(lines, heads)
        if _looks_sane(sections, min_section_words):
            return sections
    return _fixed_blocks(text, target_words)
