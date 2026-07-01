"""Step 1: load a document into plain text.

Supports .txt now, and .pdf when the optional `pypdf` dependency is installed
(`pip install -e '.[ingest]'`). PDF text extraction — especially of equations —
is the biggest known risk in Phase 1, so we keep this isolated and easy to swap.
"""

from __future__ import annotations

from pathlib import Path

from .normalize import clean_text


def load_text(path: str | Path, clean: bool = True) -> str:
    """Return the plain text of a .txt/.md/.pdf file.

    By default the text is run through `clean_text` to remove PDF extraction noise
    (ligatures, mangled ℏ, hyphenation) — measured to matter for grounding quality.
    Pass clean=False to get the raw extraction (e.g. to compare before/after).
    """
    joined = "\n".join(text for _page, text in load_pages(path, clean=False))
    return clean_text(joined) if clean else joined


def load_pages(path: str | Path, clean: bool = True) -> list[tuple[int | None, str]]:
    """Return the document as [(page_number, text), ...].

    Page numbers are 1-based for PDFs and None for plain text (which has no
    pages). Keeping the page alongside the text is what lets a citation say
    "p. 112" — something a person can actually open the book and check.
    """
    pages = _extract_pages(Path(path))
    if clean:
        pages = [(page, clean_text(text)) for page, text in pages]
    return pages


def _extract_pages(path: Path) -> list[tuple[int | None, str]]:
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        return [(None, path.read_text(encoding="utf-8", errors="replace"))]

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "Reading PDFs needs pypdf. Install with: pip install -e '.[ingest]'"
            ) from exc
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]

    raise ValueError(f"Unsupported file type: {suffix!r} (use .txt, .md or .pdf)")
