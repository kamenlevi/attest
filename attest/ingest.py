"""Step 1: load a document into plain text.

Supports .txt now, and .pdf when the optional `pypdf` dependency is installed
(`pip install -e '.[ingest]'`). PDF text extraction — especially of equations —
is the biggest known risk in Phase 1, so we keep this isolated and easy to swap.
"""

from __future__ import annotations

from pathlib import Path


def load_text(path: str | Path) -> str:
    """Return the plain text of a .txt or .pdf file."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "Reading PDFs needs pypdf. Install with: pip install -e '.[ingest]'"
            ) from exc
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    raise ValueError(f"Unsupported file type: {suffix!r} (use .txt or .pdf)")
