r"""Math-aware extraction: read the page as an IMAGE, not a glyph stream.

Glyph-based PDF extraction (pypdf) destroys equation structure — fractions, roots
and superscripts collapse into nonsense like `A ≡ mωx + ip√ 2mℏω` (Exp 8/11). The
information is simply not recoverable from the text layer.

So we don't use the text layer. We render each page to an image and ask a vision
model to transcribe it to clean Markdown with LaTeX math. A vision model reads the
*rendered* equation the way a person does, so `A ≡ (mωx+ip)/√(2mℏω)` comes back as
`A \equiv \frac{m\omega x + ip}{\sqrt{2m\hbar\omega}}`.

This is the project's "any model, local or cloud" thesis applied to ingestion: any
vision-capable OpenAI-compatible endpoint works — a cloud model today, a local MLX
vision model on the Mac later, with no code change. It's a convert-ONCE step (one
call per page), so you pay it at import time and everything downstream reads clean
text.

Needs `pymupdf` (page rendering) and a vision-capable model.
"""

from __future__ import annotations

import base64
import json
import sys
import time
import urllib.request

_PROMPT = (
    "Transcribe this page from a textbook to clean Markdown. Render ALL mathematics "
    "as LaTeX: inline math as $...$, displayed equations as $$...$$. Keep equation, "
    "section and box labels exactly as printed (e.g. (3.2a), Box 2.1). Transcribe the "
    "text faithfully and do NOT solve, summarise, or add commentary. Output only the "
    "transcription."
)


def _post_vision(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={**headers, "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class VisionExtractor:
    """Render PDF pages and transcribe each to clean Markdown+LaTeX with a vision model."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        dpi: int = 170,
        timeout: float = 180.0,
        retries: int = 3,
        backoff: float = 2.0,
        post_fn=_post_vision,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.dpi = dpi
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._post = post_fn

    def _transcribe_png(self, png: bytes) -> str:
        b64 = base64.b64encode(png).decode("ascii")
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
        }
        url = f"{self.base_url}/chat/completions"
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self._post(url, payload, headers, self.timeout)
                return resp["choices"][0]["message"]["content"]
            except Exception as err:  # noqa: BLE001 - retry transient failures
                last = err
                if attempt < self.retries - 1:
                    time.sleep(self.backoff * (attempt + 1))
        raise last  # exhausted retries

    def extract(self, pdf_path: str, pages: list[int] | None = None,
                progress: bool = True) -> str:
        """Transcribe `pages` (0-based; default all) of a PDF to one clean string."""
        return "\n\n".join(t for _p, t in self.extract_pages(pdf_path, pages, progress))

    def extract_pages(self, pdf_path: str, pages: list[int] | None = None,
                      progress: bool = True) -> list[tuple[int, str]]:
        """Transcribe pages, keeping page numbers: [(1-based page, markdown), ...].

        Keeping the page number lets an index built from a vision transcription
        still cite "p. 112" — the human-checkable citation.
        """
        try:
            import fitz  # pymupdf
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "Vision extraction needs pymupdf. Install with: pip install pymupdf"
            ) from exc
        doc = fitz.open(pdf_path)
        idxs = list(range(doc.page_count)) if pages is None else pages
        out: list[tuple[int, str]] = []
        for n, i in enumerate(idxs, 1):
            png = doc[i].get_pixmap(dpi=self.dpi).tobytes("png")
            out.append((i + 1, self._transcribe_png(png)))
            if progress:
                print(f"    transcribed page {n}/{len(idxs)} (pdf page {i + 1})",
                      file=sys.stderr)
        return out
