"""A cross-encoder reranker: the precision step after wide retrieval.

Embedding-based retrieval (bi-encoder) is fast but coarse: it compares a question
vector to chunk vectors independently, so the truly best passage can rank just
below the top-k and never reach the model (we measured exactly this — Exp 6: the
clean Schrödinger-equation and Born-rule passages existed but ranked too low).

A cross-encoder fixes that. It reads the question AND a candidate passage TOGETHER
and scores how well that passage answers the question. It's too slow to run over a
whole book, but perfect for re-scoring a few dozen candidates that retrieval
already narrowed down. So the pipeline is: retrieve wide (recall) -> rerank ->
keep the best few (precision).

The model (~80 MB) runs on CPU and is loaded lazily, so importing this costs
nothing until the first rerank.
"""

from __future__ import annotations

# A small, standard reranker trained on query/passage relevance (MS MARCO).
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER) -> None:
        self._model_name = model_name
        self._model = None  # lazy: loaded on first score()

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - optional extra
                raise ImportError(
                    "CrossEncoderReranker needs sentence-transformers. "
                    "Install with: pip install -e '.[embed-local]'"
                ) from exc
            self._model = CrossEncoder(self._model_name)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance score for each passage against the query (higher = better)."""
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        return [float(s) for s in self._ensure_model().predict(pairs)]
