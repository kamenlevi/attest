"""A real embedding backend that runs on a normal CPU (no GPU/MLX needed).

It uses a small sentence-embedding model via the `sentence-transformers` library.
The default model is tiny (~80 MB) and runs fine on an old laptop CPU, so real,
meaning-aware retrieval works on the ThinkPad today.

Install with:  pip install -e '.[embed-local]'

Unlike the mock embedder (which only matches words), this understands meaning, so
a question phrased differently from the text still finds the right passage.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import Embedder

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        # Lazy: don't load the model (torch + ~80 MB weights, several seconds) at
        # construction. Only the first embed() call pays it. This keeps actions
        # that hold an embedder but never embed — e.g. opening an index just to
        # skip already-indexed files — instant.
        self._model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise ImportError(
                    "LocalEmbedder needs sentence-transformers. "
                    "Install with: pip install -e '.[embed-local]'"
                ) from exc
            # First use downloads the model once, then it's cached locally.
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True -> unit-length vectors, so dot product == cosine.
        vecs = self._ensure_model().encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return vecs.astype(np.float32)
