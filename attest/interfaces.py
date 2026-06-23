"""The two things that will eventually need a Mac + MLX — defined as interfaces.

By programming against these abstract interfaces (not against MLX directly), we
can build and test the whole pipeline on Linux using the *mock* backend, then
drop in the *real* MLX backend on the Mac with zero changes elsewhere.

- `Embedder`: turns text into vectors (numbers capturing meaning) so we can
  search by similarity.
- `Generator`: takes a prompt string and returns the model's text answer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim) of unit-length vectors."""
        raise NotImplementedError


class Generator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the model's response to a single prompt string."""
        raise NotImplementedError
