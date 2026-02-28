"""Abstract base for embedding engines."""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingEngine(ABC):
    """Abstract embedding engine interface."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into vectors."""
        ...

    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...
