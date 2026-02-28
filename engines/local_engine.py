"""Local embedding engine using sentence-transformers."""

from typing import List
from .base import EmbeddingEngine


class LocalEngine(EmbeddingEngine):
    """Embed text using a local sentence-transformers model."""

    def __init__(self, model: str = "all-MiniLM-L6-v2", dims: int = 384):
        self._model_name = model
        self._dims = dims
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self._model_name)
            self._dims = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    def dimensions(self) -> int:
        return self._dims

    def model_name(self) -> str:
        return self._model_name
