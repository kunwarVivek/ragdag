"""OpenAI embedding engine."""

import os
from typing import List
from .base import EmbeddingEngine


class OpenAIEngine(EmbeddingEngine):
    """Embed text using OpenAI's text-embedding API."""

    def __init__(self, model: str = "text-embedding-3-small", dims: int = 1536):
        self._model = model
        self._dims = dims
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

    def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        client = openai.OpenAI(api_key=self._api_key)
        response = client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dims,
        )
        return [item.embedding for item in response.data]

    def dimensions(self) -> int:
        return self._dims

    def model_name(self) -> str:
        return self._model
