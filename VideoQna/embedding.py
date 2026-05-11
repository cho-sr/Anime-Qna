from __future__ import annotations

from typing import Optional

import numpy as np


class QwenSummaryEmbedder:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for Qwen3 embeddings."
                ) from exc

            print(f"[embedding] loading model={self.model_name}")
            try:
                self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
            except TypeError:
                self._model = SentenceTransformer(self.model_name)

    def embed_summary(self, summary: str) -> list[float]:
        if not summary.strip():
            raise ValueError("Cannot embed an empty summary.")
        self._load_model()
        vector = self._model.encode(
            summary,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vector, dtype=np.float32).reshape(-1).tolist()

