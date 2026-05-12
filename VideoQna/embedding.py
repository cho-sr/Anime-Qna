from __future__ import annotations

from typing import Any, Optional

import numpy as np


class QwenSummaryEmbedder:
    def __init__(
        self,
        model_name: str = "ibm-granite/granite-embedding-97m-multilingual-r2",
        token: str = "",
        provider: Optional[str] = None,
        timeout: float = 120.0,
        normalize: bool = True,
    ):
        if not token:
            raise RuntimeError("HF_TOKEN is required for embedding API calls.")
        self.model_name = model_name
        self.provider = provider
        self.normalize = normalize
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is required for embedding API calls."
            ) from exc

        kwargs: dict[str, Any] = {"token": token, "timeout": timeout}
        if provider:
            kwargs["provider"] = provider
        self.client = InferenceClient(**kwargs)

    def embed_summary(self, summary: str) -> list[float]:
        return self.embed_text(summary)

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text.")

        provider_label = self.provider or "auto"
        print(f"[embedding] API model={self.model_name} provider={provider_label}")
        try:
            try:
                vector = self.client.feature_extraction(
                    text,
                    model=self.model_name,
                    normalize=self.normalize,
                )
            except TypeError:
                vector = self.client.feature_extraction(text, model=self.model_name)
        except Exception as exc:
            raise RuntimeError(self._format_embedding_error(exc)) from exc

        array = np.asarray(vector, dtype=np.float32)
        if array.ndim == 0:
            raise RuntimeError("Embedding API returned a scalar instead of a vector.")
        if array.ndim == 1:
            return array.tolist()
        if array.shape[0] == 1:
            return array.reshape(-1).tolist()

        # Some generic feature-extraction backends return token-level vectors.
        # Mean-pool as a conservative fallback so Qdrant still receives one vector per summary.
        return array.mean(axis=0).reshape(-1).tolist()

    def _format_embedding_error(self, exc: Exception) -> str:
        provider_label = self.provider or "auto"
        message = str(exc)
        hint = ""
        if "404" in message or "Not Found" in message:
            hint = (
                "\nHint: the selected embedding model/provider is not available for "
                "Hugging Face feature-extraction. In .env, use a supported pair such as "
                "HF_EMBEDDING_PROVIDER=hf-inference with "
                "HF_EMBEDDING_MODEL=ibm-granite/granite-embedding-97m-multilingual-r2, "
                "or HF_EMBEDDING_PROVIDER=scaleway with "
                "HF_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B."
            )
        return (
            f"Embedding request failed for model '{self.model_name}' "
            f"with provider '{provider_label}': {message}{hint}"
        )
