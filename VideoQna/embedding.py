from __future__ import annotations

import random
import time
from typing import Any, Optional

import numpy as np


QWEN3_QUERY_INSTRUCTION = (
    "Given a Korean question about a video, retrieve scene descriptions that answer "
    "the question using visual evidence, actions, people, objects, places, emotions, "
    "and subtitles."
)


class QwenSummaryEmbedder:
    def __init__(
        self,
        model_name: str = "ibm-granite/granite-embedding-97m-multilingual-r2",
        token: str = "",
        provider: Optional[str] = None,
        timeout: float = 120.0,
        normalize: bool = True,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ):
        if not token:
            raise RuntimeError("HF_TOKEN is required for embedding API calls.")
        self.model_name = model_name
        self.provider = provider
        self.normalize = normalize
        self.max_retries = max(1, int(max_retries))
        self.retry_base_delay = max(0.1, float(retry_base_delay))
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
        return self.embed_document(summary)

    def embed_document(self, text: str) -> list[float]:
        return self.embed_text(text)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(self.format_query(query))

    def format_query(self, query: str) -> str:
        if self._is_qwen3_embedding_model():
            return f"Instruct: {QWEN3_QUERY_INSTRUCTION}\nQuery: {query}"
        return query

    def _is_qwen3_embedding_model(self) -> bool:
        normalized = self.model_name.lower()
        return "qwen3" in normalized and "embedding" in normalized

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text.")

        provider_label = self.provider or "auto"
        print(f"[embedding] API model={self.model_name} provider={provider_label}")
        try:
            vector = self._feature_extraction_with_retries(text)
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

    def _feature_extraction_once(self, text: str):
        try:
            return self.client.feature_extraction(
                text,
                model=self.model_name,
                normalize=self.normalize,
            )
        except TypeError:
            return self.client.feature_extraction(text, model=self.model_name)

    def _feature_extraction_with_retries(self, text: str):
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._feature_extraction_once(text)
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries - 1 or not self._is_retryable_error(exc):
                    raise
                self._sleep_before_retry(attempt, exc)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Embedding request failed without an exception.")

    def _sleep_before_retry(self, attempt: int, exc: Exception) -> None:
        delay = min(30.0, self.retry_base_delay * (2**attempt))
        delay += random.uniform(0.0, delay * 0.25)
        print(
            f"[retry] embedding transient error; retrying in {delay:.1f}s "
            f"({type(exc).__name__})"
        )
        time.sleep(delay)

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True

        message = str(exc).lower()
        retryable_markers = [
            "429",
            "rate limit",
            "timeout",
            "timed out",
            "temporarily",
            "try again",
            "503",
            "502",
            "504",
            "connection",
        ]
        return any(marker in message for marker in retryable_markers)

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
