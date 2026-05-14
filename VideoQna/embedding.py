from __future__ import annotations

import random
import threading
import time
from typing import Any, Optional

import numpy as np


DEFAULT_LOCAL_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

QWEN3_QUERY_INSTRUCTION = (
    "Given a Korean question about a video, retrieve scene descriptions that answer "
    "the question using visual evidence, actions, people, objects, places, emotions, "
    "and subtitles."
)


class QwenSummaryEmbedder:
    is_local = False

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


class LocalQwenSummaryEmbedder:
    is_local = True

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
        device: str = "auto",
        token: str = "",
        batch_size: int = 8,
        max_length: int = 2048,
        normalize: bool = True,
    ):
        self.model_name = model_name or DEFAULT_LOCAL_EMBEDDING_MODEL
        self.device_name = self._resolve_device(device)
        self.token = token or None
        self.batch_size = max(1, int(batch_size or 1))
        self.max_length = max(128, int(max_length or 2048))
        self.normalize = normalize
        self._lock = threading.Lock()

        try:
            import torch
            import transformers.utils as transformers_utils
            import transformers.utils.import_utils as transformers_import_utils

            transformers_utils.is_torchvision_available = lambda: False
            transformers_import_utils.is_torchvision_available = lambda: False
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local embeddings require torch and transformers. "
                "Install VideoQna/requirements.txt first."
            ) from exc

        dtype = torch.float16 if self.device_name == "cuda" else torch.float32
        kwargs: dict[str, Any] = {}
        if self.token:
            kwargs["token"] = self.token

        print(
            f"[embedding] loading local model={self.model_name} "
            f"device={self.device_name} dtype={dtype} max_length={self.max_length}"
        )
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            **kwargs,
        )
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                dtype=dtype,
                **kwargs,
            )
        except TypeError:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                **kwargs,
            )
        self.model.to(self.device_name)
        self.model.eval()

    def embed_summary(self, summary: str) -> list[float]:
        return self.embed_document(summary)

    def embed_document(self, text: str) -> list[float]:
        return self.embed_text(text)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(self.format_query(query))

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned = [str(text).strip() for text in texts]
        if any(not text for text in cleaned):
            raise ValueError("Cannot embed empty text.")

        print(f"[embedding] local model={self.model_name} batch={len(cleaned)}")
        vectors: list[list[float]] = []
        # The model is shared across API worker threads, so serialize local GPU inference.
        with self._lock:
            for start in range(0, len(cleaned), self.batch_size):
                batch_texts = cleaned[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self.device_name)
                    for key, value in encoded.items()
                }
                with self.torch.inference_mode():
                    outputs = self.model(**encoded)
                    pooled = self._last_token_pool(
                        outputs.last_hidden_state,
                        encoded["attention_mask"],
                    )
                    if self.normalize:
                        pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.detach().cpu().float().numpy().tolist())
        return vectors

    def format_query(self, query: str) -> str:
        if self._is_qwen3_embedding_model():
            return f"Instruct: {QWEN3_QUERY_INSTRUCTION}\nQuery: {query}"
        return query

    def _is_qwen3_embedding_model(self) -> bool:
        normalized = self.model_name.lower()
        return "qwen3" in normalized and "embedding" in normalized

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
        if left_padding:
            return last_hidden_states[:, -1]

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            LocalQwenSummaryEmbedder.torch_arange(
                last_hidden_states,
                batch_size,
            ),
            sequence_lengths,
        ]

    @staticmethod
    def torch_arange(tensor, batch_size: int):
        import torch

        return torch.arange(batch_size, device=tensor.device)

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        value = (device or "auto").strip().lower()
        if value in {"cpu", "cuda"}:
            return value
        try:
            import torch
        except Exception:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"


def create_summary_embedder(
    *,
    backend: str = "api",
    model_name: str,
    token: str = "",
    provider: Optional[str] = None,
    local_device: str = "auto",
    local_batch_size: int = 8,
    local_max_length: int = 2048,
):
    value = (backend or "api").strip().lower()
    if value == "local" or (provider or "").strip().lower() == "local":
        return LocalQwenSummaryEmbedder(
            model_name=model_name or DEFAULT_LOCAL_EMBEDDING_MODEL,
            device=local_device,
            token=token,
            batch_size=local_batch_size,
            max_length=local_max_length,
        )

    return QwenSummaryEmbedder(
        model_name=model_name,
        token=token,
        provider=provider,
    )
