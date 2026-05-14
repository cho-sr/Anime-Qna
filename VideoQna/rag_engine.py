from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import local
from typing import Any

from bm25 import BM25Index
from embedding import (
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    QwenSummaryEmbedder,
    create_summary_embedder,
)
from hf_clients import RAGLLMClient
from utils import ensure_str, ensure_str_list, seconds_to_timestamp
from vector_store import QdrantSummaryStore


@dataclass
class RetrievalConfig:
    top_k: int = 6
    dense_top_k: int = 40
    bm25_top_k: int = 60
    rrf_k: int = 60
    dense_workers: int = 3


def payload_to_search_text(payload: dict[str, Any]) -> str:
    subtitles = payload.get("shot_subtitles") or []
    subtitle_texts = []
    if isinstance(subtitles, list):
        for item in subtitles:
            if isinstance(item, dict):
                subtitle_texts.append(ensure_str(item.get("text")))
            else:
                subtitle_texts.append(ensure_str(item))

    fields = [
        ensure_str(payload.get("search_text")),
        ensure_str(payload.get("summary")),
        " ".join(ensure_str_list(payload.get("action"))),
        ensure_str(payload.get("context")),
        " ".join(ensure_str_list(payload.get("emotion"))),
        " ".join(ensure_str_list(payload.get("people"))),
        " ".join(ensure_str_list(payload.get("objects"))),
        " ".join(ensure_str_list(payload.get("places"))),
        " ".join(ensure_str_list(payload.get("visual_keywords"))),
        " ".join(ensure_str_list(payload.get("dialogue_keywords"))),
        ensure_str(payload.get("frame_description")),
        " ".join(subtitle_texts),
    ]
    return "\n".join(part for part in fields if part)


def payload_to_source(
    point_id: str,
    payload: dict[str, Any],
    rank: int,
    rrf_score: float,
    dense_score: float = 0.0,
    bm25_score: float = 0.0,
) -> dict[str, Any]:
    shot_start = float(payload.get("shot_start_sec") or 0.0)
    shot_end = float(payload.get("shot_end_sec") or 0.0)
    subtitles = payload.get("shot_subtitles") or []
    subtitle_texts = []
    if isinstance(subtitles, list):
        for item in subtitles:
            if isinstance(item, dict) and item.get("text"):
                subtitle_texts.append(
                    {
                        "start_time": item.get("start_time"),
                        "end_time": item.get("end_time"),
                        "text": item.get("text"),
                    }
                )

    return {
        "rank": rank,
        "id": point_id,
        "rrf_score": rrf_score,
        "dense_score": dense_score,
        "bm25_score": bm25_score,
        "shot_id": payload.get("shot_id"),
        "timestamp": f"{seconds_to_timestamp(shot_start)} ~ {seconds_to_timestamp(shot_end)}",
        "shot_start_sec": shot_start,
        "shot_end_sec": shot_end,
        "keyframe_timestamp_sec": payload.get("keyframe_timestamp_sec"),
        "image_path": payload.get("image_path"),
        "summary": payload.get("summary"),
        "action": payload.get("action") or [],
        "context": payload.get("context") or "",
        "emotion": payload.get("emotion") or [],
        "people": payload.get("people") or [],
        "objects": payload.get("objects") or [],
        "places": payload.get("places") or [],
        "visual_keywords": payload.get("visual_keywords") or [],
        "dialogue_keywords": payload.get("dialogue_keywords") or [],
        "search_text": payload.get("search_text") or "",
        "frame_description": payload.get("frame_description") or "",
        "subtitles": subtitle_texts,
    }


class HybridRAGEngine:
    def __init__(
        self,
        qdrant_path: str | Path,
        hf_token: str,
        llm_model: str,
        embedding_model: str,
        embedding_backend: str = "local",
        hf_provider: str | None = None,
        llm_provider: str | None = None,
        embedding_provider: str | None = None,
        local_embedding_device: str = "auto",
        local_embedding_batch_size: int = 8,
        local_embedding_max_length: int = 2048,
        store: QdrantSummaryStore | None = None,
    ):
        if not hf_token:
            raise RuntimeError("HF_TOKEN is required for RAG query expansion and answers.")

        self.hf_token = hf_token
        self.embedding_model = embedding_model
        self.embedding_backend = (embedding_backend or "local").strip().lower()
        self.embedding_provider = "local" if self.embedding_backend == "local" else embedding_provider or hf_provider
        self.local_embedding_device = local_embedding_device
        self.local_embedding_batch_size = int(local_embedding_batch_size or 8)
        self.local_embedding_max_length = int(local_embedding_max_length or 2048)
        self.store = store or QdrantSummaryStore(qdrant_path=qdrant_path)
        self.embedder = create_summary_embedder(
            backend=self.embedding_backend,
            model_name=embedding_model,
            token=hf_token,
            provider=self.embedding_provider,
            local_device=self.local_embedding_device,
            local_batch_size=self.local_embedding_batch_size,
            local_max_length=self.local_embedding_max_length,
        )
        self.llm = RAGLLMClient(token=hf_token, model=llm_model, provider=llm_provider or hf_provider)
        self._collection_cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_env(
        cls,
        qdrant_path: str | Path,
        store: QdrantSummaryStore | None = None,
    ) -> "HybridRAGEngine":
        hf_provider = os.getenv("HF_PROVIDER") or None
        embedding_backend = (os.getenv("EMBEDDING_BACKEND") or "local").strip().lower()
        if embedding_backend == "local":
            embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL)
            embedding_provider = "local"
        else:
            embedding_model = os.getenv(
                "HF_EMBEDDING_MODEL",
                "ibm-granite/granite-embedding-97m-multilingual-r2",
            )
            embedding_provider = os.getenv("HF_EMBEDDING_PROVIDER") or hf_provider
        return cls(
            qdrant_path=qdrant_path,
            hf_token=os.getenv("HF_TOKEN", ""),
            hf_provider=hf_provider,
            llm_provider=os.getenv("HF_LLM_PROVIDER") or hf_provider,
            embedding_provider=embedding_provider,
            llm_model=os.getenv("HF_LLM_MODEL", "Qwen/Qwen3-8B"),
            embedding_model=embedding_model,
            embedding_backend=embedding_backend,
            local_embedding_device=os.getenv("LOCAL_EMBEDDING_DEVICE", "auto"),
            local_embedding_batch_size=int(os.getenv("LOCAL_EMBEDDING_BATCH_SIZE", "8")),
            local_embedding_max_length=int(os.getenv("LOCAL_EMBEDDING_MAX_LENGTH", "2048")),
            store=store,
        )

    def _new_embedder(self) -> QwenSummaryEmbedder:
        return create_summary_embedder(
            backend=self.embedding_backend,
            model_name=self.embedding_model,
            token=self.hf_token,
            provider=self.embedding_provider,
            local_device=self.local_embedding_device,
            local_batch_size=self.local_embedding_batch_size,
            local_max_length=self.local_embedding_max_length,
        )

    def ask(self, question: str, collection: str, config: RetrievalConfig) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("question is required.")

        if not self.store.collection_exists(collection):
            return {
                "answer": f"'{collection}' 컬렉션을 찾을 수 없습니다. 먼저 영상을 인덱싱해 주세요.",
                "expanded_queries": [],
                "sources": [],
                "retrieval_debug": {
                    "collection_exists": False,
                    "dense_results": [],
                    "bm25_results": [],
                    "rrf_results": [],
                    "warnings": [],
                },
            }

        expansion_warning = None
        try:
            expanded = self.llm.expand_query(question)
        except Exception as exc:
            expanded = {"expanded_queries": [], "keywords": []}
            expansion_warning = f"query expansion failed, used original question only: {exc}"

        expanded_queries = [
            item
            for item in expanded.get("expanded_queries", [])
            if isinstance(item, str) and item.strip()
        ][:3]
        keywords = [
            item
            for item in expanded.get("keywords", [])
            if isinstance(item, str) and item.strip()
        ][:12]
        search_queries = [question] + [
            query for query in expanded_queries if query.strip() != question.strip()
        ]

        dense_rankings, dense_debug, best_dense_scores = self._dense_rankings(
            collection=collection,
            queries=search_queries,
            limit=config.dense_top_k,
            workers=config.dense_workers,
        )

        payload_records, bm25_index = self._collection_payloads_and_bm25(collection)
        bm25_query = " ".join([question, *expanded_queries, *keywords])
        bm25_results = bm25_index.search(bm25_query, top_k=config.bm25_top_k)
        best_bm25_scores = {item["id"]: float(item["score"]) for item in bm25_results}

        rrf_scores = self._rrf([*dense_rankings, [item["id"] for item in bm25_results]], config.rrf_k)
        payload_by_id = {record["id"]: record["payload"] for record in payload_records}
        for item in dense_debug:
            payload_by_id.setdefault(item["id"], item["payload"])

        ranked_ids = [
            point_id
            for point_id, _score in sorted(
                rrf_scores.items(), key=lambda item: item[1], reverse=True
            )
            if point_id in payload_by_id
        ][: config.top_k]

        sources = [
            payload_to_source(
                point_id=point_id,
                payload=payload_by_id[point_id],
                rank=rank,
                rrf_score=rrf_scores[point_id],
                dense_score=best_dense_scores.get(point_id, 0.0),
                bm25_score=best_bm25_scores.get(point_id, 0.0),
            )
            for rank, point_id in enumerate(ranked_ids, start=1)
        ]

        answer = self.llm.answer_question(question, sources)
        warnings = [expansion_warning] if expansion_warning else []
        return {
            "answer": answer,
            "expanded_queries": expanded_queries,
            "keywords": keywords,
            "sources": sources,
            "retrieval_debug": {
                "collection_exists": True,
                "config": {
                    "top_k": config.top_k,
                    "dense_top_k": config.dense_top_k,
                    "bm25_top_k": config.bm25_top_k,
                    "rrf_k": config.rrf_k,
                    "dense_workers": config.dense_workers,
                },
                "dense_results": [
                    {
                        "query": item["query"],
                        "id": item["id"],
                        "rank": item["rank"],
                        "score": item["score"],
                        "shot_id": item["payload"].get("shot_id"),
                    }
                    for item in dense_debug
                ],
                "bm25_results": [
                    {
                        "id": item["id"],
                        "rank": rank,
                        "score": item["score"],
                        "shot_id": item["payload"].get("shot_id"),
                    }
                    for rank, item in enumerate(bm25_results, start=1)
                ],
                "rrf_results": [
                    {"id": source["id"], "rank": source["rank"], "score": source["rrf_score"]}
                    for source in sources
                ],
                "warnings": warnings,
            },
        }

    def _dense_rankings(
        self,
        collection: str,
        queries: list[str],
        limit: int,
        workers: int = 1,
    ) -> tuple[list[list[str]], list[dict[str, Any]], dict[str, float]]:
        rankings: list[list[str]] = []
        debug: list[dict[str, Any]] = []
        best_scores: dict[str, float] = {}

        def embed_queries() -> list[tuple[str, list[float]]]:
            ordered_vectors: list[tuple[str, list[float]] | None] = [None] * len(queries)
            workers_count = max(1, int(workers or 1))
            if getattr(self.embedder, "is_local", False):
                workers_count = 1
            if workers_count == 1 or len(queries) <= 1:
                for index, query in enumerate(queries):
                    ordered_vectors[index] = (query, self.embedder.embed_query(query))
                return [item for item in ordered_vectors if item is not None]

            worker_state = local()

            def get_worker_embedder() -> QwenSummaryEmbedder:
                embedder = getattr(worker_state, "embedder", None)
                if embedder is None:
                    embedder = self._new_embedder()
                    worker_state.embedder = embedder
                return embedder

            def embed_one(index: int, query: str) -> tuple[int, str, list[float]]:
                return index, query, get_worker_embedder().embed_query(query)

            with ThreadPoolExecutor(max_workers=min(workers_count, len(queries))) as executor:
                futures = {
                    executor.submit(embed_one, index, query): index
                    for index, query in enumerate(queries)
                }
                for future in as_completed(futures):
                    index, query, vector = future.result()
                    ordered_vectors[index] = (query, vector)

            return [item for item in ordered_vectors if item is not None]

        for query, vector in embed_queries():
            results = self.store.dense_search(collection, vector=vector, limit=limit)
            ranking = []
            query_debug = []
            for rank, result in enumerate(results, start=1):
                ranking.append(result["id"])
                query_debug.append(
                    {
                        "query": query,
                        "rank": rank,
                        "id": result["id"],
                        "score": result["score"],
                        "payload": result["payload"],
                    }
                )
            rankings.append(ranking)
            debug.extend(query_debug)
            for row in query_debug:
                best_scores[row["id"]] = max(
                    best_scores.get(row["id"], float("-inf")),
                    float(row["score"]),
                )

        return rankings, debug, best_scores

    def _collection_payloads_and_bm25(
        self,
        collection: str,
    ) -> tuple[list[dict[str, Any]], BM25Index]:
        stats = self.store.collection_stats(collection)
        points_count = int(stats.get("points_count") or 0)
        cached = self._collection_cache.get(collection)
        if cached and cached.get("points_count") == points_count:
            return cached["payload_records"], cached["bm25_index"]

        payload_records = self.store.scroll_payloads(collection)
        documents = [
            {
                "id": record["id"],
                "payload": record["payload"],
                "text": payload_to_search_text(record["payload"]),
            }
            for record in payload_records
        ]
        bm25_index = BM25Index(documents)
        self._collection_cache[collection] = {
            "points_count": points_count,
            "payload_records": payload_records,
            "bm25_index": bm25_index,
        }
        return payload_records, bm25_index

    @staticmethod
    def _rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
        scores: dict[str, float] = {}
        for ranking in rankings:
            seen = set()
            for rank, point_id in enumerate(ranking, start=1):
                if point_id in seen:
                    continue
                seen.add(point_id)
                scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (k + rank)
        return scores
