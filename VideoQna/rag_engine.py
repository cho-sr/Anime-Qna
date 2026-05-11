from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bm25 import BM25Index
from embedding import QwenSummaryEmbedder
from hf_clients import RAGLLMClient
from utils import ensure_str, ensure_str_list, seconds_to_timestamp
from vector_store import QdrantSummaryStore


@dataclass
class RetrievalConfig:
    top_k: int = 5
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60


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
        ensure_str(payload.get("summary")),
        " ".join(ensure_str_list(payload.get("action"))),
        ensure_str(payload.get("context")),
        " ".join(ensure_str_list(payload.get("emotion"))),
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
        hf_provider: str | None = None,
    ):
        if not hf_token:
            raise RuntimeError("HF_TOKEN is required for RAG query expansion, embedding, and answers.")

        self.store = QdrantSummaryStore(qdrant_path=qdrant_path)
        self.embedder = QwenSummaryEmbedder(
            model_name=embedding_model,
            token=hf_token,
            provider=hf_provider,
        )
        self.llm = RAGLLMClient(token=hf_token, model=llm_model, provider=hf_provider)

    @classmethod
    def from_env(cls, qdrant_path: str | Path) -> "HybridRAGEngine":
        return cls(
            qdrant_path=qdrant_path,
            hf_token=os.getenv("HF_TOKEN", ""),
            hf_provider=os.getenv("HF_PROVIDER") or None,
            llm_model=os.getenv("HF_LLM_MODEL", "Qwen/Qwen3-8B"),
            embedding_model=os.getenv("HF_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
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
        )

        payload_records = self.store.scroll_payloads(collection)
        bm25_query = " ".join([question, *expanded_queries, *keywords])
        bm25_results = self._bm25_results(payload_records, bm25_query, config.bm25_top_k)
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
    ) -> tuple[list[list[str]], list[dict[str, Any]], dict[str, float]]:
        rankings: list[list[str]] = []
        debug: list[dict[str, Any]] = []
        best_scores: dict[str, float] = {}

        for query in queries:
            vector = self.embedder.embed_text(query)
            results = self.store.dense_search(collection, vector=vector, limit=limit)
            ranking = []
            for rank, result in enumerate(results, start=1):
                ranking.append(result["id"])
                best_scores[result["id"]] = max(
                    best_scores.get(result["id"], float("-inf")),
                    float(result["score"]),
                )
                debug.append(
                    {
                        "query": query,
                        "rank": rank,
                        "id": result["id"],
                        "score": result["score"],
                        "payload": result["payload"],
                    }
                )
            rankings.append(ranking)

        return rankings, debug, best_scores

    @staticmethod
    def _bm25_results(
        payload_records: list[dict[str, Any]],
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        documents = [
            {
                "id": record["id"],
                "payload": record["payload"],
                "text": payload_to_search_text(record["payload"]),
            }
            for record in payload_records
        ]
        return BM25Index(documents).search(query, top_k=top_k)

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

