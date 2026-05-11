from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any


class QdrantSummaryStore:
    def __init__(self, qdrant_path: str | Path):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("qdrant-client is required.") from exc

        self.qdrant_path = Path(qdrant_path)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.qdrant_path))

    def upsert_scene(
        self,
        collection: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> str:
        self._ensure_collection(collection, vector_size=len(vector))
        try:
            from qdrant_client.models import PointStruct
        except ImportError as exc:
            raise RuntimeError("qdrant-client models are unavailable.") from exc

        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{payload['video_path']}:{payload['shot_id']}:{payload['shot_start_sec']}",
            )
        )
        self.client.upsert(
            collection_name=collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return point_id

    def collection_stats(self, collection: str) -> dict[str, Any]:
        if not self._collection_exists(collection):
            return {
                "collection": collection,
                "exists": False,
                "qdrant_path": str(self.qdrant_path),
                "points_count": 0,
            }
        count = self.client.count(collection_name=collection, exact=True).count
        return {
            "collection": collection,
            "exists": True,
            "qdrant_path": str(self.qdrant_path),
            "points_count": count,
        }

    def _ensure_collection(self, collection: str, vector_size: int) -> None:
        try:
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise RuntimeError("qdrant-client models are unavailable.") from exc

        if not self._collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            return

        current_size = self._collection_vector_size(collection)
        if current_size and current_size != vector_size:
            raise RuntimeError(
                f"Collection '{collection}' expects vector size {current_size}, "
                f"but embedding returned {vector_size}. Use another collection or Qdrant path."
            )

    def _collection_exists(self, collection: str) -> bool:
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(collection))
        try:
            self.client.get_collection(collection)
            return True
        except Exception:
            return False

    def _collection_vector_size(self, collection: str) -> int | None:
        info = self.client.get_collection(collection)
        vectors = info.config.params.vectors
        if hasattr(vectors, "size"):
            return int(vectors.size)
        if isinstance(vectors, dict):
            first = next(iter(vectors.values()), None)
            if first and hasattr(first, "size"):
                return int(first.size)
        return None

