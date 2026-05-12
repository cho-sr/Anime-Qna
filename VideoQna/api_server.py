from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pipeline import BASE_DIR, DEFAULT_DATA_DIR, load_env
from rag_engine import HybridRAGEngine, RetrievalConfig
from vector_store import QdrantSummaryStore


load_env()

app = FastAPI(title="VideoQna Hybrid RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_HTML = BASE_DIR / "index.html"


def default_qdrant_path() -> Path:
    import os

    return Path(os.getenv("QDRANT_PATH", str(DEFAULT_DATA_DIR / "qdrant"))).expanduser().resolve()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collection: str = "video_qna"
    top_k: int = Field(default=5, ge=1, le=20)
    dense_top_k: int = Field(default=20, ge=1, le=100)
    bm25_top_k: int = Field(default=20, ge=1, le=100)
    dense_workers: int = Field(default=3, ge=1, le=8)


class Source(BaseModel):
    rank: int
    id: str
    rrf_score: float
    dense_score: float
    bm25_score: float
    shot_id: int | None = None
    timestamp: str
    shot_start_sec: float
    shot_end_sec: float
    keyframe_timestamp_sec: float | None = None
    image_path: str | None = None
    summary: str | None = None
    action: list[Any] = Field(default_factory=list)
    context: str = ""
    emotion: list[Any] = Field(default_factory=list)
    people: list[Any] = Field(default_factory=list)
    objects: list[Any] = Field(default_factory=list)
    places: list[Any] = Field(default_factory=list)
    visual_keywords: list[Any] = Field(default_factory=list)
    dialogue_keywords: list[Any] = Field(default_factory=list)
    search_text: str = ""
    frame_description: str = ""
    subtitles: list[dict[str, Any]] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    expanded_queries: list[str]
    keywords: list[str] = []
    sources: list[Source]
    retrieval_debug: dict[str, Any]


@lru_cache(maxsize=1)
def get_store() -> QdrantSummaryStore:
    return QdrantSummaryStore(default_qdrant_path())


@lru_cache(maxsize=1)
def get_engine() -> HybridRAGEngine:
    return HybridRAGEngine.from_env(default_qdrant_path(), store=get_store())


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "qdrant_path": str(default_qdrant_path()),
        "base_dir": str(BASE_DIR),
    }


@app.get("/", include_in_schema=False)
async def index():
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(INDEX_HTML)


@app.get("/index.html", include_in_schema=False)
async def index_html():
    return await index()


@app.get("/media/keyframe")
async def keyframe_image(path: str = Query(..., min_length=1)):
    image_path = Path(path).expanduser().resolve()
    allowed_root = DEFAULT_DATA_DIR.resolve()
    if allowed_root not in image_path.parents and image_path != allowed_root:
        raise HTTPException(status_code=403, detail="image path is outside VideoQna data directory")
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail="unsupported image type")
    return FileResponse(image_path)


@app.get("/stats/{collection}")
async def stats(collection: str):
    return get_store().collection_stats(collection)


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        engine = get_engine()
        result = engine.ask(
            question=req.question,
            collection=req.collection,
            config=RetrievalConfig(
                top_k=req.top_k,
                dense_top_k=req.dense_top_k,
                bm25_top_k=req.bm25_top_k,
                dense_workers=req.dense_workers,
            ),
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        detail = {"error": type(exc).__name__, "message": str(exc)}
        raise HTTPException(status_code=500, detail=detail) from exc
