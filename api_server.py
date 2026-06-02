from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from embedding import DEFAULT_LOCAL_EMBEDDING_MODEL, create_summary_embedder
from hf_clients import RAGLLMClient
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
UPLOAD_ROOT = DEFAULT_DATA_DIR / "uploads"
JOB_ROOT = DEFAULT_DATA_DIR / "index_jobs"
UPLOAD_QDRANT_ROOT = DEFAULT_DATA_DIR / "qdrant_uploads"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
JOB_LOCK = threading.Lock()
JOBS: dict[str, dict[str, Any]] = {}
WARMUP_LOCK = threading.Lock()


def default_qdrant_path() -> Path:
    import os

    return Path(os.getenv("QDRANT_PATH", str(DEFAULT_DATA_DIR / "qdrant"))).expanduser().resolve()


def qdrant_search_paths() -> list[Path]:
    paths = [default_qdrant_path()]
    upload_root = UPLOAD_QDRANT_ROOT.resolve()
    for path in sorted(DEFAULT_DATA_DIR.glob("qdrant*")):
        resolved = path.resolve()
        if path.is_dir() and resolved != upload_root and resolved not in paths:
            paths.append(resolved)
    for path in sorted(UPLOAD_QDRANT_ROOT.glob("*")):
        if path.is_dir() and path.resolve() not in paths:
            paths.append(path.resolve())
    return paths


def safe_name(value: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return value or fallback


def status_path(job_id: str) -> Path:
    return JOB_ROOT / job_id / "status.json"


def log_path(job_id: str) -> Path:
    return JOB_ROOT / job_id / "log.txt"


def write_job_status(job: dict[str, Any]) -> None:
    path = status_path(job["job_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    with JOB_LOCK:
        JOBS[job["job_id"]] = dict(job)


def read_job_status(job_id: str) -> dict[str, Any] | None:
    with JOB_LOCK:
        cached = JOBS.get(job_id)
    if cached:
        return dict(cached)
    path = status_path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_log_tail(path: Path, limit: int = 12000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", errors="replace")


def collection_exists_anywhere(collection: str) -> bool:
    for qdrant_path in qdrant_search_paths():
        try:
            if get_store_for_path(str(qdrant_path)).collection_exists(collection):
                return True
        except RuntimeError:
            continue
    return False


@lru_cache(maxsize=1)
def get_shared_embedder():
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

    return create_summary_embedder(
        backend=embedding_backend,
        model_name=embedding_model,
        token=os.getenv("HF_TOKEN", ""),
        provider=embedding_provider,
        local_device=os.getenv("LOCAL_EMBEDDING_DEVICE", "auto"),
        local_batch_size=int(os.getenv("LOCAL_EMBEDDING_BATCH_SIZE", "8")),
        local_max_length=int(os.getenv("LOCAL_EMBEDDING_MAX_LENGTH", "2048")),
    )


@lru_cache(maxsize=1)
def get_shared_llm() -> RAGLLMClient:
    hf_provider = os.getenv("HF_PROVIDER") or None
    return RAGLLMClient(
        token=os.getenv("HF_TOKEN", ""),
        model=os.getenv("HF_LLM_MODEL", "Qwen/Qwen3-8B"),
        provider=os.getenv("HF_LLM_PROVIDER") or hf_provider,
    )


def warm_collection(collection: str, qdrant_path: str | Path | None = None) -> dict[str, Any]:
    with WARMUP_LOCK:
        qdrant_path = Path(qdrant_path).resolve() if qdrant_path else Path(resolve_collection_path(collection))
        engine = get_engine_for_path(str(qdrant_path))
        timings = engine.warm_collection(collection)
        print(f"[warmup] collection={collection} qdrant_path={qdrant_path} timings={timings}", flush=True)
        return {
            "collection": collection,
            "qdrant_path": str(qdrant_path),
            "timings": timings,
        }


def run_index_job(job_id: str) -> None:
    job = read_job_status(job_id)
    if not job:
        return

    job["status"] = "running"
    job["started_at"] = datetime.now().isoformat(timespec="seconds")
    write_job_status(job)

    command = [
        sys.executable,
        str(BASE_DIR / "pipeline.py"),
        "index",
        "--video",
        job["video_path"],
        "--collection",
        job["collection"],
        "--qdrant-path",
        job["qdrant_path"],
        "--api-mode",
        "unified",
        "--api-workers",
        str(job["api_workers"]),
        "--scene-batch-size",
        str(job["scene_batch_size"]),
        "--qdrant-batch-size",
        str(job["qdrant_batch_size"]),
    ]
    language = job.get("language") or "auto"
    if language != "auto":
        command.extend(["--language", language])
    max_shots = int(job.get("max_shots") or 0)
    if max_shots > 0:
        command.extend(["--max-shots", str(max_shots)])

    job["command"] = command
    write_job_status(job)

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("HF_CHAT_TIMEOUT", "180")
    env.setdefault("HF_CHAT_MAX_RETRIES", "3")

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    log_file = log_path(job_id)
    with log_file.open("a", encoding="utf-8", errors="replace") as log:
        log.write(" ".join(command) + "\n\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=str(BASE_DIR),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=creationflags,
            )
            job["return_code"] = completed.returncode
            if completed.returncode == 0:
                resolve_collection_path.cache_clear()
                job["status"] = "warming"
                write_job_status(job)
                try:
                    job["warmup"] = warm_collection(job["collection"], job["qdrant_path"])
                except Exception as exc:
                    job["warmup_error"] = f"{type(exc).__name__}: {exc}"
                job["status"] = "done"
            else:
                job["status"] = "error"
                job["error"] = f"index process exited with code {completed.returncode}"
        except Exception as exc:
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            job["finished_at"] = datetime.now().isoformat(timespec="seconds")
            resolve_collection_path.cache_clear()
            write_job_status(job)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collection: str = "video_qna_qwen06b"
    top_k: int = Field(default=6, ge=1, le=20)
    dense_top_k: int = Field(default=40, ge=1, le=200)
    bm25_top_k: int = Field(default=60, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1, le=500)
    dense_workers: int = Field(default=3, ge=1, le=16)
    use_llm_query_expansion: bool = True
    generate_answer: bool = True


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
    character_candidates: list[Any] = Field(default_factory=list)
    search_text: str = ""
    frame_description: str = ""
    subtitles: list[dict[str, Any]] = Field(default_factory=list)
    event_id: str | None = None
    event_type: str = ""
    event_time_range: str = ""
    event_start_sec: float | None = None
    event_end_sec: float | None = None
    event_participants: list[Any] = Field(default_factory=list)
    event_evidence_shots: list[int] = Field(default_factory=list)
    event_target_supported: bool = False
    event_evidence_level: str = ""
    query_identity_matches: list[Any] = Field(default_factory=list)
    nearby_query_identity_matches: list[Any] = Field(default_factory=list)


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
    return HybridRAGEngine.from_env(
        default_qdrant_path(),
        store=get_store(),
        embedder=get_shared_embedder(),
        llm=get_shared_llm(),
    )


@lru_cache(maxsize=64)
def get_store_for_path(qdrant_path: str) -> QdrantSummaryStore:
    return QdrantSummaryStore(qdrant_path)


@lru_cache(maxsize=64)
def get_engine_for_path(qdrant_path: str) -> HybridRAGEngine:
    store = get_store_for_path(qdrant_path)
    return HybridRAGEngine.from_env(
        qdrant_path,
        store=store,
        embedder=get_shared_embedder(),
        llm=get_shared_llm(),
    )


@lru_cache(maxsize=256)
def resolve_collection_path(collection: str) -> str:
    for qdrant_path in qdrant_search_paths():
        store = get_store_for_path(str(qdrant_path))
        if store.collection_exists(collection):
            return str(qdrant_path)
    return str(default_qdrant_path())


def get_store_for_collection(collection: str) -> QdrantSummaryStore:
    return get_store_for_path(resolve_collection_path(collection))


def get_engine_for_collection(collection: str) -> HybridRAGEngine:
    return get_engine_for_path(resolve_collection_path(collection))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "qdrant_path": str(default_qdrant_path()),
        "qdrant_search_paths": [str(path) for path in qdrant_search_paths()],
        "base_dir": str(BASE_DIR),
    }


@app.post("/upload")
async def upload_video(
    video: UploadFile = File(...),
    collection: str = Form(""),
    language: str = Form("auto"),
    api_workers: int = Form(15),
    scene_batch_size: int = Form(5),
    qdrant_batch_size: int = Form(16),
    max_shots: int = Form(0),
):
    filename = Path(video.filename or "uploaded.mp4").name
    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="unsupported video file type")

    job_id = uuid.uuid4().hex[:12]
    stem = safe_name(Path(filename).stem, f"upload_{job_id}")
    collection_name = safe_name(collection, f"video_qna_{stem}_{job_id[:6]}")
    if collection_exists_anywhere(collection_name):
        raise HTTPException(status_code=409, detail=f"collection already exists: {collection_name}")

    language = (language or "auto").strip().lower()
    if language not in {"auto", "ja", "ko", "en", "zh", "es", "fr", "de"}:
        raise HTTPException(status_code=400, detail="unsupported language code")

    api_workers = max(1, min(int(api_workers or 15), 32))
    scene_batch_size = max(1, min(int(scene_batch_size or 5), 5))
    qdrant_batch_size = max(1, min(int(qdrant_batch_size or 16), 128))
    max_shots = max(0, int(max_shots or 0))

    upload_dir = UPLOAD_ROOT / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / f"{stem}{suffix}"
    with video_path.open("wb") as handle:
        while chunk := await video.read(1024 * 1024):
            handle.write(chunk)

    qdrant_path = UPLOAD_QDRANT_ROOT / collection_name
    job = {
        "job_id": job_id,
        "status": "queued",
        "collection": collection_name,
        "language": language,
        "video_path": str(video_path.resolve()),
        "qdrant_path": str(qdrant_path.resolve()),
        "log_path": str(log_path(job_id).resolve()),
        "api_workers": api_workers,
        "scene_batch_size": scene_batch_size,
        "qdrant_batch_size": qdrant_batch_size,
        "max_shots": max_shots,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_job_status(job)

    thread = threading.Thread(target=run_index_job, args=(job_id,), daemon=True)
    thread.start()

    return {**job, "log_tail": ""}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = read_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {**job, "log_tail": read_log_tail(Path(job.get("log_path") or ""))}


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
    return get_store_for_collection(collection).collection_stats(collection)


@app.post("/warmup/{collection}")
async def warmup_collection(collection: str):
    try:
        return warm_collection(collection)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        engine = get_engine_for_collection(req.collection)
        result = engine.ask(
            question=req.question,
            collection=req.collection,
            config=RetrievalConfig(
                top_k=req.top_k,
                dense_top_k=req.dense_top_k,
                bm25_top_k=req.bm25_top_k,
                rrf_k=req.rrf_k,
                dense_workers=req.dense_workers,
                use_llm_query_expansion=req.use_llm_query_expansion,
                generate_answer=req.generate_answer,
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
