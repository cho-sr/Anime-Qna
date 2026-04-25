"""
api_server.py
FastAPI 백엔드 서버

엔드포인트:
  POST /index     - 영상 인덱싱 (업로드 + 처리)
  POST /ask       - Q&A 질문
  GET  /animes    - 등록된 애니메이션 목록
  DELETE /anime   - 애니메이션 삭제
"""

import os
import shutil
import asyncio
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from video_processor import VideoProcessor
from subtitle_extractor import SubtitleExtractor
from vector_store import AnimeVectorStore, ChapterBuilder
from qa_engine import AnimeQAEngine

load_dotenv()

app = FastAPI(title="Anime RAG Q&A System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 인스턴스
DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
vector_store = AnimeVectorStore(db_path=DB_PATH)
qa_engine = AnimeQAEngine(vector_store=vector_store)

# 인덱싱 상태 관리
indexing_status: dict = {}

UPLOAD_DIR = Path("./uploads")
FRAMES_DIR = Path("./frames")
UPLOAD_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# 요청/응답 모델
# ──────────────────────────────────────────────

class IndexRequest(BaseModel):
    anime_title: str
    subtitle_method: str = "whisper"   # "srt" | "whisper" | "ocr"
    whisper_model: str = "base"
    whisper_language: Optional[str] = None
    frame_interval: float = 2.0
    chapter_duration: float = 60.0


class AskRequest(BaseModel):
    question: str
    anime_title: str
    chat_history: List[dict] = []


class AskResponse(BaseModel):
    answer: str
    timestamps: List[str]
    sources: List[dict]


# ──────────────────────────────────────────────
# 백그라운드 인덱싱 작업
# ──────────────────────────────────────────────

async def run_indexing(
    video_path: str,
    srt_path: Optional[str],
    config: IndexRequest
):
    title = config.anime_title
    try:
        indexing_status[title] = {"status": "processing", "step": "프레임 추출 중...", "progress": 0}

        # 1. 프레임 추출
        processor = VideoProcessor(
            frame_interval=config.frame_interval,
            chapter_duration=config.chapter_duration
        )
        frames = processor.extract_frames(
            video_path,
            output_dir=str(FRAMES_DIR / title)
        )
        indexing_status[title]["step"] = "자막 추출 중..."
        indexing_status[title]["progress"] = 30

        # 2. 자막 추출
        extractor = SubtitleExtractor(
            method=config.subtitle_method,
            model_size=config.whisper_model,
            language=config.whisper_language
        )
        subtitle_chunks = extractor.extract(
            video_path=video_path,
            srt_path=srt_path,
            frames=frames if config.subtitle_method == "ocr" else None,
            chapter_duration=config.chapter_duration
        )
        indexing_status[title]["step"] = "챕터 분석 중..."
        indexing_status[title]["progress"] = 60

        # 3. 챕터 구성 + LLM 요약
        builder = ChapterBuilder(chapter_duration=config.chapter_duration)
        chapters = builder.build_chapters(subtitle_chunks)
        chapters = builder.process_all_chapters(chapters)
        indexing_status[title]["step"] = "벡터 DB 저장 중..."
        indexing_status[title]["progress"] = 85

        # 4. 벡터 DB 저장
        vector_store.index_chapters(chapters, anime_title=title)

        indexing_status[title] = {
            "status": "done",
            "step": "완료",
            "progress": 100,
            "chapter_count": len(chapters)
        }

    except Exception as e:
        indexing_status[title] = {"status": "error", "step": str(e), "progress": -1}
        raise


# ──────────────────────────────────────────────
# API 엔드포인트
# ──────────────────────────────────────────────

@app.post("/index")
async def index_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    subtitle: Optional[UploadFile] = File(None),
    anime_title: str = "anime",
    subtitle_method: str = "whisper",
    whisper_model: str = "base",
    whisper_language: Optional[str] = None,
    frame_interval: float = 2.0,
    chapter_duration: float = 60.0
):
    """영상 업로드 및 인덱싱 시작"""

    # 파일 저장
    video_path = str(UPLOAD_DIR / video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    srt_path = None
    if subtitle:
        srt_path = str(UPLOAD_DIR / subtitle.filename)
        with open(srt_path, "wb") as f:
            shutil.copyfileobj(subtitle.file, f)

    config = IndexRequest(
        anime_title=anime_title,
        subtitle_method=subtitle_method,
        whisper_model=whisper_model,
        whisper_language=whisper_language,
        frame_interval=frame_interval,
        chapter_duration=chapter_duration
    )

    indexing_status[anime_title] = {"status": "queued", "step": "대기 중...", "progress": 0}
    background_tasks.add_task(run_indexing, video_path, srt_path, config)

    return {"message": f"'{anime_title}' 인덱싱 시작됨", "status_key": anime_title}


@app.get("/index/status/{anime_title}")
async def get_index_status(anime_title: str):
    """인덱싱 진행 상태 조회"""
    if anime_title not in indexing_status:
        raise HTTPException(status_code=404, detail="해당 애니메이션 인덱싱 이력 없음")
    return indexing_status[anime_title]


@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """Q&A 질문"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력하세요")

    if req.chat_history:
        result = qa_engine.ask_with_history(
            question=req.question,
            anime_title=req.anime_title,
            chat_history=req.chat_history
        )
    else:
        result = qa_engine.ask(question=req.question, anime_title=req.anime_title)

    return AskResponse(
        answer=result.answer,
        timestamps=result.timestamps,
        sources=[
            {
                "chapter_id": s["metadata"]["chapter_id"],
                "start_time": s["metadata"]["start_time_str"],
                "end_time": s["metadata"]["end_time_str"],
                "summary": s["metadata"]["summary"],
                "score": round(s["score"], 3)
            }
            for s in result.sources
        ]
    )


@app.get("/animes")
async def list_animes():
    """등록된 애니메이션 목록"""
    return {"animes": vector_store.list_animes()}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
