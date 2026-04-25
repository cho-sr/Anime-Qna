"""
vector_store.py
벡터 DB 저장/검색 모듈 (ChromaDB + OpenAI Embeddings)

저장 구조:
  - 각 문서 = 1개 챕터의 이벤트 + 개요
  - metadata: chapter_id, start_time, end_time, event_list
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from collections import defaultdict

from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    from langchain.schema import HumanMessage, SystemMessage

load_dotenv()


@dataclass
class Chapter:
    """챕터 단위 데이터"""
    chapter_id: int
    start_time: float
    end_time: float
    subtitles: List[str]          # 해당 챕터의 자막 리스트
    summary: str = ""             # LLM이 생성한 개요
    events: List[str] = field(default_factory=list)   # 주요 이벤트 리스트


class ChapterBuilder:
    """
    자막 청크들을 챕터 단위로 묶고,
    LLM을 통해 각 챕터의 이벤트 목록과 개요를 생성
    """

    def __init__(self, chapter_duration: float = 60.0, openai_api_key: Optional[str] = None):
        self.chapter_duration = chapter_duration
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key
        )

    def build_chapters(self, subtitle_chunks) -> List[Chapter]:
        """자막 청크 -> 챕터 리스트 생성"""
        # 챕터별로 자막 그룹핑
        chapter_map = defaultdict(list)
        for chunk in subtitle_chunks:
            chapter_map[chunk.chapter_id].append(chunk)

        chapters = []
        for chapter_id in sorted(chapter_map.keys()):
            chunks = chapter_map[chapter_id]
            subtitles = [c.text for c in chunks]
            start_time = chunks[0].start_time
            end_time = chunks[-1].end_time

            chapter = Chapter(
                chapter_id=chapter_id,
                start_time=start_time,
                end_time=end_time,
                subtitles=subtitles
            )
            chapters.append(chapter)

        print(f"📚 총 {len(chapters)}개 챕터 구성 완료")
        return chapters

    def generate_chapter_summary(self, chapter: Chapter) -> Chapter:
        """LLM으로 챕터 개요 + 이벤트 목록 생성"""
        subtitle_text = "\n".join(chapter.subtitles[:50])  # 최대 50줄

        system_prompt = """당신은 애니메이션 내용을 분석하는 전문가입니다.
주어진 자막을 바탕으로 JSON 형식으로 다음을 추출하세요:
1. summary: 이 챕터의 핵심 내용 요약 (2-3문장)
2. events: 주요 사건/이벤트 목록 (최대 5개, 각 1문장)

반드시 JSON만 반환하세요:
{"summary": "...", "events": ["이벤트1", "이벤트2", ...]}"""

        user_prompt = f"""[챕터 {chapter.chapter_id} 자막 ({chapter.start_time:.0f}초 ~ {chapter.end_time:.0f}초)]
{subtitle_text}"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            # JSON 파싱
            raw = response.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            chapter.summary = data.get("summary", "")
            chapter.events = data.get("events", [])
        except Exception as e:
            print(f"⚠️  챕터 {chapter.chapter_id} 요약 실패: {e}")
            chapter.summary = " ".join(chapter.subtitles[:3])
            chapter.events = []

        return chapter

    def process_all_chapters(self, chapters: List[Chapter]) -> List[Chapter]:
        """모든 챕터에 대해 요약 생성"""
        print("🤖 챕터별 개요/이벤트 생성 중...")
        for i, chapter in enumerate(chapters):
            print(f"  [{i+1}/{len(chapters)}] 챕터 {chapter.chapter_id} 처리 중...")
            self.generate_chapter_summary(chapter)
        print("✅ 모든 챕터 처리 완료")
        return chapters


class AnimeVectorStore:
    """
    ChromaDB 기반 벡터 저장소
    
    저장 형식:
      - document: "챕터 개요 + 이벤트 목록 + 원본 자막"
      - metadata: chapter_id, start_time, end_time, ...
    
    사용법:
        store = AnimeVectorStore(db_path="./chroma_db")
        store.index_chapters(chapters, anime_title="원피스 EP1")
        results = store.search("루피가 고무고무 열매를 먹는 장면")
    """

    def __init__(self, db_path: str = "./chroma_db", openai_api_key: Optional[str] = None):
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.db_path = db_path

        # ChromaDB 초기화
        self.client = chromadb.PersistentClient(path=db_path)

        # OpenAI 임베딩 함수
        self.embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )

    def get_or_create_collection(self, anime_title: str):
        """애니메이션별 컬렉션 생성/조회"""
        # 컬렉션 이름: 특수문자 제거
        collection_name = "".join(c if c.isalnum() or c == "_" else "_" for c in anime_title)
        collection_name = collection_name[:50]  # 최대 50자

        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
            metadata={"anime_title": anime_title}
        )

    def index_chapters(self, chapters: List[Chapter], anime_title: str):
        """챕터 데이터를 벡터 DB에 저장"""
        collection = self.get_or_create_collection(anime_title)

        documents = []
        metadatas = []
        ids = []

        for chapter in chapters:
            # 검색용 문서 구성: 개요 + 이벤트 + 자막 샘플
            subtitle_sample = " | ".join(chapter.subtitles[:20])
            events_text = "\n".join([f"- {e}" for e in chapter.events])

            document = f"""[챕터 {chapter.chapter_id}] {chapter.start_time:.0f}초 ~ {chapter.end_time:.0f}초
개요: {chapter.summary}
주요 이벤트:
{events_text}
자막 샘플: {subtitle_sample}"""

            metadata = {
                "chapter_id": chapter.chapter_id,
                "start_time": chapter.start_time,
                "end_time": chapter.end_time,
                "start_time_str": self._sec_to_str(chapter.start_time),
                "end_time_str": self._sec_to_str(chapter.end_time),
                "summary": chapter.summary,
                "events": json.dumps(chapter.events, ensure_ascii=False),
                "anime_title": anime_title
            }

            doc_id = f"{anime_title}_ch{chapter.chapter_id}"
            documents.append(document)
            metadatas.append(metadata)
            ids.append(doc_id)

        # 배치 저장
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ {len(chapters)}개 챕터를 벡터 DB에 저장 완료 (컬렉션: {anime_title})")

    def search(
        self,
        query: str,
        anime_title: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """쿼리로 관련 챕터 검색"""
        collection = self.get_or_create_collection(anime_title)

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count())
        )

        # 결과 정리
        retrieved = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            retrieved.append({
                "rank": i + 1,
                "document": doc,
                "metadata": meta,
                "score": 1 - dist   # 유사도 (0~1)
            })

        return retrieved

    def list_animes(self) -> List[str]:
        """저장된 애니메이션 목록 반환"""
        collections = self.client.list_collections()
        return [c.name for c in collections]

    @staticmethod
    def _sec_to_str(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
