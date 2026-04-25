"""
qa_engine.py
RAG 기반 QA 엔진
벡터 DB에서 관련 챕터를 검색하고, LLM이 답변 생성
"""

import os
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ImportError:
    from langchain.schema import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

from vector_store import AnimeVectorStore

load_dotenv()


@dataclass
class QAResult:
    """QA 결과"""
    question: str
    answer: str
    sources: List[Dict[str, Any]]      # 참조한 챕터 정보
    timestamps: List[str]              # 관련 타임스탬프


class AnimeQAEngine:
    """
    애니메이션 Q&A 엔진
    
    사용법:
        engine = AnimeQAEngine(vector_store)
        result = engine.ask("루피가 처음으로 기어 2를 사용한 장면은?", anime_title="원피스")
        print(result.answer)
    """

    SYSTEM_PROMPT = """당신은 애니메이션 전문 Q&A 어시스턴트입니다.
사용자의 질문에 대해 제공된 애니메이션 챕터 정보를 바탕으로 정확하고 친절하게 답변하세요.

답변 규칙:
1. 제공된 챕터 정보에 근거하여 답변하세요
2. 해당 장면의 타임스탬프(시간)를 언급하세요  
3. 정보가 없으면 "해당 정보를 찾을 수 없습니다"라고 솔직하게 말하세요
4. 스포일러가 포함될 수 있음을 고려하세요
5. 한국어로 답변하세요"""

    def __init__(
        self,
        vector_store: AnimeVectorStore,
        model: str = "gpt-4o-mini",
        top_k: int = 3,
        openai_api_key: Optional[str] = None
    ):
        self.store = vector_store
        self.top_k = top_k
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model=model, temperature=0.3, api_key=api_key)

    def ask(self, question: str, anime_title: str) -> QAResult:
        """질문에 대한 답변 생성"""

        # 1. 벡터 DB에서 관련 챕터 검색
        print(f"🔍 검색 중: '{question}'")
        retrieved = self.store.search(question, anime_title, top_k=self.top_k)

        if not retrieved:
            return QAResult(
                question=question,
                answer="관련 정보를 찾을 수 없습니다. 영상이 먼저 인덱싱되어야 합니다.",
                sources=[],
                timestamps=[]
            )

        # 2. 컨텍스트 구성
        context_parts = []
        timestamps = []
        for r in retrieved:
            meta = r["metadata"]
            events = json.loads(meta.get("events", "[]"))
            events_text = "\n".join([f"  - {e}" for e in events])

            context_parts.append(
                f"""[챕터 {meta['chapter_id']}] ({meta['start_time_str']} ~ {meta['end_time_str']})
개요: {meta['summary']}
이벤트:
{events_text}"""
            )
            timestamps.append(f"{meta['start_time_str']} ~ {meta['end_time_str']}")

        context = "\n\n".join(context_parts)

        # 3. LLM에게 답변 요청
        user_prompt = f"""[애니메이션: {anime_title}]

=== 관련 챕터 정보 ===
{context}

=== 질문 ===
{question}"""

        response = self.llm.invoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        return QAResult(
            question=question,
            answer=response.content,
            sources=retrieved,
            timestamps=timestamps
        )

    def ask_with_history(
        self,
        question: str,
        anime_title: str,
        chat_history: List[Dict[str, str]]
    ) -> QAResult:
        """
        대화 히스토리를 포함한 멀티턴 Q&A
        chat_history: [{"role": "user"/"assistant", "content": "..."}, ...]
        """
        retrieved = self.store.search(question, anime_title, top_k=self.top_k)

        context_parts = []
        timestamps = []
        for r in retrieved:
            meta = r["metadata"]
            events = json.loads(meta.get("events", "[]"))
            context_parts.append(
                f"[챕터 {meta['chapter_id']}] ({meta['start_time_str']}~{meta['end_time_str']})\n"
                f"개요: {meta['summary']}\n"
                f"이벤트: {', '.join(events)}"
            )
            timestamps.append(f"{meta['start_time_str']} ~ {meta['end_time_str']}")

        context = "\n\n".join(context_parts)

        messages = [SystemMessage(content=self.SYSTEM_PROMPT)]

        # 히스토리 추가
        for h in chat_history[-6:]:  # 최근 6턴
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            else:
                messages.append(AIMessage(content=h["content"]))

        # 현재 질문
        messages.append(HumanMessage(content=f"""[관련 챕터 정보]
{context}

질문: {question}"""))

        response = self.llm.invoke(messages)

        return QAResult(
            question=question,
            answer=response.content,
            sources=retrieved,
            timestamps=timestamps
        )
