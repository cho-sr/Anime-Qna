from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from difflib import SequenceMatcher
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
    use_llm_query_expansion: bool = True
    generate_answer: bool = True


@dataclass
class QueryIntent:
    answer_type: str = "general"
    relation: str = "general"
    target_terms: list[str] = field(default_factory=list)


PERSON_QUESTION_MARKERS = ["누구", "누가", "사람", "인물", "이름"]
CONFLICT_MARKERS = [
    "싸우",
    "싸웠",
    "싸움",
    "전투",
    "공격",
    "방어",
    "대결",
    "맞서",
    "상대",
    "제압",
    "충돌",
    "다투",
]
CONFLICT_EXPANSION_TERMS = [
    "싸움",
    "싸우다",
    "전투",
    "공격",
    "방어",
    "대결",
    "상대",
    "제압",
    "충돌",
    "무기",
]
DESCRIPTIVE_PERSON_MARKERS = [
    "인물",
    "사람",
    "남자",
    "남성",
    "여자",
    "여성",
    "소녀",
    "소년",
    "아이",
    "검",
    "칼",
    "무기",
    "옷",
    "머리",
    "흰색",
    "검은",
    "붉은",
    "피",
]
GENERIC_PERSON_LABELS = {
    "인물",
    "사람",
    "남자",
    "남성",
    "여자",
    "여성",
    "소녀",
    "소년",
    "아이",
    "아기",
    "한 사람",
    "다른 인물",
}

QUERY_IDENTITY_STOPWORDS = {
    "나오는",
    "나온",
    "나와",
    "등장",
    "등장하는",
    "장면",
    "부분",
    "영상",
    "사람",
    "인물",
    "남자",
    "여자",
    "남성",
    "여성",
    "아이",
    "누구",
    "누가",
    "누구야",
    "어디",
    "어디야",
    "언제",
    "무엇",
    "뭐",
    "뭐야",
    "어떤",
    "검색",
    "찾아",
    "보여",
    "알려",
    "전투",
    "전투하",
    "싸움",
    "싸우",
    "싸우는",
    "싸운",
    "공격",
    "공격하",
    "방어",
    "대결",
    "충돌",
    "제압",
    "상대",
}

KOREAN_PARTICLE_SUFFIXES = [
    "에게서",
    "에게",
    "한테",
    "으로",
    "에서",
    "하고",
    "처럼",
    "까지",
    "부터",
    "보다",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "와",
    "과",
    "도",
    "로",
]

HANGUL_INITIALS = [
    "g",
    "kk",
    "n",
    "d",
    "tt",
    "r",
    "m",
    "b",
    "pp",
    "s",
    "ss",
    "",
    "j",
    "jj",
    "ch",
    "k",
    "t",
    "p",
    "h",
]
HANGUL_VOWELS = [
    "a",
    "ae",
    "ya",
    "yae",
    "eo",
    "e",
    "yeo",
    "ye",
    "o",
    "wa",
    "wae",
    "oe",
    "yo",
    "u",
    "wo",
    "we",
    "wi",
    "yu",
    "u",
    "ui",
    "i",
]
HANGUL_FINALS = [
    "",
    "k",
    "k",
    "ks",
    "n",
    "nj",
    "nh",
    "t",
    "l",
    "lk",
    "lm",
    "lb",
    "ls",
    "lt",
    "lp",
    "lh",
    "m",
    "p",
    "ps",
    "t",
    "t",
    "ng",
    "t",
    "t",
    "k",
    "t",
    "p",
    "t",
]


def unique_nonempty(items: list[str], limit: int | None = None) -> list[str]:
    unique = []
    seen = set()
    for item in items:
        value = ensure_str(item).strip()
        key = " ".join(value.split()).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(value)
        if limit is not None and len(unique) >= limit:
            break
    return unique


def strip_korean_particle(term: str) -> str:
    value = term.strip()
    for suffix in KOREAN_PARTICLE_SUFFIXES:
        if value.endswith(suffix) and len(value) > len(suffix) + 1:
            return value[: -len(suffix)]
    return value


def extract_query_identity_terms(question: str) -> list[str]:
    terms: list[str] = []
    for match in re.finditer(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9()_.-]*", question):
        raw = match.group(0).strip("()_.-")
        if not raw:
            continue
        stripped = strip_korean_particle(raw)
        candidates = [stripped] if stripped != raw else [raw]
        for term in candidates:
            normalized = term.strip("()_.-")
            key = normalized.casefold()
            if key in QUERY_IDENTITY_STOPWORDS or normalized in QUERY_IDENTITY_STOPWORDS:
                continue
            has_korean = bool(re.search(r"[가-힣]", normalized))
            min_len = 3 if has_korean else 4
            if len(normalized) < min_len:
                continue
            terms.append(normalized)
    return unique_nonempty(terms, limit=8)


def normalize_identity_label(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", value.casefold())


def hangul_to_rough_latin(value: str) -> str:
    parts: list[str] = []
    for char in value:
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:
            syllable = code - 0xAC00
            initial = syllable // 588
            vowel = (syllable % 588) // 28
            final = syllable % 28
            parts.append(HANGUL_INITIALS[initial])
            parts.append(HANGUL_VOWELS[vowel])
            parts.append(HANGUL_FINALS[final])
        elif char.isascii() and char.isalnum():
            parts.append(char.lower())
    return "".join(parts)


def identity_label_match_type(term: str, label: str) -> str:
    term_key = normalize_identity_label(term)
    label_key = normalize_identity_label(label)
    if not term_key or not label_key:
        return ""
    if term_key == label_key:
        return "exact"
    if len(term_key) >= 3 and term_key in label_key:
        return "contains"

    term_latin = hangul_to_rough_latin(term)
    label_latin = hangul_to_rough_latin(label)
    if len(term_latin) >= 4 and len(label_latin) >= 4:
        if term_latin in label_latin or label_latin in term_latin:
            return "transliteration"
        ratio = SequenceMatcher(None, term_latin, label_latin).ratio()
        if ratio >= 0.82:
            return "transliteration"
    return ""


def infer_query_intent(question: str) -> QueryIntent:
    text = question.strip()
    answer_type = "person" if any(marker in text for marker in PERSON_QUESTION_MARKERS) else "general"
    relation = "conflict" if any(marker in text for marker in CONFLICT_MARKERS) else "general"
    target_terms = extract_relation_target_terms(text) if relation == "conflict" else []
    return QueryIntent(answer_type=answer_type, relation=relation, target_terms=target_terms)


def extract_relation_target_terms(question: str) -> list[str]:
    patterns = [
        r"(.{1,40}?)(?:와|과|랑|하고)\s*(?:싸웠|싸운|싸우|대결|전투|맞서|다툰|붙은)",
        r"(.{1,40}?)(?:을|를)\s*(?:공격|제압|상대|공격한|공격했던)",
    ]
    targets: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, question)
        if not match:
            continue
        raw = normalize_target_text(match.group(1))
        if raw:
            targets.append(raw)
    return unique_nonempty(targets, limit=4)


def normalize_target_text(value: str) -> str:
    text = re.sub(r"[\"'“”‘’?!,.。]+", " ", value)
    text = re.sub(r"\b(그|저|이)\b", " ", text)
    text = " ".join(text.split()).strip()
    if not text:
        return ""
    return text.split()[-1]


def local_query_expansion(question: str, intent: QueryIntent | None = None) -> dict[str, list[str]]:
    text = question.strip()
    if not text:
        return {"expanded_queries": [], "keywords": []}

    intent = intent or infer_query_intent(text)
    expanded_queries: list[str] = []
    keywords: list[str] = []

    if intent.answer_type == "person":
        keywords.extend(["인물", "사람", "이름", "남자", "여자", "등장인물"])

    if intent.relation == "conflict":
        keywords.extend(CONFLICT_EXPANSION_TERMS)
        expanded_queries.extend(
            [
                "싸움 전투 공격 방어 대결 상대 누구",
                "공격 방어 무기를 든 사람",
                "전투 장면 싸운 사람 누구",
            ]
        )

    keywords.extend(intent.target_terms)

    return {
        "expanded_queries": unique_nonempty(expanded_queries, limit=4),
        "keywords": unique_nonempty(keywords, limit=24),
    }


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
        " ".join(
            " ".join(ensure_str_list(candidate.get("names")) + [ensure_str(candidate.get("evidence"))])
            for candidate in payload.get("character_candidates") or []
            if isinstance(candidate, dict)
        ),
        ensure_str(payload.get("frame_description")),
        " ".join(subtitle_texts),
    ]
    return "\n".join(part for part in fields if part)


def payload_conflict_score(payload: dict[str, Any]) -> float:
    text = payload_to_search_text(payload).casefold()
    action_score = sum(1 for marker in CONFLICT_MARKERS if marker.casefold() in text)
    weapon_markers = ["검을 들", "검을 든", "칼을 들", "칼을 든", "무기를 들", "낫을 들", "sword", "weapon"]
    weapon_score = sum(1 for marker in weapon_markers if marker.casefold() in text)
    return float(action_score * 2 + weapon_score)


def payload_target_score(payload: dict[str, Any], target_terms: list[str]) -> float:
    if not target_terms:
        return 0.0
    text = payload_to_search_text(payload).casefold()
    return float(sum(1 for term in target_terms if term.casefold() in text))


def payload_identity_score(payload: dict[str, Any], identity_terms: list[str]) -> float:
    matches = payload_identity_matches(payload, identity_terms)
    score = 0.0
    weights = {"exact": 8.0, "contains": 4.0, "transliteration": 3.0}
    for match in matches:
        score += weights.get(ensure_str(match.get("match_type")), 0.0)
    return score


def payload_identity_matches(
    payload: dict[str, Any],
    identity_terms: list[str],
) -> list[dict[str, str]]:
    labels: list[str] = []
    labels.extend(ensure_str_list(payload.get("people")))
    labels.extend(ensure_str_list(payload.get("dialogue_keywords")))
    for candidate in payload.get("character_candidates") or []:
        if isinstance(candidate, dict):
            labels.extend(ensure_str_list(candidate.get("names")))

    matches: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for term in identity_terms:
        for label in labels:
            label_text = ensure_str(label).strip()
            match_type = identity_label_match_type(term, label_text)
            if not match_type:
                continue
            key = (term.casefold(), label_text.casefold(), match_type)
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                {
                    "query_name": term,
                    "stored_label": label_text,
                    "match_type": match_type,
                }
            )
    return matches


def event_time_range(payloads: list[dict[str, Any]]) -> str:
    if not payloads:
        return "시간 정보 없음"
    start = min(float(payload.get("shot_start_sec") or 0.0) for payload in payloads)
    end = max(float(payload.get("shot_end_sec") or 0.0) for payload in payloads)
    return f"{seconds_to_timestamp(start)} ~ {seconds_to_timestamp(end)}"


def payload_participant_labels(payload: dict[str, Any]) -> list[dict[str, Any]]:
    labels = []
    for label in ensure_str_list(payload.get("people")):
        labels.append({"label": label, "source": "people"})
    for candidate in payload.get("character_candidates") or []:
        if not isinstance(candidate, dict):
            continue
        for label in ensure_str_list(candidate.get("names")):
            labels.append(
                {
                    "label": label,
                    "source": "character_candidates",
                    "evidence": ensure_str(candidate.get("evidence")),
                    "confidence": ensure_str(candidate.get("confidence")),
                }
            )
    return labels


def is_descriptive_person_label(label: str) -> bool:
    normalized = " ".join(label.split())
    if not normalized:
        return True
    if normalized in GENERIC_PERSON_LABELS:
        return True
    if normalized.casefold().startswith("unknown"):
        return True
    return any(marker in normalized for marker in DESCRIPTIVE_PERSON_MARKERS)


def participant_evidence_level(label: str, payloads: list[dict[str, Any]]) -> str:
    if is_descriptive_person_label(label):
        return "descriptive"

    label_key = label.casefold()
    for payload in payloads:
        subtitles = payload.get("shot_subtitles") or []
        subtitle_text = " ".join(
            ensure_str(item.get("text") if isinstance(item, dict) else item)
            for item in subtitles
        ).casefold()
        if label_key and label_key in subtitle_text:
            return "named"
    return "unverified_name"


def participant_role_score(participant: dict[str, Any]) -> float:
    label = ensure_str(participant.get("label"))
    score = 0.0
    if any(marker in label for marker in ["검을", "검 든", "검을 든", "검을 들", "칼", "무기", "낫"]):
        score += 30.0
    if any(marker in label for marker in ["공격", "방어", "상대", "대결"]):
        score += 20.0
    if any(marker in label for marker in ["남자", "남성", "사람"]):
        score += 5.0
    if "character_candidates" in participant.get("sources", []):
        score += 3.0
    if any(marker in label for marker in ["달려가는", "쓰러진", "누워"]):
        score -= 5.0
    return score


def event_participants(payloads: list[dict[str, Any]], target_terms: list[str]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    target_keys = [term.casefold() for term in target_terms]
    for payload in payloads:
        for item in payload_participant_labels(payload):
            label = " ".join(ensure_str(item.get("label")).split())
            if not label or label in GENERIC_PERSON_LABELS:
                continue
            key = label.casefold()
            if key.startswith("unknown"):
                continue
            if key not in merged:
                merged[key] = {
                    "label": label,
                    "evidence_level": participant_evidence_level(label, payloads),
                    "sources": [],
                    "is_query_target": any(target in key or key in target for target in target_keys),
                }
            source = ensure_str(item.get("source"))
            if source and source not in merged[key]["sources"]:
                merged[key]["sources"].append(source)

    ordered = list(merged.values())
    ordered.sort(
        key=lambda item: (
            item["is_query_target"],
            -participant_role_score(item),
            {"named": 0, "descriptive": 1, "unverified_name": 2}.get(item["evidence_level"], 3),
            len(item["label"]),
        )
    )
    return ordered[:12]


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
        "character_candidates": payload.get("character_candidates") or [],
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

        total_start = time.perf_counter()
        timings: dict[str, float] = {}

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

        intent = infer_query_intent(question)
        identity_terms = extract_query_identity_terms(question)
        local_expanded = local_query_expansion(question, intent)
        expansion_warning = None
        expansion_start = time.perf_counter()
        if config.use_llm_query_expansion:
            try:
                expanded = self.llm.expand_query(question)
            except Exception as exc:
                expanded = {"expanded_queries": [], "keywords": []}
                expansion_warning = f"query expansion failed, used local fallback expansions: {exc}"
        else:
            expanded = {"expanded_queries": [], "keywords": []}
        timings["query_expansion"] = time.perf_counter() - expansion_start

        expanded_queries = unique_nonempty(
            [
                *[
                    item
                    for item in expanded.get("expanded_queries", [])
                    if isinstance(item, str) and item.strip()
                ],
                *local_expanded["expanded_queries"],
            ],
            limit=6,
        )
        keywords = unique_nonempty(
            [
                *[
                    item
                    for item in expanded.get("keywords", [])
                    if isinstance(item, str) and item.strip()
                ],
                *local_expanded["keywords"],
            ],
            limit=24,
        )
        search_queries = [question] + [
            query for query in expanded_queries if query.strip() != question.strip()
        ]

        dense_start = time.perf_counter()
        dense_rankings, dense_debug, best_dense_scores = self._dense_rankings(
            collection=collection,
            queries=search_queries,
            limit=config.dense_top_k,
            workers=config.dense_workers,
        )
        timings["dense_retrieval"] = time.perf_counter() - dense_start

        bm25_start = time.perf_counter()
        payload_records, bm25_index = self._collection_payloads_and_bm25(collection)
        bm25_query = " ".join([question, *identity_terms, *expanded_queries, *keywords])
        bm25_results = bm25_index.search(bm25_query, top_k=config.bm25_top_k)
        best_bm25_scores = {item["id"]: float(item["score"]) for item in bm25_results}
        timings["bm25_retrieval"] = time.perf_counter() - bm25_start

        fusion_start = time.perf_counter()
        rrf_scores = self._rrf([*dense_rankings, [item["id"] for item in bm25_results]], config.rrf_k)
        payload_by_id = {record["id"]: record["payload"] for record in payload_records}
        for item in dense_debug:
            payload_by_id.setdefault(item["id"], item["payload"])
            item["payload"] = payload_by_id.get(item["id"], item["payload"])
        identity_scores = {
            point_id: payload_identity_score(payload, identity_terms)
            for point_id, payload in payload_by_id.items()
            if identity_terms
        }

        event_results = self._rank_events(
            intent=intent,
            payload_records=payload_records,
            rrf_scores=rrf_scores,
            dense_scores=best_dense_scores,
            bm25_scores=best_bm25_scores,
        )
        point_event_context = {
            point_id: self._event_source_context(event)
            for event in event_results
            for point_id in event["point_ids"]
        }
        event_ranked_ids = unique_nonempty(
            [
                point_id
                for event in event_results[:2]
                for point_id in event["point_ids"]
                if point_id in payload_by_id
            ],
            limit=config.top_k,
        )

        ranked_ids = [
            point_id
            for point_id, _score in sorted(
                rrf_scores.items(),
                key=lambda item: (
                    identity_scores.get(item[0], 0.0) > 0.0,
                    identity_scores.get(item[0], 0.0),
                    item[1],
                ),
                reverse=True,
            )
            if point_id in payload_by_id
        ]
        if event_ranked_ids:
            ranked_ids = unique_nonempty([*event_ranked_ids, *ranked_ids], limit=config.top_k)
        else:
            ranked_ids = ranked_ids[: config.top_k]

        sources = []
        for rank, point_id in enumerate(ranked_ids, start=1):
            source = payload_to_source(
                point_id=point_id,
                payload=payload_by_id[point_id],
                rank=rank,
                rrf_score=rrf_scores.get(point_id, 0.0),
                dense_score=best_dense_scores.get(point_id, 0.0),
                bm25_score=best_bm25_scores.get(point_id, 0.0),
            )
            if point_id in point_event_context:
                source.update(point_event_context[point_id])
            source["query_identity_matches"] = payload_identity_matches(
                payload_by_id[point_id],
                identity_terms,
            )
            sources.append(source)
        self._attach_nearby_identity_matches(sources)
        timings["fusion_and_ranking"] = time.perf_counter() - fusion_start

        answer_start = time.perf_counter()
        if config.generate_answer:
            answer = self.llm.answer_question(question, sources)
        else:
            answer = self.llm.fallback_answer(question, sources)
        timings["answer_generation"] = time.perf_counter() - answer_start
        timings["total"] = time.perf_counter() - total_start
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
                    "use_llm_query_expansion": config.use_llm_query_expansion,
                    "generate_answer": config.generate_answer,
                },
                "timings_sec": {key: round(value, 3) for key, value in timings.items()},
                "query_intent": {
                    "answer_type": intent.answer_type,
                    "relation": intent.relation,
                    "target_terms": intent.target_terms,
                    "identity_terms": identity_terms,
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
                "event_results": [
                    {
                        "event_id": event["event_id"],
                        "event_type": event["event_type"],
                        "time_range": event["time_range"],
                        "score": event["score"],
                        "target_supported": event["target_supported"],
                        "participants": event["participants"],
                        "shot_ids": event["shot_ids"],
                    }
                    for event in event_results
                ],
                "warnings": warnings,
            },
        }

    @staticmethod
    def _source_interval(source: dict[str, Any]) -> tuple[float, float]:
        start = source.get("event_start_sec")
        end = source.get("event_end_sec")
        if start is None or end is None:
            start = source.get("shot_start_sec")
            end = source.get("shot_end_sec")
        return float(start or 0.0), float(end or start or 0.0)

    @classmethod
    def _attach_nearby_identity_matches(
        cls,
        sources: list[dict[str, Any]],
        max_gap_sec: float = 8.0,
    ) -> None:
        matched_sources = [
            source
            for source in sources
            if source.get("query_identity_matches")
        ]
        for source in sources:
            if source.get("query_identity_matches"):
                source["nearby_query_identity_matches"] = []
                continue

            start, end = cls._source_interval(source)
            nearby: list[dict[str, Any]] = []
            seen: set[tuple[str, str, str, int]] = set()
            for matched in matched_sources:
                matched_start, matched_end = cls._source_interval(matched)
                gap = max(matched_start - end, start - matched_end, 0.0)
                if gap > max_gap_sec:
                    continue
                for match in matched.get("query_identity_matches") or []:
                    key = (
                        ensure_str(match.get("query_name")).casefold(),
                        ensure_str(match.get("stored_label")).casefold(),
                        ensure_str(match.get("match_type")),
                        int(matched.get("shot_id") or -1),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    nearby.append(
                        {
                            **match,
                            "nearby_shot_id": matched.get("shot_id"),
                            "nearby_timestamp": matched.get("timestamp"),
                            "gap_sec": round(gap, 3),
                        }
                    )
            source["nearby_query_identity_matches"] = nearby[:8]

    def _rank_events(
        self,
        intent: QueryIntent,
        payload_records: list[dict[str, Any]],
        rrf_scores: dict[str, float],
        dense_scores: dict[str, float],
        bm25_scores: dict[str, float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if intent.relation != "conflict":
            return []

        ordered_records = sorted(
            [
                {
                    "index": index,
                    "id": record["id"],
                    "payload": record["payload"],
                }
                for index, record in enumerate(payload_records)
            ],
            key=lambda item: (
                ensure_str(item["payload"].get("video_path")),
                float(item["payload"].get("shot_start_sec") or 0.0),
                int(item["payload"].get("shot_id") or 0),
            ),
        )
        seeds = []
        for index, item in enumerate(ordered_records):
            point_id = item["id"]
            payload = item["payload"]
            conflict_score = payload_conflict_score(payload)
            if conflict_score <= 0:
                continue

            retrieval_score = (
                rrf_scores.get(point_id, 0.0) * 100.0
                + dense_scores.get(point_id, 0.0)
                + bm25_scores.get(point_id, 0.0) * 0.05
            )
            target_score = payload_target_score(payload, intent.target_terms)
            if retrieval_score <= 0 and target_score <= 0:
                continue

            seeds.append(
                {
                    "index": index,
                    "score": retrieval_score + conflict_score * 8.0 + target_score * 5.0,
                }
            )

        seeds.sort(key=lambda item: item["score"], reverse=True)
        events = []
        seen_windows: set[tuple[str, ...]] = set()
        for seed in seeds[:24]:
            window = self._event_window(ordered_records, seed["index"], intent)
            point_ids = tuple(item["id"] for item in window)
            if not point_ids or point_ids in seen_windows:
                continue
            seen_windows.add(point_ids)

            payloads = [item["payload"] for item in window]
            participants = event_participants(payloads, intent.target_terms)
            target_supported = any(payload_target_score(payload, intent.target_terms) > 0 for payload in payloads)
            event_score = seed["score"] + sum(
                payload_conflict_score(payload) * 2.0
                + payload_target_score(payload, intent.target_terms)
                for payload in payloads
            )
            role_boost = max(
                (
                    participant_role_score(participant)
                    for participant in participants
                    if not participant.get("is_query_target")
                ),
                default=0.0,
            )
            event_score += role_boost * 1.2
            if intent.target_terms and target_supported:
                event_score += 120.0
            shot_ids = [
                int(payload.get("shot_id"))
                for payload in payloads
                if payload.get("shot_id") is not None
            ]
            events.append(
                {
                    "event_id": f"event_{shot_ids[0] if shot_ids else len(events)}_{shot_ids[-1] if shot_ids else len(events)}",
                    "event_type": "conflict",
                    "time_range": event_time_range(payloads),
                    "start_sec": min(float(payload.get("shot_start_sec") or 0.0) for payload in payloads),
                    "end_sec": max(float(payload.get("shot_end_sec") or 0.0) for payload in payloads),
                    "score": event_score,
                    "target_supported": target_supported,
                    "participants": participants,
                    "point_ids": list(point_ids),
                    "shot_ids": shot_ids,
                }
            )

        events.sort(key=lambda item: item["score"], reverse=True)
        filtered = []
        used_ids: set[str] = set()
        for event in events:
            overlap = len(set(event["point_ids"]) & used_ids)
            if overlap >= max(1, len(event["point_ids"]) // 2):
                continue
            filtered.append(event)
            used_ids.update(event["point_ids"])
            if len(filtered) >= limit:
                break
        return filtered

    def _event_window(
        self,
        ordered_records: list[dict[str, Any]],
        seed_index: int,
        intent: QueryIntent,
        max_neighbors: int = 3,
        max_gap_sec: float = 8.0,
    ) -> list[dict[str, Any]]:
        seed = ordered_records[seed_index]
        seed_payload = seed["payload"]
        seed_video = ensure_str(seed_payload.get("video_path"))
        seed_start = float(seed_payload.get("shot_start_sec") or 0.0)
        seed_end = float(seed_payload.get("shot_end_sec") or seed_start)
        indexes = {seed_index}

        for direction in (-1, 1):
            for step in range(1, max_neighbors + 1):
                index = seed_index + direction * step
                if index < 0 or index >= len(ordered_records):
                    break
                item = ordered_records[index]
                payload = item["payload"]
                if ensure_str(payload.get("video_path")) != seed_video:
                    break
                start = float(payload.get("shot_start_sec") or 0.0)
                end = float(payload.get("shot_end_sec") or start)
                gap = seed_start - end if direction < 0 else start - seed_end
                if gap > max_gap_sec:
                    break
                has_event_evidence = (
                    payload_conflict_score(payload) > 0
                    or payload_target_score(payload, intent.target_terms) > 0
                )
                if step == 1 or has_event_evidence:
                    indexes.add(index)

        return [ordered_records[index] for index in sorted(indexes)]

    @staticmethod
    def _event_source_context(event: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_id": event["event_id"],
            "event_type": event["event_type"],
            "event_time_range": event["time_range"],
            "event_start_sec": event.get("start_sec"),
            "event_end_sec": event.get("end_sec"),
            "event_participants": event["participants"],
            "event_evidence_shots": event["shot_ids"],
            "event_target_supported": event["target_supported"],
            "event_evidence_level": "target_linked" if event["target_supported"] else "event_only",
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
            if getattr(self.embedder, "is_local", False) and hasattr(self.embedder, "embed_texts"):
                formatted_queries = [self.embedder.format_query(query) for query in queries]
                vectors = self.embedder.embed_texts(formatted_queries)
                for index, (query, vector) in enumerate(zip(queries, vectors)):
                    ordered_vectors[index] = (query, vector)
                return [item for item in ordered_vectors if item is not None]

            workers_count = max(1, int(workers or 1))
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
            results = self.store.dense_search(
                collection,
                vector=vector,
                limit=limit,
                with_payload=False,
            )
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
