from __future__ import annotations

import base64
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

from models import FrameDescription, SceneSummary, SubtitleSegment
from utils import ensure_str, ensure_str_list, extract_json_object


JSON_RETRY_PROMPT = (
    "Your previous response was empty or invalid. Return exactly one valid JSON "
    "object and nothing else. Do not include markdown, comments, or reasoning. /no_think"
)

NO_REASONING_EXTRA_BODY = {"reasoning": {"enabled": False}}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[warn] invalid {name}={value!r}; using {default}")
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[warn] invalid {name}={value!r}; using {default}")
        return default


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _message_content(response: Any) -> str:
    try:
        return _content_to_text(response.choices[0].message.content)
    except (AttributeError, IndexError, TypeError):
        pass
    if isinstance(response, dict):
        message = response.get("choices", [{}])[0].get("message", {})
        if isinstance(message, dict):
            return _content_to_text(message.get("content"))
        return _content_to_text(getattr(message, "content", ""))
    return _content_to_text(response)


def _json_retry_messages(
    messages: list[dict[str, Any]],
    invalid_content: str,
) -> list[dict[str, Any]]:
    retry_messages = list(messages)
    invalid_content = invalid_content.strip()
    if invalid_content:
        retry_messages.append({"role": "assistant", "content": invalid_content[:2000]})
    retry_messages.append({"role": "user", "content": JSON_RETRY_PROMPT})
    return retry_messages


def _content_preview(content: str, limit: int = 500) -> str:
    preview = content.strip()
    if not preview:
        return "<empty response>"
    return preview[:limit]


def _clip_text(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _clip_multiline_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _unique_str_list(value: Any, limit: int | None = None) -> list[str]:
    items = ensure_str_list(value)
    unique = []
    seen = set()
    for item in items:
        key = " ".join(item.split()).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if limit is not None and len(unique) >= limit:
            break
    return unique


def _clean_search_text(value: Any) -> str:
    text = ensure_str(value)
    if not text:
        return ""

    for label in ["행동:", "인물:", "사물:", "장소:", "화면 키워드:", "대사 키워드:"]:
        text = text.replace(f" {label}", f"\n{label}")

    empty_values = {
        "없음",
        "없다",
        "없습니다",
        "해당 없음",
        "해당없음",
        "없음.",
        "[]",
        "n/a",
        "none",
        "null",
    }
    lines = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        value_part = line.split(":", 1)[1].strip() if ":" in line else line
        if value_part.casefold() in empty_values:
            continue
        lines.append(line)
    return _clip_multiline_text("\n".join(lines), 1800)


class HuggingFaceChatClient:
    def __init__(
        self,
        token: str,
        provider: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ):
        if not token:
            raise RuntimeError("HF_TOKEN is required for Hugging Face model calls.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is required for Hugging Face Router chat calls. "
                "Install VideoQna/requirements.txt first."
            ) from exc

        self.provider = provider
        timeout = _env_float("HF_CHAT_TIMEOUT", timeout)
        max_retries = _env_int("HF_CHAT_MAX_RETRIES", max_retries)
        retry_base_delay = _env_float("HF_CHAT_RETRY_BASE_DELAY", retry_base_delay)
        self.max_retries = max(1, int(max_retries))
        self.retry_base_delay = max(0.1, float(retry_base_delay))
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=token,
            timeout=timeout,
        )

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 800,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        active_messages = messages
        content = ""
        parse_error: Exception | None = None

        for attempt in range(2):
            kwargs: dict[str, Any] = {
                "model": self._router_model(model),
                "messages": active_messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }
            if response_format:
                kwargs["response_format"] = response_format
            if extra_body:
                kwargs["extra_body"] = extra_body

            try:
                response = self._create_completion(**kwargs)
            except Exception as exc:
                if extra_body and self._is_request_param_error(exc):
                    print(
                        "[warn] HF chat rejected no-reasoning request options; "
                        "retrying without extra_body."
                    )
                    kwargs.pop("extra_body", None)
                    try:
                        response = self._create_completion(**kwargs)
                    except Exception as retry_exc:
                        raise RuntimeError(
                            self._format_hf_error(model, retry_exc)
                        ) from retry_exc
                else:
                    raise RuntimeError(self._format_hf_error(model, exc)) from exc

            content = _message_content(response)
            if not content.strip():
                parse_error = ValueError("Model response was empty.")
            else:
                try:
                    return extract_json_object(content)
                except Exception as exc:
                    parse_error = exc

            if attempt == 0:
                active_messages = _json_retry_messages(messages, content)

        raise RuntimeError(
            f"Model did not return valid JSON: {_content_preview(content)}"
        ) from parse_error

    def chat_text(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1000,
        temperature: float = 0.2,
    ) -> str:
        try:
            response = self._create_completion(
                model=self._router_model(model),
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            raise RuntimeError(self._format_hf_error(model, exc)) from exc
        return _message_content(response).strip()

    def _create_completion(self, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries - 1 or not self._is_retryable_error(exc):
                    raise
                self._sleep_before_retry(attempt, exc)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Completion request failed without an exception.")

    def _sleep_before_retry(self, attempt: int, exc: Exception) -> None:
        delay = min(30.0, self.retry_base_delay * (2**attempt))
        delay += random.uniform(0.0, delay * 0.25)
        print(
            f"[retry] HF chat transient error; retrying in {delay:.1f}s "
            f"({type(exc).__name__})"
        )
        time.sleep(delay)

    def _router_model(self, model: str) -> str:
        if ":" in model or not self.provider:
            return model
        return f"{model}:{self.provider}"

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

    @staticmethod
    def _is_request_param_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        message = str(exc).lower()
        if status_code == 400 and "invalid_request" in message:
            return True
        markers = [
            "input validation error",
            "unexpected keyword argument",
            "unknown parameter",
            "unsupported parameter",
            "invalid parameter",
        ]
        return any(marker in message for marker in markers)

    @staticmethod
    def _format_hf_error(model: str, exc: Exception) -> str:
        message = str(exc)
        hint = ""
        if "non-serverless model" in message or "model_not_available" in message:
            hint = (
                "\nHint: this model is not available through the selected Hugging Face "
                "serverless provider. Set HF_VLM_PROVIDER/HF_LLM_PROVIDER to another "
                "provider such as hf-inference, use a serverless-supported model, or "
                "create a dedicated endpoint for the model."
            )
        return f"Hugging Face request failed for model '{model}': {message}{hint}"


class VideoVLMClient:
    def __init__(self, token: str, model: str, provider: Optional[str] = None):
        self.model = model
        self.chat = HuggingFaceChatClient(token=token, provider=provider)

    def describe_keyframe(self, image_path: str | Path) -> FrameDescription:
        image_data_url = self._image_data_url(image_path)
        messages = [
            {
                "role": "system",
                "content": (
                    "You describe one video keyframe for Korean video retrieval. "
                    "Use only visible evidence in the image. Do not infer dialogue, names, "
                    "plot, relationships, emotions, or unseen context. Prefer concrete visible "
                    "objects, people, settings, text, posture, and actions over interpretation. "
                    "Return plain Korean text only. Do not return JSON, markdown, or reasoning."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                    {
                        "type": "text",
                        "text": (
                            "키프레임을 한국어 검색용으로 설명하세요.\n"
                            "- 2-4개의 짧은 문장으로 작성하세요.\n"
                            "- 보이는 인물, 복장, 자세, 행동, 물체, 배경, 읽을 수 있는 글자를 구체적으로 적으세요.\n"
                            "- 소리, 대사, 이전 줄거리, 숨은 감정, 이름, 관계는 추측하지 마세요.\n"
                            "- JSON이나 마크다운 없이 일반 텍스트만 반환하세요. /no_think"
                        ),
                    },
                ],
            },
        ]
        try:
            description = self.chat.chat_text(
                self.model,
                messages,
                max_tokens=500,
                temperature=0.0,
            )
        except RuntimeError as exc:
            reason = str(exc).splitlines()[0]
            print(f"[warn] VLM keyframe text failed; using fallback description: {reason}")
            return FrameDescription(
                frame_description="키프레임 이미지 설명을 생성하지 못했습니다.",
                visible_objects=[],
                visible_actions=[],
                people=[],
                setting="",
                visible_text=[],
                visual_keywords=[],
            )

        description = _clip_multiline_text(ensure_str(description), 1200)
        if not description:
            print("[warn] VLM keyframe text was empty; using fallback description.")
            description = "키프레임 이미지 설명을 생성하지 못했습니다."
        return FrameDescription(
            frame_description=description,
            visible_objects=[],
            visible_actions=[],
            people=[],
            setting="",
            visible_text=[],
            visual_keywords=[],
        )

    @staticmethod
    def _image_data_url(image_path: str | Path) -> str:
        image_path = Path(image_path)
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"


class SummaryLLMClient:
    def __init__(
        self,
        token: str,
        model: str,
        provider: Optional[str] = None,
        character_glossary: str = "",
    ):
        self.model = model
        self.character_glossary = character_glossary.strip()
        self.chat = HuggingFaceChatClient(token=token, provider=provider)

    def summarize_scene(
        self,
        frame_description: FrameDescription,
        shot_subtitles: list[SubtitleSegment],
    ) -> SceneSummary:
        payload = {
            "frame_description": frame_description.to_dict(),
            "shot_subtitles": [segment.to_dict() for segment in shot_subtitles],
        }
        if self.character_glossary:
            payload["character_glossary"] = self.character_glossary
        messages = [
            {
                "role": "system",
                "content": (
                    "You write retrieval-optimized Korean metadata for one video shot. "
                    "Use only the provided keyframe visual description and overlapping subtitles. "
                    "The output will be used for semantic vector search and keyword/BM25 search, "
                    "so preserve concrete nouns, actions, people, objects, places, and dialogue terms. "
                    "If a character glossary is provided, use it as allowed character names, aliases, "
                    "and visual clues. Attach a character name only when the visual description, "
                    "subtitle, or glossary clue supports it; if uncertain, phrase it as an apparent "
                    "or subtitle-mentioned character rather than a hard fact. "
                    "Do not invent facts, names, emotions, places, relationships, or story context "
                    "that are not supported by the input. Return only one valid JSON object. "
                    "Do not include markdown or reasoning."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create a concise but search-rich scene record. "
                    "Return JSON fields exactly as:\n"
                    "- summary: 1-2 factual Korean sentences combining visible evidence and subtitles\n"
                    "- action: array of concrete actions/events as short Korean verb phrases\n"
                    "- context: short factual context from subtitles and visible scene; empty string if unclear\n"
                    "- emotion: array of explicitly supported emotions or tones; [] if unclear\n"
                    "- people: array of visible or mentioned people, roles, names, or descriptions\n"
                    "- objects: array of important visible or mentioned objects/items\n"
                    "- places: array of visible or mentioned places/settings\n"
                    "- visual_keywords: array of Korean visual search terms and useful synonyms\n"
                    "- dialogue_keywords: array of important subtitle terms, names, topics, questions, goals, or events\n"
                    "- search_text: one Korean string with 5-8 short newline-separated lines optimized for retrieval\n\n"
                    "Search text rules:\n"
                    "- Use labels like 요약:, 행동:, 인물:, 사물:, 장소:, 화면 키워드:, 대사 키워드: when available.\n"
                    "- Include the summary first and make every line useful as a standalone search clue.\n"
                    "- Do not include unavailable categories in search_text; never write 없음/해당 없음 lines.\n"
                    "- Repeat the most important nouns/verbs naturally 1-2 times, but do not add unrelated keyword spam.\n"
                    "- Add common Korean synonyms only when they are faithful to the evidence.\n"
                    "- If subtitles are empty, keep dialogue_keywords as [] and focus search_text on visual evidence.\n"
                    "- Avoid poetic language; use terms a user would type in a video search question.\n\n"
                    "Return exactly one JSON object and no markdown. /no_think\n\n"
                    f"INPUT_JSON:\n{json.dumps(payload, ensure_ascii=False)}"
                ),
            },
        ]
        try:
            data = self.chat.chat_json(
                self.model,
                messages,
                max_tokens=1200,
                response_format={"type": "json_object"},
                extra_body=NO_REASONING_EXTRA_BODY,
            )
        except RuntimeError as exc:
            if "Model did not return valid JSON" not in str(exc):
                raise
            reason = str(exc).splitlines()[0]
            print(f"[warn] LLM summary JSON failed; using fallback summary: {reason}")
            return self.fallback_summary(frame_description, shot_subtitles)

        summary = SceneSummary(
            summary=ensure_str(data.get("summary")),
            action=_unique_str_list(data.get("action"), limit=12),
            context=ensure_str(data.get("context")),
            emotion=_unique_str_list(data.get("emotion"), limit=8),
            people=_unique_str_list(data.get("people"), limit=15),
            objects=_unique_str_list(data.get("objects"), limit=20),
            places=_unique_str_list(data.get("places"), limit=12),
            visual_keywords=_unique_str_list(data.get("visual_keywords"), limit=25),
            dialogue_keywords=_unique_str_list(data.get("dialogue_keywords"), limit=25),
            search_text=_clean_search_text(data.get("search_text")),
        )
        if not summary.summary:
            print("[warn] LLM summary JSON omitted summary; using fallback summary.")
            return self.fallback_summary(frame_description, shot_subtitles)
        if not summary.search_text:
            summary.search_text = self.search_text_from_summary(
                summary,
                frame_description,
                shot_subtitles,
            )
        return summary

    @staticmethod
    def fallback_summary(
        frame_description: FrameDescription,
        shot_subtitles: list[SubtitleSegment],
    ) -> SceneSummary:
        visual = ensure_str(frame_description.frame_description)
        subtitle_texts = [ensure_str(segment.text) for segment in shot_subtitles]
        subtitle_text = " ".join(text for text in subtitle_texts if text)

        if visual and subtitle_text:
            summary = f"{visual} 자막 내용: {subtitle_text}"
        elif subtitle_text:
            summary = subtitle_text
        elif visual:
            summary = visual
        else:
            summary = "장면 요약을 생성할 수 있는 화면 설명이나 자막이 부족합니다."

        context_parts = []
        if visual:
            context_parts.append(f"화면: {visual}")
        if subtitle_text:
            context_parts.append(f"자막: {subtitle_text}")

        fallback = SceneSummary(
            summary=_clip_text(summary, 700),
            action=_unique_str_list(frame_description.visible_actions, limit=12),
            context=_clip_text(" ".join(context_parts), 1000),
            emotion=[],
            people=_unique_str_list(frame_description.people, limit=15),
            objects=_unique_str_list(frame_description.visible_objects, limit=20),
            places=_unique_str_list(frame_description.setting, limit=12),
            visual_keywords=_unique_str_list(frame_description.visual_keywords, limit=25),
            dialogue_keywords=[],
        )
        fallback.search_text = SummaryLLMClient.search_text_from_summary(
            fallback,
            frame_description,
            shot_subtitles,
        )
        return fallback

    @staticmethod
    def search_text_from_summary(
        summary: SceneSummary,
        frame_description: FrameDescription,
        shot_subtitles: list[SubtitleSegment],
    ) -> str:
        subtitle_text = " ".join(
            ensure_str(segment.text) for segment in shot_subtitles if ensure_str(segment.text)
        )
        parts = [
            f"요약: {summary.summary}",
            f"행동: {', '.join(summary.action)}" if summary.action else "",
            f"맥락: {summary.context}" if summary.context else "",
            f"인물: {', '.join(summary.people)}" if summary.people else "",
            f"사물: {', '.join(summary.objects)}" if summary.objects else "",
            f"장소: {', '.join(summary.places)}" if summary.places else "",
            f"화면 키워드: {', '.join(summary.visual_keywords)}" if summary.visual_keywords else "",
            (
                f"대사 키워드: {', '.join(summary.dialogue_keywords)}"
                if summary.dialogue_keywords
                else ""
            ),
            f"화면 설명: {frame_description.frame_description}",
            f"자막: {_clip_text(subtitle_text, 500)}" if subtitle_text else "",
        ]
        return _clip_multiline_text("\n".join(part for part in parts if part), 1800)


class UnifiedSceneClient:
    def __init__(
        self,
        token: str,
        model: str,
        provider: Optional[str] = None,
        character_glossary: str = "",
    ):
        self.model = model
        self.character_glossary = character_glossary.strip()
        self.chat = HuggingFaceChatClient(token=token, provider=provider)

    def summarize_scenes(self, scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not scenes:
            return []
        try:
            return self._summarize_scenes_once(scenes)
        except RuntimeError as exc:
            if len(scenes) <= 1:
                raise
            midpoint = max(1, len(scenes) // 2)
            print(
                "[warn] unified scene batch failed; splitting batch "
                f"size={len(scenes)}: {str(exc).splitlines()[0]}"
            )
            return [
                *self.summarize_scenes(scenes[:midpoint]),
                *self.summarize_scenes(scenes[midpoint:]),
            ]

    def _summarize_scenes_once(self, scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Analyze the following video shots in one API call. For each shot, do "
                    "the work in two internal passes: first inspect only the keyframe image "
                    "for visible evidence, then combine that visual evidence with the shot "
                    "subtitles to produce the final retrieval JSON. Return one result for "
                    "every input shot_id."
                ),
            }
        ]
        for scene in scenes:
            shot = scene["shot"]
            subtitles = scene.get("shot_subtitles") or []
            subtitle_rows = [
                {
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "text": segment.text,
                }
                for segment in subtitles
            ]
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"SHOT_ID: {shot.shot_id}\n"
                        f"TIME: {shot.start_time:.2f}-{shot.end_time:.2f}\n"
                        "PASS_1_INPUT: inspect the next keyframe image only."
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": VideoVLMClient._image_data_url(scene["keyframe"].image_path)
                    },
                }
            )
            content.append(
                {
                    "type": "text",
                    "text": (
                        "PASS_2_INPUT: now combine the visual evidence from the keyframe "
                        "with these subtitles for the same shot.\n"
                        f"SUBTITLES_JSON: {json.dumps(subtitle_rows, ensure_ascii=False)}"
                    ),
                }
            )

        if self.character_glossary:
            content.append(
                {
                    "type": "text",
                    "text": f"CHARACTER_GLOSSARY:\n{self.character_glossary}",
                }
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a multimodal Korean video indexing model. This is a single "
                    "API request, but you must internally process each shot in two passes. "
                    "Pass 1: inspect the keyframe image only and derive visible evidence. "
                    "Pass 2: combine that visible evidence with subtitles and the optional "
                    "character glossary to produce the final scene JSON. Do not let subtitles "
                    "change what is visibly present in frame_description. If a name is only "
                    "implied by subtitles or glossary clues, phrase it conservatively. Do not "
                    "invent facts, relationships, emotions, places, or names. Return only one "
                    "valid JSON object."
                ),
            },
            {
                "role": "user",
                "content": [
                    *content,
                    {
                        "type": "text",
                        "text": (
                            "Return JSON exactly as {\"scenes\": [...]}.\n"
                            "Each scene object must include:\n"
                            "- shot_id: integer matching the input\n"
                            "- frame_description: 1 factual Korean sentence from Pass 1 image-only visible evidence\n"
                            "- summary: 1-2 Korean sentences combining visible evidence and subtitles\n"
                            "- action: array of concrete Korean action phrases\n"
                            "- context: short factual context; empty string if unclear\n"
                            "- emotion: array of explicitly supported emotions or tones; [] if unclear\n"
                            "- people: array of visible or mentioned people, roles, names, or descriptions\n"
                            "- objects: array of important visible or mentioned objects/items\n"
                            "- places: array of visible or mentioned places/settings\n"
                            "- visual_keywords: Korean visual search terms and useful synonyms\n"
                            "- dialogue_keywords: important subtitle terms, names, topics, goals, or events\n"
                            "- search_text: one Korean string with short newline-separated retrieval lines\n\n"
                            "Do not output the intermediate two-pass notes. Only output the final JSON. "
                            "Search_text should use labels such as 요약:, 행동:, 인물:, 사물:, 장소:, "
                            "화면 키워드:, 대사 키워드:. Never include 없음/해당 없음 lines. "
                            "Return exactly one JSON object and no markdown. /no_think"
                        ),
                    },
                ],
            },
        ]
        data = self.chat.chat_json(
            self.model,
            messages,
            max_tokens=max(1200, 650 * len(scenes)),
            response_format={"type": "json_object"},
            extra_body=NO_REASONING_EXTRA_BODY,
        )
        scene_items = data.get("scenes")
        if not isinstance(scene_items, list):
            raise RuntimeError("Unified scene model did not return a scenes array.")

        by_id = {}
        for item in scene_items:
            if not isinstance(item, dict):
                continue
            try:
                by_id[int(item.get("shot_id"))] = item
            except (TypeError, ValueError):
                continue

        results = []
        for scene in scenes:
            shot_id = int(scene["shot"].shot_id)
            item = by_id.get(shot_id)
            if item is None:
                raise RuntimeError(f"Unified scene response omitted shot_id={shot_id}.")
            frame_description = self._frame_description_from_item(item)
            summary = self._summary_from_item(item, frame_description, scene["shot_subtitles"])
            results.append(
                {
                    "shot": scene["shot"],
                    "keyframe": scene["keyframe"],
                    "shot_subtitles": scene["shot_subtitles"],
                    "frame_description": frame_description,
                    "summary": summary,
                }
            )
        return results

    @staticmethod
    def _frame_description_from_item(item: dict[str, Any]) -> FrameDescription:
        return FrameDescription(
            frame_description=ensure_str(
                item.get("frame_description") or item.get("description") or item.get("summary")
            ),
            visible_objects=_unique_str_list(item.get("objects") or item.get("visible_objects"), limit=12),
            visible_actions=_unique_str_list(item.get("action") or item.get("visible_actions"), limit=12),
            people=_unique_str_list(item.get("people"), limit=12),
            setting=ensure_str(item.get("setting") or item.get("context")),
            visible_text=_unique_str_list(item.get("visible_text"), limit=12),
            visual_keywords=_unique_str_list(item.get("visual_keywords"), limit=20),
        )

    @staticmethod
    def _summary_from_item(
        item: dict[str, Any],
        frame_description: FrameDescription,
        shot_subtitles: list[SubtitleSegment],
    ) -> SceneSummary:
        summary = SceneSummary(
            summary=ensure_str(item.get("summary")),
            action=_unique_str_list(item.get("action"), limit=12),
            context=ensure_str(item.get("context")),
            emotion=_unique_str_list(item.get("emotion"), limit=8),
            people=_unique_str_list(item.get("people"), limit=15),
            objects=_unique_str_list(item.get("objects"), limit=20),
            places=_unique_str_list(item.get("places"), limit=12),
            visual_keywords=_unique_str_list(item.get("visual_keywords"), limit=25),
            dialogue_keywords=_unique_str_list(item.get("dialogue_keywords"), limit=25),
            search_text=_clean_search_text(item.get("search_text")),
        )
        if not summary.summary:
            return SummaryLLMClient.fallback_summary(frame_description, shot_subtitles)
        if not summary.search_text:
            summary.search_text = SummaryLLMClient.search_text_from_summary(
                summary,
                frame_description,
                shot_subtitles,
            )
        return summary


class RAGLLMClient:
    def __init__(self, token: str, model: str, provider: Optional[str] = None):
        self.model = model
        self.chat = HuggingFaceChatClient(token=token, provider=provider)

    def expand_query(self, question: str) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You expand Korean video search questions. Return only JSON. "
                    "Do not include reasoning."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create search expansions for this video RAG question. "
                    "Return JSON fields exactly as: expanded_queries (array of up to 3 Korean strings), "
                    "keywords (array of Korean or English terms). "
                    "Keep expansions faithful to the question. "
                    "Return exactly one JSON object and no markdown. /no_think\n\n"
                    f"QUESTION: {question}"
                ),
            },
        ]
        data = self.chat.chat_json(self.model, messages, max_tokens=500)
        expanded = ensure_str_list(data.get("expanded_queries"))[:3]
        keywords = ensure_str_list(data.get("keywords"))[:12]
        return {"expanded_queries": expanded, "keywords": keywords}

    def answer_question(self, question: str, sources: list[dict[str, Any]]) -> str:
        if not sources:
            return "저장된 영상 정보에서 질문과 관련된 내용을 찾기 어렵습니다."

        context = []
        for source in sources:
            context.append(
                {
                    "rank": source.get("rank"),
                    "shot_id": source.get("shot_id"),
                    "timestamp": source.get("timestamp"),
                    "summary": source.get("summary"),
                    "action": source.get("action"),
                    "context": source.get("context"),
                    "emotion": source.get("emotion"),
                    "people": source.get("people"),
                    "objects": source.get("objects"),
                    "places": source.get("places"),
                    "visual_keywords": source.get("visual_keywords"),
                    "dialogue_keywords": source.get("dialogue_keywords"),
                    "frame_description": source.get("frame_description"),
                    "subtitles": source.get("subtitles"),
                }
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You answer questions about an indexed video. "
                    "Use only the provided retrieved shot context. "
                    "Answer in Korean. Mention relevant timestamps when useful. "
                    "If the context is insufficient, say that the stored video information "
                    "does not contain enough evidence."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"RETRIEVED_CONTEXT_JSON:\n{json.dumps(context, ensure_ascii=False)}"
                ),
            },
        ]
        try:
            answer = self.chat.chat_text(self.model, messages, max_tokens=1100, temperature=0.2)
        except Exception as exc:
            print(f"[warn] RAG answer generation failed; using fallback answer: {exc}")
            return self.fallback_answer(sources)

        answer = ensure_str(answer)
        if answer:
            return answer

        print("[warn] RAG answer generation returned empty text; using fallback answer.")
        return self.fallback_answer(sources)

    @staticmethod
    def fallback_answer(sources: list[dict[str, Any]]) -> str:
        if not sources:
            return "저장된 영상 정보에서 질문과 관련된 내용을 찾기 어렵습니다."

        lines = ["검색된 장면 기준으로는 다음 내용이 관련 있어 보입니다:"]
        for source in sources[:3]:
            timestamp = ensure_str(source.get("timestamp")) or "시간 정보 없음"
            summary = ensure_str(source.get("summary")) or ensure_str(source.get("frame_description"))
            if not summary:
                summary = "요약 정보가 부족한 장면입니다."
            lines.append(f"- {timestamp}: {_clip_text(summary, 220)}")
        return "\n".join(lines)


class VideoEntitySweepClient:
    def __init__(self, token: str, model: str, provider: Optional[str] = None):
        self.model = model
        self.chat = HuggingFaceChatClient(token=token, provider=provider)

    def infer_candidates(self, scenes: list[dict[str, Any]]) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You infer recurring character or person identity candidates from indexed "
                    "video scene metadata. Use only the provided scene summaries, visual "
                    "descriptions, people fields, keywords, and subtitles. Do not use outside "
                    "knowledge. Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Extract recurring named or strongly identifiable characters. Prefer real "
                    "names and aliases from subtitles. If a name is absent, include a stable "
                    "descriptive label only when the same visual identity appears repeatedly. "
                    "Return JSON exactly as {\"characters\": [...]}. Each character object must "
                    "have: canonical_name, aliases, visual_clues, evidence_shot_ids, confidence "
                    "(low/medium/high). Keep only candidates supported by evidence. Return at "
                    "most 8 characters for this chunk. Keep aliases and visual_clues as short "
                    "arrays of strings, and keep evidence_shot_ids to the strongest shots only.\n\n"
                    f"SCENES_JSON:\n{json.dumps(scenes, ensure_ascii=False)}"
                ),
            },
        ]
        max_tokens = min(6000, max(2200, 1000 + 90 * len(scenes)))
        return self.chat.chat_json(
            self.model,
            messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body=NO_REASONING_EXTRA_BODY,
        )

    def merge_candidates(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You merge character identity candidates extracted from chunks of one video. "
                    "Use only the provided candidates. Deduplicate aliases and evidence. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Merge duplicate candidates and keep conservative evidence. Return JSON "
                    "exactly as {\"characters\": [...]}. Each character object must have: "
                    "canonical_name, aliases, visual_clues, evidence_shot_ids, confidence "
                    "(low/medium/high). Do not add names that are not present in candidates. "
                    "Return at most 24 final characters, and keep each field concise.\n\n"
                    f"CANDIDATES_JSON:\n{json.dumps(candidates, ensure_ascii=False)}"
                ),
            },
        ]
        max_tokens = min(7000, max(2600, 800 + 90 * len(candidates)))
        return self.chat.chat_json(
            self.model,
            messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body=NO_REASONING_EXTRA_BODY,
        )

    def assign_entities(
        self,
        entities: dict[str, Any],
        scenes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You assign known video character candidates to scene records. Use only "
                    "the provided entity list and scene evidence. Be conservative: assign a "
                    "name only when subtitles mention it or visual clues match. Return only "
                    "valid JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    "For each scene with enough evidence, return an assignment. Return JSON "
                    "exactly as {\"assignments\": [...]}. Each assignment must have: shot_id, "
                    "names, evidence, confidence (low/medium/high). Omit scenes with no useful "
                    "assignment. Keep evidence to one short sentence.\n\n"
                    f"ENTITIES_JSON:\n{json.dumps(entities, ensure_ascii=False)}\n\n"
                    f"SCENES_JSON:\n{json.dumps(scenes, ensure_ascii=False)}"
                ),
            },
        ]
        max_tokens = min(7000, max(2600, 900 + 80 * len(scenes)))
        return self.chat.chat_json(
            self.model,
            messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body=NO_REASONING_EXTRA_BODY,
        )
