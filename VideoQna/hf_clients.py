from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Optional

from models import FrameDescription, SceneSummary, SubtitleSegment
from utils import ensure_str, ensure_str_list, extract_json_object


JSON_RETRY_PROMPT = (
    "Your previous response was empty or invalid. Return exactly one valid JSON "
    "object and nothing else. Do not include markdown, comments, or reasoning. /no_think"
)


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


class HuggingFaceChatClient:
    def __init__(
        self,
        token: str,
        provider: Optional[str] = None,
        timeout: float = 120.0,
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
    ) -> dict[str, Any]:
        active_messages = messages
        content = ""
        parse_error: Exception | None = None

        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self._router_model(model),
                    messages=active_messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
            except Exception as exc:
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
            response = self.client.chat.completions.create(
                model=self._router_model(model),
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            raise RuntimeError(self._format_hf_error(model, exc)) from exc
        return _message_content(response).strip()

    def _router_model(self, model: str) -> str:
        if ":" in model or not self.provider:
            return model
        return f"{model}:{self.provider}"

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
                    "You describe a single video keyframe. Return only JSON in Korean. "
                    "Do not include reasoning."
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
                            "Describe only what is visible in this keyframe. "
                            "Do not infer dialogue or unseen context. "
                            "Return JSON with fields: frame_description, "
                            "visible_objects, visible_actions. "
                            "Return exactly one JSON object and no markdown. /no_think"
                        ),
                    },
                ],
            },
        ]
        try:
            data = self.chat.chat_json(self.model, messages, max_tokens=700)
        except RuntimeError as exc:
            if "Model did not return valid JSON" not in str(exc):
                raise
            reason = str(exc).splitlines()[0]
            print(f"[warn] VLM keyframe JSON failed; using fallback description: {reason}")
            return FrameDescription(
                frame_description="키프레임 이미지 설명을 생성하지 못했습니다.",
                visible_objects=[],
                visible_actions=[],
            )

        return FrameDescription(
            frame_description=ensure_str(
                data.get("frame_description") or data.get("description")
            ),
            visible_objects=ensure_str_list(data.get("visible_objects")),
            visible_actions=ensure_str_list(data.get("visible_actions")),
        )

    @staticmethod
    def _image_data_url(image_path: str | Path) -> str:
        image_path = Path(image_path)
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"


class SummaryLLMClient:
    def __init__(self, token: str, model: str, provider: Optional[str] = None):
        self.model = model
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
        messages = [
            {
                "role": "system",
                "content": (
                    "You summarize a video shot using a visual description and "
                    "the full subtitles overlapping that shot. Return only JSON in Korean."
                    " Do not include reasoning."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create a concise searchable scene summary. "
                    "Return JSON fields exactly as: summary (string), "
                    "action (array of strings), context (string), emotion (array of strings).\n\n"
                    "Return exactly one JSON object and no markdown. /no_think\n\n"
                    f"INPUT_JSON:\n{json.dumps(payload, ensure_ascii=False)}"
                ),
            },
        ]
        try:
            data = self.chat.chat_json(self.model, messages, max_tokens=900)
        except RuntimeError as exc:
            if "Model did not return valid JSON" not in str(exc):
                raise
            reason = str(exc).splitlines()[0]
            print(f"[warn] LLM summary JSON failed; using fallback summary: {reason}")
            return self.fallback_summary(frame_description, shot_subtitles)

        summary = SceneSummary(
            summary=ensure_str(data.get("summary")),
            action=ensure_str_list(data.get("action")),
            context=ensure_str(data.get("context")),
            emotion=ensure_str_list(data.get("emotion")),
        )
        if not summary.summary:
            print("[warn] LLM summary JSON omitted summary; using fallback summary.")
            return self.fallback_summary(frame_description, shot_subtitles)
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

        return SceneSummary(
            summary=_clip_text(summary, 700),
            action=ensure_str_list(frame_description.visible_actions),
            context=_clip_text(" ".join(context_parts), 1000),
            emotion=[],
        )


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
        return self.chat.chat_text(self.model, messages, max_tokens=1100, temperature=0.2)
