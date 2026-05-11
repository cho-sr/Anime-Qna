from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Optional

from models import FrameDescription, SceneSummary, SubtitleSegment
from utils import ensure_str, ensure_str_list, extract_json_object


def _message_content(response: Any) -> str:
    try:
        return response.choices[0].message.content
    except AttributeError:
        pass
    if isinstance(response, dict):
        return response["choices"][0]["message"]["content"]
    return str(response)


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
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is required. Install VideoQna/requirements.txt first."
            ) from exc

        kwargs: dict[str, Any] = {"token": token, "timeout": timeout}
        if provider:
            kwargs["provider"] = provider
        self.client = InferenceClient(**kwargs)

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 800,
    ) -> dict[str, Any]:
        response = self.client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        content = _message_content(response)
        try:
            return extract_json_object(content)
        except Exception as exc:
            raise RuntimeError(f"Model did not return valid JSON: {content[:500]}") from exc

    def chat_text(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1000,
        temperature: float = 0.2,
    ) -> str:
        response = self.client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return _message_content(response).strip()


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
                    "You describe a single video keyframe. Return only JSON in Korean."
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
                            "visible_objects, visible_actions."
                        ),
                    },
                ],
            },
        ]
        data = self.chat.chat_json(self.model, messages, max_tokens=700)
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
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create a concise searchable scene summary. "
                    "Return JSON fields exactly as: summary (string), "
                    "action (array of strings), context (string), emotion (array of strings).\n\n"
                    f"INPUT_JSON:\n{json.dumps(payload, ensure_ascii=False)}"
                ),
            },
        ]
        data = self.chat.chat_json(self.model, messages, max_tokens=900)
        return SceneSummary(
            summary=ensure_str(data.get("summary")),
            action=ensure_str_list(data.get("action")),
            context=ensure_str(data.get("context")),
            emotion=ensure_str_list(data.get("emotion")),
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
                    "You expand Korean video search questions. Return only JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create search expansions for this video RAG question. "
                    "Return JSON fields exactly as: expanded_queries (array of up to 3 Korean strings), "
                    "keywords (array of Korean or English terms). "
                    "Keep expansions faithful to the question.\n\n"
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
