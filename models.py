from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SubtitleSegment:
    index: int
    start_time: float
    end_time: float
    text: str
    source: str = "whisper"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Shot:
    shot_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Keyframe:
    shot_id: int
    frame_index: int
    timestamp_sec: float
    timestamp_str: str
    image_path: str
    sharpness: float
    vlm_image_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameDescription:
    frame_description: str
    visible_objects: list[str] = field(default_factory=list)
    visible_actions: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    setting: str = ""
    visible_text: list[str] = field(default_factory=list)
    visual_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneSummary:
    summary: str
    action: list[str] = field(default_factory=list)
    context: str = ""
    emotion: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    places: list[str] = field(default_factory=list)
    visual_keywords: list[str] = field(default_factory=list)
    dialogue_keywords: list[str] = field(default_factory=list)
    search_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

