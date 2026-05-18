from __future__ import annotations

from models import Shot, SubtitleSegment


def subtitles_for_shot(
    subtitles: list[SubtitleSegment],
    shot: Shot,
    padding_sec: float = 0.5,
) -> list[SubtitleSegment]:
    start = max(0.0, shot.start_time - padding_sec)
    end = shot.end_time + padding_sec
    return [
        segment
        for segment in subtitles
        if segment.end_time >= start and segment.start_time <= end
    ]

