"""
subtitle_extractor.py
자막 추출 모듈: OCR / Whisper STT / SRT 파일 파싱 지원
"""

import re
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class SubtitleChunk:
    """자막 단위 데이터"""
    start_time: float      # 시작 시각 (초)
    end_time: float        # 종료 시각 (초)
    text: str              # 자막 텍스트
    chapter_id: int        # 챕터 번호
    source: str            # "ocr" | "whisper" | "srt"


# ──────────────────────────────────────────────
# 1. SRT 파일 파싱
# ──────────────────────────────────────────────

class SRTParser:
    """
    .srt 자막 파일 파싱
    
    사용법:
        parser = SRTParser()
        chunks = parser.parse("subtitle.srt", chapter_duration=60)
    """

    def parse(self, srt_path: str, chapter_duration: float = 60.0) -> List[SubtitleChunk]:
        with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # SRT 블록 파싱 (번호 / 타임코드 / 텍스트)
        pattern = re.compile(
            r"\d+\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)",
            re.DOTALL
        )
        chunks = []
        for match in pattern.finditer(content):
            start = self._time_to_sec(match.group(1))
            end = self._time_to_sec(match.group(2))
            text = match.group(3).strip().replace("\n", " ")
            # HTML 태그 제거
            text = re.sub(r"<[^>]+>", "", text)
            if not text:
                continue
            chunks.append(SubtitleChunk(
                start_time=start,
                end_time=end,
                text=text,
                chapter_id=int(start // chapter_duration),
                source="srt"
            ))
        print(f"✅ SRT 파싱 완료: {len(chunks)}개 자막")
        return chunks

    @staticmethod
    def _time_to_sec(time_str: str) -> float:
        """'HH:MM:SS,mmm' -> 초"""
        time_str = time_str.replace(",", ".")
        h, m, s = time_str.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)


# ──────────────────────────────────────────────
# 2. Whisper 기반 STT (음성 -> 자막)
# ──────────────────────────────────────────────

class WhisperExtractor:
    """
    faster-whisper를 사용한 음성 인식
    
    사용법:
        extractor = WhisperExtractor(model_size="base")
        chunks = extractor.transcribe("anime.mp4", language="ja")
    """

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """
        model_size: tiny / base / small / medium / large-v2
        device: cpu / cuda
        """
        self.model_size = model_size
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            print(f"🔊 Whisper 모델 로딩: {self.model_size} ({self.device})")
            self._model = WhisperModel(self.model_size, device=self.device, compute_type="int8")

    def transcribe(
        self,
        video_path: str,
        language: Optional[str] = None,      # None이면 자동 감지. "ja"(일본어), "ko"(한국어)
        chapter_duration: float = 60.0
    ) -> List[SubtitleChunk]:
        self._load_model()

        print(f"🎙️  음성 인식 시작: {video_path}")
        segments, info = self._model.transcribe(
            video_path,
            language=language,
            beam_size=5,
            vad_filter=True   # 무음 구간 자동 제거
        )

        chunks = []
        for seg in segments:
            if not seg.text.strip():
                continue
            chunks.append(SubtitleChunk(
                start_time=seg.start,
                end_time=seg.end,
                text=seg.text.strip(),
                chapter_id=int(seg.start // chapter_duration),
                source="whisper"
            ))

        print(f"✅ Whisper 인식 완료: {len(chunks)}개 세그먼트 (언어: {info.language})")
        return chunks


# ──────────────────────────────────────────────
# 3. OCR 기반 자막 추출 (화면 자막 인식)
# ──────────────────────────────────────────────

class OCRExtractor:
    """
    EasyOCR을 사용해 프레임 하단 자막 영역 텍스트 추출
    
    사용법:
        from video_processor import Frame
        extractor = OCRExtractor(languages=["ko", "ja", "en"])
        chunks = extractor.extract_from_frames(frames)
    """

    def __init__(self, languages: List[str] = ["ko", "ja", "en"]):
        self.languages = languages
        self._reader = None

    def _load_reader(self):
        if self._reader is None:
            import easyocr
            print(f"👁️  EasyOCR 로딩: {self.languages}")
            self._reader = easyocr.Reader(self.languages, gpu=False)

    def extract_from_frames(
        self,
        frames,                          # List[Frame]
        subtitle_region_ratio: float = 0.75,   # 화면 하단 몇 %부터 자막 영역
        chapter_duration: float = 60.0,
        confidence_threshold: float = 0.5
    ) -> List[SubtitleChunk]:
        """
        프레임 리스트에서 OCR로 자막 추출
        subtitle_region_ratio: 0.75이면 화면의 75% 아래 영역을 자막 영역으로 간주
        """
        self._load_reader()

        chunks = []
        prev_text = ""

        print(f"👁️  OCR 자막 추출 시작: {len(frames)}개 프레임...")
        for frame in frames:
            img = frame.image
            h, w = img.shape[:2]
            subtitle_y = int(h * subtitle_region_ratio)
            subtitle_region = img[subtitle_y:, :]   # 하단 영역 크롭

            results = self._reader.readtext(subtitle_region, detail=1)
            texts = [r[1] for r in results if r[2] >= confidence_threshold]
            text = " ".join(texts).strip()

            # 이전 프레임과 동일한 자막이면 스킵 (중복 방지)
            if not text or text == prev_text:
                continue

            prev_text = text
            chunks.append(SubtitleChunk(
                start_time=frame.timestamp,
                end_time=frame.timestamp + 2.0,  # 근사값
                text=text,
                chapter_id=frame.chapter_id,
                source="ocr"
            ))

        print(f"✅ OCR 완료: {len(chunks)}개 자막 추출")
        return chunks


# ──────────────────────────────────────────────
# 통합 자막 추출기
# ──────────────────────────────────────────────

class SubtitleExtractor:
    """
    세 가지 방법 중 하나를 선택해서 자막 추출
    
    method: "srt" | "whisper" | "ocr"
    """

    def __init__(self, method: str = "whisper", **kwargs):
        self.method = method
        self.kwargs = kwargs

    def extract(
        self,
        video_path: str,
        srt_path: Optional[str] = None,
        frames=None,
        chapter_duration: float = 60.0
    ) -> List[SubtitleChunk]:

        if self.method == "srt":
            if not srt_path:
                raise ValueError("SRT 모드에서는 srt_path가 필요합니다.")
            return SRTParser().parse(srt_path, chapter_duration)

        elif self.method == "whisper":
            model_size = self.kwargs.get("model_size", "base")
            language = self.kwargs.get("language", None)
            return WhisperExtractor(model_size=model_size).transcribe(
                video_path, language=language, chapter_duration=chapter_duration
            )

        elif self.method == "ocr":
            if not frames:
                raise ValueError("OCR 모드에서는 frames가 필요합니다.")
            languages = self.kwargs.get("languages", ["ko", "ja", "en"])
            return OCRExtractor(languages=languages).extract_from_frames(
                frames, chapter_duration=chapter_duration
            )

        else:
            raise ValueError(f"알 수 없는 method: {self.method}")
