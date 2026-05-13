from __future__ import annotations

from pathlib import Path
from typing import Optional

from device_utils import resolve_whisper_compute_type, resolve_whisper_device
from models import SubtitleSegment


class WhisperSubtitleExtractor:
    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: str = "auto",
        compute_type: str = "auto",
    ):
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is required. Install VideoQna/requirements.txt first."
                ) from exc

            device = resolve_whisper_device(self.device)
            compute_type = resolve_whisper_compute_type(self.compute_type, device)
            print(
                f"[whisper] loading model={self.model_size} "
                f"device={device} compute_type={compute_type}"
            )
            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
            )

    def transcribe(self, video_path: str | Path) -> list[SubtitleSegment]:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._load_model()
        print(f"[whisper] transcribing: {video_path}")
        try:
            segments, info = self._transcribe_with_vad(video_path, vad_filter=True)
        except RuntimeError as exc:
            if "VAD" not in str(exc) and "onnxruntime" not in str(exc).lower():
                raise
            print(f"[warn] whisper VAD unavailable; retrying without VAD: {exc}")
            segments, info = self._transcribe_with_vad(video_path, vad_filter=False)

        subtitles: list[SubtitleSegment] = []
        for index, segment in enumerate(segments):
            text = segment.text.strip()
            if not text:
                continue
            subtitles.append(
                SubtitleSegment(
                    index=len(subtitles),
                    start_time=float(segment.start),
                    end_time=float(segment.end),
                    text=text,
                )
            )

        print(f"[whisper] done: {len(subtitles)} segments, language={info.language}")
        return subtitles

    def _transcribe_with_vad(self, video_path: Path, *, vad_filter: bool):
        return self._model.transcribe(
            str(video_path),
            language=self.language,
            beam_size=5,
            vad_filter=vad_filter,
        )

