from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from models import Keyframe, Shot
from utils import seconds_to_timestamp


@dataclass
class CandidateFrame:
    frame_index: int
    timestamp_sec: float
    image: np.ndarray
    feature: np.ndarray
    sharpness: float


class KMeansKeyframeSelector:
    def __init__(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        candidate_stride: float = 0.5,
    ):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.candidate_stride = max(0.1, float(candidate_stride))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
        if self.fps <= 0:
            self.close()
            raise RuntimeError(f"Cannot read FPS from video: {self.video_path}")

    def close(self) -> None:
        if getattr(self, "cap", None) is not None:
            self.cap.release()

    def select_one(self, shot: Shot) -> Keyframe:
        candidates = self._sample_candidates(shot)
        if not candidates:
            raise RuntimeError(f"No candidate frames found for shot {shot.shot_id}.")

        selected = candidates[0] if len(candidates) == 1 else self._kmeans_centroid_pick(candidates)
        timestamp = seconds_to_timestamp(selected.timestamp_sec).replace(":", "-")
        image_path = self.output_dir / f"shot_{shot.shot_id:04d}_{timestamp}.jpg"
        cv2.imwrite(str(image_path), selected.image)

        return Keyframe(
            shot_id=shot.shot_id,
            frame_index=selected.frame_index,
            timestamp_sec=selected.timestamp_sec,
            timestamp_str=seconds_to_timestamp(selected.timestamp_sec),
            image_path=str(image_path),
            sharpness=selected.sharpness,
        )

    def _sample_candidates(self, shot: Shot) -> list[CandidateFrame]:
        start = max(0.0, shot.start_time)
        end = max(start, shot.end_time)
        if end - start < self.candidate_stride:
            timestamps = [start + (end - start) / 2]
        else:
            timestamps = list(np.arange(start, end, self.candidate_stride))
            midpoint = start + (end - start) / 2
            if all(abs(ts - midpoint) > 0.05 for ts in timestamps):
                timestamps.append(midpoint)
            timestamps = sorted(set(round(float(ts), 3) for ts in timestamps))

        candidates: list[CandidateFrame] = []
        for timestamp in timestamps:
            frame = self._read_frame_at(timestamp)
            if frame is None:
                continue
            frame_index = int(round(timestamp * self.fps))
            candidates.append(
                CandidateFrame(
                    frame_index=frame_index,
                    timestamp_sec=float(timestamp),
                    image=frame,
                    feature=self._feature(frame),
                    sharpness=self._sharpness(frame),
                )
            )
        return candidates

    def _read_frame_at(self, timestamp: float) -> np.ndarray | None:
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000)
        ok, frame = self.cap.read()
        return frame if ok else None

    @staticmethod
    def _feature(frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return np.concatenate([hist.astype(np.float32), [gray.mean(), gray.std()]]).astype(np.float32)

    @staticmethod
    def _sharpness(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _kmeans_centroid_pick(candidates: list[CandidateFrame]) -> CandidateFrame:
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required for K-Means keyframe selection."
            ) from exc

        features = np.stack([candidate.feature for candidate in candidates])
        try:
            model = KMeans(n_clusters=1, random_state=42, n_init="auto")
        except TypeError:
            model = KMeans(n_clusters=1, random_state=42, n_init=10)
        model.fit(features)

        centroid = model.cluster_centers_[0]
        distances = np.linalg.norm(features - centroid, axis=1)
        min_distance = float(distances.min())
        tied = [
            candidate
            for candidate, distance in zip(candidates, distances)
            if abs(float(distance) - min_distance) <= 1e-9
        ]
        return max(tied, key=lambda candidate: candidate.sharpness)

