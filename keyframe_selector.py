from __future__ import annotations

from collections import defaultdict
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


@dataclass
class CandidateSample:
    frame_index: int
    timestamp_sec: float
    feature: np.ndarray
    sharpness: float


@dataclass
class CandidateTarget:
    shot: Shot
    frame_index: int
    timestamp_sec: float


class CentroidKeyframeSelector:
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

        selected = candidates[0] if len(candidates) == 1 else self._centroid_pick(candidates)
        return self._write_keyframe(
            output_dir=self.output_dir,
            shot=shot,
            frame=selected.image,
            frame_index=selected.frame_index,
            timestamp_sec=selected.timestamp_sec,
            sharpness=selected.sharpness,
        )

    def _sample_candidates(self, shot: Shot) -> list[CandidateFrame]:
        candidates: list[CandidateFrame] = []
        for timestamp in self.candidate_timestamps(shot, self.candidate_stride):
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
    def _centroid_pick(candidates):
        features = np.stack([candidate.feature for candidate in candidates])
        centroid = features.mean(axis=0)
        distances = np.linalg.norm(features - centroid, axis=1)
        min_distance = float(distances.min())
        tied = [
            candidate
            for candidate, distance in zip(candidates, distances)
            if abs(float(distance) - min_distance) <= 1e-9
        ]
        return max(tied, key=lambda candidate: candidate.sharpness)

    @staticmethod
    def candidate_timestamps(shot: Shot, candidate_stride: float) -> list[float]:
        candidate_stride = max(0.1, float(candidate_stride))
        start = max(0.0, shot.start_time)
        end = max(start, shot.end_time)
        if end - start < candidate_stride:
            return [start + (end - start) / 2]

        timestamps = list(np.arange(start, end, candidate_stride))
        midpoint = start + (end - start) / 2
        if all(abs(ts - midpoint) > 0.05 for ts in timestamps):
            timestamps.append(midpoint)
        return sorted(set(round(float(ts), 3) for ts in timestamps))

    @staticmethod
    def _write_keyframe(
        output_dir: Path,
        shot: Shot,
        frame: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
        sharpness: float,
    ) -> Keyframe:
        timestamp = seconds_to_timestamp(timestamp_sec).replace(":", "-")
        image_path = output_dir / f"shot_{shot.shot_id:04d}_{timestamp}.jpg"
        cv2.imwrite(str(image_path), frame)

        return Keyframe(
            shot_id=shot.shot_id,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            timestamp_str=seconds_to_timestamp(timestamp_sec),
            image_path=str(image_path),
            sharpness=sharpness,
        )


def select_keyframe_for_shot(
    video_path: str | Path,
    output_dir: str | Path,
    candidate_stride: float,
    shot: Shot,
) -> Keyframe:
    selector = CentroidKeyframeSelector(
        video_path=video_path,
        output_dir=output_dir,
        candidate_stride=candidate_stride,
    )
    try:
        return selector.select_one(shot)
    finally:
        selector.close()


def select_keyframes_single_pass(
    video_path: str | Path,
    output_dir: str | Path,
    candidate_stride: float,
    shots: list[Shot],
) -> dict[int, Keyframe]:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not shots:
        return {}

    fps = _video_fps(video_path)
    targets_by_frame: dict[int, list[CandidateTarget]] = defaultdict(list)
    for shot in shots:
        for timestamp in CentroidKeyframeSelector.candidate_timestamps(shot, candidate_stride):
            frame_index = max(0, int(round(timestamp * fps)))
            targets_by_frame[frame_index].append(
                CandidateTarget(
                    shot=shot,
                    frame_index=frame_index,
                    timestamp_sec=float(timestamp),
                )
            )

    samples_by_shot: dict[int, list[CandidateSample]] = defaultdict(list)

    def collect_sample(frame_index: int, frame: np.ndarray) -> None:
        feature = CentroidKeyframeSelector._feature(frame)
        sharpness = CentroidKeyframeSelector._sharpness(frame)
        for target in targets_by_frame[frame_index]:
            samples_by_shot[target.shot.shot_id].append(
                CandidateSample(
                    frame_index=frame_index,
                    timestamp_sec=target.timestamp_sec,
                    feature=feature,
                    sharpness=sharpness,
                )
            )

    _visit_frames_in_order(video_path, sorted(targets_by_frame), collect_sample)

    selected_by_shot: dict[int, CandidateSample] = {}
    for shot in shots:
        samples = samples_by_shot.get(shot.shot_id, [])
        if not samples:
            raise RuntimeError(f"No candidate frames found for shot {shot.shot_id}.")
        selected_by_shot[shot.shot_id] = (
            samples[0] if len(samples) == 1 else CentroidKeyframeSelector._centroid_pick(samples)
        )

    selected_frames: dict[int, list[tuple[Shot, CandidateSample]]] = defaultdict(list)
    for shot in shots:
        selected = selected_by_shot[shot.shot_id]
        selected_frames[selected.frame_index].append((shot, selected))

    keyframes: dict[int, Keyframe] = {}

    def write_selected(frame_index: int, frame: np.ndarray) -> None:
        for shot, selected in selected_frames[frame_index]:
            keyframes[shot.shot_id] = CentroidKeyframeSelector._write_keyframe(
                output_dir=output_dir,
                shot=shot,
                frame=frame,
                frame_index=selected.frame_index,
                timestamp_sec=selected.timestamp_sec,
                sharpness=selected.sharpness,
            )

    _visit_frames_in_order(video_path, sorted(selected_frames), write_selected)

    missing = [shot.shot_id for shot in shots if shot.shot_id not in keyframes]
    if missing:
        raise RuntimeError(f"Could not write selected keyframes for shots: {missing}")
    return keyframes


def _video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if fps <= 0:
            raise RuntimeError(f"Cannot read FPS from video: {video_path}")
        return fps
    finally:
        cap.release()


def _visit_frames_in_order(video_path: Path, frame_indices: list[int], visit) -> None:
    if not frame_indices:
        return

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        for target_frame in frame_indices:
            while current_frame < target_frame:
                ok = cap.grab()
                if not ok:
                    return
                current_frame += 1

            ok, frame = cap.read()
            if not ok:
                return
            actual_frame = current_frame
            current_frame += 1

            if actual_frame == target_frame:
                visit(target_frame, frame)
    finally:
        cap.release()

