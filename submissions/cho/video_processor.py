"""
video_processor.py
영상 -> 프레임 분할 모듈
"""

import cv2
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm


@dataclass
class Frame:
    """추출된 프레임 데이터"""
    frame_id: int
    timestamp: float          # 초 단위
    timestamp_str: str        # "00:01:23" 형식
    image: np.ndarray         # OpenCV 이미지
    image_path: str           # 저장된 경로
    chapter_id: int           # 챕터 번호


def seconds_to_str(seconds: float) -> str:
    """초를 HH:MM:SS 형식으로 변환"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class VideoProcessor:
    """
    영상을 입력받아 일정 간격으로 프레임을 추출하는 클래스
    
    사용법:
        processor = VideoProcessor(frame_interval=2, chapter_duration=60)
        frames = processor.extract_frames("anime.mp4", output_dir="frames/")
    """

    def __init__(
        self,
        frame_interval: float = 2.0,     # 프레임 추출 간격 (초)
        chapter_duration: float = 60.0,   # 챕터 길이 (초)
        save_frames: bool = True,         # 프레임 이미지 저장 여부
        resize_width: int = 640           # 저장 시 리사이즈 너비 (None이면 원본)
    ):
        self.frame_interval = frame_interval
        self.chapter_duration = chapter_duration
        self.save_frames = save_frames
        self.resize_width = resize_width

    def extract_frames(
        self,
        video_path: str,
        output_dir: str = "frames"
    ) -> List[Frame]:
        """
        영상에서 프레임 추출
        
        Returns:
            List[Frame]: 추출된 프레임 리스트
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        frame_skip = int(fps * self.frame_interval)

        print(f"📹 영상 정보: {duration:.1f}초 / {fps:.1f}fps / 총 {total_frames}프레임")
        print(f"⏱️  {self.frame_interval}초 간격으로 프레임 추출 시작...")

        extracted_frames = []
        frame_id = 0
        current_frame_idx = 0

        with tqdm(total=int(duration / self.frame_interval)) as pbar:
            while True:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, image = cap.read()
                if not ret:
                    break

                timestamp = current_frame_idx / fps
                chapter_id = int(timestamp // self.chapter_duration)
                timestamp_str = seconds_to_str(timestamp)

                # 이미지 리사이즈
                if self.resize_width and image.shape[1] > self.resize_width:
                    ratio = self.resize_width / image.shape[1]
                    new_h = int(image.shape[0] * ratio)
                    image = cv2.resize(image, (self.resize_width, new_h))

                # 프레임 저장
                image_path = ""
                if self.save_frames:
                    image_path = str(output_dir / f"frame_{frame_id:06d}_{timestamp_str.replace(':', '-')}.jpg")
                    cv2.imwrite(image_path, image)

                frame = Frame(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    timestamp_str=timestamp_str,
                    image=image,
                    image_path=image_path,
                    chapter_id=chapter_id
                )
                extracted_frames.append(frame)

                frame_id += 1
                current_frame_idx += frame_skip
                pbar.update(1)

        cap.release()
        print(f"✅ 총 {len(extracted_frames)}개 프레임 추출 완료")
        return extracted_frames

    def get_video_info(self, video_path: str) -> dict:
        """영상 기본 정보 반환"""
        cap = cv2.VideoCapture(video_path)
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        info["duration"] = info["total_frames"] / info["fps"]
        cap.release()
        return info
