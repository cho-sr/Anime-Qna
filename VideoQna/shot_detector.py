from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from stat import S_IXGRP, S_IXOTH, S_IXUSR

import cv2
from tqdm import tqdm

from device_utils import resolve_torch_device
from models import Shot


def get_video_info(video_path: str | Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0:
        raise RuntimeError(f"Cannot read FPS from video: {video_path}")

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps if frame_count else 0.0,
    }


def create_lowres_proxy(
    video_path: str | Path,
    output_path: str | Path,
    target_width: int = 320,
) -> Path:
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or original_width <= 0 or original_height <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video metadata: {video_path}")

    target_width = max(32, int(target_width))
    target_height = int(round(original_height * target_width / original_width))
    target_width += target_width % 2
    target_height += target_height % 2

    print(
        f"[transnet] creating proxy {original_width}x{original_height} -> "
        f"{target_width}x{target_height}"
    )
    start = time.perf_counter()
    if _create_lowres_proxy_with_ffmpeg(video_path, output_path, target_width):
        cap.release()
        print(f"[transnet] proxy created with ffmpeg in {time.perf_counter() - start:.1f}s")
        return output_path

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create low-resolution proxy: {output_path}")

    with tqdm(total=frame_count or None, desc="proxy", unit="frame") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            writer.write(resized)
            pbar.update(1)

    cap.release()
    writer.release()
    print(f"[transnet] proxy created with OpenCV in {time.perf_counter() - start:.1f}s")
    return output_path


def _create_lowres_proxy_with_ffmpeg(
    video_path: Path,
    output_path: Path,
    target_width: int,
) -> bool:
    if not shutil.which("ffmpeg"):
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.unlink(missing_ok=True)
    except OSError:
        pass

    common_args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"scale={target_width}:-2",
    ]
    encoder_args = [
        ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p"],
        ["-c:v", "mpeg4", "-q:v", "5"],
    ]

    last_error = ""
    for encoder in encoder_args:
        completed = subprocess.run(
            [*common_args, *encoder, str(output_path)],
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True
        last_error = (completed.stderr or completed.stdout or "").strip()
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass

    if last_error:
        print(f"[warn] ffmpeg proxy failed; falling back to OpenCV: {last_error[-500:]}")
    return False


class TransNetShotDetector:
    def __init__(
        self,
        threshold: float = 0.5,
        device: str = "auto",
        proxy_width: int = 320,
        weights_path: str | Path | None = None,
    ):
        self.threshold = threshold
        self.device = device
        self.proxy_width = proxy_width
        self.weights_path = Path(weights_path).expanduser() if weights_path else None

    def detect(self, video_path: str | Path, work_dir: str | Path) -> list[Shot]:
        video_path = Path(video_path)
        work_dir = Path(work_dir)
        self._ensure_ffmpeg()
        info = get_video_info(video_path)
        proxy_path = work_dir / f"{video_path.stem}_transnet_proxy.mp4"
        create_lowres_proxy(video_path, proxy_path, self.proxy_width)

        try:
            from transnetv2_pytorch import TransNetV2
        except ImportError as exc:
            raise RuntimeError(
                "transnetv2-pytorch is required. Install VideoQna/requirements.txt first."
            ) from exc

        device = resolve_torch_device(self.device, label="transnet")
        print(f"[transnet] detecting shots threshold={self.threshold} device={device}")
        model = TransNetV2(device=device)
        self._load_weights_if_available(model)
        model.eval()

        if hasattr(model, "detect_scenes"):
            raw_scenes = model.detect_scenes(str(proxy_path), threshold=self.threshold)
        elif hasattr(model, "analyze_video"):
            raw_scenes = model.analyze_video(str(proxy_path), threshold=self.threshold)["scenes"]
        else:
            raise RuntimeError("Installed transnetv2-pytorch does not expose detect_scenes.")

        shots = self._normalize_scenes(raw_scenes, fps=info["fps"], duration=info["duration"])
        if not shots:
            shots = [
                Shot(
                    shot_id=0,
                    start_frame=0,
                    end_frame=max(0, int(info["frame_count"]) - 1),
                    start_time=0.0,
                    end_time=float(info["duration"]),
                )
            ]

        print(f"[transnet] done: {len(shots)} shots")
        return shots

    @staticmethod
    def _ensure_ffmpeg() -> None:
        if shutil.which("ffmpeg"):
            return
        if TransNetShotDetector._install_imageio_ffmpeg_shim():
            return
        raise RuntimeError(
            "ffmpeg executable was not found. TransNetV2 requires the system ffmpeg command. "
            "Install it with one of:\n"
            "  conda install -c conda-forge ffmpeg\n"
            "  brew install ffmpeg\n"
            "  pip install imageio-ffmpeg\n"
            "Then restart the terminal and run the indexing command again."
        )

    @staticmethod
    def _install_imageio_ffmpeg_shim() -> bool:
        try:
            import imageio_ffmpeg
        except ImportError:
            return False

        ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe()).expanduser().resolve()
        if not ffmpeg_exe.exists():
            return False

        shim_dir = Path(__file__).resolve().parent / "data" / "bin"
        shim_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            shim_path = shim_dir / "ffmpeg.exe"
            if not shim_path.exists() or shim_path.stat().st_size != ffmpeg_exe.stat().st_size:
                shutil.copy2(ffmpeg_exe, shim_path)
        else:
            shim_path = shim_dir / "ffmpeg"
            shim_path.write_text(f'#!/bin/sh\nexec "{ffmpeg_exe}" "$@"\n', encoding="utf-8")
            shim_path.chmod(shim_path.stat().st_mode | S_IXUSR | S_IXGRP | S_IXOTH)

        os.environ["PATH"] = f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        if shutil.which("ffmpeg"):
            print(f"[transnet] using imageio-ffmpeg shim: {ffmpeg_exe}")
            return True
        return False

    def _load_weights_if_available(self, model) -> None:
        weights_path = self.weights_path or self._find_weights_path()
        if not weights_path:
            print("[transnet] no explicit weights file found; using package defaults")
            return

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required to load TransNetV2 weights.") from exc

        print(f"[transnet] loading weights: {weights_path}")
        state_dict = torch.load(str(weights_path), map_location=model.device)
        model.load_state_dict(state_dict)

    @staticmethod
    def _find_weights_path() -> Path | None:
        env_path = os.getenv("TRANSNET_WEIGHTS")
        if env_path:
            path = Path(env_path).expanduser()
            if path.exists():
                return path

        local_candidates = [
            Path.cwd() / "transnetv2-pytorch-weights.pth",
            Path(__file__).resolve().parent / "transnetv2-pytorch-weights.pth",
        ]
        for candidate in local_candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _normalize_scenes(raw_scenes, fps: float, duration: float) -> list[Shot]:
        shots: list[Shot] = []
        for idx, scene in enumerate(raw_scenes):
            if isinstance(scene, dict):
                start_time = float(scene.get("start_time", 0.0) or 0.0)
                end_time = float(scene.get("end_time", duration) or duration)
                start_frame = int(scene.get("start_frame", round(start_time * fps)) or 0)
                end_frame = int(scene.get("end_frame", round(end_time * fps)) or start_frame)
                shot_id = int(scene.get("shot_id", idx))
            else:
                start_frame = int(scene[0])
                end_frame = int(scene[1])
                start_time = start_frame / fps
                end_time = end_frame / fps
                shot_id = idx

            start_time = max(0.0, min(start_time, duration))
            end_time = max(start_time, min(end_time, duration))
            shots.append(
                Shot(
                    shot_id=shot_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

        shots.sort(key=lambda shot: (shot.start_time, shot.end_time))
        for new_id, shot in enumerate(shots):
            shot.shot_id = new_id
        return shots
