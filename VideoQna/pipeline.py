from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from utils import safe_stem, write_json


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"


class IndexTimer:
    def __init__(self):
        self.started_at = time.perf_counter()
        self.steps: list[dict] = []

    @contextmanager
    def step(self, name: str, **metadata):
        print(f"[time] start {name}")
        start = time.perf_counter()
        status = "ok"
        error = None
        try:
            yield
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = time.perf_counter() - start
            record = {
                "name": name,
                "elapsed_sec": round(elapsed, 3),
                "status": status,
                **metadata,
            }
            if error:
                record["error"] = error
            self.steps.append(record)
            print(f"[time] {name}: {elapsed:.2f}s ({status})")

    def total_sec(self) -> float:
        return time.perf_counter() - self.started_at

    def to_dict(self) -> dict:
        return {
            "total_elapsed_sec": round(self.total_sec(), 3),
            "steps": self.steps,
        }


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
    load_dotenv(BASE_DIR / ".env", override=False)


def cmd_index(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    from embedding import QwenSummaryEmbedder
    from hf_clients import SummaryLLMClient, VideoVLMClient
    from keyframe_selector import KMeansKeyframeSelector
    from shot_detector import TransNetShotDetector
    from subtitle_context import subtitles_for_shot
    from subtitle_extractor import WhisperSubtitleExtractor
    from vector_store import QdrantSummaryStore

    load_env()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    data_dir = Path(args.data_dir).expanduser().resolve()
    run_id = f"{safe_stem(str(video_path))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = data_dir / "runs" / run_id
    keyframe_dir = data_dir / "keyframes" / run_id
    qdrant_path = Path(args.qdrant_path).expanduser().resolve()

    timer = IndexTimer()
    print(f"[index] video={video_path}")
    print(f"[index] run_dir={run_dir}")

    try:
        with timer.step("whisper", model=args.whisper_model, device=args.whisper_device):
            subtitles = WhisperSubtitleExtractor(
                model_size=args.whisper_model,
                language=args.language,
                device=args.whisper_device,
                compute_type=args.whisper_compute_type,
            ).transcribe(video_path)
            write_json(run_dir / "subtitles.json", [segment.to_dict() for segment in subtitles])

        with timer.step("transnet", threshold=args.transnet_threshold, device=args.transnet_device):
            shots = TransNetShotDetector(
                threshold=args.transnet_threshold,
                device=args.transnet_device,
                proxy_width=args.proxy_width,
                weights_path=args.transnet_weights,
            ).detect(video_path, work_dir=run_dir / "transnet")
            if args.max_shots:
                shots = shots[: args.max_shots]
            write_json(run_dir / "shots.json", [shot.to_dict() for shot in shots])

        hf_token = os.getenv("HF_TOKEN", "")
        hf_provider = os.getenv("HF_PROVIDER") or None
        vlm_provider = os.getenv("HF_VLM_PROVIDER") or hf_provider
        llm_provider = os.getenv("HF_LLM_PROVIDER") or hf_provider
        embedding_provider = os.getenv("HF_EMBEDDING_PROVIDER") or hf_provider
        vlm_model = os.getenv("HF_VLM_MODEL", args.vlm_model)
        llm_model = os.getenv("HF_LLM_MODEL", args.llm_model)
        embedding_model = os.getenv("HF_EMBEDDING_MODEL", args.embedding_model)

        with timer.step("client_setup", vlm_model=vlm_model, llm_model=llm_model, embedding_model=embedding_model):
            vlm = VideoVLMClient(token=hf_token, model=vlm_model, provider=vlm_provider)
            llm = SummaryLLMClient(token=hf_token, model=llm_model, provider=llm_provider)
            embedder = QwenSummaryEmbedder(
                model_name=embedding_model,
                token=hf_token,
                provider=embedding_provider,
            )
            store = QdrantSummaryStore(qdrant_path=qdrant_path)

        records = []
        selector = KMeansKeyframeSelector(
            video_path=video_path,
            output_dir=keyframe_dir,
            candidate_stride=args.candidate_stride,
        )
        try:
            for shot in tqdm(shots, desc="shots", unit="shot"):
                shot_label = f"shot_{shot.shot_id:04d}"
                with timer.step(f"{shot_label}.keyframe", shot_id=shot.shot_id):
                    keyframe = selector.select_one(shot)
                    shot_subtitles = subtitles_for_shot(
                        subtitles,
                        shot,
                        padding_sec=args.subtitle_padding,
                    )

                with timer.step(f"{shot_label}.vlm", shot_id=shot.shot_id, model=vlm_model):
                    frame_description = vlm.describe_keyframe(keyframe.image_path)

                with timer.step(f"{shot_label}.llm_summary", shot_id=shot.shot_id, model=llm_model):
                    summary = llm.summarize_scene(frame_description, shot_subtitles)

                with timer.step(f"{shot_label}.embedding", shot_id=shot.shot_id, model=embedding_model):
                    # The vector is intentionally generated from summary only.
                    vector = embedder.embed_summary(summary.summary)

                with timer.step(f"{shot_label}.qdrant_upsert", shot_id=shot.shot_id):
                    payload = {
                        "video_path": str(video_path),
                        "shot_id": shot.shot_id,
                        "shot_start_sec": shot.start_time,
                        "shot_end_sec": shot.end_time,
                        "keyframe_timestamp_sec": keyframe.timestamp_sec,
                        "image_path": keyframe.image_path,
                        "frame_description": frame_description.frame_description,
                        "shot_subtitles": [segment.to_dict() for segment in shot_subtitles],
                        "summary": summary.summary,
                        "action": summary.action,
                        "context": summary.context,
                        "emotion": summary.emotion,
                        "vlm_model": vlm_model,
                        "llm_model": llm_model,
                        "embedding_model": embedding_model,
                    }
                    point_id = store.upsert_scene(args.collection, vector=vector, payload=payload)

                record = {
                    "point_id": point_id,
                    "shot": shot.to_dict(),
                    "keyframe": keyframe.to_dict(),
                    "frame_description": frame_description.to_dict(),
                    "summary": summary.to_dict(),
                    "shot_subtitles": [segment.to_dict() for segment in shot_subtitles],
                }
                records.append(record)
                write_json(run_dir / "indexed_scenes.json", records)
        finally:
            selector.close()

        print(f"[index] done: {len(records)} shots indexed into collection={args.collection}")
        print(f"[index] artifacts={run_dir}")
    finally:
        write_json(run_dir / "timings.json", timer.to_dict())
        print(f"[time] total: {timer.total_sec():.2f}s")
        print(f"[time] timings={run_dir / 'timings.json'}")


def cmd_stats(args: argparse.Namespace) -> None:
    from vector_store import QdrantSummaryStore

    store = QdrantSummaryStore(qdrant_path=Path(args.qdrant_path).expanduser().resolve())
    print(json.dumps(store.collection_stats(args.collection), ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VideoQna ingestion pipeline")
    default_qdrant_path = str(DEFAULT_DATA_DIR / "qdrant")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_index = subparsers.add_parser("index", help="Index a local video")
    p_index.add_argument("--video", required=True, help="Local video path")
    p_index.add_argument("--collection", default="video_qna", help="Qdrant collection")
    p_index.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Artifacts directory")
    p_index.add_argument("--qdrant-path", default=default_qdrant_path, help="Local Qdrant path")
    p_index.add_argument("--max-shots", type=int, default=None, help="Process only first N shots")

    p_index.add_argument("--whisper-model", default="base", help="faster-whisper model size")
    p_index.add_argument("--language", default=None, help="Whisper language code, e.g. ko/ja/en")
    p_index.add_argument("--whisper-device", default="cpu", help="Whisper device")
    p_index.add_argument("--whisper-compute-type", default="int8", help="Whisper compute type")

    p_index.add_argument("--transnet-threshold", type=float, default=0.5)
    p_index.add_argument("--transnet-device", default="auto")
    p_index.add_argument("--transnet-weights", default=None, help="Optional TransNetV2 weights path")
    p_index.add_argument("--proxy-width", type=int, default=320)
    p_index.add_argument("--candidate-stride", type=float, default=0.5)
    p_index.add_argument("--subtitle-padding", type=float, default=0.5)

    p_index.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    p_index.add_argument("--llm-model", default="Qwen/Qwen3-8B")
    p_index.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    p_index.set_defaults(func=cmd_index)

    p_stats = subparsers.add_parser("stats", help="Show Qdrant collection stats")
    p_stats.add_argument("--collection", default="video_qna", help="Qdrant collection")
    p_stats.add_argument("--qdrant-path", default=default_qdrant_path, help="Local Qdrant path")
    p_stats.set_defaults(func=cmd_stats)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
