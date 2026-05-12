from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import local

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


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] skipping invalid JSONL row {path}:{line_number}: {exc}")
    return rows


def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


def worker_count(value: int | None) -> int:
    return max(1, int(value or 1))


def record_sort_key(record: dict) -> tuple[int, int]:
    shot = record.get("shot") if isinstance(record, dict) else None
    if isinstance(shot, dict) and shot.get("shot_id") is not None:
        return (0, int(shot["shot_id"]))
    return (1, 0)


def cmd_index(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    from embedding import QwenSummaryEmbedder
    from hf_clients import SummaryLLMClient, VideoVLMClient
    from keyframe_selector import (
        select_keyframe_for_shot,
        select_keyframes_single_pass,
    )
    from models import FrameDescription, Keyframe, SceneSummary, Shot, SubtitleSegment
    from shot_detector import TransNetShotDetector
    from subtitle_context import subtitles_for_shot
    from subtitle_extractor import WhisperSubtitleExtractor
    from vector_store import QdrantSummaryStore

    load_env()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    data_dir = Path(args.data_dir).expanduser().resolve()
    qdrant_path_value = (
        os.getenv("QDRANT_PATH")
        if args.qdrant_path == str(DEFAULT_DATA_DIR / "qdrant")
        else args.qdrant_path
    )
    qdrant_path = Path(qdrant_path_value or args.qdrant_path).expanduser().resolve()
    if args.resume_run:
        run_dir = Path(args.resume_run).expanduser()
        if not run_dir.is_absolute():
            run_dir = (BASE_DIR / run_dir).resolve()
        run_id = run_dir.name
        if not run_dir.exists():
            raise FileNotFoundError(f"resume run directory not found: {run_dir}")
    else:
        run_id = f"{safe_stem(str(video_path))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = data_dir / "runs" / run_id
    keyframe_dir = data_dir / "keyframes" / run_id

    timer = IndexTimer()
    print(f"[index] video={video_path}")
    print(f"[index] run_dir={run_dir}")

    store = None
    try:
        with timer.step("qdrant_setup", qdrant_path=str(qdrant_path)):
            store = QdrantSummaryStore(qdrant_path=qdrant_path)

        subtitles_path = run_dir / "subtitles.json"
        shots_path = run_dir / "shots.json"

        if args.resume_run and subtitles_path.exists():
            with timer.step("load_subtitles", source=str(subtitles_path)):
                subtitles = [SubtitleSegment(**item) for item in read_json(subtitles_path)]
        else:
            with timer.step("whisper", model=args.whisper_model, device=args.whisper_device):
                subtitles = WhisperSubtitleExtractor(
                    model_size=args.whisper_model,
                    language=args.language,
                    device=args.whisper_device,
                    compute_type=args.whisper_compute_type,
                ).transcribe(video_path)
                write_json(subtitles_path, [segment.to_dict() for segment in subtitles])

        if args.resume_run and shots_path.exists():
            with timer.step("load_shots", source=str(shots_path)):
                shots = [Shot(**item) for item in read_json(shots_path)]
                if args.max_shots:
                    shots = shots[: args.max_shots]
        else:
            with timer.step("transnet", threshold=args.transnet_threshold, device=args.transnet_device):
                shots = TransNetShotDetector(
                    threshold=args.transnet_threshold,
                    device=args.transnet_device,
                    proxy_width=args.proxy_width,
                    weights_path=args.transnet_weights,
                ).detect(video_path, work_dir=run_dir / "transnet")
                if args.max_shots:
                    shots = shots[: args.max_shots]
                write_json(shots_path, [shot.to_dict() for shot in shots])

        hf_token = os.getenv("HF_TOKEN", "")
        hf_provider = os.getenv("HF_PROVIDER") or None
        vlm_provider = os.getenv("HF_VLM_PROVIDER") or hf_provider
        llm_provider = os.getenv("HF_LLM_PROVIDER") or hf_provider
        embedding_provider = os.getenv("HF_EMBEDDING_PROVIDER") or hf_provider
        vlm_model = os.getenv("HF_VLM_MODEL", args.vlm_model)
        llm_model = os.getenv("HF_LLM_MODEL", args.llm_model)
        embedding_model = os.getenv("HF_EMBEDDING_MODEL", args.embedding_model)

        effective_llm_model = "skipped" if args.skip_llm_summary else llm_model
        with timer.step(
            "client_setup",
            vlm_model=vlm_model,
            llm_model=effective_llm_model,
            embedding_model=embedding_model,
        ):
            vlm = VideoVLMClient(token=hf_token, model=vlm_model, provider=vlm_provider)
            llm = None
            if not args.skip_llm_summary:
                llm = SummaryLLMClient(token=hf_token, model=llm_model, provider=llm_provider)
            embedder = QwenSummaryEmbedder(
                model_name=embedding_model,
                token=hf_token,
                provider=embedding_provider,
            )

        records_path = run_dir / "indexed_scenes.json"
        api_results_path = run_dir / "api_results.jsonl"
        records = read_json(records_path) if args.resume_run and records_path.exists() else []
        processed_shot_ids = {
            int(item["shot"]["shot_id"])
            for item in records
            if isinstance(item, dict)
            and isinstance(item.get("shot"), dict)
            and item["shot"].get("shot_id") is not None
        }
        if processed_shot_ids:
            print(f"[index] resume: skipping completed shots={sorted(processed_shot_ids)}")

        def api_checkpoint_record(result) -> dict:
            return {
                "shot": result["shot"].to_dict(),
                "keyframe": result["keyframe"].to_dict(),
                "frame_description": result["frame_description"].to_dict(),
                "summary": result["summary"].to_dict(),
                "shot_subtitles": [
                    segment.to_dict() for segment in result["shot_subtitles"]
                ],
                "vector": result["vector"],
            }

        def result_from_api_checkpoint(item: dict):
            shot = Shot(**item["shot"])
            return {
                "shot": shot,
                "keyframe": Keyframe(**item["keyframe"]),
                "shot_subtitles": [
                    SubtitleSegment(**segment)
                    for segment in item.get("shot_subtitles", [])
                ],
                "frame_description": FrameDescription(**item["frame_description"]),
                "summary": SceneSummary(**item["summary"]),
                "vector": item["vector"],
            }

        checkpoint_by_shot = {}
        for item in read_jsonl(api_results_path):
            try:
                result = result_from_api_checkpoint(item)
            except Exception as exc:
                print(f"[warn] skipping invalid API checkpoint row: {type(exc).__name__}: {exc}")
                continue

            shot_id = result["shot"].shot_id
            if shot_id not in processed_shot_ids:
                checkpoint_by_shot[shot_id] = result

        if checkpoint_by_shot:
            print(
                "[index] resume: restoring API results for shots="
                f"{sorted(checkpoint_by_shot)}"
            )

        keyframe_workers = worker_count(args.keyframe_workers)
        api_workers = worker_count(args.api_workers)
        qdrant_batch_size = worker_count(args.qdrant_batch_size)
        pending_shots = []
        for shot in shots:
            shot_label = f"shot_{shot.shot_id:04d}"
            if shot.shot_id in processed_shot_ids:
                print(f"[index] skip {shot_label}: already in indexed_scenes.json")
                continue
            if shot.shot_id in checkpoint_by_shot:
                print(f"[index] skip {shot_label}: found in api_results.jsonl")
                continue
            pending_shots.append(shot)

        keyframes_by_shot = {}
        if pending_shots:
            with timer.step("keyframes", shots=len(pending_shots), workers=keyframe_workers):
                if keyframe_workers == 1:
                    keyframes_by_shot = select_keyframes_single_pass(
                        video_path=video_path,
                        output_dir=keyframe_dir,
                        candidate_stride=args.candidate_stride,
                        shots=pending_shots,
                    )
                else:
                    with ThreadPoolExecutor(max_workers=keyframe_workers) as executor:
                        futures = {
                            executor.submit(
                                select_keyframe_for_shot,
                                video_path,
                                keyframe_dir,
                                args.candidate_stride,
                                shot,
                            ): shot
                            for shot in pending_shots
                        }
                        for future in tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc="keyframes",
                            unit="shot",
                        ):
                            shot = futures[future]
                            keyframes_by_shot[shot.shot_id] = future.result()

        prepared_scenes = []
        for shot in pending_shots:
            shot_subtitles = subtitles_for_shot(
                subtitles,
                shot,
                padding_sec=args.subtitle_padding,
            )
            prepared_scenes.append(
                {
                    "shot": shot,
                    "keyframe": keyframes_by_shot[shot.shot_id],
                    "shot_subtitles": shot_subtitles,
                }
            )

        worker_state = local()

        def get_api_clients():
            if api_workers == 1:
                return vlm, llm, embedder

            clients = getattr(worker_state, "clients", None)
            if clients is None:
                worker_vlm = VideoVLMClient(
                    token=hf_token,
                    model=vlm_model,
                    provider=vlm_provider,
                )
                worker_llm = None
                if not args.skip_llm_summary:
                    worker_llm = SummaryLLMClient(
                        token=hf_token,
                        model=llm_model,
                        provider=llm_provider,
                    )
                worker_embedder = QwenSummaryEmbedder(
                    model_name=embedding_model,
                    token=hf_token,
                    provider=embedding_provider,
                )
                clients = (worker_vlm, worker_llm, worker_embedder)
                worker_state.clients = clients
            return clients

        def process_scene_api(scene):
            shot = scene["shot"]
            keyframe = scene["keyframe"]
            shot_subtitles = scene["shot_subtitles"]
            scene_vlm, scene_llm, scene_embedder = get_api_clients()

            frame_description = scene_vlm.describe_keyframe(keyframe.image_path)
            if args.skip_llm_summary:
                summary = SummaryLLMClient.fallback_summary(
                    frame_description,
                    shot_subtitles,
                )
            else:
                if scene_llm is None:
                    raise RuntimeError("LLM client is not initialized.")
                summary = scene_llm.summarize_scene(frame_description, shot_subtitles)

            embedding_text = summary.search_text or summary.summary
            vector = scene_embedder.embed_document(embedding_text)
            return {
                "shot": shot,
                "keyframe": keyframe,
                "shot_subtitles": shot_subtitles,
                "frame_description": frame_description,
                "summary": summary,
                "vector": vector,
            }

        def scene_payload(result):
            shot = result["shot"]
            keyframe = result["keyframe"]
            shot_subtitles = result["shot_subtitles"]
            frame_description = result["frame_description"]
            summary = result["summary"]
            return {
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
                "people": summary.people,
                "objects": summary.objects,
                "places": summary.places,
                "visual_keywords": summary.visual_keywords,
                "dialogue_keywords": summary.dialogue_keywords,
                "search_text": summary.search_text,
                "vlm_model": vlm_model,
                "llm_model": llm_model,
                "embedding_model": embedding_model,
            }

        def scene_record(point_id: str, result):
            return {
                "point_id": point_id,
                "shot": result["shot"].to_dict(),
                "keyframe": result["keyframe"].to_dict(),
                "frame_description": result["frame_description"].to_dict(),
                "summary": result["summary"].to_dict(),
                "shot_subtitles": [
                    segment.to_dict() for segment in result["shot_subtitles"]
                ],
            }

        pending_upserts = []
        batch_state = {"index": 0}

        def flush_scene_results() -> None:
            if not pending_upserts:
                return

            batch = list(pending_upserts)
            pending_upserts.clear()
            payloads = [scene_payload(result) for result in batch]
            scenes = [
                (result["vector"], payload)
                for result, payload in zip(batch, payloads)
            ]
            batch_state["index"] += 1
            with timer.step(
                "qdrant_upsert_batch",
                batch=batch_state["index"],
                shots=len(batch),
            ):
                point_ids = store.upsert_scenes(args.collection, scenes)

            for point_id, result in zip(point_ids, batch):
                records.append(scene_record(point_id, result))
            records.sort(key=record_sort_key)
            write_json(records_path, records)

        def queue_scene_result(result, *, persist_checkpoint: bool = True) -> None:
            if persist_checkpoint:
                append_jsonl(api_results_path, api_checkpoint_record(result))
            pending_upserts.append(result)
            if len(pending_upserts) >= qdrant_batch_size:
                flush_scene_results()

        if checkpoint_by_shot:
            with timer.step(
                "api_checkpoint_restore",
                shots=len(checkpoint_by_shot),
                qdrant_batch_size=qdrant_batch_size,
            ):
                for result in checkpoint_by_shot.values():
                    queue_scene_result(result, persist_checkpoint=False)
                flush_scene_results()

        if prepared_scenes:
            with timer.step(
                "shot_api",
                shots=len(prepared_scenes),
                workers=api_workers,
                vlm_model=vlm_model,
                llm_model=effective_llm_model,
                embedding_model=embedding_model,
                qdrant_batch_size=qdrant_batch_size,
            ):
                try:
                    if api_workers == 1:
                        for scene in tqdm(prepared_scenes, desc="api", unit="shot"):
                            queue_scene_result(process_scene_api(scene))
                    else:
                        with ThreadPoolExecutor(max_workers=api_workers) as executor:
                            futures = {
                                executor.submit(process_scene_api, scene): scene["shot"]
                                for scene in prepared_scenes
                            }
                            for future in tqdm(
                                as_completed(futures),
                                total=len(futures),
                                desc="api",
                                unit="shot",
                            ):
                                queue_scene_result(future.result())
                    flush_scene_results()
                except Exception:
                    flush_scene_results()
                    raise

        print(f"[index] done: {len(records)} shots indexed into collection={args.collection}")
        print(f"[index] artifacts={run_dir}")
    finally:
        write_json(run_dir / "timings.json", timer.to_dict())
        if store is not None:
            store.close()
        print(f"[time] total: {timer.total_sec():.2f}s")
        print(f"[time] timings={run_dir / 'timings.json'}")


def cmd_stats(args: argparse.Namespace) -> None:
    from vector_store import QdrantSummaryStore

    store = QdrantSummaryStore(qdrant_path=Path(args.qdrant_path).expanduser().resolve())
    try:
        print(json.dumps(store.collection_stats(args.collection), ensure_ascii=False, indent=2))
    finally:
        store.close()


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
    p_index.add_argument(
        "--keyframe-workers",
        type=int,
        default=1,
        help="Keyframe workers. 1 uses the single-pass sampler; values above 1 use parallel seeks.",
    )
    p_index.add_argument(
        "--api-workers",
        type=int,
        default=3,
        help="Parallel workers for per-shot VLM, LLM, and embedding API calls",
    )
    p_index.add_argument(
        "--qdrant-batch-size",
        type=int,
        default=16,
        help="Number of finished shots to write in each Qdrant upsert batch",
    )
    p_index.add_argument("--resume-run", default=None, help="Reuse subtitles.json and shots.json from an existing run directory")

    p_index.add_argument("--whisper-model", default="base", help="faster-whisper model size")
    p_index.add_argument("--language", default=None, help="Whisper language code, e.g. ko/ja/en")
    p_index.add_argument("--whisper-device", default="auto", help="Whisper device: auto/cuda/cpu")
    p_index.add_argument(
        "--whisper-compute-type",
        default="auto",
        help="Whisper compute type: auto uses float16 on CUDA and int8 on CPU",
    )

    p_index.add_argument("--transnet-threshold", type=float, default=0.5)
    p_index.add_argument("--transnet-device", default="auto", help="TransNetV2 device: auto/cuda/cpu")
    p_index.add_argument("--transnet-weights", default=None, help="Optional TransNetV2 weights path")
    p_index.add_argument("--proxy-width", type=int, default=320)
    p_index.add_argument("--candidate-stride", type=float, default=0.5)
    p_index.add_argument("--subtitle-padding", type=float, default=0.5)
    p_index.add_argument(
        "--skip-llm-summary",
        action="store_true",
        help="Skip remote LLM summary calls and build summaries from VLM output plus subtitles",
    )

    p_index.add_argument("--vlm-model", default="Qwen/Qwen3.5-9B:together")
    p_index.add_argument("--llm-model", default="Qwen/Qwen3-8B")
    p_index.add_argument("--embedding-model", default="ibm-granite/granite-embedding-97m-multilingual-r2")
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
