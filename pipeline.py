from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import local

from utils import ensure_str, ensure_str_list, safe_stem, seconds_to_timestamp, write_json


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"


class IndexTimer:
    def __init__(self):
        self.started_at = time.perf_counter()
        self.steps: list[dict] = []

    @staticmethod
    def format_duration(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        whole_seconds = int(round(seconds))
        hours, remainder = divmod(whole_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @contextmanager
    def step(self, name: str, **metadata):
        print(f"[time] start {name}", flush=True)
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
                "elapsed_hms": self.format_duration(elapsed),
                "status": status,
                **metadata,
            }
            if error:
                record["error"] = error
            self.steps.append(record)
            print(f"[time] {name}: {elapsed:.2f}s ({status})", flush=True)

    def total_sec(self) -> float:
        return time.perf_counter() - self.started_at

    def to_dict(self) -> dict:
        total = self.total_sec()
        return {
            "total_elapsed_sec": round(total, 3),
            "total_elapsed_hms": self.format_duration(total),
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


def read_optional_text(path_value: str | None) -> str:
    if not path_value:
        return ""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"character glossary file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def character_glossary_sidecar(video_path: Path) -> Path | None:
    candidates = [
        video_path.with_suffix(".characters.txt"),
        video_path.with_suffix(".cast.txt"),
        video_path.with_suffix(".glossary.txt"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


def worker_count(value: int | None) -> int:
    return max(1, int(value or 1))


def chunked(items: list, size: int):
    size = max(1, int(size or 1))
    for start in range(0, len(items), size):
        yield items[start : start + size]


def record_sort_key(record: dict) -> tuple[int, int]:
    shot = record.get("shot") if isinstance(record, dict) else None
    if isinstance(shot, dict) and shot.get("shot_id") is not None:
        return (0, int(shot["shot_id"]))
    return (1, 0)


def compact_scene_for_entities(record: dict) -> dict:
    shot = record.get("shot") if isinstance(record, dict) else {}
    summary = record.get("summary") if isinstance(record, dict) else {}
    frame_description = record.get("frame_description") if isinstance(record, dict) else {}
    subtitles = record.get("shot_subtitles") if isinstance(record, dict) else []
    shot_start = float(shot.get("start_time") or 0.0)
    shot_end = float(shot.get("end_time") or 0.0)
    subtitle_text = " ".join(
        ensure_str(item.get("text"))
        for item in subtitles
        if isinstance(item, dict) and ensure_str(item.get("text"))
    )
    return {
        "shot_id": shot.get("shot_id"),
        "timestamp": f"{seconds_to_timestamp(shot_start)} ~ {seconds_to_timestamp(shot_end)}",
        "summary": ensure_str(summary.get("summary"))[:700],
        "people": ensure_str_list(summary.get("people"))[:12],
        "visual_keywords": ensure_str_list(summary.get("visual_keywords"))[:18],
        "dialogue_keywords": ensure_str_list(summary.get("dialogue_keywords"))[:18],
        "frame_description": ensure_str(frame_description.get("frame_description"))[:700],
        "subtitles": subtitle_text[:700],
    }


def normalize_entity_candidates(entity_data: dict) -> list[dict]:
    characters = entity_data.get("characters") if isinstance(entity_data, dict) else []
    if not isinstance(characters, list):
        return []
    normalized = []
    seen = set()
    for item in characters:
        if not isinstance(item, dict):
            continue
        name = ensure_str(item.get("canonical_name") or item.get("name"))
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        shot_ids = []
        for shot_id in item.get("evidence_shot_ids") or []:
            try:
                shot_ids.append(int(shot_id))
            except (TypeError, ValueError):
                continue
        normalized.append(
            {
                "canonical_name": name,
                "aliases": ensure_str_list(item.get("aliases"))[:12],
                "visual_clues": ensure_str_list(item.get("visual_clues"))[:16],
                "evidence_shot_ids": sorted(set(shot_ids))[:80],
                "confidence": ensure_str(item.get("confidence")) or "medium",
            }
        )
    return normalized


def merge_entity_candidates_locally(candidates: list[dict], max_candidates: int = 24) -> list[dict]:
    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    merged: list[dict] = []

    def aliases_for(candidate: dict) -> set[str]:
        values = [candidate.get("canonical_name"), *ensure_str_list(candidate.get("aliases"))]
        return {
            " ".join(ensure_str(value).split()).casefold()
            for value in values
            if ensure_str(value)
        }

    for candidate in normalize_entity_candidates({"characters": candidates}):
        candidate_aliases = aliases_for(candidate)
        target = None
        for existing in merged:
            if candidate_aliases & aliases_for(existing):
                target = existing
                break

        if target is None:
            merged.append(candidate)
            continue

        existing_aliases = {
            " ".join(alias.split()).casefold()
            for alias in ensure_str_list(target.get("aliases"))
        }
        aliases = ensure_str_list(target.get("aliases"))
        for alias in ensure_str_list(candidate.get("aliases")) + [candidate["canonical_name"]]:
            alias_key = " ".join(alias.split()).casefold()
            if alias_key and alias_key not in existing_aliases and alias_key != target["canonical_name"].casefold():
                existing_aliases.add(alias_key)
                aliases.append(alias)
        target["aliases"] = aliases[:12]

        clues = ensure_str_list(target.get("visual_clues"))
        seen_clues = {" ".join(clue.split()).casefold() for clue in clues}
        for clue in ensure_str_list(candidate.get("visual_clues")):
            clue_key = " ".join(clue.split()).casefold()
            if clue_key and clue_key not in seen_clues:
                seen_clues.add(clue_key)
                clues.append(clue)
        target["visual_clues"] = clues[:16]

        shot_ids = set(target.get("evidence_shot_ids") or [])
        shot_ids.update(candidate.get("evidence_shot_ids") or [])
        target["evidence_shot_ids"] = sorted(shot_ids)[:80]

        if confidence_rank.get(candidate.get("confidence"), 1) > confidence_rank.get(target.get("confidence"), 1):
            target["confidence"] = candidate.get("confidence") or target.get("confidence") or "medium"

    merged.sort(
        key=lambda item: (
            -len(item.get("evidence_shot_ids") or []),
            -confidence_rank.get(item.get("confidence"), 1),
            ensure_str(item.get("canonical_name")).casefold(),
        )
    )
    return merged[:max_candidates]


def normalize_entity_assignments(assignment_data: dict) -> dict[int, list[dict]]:
    assignments = assignment_data.get("assignments") if isinstance(assignment_data, dict) else []
    if not isinstance(assignments, list):
        return {}
    by_shot: dict[int, list[dict]] = {}
    for item in assignments:
        if not isinstance(item, dict):
            continue
        try:
            shot_id = int(item.get("shot_id"))
        except (TypeError, ValueError):
            continue
        names = ensure_str_list(item.get("names"))[:6]
        if not names:
            continue
        by_shot.setdefault(shot_id, []).append(
            {
                "names": names,
                "evidence": ensure_str(item.get("evidence"))[:500],
                "confidence": ensure_str(item.get("confidence")) or "medium",
            }
        )
    return by_shot


def merge_assignment_maps(items: list[dict[int, list[dict]]]) -> dict[int, list[dict]]:
    merged: dict[int, list[dict]] = {}
    for item in items:
        for shot_id, assignments in item.items():
            existing_names = {
                name.casefold()
                for assignment in merged.setdefault(shot_id, [])
                for name in assignment.get("names", [])
            }
            for assignment in assignments:
                names = [
                    name
                    for name in assignment.get("names", [])
                    if name.casefold() not in existing_names
                ]
                if not names:
                    continue
                existing_names.update(name.casefold() for name in names)
                merged[shot_id].append({**assignment, "names": names})
    return merged


def enrich_records_with_entities(records: list[dict], assignments_by_shot: dict[int, list[dict]]) -> list[dict]:
    changed = []
    for record in records:
        shot = record.get("shot") if isinstance(record, dict) else {}
        try:
            shot_id = int(shot.get("shot_id"))
        except (TypeError, ValueError):
            continue
        assignments = assignments_by_shot.get(shot_id)
        if not assignments:
            continue

        summary = record.setdefault("summary", {})
        names = []
        evidence_parts = []
        for assignment in assignments:
            for name in assignment.get("names", []):
                if name and name not in names:
                    names.append(name)
            evidence = ensure_str(assignment.get("evidence"))
            if evidence:
                evidence_parts.append(evidence)

        if not names:
            continue

        people = ensure_str_list(summary.get("people"))
        for name in names:
            if name not in people:
                people.append(name)
        summary["people"] = people

        dialogue_keywords = ensure_str_list(summary.get("dialogue_keywords"))
        for name in names:
            if name not in dialogue_keywords:
                dialogue_keywords.append(name)
        summary["dialogue_keywords"] = dialogue_keywords

        character_candidates = summary.get("character_candidates")
        if not isinstance(character_candidates, list):
            character_candidates = []
        character_candidates.extend(assignments)
        summary["character_candidates"] = character_candidates

        search_text = ensure_str(summary.get("search_text") or summary.get("summary"))
        entity_line = f"등장인물 후보: {', '.join(names)}"
        evidence_line = ""
        if evidence_parts:
            evidence_line = f"등장인물 근거: {' / '.join(evidence_parts[:2])}"
        additions = "\n".join(part for part in [entity_line, evidence_line] if part)
        if additions and additions not in search_text:
            summary["search_text"] = "\n".join(part for part in [search_text, additions] if part)
        changed.append(record)
    return changed


def payload_from_enriched_record(
    record: dict,
    *,
    existing_payload: dict | None = None,
    embedding_model: str,
    embedding_backend: str,
) -> dict:
    shot = record["shot"]
    keyframe = record["keyframe"]
    frame_description = record.get("frame_description", {})
    summary = record.get("summary", {})
    payload = dict(existing_payload or {})
    payload.update(
        {
        "video_path": str(payload.get("video_path") or record.get("video_path") or ""),
        "shot_id": shot.get("shot_id"),
        "shot_start_sec": shot.get("start_time"),
        "shot_end_sec": shot.get("end_time"),
        "keyframe_timestamp_sec": keyframe.get("timestamp_sec"),
        "image_path": keyframe.get("image_path"),
        "frame_description": ensure_str(frame_description.get("frame_description")),
        "shot_subtitles": record.get("shot_subtitles") or [],
        "summary": ensure_str(summary.get("summary")),
        "action": ensure_str_list(summary.get("action")),
        "context": ensure_str(summary.get("context")),
        "emotion": ensure_str_list(summary.get("emotion")),
        "people": ensure_str_list(summary.get("people")),
        "objects": ensure_str_list(summary.get("objects")),
        "places": ensure_str_list(summary.get("places")),
        "visual_keywords": ensure_str_list(summary.get("visual_keywords")),
        "dialogue_keywords": ensure_str_list(summary.get("dialogue_keywords")),
        "character_candidates": summary.get("character_candidates") or [],
        "search_text": ensure_str(summary.get("search_text") or summary.get("summary")),
        "vlm_model": payload.get("vlm_model") or record.get("vlm_model"),
        "llm_model": payload.get("llm_model") or record.get("llm_model"),
        "embedding_model": embedding_model,
        "embedding_backend": embedding_backend,
        }
    )
    return payload


def cmd_index(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    from embedding import DEFAULT_LOCAL_EMBEDDING_MODEL, create_summary_embedder
    from hf_clients import SummaryLLMClient, UnifiedSceneClient, VideoVLMClient
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
    run_dir.mkdir(parents=True, exist_ok=True)

    timer = IndexTimer()
    print(f"[index] video={video_path}")
    print(f"[index] run_dir={run_dir}")
    if args.api_mode == "unified" and args.skip_llm_summary:
        print(
            "[warn] --skip-llm-summary is ignored with --api-mode unified; "
            "unified mode uses one multimodal scene-summary call.",
            flush=True,
        )

    store = None
    whisper_extractor = None
    completed_ok = False
    try:
        with timer.step("qdrant_setup", qdrant_path=str(qdrant_path)):
            store = QdrantSummaryStore(qdrant_path=qdrant_path)

        subtitles_path = run_dir / "subtitles.json"
        shots_path = run_dir / "shots.json"
        whisper_initial_prompt = (
            os.getenv("WHISPER_INITIAL_PROMPT") or args.whisper_initial_prompt or ""
        )

        if args.resume_run and subtitles_path.exists():
            with timer.step("load_subtitles", source=str(subtitles_path)):
                subtitles = [SubtitleSegment(**item) for item in read_json(subtitles_path)]
        else:
            with timer.step("whisper", model=args.whisper_model, device=args.whisper_device):
                whisper_extractor = WhisperSubtitleExtractor(
                    model_size=args.whisper_model,
                    language=args.language,
                    device=args.whisper_device,
                    compute_type=args.whisper_compute_type,
                    vad_filter=args.whisper_vad,
                    beam_size=args.whisper_beam_size,
                    initial_prompt=whisper_initial_prompt,
                )
                subtitles = whisper_extractor.transcribe(video_path)
                print(f"[index] saving subtitles={subtitles_path}", flush=True)
                write_json(subtitles_path, [segment.to_dict() for segment in subtitles])
                print(f"[index] saved subtitles: {len(subtitles)} segments", flush=True)

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
        embedding_backend = (os.getenv("EMBEDDING_BACKEND") or args.embedding_backend).strip().lower()
        embedding_provider = os.getenv("HF_EMBEDDING_PROVIDER") or hf_provider
        vlm_model = os.getenv("HF_VLM_MODEL", args.vlm_model)
        llm_model = os.getenv("HF_LLM_MODEL", args.llm_model)
        if embedding_backend == "local":
            embedding_model = os.getenv(
                "LOCAL_EMBEDDING_MODEL",
                args.local_embedding_model or DEFAULT_LOCAL_EMBEDDING_MODEL,
            )
            embedding_provider = "local"
        else:
            embedding_model = os.getenv("HF_EMBEDDING_MODEL", args.embedding_model)
        local_embedding_device = os.getenv("LOCAL_EMBEDDING_DEVICE", args.local_embedding_device)
        local_embedding_batch_size = max(
            1,
            int(os.getenv("LOCAL_EMBEDDING_BATCH_SIZE", args.local_embedding_batch_size)),
        )
        local_embedding_max_length = int(
            os.getenv("LOCAL_EMBEDDING_MAX_LENGTH", args.local_embedding_max_length)
        )
        character_glossary = os.getenv("CHARACTER_GLOSSARY", "").strip()
        character_glossary_path = os.getenv("CHARACTER_GLOSSARY_PATH") or args.character_glossary
        if not character_glossary and not character_glossary_path:
            sidecar_path = character_glossary_sidecar(video_path)
            if sidecar_path is not None:
                character_glossary_path = str(sidecar_path)
        if character_glossary_path:
            character_glossary = read_optional_text(character_glossary_path)

        effective_llm_model = "skipped" if args.skip_llm_summary else llm_model
        with timer.step(
            "client_setup",
            vlm_model=vlm_model,
            llm_model=effective_llm_model,
            embedding_model=embedding_model,
            embedding_backend=embedding_backend,
            character_glossary=bool(character_glossary),
        ):
            vlm = VideoVLMClient(token=hf_token, model=vlm_model, provider=vlm_provider)
            unified = UnifiedSceneClient(
                token=hf_token,
                model=vlm_model,
                provider=vlm_provider,
                character_glossary=character_glossary,
            )
            llm = None
            if not args.skip_llm_summary:
                llm = SummaryLLMClient(
                    token=hf_token,
                    model=llm_model,
                    provider=llm_provider,
                    character_glossary=character_glossary,
                )
            embedder = create_summary_embedder(
                backend=embedding_backend,
                model_name=embedding_model,
                token=hf_token,
                provider=embedding_provider,
                local_device=local_embedding_device,
                local_batch_size=local_embedding_batch_size,
                local_max_length=local_embedding_max_length,
            )
            embedding_vector_size = getattr(embedder, "vector_size", None)
            if embedding_vector_size:
                store.ensure_collection_compatible(args.collection, embedding_vector_size)

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
        vlm_workers = worker_count(args.vlm_workers or api_workers)
        llm_workers = worker_count(args.llm_workers or api_workers)
        embedding_workers = worker_count(args.embedding_workers or api_workers)
        if getattr(embedder, "is_local", False):
            embedding_workers = 1
        qdrant_batch_size = worker_count(args.qdrant_batch_size)
        scene_batch_size = worker_count(args.scene_batch_size)
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
            with timer.step(
                "keyframes",
                shots=len(pending_shots),
                workers=keyframe_workers,
                save_original=args.save_original_keyframes,
            ):
                if keyframe_workers == 1:
                    keyframes_by_shot = select_keyframes_single_pass(
                        video_path=video_path,
                        output_dir=keyframe_dir,
                        candidate_stride=args.candidate_stride,
                        shots=pending_shots,
                        save_original=args.save_original_keyframes,
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
                                args.save_original_keyframes,
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
                        character_glossary=character_glossary,
                    )
                if getattr(embedder, "is_local", False):
                    worker_embedder = embedder
                else:
                    worker_embedder = create_summary_embedder(
                        backend=embedding_backend,
                        model_name=embedding_model,
                        token=hf_token,
                        provider=embedding_provider,
                        local_device=local_embedding_device,
                        local_batch_size=local_embedding_batch_size,
                        local_max_length=local_embedding_max_length,
                    )
                clients = (worker_vlm, worker_llm, worker_embedder)
                worker_state.clients = clients
            return clients

        vlm_worker_state = local()
        unified_worker_state = local()
        llm_worker_state = local()
        embedding_worker_state = local()

        def get_vlm_client():
            if vlm_workers == 1:
                return vlm

            client = getattr(vlm_worker_state, "client", None)
            if client is None:
                client = VideoVLMClient(
                    token=hf_token,
                    model=vlm_model,
                    provider=vlm_provider,
                )
                vlm_worker_state.client = client
            return client

        def get_unified_client():
            if api_workers == 1:
                return unified

            client = getattr(unified_worker_state, "client", None)
            if client is None:
                client = UnifiedSceneClient(
                    token=hf_token,
                    model=vlm_model,
                    provider=vlm_provider,
                    character_glossary=character_glossary,
                )
                unified_worker_state.client = client
            return client

        def get_llm_client():
            if args.skip_llm_summary:
                return None
            if llm_workers == 1:
                return llm

            client = getattr(llm_worker_state, "client", None)
            if client is None:
                client = SummaryLLMClient(
                    token=hf_token,
                    model=llm_model,
                    provider=llm_provider,
                    character_glossary=character_glossary,
                )
                llm_worker_state.client = client
            return client

        def get_embedding_client():
            if getattr(embedder, "is_local", False) or embedding_workers == 1:
                return embedder

            client = getattr(embedding_worker_state, "client", None)
            if client is None:
                client = create_summary_embedder(
                    backend=embedding_backend,
                    model_name=embedding_model,
                    token=hf_token,
                    provider=embedding_provider,
                    local_device=local_embedding_device,
                    local_batch_size=local_embedding_batch_size,
                    local_max_length=local_embedding_max_length,
                )
                embedding_worker_state.client = client
            return client

        def process_scene_api(scene):
            shot = scene["shot"]
            keyframe = scene["keyframe"]
            shot_subtitles = scene["shot_subtitles"]
            scene_vlm, scene_llm, scene_embedder = get_api_clients()

            frame_description = scene_vlm.describe_keyframe(
                keyframe.vlm_image_path or keyframe.image_path
            )
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

        def process_vlm_stage(scene):
            keyframe = scene["keyframe"]
            frame_description = get_vlm_client().describe_keyframe(
                keyframe.vlm_image_path or keyframe.image_path
            )
            return {
                "shot": scene["shot"],
                "keyframe": scene["keyframe"],
                "shot_subtitles": scene["shot_subtitles"],
                "frame_description": frame_description,
            }

        def process_llm_stage(vlm_result):
            frame_description = vlm_result["frame_description"]
            shot_subtitles = vlm_result["shot_subtitles"]
            if args.skip_llm_summary:
                summary = SummaryLLMClient.fallback_summary(
                    frame_description,
                    shot_subtitles,
                )
            else:
                scene_llm = get_llm_client()
                if scene_llm is None:
                    raise RuntimeError("LLM client is not initialized.")
                summary = scene_llm.summarize_scene(frame_description, shot_subtitles)
            return {
                **vlm_result,
                "summary": summary,
            }

        def process_embedding_stage(summary_result):
            embedding_text = summary_result["summary"].search_text or summary_result["summary"].summary
            vector = get_embedding_client().embed_document(embedding_text)
            return {
                **summary_result,
                "vector": vector,
            }

        def process_embedding_batch_stage(summary_results):
            scene_embedder = get_embedding_client()
            texts = [
                result["summary"].search_text or result["summary"].summary
                for result in summary_results
            ]
            embed_texts = getattr(scene_embedder, "embed_texts", None)
            if callable(embed_texts):
                vectors = embed_texts(texts)
            else:
                vectors = [scene_embedder.embed_document(text) for text in texts]
            if len(vectors) != len(summary_results):
                raise RuntimeError(
                    f"Embedding batch returned {len(vectors)} vectors for "
                    f"{len(summary_results)} texts."
                )
            return [
                {
                    **result,
                    "vector": vector,
                }
                for result, vector in zip(summary_results, vectors)
            ]

        def process_unified_stage(scene_batch):
            return get_unified_client().summarize_scenes(scene_batch)

        def process_unified_pipeline(scenes, progress) -> None:
            embedding_batch_size = (
                local_embedding_batch_size
                if getattr(embedder, "is_local", False)
                else 1
            )
            scene_batches = list(chunked(scenes, scene_batch_size))
            print(
                "[api] unified pipeline "
                f"api_workers={api_workers} scene_batch_size={scene_batch_size} "
                f"embedding_workers={embedding_workers} "
                f"embedding_batch_size={embedding_batch_size}",
                flush=True,
            )
            with (
                ThreadPoolExecutor(max_workers=api_workers) as unified_executor,
                ThreadPoolExecutor(max_workers=embedding_workers) as embedding_executor,
            ):
                active = {}
                pending_embedding = []

                def submit_embedding_batch(*, force: bool = False) -> None:
                    nonlocal pending_embedding
                    while pending_embedding and (
                        force or len(pending_embedding) >= embedding_batch_size
                    ):
                        if force:
                            batch = list(pending_embedding)
                            pending_embedding.clear()
                        else:
                            batch = pending_embedding[:embedding_batch_size]
                            del pending_embedding[:embedding_batch_size]
                        active[
                            embedding_executor.submit(
                                process_embedding_batch_stage,
                                batch,
                            )
                        ] = "embedding_batch"

                for scene_batch in scene_batches:
                    active[unified_executor.submit(process_unified_stage, scene_batch)] = "unified"

                while active or pending_embedding:
                    if pending_embedding and not any(
                        stage == "unified" for stage in active.values()
                    ):
                        submit_embedding_batch(force=True)

                    if not active:
                        continue

                    done, _pending = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        stage = active.pop(future)
                        result = future.result()
                        if stage == "unified":
                            pending_embedding.extend(result)
                            submit_embedding_batch()
                        else:
                            for item in result:
                                queue_scene_result(item)
                            progress.update(len(result))

        def process_stage_pipeline(scenes, progress) -> None:
            embedding_batch_size = (
                local_embedding_batch_size
                if getattr(embedder, "is_local", False)
                else 1
            )
            print(
                "[api] stage pipeline "
                f"vlm_workers={vlm_workers} llm_workers={llm_workers} "
                f"embedding_workers={embedding_workers} "
                f"embedding_batch_size={embedding_batch_size}",
                flush=True,
            )
            with (
                ThreadPoolExecutor(max_workers=vlm_workers) as vlm_executor,
                ThreadPoolExecutor(max_workers=llm_workers) as llm_executor,
                ThreadPoolExecutor(max_workers=embedding_workers) as embedding_executor,
            ):
                active = {}
                pending_embedding = []

                def submit_embedding_batch(*, force: bool = False) -> None:
                    nonlocal pending_embedding
                    while pending_embedding and (
                        force or len(pending_embedding) >= embedding_batch_size
                    ):
                        if force:
                            batch = list(pending_embedding)
                            pending_embedding.clear()
                        else:
                            batch = pending_embedding[:embedding_batch_size]
                            del pending_embedding[:embedding_batch_size]
                        active[
                            embedding_executor.submit(
                                process_embedding_batch_stage,
                                batch,
                            )
                        ] = "embedding_batch"

                for scene in scenes:
                    active[vlm_executor.submit(process_vlm_stage, scene)] = "vlm"

                while active or pending_embedding:
                    if pending_embedding and not any(
                        stage in {"vlm", "llm"} for stage in active.values()
                    ):
                        submit_embedding_batch(force=True)

                    done, _pending = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        stage = active.pop(future)
                        result = future.result()
                        if stage == "vlm":
                            active[llm_executor.submit(process_llm_stage, result)] = "llm"
                        elif stage == "llm":
                            pending_embedding.append(result)
                            submit_embedding_batch()
                        else:
                            for item in result:
                                queue_scene_result(item)
                            progress.update(len(result))

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
                "embedding_backend": embedding_backend,
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
                api_mode=args.api_mode,
                scene_batch_size=scene_batch_size,
                vlm_workers=vlm_workers,
                llm_workers=llm_workers,
                embedding_workers=embedding_workers,
                vlm_model=vlm_model,
                llm_model=effective_llm_model,
                embedding_model=embedding_model,
                embedding_backend=embedding_backend,
                qdrant_batch_size=qdrant_batch_size,
            ):
                try:
                    if args.api_mode == "unified":
                        with tqdm(
                            total=len(prepared_scenes),
                            desc="api",
                            unit="shot",
                        ) as progress:
                            process_unified_pipeline(prepared_scenes, progress)
                    elif args.api_mode == "stage":
                        with tqdm(
                            total=len(prepared_scenes),
                            desc="api",
                            unit="shot",
                        ) as progress:
                            process_stage_pipeline(prepared_scenes, progress)
                    elif api_workers == 1:
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
        completed_ok = True
    finally:
        timing_data = timer.to_dict()
        write_json(run_dir / "timings.json", timing_data)
        if store is not None:
            store.close()
        print(
            "[time] TOTAL elapsed="
            f"{timing_data['total_elapsed_hms']} ({timing_data['total_elapsed_sec']:.2f}s)",
            flush=True,
        )
        print(f"[time] timings={run_dir / 'timings.json'}", flush=True)
        if completed_ok and getattr(args, "_exit_after_success", False):
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


def cmd_stats(args: argparse.Namespace) -> None:
    from vector_store import QdrantSummaryStore

    store = QdrantSummaryStore(qdrant_path=Path(args.qdrant_path).expanduser().resolve())
    try:
        print(json.dumps(store.collection_stats(args.collection), ensure_ascii=False, indent=2))
    finally:
        store.close()


def cmd_entities(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    from embedding import DEFAULT_LOCAL_EMBEDDING_MODEL, create_summary_embedder
    from hf_clients import VideoEntitySweepClient
    from vector_store import QdrantSummaryStore

    load_env()

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = (BASE_DIR / run_dir).resolve()
    records_path = run_dir / "indexed_scenes.json"
    if not records_path.exists():
        raise FileNotFoundError(f"indexed scenes file not found: {records_path}")

    records = read_json(records_path)
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"No indexed scene records found in {records_path}")

    hf_token = os.getenv("HF_TOKEN", "")
    hf_provider = os.getenv("HF_PROVIDER") or None
    llm_provider = os.getenv("HF_LLM_PROVIDER") or hf_provider
    llm_model = os.getenv("HF_LLM_MODEL", args.llm_model)
    embedding_backend = (os.getenv("EMBEDDING_BACKEND") or args.embedding_backend).strip().lower()
    embedding_provider = os.getenv("HF_EMBEDDING_PROVIDER") or hf_provider
    if embedding_backend == "local":
        embedding_model = os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            args.local_embedding_model or DEFAULT_LOCAL_EMBEDDING_MODEL,
        )
        embedding_provider = "local"
    else:
        embedding_model = os.getenv("HF_EMBEDDING_MODEL", args.embedding_model)
    local_embedding_device = os.getenv("LOCAL_EMBEDDING_DEVICE", args.local_embedding_device)
    local_embedding_batch_size = max(
        1,
        int(os.getenv("LOCAL_EMBEDDING_BATCH_SIZE", args.local_embedding_batch_size)),
    )
    local_embedding_max_length = int(
        os.getenv("LOCAL_EMBEDDING_MAX_LENGTH", args.local_embedding_max_length)
    )

    timer = IndexTimer()
    print(f"[entities] run_dir={run_dir}")
    print(f"[entities] records={len(records)} collection={args.collection}")

    store = None
    existing_payload_by_id = {}
    try:
        with timer.step("entity_client_setup", llm_model=llm_model):
            entity_client = VideoEntitySweepClient(
                token=hf_token,
                model=llm_model,
                provider=llm_provider,
            )
            embedder = create_summary_embedder(
                backend=embedding_backend,
                model_name=embedding_model,
                token=hf_token,
                provider=embedding_provider,
                local_device=local_embedding_device,
                local_batch_size=local_embedding_batch_size,
                local_max_length=local_embedding_max_length,
            )

        if not args.dry_run:
            qdrant_path_value = (
                os.getenv("QDRANT_PATH")
                if args.qdrant_path == str(DEFAULT_DATA_DIR / "qdrant")
                else args.qdrant_path
            )
            qdrant_path = Path(qdrant_path_value or args.qdrant_path).expanduser().resolve()
            with timer.step("entity_qdrant_setup", qdrant_path=str(qdrant_path)):
                store = QdrantSummaryStore(qdrant_path=qdrant_path)
                existing_payload_by_id = {
                    item["id"]: item["payload"]
                    for item in store.scroll_payloads(args.collection)
                }

        compact_scenes = [compact_scene_for_entities(record) for record in records]
        compact_scenes = [scene for scene in compact_scenes if scene.get("shot_id") is not None]

        failed_candidate_chunks_path = run_dir / "entity_failed_candidate_chunks.jsonl"
        failed_assignment_chunks_path = run_dir / "entity_failed_assignment_chunks.jsonl"
        candidate_batches_path = run_dir / "entity_candidate_batches.json"

        def should_skip_entity_chunk(batch: list[dict], exc: RuntimeError) -> bool:
            message = str(exc).casefold()
            return len(batch) <= 1 or (
                len(batch) <= 5
                and (
                    "empty response" in message
                    or "did not return valid json" in message
                )
            )

        def failed_chunk_record(batch: list[dict], exc: RuntimeError) -> dict:
            shot_ids = [scene.get("shot_id") for scene in batch if scene.get("shot_id") is not None]
            return {
                "shot_ids": shot_ids,
                "size": len(batch),
                "error": str(exc).splitlines()[0],
            }

        def infer_candidate_batches(batch: list[dict]) -> list[dict]:
            try:
                data = entity_client.infer_candidates(batch)
                return [{"characters": normalize_entity_candidates(data)}]
            except RuntimeError as exc:
                if should_skip_entity_chunk(batch, exc):
                    append_jsonl(failed_candidate_chunks_path, failed_chunk_record(batch, exc))
                    print(
                        "[warn] entity candidate chunk skipped "
                        f"size={len(batch)} shots="
                        f"{[scene.get('shot_id') for scene in batch]}: "
                        f"{str(exc).splitlines()[0]}",
                        flush=True,
                    )
                    return []
                midpoint = max(1, len(batch) // 2)
                print(
                    "[warn] entity candidate chunk failed; splitting "
                    f"size={len(batch)} -> {midpoint}+{len(batch) - midpoint}: "
                    f"{str(exc).splitlines()[0]}",
                    flush=True,
                )
                return [
                    *infer_candidate_batches(batch[:midpoint]),
                    *infer_candidate_batches(batch[midpoint:]),
                ]

        def assign_entity_batches(batch: list[dict]) -> list[dict[int, list[dict]]]:
            try:
                data = entity_client.assign_entities(entities, batch)
                return [normalize_entity_assignments(data)]
            except RuntimeError as exc:
                if should_skip_entity_chunk(batch, exc):
                    append_jsonl(failed_assignment_chunks_path, failed_chunk_record(batch, exc))
                    print(
                        "[warn] entity assignment chunk skipped "
                        f"size={len(batch)} shots="
                        f"{[scene.get('shot_id') for scene in batch]}: "
                        f"{str(exc).splitlines()[0]}",
                        flush=True,
                    )
                    return []
                midpoint = max(1, len(batch) // 2)
                print(
                    "[warn] entity assignment chunk failed; splitting "
                    f"size={len(batch)} -> {midpoint}+{len(batch) - midpoint}: "
                    f"{str(exc).splitlines()[0]}",
                    flush=True,
                )
                return [
                    *assign_entity_batches(batch[:midpoint]),
                    *assign_entity_batches(batch[midpoint:]),
                ]

        if candidate_batches_path.exists():
            with timer.step("load_entity_candidate_batches", source=str(candidate_batches_path)):
                candidate_batches = read_json(candidate_batches_path)
                if not isinstance(candidate_batches, list):
                    candidate_batches = []
        else:
            candidate_batches = []
            with timer.step("entity_candidate_sweep", scenes=len(compact_scenes), chunk_size=args.entity_chunk_size):
                for batch in tqdm(
                    list(chunked(compact_scenes, args.entity_chunk_size)),
                    desc="entity-candidates",
                    unit="chunk",
                ):
                    candidate_batches.extend(infer_candidate_batches(batch))
                    write_json(candidate_batches_path, candidate_batches)

        all_candidates = [
            character
            for batch in candidate_batches
            for character in batch.get("characters", [])
        ]
        with timer.step("entity_candidate_merge", candidates=len(all_candidates)):
            if not all_candidates:
                entities = {"characters": []}
            else:
                entities = {"characters": merge_entity_candidates_locally(all_candidates)}
            write_json(run_dir / "video_entities.json", entities)
            print(f"[entities] candidates={len(entities['characters'])}")

        assignment_maps = []
        with timer.step("entity_scene_assignment", scenes=len(compact_scenes), chunk_size=args.entity_chunk_size):
            if entities["characters"]:
                for batch in tqdm(
                    list(chunked(compact_scenes, args.entity_chunk_size)),
                    desc="entity-assign",
                    unit="chunk",
                ):
                    assignment_maps.extend(assign_entity_batches(batch))
            assignments_by_shot = merge_assignment_maps(assignment_maps)
            assignment_rows = [
                {"shot_id": shot_id, "assignments": assignments}
                for shot_id, assignments in sorted(assignments_by_shot.items())
            ]
            write_json(run_dir / "entity_assignments.json", assignment_rows)
            print(f"[entities] assigned_shots={len(assignments_by_shot)}")

        changed_records = enrich_records_with_entities(records, assignments_by_shot)
        print(f"[entities] changed_records={len(changed_records)}")
        if args.dry_run or not changed_records:
            return

        backup_path = run_dir / "indexed_scenes.pre_entities.json"
        if not backup_path.exists():
            write_json(backup_path, read_json(records_path))
        write_json(records_path, records)

        embed_texts = getattr(embedder, "embed_texts", None)
        with timer.step(
            "entity_reembed_upsert",
            records=len(changed_records),
            qdrant_batch_size=args.qdrant_batch_size,
            embedding_model=embedding_model,
            embedding_backend=embedding_backend,
        ):
            for batch in tqdm(
                list(chunked(changed_records, args.qdrant_batch_size)),
                desc="entity-upsert",
                unit="batch",
            ):
                texts = [
                    ensure_str(record.get("summary", {}).get("search_text"))
                    or ensure_str(record.get("summary", {}).get("summary"))
                    for record in batch
                ]
                vectors = embed_texts(texts) if callable(embed_texts) else [
                    embedder.embed_document(text) for text in texts
                ]
                points = []
                for record, vector in zip(batch, vectors):
                    point_id = ensure_str(record.get("point_id"))
                    if not point_id:
                        raise RuntimeError("Cannot update entity-enriched record without point_id.")
                    payload = payload_from_enriched_record(
                        record,
                        existing_payload=existing_payload_by_id.get(point_id),
                        embedding_model=embedding_model,
                        embedding_backend=embedding_backend,
                    )
                    points.append((point_id, vector, payload))
                store.upsert_points(args.collection, points)

        print(f"[entities] done: updated {len(changed_records)} records in collection={args.collection}")
    finally:
        timing_data = timer.to_dict()
        write_json(run_dir / "entity_timings.json", timing_data)
        if store is not None:
            store.close()
        print(
            "[time] ENTITIES elapsed="
            f"{timing_data['total_elapsed_hms']} ({timing_data['total_elapsed_sec']:.2f}s)",
            flush=True,
        )
        print(f"[time] entity timings={run_dir / 'entity_timings.json'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VideoQna ingestion pipeline")
    default_qdrant_path = str(DEFAULT_DATA_DIR / "qdrant")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_index = subparsers.add_parser("index", help="Index a local video")
    p_index.add_argument("--video", required=True, help="Local video path")
    p_index.add_argument("--collection", default="video_qna_qwen06b", help="Qdrant collection")
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
        "--save-original-keyframes",
        action="store_true",
        help="Also save full-resolution keyframes. By default only the downscaled VLM/answer image is saved.",
    )
    p_index.add_argument(
        "--api-workers",
        type=int,
        default=3,
        help="Parallel workers for per-shot VLM, LLM, and embedding API calls",
    )
    p_index.add_argument(
        "--api-mode",
        default="unified",
        choices=["unified", "stage", "shot"],
        help="unified sends image+subtitles to one multimodal model call; stage/shot keep the older VLM->LLM flow",
    )
    p_index.add_argument(
        "--scene-batch-size",
        type=int,
        default=8,
        help="Number of scenes per unified multimodal API request",
    )
    p_index.add_argument("--vlm-workers", type=int, default=None, help="VLM stage workers; defaults to --api-workers")
    p_index.add_argument("--llm-workers", type=int, default=None, help="LLM stage workers; defaults to --api-workers")
    p_index.add_argument(
        "--embedding-workers",
        type=int,
        default=None,
        help="Embedding stage workers; defaults to --api-workers, forced to 1 for local embeddings",
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
    p_index.add_argument("--whisper-device", default="cuda", help="Whisper device: auto/cuda/cpu")
    p_index.add_argument(
        "--whisper-compute-type",
        default="int8",
        help="Whisper compute type, e.g. int8/int8_float32/float32/auto",
    )
    p_index.add_argument(
        "--whisper-vad",
        action="store_true",
        help="Enable faster-whisper's optional VAD filter. Disabled by default because it imports ONNX Runtime.",
    )
    p_index.add_argument(
        "--whisper-beam-size",
        type=int,
        default=5,
        help="Whisper beam size. Higher values can improve ASR quality but are slower.",
    )
    p_index.add_argument(
        "--whisper-initial-prompt",
        default="",
        help="Optional Whisper prompt with language/style/name hints, e.g. Japanese character names.",
    )

    p_index.add_argument("--transnet-threshold", type=float, default=0.5)
    p_index.add_argument("--transnet-device", default="auto", help="TransNetV2 device: auto/cuda/cpu")
    p_index.add_argument("--transnet-weights", default=None, help="Optional TransNetV2 weights path")
    p_index.add_argument("--proxy-width", type=int, default=320)
    p_index.add_argument("--candidate-stride", type=float, default=0.5)
    p_index.add_argument("--subtitle-padding", type=float, default=0.5)
    p_index.add_argument(
        "--character-glossary",
        default=None,
        help="Optional UTF-8 text file with character names, aliases, and visual clues for scene summaries",
    )
    p_index.add_argument(
        "--skip-llm-summary",
        action="store_true",
        help="Skip remote LLM summary calls and build summaries from VLM output plus subtitles",
    )

    p_index.add_argument("--vlm-model", default="Qwen/Qwen3.5-9B:together")
    p_index.add_argument("--llm-model", default="Qwen/Qwen3-8B")
    p_index.add_argument("--embedding-model", default="ibm-granite/granite-embedding-97m-multilingual-r2")
    p_index.add_argument("--embedding-backend", default="local", choices=["local", "api"])
    p_index.add_argument("--local-embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    p_index.add_argument("--local-embedding-device", default="auto", help="Local embedding device: auto/cuda/cpu")
    p_index.add_argument("--local-embedding-batch-size", type=int, default=8)
    p_index.add_argument("--local-embedding-max-length", type=int, default=2048)
    p_index.set_defaults(func=cmd_index)

    p_stats = subparsers.add_parser("stats", help="Show Qdrant collection stats")
    p_stats.add_argument("--collection", default="video_qna_qwen06b", help="Qdrant collection")
    p_stats.add_argument("--qdrant-path", default=default_qdrant_path, help="Local Qdrant path")
    p_stats.set_defaults(func=cmd_stats)

    p_entities = subparsers.add_parser(
        "entities",
        help="Infer video-level character candidates and enrich indexed scenes",
    )
    p_entities.add_argument("--run-dir", required=True, help="Run directory containing indexed_scenes.json")
    p_entities.add_argument("--collection", default="video_qna_qwen06b", help="Qdrant collection to update")
    p_entities.add_argument("--qdrant-path", default=default_qdrant_path, help="Local Qdrant path")
    p_entities.add_argument("--llm-model", default="Qwen/Qwen3.5-9B:together")
    p_entities.add_argument("--entity-chunk-size", type=int, default=50)
    p_entities.add_argument("--qdrant-batch-size", type=int, default=32)
    p_entities.add_argument("--dry-run", action="store_true", help="Write entity JSON files but do not update records or Qdrant")
    p_entities.add_argument("--embedding-model", default="ibm-granite/granite-embedding-97m-multilingual-r2")
    p_entities.add_argument("--embedding-backend", default="local", choices=["local", "api"])
    p_entities.add_argument("--local-embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    p_entities.add_argument("--local-embedding-device", default="auto", help="Local embedding device: auto/cuda/cpu")
    p_entities.add_argument("--local-embedding-batch-size", type=int, default=8)
    p_entities.add_argument("--local-embedding-max-length", type=int, default=2048)
    p_entities.set_defaults(func=cmd_entities)

    return parser


def main() -> None:
    faulthandler.enable()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    parser = build_parser()
    args = parser.parse_args()
    if os.name == "nt" and args.func is cmd_index:
        setattr(args, "_exit_after_success", True)
    try:
        args.func(args)
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exception(exc, file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        if os.name == "nt" and args.func is cmd_index:
            os._exit(1)
        raise SystemExit(1) from None
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt" and args.func is cmd_index:
        os._exit(0)


if __name__ == "__main__":
    main()
