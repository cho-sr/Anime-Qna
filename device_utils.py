from __future__ import annotations

import os
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []
_ADDED_DLL_DIRS: set[str] = set()


def _path_has_dll(name: str) -> bool:
    path_value = os.environ.get("PATH", "")
    for path_item in path_value.split(os.pathsep):
        if not path_item:
            continue
        try:
            if (Path(path_item) / name).exists():
                return True
        except OSError:
            continue
    return False


def _iter_cuda_bin_candidates():
    seen: set[Path] = set()

    for key, value in os.environ.items():
        if not key.upper().startswith("CUDA_PATH") or not value:
            continue
        path = Path(value)
        candidates = [path] if path.name.lower() == "bin" else [path / "bin", path]
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                yield candidate

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    roots = [
        Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA",
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"),
    ]
    for root in roots:
        try:
            versions = sorted(root.iterdir(), reverse=True)
        except OSError:
            continue
        for version_dir in versions:
            candidate = version_dir / "bin"
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def _ensure_windows_dll_available(name: str) -> bool:
    if os.name != "nt":
        return True
    if _path_has_dll(name):
        return True

    for candidate in _iter_cuda_bin_candidates():
        try:
            if not (candidate / name).exists():
                continue
        except OSError:
            continue

        candidate_text = str(candidate)
        if candidate_text not in _ADDED_DLL_DIRS:
            os.environ["PATH"] = candidate_text + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(candidate_text))
            _ADDED_DLL_DIRS.add(candidate_text)
            print(f"[device] whisper: added CUDA DLL path {candidate_text}")
        return True

    return False


def cuda_is_available() -> tuple[bool, str | None]:
    try:
        import torch
    except Exception:
        return False, None

    if not torch.cuda.is_available():
        return False, None

    try:
        return True, torch.cuda.get_device_name(0)
    except Exception:
        return True, None


def ctranslate2_cuda_is_available() -> bool:
    try:
        import ctranslate2
    except Exception:
        return False

    try:
        if int(ctranslate2.get_cuda_device_count()) <= 0:
            return False
    except Exception:
        return False
    return True


def whisper_cuda_runtime_is_available() -> bool:
    if not ctranslate2_cuda_is_available():
        return False
    return _ensure_windows_dll_available("cudnn_ops_infer64_8.dll")


def ctranslate2_supported_compute_types(device: str) -> set[str]:
    try:
        import ctranslate2
    except Exception:
        return set()

    try:
        return {str(item) for item in ctranslate2.get_supported_compute_types(device)}
    except Exception:
        return set()


def resolve_torch_device(requested: str | None = "auto", *, label: str = "model") -> str:
    value = (requested or "auto").strip().lower()
    if value not in {"auto", "cuda"}:
        return requested or "cpu"

    available, device_name = cuda_is_available()
    if value == "cuda":
        if available:
            suffix = f" ({device_name})" if device_name else ""
            print(f"[device] {label}: using cuda{suffix}")
        else:
            print(f"[device] {label}: cuda requested; PyTorch does not report CUDA availability")
        return "cuda"

    if available:
        suffix = f" ({device_name})" if device_name else ""
        print(f"[device] {label}: auto selected cuda{suffix}")
        return "cuda"

    print(f"[device] {label}: cuda not available; using cpu")
    return "cpu"


def resolve_whisper_device(requested: str | None = "auto") -> str:
    value = (requested or "auto").strip().lower()
    if value not in {"auto", "cuda"}:
        return requested or "cpu"

    if value == "cuda":
        if whisper_cuda_runtime_is_available():
            print("[device] whisper: using cuda (CTranslate2)")
        elif ctranslate2_cuda_is_available():
            print(
                "[device] whisper: cuda requested, but cuDNN 8 runtime DLL was not found"
            )
        else:
            print("[device] whisper: cuda requested; CTranslate2 does not report CUDA availability")
        return "cuda"

    if whisper_cuda_runtime_is_available():
        print("[device] whisper: auto selected cuda (CTranslate2)")
        return "cuda"
    if ctranslate2_cuda_is_available():
        print("[device] whisper: CTranslate2 cuda found, but cuDNN 8 DLL is missing; using cpu")
        return "cpu"

    print("[device] whisper: CTranslate2 cuda not available; using cpu")
    return "cpu"


def resolve_whisper_compute_type(requested: str | None, device: str) -> str:
    value = (requested or "auto").strip().lower()
    if value != "auto":
        return requested or "int8"

    if (device or "").lower().startswith("cuda"):
        supported = ctranslate2_supported_compute_types("cuda")
        for candidate in ("float16", "int8_float16", "int8_float32", "int8", "float32"):
            if candidate in supported:
                print(f"[device] whisper: auto selected compute_type={candidate}")
                return candidate
        print("[device] whisper: no CUDA compute type reported; using int8")
        return "int8"

    supported = ctranslate2_supported_compute_types("cpu")
    if "int8" in supported:
        print("[device] whisper: auto selected compute_type=int8")
        return "int8"
    if "float32" in supported:
        print("[device] whisper: auto selected compute_type=float32")
        return "float32"
    return "int8"
