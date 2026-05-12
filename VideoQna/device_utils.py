from __future__ import annotations


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


def resolve_whisper_compute_type(requested: str | None, device: str) -> str:
    value = (requested or "auto").strip().lower()
    if value != "auto":
        return requested or "int8"

    if (device or "").lower().startswith("cuda"):
        return "float16"
    return "int8"
