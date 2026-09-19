"""Small hardware/model-stats helpers used by training/train.py.

Deliberately separate from scripts/check_hardware.py (system-level VRAM/GPU
detection owned by another part of the project) — this module only deals
with picking a compute dtype and reporting on an already-loaded model.

torch is imported lazily inside each function (not at module import time) so
that importing this module doesn't require torch to be installed, and so a
missing torch gives one clear error at the call site instead of an import
crash anywhere this module happens to be imported from.
"""
from __future__ import annotations


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch installed
        raise ImportError(
            "PyTorch is required for training/evaluation but is not installed. "
            "Install it (a CUDA build matching your GPU driver) before running "
            "training/train.py or training/evaluate.py."
        ) from exc
    return torch


def get_compute_dtype():
    """Returns torch.bfloat16 if the GPU supports it, else torch.float16 if a
    GPU is present, else torch.float32 on CPU-only machines."""
    torch = _require_torch()
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def get_gpu_info() -> dict:
    """Returns {'available': bool, 'name': str | None, 'total_vram_mb': float | None}."""
    torch = _require_torch()
    if not torch.cuda.is_available():
        return {"available": False, "name": None, "total_vram_mb": None}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "total_vram_mb": round(props.total_memory / (1024 ** 2), 1),
    }


def print_gpu_info() -> None:
    info = get_gpu_info()
    if info["available"]:
        print(f"GPU: {info['name']} ({info['total_vram_mb']:.0f} MiB VRAM)")
    else:
        print("GPU: none detected — running on CPU (will be very slow for fine-tuning).")


def print_model_stats(model) -> dict:
    """Prints and returns {'trainable_params', 'total_params', 'trainable_pct'}."""
    trainable_params = 0
    total_params = 0
    for _, param in model.named_parameters():
        num = param.numel()
        total_params += num
        if param.requires_grad:
            trainable_params += num

    trainable_pct = (100 * trainable_params / total_params) if total_params else 0.0
    print(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({trainable_pct:.4f}% trainable)"
    )
    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": trainable_pct,
    }
