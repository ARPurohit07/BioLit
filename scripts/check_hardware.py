"""Reports GPU/CPU/RAM/disk/Ollama availability and prints a recommended fine-tuning config.

Run with: python scripts/check_hardware.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys


def check_gpu() -> dict:
    info: dict = {"available": False, "name": None, "vram_gb": None, "cuda_version": None, "note": None}
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
            info["cuda_version"] = torch.version.cuda
        else:
            info["note"] = "torch installed but no CUDA device detected"
    except ImportError:
        info["note"] = "torch not installed, GPU details limited"
        info.update(_gpu_via_nvidia_smi())
    return info


def _gpu_via_nvidia_smi() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            name, mem = [p.strip() for p in result.stdout.strip().splitlines()[0].split(",")]
            mem_mb = float(mem.lower().replace("mib", "").strip())
            return {"available": True, "name": name, "vram_gb": round(mem_mb / 1024, 2)}
    except Exception:
        pass
    return {}


def check_ram() -> str:
    try:
        import psutil
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        return f"{total_gb:.1f} GB"
    except ImportError:
        return _ram_fallback()


def _ram_fallback() -> str:
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return f"{stat.ullTotalPhys / (1024 ** 3):.1f} GB"
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return f"{kb / (1024 ** 2):.1f} GB"
    except Exception:
        pass
    return "unknown"


def check_disk(path: str = ".") -> str:
    try:
        usage = shutil.disk_usage(path)
        return f"{usage.free / (1024 ** 3):.1f} GB free / {usage.total / (1024 ** 3):.1f} GB total"
    except Exception:
        return "unknown"


def check_ollama() -> bool:
    try:
        import requests
        resp = requests.get("http://localhost:11434", timeout=2)
        return resp.status_code < 500
    except Exception:
        return False


def recommend_config(vram_gb: float | None) -> str:
    if vram_gb is None:
        return ("VRAM unknown -> assume the most conservative config: "
                "Qwen2.5-1.5B-Instruct + QLoRA 4-bit + batch_size 1 + grad_accum 16 + seq_len 2048")
    if vram_gb < 6:
        return (f"Detected {vram_gb:.1f} GB VRAM -> limited VRAM, use conservative QLoRA config: "
                "Qwen2.5-1.5B-Instruct + QLoRA 4-bit + batch_size 1 + grad_accum 16 + seq_len 2048")
    if vram_gb <= 12:
        return (f"Detected {vram_gb:.1f} GB VRAM -> Qwen2.5-3B-Instruct + QLoRA 4-bit + "
                "batch_size 2 + grad_accum 8 + seq_len 2048")
    if vram_gb <= 16:
        return (f"Detected {vram_gb:.1f} GB VRAM -> Qwen2.5-3B/7B-Instruct + QLoRA 4-bit + "
                "batch_size 4 + grad_accum 4 + seq_len 4096")
    return (f"Detected {vram_gb:.1f} GB VRAM -> Qwen2.5-7B-Instruct + larger batch_size (8+) + "
            "grad_accum 2 + seq_len 4096; LoRA or full fine-tune feasible")


def main() -> None:
    print("=== BioLit Hardware Check ===\n")
    print(f"Python version: {sys.version.split()[0]}")

    gpu = check_gpu()
    if gpu["available"]:
        vram = f"{gpu['vram_gb']} GB VRAM" if gpu["vram_gb"] is not None else "VRAM unknown"
        print(f"GPU: {gpu['name']} ({vram})")
        if gpu["cuda_version"]:
            print(f"CUDA version: {gpu['cuda_version']}")
    else:
        print("GPU: not detected")
    if gpu["note"]:
        print(f"  note: {gpu['note']}")

    print(f"System RAM: {check_ram()}")
    print(f"Disk space: {check_disk()}")

    ollama_ok = check_ollama()
    print(f"Ollama reachable at http://localhost:11434: {'yes' if ollama_ok else 'no'}")

    print("\n=== Recommended training config ===")
    print(recommend_config(gpu["vram_gb"]))


if __name__ == "__main__":
    main()
