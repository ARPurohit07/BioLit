"""Creates the Ollama model BioLit talks to.

    python scripts/setup_ollama.py --mode base        # pulls the stock Qwen instruct model,
                                                        # wraps it with the BioLit system prompt
                                                        # as "<model_name>-base" (e.g. biolit-qwen-base)
                                                        # so the app has something to talk to
                                                        # before fine-tuning is done.
    python scripts/setup_ollama.py --mode finetuned    # uses models/gguf/Modelfile produced by
                                                        # training/export.py, creates "<model_name>"
                                                        # (e.g. biolit-qwen).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "training"))
from config import load_yaml_config  # noqa: E402
from export import SYSTEM_PROMPT  # noqa: E402

OLLAMA_INSTALL_URL = "https://ollama.com/download"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the BioLit Ollama model (base or fine-tuned)")
    parser.add_argument("--mode", choices=["base", "finetuned"], required=True)
    return parser.parse_args()


def check_ollama_on_path() -> str:
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        print(
            f"[setup_ollama] `ollama` was not found on PATH. Install it from {OLLAMA_INSTALL_URL}, "
            "then re-run this script."
        )
        sys.exit(1)
    return ollama_bin


def check_daemon_reachable(host: str) -> None:
    import requests

    try:
        resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        print(
            f"[setup_ollama] Could not reach the Ollama daemon at {host} ({exc}). "
            "Start Ollama Desktop, or run `ollama serve` in another terminal, then re-run this script."
        )
        sys.exit(1)


def run_streaming(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    # ollama's progress-bar output uses UTF-8 block characters that aren't valid
    # under Windows' default cp1252 console encoding — decode as UTF-8 explicitly
    # and replace anything undecodable rather than crashing mid-pull.
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
    )
    assert process.stdout is not None
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    for line in process.stdout:
        print(line, end="")
    process.wait()
    return process.returncode


def build_base_modelfile(base_model_name: str, temperature: float, top_p: float, num_ctx: int) -> str:
    return f'''FROM {base_model_name}

PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER num_ctx {num_ctx}

SYSTEM """
{SYSTEM_PROMPT}
"""
'''


def main() -> None:
    args = parse_args()
    check_ollama_on_path()

    models_config = load_yaml_config(_REPO_ROOT / "configs" / "models.yaml")
    ollama_cfg = models_config.get("ollama", {})
    host = ollama_cfg.get("host", "http://localhost:11434")
    model_name = ollama_cfg.get("model_name", "biolit-qwen")
    base_model_name = ollama_cfg.get("base_model_name", "qwen2.5:1.5b-instruct")
    temperature = ollama_cfg.get("temperature", 0.1)
    top_p = ollama_cfg.get("top_p", 0.9)
    num_ctx = ollama_cfg.get("num_ctx", 8192)

    check_daemon_reachable(host)

    if args.mode == "base":
        target_name = f"{model_name}-base"
        print(f"Pulling {base_model_name} via Ollama (this can take a while the first time)...")
        rc = run_streaming(["ollama", "pull", base_model_name])
        if rc != 0:
            print(f"[setup_ollama] `ollama pull {base_model_name}` failed (exit code {rc}).")
            sys.exit(1)

        modelfile_dir = _REPO_ROOT / "models" / "ollama"
        modelfile_dir.mkdir(parents=True, exist_ok=True)
        modelfile_path = modelfile_dir / "base.Modelfile"
        modelfile_path.write_text(
            build_base_modelfile(base_model_name, temperature, top_p, num_ctx), encoding="utf-8"
        )
        print(f"Wrote {modelfile_path}")

        rc = run_streaming(["ollama", "create", target_name, "-f", str(modelfile_path)])
        if rc != 0:
            print(f"[setup_ollama] `ollama create {target_name}` failed (exit code {rc}).")
            sys.exit(1)
        print(f"\nCreated Ollama model '{target_name}' from the un-finetuned base model.")

    else:  # finetuned
        modelfile_path = _REPO_ROOT / "models" / "gguf" / "Modelfile"
        if not modelfile_path.exists():
            print(
                f"[setup_ollama] {modelfile_path} not found. Run `python training/export.py` first "
                "to merge the adapter, convert to GGUF, and write this Modelfile."
            )
            sys.exit(1)

        rc = run_streaming(["ollama", "create", model_name, "-f", str(modelfile_path)])
        if rc != 0:
            print(f"[setup_ollama] `ollama create {model_name}` failed (exit code {rc}).")
            sys.exit(1)
        print(f"\nCreated Ollama model '{model_name}' from the fine-tuned GGUF export.")


if __name__ == "__main__":
    main()
