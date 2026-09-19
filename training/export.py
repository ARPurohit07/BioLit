"""Export pipeline: merge LoRA adapter -> GGUF conversion -> Ollama Modelfile.

    python training/export.py
    python training/export.py --llama-cpp-path /path/to/llama.cpp

Steps:
  (a) Merge the LoRA adapter into the base model in fp16 and save it to
      models/merged/biolit-qwen-merged. Merging is done against an
      un-quantized (fp16/bf16) copy of the base model rather than the 4-bit
      QLoRA copy used for training — merging LoRA deltas into 4-bit weights
      is lossy/unsupported in general, so a full-precision merge is the
      correct way to get clean merged weights before quantizing to GGUF.
  (b) GGUF conversion via llama.cpp's convert_hf_to_gguf.py, if a local
      llama.cpp checkout is available (LLAMA_CPP_PATH env var or
      --llama-cpp-path). This is an optional external toolchain — if it's
      not available, we print manual instructions and exit gracefully
      rather than failing the whole export.
  (c) Write models/gguf/Modelfile from the BioLit system-prompt template
      (read from configs/models.yaml where possible), pointing at whatever
      GGUF file conversion actually produced (or the default expected
      filename if conversion was skipped).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_yaml_config, repo_root  # noqa: E402
from train import resolve_compute_dtype  # noqa: E402

DEFAULT_GGUF_FILENAME = "biolit-qwen.Q4_K_M.gguf"

SYSTEM_PROMPT = """You are BioLit, a biomedical scientific literature assistant.

Use only supplied evidence for factual claims.
Do not invent citations.
Clearly distinguish evidence from interpretation.
If evidence is insufficient, say so."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter, convert to GGUF, write Ollama Modelfile")
    parser.add_argument("--adapter_path", default=str(repo_root() / "models" / "adapters" / "biolit-qwen-lora"))
    parser.add_argument("--config", default=str(repo_root() / "configs" / "training.yaml"))
    parser.add_argument("--merged_output", default=str(repo_root() / "models" / "merged" / "biolit-qwen-merged"))
    parser.add_argument("--gguf_output_dir", default=str(repo_root() / "models" / "gguf"))
    parser.add_argument("--llama-cpp-path", default=os.environ.get("LLAMA_CPP_PATH"))
    parser.add_argument("--skip-quantize", action="store_true", help="Convert to f16 GGUF but skip Q4_K_M quantization")
    return parser.parse_args()


def merge_adapter(adapter_path: Path, config: dict, merged_output: Path) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_name = config["model_name"]
    dtype = resolve_compute_dtype(config)
    print(f"Loading base model {model_name} in {dtype} for merge (fp16/bf16, not quantized)...")
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))

    print(f"Loading LoRA adapter from {adapter_path} and merging...")
    merged = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged = merged.merge_and_unload()

    merged_output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_output))
    tokenizer.save_pretrained(str(merged_output))
    print(f"Saved merged fp16 model to {merged_output}")
    return merged_output


def _print_manual_gguf_instructions(gguf_output_dir: Path) -> None:
    print(
        "\nGGUF conversion requires a local llama.cpp checkout (an external toolchain not "
        "installed by this project). To convert manually:\n"
        "  1. git clone https://github.com/ggerganov/llama.cpp\n"
        "  2. pip install -r llama.cpp/requirements.txt\n"
        f"  3. python llama.cpp/convert_hf_to_gguf.py {repo_root() / 'models' / 'merged' / 'biolit-qwen-merged'} "
        f"--outfile {gguf_output_dir / 'biolit-qwen-f16.gguf'} --outtype f16\n"
        "  4. (optional, recommended for 4GB VRAM) quantize to Q4_K_M:\n"
        "     llama.cpp/build/bin/llama-quantize "
        f"{gguf_output_dir / 'biolit-qwen-f16.gguf'} {gguf_output_dir / DEFAULT_GGUF_FILENAME} Q4_K_M\n"
        "Then re-run `python training/export.py --llama-cpp-path /path/to/llama.cpp` "
        "(or just re-run `python scripts/setup_ollama.py --mode finetuned` once the "
        f".gguf file exists at {gguf_output_dir}).\n"
    )


def convert_to_gguf(llama_cpp_path: Optional[str], merged_output: Path, gguf_output_dir: Path, skip_quantize: bool) -> Optional[Path]:
    gguf_output_dir.mkdir(parents=True, exist_ok=True)

    if not llama_cpp_path:
        print("No --llama-cpp-path / LLAMA_CPP_PATH given; skipping automatic GGUF conversion.")
        _print_manual_gguf_instructions(gguf_output_dir)
        return None

    llama_cpp_dir = Path(llama_cpp_path)
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"convert_hf_to_gguf.py not found at {convert_script}; skipping automatic GGUF conversion.")
        _print_manual_gguf_instructions(gguf_output_dir)
        return None

    f16_output = gguf_output_dir / "biolit-qwen-f16.gguf"
    cmd = [sys.executable, str(convert_script), str(merged_output), "--outfile", str(f16_output), "--outtype", "f16"]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"GGUF conversion failed ({exc}); skipping automatic conversion.")
        _print_manual_gguf_instructions(gguf_output_dir)
        return None

    if skip_quantize:
        print(f"Wrote {f16_output} (unquantized f16). Pass without --skip-quantize to also produce Q4_K_M.")
        return f16_output

    quantize_bin = shutil.which("llama-quantize") or next(
        (str(p) for p in [
            llama_cpp_dir / "build" / "bin" / "llama-quantize",
            llama_cpp_dir / "build" / "bin" / "Release" / "llama-quantize.exe",
            llama_cpp_dir / "llama-quantize",
        ] if p.exists()),
        None,
    )
    if not quantize_bin:
        print(
            f"llama-quantize binary not found (checked PATH and common build locations under {llama_cpp_dir}). "
            f"Leaving {f16_output} unquantized — for a 4GB-VRAM GPU, quantizing to Q4_K_M is strongly recommended. "
            "Build llama.cpp (cmake) and re-run, or quantize manually."
        )
        return f16_output

    quantized_output = gguf_output_dir / DEFAULT_GGUF_FILENAME
    cmd = [quantize_bin, str(f16_output), str(quantized_output), "Q4_K_M"]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"Quantization failed ({exc}); keeping unquantized {f16_output}.")
        return f16_output

    print(f"Wrote quantized GGUF to {quantized_output}")
    return quantized_output


def write_modelfile(gguf_output_dir: Path, gguf_path: Optional[Path]) -> Path:
    gguf_filename = gguf_path.name if gguf_path else DEFAULT_GGUF_FILENAME

    temperature, top_p, num_ctx = 0.1, 0.9, 8192
    try:
        models_config = load_yaml_config(repo_root() / "configs" / "models.yaml")
        ollama_cfg = models_config.get("ollama", {})
        temperature = ollama_cfg.get("temperature", temperature)
        top_p = ollama_cfg.get("top_p", top_p)
        num_ctx = ollama_cfg.get("num_ctx", num_ctx)
    except FileNotFoundError:
        pass

    modelfile_content = f'''FROM ./{gguf_filename}

PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER num_ctx {num_ctx}

SYSTEM """
{SYSTEM_PROMPT}
"""
'''
    gguf_output_dir.mkdir(parents=True, exist_ok=True)
    modelfile_path = gguf_output_dir / "Modelfile"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    print(f"Wrote {modelfile_path}")
    return modelfile_path


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter_path)
    if not (adapter_path / "adapter_config.json").exists():
        print(f"[export] No LoRA adapter found at {adapter_path}. Run training/train.py first.")
        sys.exit(1)

    config = load_yaml_config(args.config)

    try:
        merged_output = merge_adapter(adapter_path, config, Path(args.merged_output))
    except ImportError as exc:
        print(f"[export] Missing/broken ML dependency: {exc}")
        sys.exit(1)

    gguf_path = convert_to_gguf(args.llama_cpp_path, merged_output, Path(args.gguf_output_dir), args.skip_quantize)
    write_modelfile(Path(args.gguf_output_dir), gguf_path)

    if gguf_path:
        print("\nExport complete. Next: python scripts/setup_ollama.py --mode finetuned")
    else:
        print(
            "\nMerge complete, but no GGUF file was produced automatically. Follow the manual "
            "steps above to produce models/gguf/*.gguf, then run: "
            "python scripts/setup_ollama.py --mode finetuned"
        )


if __name__ == "__main__":
    main()
