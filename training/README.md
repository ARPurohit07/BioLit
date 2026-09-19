# BioLit fine-tuning pipeline

Conservative defaults throughout are sized for an **NVIDIA RTX 3050 Laptop
GPU (4096 MiB / 4GB VRAM)**: 4-bit QLoRA on `Qwen/Qwen2.5-1.5B-Instruct`,
batch size 1 with gradient accumulation, gradient checkpointing, and a
2048-token sequence length (see `configs/training.yaml`). On a 4GB card
this is realistically the largest model size that fits — do not switch to
the 3B/7B Qwen variants on this GPU without a lot more VRAM.

## Commands, in order

```bash
python scripts/check_hardware.py                                   # confirm VRAM / CUDA setup
python scripts/ingest.py --input data/raw --output data/processed  # parse PDFs in data/raw
python scripts/build_index.py                                      # build the retrieval index
python training/prepare_dataset.py                                 # build train/val/test JSONL
python training/train.py --config configs/training.yaml            # QLoRA SFT
python training/evaluate.py                                        # eval loss/perplexity + samples
python training/export.py                                          # merge -> GGUF -> Modelfile
python scripts/setup_ollama.py --mode finetuned                    # register with Ollama
```

Before any papers are ingested, you can still stand up something for the
app to talk to:

```bash
python scripts/setup_ollama.py --mode base   # pulls stock Qwen2.5-1.5B-Instruct,
                                              # creates "biolit-qwen-base" in Ollama
```

`training/prepare_dataset.py` (and its thin wrapper `scripts/build_dataset.py`)
also run cleanly with an empty `data/processed/` — they print a warning and
write empty-but-valid JSONL files rather than crashing, so the rest of the
pipeline can be exercised end-to-end before any real papers exist.

## Expected runtime

Step time on a 4GB laptop GPU depends heavily on paper count, dataset size,
and thermal throttling — there is no fabricated "X seconds/step" number
here. As an order-of-magnitude expectation: with QLoRA on a 1.5B model,
batch size 1 + grad accumulation 16, a single optimizer step (16
micro-batches at up to 2048 tokens) is on the order of tens of seconds to a
few minutes on this class of GPU; `training/train.py` prints an estimated
total step count before training starts so you can gauge wall-clock time
for your actual dataset. Run `scripts/check_hardware.py` if you want a
hardware-specific estimate.

## Dataset construction and response generation

`training/dataset_builder.py` builds instruction/context/provenance from
real ingested paper chunks, but does **not** invent the `response` text
itself — there is no authorized cloud LLM call in this pipeline, and no
real gold-label data exists to draw from. Instead it calls a local Ollama
model (see the module docstring in `training/dataset_builder.py`) with a
prompt that restricts the model to only what's in the supplied context,
using a SUPPORTED CLAIM / EVIDENCE / INTERPRETATION / LIMITATION structure
and forbidding invented citations. **Spot-check a sample of
`data/training/train.jsonl` by hand before training on it** — this is a
bootstrapping technique, not a substitute for review. If Ollama isn't
running, the builder emits 0 examples with a clear message instead of
fabricating text.

## If you hit CUDA out-of-memory

In order of least to most disruptive:

1. Lower `max_seq_length` in `configs/training.yaml` (e.g. 2048 -> 1024 or 512).
2. Increase `gradient_accumulation_steps` further — you generally cannot
   lower `per_device_train_batch_size` below 1, so this is the main lever
   left once batch size is already 1.
3. Confirm `load_in_4bit: true` and `gradient_checkpointing: true` are both
   still set (they're the defaults; don't disable them on a 4GB card).
4. Close other GPU-using applications (browser hardware acceleration, other
   local model servers, games) — 4GB leaves very little headroom.
5. Switch to a smaller base model in `configs/training.yaml`. On 4GB VRAM,
   `Qwen/Qwen2.5-1.5B-Instruct` is already close to the practical ceiling
   for QLoRA fine-tuning; larger Qwen variants require more VRAM than this
   GPU has.

`training/train.py` catches CUDA OOM errors and prints this same guidance
instead of a raw traceback.

## Notes on GGUF export

`training/export.py` merges the LoRA adapter into the base model in
fp16/bf16 (not the 4-bit training copy — merging into quantized weights is
lossy/unsupported), then optionally shells out to a local `llama.cpp`
checkout (`--llama-cpp-path` or `LLAMA_CPP_PATH` env var) to convert to
GGUF and quantize to Q4_K_M. llama.cpp is an external toolchain this
project does not install for you; if it isn't available, `export.py`
prints step-by-step manual instructions and exits cleanly rather than
failing — merging still succeeds either way.
