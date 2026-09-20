# BioLit fine-tuning pipeline (experiment; not used by the app)

This is an evaluated side study: QLoRA taught a small model to cite, but a better prompt on the served `qwen2.5:3b` did about as well, so nothing here is deployed. The results and reasoning are in the main README, section 18 ("Experiment: QLoRA fine-tuning").

Conservative defaults throughout are sized for an **NVIDIA RTX 3050 Laptop
GPU (4096 MiB / 4GB VRAM)**: 4-bit QLoRA on `Qwen/Qwen2.5-1.5B-Instruct`,
batch size 1 with gradient accumulation, gradient checkpointing, and a
2560-token sequence length (see `configs/training.yaml`; the citation-grounded
examples run about 2.0-2.3k tokens each). On a 4GB card
this is realistically the largest model size that fits — do not switch to
the 3B/7B Qwen variants on this GPU without a lot more VRAM.

## Commands, in order

```bash
python scripts/check_hardware.py                                   # confirm VRAM / CUDA setup
python scripts/ingest.py --input data/raw --output data/processed  # parse PDFs in data/raw
python scripts/build_index.py                                      # build the retrieval index
python training/build_cited_dataset.py --train-questions-per-chunk 3  # citation-grounded train/val/test JSONL (resumable)
python training/train.py --config configs/training.yaml            # QLoRA SFT
python training/eval_citations.py --adapter_path models/adapters/biolit-qwen-lora   # cites [n]? base vs adapter
python training/evaluate.py                                        # eval loss/perplexity + samples
python training/export.py                                          # merge -> GGUF -> Modelfile (needs llama.cpp)
python scripts/setup_ollama.py --mode finetuned                    # register with Ollama
```

Before any papers are ingested, you can still stand up something for the
app to talk to:

```bash
python scripts/setup_ollama.py --mode base   # pulls stock Qwen2.5-1.5B-Instruct,
                                              # creates "biolit-qwen-base" in Ollama
```

`training/prepare_dataset.py` (the original builder, and its thin wrapper
`scripts/build_dataset.py`) is superseded by `build_cited_dataset.py` (see below) but
still runs cleanly with an empty `data/processed/` — they print a warning and
write empty-but-valid JSONL files rather than crashing, so the rest of the
pipeline can be exercised end-to-end before any real papers exist.

## Measured runtime and memory

Measured on the development laptop (RTX 3050 Laptop GPU, 4 GB; 16 GB RAM), batch size 1,
gradient accumulation 8, examples of about 2.0-2.3k tokens:

- **About 93 s per optimizer step** (8 micro-batches). The first run (66 examples, 32 steps)
  took about 49 minutes; the second (115 examples, 42 steps) was projected at about 65 minutes.
- **VRAM sits near the limit** (about 3.9 of 4.0 GB during training). A longer sequence length is
  likely to run out of memory, which is why the dataset builder enforces a token budget.
- **RAM: about 3.5-3.9 GB** for the training process. On a 16 GB machine with a browser open this
  was enough pressure that long background jobs were killed twice; close other memory-hungry apps.
- **The GPU must be free.** Stop the backend and unload Ollama models (`ollama stop <model>`) first.

`training/train.py` prints an estimated total step count before training starts (it can overcount by
a step or two because an incomplete final accumulation batch is dropped).

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

## Citation-grounded dataset (what the reported runs used)

`training/build_cited_dataset.py` builds examples in the exact prompt format the RAG pipeline uses
(system prompt + numbered-evidence user prompt) with answers that cite `[n]`. A local Ollama teacher
drafts answers; only those passing deterministic checks are kept (valid citation ids, sentence
coverage, numbers and wording grounded in the cited evidence, per-citation relevance, no filler, no
restated questions). Splits are by paper, and retrieval for each split is restricted to its own
papers. Every attempt and rejection reason is logged to `data/training/cited_attempts.jsonl`, so
the run is resumable and auditable. See the main README (section 8) for the full method and its
limits.

Lessons that shaped it, worth knowing if you change the data:

- **Never let the trainer truncate silently.** The first dataset used whole papers as context
  (3k-32k tokens) against a 2048-token limit, so every response was cut off and the adapter never saw
  an answer. `train.py` now drops over-length examples and prints the count.
- **Compute loss on the answer only.** About 97% of each example is prompt; `completion_only_loss: true`
  masks it (verified: the loss-bearing tokens equal the answer tokens exactly).
- **Watch for overfitting on small data.** With 66 examples, eval loss was best at step 20 of 32 and
  rose after; `load_best_model_at_end` keeps the best checkpoint.

## Evaluating an adapter

```bash
python training/eval_citations.py --adapter_path models/adapters/biolit-qwen-lora        --extra_adapter v2=models/adapters/biolit-qwen-lora-v2/checkpoint-21     # base vs each adapter
python training/eval_ollama_citations.py --models qwen2.5:3b [--style_hint]      # served-model baseline
python scripts/summarize_citation_evals.py                                       # merge for the UI
```

Read the results with their caveats: a 14-prompt held-out set, and checks that reward the style the
data was built to teach. Loss values alone did not settle the comparison: the second adapter's eval loss was still
improving when it was stopped (0.473, 0.437, 0.429), yet its citation results matched the first's.
