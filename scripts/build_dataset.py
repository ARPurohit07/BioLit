"""Thin CLI wrapper around training/dataset_builder.py + training/prepare_dataset.py.

Kept for structural consistency with the product spec's `scripts/build_dataset.py`
entry point. All logic lives in training/ — this just forwards argv there.

    python scripts/build_dataset.py --processed-dir data/processed --output-dir data/
"""
from __future__ import annotations

import sys
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parent.parent / "training"
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from prepare_dataset import main  # noqa: E402

if __name__ == "__main__":
    main()
