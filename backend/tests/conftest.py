"""Ensures the repo root is on sys.path so `from backend.app... import ...` resolves
regardless of how pytest's rootdir/package auto-detection decides to run.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
