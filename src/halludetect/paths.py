"""Repository paths (data, results) resolved relative to the project root."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("HALLUDETECT_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RESULTS = ROOT / "results"
GITA_DATA = DATA / "gita"
ODQA_DATA = DATA / "odqa"
