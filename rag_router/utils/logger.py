"""
Logger Utility
===============
JSON-lines logging for experiment results and training metrics.
All experiment scripts append to results/training_log.jsonl so the
full experimental history is preserved.
"""

import json
import numpy as np
from datetime import datetime, timezone

from config import LOG_DIR


def _convert(obj):
    """Make numpy types JSON-serialisable."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def log_results(results: list[dict], file_path: str = "logs.json") -> None:
    """Append a list of result dicts to a JSON-lines file."""
    with open(file_path, "a", encoding="utf-8") as f:
        for r in results:
            r_clean = json.loads(json.dumps(r, default=_convert))
            f.write(json.dumps(r_clean) + "\n")


def log_training_event(event: dict) -> None:
    """Append a timestamped training event to results/training_log.jsonl."""
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    path = LOG_DIR / "training_log.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=_convert) + "\n")
