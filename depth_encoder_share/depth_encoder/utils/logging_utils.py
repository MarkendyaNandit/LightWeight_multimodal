"""
utils/logging_utils.py

Lightweight JSON-lines training metrics logger.

Each call to `MetricsLogger.log()` appends one JSON line to the log file.
This format is:
  - Human-readable
  - Easy to parse with pandas: `pd.read_json("metrics.jsonl", lines=True)`
  - Streaming — no file re-write needed

Also provides a simple console summary printer.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union


class MetricsLogger:
    """
    Appends training metrics as JSON lines to a log file.

    Usage
    -----
    logger = MetricsLogger("runs/metrics.jsonl")
    logger.log({"epoch": 1, "train/loss": 0.42, "val/loss": 0.38})

    To load in pandas:
    >>> import pandas as pd
    >>> df = pd.read_json("runs/metrics.jsonl", lines=True)

    Parameters
    ----------
    path : str | Path
        Path to the JSONL log file.  Parent directory is created if needed.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()

        # Write a header comment (not valid JSON — just for human readers)
        if not self.path.exists():
            with self.path.open("w") as f:
                f.write(
                    f"# VICReg Depth Encoder training log\n"
                    f"# Started: {datetime.now().isoformat()}\n"
                )

    def log(self, metrics: Dict[str, Any]) -> None:
        """
        Append one row of metrics to the log file.

        Parameters
        ----------
        metrics : dict
            Mapping of metric names to values.  All values must be
            JSON-serialisable (int, float, str, bool, None).
        """
        record = {"timestamp": round(time.time() - self._start_time, 2)}
        record.update(metrics)

        with self.path.open("a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")

    def __repr__(self) -> str:
        return f"MetricsLogger(path={self.path})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Fallback JSON serialiser for non-standard types."""
    if hasattr(obj, "item"):
        return obj.item()       # torch.Tensor scalar
    if hasattr(obj, "tolist"):
        return obj.tolist()     # numpy array
    return str(obj)


def print_summary(metrics_path: Union[str, Path]) -> None:
    """
    Print a concise summary table of training metrics from a JSONL log.

    Parameters
    ----------
    metrics_path : str | Path
        Path to the JSONL metrics file produced by MetricsLogger.
    """
    path = Path(metrics_path)
    if not path.exists():
        print(f"[Logger] No metrics file found at {path}")
        return

    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print("[Logger] No records found.")
        return

    first = records[0]
    last  = records[-1]
    print("\n" + "="*55)
    print(f"  Training Summary  ({len(records)} epochs logged)")
    print("="*55)
    for key in last:
        if key == "timestamp":
            continue
        v_first = first.get(key, "—")
        v_last  = last.get(key, "—")
        print(f"  {key:<30} {_fmt(v_first)} → {_fmt(v_last)}")
    print("="*55 + "\n")


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)
