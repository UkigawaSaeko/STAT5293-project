from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from utils.io import ensure_dir, write_json


def save_predictions_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            flat = {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()}
            w.writerow(flat)


def save_run_summary(summary: dict[str, Any], path: Path) -> None:
    write_json(path, summary)
