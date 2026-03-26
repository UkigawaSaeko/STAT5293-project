"""Optional plotting helpers for ablation / summary JSON files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.io import read_json


def plot_bar_from_summaries(
    summary_paths: list[Path],
    metric: str = "mean_f1",
    out_path: Path | None = None,
    title: str = "Metrics",
) -> None:
    import matplotlib.pyplot as plt

    labels: list[str] = []
    values: list[float] = []
    for p in summary_paths:
        d: dict[str, Any] = read_json(p)
        labels.append(str(d.get("tag") or p.stem))
        values.append(float(d.get(metric, 0.0)))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), 4))
    ax.bar(labels, values)
    ax.set_ylabel(metric)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    if out_path is None:
        out_path = summary_paths[0].parent / f"{metric}_bar.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
