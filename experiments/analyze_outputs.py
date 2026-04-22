"""
Post-hoc analysis for baseline predictions (steps 2-6).

Outputs under outputs/analysis:
- aligned_predictions.csv
- overall_metrics.csv
- paired_differences.csv
- paired_bootstrap_tests.json
- fig_cost_vs_quality_f1.png
- fig_cost_vs_citation.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs" / "predictions"
OUT_DIR = ROOT / "outputs" / "analysis"

METHOD_FILES = {
    "no_rag": PRED_DIR / "no_rag_predictions.csv",
    "vector_rag": PRED_DIR / "vector_rag_predictions.csv",
    "toc_rag": PRED_DIR / "toc_rag_predictions.csv",
    "hybrid_rag": PRED_DIR / "hybrid_rag_predictions.csv",
}

MAIN_METRICS = [
    "f1",
    "evidence_hit_rate",
    "citation_hit_rate",
    "heuristic_hallucination",
    "prompt_tokens",
]

ALL_SUMMARY_METRICS = [
    "em",
    "f1",
    "evidence_hit_rate",
    "citation_hit_rate",
    "citation_precision",
    "heuristic_hallucination",
    "abstain",
    "prompt_tokens",
    "completion_tokens",
]


def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_predictions() -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for method, path in METHOD_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file: {path}")
        df = pd.read_csv(path)
        if "question_id" not in df.columns:
            raise ValueError(f"`question_id` not found in {path}")
        df = df.copy()
        for c in ALL_SUMMARY_METRICS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs[method] = df
    return dfs


def build_aligned_frame(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base_cols = ["question_id", "doc_id", "question"]
    methods = list(dfs.keys())
    if "no_rag" not in methods:
        raise ValueError("`no_rag` predictions are required as alignment base.")
    aligned = dfs["no_rag"][base_cols + ALL_SUMMARY_METRICS].copy()
    aligned = aligned.rename(columns={c: f"{c}_no_rag" for c in ALL_SUMMARY_METRICS})

    for method in methods:
        if method == "no_rag":
            continue
        sub = dfs[method][["question_id"] + ALL_SUMMARY_METRICS].copy()
        sub = sub.rename(columns={c: f"{c}_{method}" for c in ALL_SUMMARY_METRICS})
        aligned = aligned.merge(sub, on="question_id", how="inner")

    return aligned


def overall_metrics_table(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, df in dfs.items():
        row: dict[str, Any] = {"method": method, "n": int(len(df))}
        for metric in ALL_SUMMARY_METRICS:
            row[f"avg_{metric}"] = float(df[metric].mean()) if metric in df.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by="method")


def paired_diff_table(aligned: pd.DataFrame) -> pd.DataFrame:
    methods = sorted({c[len("f1_") :] for c in aligned.columns if c.startswith("f1_")})
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(methods):
        for b in methods[i + 1 :]:
            # keep higher-performing methods first for readability where possible
            if b == "no_rag":
                pairs.append((a, b))
            elif a == "no_rag":
                pairs.append((b, a))
            else:
                pairs.append((a, b))
    rows: list[dict[str, Any]] = []
    for a, b in pairs:
        for metric in MAIN_METRICS:
            c1 = f"{metric}_{a}"
            c2 = f"{metric}_{b}"
            d = (aligned[c1] - aligned[c2]).dropna()
            rows.append(
                {
                    "comparison": f"{a} - {b}",
                    "metric": metric,
                    "n": int(len(d)),
                    "mean_diff": float(d.mean()) if len(d) else np.nan,
                    "median_diff": float(d.median()) if len(d) else np.nan,
                    "win_rate": float((d > 0).mean()) if len(d) else np.nan,
                    "lose_rate": float((d < 0).mean()) if len(d) else np.nan,
                    "tie_rate": float((d == 0).mean()) if len(d) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def paired_bootstrap(
    aligned: pd.DataFrame,
    metric: str,
    a: str,
    b: str,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    c1 = f"{metric}_{a}"
    c2 = f"{metric}_{b}"
    d = (aligned[c1] - aligned[c2]).dropna().to_numpy(dtype=float)
    n = len(d)
    if n == 0:
        return {
            "comparison": f"{a} - {b}",
            "metric": metric,
            "n": 0,
            "mean_diff": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_two_sided": np.nan,
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = d[idx].mean(axis=1)
    obs = float(d.mean())
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975]).tolist()

    # Two-sided p-value from bootstrap null around 0.
    if obs >= 0:
        p = 2.0 * float((boot_means <= 0).mean())
    else:
        p = 2.0 * float((boot_means >= 0).mean())
    p = min(max(p, 0.0), 1.0)

    return {
        "comparison": f"{a} - {b}",
        "metric": metric,
        "n": int(n),
        "mean_diff": obs,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_two_sided": p,
    }


def run_paired_tests(aligned: pd.DataFrame) -> dict[str, Any]:
    methods = sorted({c[len("f1_") :] for c in aligned.columns if c.startswith("f1_")})
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(methods):
        for b in methods[i + 1 :]:
            if b == "no_rag":
                pairs.append((a, b))
            elif a == "no_rag":
                pairs.append((b, a))
            else:
                pairs.append((a, b))
    metrics = ["f1", "evidence_hit_rate", "citation_hit_rate"]
    results: list[dict[str, Any]] = []
    for a, b in pairs:
        for metric in metrics:
            results.append(paired_bootstrap(aligned, metric, a, b))
    return {
        "method": "paired_bootstrap",
        "n_boot": 5000,
        "results": results,
    }


def _plot_cost_quality(overall: pd.DataFrame, y_col: str, out_name: str, y_label: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = overall["avg_prompt_tokens"].to_numpy(dtype=float)
    y = overall[y_col].to_numpy(dtype=float)
    labels = overall["method"].tolist()
    ax.scatter(x, y, s=100)
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Average Prompt Tokens")
    ax.set_ylabel(y_label)
    ax.set_title(f"Cost-Quality Trade-off ({y_label})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=180)
    plt.close(fig)


def main() -> None:
    _ensure_out_dir()
    dfs = _load_predictions()
    aligned = build_aligned_frame(dfs)
    overall = overall_metrics_table(dfs)
    paired = paired_diff_table(aligned)
    tests = run_paired_tests(aligned)

    aligned.to_csv(OUT_DIR / "aligned_predictions.csv", index=False, encoding="utf-8")
    overall.to_csv(OUT_DIR / "overall_metrics.csv", index=False, encoding="utf-8")
    paired.to_csv(OUT_DIR / "paired_differences.csv", index=False, encoding="utf-8")
    (OUT_DIR / "paired_bootstrap_tests.json").write_text(
        json.dumps(tests, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _plot_cost_quality(overall, "avg_f1", "fig_cost_vs_quality_f1.png", "Average F1")
    _plot_cost_quality(
        overall,
        "avg_citation_hit_rate",
        "fig_cost_vs_citation.png",
        "Average Citation Hit Rate",
    )

    print("Analysis outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
