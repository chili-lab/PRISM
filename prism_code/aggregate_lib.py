"""Shared utilities for PRISM aggregate scripts.

Used by aggregate_prism_top.py and aggregate_prism_bottom.py.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Constants ────────────────────────────────────────────────────────
C = 4  # FA, SR, AC, UV (ignore unknown)
TAG_SHORT = ["FA", "SR", "AC", "UV"]
CAT_PAIRS = ["FA-SR", "FA-AC", "FA-UV", "SR-AC", "SR-UV", "AC-UV"]
CAT_COLORS = {"FA": "#E91E63", "SR": "#9E9E9E", "AC": "#009E73", "UV": "#F0E442"}
CORR_COLOR, INCORR_COLOR = "#4CAF50", "#F44336"
SAVE_KW = dict(dpi=200, bbox_inches="tight", facecolor="white")


# ── Style ────────────────────────────────────────────────────────────

def setup_style():
    """Configure matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })


# ── Discovery & loading ─────────────────────────────────────────────

def discover_jsons(base_dir: str, filename: str = "analysis.json",
                   models: List[str] = None,
                   datasets: List[str] = None) -> List[Tuple[str, str]]:
    results = []
    base = Path(base_dir)
    for p in sorted(base.rglob(filename)):
        parts = p.relative_to(base).parts
        if len(parts) >= 4:
            model, dataset, seed = parts[-4], parts[-3], parts[-2]
            if models and model not in models:
                continue
            if datasets and dataset not in datasets:
                continue
            label = f"{model}/{dataset}/{seed}"
        elif len(parts) >= 2:
            label = "/".join(parts[:-1])
        else:
            label = str(p.relative_to(base).parent)
        results.append((label, str(p)))
    return results


def load_json(path: str) -> Optional[Dict]:
    """Load a JSON file, return None on error."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] {path}: {e}")
        return None


# ── Aggregation ──────────────────────────────────────────────────────

def aggregate(all_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute mean +/- std for each metric across runs."""
    collected = defaultdict(list)
    for m in all_metrics:
        for k, v in m.items():
            if v is not None:
                collected[k].append(v)

    result = {}
    for k, vals in collected.items():
        arr = np.array(vals)
        result[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "n": len(vals),
        }
    return result


# ── Plotting ─────────────────────────────────────────────────────────

def _bar_annotate(ax, bars, fmt=".3f"):
    """Annotate bar chart with value labels."""
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f"{h:{fmt}}", ha="center", va="bottom", fontsize=10)


def plot_matrix(mat: np.ndarray, title: str, path: str,
                cmap="Blues", vmin=0, vmax=1, labels=None):
    """Plot a C×C heatmap with cell values."""
    labels = labels or TAG_SHORT
    n = mat.shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(n):
        for j in range(n):
            color = "white" if mat[i, j] > (vmax + vmin) / 2 else "black"
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=12, color=color)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels[:n])
    ax.set_yticklabels(labels[:n])
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)


def plot_diff_matrix(mat: np.ndarray, title: str, path: str):
    """Plot a difference matrix with diverging colormap."""
    vlim = max(abs(mat.min()), abs(mat.max()), 0.01)
    plot_matrix(mat, title, path, cmap="RdBu_r", vmin=-vlim, vmax=vlim)


def renormalize_4x4(mat5: list) -> np.ndarray:
    """Extract 4×4 submatrix from 5×5 and re-normalize rows."""
    arr = np.array(mat5, dtype=np.float64)
    mat4 = arr[:C, :C]
    row_sums = mat4.sum(axis=1, keepdims=True)
    safe = np.where(row_sums > 0, row_sums, 1.0)
    return np.where(row_sums > 0, mat4 / safe, 0.0)
