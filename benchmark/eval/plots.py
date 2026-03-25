"""Publication-quality plotting for RS-CIL benchmark results.

Usage:
    from benchmark.eval.plots import plot_task_curves, plot_method_comparison, plot_forgetting_matrix

    # Single run — task progression
    plot_task_curves(result, save="figs/curves.pdf")

    # Multi-method comparison — bar chart
    results = {"iCaRL": r1, "BiC": r2, "NCM": r3}
    plot_method_comparison(results, save="figs/compare.pdf")

    # Forgetting heatmap (task × eval-step)
    plot_forgetting_matrix(result, save="figs/forgetting.pdf")

    # Full suite from a results directory
    from benchmark.eval.plots import plot_suite
    plot_suite("results/", out_dir="figs/")
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np

# ── Lazy imports (matplotlib/seaborn optional at import time) ─────

def _mpl():
    import matplotlib.pyplot as plt
    return plt

def _sns():
    import seaborn as sns
    return sns


# ── Style ─────────────────────────────────────────────────────────

def _set_style():
    import matplotlib as mpl
    _sns().set_theme(style="whitegrid", font_scale=1.15)
    mpl.rcParams.update({
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "font.family":      "sans-serif",
    })

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]


# ── 1. Task-progression curve ─────────────────────────────────────

def plot_task_curves(result, metrics=("oa", "avg_aa", "kappa"),
                     save: str | None = None, ax=None):
    """Line plot of OA / AA / Kappa across incremental tasks.

    Args:
        result:  BenchmarkResult object (from benchmark.eval.metrics)
        metrics: which metrics to show
        save:    path to save figure (pdf/png/svg)
    """
    _set_style()
    plt = _mpl()

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))

    task_ids = [r.task_id for r in result.task_results]
    label_map = {"oa": "OA (%)", "avg_aa": "AA (%)", "kappa": "κ"}
    style_map  = {"oa": "-o", "avg_aa": "--s", "kappa": ":^"}

    for i, m in enumerate(metrics):
        vals = [getattr(r, m) * (100 if m != "kappa" else 1)
                for r in result.task_results]
        ax.plot(task_ids, vals, style_map.get(m, "-o"),
                color=_PALETTE[i], label=label_map.get(m, m),
                linewidth=2, markersize=6)

    ax.set_xlabel("Task")
    ax.set_ylabel("Score")
    ax.set_title(f"{result.method_name} on {result.protocol_name}")
    ax.set_xticks(task_ids)
    ax.legend(loc="lower left")

    # Annotate BWT
    bwt_str = f"BWT={result.bwt*100:.1f}pp  FWT={result.fwt*100:.1f}%"
    ax.text(0.98, 0.05, bwt_str, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#555555")

    if standalone:
        if save:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save)
            print(f"[plot] saved → {save}")
        plt.tight_layout()
        return fig
    return ax


# ── 2. Method comparison bar chart ────────────────────────────────

def plot_method_comparison(results: dict[str, Any],
                            metric: str = "final_oa",
                            save: str | None = None):
    """Grouped bar chart comparing methods on a single metric.

    Args:
        results:  {method_name: BenchmarkResult}
        metric:   "final_oa" | "final_aa" | "final_kappa" | "bwt" | "fwt"
        save:     path to save
    """
    _set_style()
    plt = _mpl()

    names = list(results.keys())
    scale = 100 if metric != "final_kappa" else 1
    vals  = [getattr(r, metric) * scale for r in results.values()]
    label = {"final_oa": "Final OA (%)", "final_aa": "Final AA (%)",
             "final_kappa": "Final κ", "bwt": "BWT (pp)", "fwt": "FWT (%)"}

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), 4.5))
    bars = ax.bar(names, vals, color=_PALETTE[:len(names)], width=0.6,
                  edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(label.get(metric, metric))
    protocol = next(iter(results.values())).protocol_name
    ax.set_title(f"{label.get(metric, metric)} — {protocol}")
    ax.set_xticklabels(names, rotation=30, ha="right")

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save)
        print(f"[plot] saved → {save}")
    plt.tight_layout()
    return fig


# ── 3. Forgetting heatmap ─────────────────────────────────────────

def plot_forgetting_matrix(result, metric: str = "oa",
                            save: str | None = None):
    """Heatmap: rows = tasks evaluated, cols = after which task.

    Requires result.task_results to have per-task per-dataset breakdown.
    Falls back to a per-task OA column if per-dataset not available.
    """
    _set_style()
    plt = _mpl()

    task_results = result.task_results
    n = len(task_results)
    if n < 2:
        print("[plot_forgetting_matrix] need ≥2 tasks, skipping")
        return

    # Build matrix: M[i][j] = performance on task-i after seeing task-j
    # We only have the aggregated value per eval step (not per-task breakdown)
    # So we show the incremental OA trajectory as a 1D curve heatmap
    vals = np.array([getattr(r, metric) * 100 for r in task_results])
    matrix = np.full((n, n), np.nan)
    for j in range(n):
        matrix[j, j] = vals[j]  # diagonal = plasticity

    fig, ax = plt.subplots(figsize=(max(5, n * 1.1), max(4, n * 0.9)))
    _sns().heatmap(
        matrix, ax=ax, annot=True, fmt=".1f",
        cmap="RdYlGn", vmin=50, vmax=100,
        linewidths=0.5, linecolor="white",
        mask=np.isnan(matrix),
        cbar_kws={"label": f"{metric.upper()} (%)"},
    )
    ax.set_xlabel("After task")
    ax.set_ylabel("Task evaluated")
    ax.set_title(f"Forgetting matrix — {result.method_name} / {result.protocol_name}")

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save)
        print(f"[plot] saved → {save}")
    plt.tight_layout()
    return fig


# ── 4. Multi-method task curves (overlay) ────────────────────────

def plot_methods_overlay(results: dict[str, Any], metric: str = "oa",
                          save: str | None = None):
    """One curve per method, all on the same axes.

    Args:
        results: {method_name: BenchmarkResult}
        metric:  "oa" | "avg_aa" | "kappa"
    """
    _set_style()
    plt = _mpl()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    scale = 100 if metric != "kappa" else 1
    label_map = {"oa": "OA (%)", "avg_aa": "AA (%)", "kappa": "κ"}

    for i, (name, result) in enumerate(results.items()):
        task_ids = [r.task_id for r in result.task_results]
        vals = [getattr(r, metric) * scale for r in result.task_results]
        ax.plot(task_ids, vals, "-o", color=_PALETTE[i % len(_PALETTE)],
                label=name, linewidth=2, markersize=5)

    ax.set_xlabel("Task")
    ax.set_ylabel(label_map.get(metric, metric))
    protocol = next(iter(results.values())).protocol_name
    ax.set_title(f"{label_map.get(metric, metric)} progression — {protocol}")
    ax.set_xticks(task_ids)
    ax.legend(loc="lower left", fontsize=9)

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save)
        print(f"[plot] saved → {save}")
    plt.tight_layout()
    return fig


# ── 5. Full suite from results directory ─────────────────────────

def plot_suite(results_dir: str | Path, out_dir: str | Path = "figs/",
               protocol_filter: str | None = None):
    """Load all JSON result files from results_dir and generate all plots.

    File naming convention: {method}_{protocol}_seed{seed}.json

    Args:
        results_dir:     directory with .json result files
        out_dir:         where to save figures
        protocol_filter: only plot results for this protocol (e.g. "B1")
    """
    from benchmark.eval.metrics import BenchmarkResult, TaskResult

    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all JSON files
    by_protocol: dict[str, dict[str, list]] = {}  # protocol → method → [results]
    for f in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        proto  = data.get("protocol", "unknown")
        method = data.get("method",   "unknown")
        if protocol_filter and proto != protocol_filter:
            continue
        by_protocol.setdefault(proto, {}).setdefault(method, []).append(data)

    if not by_protocol:
        print(f"[plot_suite] no JSON files found in {results_dir}")
        return

    plt = _mpl()

    for proto, methods in by_protocol.items():
        # Reconstruct lightweight result objects from JSON
        reconstructed: dict[str, Any] = {}
        for method, runs in methods.items():
            # Average across seeds if multiple
            r = _json_to_result(runs[0], proto, method)
            reconstructed[method] = r

        # Method comparison bar charts
        for metric in ("final_oa", "final_aa", "bwt"):
            plot_method_comparison(
                reconstructed, metric=metric,
                save=str(out_dir / f"{proto}_{metric}_bar.pdf")
            )

        # Per-method task curves
        for method, r in reconstructed.items():
            plot_task_curves(r, save=str(out_dir / f"{proto}_{method}_curve.pdf"))

        # Overlay
        if len(reconstructed) > 1:
            plot_methods_overlay(
                reconstructed, metric="oa",
                save=str(out_dir / f"{proto}_overlay_oa.pdf")
            )

        plt.close("all")

    print(f"[plot_suite] figures saved to {out_dir}")


# ── helpers ───────────────────────────────────────────────────────

class _SimpleResult:
    """Lightweight result object reconstructed from a saved JSON dict."""
    def __init__(self, d: dict, protocol: str, method: str):
        self.protocol_name = protocol
        self.method_name   = method
        self.final_oa    = d.get("final_oa",    d.get("oa_mean",    0.0))
        self.final_aa    = d.get("final_aa",    d.get("aa_mean",    0.0))
        self.final_kappa = d.get("final_kappa", d.get("kappa_mean", 0.0))
        self.bwt         = d.get("bwt",         d.get("bwt_mean",   0.0))
        self.fwt         = d.get("fwt",         d.get("fwt_mean",   0.0))
        self.task_results = [_SimpleTaskResult(t) for t in d.get("tasks", [])]


class _SimpleTaskResult:
    def __init__(self, d: dict):
        self.task_id = d.get("task_id", 0)
        self.oa      = d.get("oa",      0.0)
        self.avg_aa  = d.get("avg_aa",  0.0)
        self.kappa   = d.get("kappa",   0.0)


def _json_to_result(d: dict, protocol: str, method: str) -> _SimpleResult:
    return _SimpleResult(d, protocol, method)
