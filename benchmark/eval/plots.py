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


# ══════════════════════════════════════════════════════════════════
# 6. Classification map (RGB)
# ══════════════════════════════════════════════════════════════════

def plot_classification_map(
    gt_map: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
    dataset_name: str | None = None,
    title: str = "",
    save: str | None = None,
    show_legend: bool = True,
    show_errors: bool = False,
):
    """Render a classification map overlaying predictions on the ground truth.

    Args:
        gt_map:       (H, W) int array, 1-indexed (0 = background).
        preds:        (N,) predicted class IDs (global).
        targets:      (N,) ground-truth class IDs (global).
        class_names:  List of class names for the legend.
        dataset_name: Dataset name for colour palette lookup.
        title:        Figure title.
        save:         Path to save (pdf/png/svg).
        show_legend:  Whether to show the colour legend.
        show_errors:  If True, highlight misclassifications in red.
    """
    from benchmark.eval.colors import get_colormap, label_map_to_rgb

    _set_style()
    plt = _mpl()

    num_classes = len(class_names)
    ds_name = dataset_name or ""
    cmap = get_colormap(ds_name, num_classes)

    # Build the RGB image from ground truth (faded background)
    gt_rgb = label_map_to_rgb(gt_map, ds_name, num_classes).astype(np.float32)
    # Fade non-test background
    gt_rgb *= 0.3

    # Reconstruct prediction map from (preds, targets) — map back to spatial coords
    # We overlay predictions onto the GT map where test pixels exist
    pred_map = np.zeros_like(gt_map)
    test_coords = _find_test_coords(gt_map, targets)
    if test_coords is not None:
        for (r, c), p in zip(test_coords, preds):
            pred_map[r, c] = p

        # Paint predictions with full saturation
        for (r, c), p in zip(test_coords, preds):
            if p > 0 and p <= num_classes:
                gt_rgb[r, c] = cmap[int(p)].astype(np.float32)

        if show_errors:
            for (r, c), p, t in zip(test_coords, preds, targets):
                if p != t:
                    gt_rgb[r, c] = [255, 0, 0]  # red for errors

    gt_rgb = np.clip(gt_rgb, 0, 255).astype(np.uint8)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(gt_rgb)
    ax.set_title(title or f"Classification Map — {ds_name}", fontsize=12)
    ax.axis("off")

    if show_legend and class_names:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=np.array(cmap[i + 1]) / 255.0, label=name)
            for i, name in enumerate(class_names)
        ]
        ax.legend(handles=legend_elements, loc="lower right",
                  fontsize=7, ncol=max(1, num_classes // 8),
                  framealpha=0.8)

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, bbox_inches="tight")
        print(f"[plot] saved → {save}")
    plt.tight_layout()
    return fig


def _find_test_coords(gt_map, targets):
    """Match test pixel targets back to their (row, col) in gt_map.

    Returns (N, 2) array of (row, col) or None if matching fails.
    """
    # Build a mapping from label → list of coords in gt_map
    unique_targets = np.unique(targets)
    coords_by_class = {}
    for c in unique_targets:
        rows, cols = np.where(gt_map == c)
        coords_by_class[c] = list(zip(rows, cols))

    # Try to assign coords: iterate through targets, pop from coords_by_class
    # This is approximate (test pixels are a subset of gt_map pixels)
    result = []
    usage = {c: 0 for c in unique_targets}
    for t in targets:
        c = int(t)
        if c in coords_by_class and usage[c] < len(coords_by_class[c]):
            result.append(coords_by_class[c][usage[c]])
            usage[c] += 1
        else:
            result.append((0, 0))  # fallback

    return np.array(result)


# ══════════════════════════════════════════════════════════════════
# 7. Classification maps per task (CIL evolution)
# ══════════════════════════════════════════════════════════════════

def plot_classification_maps_per_task(
    gt_map: np.ndarray,
    predictions_per_task: list[dict],
    class_names: list[str],
    dataset_name: str | None = None,
    save: str | None = None,
):
    """Show classification evolution across CIL tasks as a subplot grid.

    Args:
        gt_map:               (H, W) ground truth map.
        predictions_per_task: List of dicts, each with keys:
                              "preds", "targets", "task_id".
        class_names:          Class names for legend.
        dataset_name:         For colour palette.
        save:                 Path to save figure.
    """
    from benchmark.eval.colors import get_colormap, label_map_to_rgb

    _set_style()
    plt = _mpl()

    n_tasks = len(predictions_per_task)
    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 5))
    if n_tasks == 1:
        axes = [axes]

    num_classes = len(class_names)
    ds_name = dataset_name or ""
    cmap = get_colormap(ds_name, num_classes)

    for ax, task_data in zip(axes, predictions_per_task):
        gt_rgb = label_map_to_rgb(gt_map, ds_name, num_classes).astype(np.float32)
        gt_rgb *= 0.3

        test_coords = _find_test_coords(gt_map, task_data["targets"])
        if test_coords is not None:
            for (r, c), p in zip(test_coords, task_data["preds"]):
                if 0 < p <= num_classes:
                    gt_rgb[r, c] = cmap[int(p)].astype(np.float32)

        gt_rgb = np.clip(gt_rgb, 0, 255).astype(np.uint8)
        ax.imshow(gt_rgb)
        ax.set_title(f"Task {task_data.get('task_id', '?')}", fontsize=10)
        ax.axis("off")

    plt.suptitle(f"CIL Evolution — {ds_name}", fontsize=13)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, bbox_inches="tight")
        print(f"[plot] saved → {save}")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# 8. Confusion matrix
# ══════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    class_names: list[str] | None = None,
    normalize: bool = True,
    save: str | None = None,
    title: str = "",
):
    """Plot a confusion matrix heatmap.

    Args:
        preds:       (N,) predicted labels.
        targets:     (N,) true labels.
        class_names: Labels for axes. If None, uses sorted unique targets.
        normalize:   If True, normalize rows to sum to 1 (recall-based).
        save:        Path to save.
        title:       Figure title.
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    _set_style()
    plt = _mpl()
    sns = _sns()

    labels = sorted(set(targets.tolist()) | set(preds.tolist()))
    cm = sk_cm(targets, preds, labels=labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.where(row_sums > 0, cm / row_sums, 0) * 100

    names = class_names or [str(l) for l in labels]
    # Truncate names if too long
    names = [n[:15] for n in names]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.5)))
    sns.heatmap(
        cm, ax=ax, annot=True,
        fmt=".1f" if normalize else "d",
        cmap="Blues", vmin=0, vmax=100 if normalize else None,
        xticklabels=names[:n], yticklabels=names[:n],
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Recall (%)" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion Matrix")

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, bbox_inches="tight")
        print(f"[plot] saved → {save}")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# 9. Per-class accuracy bar chart
# ══════════════════════════════════════════════════════════════════

def plot_per_class_accuracy(
    preds: np.ndarray,
    targets: np.ndarray,
    class_names: list[str] | None = None,
    save: str | None = None,
    title: str = "",
):
    """Horizontal bar chart of per-class recall.

    Args:
        preds:       (N,) predicted labels.
        targets:     (N,) true labels.
        class_names: Labels for bars.
        save:        Path to save.
        title:       Figure title.
    """
    _set_style()
    plt = _mpl()

    labels = sorted(set(targets.tolist()))
    accs = []
    for c in labels:
        mask = targets == c
        accs.append((preds[mask] == c).mean() * 100 if mask.sum() > 0 else 0)

    names = class_names or [str(l) for l in labels]
    names = [n[:20] for n in names]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.35)))
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]
    bars = ax.barh(range(len(labels)), accs, color=colors, height=0.7)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", ha="left", va="center", fontsize=8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(names[:len(labels)], fontsize=9)
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 105)
    ax.set_title(title or "Per-Class Accuracy")
    ax.invert_yaxis()

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, bbox_inches="tight")
        print(f"[plot] saved → {save}")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# 10. Radar / spider chart (per-dataset comparison)
# ══════════════════════════════════════════════════════════════════

def plot_radar_comparison(
    results: dict[str, dict[str, float]],
    save: str | None = None,
    title: str = "Per-Dataset Accuracy",
):
    """Spider chart comparing methods across datasets.

    Args:
        results: {method_name: {dataset_name: AA_percentage}}.
        save:    Path to save.
        title:   Figure title.
    """
    _set_style()
    plt = _mpl()

    # Get all dataset names (union across methods)
    all_datasets = sorted({ds for m in results.values() for ds in m})
    n = len(all_datasets)
    if n < 3:
        print("[plot_radar] need ≥3 datasets for radar chart, skipping")
        return

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, (method_name, ds_accs) in enumerate(results.items()):
        vals = [ds_accs.get(ds, 0) for ds in all_datasets]
        vals += vals[:1]
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(angles, vals, "o-", color=color, linewidth=2,
                label=method_name, markersize=5)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_datasets, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title(title, y=1.08, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, bbox_inches="tight")
        print(f"[plot] saved → {save}")
    return fig


# ══════════════════════════════════════════════════════════════════
# 11. Method comparison table (matplotlib rendered)
# ══════════════════════════════════════════════════════════════════

def plot_method_comparison_table(
    results: dict[str, dict],
    metrics: tuple[str, ...] = ("final_oa", "final_aa", "bwt"),
    save: str | None = None,
):
    """Render a styled table comparing methods on multiple metrics.

    Args:
        results: {method_name: dict with metric keys}.
        metrics: Which metrics to show.
        save:    Path to save.
    """
    _set_style()
    plt = _mpl()

    methods = list(results.keys())
    label_map = {
        "final_oa": "OA (%)", "final_aa": "AA (%)", "final_kappa": "κ",
        "bwt": "BWT (pp)", "fwt": "FWT (%)",
    }
    headers = [label_map.get(m, m) for m in metrics]

    cell_text = []
    for method in methods:
        row = []
        for m in metrics:
            val = results[method].get(m, 0)
            scale = 100 if m not in ("final_kappa",) else 1
            row.append(f"{val * scale:.2f}")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 1.5),
                                     max(3, len(methods) * 0.4 + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=methods,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Highlight best per column
    for col_idx in range(len(metrics)):
        vals = [float(cell_text[r][col_idx]) for r in range(len(methods))]
        best_row = int(np.argmax(vals)) if metrics[col_idx] != "bwt" else int(np.argmin([abs(v) for v in vals]))
        table[best_row + 1, col_idx].set_facecolor("#d4edda")

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, bbox_inches="tight")
        print(f"[plot] saved → {save}")
    return fig
