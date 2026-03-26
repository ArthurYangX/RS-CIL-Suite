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
from collections import defaultdict
import json
from pathlib import Path
from typing import Any
import warnings

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


def _save_publication_figure(fig, save: str | None):
    """Save both PDF and PNG variants for publication use."""
    if not save:
        return
    base = Path(save)
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        out = base.with_suffix(suffix)
        fig.savefig(out, bbox_inches="tight")
        print(f"[plot] saved → {out}")


def _coerce_runs(result_or_runs: Any) -> list[Any]:
    if isinstance(result_or_runs, (list, tuple)):
        return list(result_or_runs)
    return [result_or_runs]


def _run_attr(run: Any, name: str, default=None):
    if isinstance(run, dict):
        return run.get(name, default)
    return getattr(run, name, default)


def _method_name(run: Any) -> str:
    return _run_attr(run, "method_name", _run_attr(run, "method", "unknown"))


def _protocol_name(run: Any) -> str:
    return _run_attr(run, "protocol_name", _run_attr(run, "protocol", "unknown"))


def _extract_task_results(run: Any) -> list[dict]:
    task_results = _run_attr(run, "task_results", [])
    out: list[dict] = []
    for record in task_results:
        if isinstance(record, dict):
            out.append({
                "task_id": int(record.get("task_id", 0)),
                "oa": float(record.get("oa", 0.0)),
                "avg_aa": float(record.get("avg_aa", record.get("aa", 0.0))),
                "kappa": float(record.get("kappa", 0.0)),
            })
        else:
            out.append({
                "task_id": int(getattr(record, "task_id", 0)),
                "oa": float(getattr(record, "oa", 0.0)),
                "avg_aa": float(getattr(record, "avg_aa", getattr(record, "aa", 0.0))),
                "kappa": float(getattr(record, "kappa", 0.0)),
            })
    return out


def _extract_task_feedback_records(run: Any) -> list[dict]:
    """Return normalized task-feedback records from a saved run/result object."""
    task_feedback = _run_attr(run, "task_feedback", None)
    records: list[dict] = []

    if task_feedback:
        for record in task_feedback:
            if isinstance(record, dict):
                records.append({
                    "after_task_id": int(record.get("after_task_id", 0)),
                    "eval_task_id": int(record.get("eval_task_id", 0)),
                    "dataset_name": str(record.get("dataset_name", "")),
                    "oa": float(record.get("oa", 0.0)),
                    "aa": float(record.get("aa", record.get("avg_aa", 0.0))),
                    "kappa": float(record.get("kappa", 0.0)),
                    "num_samples": int(record.get("num_samples", 0)),
                })
            else:
                records.append({
                    "after_task_id": int(getattr(record, "after_task_id", 0)),
                    "eval_task_id": int(getattr(record, "eval_task_id", 0)),
                    "dataset_name": str(getattr(record, "dataset_name", "")),
                    "oa": float(getattr(record, "oa", 0.0)),
                    "aa": float(getattr(record, "aa", getattr(record, "avg_aa", 0.0))),
                    "kappa": float(getattr(record, "kappa", 0.0)),
                    "num_samples": int(getattr(record, "num_samples", 0)),
                })
        return records

    # Backward-compatible fallback: only diagonal entries are available.
    task_results = _extract_task_results(run)
    if task_results:
        warnings.warn(
            "Task-feedback artifacts are unavailable; falling back to diagonal-only task results.",
            RuntimeWarning,
        )
    for record in task_results:
        records.append({
            "after_task_id": record["task_id"],
            "eval_task_id": record["task_id"],
            "dataset_name": "",
            "oa": record["oa"],
            "aa": record["avg_aa"],
            "kappa": record["kappa"],
            "num_samples": 0,
        })
    return records


def _aggregate_task_curves(runs: list[Any]) -> dict[str, Any] | None:
    task_results_per_run = [_extract_task_results(run) for run in runs]
    task_ids = sorted({r["task_id"] for run in task_results_per_run for r in run})
    if not task_ids:
        return None

    metrics = ("oa", "avg_aa", "kappa")
    stack = {metric: [] for metric in metrics}
    for run_records in task_results_per_run:
        by_task = {r["task_id"]: r for r in run_records}
        for metric in metrics:
            vals = np.full(len(task_ids), np.nan, dtype=np.float64)
            for idx, task_id in enumerate(task_ids):
                if task_id in by_task:
                    vals[idx] = float(by_task[task_id][metric])
            stack[metric].append(vals)

    return {
        "task_ids": task_ids,
        "mean": {m: np.nanmean(np.vstack(v), axis=0) for m, v in stack.items()},
        "std": {m: np.nanstd(np.vstack(v), axis=0) for m, v in stack.items()},
        "num_runs": len(runs),
    }


def _aggregate_task_feedback(result_or_runs: Any, metric: str = "oa") -> dict[str, Any] | None:
    runs = _coerce_runs(result_or_runs)
    metric_key = {"avg_aa": "aa"}.get(metric, metric)
    records_per_run = [_extract_task_feedback_records(run) for run in runs]

    after_ids = sorted({r["after_task_id"] for run in records_per_run for r in run})
    eval_ids = sorted({r["eval_task_id"] for run in records_per_run for r in run})
    if not after_ids or not eval_ids:
        return None

    dataset_by_eval: dict[int, str] = {}
    matrices = []
    for run_records in records_per_run:
        matrix = np.full((len(eval_ids), len(after_ids)), np.nan, dtype=np.float64)
        for record in run_records:
            i = eval_ids.index(record["eval_task_id"])
            j = after_ids.index(record["after_task_id"])
            matrix[i, j] = float(record.get(metric_key, 0.0))
            dataset_by_eval.setdefault(record["eval_task_id"], record.get("dataset_name", ""))
        matrices.append(matrix)

    stack = np.stack(matrices, axis=0)
    return {
        "after_ids": after_ids,
        "eval_ids": eval_ids,
        "dataset_by_eval": dataset_by_eval,
        "matrices": stack,
        "mean": np.nanmean(stack, axis=0),
        "std": np.nanstd(stack, axis=0),
        "num_runs": len(runs),
    }


def _task_eval_records(run: Any) -> list[dict]:
    return list(_run_attr(run, "task_evals", []))


def _dataset_mappings(run: Any) -> dict[str, dict]:
    return dict(_run_attr(run, "dataset_mappings", {}) or {})


def _artifact_path(run: Any) -> Path | None:
    source_file = _run_attr(run, "source_file", None)
    artifacts_file = _run_attr(run, "artifacts_file", None)
    if source_file is None or not artifacts_file:
        return None
    return Path(source_file).with_name(artifacts_file)


def _load_artifact_arrays(run: Any):
    path = _artifact_path(run)
    if path is None:
        return None
    if not path.exists():
        warnings.warn(f"Artifact file not found: {path}", RuntimeWarning)
        return None
    return np.load(path, allow_pickle=False)


def _materialize_task_eval(record: dict, artifact_arrays=None) -> dict:
    out = dict(record)
    for field, key_field in (("preds", "preds_key"), ("targets", "targets_key"), ("coords", "coords_key")):
        if field in out:
            continue
        key = out.get(key_field)
        if key is not None and artifact_arrays is not None and key in artifact_arrays:
            out[field] = artifact_arrays[key]
    return out


def _dataset_global_to_local(record: dict, dataset_mapping: dict | None = None) -> dict[int, int]:
    mapping = dataset_mapping or {}
    global_ids = mapping.get("global_class_ids", record.get("dataset_global_class_ids", record.get("global_class_ids", [])))
    local_ids = mapping.get("local_class_ids", record.get("dataset_local_class_ids", record.get("local_class_ids", [])))
    return {
        int(global_id): int(local_id)
        for global_id, local_id in zip(global_ids, local_ids)
    }


def _global_to_local_labels(values: np.ndarray, global_to_local: dict[int, int] | None) -> np.ndarray:
    arr = np.asarray(values)
    if global_to_local is None:
        return arr.astype(np.int32, copy=False)
    out = np.zeros(arr.shape[0], dtype=np.int32)
    for idx, value in enumerate(arr.tolist()):
        if int(value) in global_to_local:
            out[idx] = global_to_local[int(value)] + 1
    return out


def _build_prediction_snapshots(
    task_eval_records: list[dict],
    dataset_name: str,
    dataset_mapping: dict | None = None,
    artifact_arrays=None,
) -> list[dict]:
    """Combine per-task artifacts into per-after-task scene snapshots."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in task_eval_records:
        if record.get("dataset_name") != dataset_name:
            continue
        grouped[int(record.get("after_task_id", 0))].append(
            _materialize_task_eval(record, artifact_arrays)
        )

    snapshots: list[dict] = []
    for after_task_id in sorted(grouped):
        preds_parts, targets_parts, coords_parts = [], [], []
        oa_values = []
        global_to_local = _dataset_global_to_local(grouped[after_task_id][0], dataset_mapping)
        for record in grouped[after_task_id]:
            preds = record.get("preds")
            targets = record.get("targets")
            coords = record.get("coords")
            if preds is None or targets is None or coords is None:
                continue
            preds_parts.append(np.asarray(preds))
            targets_parts.append(np.asarray(targets))
            coords_parts.append(np.asarray(coords))
            oa_values.append(float(record.get("oa", 0.0)))

        if not coords_parts:
            continue

        preds = np.concatenate(preds_parts)
        targets = np.concatenate(targets_parts)
        coords = np.concatenate(coords_parts).astype(np.int32, copy=False)
        snapshots.append({
            "after_task_id": after_task_id,
            "preds": preds,
            "targets": targets,
            "coords": coords,
            "global_to_local": global_to_local,
            "oa": float((preds == targets).mean()) if len(targets) > 0 else (
                float(np.mean(oa_values)) if oa_values else 0.0
            ),
        })
    return snapshots


def _render_classification_map_rgb(
    gt_map: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    dataset_name: str = "",
    coords: np.ndarray | None = None,
    global_to_local: dict[int, int] | None = None,
    show_errors: bool = False,
) -> np.ndarray:
    from benchmark.eval.colors import get_colormap, label_map_to_rgb

    cmap = get_colormap(dataset_name, num_classes)
    rgb = label_map_to_rgb(gt_map, dataset_name, num_classes).astype(np.float32)
    rgb *= 0.30

    if coords is None:
        warnings.warn(
            "Exact spatial coordinates are unavailable; falling back to approximate matching.",
            RuntimeWarning,
        )
        coords = _find_test_coords(gt_map, targets)

    if coords is None:
        return np.clip(rgb, 0, 255).astype(np.uint8)

    map_preds = _global_to_local_labels(np.asarray(preds), global_to_local)
    map_targets = _global_to_local_labels(np.asarray(targets), global_to_local)

    for (row, col), pred in zip(coords, map_preds):
        row_i, col_i = int(row), int(col)
        if not (0 <= row_i < gt_map.shape[0] and 0 <= col_i < gt_map.shape[1]):
            continue
        if 0 < pred <= num_classes:
            rgb[row_i, col_i] = cmap[int(pred)].astype(np.float32)

    if show_errors:
        for (row, col), pred, target in zip(coords, map_preds, map_targets):
            row_i, col_i = int(row), int(col)
            if not (0 <= row_i < gt_map.shape[0] and 0 <= col_i < gt_map.shape[1]):
                continue
            if pred != target:
                rgb[row_i, col_i] = [255, 0, 0]

    return np.clip(rgb, 0, 255).astype(np.uint8)


# ── 1. Task-progression curve ─────────────────────────────────────

def plot_task_curves(result, metrics=("oa", "avg_aa", "kappa"),
                     save: str | None = None, ax=None):
    """Line plot of OA / AA / Kappa across incremental tasks.

    Args:
        result:  BenchmarkResult object or a list of per-seed results
        metrics: which metrics to show
        save:    path to save figure (pdf/png/svg)
    """
    _set_style()
    plt = _mpl()
    runs = _coerce_runs(result)
    aggregated = _aggregate_task_curves(runs)
    if aggregated is None:
        warnings.warn("No task results available for plot_task_curves().", RuntimeWarning)
        return None

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))

    task_ids = aggregated["task_ids"]
    label_map = {"oa": "OA (%)", "avg_aa": "AA (%)", "kappa": "κ"}
    style_map  = {"oa": "-o", "avg_aa": "--s", "kappa": ":^"}

    for i, m in enumerate(metrics):
        scale = 100 if m != "kappa" else 1
        vals = aggregated["mean"][m] * scale
        stds = aggregated["std"][m] * scale
        ax.plot(task_ids, vals, style_map.get(m, "-o"),
                color=_PALETTE[i], label=label_map.get(m, m),
                linewidth=2, markersize=6)
        if aggregated["num_runs"] > 1:
            ax.fill_between(
                task_ids,
                vals - stds,
                vals + stds,
                color=_PALETTE[i],
                alpha=0.15,
                linewidth=0,
            )

    ax.set_xlabel("Task")
    ax.set_ylabel("Score")
    ax.set_title(f"{_method_name(runs[0])} on {_protocol_name(runs[0])}")
    ax.set_xticks(task_ids)
    ax.legend(loc="lower left")

    # Annotate BWT
    bwt_vals = [_run_attr(run, "bwt", 0.0) for run in runs]
    fwt_vals = [_run_attr(run, "fwt", 0.0) for run in runs]
    bwt_str = (
        f"BWT={np.mean(bwt_vals)*100:.1f}pp  "
        f"FWT={np.mean(fwt_vals)*100:.1f}%"
    )
    ax.text(0.98, 0.05, bwt_str, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#555555")

    if standalone:
        if save:
            _save_publication_figure(fig, save)
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

    If task-feedback artifacts are available, this renders the true
    task-aware matrix. Otherwise it falls back to the legacy diagonal-only
    approximation for backward compatibility.
    """
    return plot_task_accuracy_matrix(result, metric=metric, save=save)


def plot_task_accuracy_matrix(
    result_or_runs: Any,
    metric: str = "oa",
    save: str | None = None,
    ax=None,
):
    """True task-aware CIL accuracy matrix.

    Rows are evaluated tasks, columns are the learner state after task `t`.
    Values are task-specific OA / AA / Kappa. Supports multi-seed mean ± std.
    """
    _set_style()
    plt = _mpl()
    aggregated = _aggregate_task_feedback(result_or_runs, metric=metric)
    if aggregated is None:
        warnings.warn("Task-feedback records are unavailable; skipping task matrix.", RuntimeWarning)
        return None

    scale = 100 if metric != "kappa" else 1
    matrix = aggregated["mean"] * scale
    std = aggregated["std"] * scale
    after_ids = aggregated["after_ids"]
    eval_ids = aggregated["eval_ids"]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=(max(5.5, len(after_ids) * 1.05), max(4.5, len(eval_ids) * 0.8))
        )

    annotations = np.empty(matrix.shape, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                annotations[i, j] = ""
            elif aggregated["num_runs"] > 1:
                annotations[i, j] = f"{matrix[i, j]:.1f}\n±{std[i, j]:.1f}"
            else:
                annotations[i, j] = f"{matrix[i, j]:.1f}"

    label_map = {"oa": "OA (%)", "aa": "AA (%)", "avg_aa": "AA (%)", "kappa": "κ"}
    sns = _sns()
    sns.heatmap(
        matrix,
        ax=ax,
        annot=annotations,
        fmt="",
        cmap="YlGnBu",
        vmin=0 if metric == "kappa" else max(0, np.nanmin(matrix) - 5),
        vmax=1 if metric == "kappa" else min(100, np.nanmax(matrix) + 5),
        linewidths=0.6,
        linecolor="white",
        mask=np.isnan(matrix),
        cbar_kws={"label": label_map.get(metric, metric)},
        annot_kws={"fontsize": 9 if aggregated["num_runs"] > 1 else 10},
    )
    ax.set_xlabel("After Learning Task")
    ax.set_ylabel("Evaluated Task")
    ax.set_xticklabels([f"T{task_id + 1}" for task_id in after_ids], rotation=0)
    ax.set_yticklabels([f"T{task_id + 1}" for task_id in eval_ids], rotation=0)
    ax.set_title(
        f"Task-Aware {label_map.get(metric, metric)} Matrix — "
        f"{_method_name(_coerce_runs(result_or_runs)[0])} / {_protocol_name(_coerce_runs(result_or_runs)[0])}"
    )

    if standalone:
        if save:
            _save_publication_figure(fig, save)
        plt.tight_layout()
        return fig
    return ax


def plot_task_feedback_curve(
    result_or_runs: Any,
    metric: str = "oa",
    save: str | None = None,
    ax=None,
):
    """Companion task-feedback curve.

    Shows average seen-task performance, current-task performance, and
    old-task performance as the learner progresses.
    """
    _set_style()
    plt = _mpl()
    aggregated = _aggregate_task_feedback(result_or_runs, metric=metric)
    if aggregated is None:
        warnings.warn("Task-feedback records are unavailable; skipping feedback curve.", RuntimeWarning)
        return None

    matrices = aggregated["matrices"]
    after_ids = aggregated["after_ids"]
    eval_ids = aggregated["eval_ids"]
    scale = 100 if metric != "kappa" else 1

    eval_to_row = {task_id: idx for idx, task_id in enumerate(eval_ids)}
    seen_curves = []
    current_curves = []
    old_curves = []
    for matrix in matrices:
        seen_vals, current_vals, old_vals = [], [], []
        for col_idx, after_task_id in enumerate(after_ids):
            seen_vals.append(np.nanmean(matrix[:, col_idx]))
            row_idx = eval_to_row.get(after_task_id)
            current_vals.append(matrix[row_idx, col_idx] if row_idx is not None else np.nan)

            old_row_indices = [eval_to_row[t] for t in eval_ids if t < after_task_id]
            if old_row_indices:
                old_vals.append(np.nanmean(matrix[old_row_indices, col_idx]))
            else:
                old_vals.append(np.nan)
        seen_curves.append(seen_vals)
        current_curves.append(current_vals)
        old_curves.append(old_vals)

    seen_arr = np.asarray(seen_curves, dtype=np.float64) * scale
    current_arr = np.asarray(current_curves, dtype=np.float64) * scale
    old_arr = np.asarray(old_curves, dtype=np.float64) * scale

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7.4, 4.6))

    series = [
        ("Seen-task mean", seen_arr, _PALETTE[0], "-o"),
        ("Current task", current_arr, _PALETTE[1], "--s"),
        ("Old-task mean", old_arr, _PALETTE[2], ":^"),
    ]
    for label, values, color, style in series:
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        ax.plot(after_ids, mean, style, color=color, linewidth=2.2, markersize=6, label=label)
        if values.shape[0] > 1:
            ax.fill_between(after_ids, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)

    label_map = {"oa": "OA (%)", "aa": "AA (%)", "avg_aa": "AA (%)", "kappa": "κ"}
    ax.set_xlabel("After Learning Task")
    ax.set_ylabel(label_map.get(metric, metric))
    ax.set_xticks(after_ids)
    ax.set_xticklabels([f"T{task_id + 1}" for task_id in after_ids])
    ax.set_title(
        f"Task Feedback — {_method_name(_coerce_runs(result_or_runs)[0])} / "
        f"{_protocol_name(_coerce_runs(result_or_runs)[0])}"
    )
    ax.legend(loc="lower left", fontsize=9)

    if standalone:
        if save:
            _save_publication_figure(fig, save)
        plt.tight_layout()
        return fig
    return ax


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

    Properly aggregates multiple seeds (mean ± std) instead of using only
    the first run.  Also generates task-feedback matrix and feedback curves
    when ``task_feedback`` data is available in the JSON files.

    Args:
        results_dir:     directory with .json result files
        out_dir:         where to save figures
        protocol_filter: only plot results for this protocol (e.g. "B1")
    """
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all JSON files (recursive)
    by_protocol: dict[str, dict[str, list]] = {}
    for f in sorted(results_dir.glob("**/*.json")):
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
        reconstructed: dict[str, Any] = {}
        for method, runs in methods.items():
            # Real multi-seed aggregation
            r = _json_to_result_averaged(runs, proto, method)
            reconstructed[method] = r

        # Method comparison bar charts
        for metric in ("final_oa", "final_aa", "bwt"):
            plot_method_comparison(
                reconstructed, metric=metric,
                save=str(out_dir / f"{proto}_{metric}_bar.pdf")
            )

        # Per-method plots
        for method, r in reconstructed.items():
            plot_task_curves(r, save=str(out_dir / f"{proto}_{method}_curve.pdf"))

            # Task-feedback plots (if data available)
            if getattr(r, "_task_feedback_raw", None):
                plot_task_accuracy_matrix(
                    r, metric="oa",
                    save=str(out_dir / f"{proto}_{method}_task_matrix.pdf"),
                )
                plot_task_feedback_curve(
                    r, metric="oa",
                    save=str(out_dir / f"{proto}_{method}_task_feedback_curve.pdf"),
                )

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
        # Task feedback (if available)
        self.task_feedback = d.get("task_feedback", [])
        self._task_feedback_raw = d.get("task_feedback", [])


class _SimpleTaskResult:
    def __init__(self, d: dict):
        self.task_id = d.get("task_id", 0)
        self.oa      = d.get("oa",      0.0)
        self.avg_aa  = d.get("avg_aa",  0.0)
        self.kappa   = d.get("kappa",   0.0)


def _json_to_result(d: dict, protocol: str, method: str) -> _SimpleResult:
    return _SimpleResult(d, protocol, method)


def _json_to_result_averaged(runs: list[dict], protocol: str,
                              method: str) -> _SimpleResult:
    """Average multiple seed runs into a single result object.

    Task-level metrics are averaged element-wise across runs that have
    the same number of tasks.
    """
    if len(runs) == 1:
        return _SimpleResult(runs[0], protocol, method)

    # Average scalar metrics
    def _mean(key, fallback_key=None):
        vals = []
        for r in runs:
            v = r.get(key)
            if v is None and fallback_key:
                v = r.get(fallback_key)
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else 0.0

    averaged: dict = {
        "final_oa":    _mean("final_oa", "oa_mean"),
        "final_aa":    _mean("final_aa", "aa_mean"),
        "final_kappa": _mean("final_kappa", "kappa_mean"),
        "bwt":         _mean("bwt", "bwt_mean"),
        "fwt":         _mean("fwt", "fwt_mean"),
    }

    # Average per-task metrics
    task_lists = [r.get("tasks", []) for r in runs]
    n_tasks = min(len(t) for t in task_lists) if task_lists else 0
    tasks_avg = []
    for i in range(n_tasks):
        tasks_avg.append({
            "task_id": task_lists[0][i].get("task_id", i),
            "oa":      float(np.mean([t[i].get("oa", 0) for t in task_lists])),
            "avg_aa":  float(np.mean([t[i].get("avg_aa", 0) for t in task_lists])),
            "kappa":   float(np.mean([t[i].get("kappa", 0) for t in task_lists])),
        })
    averaged["tasks"] = tasks_avg

    # Average task_feedback (use first run's structure, average values)
    fb_lists = [r.get("task_feedback", []) for r in runs]
    if all(fb_lists) and len(set(len(fb) for fb in fb_lists)) == 1:
        fb_avg = []
        for i in range(len(fb_lists[0])):
            entry = dict(fb_lists[0][i])  # copy structure
            for key in ("oa", "aa", "kappa"):
                vals = [fb[i].get(key, 0) for fb in fb_lists]
                entry[key] = float(np.mean(vals))
            fb_avg.append(entry)
        averaged["task_feedback"] = fb_avg

    return _SimpleResult(averaged, protocol, method)


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
    method_name: str | None = None,
    protocol_name: str | None = None,
    save: str | None = None,
):
    """HyperKD Fig.8-style classification map grid showing CIL evolution.

    Layout: columns = [Ground Truth, After Task 0, After Task 1, …]
    Each panel is a spatial classification map on the same scene using a
    fixed colour palette.  Uses exact pixel coordinates when available
    (stored in ``task_data["coords"]``), falls back to ``_find_test_coords``
    for legacy result files.

    Args:
        gt_map:               (H, W) int, 1-indexed ground truth (0 = bg).
        predictions_per_task: List of dicts with keys:
            - ``preds``   : (N,) int array — predicted *local* class IDs
            - ``targets`` : (N,) int array — ground-truth *local* class IDs
            - ``coords``  : (N, 2) int array or None — exact (row, col)
            - ``after_task_id``: int
            - ``oa``      : float (optional, shown as annotation)
        class_names:  Class names for the legend.
        dataset_name: For colour palette lookup.
        method_name:  Shown in suptitle.
        protocol_name: Shown in suptitle.
        save:         Path to save (pdf/png).
    """
    from benchmark.eval.colors import get_colormap, label_map_to_rgb

    _set_style()
    plt = _mpl()
    from matplotlib.patches import Patch

    num_classes = len(class_names)
    ds_name = dataset_name or ""
    cmap = get_colormap(ds_name, num_classes)
    n_panels = 1 + len(predictions_per_task)  # GT + per-task

    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    # Column 0: Ground Truth
    gt_rgb = label_map_to_rgb(gt_map, ds_name, num_classes)
    axes[0].imshow(gt_rgb)
    axes[0].set_title("Ground Truth", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    # Columns 1..T: predictions after each task
    for col, task_data in enumerate(predictions_per_task, start=1):
        img = label_map_to_rgb(gt_map, ds_name, num_classes).astype(np.float32)
        img *= 0.25  # faded background

        preds = task_data["preds"]
        coords = task_data.get("coords")
        if coords is None:
            print(f"[WARN] No exact coords for task {task_data.get('after_task_id','?')}; "
                  "using approximate reconstruction")
            coords = _find_test_coords(gt_map, task_data["targets"])

        if coords is not None:
            for (r, c), p in zip(coords, preds):
                if 0 < int(p) <= num_classes:
                    img[int(r), int(c)] = cmap[int(p)].astype(np.float32)

        img = np.clip(img, 0, 255).astype(np.uint8)
        axes[col].imshow(img)
        tid = task_data.get("after_task_id", col - 1)
        title = f"After Task {tid}"
        oa = task_data.get("oa")
        if oa is not None:
            title += f"\nOA={oa*100:.1f}%"
        axes[col].set_title(title, fontsize=8)
        axes[col].axis("off")

    # Legend below
    legend_elements = [
        Patch(facecolor=np.array(cmap[i + 1]) / 255.0, label=name)
        for i, name in enumerate(class_names)
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               fontsize=6, ncol=min(num_classes, 8),
               framealpha=0.9, borderpad=0.5)

    suptitle = ds_name
    if method_name:
        suptitle = f"{method_name} — {suptitle}"
    if protocol_name:
        suptitle += f" ({protocol_name})"
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.02)

    fig.subplots_adjust(bottom=0.12)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        for ext in (".pdf", ".png"):
            fig.savefig(str(Path(save).with_suffix(ext)),
                        bbox_inches="tight", dpi=300)
        print(f"[plot] saved → {save}")
    return fig


def plot_multi_method_maps(
    gt_map: np.ndarray,
    method_predictions: dict[str, list[dict]],
    class_names: list[str],
    dataset_name: str | None = None,
    task_id: int = -1,
    save: str | None = None,
):
    """Side-by-side classification maps comparing multiple methods.

    Layout: columns = [Ground Truth, Method1, Method2, …]

    Args:
        gt_map:             (H, W) int ground truth.
        method_predictions: {method_name: [artifact_dicts]} — use
                            the artifact whose after_task_id == task_id
                            (or the last one if task_id == -1).
        class_names:        For colour palette.
        dataset_name:       For colour palette.
        task_id:            Which task snapshot to show (-1 = last).
        save:               Path to save.
    """
    from benchmark.eval.colors import get_colormap, label_map_to_rgb
    from matplotlib.patches import Patch

    _set_style()
    plt = _mpl()

    num_classes = len(class_names)
    ds_name = dataset_name or ""
    cmap = get_colormap(ds_name, num_classes)
    methods = list(method_predictions.keys())
    n_cols = 1 + len(methods)

    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 4.5))
    if n_cols == 1:
        axes = [axes]

    # GT
    axes[0].imshow(label_map_to_rgb(gt_map, ds_name, num_classes))
    axes[0].set_title("Ground Truth", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    for col, method in enumerate(methods, start=1):
        artifacts = method_predictions[method]
        if task_id == -1:
            art = artifacts[-1]
        else:
            art = next((a for a in artifacts if a.get("after_task_id") == task_id),
                       artifacts[-1])

        img = label_map_to_rgb(gt_map, ds_name, num_classes).astype(np.float32)
        img *= 0.25
        coords = art.get("coords")
        if coords is None:
            coords = _find_test_coords(gt_map, art["targets"])
        if coords is not None:
            for (r, c), p in zip(coords, art["preds"]):
                if 0 < int(p) <= num_classes:
                    img[int(r), int(c)] = cmap[int(p)].astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        axes[col].imshow(img)
        oa = art.get("oa")
        title = method
        if oa is not None:
            title += f"\nOA={oa*100:.1f}%"
        axes[col].set_title(title, fontsize=8)
        axes[col].axis("off")

    legend_elements = [
        Patch(facecolor=np.array(cmap[i + 1]) / 255.0, label=name)
        for i, name in enumerate(class_names)
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               fontsize=6, ncol=min(num_classes, 8), framealpha=0.9)
    fig.subplots_adjust(bottom=0.12)

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        for ext in (".pdf", ".png"):
            fig.savefig(str(Path(save).with_suffix(ext)),
                        bbox_inches="tight", dpi=300)
        print(f"[plot] saved → {save}")
    return fig


# ══════════════════════════════════════════════════════════════════
# 7b. Task-feedback accuracy matrix (true CIL heatmap)
# ══════════════════════════════════════════════════════════════════

def plot_task_accuracy_matrix(
    result,
    metric: str = "oa",
    save: str | None = None,
):
    """True CIL task-feedback accuracy heatmap.

    Rows   = evaluated task (task whose classes are being tested)
    Columns = after learning task t (training stage)
    Value  = OA or AA on that specific task at that training stage

    Requires ``result.task_feedback`` to be populated (list of
    ``TaskFeedbackResult`` with ``after_task_id``, ``eval_task_id``,
    ``oa``, ``aa``).

    Args:
        result:  BenchmarkResult with .task_feedback populated.
        metric:  "oa" or "aa".
        save:    Path to save (both .pdf and .png are generated).
    """
    _set_style()
    plt = _mpl()
    sns = _sns()

    feedback = getattr(result, "task_feedback", None)
    if not feedback:
        # Try to reconstruct from JSON-loaded data
        feedback = getattr(result, "_task_feedback_raw", None)
    if not feedback:
        print("[plot_task_accuracy_matrix] no task_feedback data, skipping")
        return None

    # Build the matrix
    after_ids = sorted({f.after_task_id if hasattr(f, 'after_task_id') else f["after_task_id"]
                        for f in feedback})
    eval_ids = sorted({f.eval_task_id if hasattr(f, 'eval_task_id') else f["eval_task_id"]
                       for f in feedback})
    n_after = len(after_ids)
    n_eval = len(eval_ids)

    matrix = np.full((n_eval, n_after), np.nan)
    after_map = {t: i for i, t in enumerate(after_ids)}
    eval_map = {t: i for i, t in enumerate(eval_ids)}

    for f in feedback:
        at = f.after_task_id if hasattr(f, 'after_task_id') else f["after_task_id"]
        et = f.eval_task_id if hasattr(f, 'eval_task_id') else f["eval_task_id"]
        val = getattr(f, metric, None) or f.get(metric, f.get("oa", 0))
        if at in after_map and et in eval_map:
            matrix[eval_map[et], after_map[at]] = val * 100

    fig, ax = plt.subplots(figsize=(max(5, n_after * 0.9 + 1),
                                     max(4, n_eval * 0.7 + 1)))
    sns.heatmap(
        matrix, ax=ax, annot=True, fmt=".1f",
        cmap="RdYlGn", vmin=30, vmax=100,
        linewidths=0.5, linecolor="white",
        mask=np.isnan(matrix),
        xticklabels=[str(t) for t in after_ids],
        yticklabels=[str(t) for t in eval_ids],
        cbar_kws={"label": f"{metric.upper()} (%)"},
    )
    ax.set_xlabel("After learning task", fontsize=10)
    ax.set_ylabel("Evaluated task", fontsize=10)
    method = getattr(result, "method_name", "")
    proto = getattr(result, "protocol_name", "")
    ax.set_title(f"Task Accuracy Matrix — {method} / {proto}", fontsize=11)

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        for ext in (".pdf", ".png"):
            fig.savefig(str(Path(save).with_suffix(ext)),
                        bbox_inches="tight", dpi=300)
        print(f"[plot] saved → {save}")
    plt.tight_layout()
    return fig


def plot_task_feedback_curve(
    result,
    metric: str = "oa",
    save: str | None = None,
):
    """Line plot showing how old-task and new-task accuracy evolve.

    Three curves:
      - Mean accuracy on all seen tasks (average OA across evaluated tasks)
      - Mean accuracy on OLD tasks only (forgetting indicator)
      - Accuracy on the CURRENT task (plasticity indicator)

    Args:
        result: BenchmarkResult with .task_feedback.
        metric: "oa" or "aa".
        save:   Path to save.
    """
    _set_style()
    plt = _mpl()

    feedback = getattr(result, "task_feedback", None)
    if not feedback:
        feedback = getattr(result, "_task_feedback_raw", None)
    if not feedback:
        print("[plot_task_feedback_curve] no task_feedback data, skipping")
        return None

    # Group by after_task_id
    by_after: dict[int, list] = {}
    for f in feedback:
        at = f.after_task_id if hasattr(f, 'after_task_id') else f["after_task_id"]
        by_after.setdefault(at, []).append(f)

    after_ids = sorted(by_after)
    mean_all = []
    mean_old = []
    current_task = []

    for at in after_ids:
        entries = by_after[at]
        vals = []
        old_vals = []
        curr_val = None
        for f in entries:
            et = f.eval_task_id if hasattr(f, 'eval_task_id') else f["eval_task_id"]
            v = getattr(f, metric, None) or f.get(metric, f.get("oa", 0))
            vals.append(v)
            if et < at:
                old_vals.append(v)
            elif et == at:
                curr_val = v
        mean_all.append(np.mean(vals) * 100 if vals else 0)
        mean_old.append(np.mean(old_vals) * 100 if old_vals else np.nan)
        current_task.append(curr_val * 100 if curr_val is not None else np.nan)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(after_ids, mean_all, "-o", color=_PALETTE[0], linewidth=2,
            markersize=6, label="Mean (all seen tasks)")
    if not all(np.isnan(v) for v in mean_old):
        ax.plot(after_ids, mean_old, "--s", color=_PALETTE[3], linewidth=2,
                markersize=5, label="Mean (old tasks only)")
    if not all(np.isnan(v) for v in current_task):
        ax.plot(after_ids, current_task, ":^", color=_PALETTE[2], linewidth=2,
                markersize=5, label="Current task (plasticity)")

    ax.set_xlabel("After learning task")
    ax.set_ylabel(f"{metric.upper()} (%)")
    ax.set_xticks(after_ids)
    method = getattr(result, "method_name", "")
    proto = getattr(result, "protocol_name", "")
    ax.set_title(f"Task Feedback — {method} / {proto}", fontsize=11)
    ax.legend(loc="lower left", fontsize=9)

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        for ext in (".pdf", ".png"):
            fig.savefig(str(Path(save).with_suffix(ext)),
                        bbox_inches="tight", dpi=300)
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
