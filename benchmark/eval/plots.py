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


def _nanmean(values: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def _nanstd(values: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(values, axis=axis)


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
        "mean": {m: _nanmean(np.vstack(v), axis=0) for m, v in stack.items()},
        "std": {m: _nanstd(np.vstack(v), axis=0) for m, v in stack.items()},
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
        "mean": _nanmean(stack, axis=0),
        "std": _nanstd(stack, axis=0),
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
    vals = []
    stds = []
    for result in results.values():
        runs = _coerce_runs(result)
        series = [float(_run_attr(run, metric, 0.0)) for run in runs]
        vals.append(float(np.mean(series)) * scale if series else 0.0)
        stds.append(float(np.std(series)) * scale if len(series) > 1 else 0.0)
    label = {"final_oa": "Final OA (%)", "final_aa": "Final AA (%)",
             "final_kappa": "Final κ", "bwt": "BWT (pp)", "fwt": "FWT (%)"}

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), 4.5))
    bars = ax.bar(names, vals, color=_PALETTE[:len(names)], width=0.6,
                  edgecolor="white", linewidth=0.8)
    if any(stds):
        ax.errorbar(names, vals, yerr=stds, fmt="none", ecolor="#333333",
                    elinewidth=1, capsize=3, capthick=1)

    # Value labels on bars
    for bar, v, s in zip(bars, vals, stds):
        txt = f"{v:.1f}" if s == 0 else f"{v:.1f}\n±{s:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                txt, ha="center", va="bottom", fontsize=8)

    ax.set_ylabel(label.get(metric, metric))
    protocol = _protocol_name(_coerce_runs(next(iter(results.values())))[0])
    ax.set_title(f"{label.get(metric, metric)} — {protocol}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")

    if save:
        _save_publication_figure(fig, save)
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
            seen_vals.append(_nanmean(matrix[:, col_idx]))
            row_idx = eval_to_row.get(after_task_id)
            current_vals.append(matrix[row_idx, col_idx] if row_idx is not None else np.nan)

            old_row_indices = [eval_to_row[t] for t in eval_ids if t < after_task_id]
            if old_row_indices:
                old_vals.append(_nanmean(matrix[old_row_indices, col_idx]))
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
        mean = _nanmean(values, axis=0)
        std = _nanstd(values, axis=0)
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
        aggregated = _aggregate_task_curves(_coerce_runs(result))
        if aggregated is None:
            continue
        task_ids = aggregated["task_ids"]
        vals = aggregated["mean"][metric] * scale
        stds = aggregated["std"][metric] * scale
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(task_ids, vals, "-o", color=color,
                label=name, linewidth=2, markersize=5)
        if aggregated["num_runs"] > 1:
            ax.fill_between(task_ids, vals - stds, vals + stds,
                            color=color, alpha=0.14, linewidth=0)

    ax.set_xlabel("Task")
    ax.set_ylabel(label_map.get(metric, metric))
    protocol = _protocol_name(_coerce_runs(next(iter(results.values())))[0])
    ax.set_title(f"{label_map.get(metric, metric)} progression — {protocol}")
    ax.set_xticks(task_ids)
    ax.legend(loc="lower left", fontsize=9)

    if save:
        _save_publication_figure(fig, save)
    plt.tight_layout()
    return fig


# ── 5. Full suite from results directory ─────────────────────────

def plot_suite(results_dir: str | Path, out_dir: str | Path = "figs/",
               protocol_filter: str | None = None):
    """Generate the full plotting suite from saved result files."""
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_protocol: dict[str, dict[str, list[_SimpleResult]]] = {}
    for f in sorted(results_dir.glob("**/*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if "method" not in data or "protocol" not in data:
            continue
        # Skip multi-seed summary files when per-seed files are also present.
        if "seeds" in data and "seed" not in data and not data.get("task_evals"):
            continue

        proto = data.get("protocol", "unknown")
        method = data.get("method", "unknown")
        if protocol_filter and proto != protocol_filter:
            continue
        by_protocol.setdefault(proto, {}).setdefault(method, []).append(
            _json_to_result(data, proto, method, source_file=f)
        )

    if not by_protocol:
        print(f"[plot_suite] no JSON files found in {results_dir}")
        return

    plt = _mpl()

    for proto, methods in by_protocol.items():
        for metric in ("final_oa", "final_aa", "bwt"):
            plot_method_comparison(
                methods,
                metric=metric,
                save=str(out_dir / f"{proto}_{metric}_bar.pdf"),
            )

        for method, runs in methods.items():
            plot_task_curves(runs, save=str(out_dir / f"{proto}_{method}_curve.pdf"))
            plot_task_accuracy_matrix(
                runs, metric="oa",
                save=str(out_dir / f"{proto}_{method}_task_matrix.pdf"),
            )
            plot_task_feedback_curve(
                runs, metric="oa",
                save=str(out_dir / f"{proto}_{method}_task_feedback_curve.pdf"),
            )

            # Maps-per-task use the first run that has exact saved artifacts.
            map_run = next((run for run in runs if _task_eval_records(run) and _artifact_path(run)), None)
            if map_run is not None:
                artifact_arrays = _load_artifact_arrays(map_run)
                if artifact_arrays is not None:
                    for ds_name, mapping in _dataset_mappings(map_run).items():
                        gt_key = mapping.get("gt_map_key")
                        if gt_key is None or gt_key not in artifact_arrays:
                            warnings.warn(
                                f"GT map unavailable for {proto}/{method}/{ds_name}; skipping maps-per-task.",
                                RuntimeWarning,
                            )
                            continue
                        task_evals = [
                            _materialize_task_eval(record, artifact_arrays)
                            for record in _task_eval_records(map_run)
                            if record.get("dataset_name") == ds_name
                        ]
                        if not task_evals:
                            continue
                        suffix = "" if len(_dataset_mappings(map_run)) == 1 else f"_{ds_name}"
                        plot_classification_maps_per_task(
                            gt_map=artifact_arrays[gt_key],
                            predictions_per_task=task_evals,
                            class_names=mapping.get("class_names", []),
                            dataset_name=ds_name,
                            method_name=method,
                            protocol_name=proto,
                            save=str(out_dir / f"{proto}_{method}{suffix}_maps_per_task.pdf"),
                        )

        if len(methods) > 1:
            all_datasets = sorted({
                ds_name
                for runs in methods.values()
                for run in runs
                for ds_name in _dataset_mappings(run).keys()
            })
            for ds_name in all_datasets:
                method_predictions: dict[str, list[dict]] = {}
                class_names: list[str] | None = None
                gt_map = None
                for method, runs in methods.items():
                    map_run = next(
                        (run for run in runs if ds_name in _dataset_mappings(run) and _task_eval_records(run) and _artifact_path(run)),
                        None,
                    )
                    if map_run is None:
                        continue
                    artifact_arrays = _load_artifact_arrays(map_run)
                    if artifact_arrays is None:
                        continue
                    mapping = _dataset_mappings(map_run).get(ds_name, {})
                    gt_key = mapping.get("gt_map_key")
                    if gt_key is None or gt_key not in artifact_arrays:
                        continue
                    gt_map = artifact_arrays[gt_key]
                    class_names = mapping.get("class_names", [])
                    task_evals = [
                        _materialize_task_eval(record, artifact_arrays)
                        for record in _task_eval_records(map_run)
                        if record.get("dataset_name") == ds_name
                    ]
                    if task_evals:
                        method_predictions[method] = task_evals

                if gt_map is not None and class_names and len(method_predictions) > 1:
                    suffix = "" if len(all_datasets) == 1 else f"_{ds_name}"
                    plot_multi_method_maps(
                        gt_map=gt_map,
                        method_predictions=method_predictions,
                        class_names=class_names,
                        dataset_name=ds_name,
                        protocol_name=proto,
                        save=str(out_dir / f"{proto}{suffix}_multi_method_maps.pdf"),
                    )

        if len(methods) > 1:
            plot_methods_overlay(
                methods,
                metric="oa",
                save=str(out_dir / f"{proto}_overlay_oa.pdf"),
            )

        plt.close("all")

    print(f"[plot_suite] figures saved to {out_dir}")


# ── helpers ───────────────────────────────────────────────────────

class _SimpleResult:
    """Lightweight result object reconstructed from a saved JSON dict."""
    def __init__(self, d: dict, protocol: str, method: str, source_file: str | Path | None = None):
        self.protocol_name = protocol
        self.method_name   = method
        self.seed = d.get("seed")
        self.source_file = str(source_file) if source_file is not None else None
        self.final_oa    = d.get("final_oa",    d.get("oa_mean",    0.0))
        self.final_aa    = d.get("final_aa",    d.get("aa_mean",    0.0))
        self.final_kappa = d.get("final_kappa", d.get("kappa_mean", 0.0))
        self.bwt         = d.get("bwt",         d.get("bwt_mean",   0.0))
        self.fwt         = d.get("fwt",         d.get("fwt_mean",   0.0))
        self.task_results = [_SimpleTaskResult(t) for t in d.get("tasks", [])]
        self.task_feedback = d.get("task_feedback", [])
        self.task_evals = d.get("task_evals", [])
        self.dataset_mappings = d.get("dataset_mappings", {})
        self.artifacts_file = d.get("artifacts_file")


class _SimpleTaskResult:
    def __init__(self, d: dict):
        self.task_id = d.get("task_id", 0)
        self.oa      = d.get("oa",      0.0)
        self.avg_aa  = d.get("avg_aa",  0.0)
        self.kappa   = d.get("kappa",   0.0)


def _json_to_result(
    d: dict,
    protocol: str,
    method: str,
    source_file: str | Path | None = None,
) -> _SimpleResult:
    return _SimpleResult(d, protocol, method, source_file=source_file)


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
    coords: np.ndarray | None = None,
    global_to_local: dict[int, int] | None = None,
):
    """Render a classification map using exact spatial coordinates when available."""
    from benchmark.eval.colors import get_colormap

    _set_style()
    plt = _mpl()

    num_classes = len(class_names)
    ds_name = dataset_name or ""
    cmap = get_colormap(ds_name, num_classes)
    gt_rgb = _render_classification_map_rgb(
        gt_map=gt_map,
        preds=preds,
        targets=targets,
        num_classes=num_classes,
        dataset_name=ds_name,
        coords=coords,
        global_to_local=global_to_local,
        show_errors=show_errors,
    )

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
        _save_publication_figure(fig, save)
    plt.tight_layout()
    return fig


def _find_test_coords(gt_map, targets, global_to_local=None):
    """Match test pixel targets back to their (row, col) in gt_map.

    This is an **approximate fallback** for legacy result files that lack
    exact saved coordinates.  New experiments should always use exact coords.

    Args:
        gt_map:          (H, W) int array, 1-indexed local labels.
        targets:         (N,) array, may be global IDs.
        global_to_local: Optional mapping {global_id: local_id} for cross-scene
                         protocols where targets use global IDs but gt_map uses
                         1-indexed local IDs.

    Returns:
        (N, 2) array of (row, col) or None if matching fails.
    """
    import warnings
    warnings.warn(
        "Using approximate coordinate reconstruction. "
        "New experiments should rely on exact saved coords instead.",
        RuntimeWarning,
        stacklevel=2,
    )

    # Map targets to local labels that match gt_map (1-indexed)
    if global_to_local is not None:
        local_targets = np.array([global_to_local.get(int(t), int(t)) + 1
                                  for t in targets])
    else:
        local_targets = targets

    unique_targets = np.unique(local_targets)
    coords_by_class = {}
    for c in unique_targets:
        rows, cols = np.where(gt_map == c)
        coords_by_class[c] = list(zip(rows, cols))

    result = []
    usage = {c: 0 for c in unique_targets}
    for t in local_targets:
        c = int(t)
        if c in coords_by_class and usage[c] < len(coords_by_class[c]):
            result.append(coords_by_class[c][usage[c]])
            usage[c] += 1
        else:
            result.append((0, 0))

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
    show_oa: bool = True,
    max_cols: int = 5,
):
    """HyperKD-style exact-coordinate classification evolution figure."""
    from benchmark.eval.colors import get_colormap, label_map_to_rgb
    from matplotlib.patches import Patch

    _set_style()
    plt = _mpl()

    num_classes = len(class_names)
    ds_name = dataset_name or ""
    dataset_mapping = {
        "class_names": list(class_names),
        "local_class_ids": predictions_per_task[0].get("dataset_local_class_ids", list(range(len(class_names)))),
        "global_class_ids": predictions_per_task[0].get("dataset_global_class_ids", []),
    } if predictions_per_task else {"class_names": list(class_names)}
    snapshots = _build_prediction_snapshots(
        predictions_per_task,
        dataset_name=ds_name,
        dataset_mapping=dataset_mapping,
    )
    if not snapshots:
        warnings.warn(
            f"Exact coordinates are unavailable for dataset '{ds_name}'; skipping maps-per-task figure.",
            RuntimeWarning,
        )
        return None

    cmap = get_colormap(ds_name, num_classes)
    panels = [{"title": "Ground Truth", "rgb": label_map_to_rgb(gt_map, ds_name, num_classes)}]
    for snapshot in snapshots:
        title = f"After T{snapshot['after_task_id'] + 1}"
        if show_oa:
            title += f"\nOA={snapshot['oa'] * 100:.1f}%"
        panels.append({
            "title": title,
            "rgb": _render_classification_map_rgb(
                gt_map=gt_map,
                preds=snapshot["preds"],
                targets=snapshot["targets"],
                num_classes=num_classes,
                dataset_name=ds_name,
                coords=snapshot["coords"],
                global_to_local=snapshot.get("global_to_local"),
            ),
        })

    n_panels = len(panels)
    ncols = min(max_cols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.9 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, panel in zip(axes.flat, panels):
        ax.imshow(panel["rgb"])
        ax.set_title(panel["title"], fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(panels):]:
        ax.axis("off")

    legend_elements = [
        Patch(facecolor=np.array(cmap[i + 1]) / 255.0, label=name)
        for i, name in enumerate(class_names)
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               fontsize=6, ncol=min(num_classes, 8), framealpha=0.9)

    title_bits = [bit for bit in (method_name, ds_name, protocol_name) if bit]
    fig.suptitle(" — ".join(title_bits) if title_bits else ds_name, fontsize=11, y=1.01)
    fig.subplots_adjust(bottom=0.12)
    if save:
        _save_publication_figure(fig, save)
    return fig


def plot_multi_method_maps(
    gt_map: np.ndarray,
    method_predictions: dict[str, list[dict]],
    class_names: list[str],
    dataset_name: str | None = None,
    protocol_name: str | None = None,
    save: str | None = None,
    show_oa: bool = True,
):
    """Multi-method exact-coordinate comparison figure.

    Rows are incremental stages, columns are GT + methods.
    """
    from benchmark.eval.colors import get_colormap, label_map_to_rgb
    from matplotlib.patches import Patch

    _set_style()
    plt = _mpl()

    num_classes = len(class_names)
    ds_name = dataset_name or ""
    cmap = get_colormap(ds_name, num_classes)
    methods = list(method_predictions.keys())
    snapshots_by_method = {
        method: _build_prediction_snapshots(preds, dataset_name=ds_name)
        for method, preds in method_predictions.items()
    }
    available_methods = {
        method: snapshots for method, snapshots in snapshots_by_method.items() if snapshots
    }
    if not available_methods:
        warnings.warn(
            f"No exact-coordinate snapshots available for dataset '{ds_name}'.",
            RuntimeWarning,
        )
        return None

    after_ids = sorted({
        snapshot["after_task_id"]
        for snapshots in available_methods.values()
        for snapshot in snapshots
    })
    nrows = len(after_ids)
    ncols = 1 + len(available_methods)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    method_names = list(available_methods.keys())
    for row_idx, after_task_id in enumerate(after_ids):
        axes[row_idx, 0].imshow(label_map_to_rgb(gt_map, ds_name, num_classes))
        axes[row_idx, 0].set_title("Ground Truth" if row_idx == 0 else "")
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_ylabel(f"After T{after_task_id + 1}", fontsize=9)

        for col_idx, method_name in enumerate(method_names, start=1):
            snapshot = next(
                (item for item in available_methods[method_name] if item["after_task_id"] == after_task_id),
                None,
            )
            if snapshot is None:
                axes[row_idx, col_idx].axis("off")
                continue
            rgb = _render_classification_map_rgb(
                gt_map=gt_map,
                preds=snapshot["preds"],
                targets=snapshot["targets"],
                num_classes=num_classes,
                dataset_name=ds_name,
                coords=snapshot["coords"],
                global_to_local=snapshot.get("global_to_local"),
            )
            axes[row_idx, col_idx].imshow(rgb)
            title = method_name if row_idx == 0 else ""
            if row_idx == 0 and show_oa:
                title += f"\nOA={snapshot['oa'] * 100:.1f}%"
            elif show_oa:
                title = f"OA={snapshot['oa'] * 100:.1f}%"
            axes[row_idx, col_idx].set_title(title, fontsize=8)
            axes[row_idx, col_idx].axis("off")

    legend_elements = [
        Patch(facecolor=np.array(cmap[i + 1]) / 255.0, label=name)
        for i, name in enumerate(class_names)
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               fontsize=6, ncol=min(num_classes, 8), framealpha=0.9)
    title_bits = [bit for bit in (protocol_name, ds_name) if bit]
    fig.suptitle(" — ".join(title_bits) if title_bits else ds_name, fontsize=11, y=1.01)
    fig.subplots_adjust(bottom=0.12)

    if save:
        _save_publication_figure(fig, save)
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
