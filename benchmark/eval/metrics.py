"""Evaluation metrics for RS-CIL benchmark.

Standard RS metrics:
  OA   — Overall Accuracy
  AA   — Average per-class Accuracy
  Kappa — Cohen's Kappa coefficient

CL metrics (computed over the task sequence):
  Forgetting  — drop from peak accuracy to final accuracy per dataset
  Plasticity  — accuracy on new classes at the task they are introduced
  BWT  — Backward Transfer (average forgetting)
  Plasticity (mean) — mean first-seen AA across all datasets
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class TaskResult:
    """Evaluation result after completing task `task_id`."""
    task_id: int
    # Per-dataset accuracy (AA over classes seen so far in that dataset)
    per_dataset: Dict[str, float]   # {dataset_name: AA}
    avg_aa: float                   # mean of per_dataset values
    oa: float                       # overall accuracy over all seen classes
    kappa: float


@dataclass
class TaskFeedbackResult:
    """Task-aware evaluation result for one (after_task, eval_task) pair."""
    after_task_id: int
    eval_task_id: int
    dataset_name: str
    oa: float
    aa: float
    kappa: float
    num_samples: int = 0


@dataclass
class BenchmarkResult:
    """Accumulated results over the full CIL sequence."""
    protocol_name: str
    method_name: str
    task_results: List[TaskResult] = field(default_factory=list)
    task_feedback: List[TaskFeedbackResult] = field(default_factory=list)

    # Derived (filled by compute_cl_metrics)
    forgetting:  Dict[str, float] = field(default_factory=dict)   # per dataset: peak − final (≥0)
    plasticity:  Dict[str, float] = field(default_factory=dict)   # per dataset: AA when first seen
    bwt: float = 0.0    # backward transfer: final − peak (≤0 means forgetting)
    fwt: float = 0.0    # mean first-seen AA (plasticity); kept as 'fwt' for backward compat
    final_aa:    float = 0.0
    final_oa:    float = 0.0
    final_kappa: float = 0.0

    # Internal: first-appearance AA per dataset (for plasticity/FWT)
    _first_aa: Dict[str, float] = field(default_factory=dict, repr=False)

    def add(self, result: TaskResult):
        # Record AA the first time each dataset appears (plasticity)
        for ds, aa in result.per_dataset.items():
            if ds not in self._first_aa:
                self._first_aa[ds] = aa
        self.task_results.append(result)

    def add_task_feedback(self, result: TaskFeedbackResult):
        self.task_feedback.append(result)

    def compute_cl_metrics(self):
        if not self.task_results:
            return
        final = self.task_results[-1]
        self.final_aa    = final.avg_aa
        self.final_oa    = final.oa
        self.final_kappa = final.kappa

        # Forgetting: peak AA → final AA per dataset
        self.forgetting = {}
        for ds in final.per_dataset:
            accs = [r.per_dataset.get(ds, 0.0) for r in self.task_results
                    if ds in r.per_dataset]
            if accs:
                peak = max(accs)
                self.forgetting[ds] = peak - final.per_dataset.get(ds, 0.0)

        # BWT = final − peak (negative = forgetting, standard CL convention)
        self.bwt = -float(np.mean(list(self.forgetting.values()))) if self.forgetting else 0.0

        # Plasticity: AA at first introduction
        self.plasticity = dict(self._first_aa)
        self.fwt = float(np.mean(list(self.plasticity.values()))) if self.plasticity else 0.0

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Method: {self.method_name}  |  Protocol: {self.protocol_name}",
            f"{'='*60}",
            f"  Final OA   : {self.final_oa*100:.2f}%",
            f"  Final AA   : {self.final_aa*100:.2f}%",
            f"  Final Kappa: {self.final_kappa:.4f}",
            f"  BWT        : {self.bwt*100:.2f}pp  (negative = forgetting)",
            f"  Plasticity : {self.fwt*100:.2f}%   (mean first-seen AA)",
        ]
        for ds, f in self.forgetting.items():
            pl = self.plasticity.get(ds, 0.0)
            lines.append(f"    [{ds}] forgetting={f*100:.2f}pp  plasticity={pl*100:.2f}%")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Core metric functions
# ══════════════════════════════════════════════════════════════════

def overall_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """OA: fraction of correctly classified pixels."""
    return float((preds == targets).mean())


def average_accuracy(preds: np.ndarray, targets: np.ndarray,
                     class_ids: List[int]) -> float:
    """AA: mean per-class recall over the given class IDs."""
    accs = []
    for c in class_ids:
        mask = targets == c
        if mask.sum() > 0:
            accs.append(float((preds[mask] == c).mean()))
    return float(np.mean(accs)) if accs else 0.0


def cohen_kappa(preds: np.ndarray, targets: np.ndarray,
                class_ids: List[int]) -> float:
    """Cohen's Kappa coefficient."""
    n = len(targets)
    if n == 0:
        return 0.0
    # Observed agreement
    p_o = float((preds == targets).mean())
    # Expected agreement
    p_e = 0.0
    for c in class_ids:
        p_pred = float((preds == c).mean())
        p_true = float((targets == c).mean())
        p_e += p_pred * p_true
    if abs(1.0 - p_e) < 1e-10:
        return 1.0 if p_o >= p_e else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def evaluate(
    preds: np.ndarray,
    targets: np.ndarray,
    seen_classes: List[int],
    class_to_dataset: Dict[int, str],
    dataset_order: List[str],
) -> TaskResult:
    """Compute all metrics for one evaluation snapshot.

    Args:
        preds:             (N,) predicted class IDs (global)
        targets:           (N,) ground-truth class IDs (global)
        seen_classes:      list of all global class IDs seen so far
        class_to_dataset:  maps global class ID → dataset name
        dataset_order:     list of dataset names for ordering output
    """
    oa    = overall_accuracy(preds, targets)
    aa    = average_accuracy(preds, targets, seen_classes)
    kappa = cohen_kappa(preds, targets, seen_classes)

    per_dataset: Dict[str, float] = {}
    for ds in dataset_order:
        ds_classes = [c for c in seen_classes if class_to_dataset.get(c) == ds]
        if not ds_classes:
            continue
        mask = np.isin(targets, ds_classes)
        if mask.sum() > 0:
            per_dataset[ds] = average_accuracy(preds[mask], targets[mask], ds_classes)

    avg_aa = float(np.mean(list(per_dataset.values()))) if per_dataset else 0.0

    return TaskResult(
        task_id=-1,   # caller sets this
        per_dataset=per_dataset,
        avg_aa=avg_aa,
        oa=oa,
        kappa=kappa,
    )
