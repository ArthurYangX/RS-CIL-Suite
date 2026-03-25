"""CIL protocol definitions.

A protocol specifies:
  - which datasets participate
  - in what order
  - how classes are partitioned into tasks

Three protocols:
  A  within_scene   — single dataset, classes split incrementally
  B  cross_scene    — multiple datasets in sequence (our main focus)
  C  cross_sensor   — datasets grouped by sensor type
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Task:
    task_id: int
    dataset_name: str
    class_ids: List[int]          # 0-indexed within the dataset
    global_class_ids: List[int]   # offset-corrected ids for the full sequence
    n_train: int = 0              # filled after data loading
    n_test:  int = 0


@dataclass
class CILProtocol:
    name: str
    tasks: List[Task]
    dataset_order: List[str]
    # class offset per dataset (for building a unified label space)
    offsets: Dict[str, int] = field(default_factory=dict)

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    @property
    def total_classes(self) -> int:
        return sum(len(t.class_ids) for t in self.tasks)

    def summary(self) -> str:
        lines = [f"Protocol: {self.name}  ({self.num_tasks} tasks, "
                 f"{self.total_classes} classes total)"]
        for t in self.tasks:
            lines.append(
                f"  Task {t.task_id:2d}: [{t.dataset_name}] "
                f"classes {t.class_ids}  "
                f"(global {t.global_class_ids})"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Protocol builders
# ══════════════════════════════════════════════════════════════════

def build_cross_scene(
    dataset_order: List[str],
    class_splits: Dict[str, List[int]],
    num_classes: Dict[str, int],
) -> CILProtocol:
    """Protocol B: cross-scene CIL.

    Each entry in class_splits[ds] is the number of classes per task for that
    dataset.  Tasks within a dataset are presented consecutively.

    Example:
        dataset_order = ["Trento", "Houston2013", "MUUFL"]
        class_splits  = {"Trento": [2,2,2], "Houston2013": [5,5,5], "MUUFL": [4,4,3]}
    """
    offsets: Dict[str, int] = {}
    off = 0
    for ds in dataset_order:
        offsets[ds] = off
        off += num_classes[ds]

    tasks = []
    task_id = 0
    for ds in dataset_order:
        local = 0
        for n in class_splits[ds]:
            local_ids  = list(range(local, local + n))
            global_ids = [c + offsets[ds] for c in local_ids]
            tasks.append(Task(task_id=task_id, dataset_name=ds,
                              class_ids=local_ids, global_class_ids=global_ids))
            local += n
            task_id += 1

    return CILProtocol(
        name=f"CrossScene({'→'.join(dataset_order)})",
        tasks=tasks,
        dataset_order=dataset_order,
        offsets=offsets,
    )


def build_within_scene(
    dataset_name: str,
    class_splits: List[int],
    num_classes: int,
) -> CILProtocol:
    """Protocol A: within-scene CIL.

    class_splits: list of group sizes, must sum to num_classes.

    Example:
        build_within_scene("Houston2013", [5, 5, 5], 15)
    """
    assert sum(class_splits) == num_classes, \
        f"class_splits {class_splits} must sum to {num_classes}"

    tasks = []
    local = 0
    for task_id, n in enumerate(class_splits):
        ids = list(range(local, local + n))
        tasks.append(Task(task_id=task_id, dataset_name=dataset_name,
                          class_ids=ids, global_class_ids=ids))
        local += n

    return CILProtocol(
        name=f"WithinScene({dataset_name})",
        tasks=tasks,
        dataset_order=[dataset_name],
        offsets={dataset_name: 0},
    )


# ══════════════════════════════════════════════════════════════════
# Pre-defined standard protocols
# ══════════════════════════════════════════════════════════════════

# Number of classes per dataset (fixed by ground-truth labels)
NUM_CLASSES = {
    # HSI + LiDAR
    "Trento":         6,
    "Houston2013":    15,
    "MUUFL":          11,
    "Augsburg":       7,
    "Houston2018":    20,
    # HSI only
    "IndianPines":    16,
    "PaviaU":         9,
    "Salinas":        16,
    # HSI + SAR
    "Berlin":         8,
    # UAV HSI
    "WHU-Hi-LongKou": 9,
}

# ── B1: Trento → Houston2013 → MUUFL  (3-dataset, 9-task) ────────
PROTOCOL_B1 = build_cross_scene(
    dataset_order=["Trento", "Houston2013", "MUUFL"],
    class_splits={"Trento": [2, 2, 2], "Houston2013": [5, 5, 5], "MUUFL": [4, 4, 3]},
    num_classes=NUM_CLASSES,
)

# ── B2: add Augsburg (4-dataset, 12-task) ─────────────────────────
PROTOCOL_B2 = build_cross_scene(
    dataset_order=["Trento", "Houston2013", "MUUFL", "Augsburg"],
    class_splits={"Trento": [2,2,2], "Houston2013": [5,5,5],
                  "MUUFL": [4,4,3], "Augsburg": [3,2,2]},
    num_classes=NUM_CLASSES,
)

# ── B3: add Houston2018 (5-dataset) ──────────────────────────────
PROTOCOL_B3 = build_cross_scene(
    dataset_order=["Trento", "Houston2013", "MUUFL", "Augsburg", "Houston2018"],
    class_splits={"Trento": [2,2,2], "Houston2013": [5,5,5],
                  "MUUFL": [4,4,3],  "Augsburg": [3,2,2], "Houston2018": [5,5,5,5]},
    num_classes=NUM_CLASSES,
)

# ── A1–A10: within-scene protocols ────────────────────────────────
PROTOCOL_A_TRENTO         = build_within_scene("Trento",         [2, 2, 2],       6)
PROTOCOL_A_HOUSTON2013    = build_within_scene("Houston2013",    [5, 5, 5],      15)
PROTOCOL_A_MUUFL          = build_within_scene("MUUFL",          [4, 4, 3],      11)
PROTOCOL_A_AUGSBURG       = build_within_scene("Augsburg",       [3, 2, 2],       7)
PROTOCOL_A_HOUSTON2018    = build_within_scene("Houston2018",    [5, 5, 5, 5],   20)
PROTOCOL_A_INDIANPINES    = build_within_scene("IndianPines",    [4, 4, 4, 4],   16)
PROTOCOL_A_PAVIAU         = build_within_scene("PaviaU",         [3, 3, 3],       9)
PROTOCOL_A_SALINAS        = build_within_scene("Salinas",        [4, 4, 4, 4],   16)
PROTOCOL_A_BERLIN         = build_within_scene("Berlin",         [3, 3, 2],       8)
PROTOCOL_A_WHUHILONGKOU   = build_within_scene("WHU-Hi-LongKou", [3, 3, 3],       9)

# ── B4: large cross-scene (all HSI+LiDAR datasets) ────────────────
PROTOCOL_B4 = build_cross_scene(
    dataset_order=["Trento", "Houston2013", "MUUFL", "Augsburg", "Houston2018"],
    class_splits={"Trento": [2,2,2], "Houston2013": [5,5,5],
                  "MUUFL": [4,4,3],  "Augsburg": [3,2,2], "Houston2018": [5,5,5,5]},
    num_classes=NUM_CLASSES,
)

# ── B5: cross-modality (HSI-only datasets) ─────────────────────────
PROTOCOL_B5 = build_cross_scene(
    dataset_order=["IndianPines", "PaviaU", "Salinas"],
    class_splits={"IndianPines": [4,4,4,4], "PaviaU": [3,3,3], "Salinas": [4,4,4,4]},
    num_classes=NUM_CLASSES,
)

# Registry
PROTOCOLS = {
    "B1": PROTOCOL_B1,
    "B2": PROTOCOL_B2,
    "B3": PROTOCOL_B3,
    "B4": PROTOCOL_B4,
    "B5": PROTOCOL_B5,
    "A_Trento":         PROTOCOL_A_TRENTO,
    "A_Houston2013":    PROTOCOL_A_HOUSTON2013,
    "A_MUUFL":          PROTOCOL_A_MUUFL,
    "A_Augsburg":       PROTOCOL_A_AUGSBURG,
    "A_Houston2018":    PROTOCOL_A_HOUSTON2018,
    "A_IndianPines":    PROTOCOL_A_INDIANPINES,
    "A_PaviaU":         PROTOCOL_A_PAVIAU,
    "A_Salinas":        PROTOCOL_A_SALINAS,
    "A_Berlin":         PROTOCOL_A_BERLIN,
    "A_WHUHiLongKou":   PROTOCOL_A_WHUHILONGKOU,
}
