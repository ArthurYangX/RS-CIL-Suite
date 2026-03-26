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
    # Validate class_splits
    for ds in dataset_order:
        if ds not in class_splits:
            raise ValueError(f"class_splits missing entry for dataset '{ds}'")
        if ds not in num_classes:
            raise ValueError(f"num_classes missing entry for dataset '{ds}'")
        split_sum = sum(class_splits[ds])
        if split_sum != num_classes[ds]:
            raise ValueError(
                f"class_splits['{ds}'] sums to {split_sum} but "
                f"num_classes['{ds}']={num_classes[ds]}")
        if any(n <= 0 for n in class_splits[ds]):
            raise ValueError(
                f"class_splits['{ds}'] contains non-positive values: "
                f"{class_splits[ds]}")

    offsets: Dict[str, int] = {}
    off = 0
    for ds in dataset_order:
        offsets[ds] = off
        off += num_classes[ds]
    total_classes = off

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

    # Sanity check: max global ID must be < total_classes
    max_gid = max(gid for t in tasks for gid in t.global_class_ids)
    assert max_gid + 1 == total_classes, \
        f"max global_class_id={max_gid} but total_classes={total_classes}"

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
    if any(n <= 0 for n in class_splits):
        raise ValueError(
            f"class_splits contains non-positive values: {class_splits}")
    if sum(class_splits) != num_classes:
        raise ValueError(
            f"class_splits {class_splits} sums to {sum(class_splits)} "
            f"but num_classes={num_classes}")

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
    "Augsburg":       8,
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
                  "MUUFL": [4,4,3], "Augsburg": [3,3,2]},
    num_classes=NUM_CLASSES,
)

# ── B3: HSI-only cross-scene (3 datasets) ────────────────────────
PROTOCOL_B3 = build_cross_scene(
    dataset_order=["IndianPines", "PaviaU", "Salinas"],
    class_splits={"IndianPines": [4,4,4,4], "PaviaU": [3,3,3], "Salinas": [4,4,4,4]},
    num_classes=NUM_CLASSES,
)

# ── A1–A10: within-scene protocols ────────────────────────────────
PROTOCOL_A_TRENTO         = build_within_scene("Trento",         [2, 2, 2],       6)
PROTOCOL_A_HOUSTON2013    = build_within_scene("Houston2013",    [5, 5, 5],      15)
PROTOCOL_A_MUUFL          = build_within_scene("MUUFL",          [4, 4, 3],      11)
PROTOCOL_A_AUGSBURG       = build_within_scene("Augsburg",       [3, 3, 2],       8)
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
                  "MUUFL": [4,4,3],  "Augsburg": [3,3,2], "Houston2018": [5,5,5,5]},
    num_classes=NUM_CLASSES,
)

# ── B5: HSI-only + SAR + UAV cross-scene (5 datasets) ────────────
PROTOCOL_B5 = build_cross_scene(
    dataset_order=["IndianPines", "PaviaU", "Salinas", "Berlin", "WHU-Hi-LongKou"],
    class_splits={"IndianPines": [4,4,4,4], "PaviaU": [3,3,3],
                  "Salinas": [4,4,4,4], "Berlin": [3,3,2],
                  "WHU-Hi-LongKou": [3,3,3]},
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


# ══════════════════════════════════════════════════════════════════
# YAML-based custom protocol loader
# ══════════════════════════════════════════════════════════════════

def load_protocol_yaml(path: str) -> CILProtocol:
    """Load a custom protocol from a YAML file.

    YAML format::

        name: MyProtocol
        type: cross_scene           # or "within_scene"
        dataset_order: [Trento, Houston2013, MUUFL]
        class_splits:
          Trento: [2, 2, 2]
          Houston2013: [5, 5, 5]
          MUUFL: [4, 4, 3]

        # Optional:
        train_ratio: 0.15           # override default 10% (stored, applied by runner)
        shuffle_classes: false       # if true, permute class order within each dataset
        class_order_seed: 42        # seed for class permutation
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required: pip install pyyaml")

    from pathlib import Path
    with open(Path(path)) as f:
        cfg = yaml.safe_load(f)

    name = cfg.get("name", Path(path).stem)
    proto_type = cfg.get("type", "cross_scene")
    dataset_order = cfg["dataset_order"]
    class_splits = cfg["class_splits"]
    shuffle = cfg.get("shuffle_classes", False)
    seed = cfg.get("class_order_seed", 42)

    # Optionally shuffle class ordering within each dataset
    # This permutes which classes go to which tasks
    permutations: dict[str, list[int]] = {}
    if shuffle:
        import random
        rng = random.Random(seed)
        for ds in dataset_order:
            nc = NUM_CLASSES[ds]
            perm = list(range(nc))
            rng.shuffle(perm)
            permutations[ds] = perm

    if proto_type == "within_scene":
        assert len(dataset_order) == 1, "within_scene requires exactly 1 dataset"
        ds = dataset_order[0]
        splits = class_splits[ds] if isinstance(class_splits, dict) else class_splits
        protocol = build_within_scene(ds, splits, NUM_CLASSES[ds])
    else:
        protocol = build_cross_scene(dataset_order, class_splits, NUM_CLASSES)

    # Apply permutation to task class IDs
    if permutations:
        for task in protocol.tasks:
            ds = task.dataset_name
            if ds in permutations:
                perm = permutations[ds]
                task.class_ids = [perm[c] for c in task.class_ids]
                offset = protocol.offsets.get(ds, 0)
                task.global_class_ids = [c + offset for c in task.class_ids]

    protocol.name = name

    # Attach optional metadata for the runner
    protocol.train_ratio = cfg.get("train_ratio", None)
    protocol.class_order_seed = seed
    protocol.shuffle_classes = shuffle
    protocol._class_permutations = permutations

    return protocol


def get_protocol(name_or_path: str) -> CILProtocol:
    """Get a protocol by registry name or YAML file path.

    Args:
        name_or_path: Either a key in PROTOCOLS (e.g. "B1") or a path
                      to a custom YAML protocol file.

    Returns:
        CILProtocol instance.
    """
    if name_or_path in PROTOCOLS:
        return PROTOCOLS[name_or_path]

    from pathlib import Path
    p = Path(name_or_path)
    if p.exists() and p.suffix in (".yaml", ".yml"):
        return load_protocol_yaml(name_or_path)

    # Also check configs/protocols/ directory
    configs_dir = Path(__file__).parent.parent / "configs" / "protocols"
    candidate = configs_dir / f"{name_or_path}.yaml"
    if candidate.exists():
        return load_protocol_yaml(str(candidate))

    raise ValueError(
        f"Unknown protocol '{name_or_path}'. "
        f"Available: {sorted(PROTOCOLS)} or provide a YAML file path."
    )
