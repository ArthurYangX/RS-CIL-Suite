"""Dataset registry — maps names to classes and default data paths."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Type

from .base import RSDataset
from .hsi_lidar import Trento, Houston2013, MUUFL, Augsburg, Houston2018
from .hsi_only  import IndianPines, PaviaU, Salinas, Berlin, WHUHiLongKou

# ── Registry ─────────────────────────────────────────────────────

DATASETS: Dict[str, Type[RSDataset]] = {
    # HSI + LiDAR
    "Trento":         Trento,
    "Houston2013":    Houston2013,
    "MUUFL":          MUUFL,
    "Augsburg":       Augsburg,
    "Houston2018":    Houston2018,
    # HSI only
    "IndianPines":    IndianPines,
    "PaviaU":         PaviaU,
    "Salinas":        Salinas,
    # HSI + SAR
    "Berlin":         Berlin,
    # UAV HSI
    "WHU-Hi-LongKou": WHUHiLongKou,
}

# Default server paths (override via config or env)
_DEFAULT_ROOTS: Dict[str, str] = {
    "Trento":         "~/autodl-tmp/datasets/Trento",
    "Houston2013":    "~/autodl-tmp/datasets/Houston2013",
    "MUUFL":          "~/autodl-tmp/datasets/MUUFL",
    "Augsburg":       "~/autodl-tmp/datasets/Augsburg",
    "Houston2018":    "~/autodl-tmp/datasets/Houston2018",
    "IndianPines":    "~/autodl-tmp/datasets/IndianPines",
    "PaviaU":         "~/autodl-tmp/datasets/PaviaU",
    "Salinas":        "~/autodl-tmp/datasets/Salinas",
    "Berlin":         "~/autodl-tmp/datasets/Berlin",
    "WHU-Hi-LongKou": "~/autodl-tmp/datasets/WHU-Hi-LongKou",
}


def get_dataset(
    name: str,
    root: str | Path | None = None,
    patch_size: int = 7,
    pca_components: int = 36,
) -> RSDataset:
    """Instantiate a dataset by name.

    Args:
        name:          Dataset key (e.g. "Trento").
        root:          Path to the dataset directory.
                       Falls back to _DEFAULT_ROOTS if None.
        patch_size:    Spatial patch size (default 7).
        pca_components: PCA reduction for HSI (default 36).
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    cls = DATASETS[name]
    if root is None:
        root = Path(_DEFAULT_ROOTS[name]).expanduser()
    return cls(root=root, patch_size=patch_size, pca_components=pca_components)


def list_datasets() -> list[str]:
    return list(DATASETS.keys())
