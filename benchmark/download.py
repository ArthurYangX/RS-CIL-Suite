"""Dataset download helper for the RS-CIL benchmark.

Downloads all 10 datasets to ~/autodl-tmp/datasets/ (default) or a custom root.

Usage:
    python benchmark/download.py --dataset all --root ~/autodl-tmp/datasets
    python benchmark/download.py --dataset IndianPines
    python benchmark/download.py --list
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


# ── Download metadata ─────────────────────────────────────────────
#
# Source preference:
#   1. gdown (Google Drive)  — when a Drive ID is known
#   2. wget/curl             — for direct .zip or .mat links
#
# Known download sources (verified public mirrors):
DATASET_SOURCES: dict[str, dict] = {

    # ── HSI + LiDAR ───────────────────────────────────────────────
    # All four HSI+LiDAR datasets are available via rs-fusion-datasets-dist
    # (pre-processed to unified .mat format with TRLabel/TSLabel or index files)
    # Release URL: https://github.com/songyz2019/rs-fusion-datasets-dist/releases/tag/v1.0.0

    "Trento": {
        "desc": "Trento HSI+LiDAR (6 classes, 166×600)",
        "method": "direct",
        "files": [
            # tyust-dayu/Trento GitHub raw files (confirmed public, no login)
            {"url": "https://raw.githubusercontent.com/tyust-dayu/Trento/main/Italy_hsi.mat",
             "dest": "Italy_hsi.mat"},
            {"url": "https://raw.githubusercontent.com/tyust-dayu/Trento/main/Italy_lidar.mat",
             "dest": "Italy_lidar.mat"},
            {"url": "https://raw.githubusercontent.com/tyust-dayu/Trento/main/allgrd.mat",
             "dest": "allgrd.mat"},
        ],
        "extract_dir": "Trento",
        "manual_url": "https://github.com/tyust-dayu/Trento",
    },

    "Houston2013": {
        "desc": "Houston 2013 HSI+LiDAR (15 classes, 349×1905)",
        "method": "direct_zip",
        "zip_url": (
            "https://github.com/songyz2019/rs-fusion-datasets-dist/"
            "releases/download/v1.0.0/houston2013-mmr.zip"
        ),
        "zip_name": "houston2013.zip",
        "extract_dir": "Houston2013",
        "expected_files": ["HSI.mat", "LiDAR.mat", "TRLabel.mat", "TSLabel.mat"],
        "manual_url": "https://github.com/songyz2019/rs-fusion-datasets-dist",
    },

    "MUUFL": {
        "desc": "MUUFL Gulfport HSI+LiDAR (11 classes, 325×220)",
        "method": "direct_zip",
        "zip_url": (
            "https://github.com/GatorSense/MUUFLGulfport/"
            "archive/refs/tags/v0.1.zip"
        ),
        "zip_name": "MUUFL.zip",
        "extract_dir": "MUUFL",
        "expected_files": ["muufl_gulfport_campus_1_hsi_220_label.mat"],
        "note": (
            "Main file: muufl_gulfport_campus_1_hsi_220_label.mat. "
            "This is the 220-band version with embedded scene labels."
        ),
        "manual_url": "https://github.com/GatorSense/MUUFLGulfport",
    },

    "Augsburg": {
        "desc": "Augsburg HSI+SAR+DSM (8 classes, 332×485)",
        "method": "direct_zip",
        "zip_url": (
            "https://github.com/songyz2019/rs-fusion-datasets-dist/"
            "releases/download/v1.0.0/augsburg-ouc.zip"
        ),
        "zip_name": "augsburg.zip",
        "extract_dir": "Augsburg",
        "expected_files": ["augsburg_hsi.mat", "augsburg_sar.mat",
                           "augsburg_gt.mat", "augsburg_index.mat"],
        "manual_url": "https://github.com/songyz2019/rs-fusion-datasets-dist",
    },

    "Houston2018": {
        "desc": "Houston 2018 HSI+LiDAR (20 classes, 1202×4768)",
        "method": "direct_zip",
        "zip_url": (
            "https://github.com/songyz2019/rs-fusion-datasets-dist/"
            "releases/download/v1.0.0/houston2018-ouc.zip"
        ),
        "zip_name": "houston2018.zip",
        "extract_dir": "Houston2018",
        "expected_files": ["houston_hsi.mat", "houston_lidar.mat",
                           "houston_gt.mat", "houston_index.mat"],
        "manual_url": "https://github.com/songyz2019/rs-fusion-datasets-dist",
    },

    # ── HSI only ──────────────────────────────────────────────────
    # Available on HuggingFace (danaroth mirror — direct .mat download)

    "IndianPines": {
        "desc": "Indian Pines AVIRIS (16 classes, 145×145)",
        "method": "direct",
        "files": [
            # EHU/GIC direct download (confirmed accessible, ~5.7MB + 1KB)
            {"url": "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
             "dest": "Indian_pines_corrected.mat"},
            {"url": "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
             "dest": "Indian_pines_gt.mat"},
        ],
        "extract_dir": "IndianPines",
        "manual_url": "https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
    },

    "PaviaU": {
        "desc": "Pavia University ROSIS (9 classes, 610×340)",
        "method": "direct",
        "files": [
            # EHU/GIC direct download (~33.2MB + 10KB)
            {"url": "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
             "dest": "PaviaU.mat"},
            {"url": "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
             "dest": "PaviaU_gt.mat"},
        ],
        "extract_dir": "PaviaU",
        "manual_url": "https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
    },

    "Salinas": {
        "desc": "Salinas Valley AVIRIS (16 classes, 512×217)",
        "method": "direct",
        "files": [
            # EHU/GIC direct download (~25.3MB + 4KB)
            {"url": "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
             "dest": "Salinas_corrected.mat"},
            {"url": "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
             "dest": "Salinas_gt.mat"},
        ],
        "extract_dir": "Salinas",
        "manual_url": "https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
    },

    # ── HSI + SAR ─────────────────────────────────────────────────

    "Berlin": {
        "desc": "Berlin EnMAP+Sentinel-1 SAR (8 classes, 476×1723)",
        "method": "direct_zip",
        "zip_url": (
            "https://github.com/songyz2019/rs-fusion-datasets-dist/"
            "releases/download/v1.0.0/berlin-ouc.zip"
        ),
        "zip_name": "berlin.zip",
        "extract_dir": "Berlin",
        "expected_files": ["berlin_hsi.mat", "berlin_sar.mat",
                           "berlin_gt.mat", "berlin_index.mat"],
        "manual_url": "https://github.com/songyz2019/rs-fusion-datasets-dist",
    },

    # ── UAV HSI ───────────────────────────────────────────────────

    "WHU-Hi-LongKou": {
        "desc": "WHU-Hi-LongKou UAV HSI (9 classes, 550×400) — ENVI .bsq format",
        "method": "huggingface",
        "hf_repo": "danaroth/whu_hi",
        # Files are in ENVI .bsq format (no .mat available without registration)
        # pip install spectral   to read .bsq files
        "files": [
            "WHU-Hi-LongKou/WHU-Hi-LongKou.bsq",
            "WHU-Hi-LongKou/WHU-Hi-LongKou.hdr",
            "WHU-Hi-LongKou/WHU-Hi-LongKou_gt.bsq",
            "WHU-Hi-LongKou/WHU-Hi-LongKou_gt.hdr",
        ],
        "dest_names": [
            "WHU-Hi-LongKou.bsq",
            "WHU-Hi-LongKou.hdr",
            "WHU-Hi-LongKou_gt.bsq",
            "WHU-Hi-LongKou_gt.hdr",
        ],
        "extract_dir": "WHU-Hi-LongKou",
        "note": "Requires: pip install spectral (for reading .bsq format)",
        "manual_url": "https://huggingface.co/datasets/danaroth/whu_hi",
    },
}


def _run(cmd: list[str], check: bool = True) -> int:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        print(f"  [WARN] command returned {result.returncode}")
    return result.returncode


def _check_tool(name: str) -> bool:
    return subprocess.run(["which", name], capture_output=True).returncode == 0


def download_gdrive(gdrive_id: str, dest_path: Path):
    """Download a file from Google Drive using gdown."""
    if not _check_tool("gdown"):
        print("  [ERROR] gdown not installed. Run: pip install gdown")
        return False
    _run(["gdown", f"https://drive.google.com/uc?id={gdrive_id}",
          "-O", str(dest_path)])
    return dest_path.exists()


def download_direct(url: str, dest_path: Path):
    """Download a direct URL with wget or curl."""
    if _check_tool("wget"):
        _run(["wget", "-q", "--show-progress", "-O", str(dest_path), url])
    elif _check_tool("curl"):
        _run(["curl", "-L", "-o", str(dest_path), url])
    else:
        print("  [ERROR] Neither wget nor curl found.")
        return False
    return dest_path.exists()


def download_huggingface(repo: str, filename: str, dest_path: Path):
    """Download a file from a HuggingFace dataset repo."""
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
    return download_direct(url, dest_path)


def extract_zip(zip_path: Path, dest_dir: Path, subdir: str | None = None):
    """Extract zip to dest_dir, optionally moving a subfolder up."""
    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    if subdir:
        src = dest_dir / subdir
        if src.exists():
            import shutil
            for item in src.iterdir():
                shutil.move(str(item), str(dest_dir))
            src.rmdir()


def download_dataset(name: str, root: Path) -> bool:
    if name not in DATASET_SOURCES:
        print(f"[ERROR] Unknown dataset '{name}'. Available: {list(DATASET_SOURCES)}")
        return False

    src = DATASET_SOURCES[name]
    ds_dir = root / src.get("extract_dir", name)
    ds_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {name}  —  {src['desc']}")
    print(f"Target:  {ds_dir}")

    method = src["method"]

    if method == "manual":
        print(f"  [MANUAL] This dataset requires manual registration.")
        print(f"  URL: {src['manual_url']}")
        if "note" in src:
            print(f"  Note: {src['note']}")
        return False

    if method == "direct":
        ok = True
        for f in src["files"]:
            dest = ds_dir / f["dest"]
            if dest.exists():
                print(f"  [SKIP] {f['dest']} already exists.")
                continue
            print(f"  Downloading {f['dest']} ...")
            if not download_direct(f["url"], dest):
                print(f"  [WARN] Failed: {f['url']}")
                print(f"  Manual URL: {src.get('manual_url', 'N/A')}")
                ok = False
        return ok

    if method == "direct_zip":
        # Check if target files already exist
        expected = src.get("expected_files", [])
        if expected and all((ds_dir / f).exists() for f in expected):
            print(f"  [SKIP] All expected files already exist.")
            return True
        zip_path = root / src["zip_name"]
        if not zip_path.exists():
            print(f"  Downloading {src['zip_name']} ...")
            if not download_direct(src["zip_url"], zip_path):
                print(f"  [WARN] Download failed. Manual URL: {src.get('manual_url', 'N/A')}")
                return False
        else:
            print(f"  [SKIP] {src['zip_name']} already downloaded.")
        print(f"  Extracting to {ds_dir} ...")
        # Extract flat into ds_dir (files may be in a subdirectory inside zip)
        import shutil
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(ds_dir)
        # Flatten one level if all files landed in a subdirectory
        subdirs = [p for p in ds_dir.iterdir() if p.is_dir()]
        if len(subdirs) == 1 and not any(ds_dir.glob("*.mat")):
            subdir = subdirs[0]
            for item in list(subdir.iterdir()):
                shutil.move(str(item), str(ds_dir))
            subdir.rmdir()
        if "note" in src:
            print(f"  Note: {src['note']}")
        return True

    if method == "huggingface":
        ok = True
        fnames = src["files"]
        dest_names = src.get("dest_names", fnames)
        for fname, dname in zip(fnames, dest_names):
            dest = ds_dir / Path(dname).name
            if dest.exists():
                print(f"  [SKIP] {dest.name} already exists.")
                continue
            print(f"  Downloading {fname} from HuggingFace ...")
            if not download_huggingface(src["hf_repo"], fname, dest):
                print(f"  [WARN] Failed. Manual URL: {src.get('manual_url', 'N/A')}")
                ok = False
        return ok

    print(f"  [ERROR] Unknown method '{method}'")
    return False


def preprocess_dataset(name: str, root: Path):
    """Force preprocessing of a dataset: .mat → PCA → patch → .npz cache."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from benchmark.datasets.registry import get_dataset, DATASETS
    if name not in DATASETS:
        print(f"  [SKIP] {name}: not in registry")
        return
    ds_dir = root / name
    if not ds_dir.exists():
        print(f"  [SKIP] {name}: directory not found ({ds_dir})")
        return
    print(f"\n{'='*60}")
    print(f"Preprocessing: {name}")
    try:
        ds = get_dataset(name, root=ds_dir)
        _ = ds.train   # triggers _load_and_cache → saves .npz
        print(f"  [OK] train={len(ds.train)} test={len(ds.test)} samples")
    except Exception as e:
        print(f"  [ERROR] {e}")


def main():
    p = argparse.ArgumentParser(description="RS-CIL Dataset Downloader")
    p.add_argument("--dataset", default="all",
                   help="Dataset name or 'all'. Use --list to see options.")
    p.add_argument("--root", default="~/datasets/rs_cil",
                   help="Root directory for dataset storage (default: ~/datasets/rs_cil).")
    p.add_argument("--list", action="store_true",
                   help="List available datasets and exit.")
    p.add_argument("--preprocess", action="store_true",
                   help="After downloading, convert .mat → .npz cache (PCA + patches).")
    args = p.parse_args()

    if args.list:
        print("Available datasets:")
        for name, src in DATASET_SOURCES.items():
            status = "[MANUAL]" if src["method"] == "manual" else "[AUTO]  "
            print(f"  {status} {name:20s} — {src['desc']}")
        return

    root = Path(args.root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    targets = list(DATASET_SOURCES.keys()) if args.dataset == "all" else [args.dataset]

    results = {}
    for name in targets:
        results[name] = download_dataset(name, root)

    print(f"\n{'='*60}")
    print("Summary:")
    for name, ok in results.items():
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {name}")

    if args.preprocess:
        print(f"\n{'='*60}")
        print("Preprocessing (mat → npz) ...")
        for name in targets:
            if results.get(name):
                preprocess_dataset(name, root)


if __name__ == "__main__":
    main()
