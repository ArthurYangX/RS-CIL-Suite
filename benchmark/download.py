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

    "Trento": {
        "desc": "Trento HSI+LiDAR (6 classes, 166×600)",
        "method": "gdrive",
        "files": {
            # MDL-RS dataset package (Hong et al., IEEE TGRS 2020)
            "gdrive_id": "1A0MnMrFEPgSMqS2MN3sMDDyiD_SHJrx8",  # MDL-RS zip
            "zip_name":  "MDL-RS.zip",
            "extract_dir": "Trento",
            "note": "Contains Trento, Houston2013, MUUFL in one zip.",
        },
        "manual_url": "https://github.com/danfenghong/IEEE_TGRS_MDL-RS",
    },

    "Houston2013": {
        "desc": "Houston 2013 HSI+LiDAR (15 classes, 349×1905)",
        "method": "shared",  # included in MDL-RS zip above
        "extract_dir": "Houston2013",
        "manual_url": "https://hyperspectral.ee.uh.edu/?page_id=459",
    },

    "MUUFL": {
        "desc": "MUUFL Gulfport HSI+LiDAR (11 classes, 325×220)",
        "method": "gdrive",
        "files": {
            "gdrive_id": "1A0MnMrFEPgSMqS2MN3sMDDyiD_SHJrx8",
            "zip_name":  "MDL-RS.zip",
            "extract_dir": "MUUFL",
        },
        "manual_url": "https://github.com/danfenghong/IEEE_TGRS_MDL-RS",
    },

    "Augsburg": {
        "desc": "Augsburg HSI+LiDAR (7 classes, 332×485)",
        "method": "gdrive",
        "files": {
            # Hong et al., IEEE TGRS 2021 (MFT paper supplementary)
            "gdrive_id": "1EHSoNeS-U4e7K2MjlrfLEG3e1C0h6Ya3",
            "zip_name":  "Augsburg.zip",
            "extract_dir": "Augsburg",
        },
        "manual_url": "https://github.com/danfenghong/IEEE_TGRS_MFT",
    },

    "Houston2018": {
        "desc": "Houston 2018 HSI+LiDAR (20 classes, 601×2384)",
        "method": "manual",
        "manual_url": "https://hyperspectral.ee.uh.edu/?page_id=1075",
        "note": (
            "Requires IEEE GRSS Data Fusion Contest registration. "
            "After downloading, place Phase2_*.mat files in $ROOT/Houston2018/"
        ),
    },

    # ── HSI only ──────────────────────────────────────────────────

    "IndianPines": {
        "desc": "Indian Pines AVIRIS (16 classes, 145×145)",
        "method": "direct",
        "files": [
            {
                "url": "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
                "dest": "Indian_pines_corrected.mat",
            },
            {
                "url": "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
                "dest": "Indian_pines_gt.mat",
            },
        ],
        "extract_dir": "IndianPines",
        "manual_url": "http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
    },

    "PaviaU": {
        "desc": "Pavia University ROSIS (9 classes, 610×340)",
        "method": "direct",
        "files": [
            {
                "url": "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
                "dest": "PaviaU.mat",
            },
            {
                "url": "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
                "dest": "PaviaU_gt.mat",
            },
        ],
        "extract_dir": "PaviaU",
        "manual_url": "http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
    },

    "Salinas": {
        "desc": "Salinas Valley AVIRIS (16 classes, 512×217)",
        "method": "direct",
        "files": [
            {
                "url": "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
                "dest": "Salinas_corrected.mat",
            },
            {
                "url": "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
                "dest": "Salinas_gt.mat",
            },
        ],
        "extract_dir": "Salinas",
        "manual_url": "http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
    },

    # ── HSI + SAR ─────────────────────────────────────────────────

    "Berlin": {
        "desc": "Berlin EnMAP+Sentinel-1 (8 classes, 1723×476)",
        "method": "gdrive",
        "files": {
            # Hong et al., ISPRS 2021 (S2FL paper)
            "gdrive_id": "1EHRkknDVRBCiEFC4D0P4G4I4V_TbBBQw",
            "zip_name":  "Berlin.zip",
            "extract_dir": "Berlin",
        },
        "manual_url": "https://github.com/danfenghong/ISPRS_S2FL",
    },

    # ── UAV HSI ───────────────────────────────────────────────────

    "WHU-Hi-LongKou": {
        "desc": "WHU-Hi-LongKou UAV HSI (9 classes, 550×400)",
        "method": "huggingface",
        "hf_repo": "WangHongbo/WHU-Hi",
        "files": [
            "WHU_Hi_LongKou.mat",
            "WHU_Hi_LongKou_gt.mat",
        ],
        "extract_dir": "WHU-Hi-LongKou",
        "manual_url": "https://huggingface.co/datasets/WangHongbo/WHU-Hi",
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

    if method == "shared":
        print(f"  [INFO] Included in another dataset's download (e.g., Trento/MDL-RS).")
        print(f"  Manual URL: {src['manual_url']}")
        return True

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
                print(f"         Manual URL: {src.get('manual_url', 'N/A')}")
                ok = False
        return ok

    if method == "gdrive":
        files = src["files"]
        zip_path = root / files["zip_name"]
        if not zip_path.exists():
            print(f"  Downloading {files['zip_name']} from Google Drive ...")
            download_gdrive(files["gdrive_id"], zip_path)
        else:
            print(f"  [SKIP] {files['zip_name']} already downloaded.")
        if zip_path.exists():
            extract_zip(zip_path, root, subdir=files.get("extract_dir"))
            return True
        else:
            print(f"  [WARN] Download failed. Manual URL: {src.get('manual_url', 'N/A')}")
            return False

    if method == "huggingface":
        ok = True
        for fname in src["files"]:
            dest = ds_dir / fname
            if dest.exists():
                print(f"  [SKIP] {fname} already exists.")
                continue
            print(f"  Downloading {fname} from HuggingFace ...")
            if not download_huggingface(src["hf_repo"], fname, dest):
                print(f"  [WARN] Failed. Manual URL: {src.get('manual_url', 'N/A')}")
                ok = False
        return ok

    print(f"  [ERROR] Unknown method '{method}'")
    return False


def main():
    p = argparse.ArgumentParser(description="RS-CIL Dataset Downloader")
    p.add_argument("--dataset", default="all",
                   help="Dataset name or 'all'. Use --list to see options.")
    p.add_argument("--root", default="~/datasets/rs_cil",
                   help="Root directory for dataset storage (default: ~/datasets/rs_cil).")
    p.add_argument("--list", action="store_true",
                   help="List available datasets and exit.")
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


if __name__ == "__main__":
    main()
