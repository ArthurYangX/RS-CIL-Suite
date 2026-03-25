"""Shared preprocessing utilities (PCA, normalisation, patch extraction).

Ported from the S2CM codebase and cleaned up for reuse across all datasets.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA


# ── PCA ───────────────────────────────────────────────────────────

def apply_pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """Apply PCA whitening to (H, W, C) hyperspectral cube.
    Returns (H, W, n_components).
    """
    H, W, C = X.shape
    n_components = min(n_components, C, H * W)
    flat = X.reshape(-1, C)
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    reduced = pca.fit_transform(flat)
    return reduced.reshape(H, W, n_components)


# ── Per-band min-max normalisation ───────────────────────────────

def normalize(data: np.ndarray) -> np.ndarray:
    """Min-max normalise each channel independently. (H, W, C) → (H, W, C)."""
    out = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[2]):
        lo, hi = data[:, :, i].min(), data[:, :, i].max()
        out[:, :, i] = 0.0 if hi == lo else (data[:, :, i] - lo) / (hi - lo)
    return out


# ── Mirror padding ────────────────────────────────────────────────

def mirror_pad(img: np.ndarray, pad: int) -> np.ndarray:
    """Reflect-pad (H, W, C) image by `pad` pixels on each side."""
    H, W, C = img.shape
    out = np.zeros((H + 2*pad, W + 2*pad, C), dtype=img.dtype)
    out[pad:pad+H, pad:pad+W] = img
    # left / right
    for i in range(pad):
        out[pad:pad+H, i]           = img[:, pad-i-1]
        out[pad:pad+H, W+pad+i]     = img[:, W-1-i]
    # top / bottom (after left/right are filled)
    for i in range(pad):
        out[i]                      = out[2*pad-i-1]
        out[H+pad+i]                = out[H+pad-1-i]
    return out


# ── Sample coordinate helpers ─────────────────────────────────────

def get_sample_coords(label_map: np.ndarray, num_classes: int) -> tuple[np.ndarray, list[int]]:
    """Return (N,2) coordinates and per-class counts for all labelled pixels."""
    coords_per_class = [np.argwhere(label_map == (c + 1)) for c in range(num_classes)]
    counts = [len(c) for c in coords_per_class]
    all_coords = np.concatenate(coords_per_class, axis=0).astype(np.int32)
    return all_coords, counts


# ── Patch extraction ──────────────────────────────────────────────

def extract_patches(
    padded_hsi: np.ndarray,   # (H+2p, W+2p, C_hsi)
    padded_lidar: np.ndarray, # (H+2p, W+2p, C_lid)
    coords: np.ndarray,       # (N, 2)
    patch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (patch, patch) neighbourhoods around each coordinate.

    Returns:
        hsi_patches   : (N, C_hsi, patch, patch)  float32
        lidar_patches : (N, C_lid, patch, patch)  float32
    """
    N = len(coords)
    C_hsi  = padded_hsi.shape[2]
    C_lid  = padded_lidar.shape[2]

    hsi_out   = np.zeros((N, C_hsi, patch, patch), dtype=np.float32)
    lidar_out = np.zeros((N, C_lid, patch, patch), dtype=np.float32)

    for i, (r, c) in enumerate(coords):
        h_patch = padded_hsi  [r:r+patch, c:c+patch, :]   # (P, P, C)
        l_patch = padded_lidar[r:r+patch, c:c+patch, :]
        hsi_out[i]   = h_patch.transpose(2, 0, 1)
        lidar_out[i] = l_patch.transpose(2, 0, 1)

    return hsi_out, lidar_out


def build_labels(counts_per_class: list[int]) -> np.ndarray:
    """Build 0-indexed label array from per-class sample counts."""
    labels = []
    for c, n in enumerate(counts_per_class):
        labels.extend([c] * n)
    return np.array(labels, dtype=np.int64)


# ── Full pipeline ─────────────────────────────────────────────────

def preprocess_hsi_lidar(
    hsi: np.ndarray,          # (H, W, C_raw)
    lidar: np.ndarray,        # (H, W, C_lid)
    tr_map: np.ndarray,       # (H, W) int, values 1..K for training pixels, 0=bg
    te_map: np.ndarray,       # (H, W) int, values 1..K for test pixels, 0=bg
    num_classes: int,
    patch: int = 7,
    pca_components: int = 36,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Full preprocessing pipeline used by all HSI+LiDAR datasets.

    Returns:
        x_train_hsi, x_train_lidar, y_train,
        x_test_hsi,  x_test_lidar,  y_test
    All arrays are numpy, labels are 0-indexed.
    """
    # Ensure LiDAR is 3-D
    if lidar.ndim == 2:
        lidar = lidar[:, :, np.newaxis]

    # PCA on HSI
    hsi_pca = apply_pca(hsi, pca_components)

    # Normalise
    hsi_norm   = normalize(hsi_pca)
    lidar_norm = normalize(lidar.astype(np.float32))

    # Mirror pad
    pad = patch // 2
    hsi_pad   = mirror_pad(hsi_norm,   pad)
    lidar_pad = mirror_pad(lidar_norm, pad)

    # Training samples
    tr_coords, tr_counts = get_sample_coords(tr_map, num_classes)
    x_tr_hsi, x_tr_lid  = extract_patches(hsi_pad, lidar_pad, tr_coords, patch)
    y_tr = build_labels(tr_counts)

    # Test samples
    te_coords, te_counts = get_sample_coords(te_map, num_classes)
    x_te_hsi, x_te_lid  = extract_patches(hsi_pad, lidar_pad, te_coords, patch)
    y_te = build_labels(te_counts)

    return x_tr_hsi, x_tr_lid, y_tr, x_te_hsi, x_te_lid, y_te
