"""SimpleEncoder — lightweight CNN backbone for RS-CIL benchmark.

Supports optional intermediate feature extraction for knowledge distillation
methods (PODNet, AFC, FOSTER, etc.).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEncoder(nn.Module):
    """Lightweight CNN that encodes (C, 7, 7) patches → d-dim feature vector.

    Input:
        xh: (B, hsi_ch, H, W)   — HSI patch (post-PCA, typically 36 channels)
        xl: (B, lidar_ch, H, W) — LiDAR / SAR patch (1-4 channels)

    Output:
        If ``return_features=False`` (default):
            (B, d) L2-normalised feature vector.
        If ``return_features=True``:
            tuple of (features, intermediates) where intermediates is a dict
            mapping layer names to activation tensors.
    """

    def __init__(self, hsi_ch: int, lidar_ch: int, d: int = 128):
        super().__init__()
        in_ch = hsi_ch + lidar_ch

        # Named sub-modules for intermediate feature access
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, d)

        # Also keep nn.Sequential for backward-compat (some methods access self.net)
        self.net = nn.Sequential(
            self.conv1[0], self.conv1[1], self.conv1[2],
            self.conv2[0], self.conv2[1], self.conv2[2],
            self.pool, self.flatten, self.fc,
        )

    def forward(self, xh, xl, return_features=False):
        x = torch.cat([xh, xl], dim=1)

        if not return_features:
            return F.normalize(self.net(x), dim=1)

        # Extract intermediates for KD methods
        intermediates = {}
        h1 = self.conv1(x)
        intermediates["conv1"] = h1                     # (B, 64, H, W)
        h2 = self.conv2(h1)
        intermediates["conv2"] = h2                     # (B, 128, H, W)
        pooled = self.pool(h2)
        intermediates["pooled"] = pooled                # (B, 128, 1, 1)
        flat = self.flatten(pooled)
        feat = self.fc(flat)
        intermediates["pre_norm"] = feat                # (B, d) before L2 norm
        out = F.normalize(feat, dim=1)

        return out, intermediates
