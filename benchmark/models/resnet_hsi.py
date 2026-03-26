"""ResNet backbones adapted for HSI+LiDAR patch classification.

Provides ResNet-18 and ResNet-34 variants that accept multi-channel
HSI+LiDAR input and produce d-dimensional L2-normalised embeddings.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_backbone


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.gelu(out)


class ResNetHSI(nn.Module):
    """ResNet backbone for HSI+LiDAR patches.

    Input:  (B, hsi_ch + lidar_ch, H, W)
    Output: (B, d) L2-normalised embeddings

    Adapted for small patches (7x7): uses stride=1 in layer1/layer2
    to avoid aggressive downsampling.
    """

    def __init__(self, hsi_ch: int, lidar_ch: int, d: int = 128,
                 layers: list[int] = None):
        super().__init__()
        if layers is None:
            layers = [2, 2, 2, 2]  # ResNet-18

        in_ch = hsi_ch + lidar_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.layer1 = self._make_layer(64, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, d)

        self._current_in = None  # used by _make_layer

    def _make_layer(self, in_ch: int, out_ch: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        c = in_ch
        for s in strides:
            layers.append(_BasicBlock(c, out_ch, s))
            c = out_ch
        return nn.Sequential(*layers)

    def forward(self, xh, xl, return_features=False):
        x = torch.cat([xh, xl], dim=1)
        h1 = self.conv1(x)
        h2 = self.layer1(h1)
        h3 = self.layer2(h2)
        h4 = self.layer3(h3)
        h5 = self.layer4(h4)
        pooled = self.pool(h5).flatten(1)
        feat = self.fc(pooled)
        out = F.normalize(feat, dim=1)

        if not return_features:
            return out

        return out, {
            "conv1": h1, "layer1": h2, "layer2": h3,
            "layer3": h4, "layer4": h5,
            "pooled": pooled, "pre_norm": feat,
        }


@register_backbone("resnet18_hsi")
class ResNet18HSI(ResNetHSI):
    """ResNet-18 for HSI+LiDAR (11.2M params with d=128)."""
    def __init__(self, hsi_ch: int, lidar_ch: int, d: int = 128, **kwargs):
        super().__init__(hsi_ch, lidar_ch, d, layers=[2, 2, 2, 2])


@register_backbone("resnet34_hsi")
class ResNet34HSI(ResNetHSI):
    """ResNet-34 for HSI+LiDAR (21.3M params with d=128)."""
    def __init__(self, hsi_ch: int, lidar_ch: int, d: int = 128, **kwargs):
        super().__init__(hsi_ch, lidar_ch, d, layers=[3, 4, 6, 3])
