"""Vision Transformer backbones for HSI+LiDAR patch classification.

Provides ViT-Tiny and ViT-Small adapted for small 7x7 HSI patches.
Each patch pixel is treated as a token (49 tokens for 7x7 patches).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_backbone


class _MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MultiHeadAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTHSI(nn.Module):
    """Vision Transformer for HSI+LiDAR patches.

    Treats each pixel in a 7x7 patch as a token (49 tokens).
    Each token has (hsi_ch + lidar_ch) channels, projected to `embed_dim`.

    Input:  xh (B, hsi_ch, H, W), xl (B, lidar_ch, H, W)
    Output: (B, d) L2-normalised embeddings
    """

    def __init__(self, hsi_ch: int, lidar_ch: int, d: int = 128,
                 embed_dim: int = 192, depth: int = 4, n_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.1, **kwargs):
        super().__init__()
        in_ch = hsi_ch + lidar_ch
        self.embed_dim = embed_dim

        # Linear projection: each pixel → embed_dim
        self.patch_proj = nn.Linear(in_ch, embed_dim)

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 49 pixel tokens + 1 CLS token = 50 positions
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.Sequential(*[
            _TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, d)

    def forward(self, xh, xl, return_features=False):
        x = torch.cat([xh, xl], dim=1)       # (B, C, H, W)
        B, C, H, W = x.shape
        # Reshape pixels to tokens: (B, C, H, W) → (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)  # (B, 49, C)
        tokens = self.patch_proj(tokens)        # (B, 49, embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 50, embed_dim)

        # Add positional embedding (handle variable patch sizes)
        if tokens.shape[1] <= self.pos_embed.shape[1]:
            tokens = tokens + self.pos_embed[:, :tokens.shape[1]]
        else:
            tokens = tokens + F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=tokens.shape[1], mode='linear', align_corners=False
            ).transpose(1, 2)

        h = self.blocks(tokens)
        h = self.norm(h)
        cls_out = h[:, 0]  # CLS token output
        feat = self.fc(cls_out)
        out = F.normalize(feat, dim=1)

        if not return_features:
            return out

        return out, {
            "cls_token": cls_out,
            "all_tokens": h,
            "pre_norm": feat,
        }


@register_backbone("vit_tiny_hsi")
class ViTTinyHSI(ViTHSI):
    """ViT-Tiny for HSI (embed=192, depth=4, heads=3, ~1.5M params)."""
    def __init__(self, hsi_ch: int, lidar_ch: int, d: int = 128, **kwargs):
        super().__init__(hsi_ch, lidar_ch, d,
                         embed_dim=192, depth=4, n_heads=3,
                         mlp_ratio=4.0, dropout=0.1)


@register_backbone("vit_small_hsi")
class ViTSmallHSI(ViTHSI):
    """ViT-Small for HSI (embed=384, depth=6, heads=6, ~8.5M params)."""
    def __init__(self, hsi_ch: int, lidar_ch: int, d: int = 128, **kwargs):
        super().__init__(hsi_ch, lidar_ch, d,
                         embed_dim=384, depth=6, n_heads=6,
                         mlp_ratio=4.0, dropout=0.1)
