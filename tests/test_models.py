"""Test backbone models."""
import torch
from benchmark.models import SimpleEncoder, build_backbone


def test_simple_encoder_forward():
    enc = SimpleEncoder(hsi_ch=36, lidar_ch=1, d=128)
    xh = torch.randn(4, 36, 7, 7)
    xl = torch.randn(4, 1, 7, 7)
    out = enc(xh, xl)
    assert out.shape == (4, 128)
    # L2 normalized
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_simple_encoder_features():
    enc = SimpleEncoder(hsi_ch=36, lidar_ch=1, d=128)
    xh = torch.randn(2, 36, 7, 7)
    xl = torch.randn(2, 1, 7, 7)
    out, feats = enc(xh, xl, return_features=True)
    assert out.shape == (2, 128)
    assert "conv1" in feats
    assert "conv2" in feats
    assert "pooled" in feats
    assert "pre_norm" in feats
    assert feats["conv1"].shape == (2, 64, 7, 7)
    assert feats["conv2"].shape == (2, 128, 7, 7)


def test_build_backbone():
    enc = build_backbone("simple_encoder", hsi_ch=10, lidar_ch=2, d=64)
    xh = torch.randn(2, 10, 7, 7)
    xl = torch.randn(2, 2, 7, 7)
    out = enc(xh, xl)
    assert out.shape == (2, 64)


def test_build_backbone_unknown():
    import pytest
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_backbone("nonexistent_backbone")
