"""Backbone model registry for RS-CIL benchmark.

Usage:
    from benchmark.models import build_backbone, list_backbones

    model = build_backbone("simple_encoder", hsi_ch=36, lidar_ch=1, d=128)
"""
from __future__ import annotations

from .simple_encoder import SimpleEncoder

_BACKBONE_REGISTRY: dict[str, type] = {
    "simple_encoder": SimpleEncoder,
}


def register_backbone(name: str):
    """Class decorator to register a new backbone.

    Example::

        @register_backbone("resnet18")
        class ResNet18Backbone(nn.Module):
            ...
    """
    def wrapper(cls):
        _BACKBONE_REGISTRY[name] = cls
        return cls
    return wrapper


def build_backbone(name: str, **kwargs) -> "torch.nn.Module":
    """Instantiate a backbone by name.

    Args:
        name:   Key in the backbone registry (e.g. "simple_encoder").
        **kwargs: Forwarded to the backbone constructor
                  (typically hsi_ch, lidar_ch, d).
    """
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. Available: {sorted(_BACKBONE_REGISTRY)}"
        )
    return _BACKBONE_REGISTRY[name](**kwargs)


def list_backbones() -> list[str]:
    """Return sorted list of registered backbone names."""
    return sorted(_BACKBONE_REGISTRY.keys())
