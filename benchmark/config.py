"""YAML-based configuration system for RS-CIL benchmark.

Layered config resolution:
    defaults.yaml  ←  {method}.yaml  ←  --config file  ←  --opts CLI overrides

Usage:
    from benchmark.config import load_config

    cfg = load_config("icarl")
    # → {'model': {'backbone': 'simple_encoder', 'd': 128},
    #    'training': {'epochs': 50, 'lr': 0.001, ...},
    #    'method': {'memory_size': 2000, 'T': 2.0}}

    cfg = load_config("icarl", cli_overrides=["training.lr=0.0005", "method.T=3.0"])
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_CONFIGS_DIR = Path(__file__).parent / "configs"


def load_config(
    method_name: str,
    config_path: str | Path | None = None,
    cli_overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load and merge configuration for a method.

    Args:
        method_name:   Method key (e.g. "icarl"). Used to find
                       ``configs/{method_name}.yaml``.
        config_path:   Explicit YAML path (overrides auto-discovery).
        cli_overrides: List of "dotted.key=value" strings from ``--opts``.

    Returns:
        Merged config dict with sections: ``model``, ``training``, ``method``.
    """
    cfg: dict[str, Any] = {}

    # 1. Defaults
    defaults_path = _CONFIGS_DIR / "defaults.yaml"
    if defaults_path.exists():
        cfg = _load_yaml(defaults_path)

    # 2. Method-specific (auto-discovered or explicit)
    if config_path:
        method_cfg = _load_yaml(Path(config_path))
    else:
        method_path = _CONFIGS_DIR / f"{method_name}.yaml"
        method_cfg = _load_yaml(method_path) if method_path.exists() else {}

    cfg = _deep_merge(cfg, method_cfg)

    # 3. CLI overrides
    if cli_overrides:
        cfg = _apply_overrides(cfg, cli_overrides)

    return cfg


def flatten_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested config into method-constructor-friendly kwargs.

    Merges ``training.*`` and ``method.*`` into a flat dict,
    plus ``d`` from ``model.d``.
    """
    flat: dict[str, Any] = {}
    flat.update(cfg.get("training", {}))
    flat.update(cfg.get("method", {}))
    if "model" in cfg:
        if "d" in cfg["model"]:
            flat["d"] = cfg["model"]["d"]
        if "backbone" in cfg["model"]:
            flat["_backbone"] = cfg["model"]["backbone"]
    return flat


# ── Internal helpers ──────────────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for config loading. Install: pip install pyyaml"
        )
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins)."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dotted-key=value overrides (e.g. ``training.lr=0.001``)."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}': expected key=value")
        key, val_str = item.split("=", 1)
        val = _parse_value(val_str)
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg


def _parse_value(s: str) -> Any:
    """Best-effort cast: int → float → bool → str."""
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() == "none":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s
