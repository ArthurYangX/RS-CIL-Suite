"""Test YAML config system."""
import pytest
from benchmark.config import load_config, flatten_config


def test_load_defaults():
    cfg = load_config("nonexistent_method_xyz")
    assert "model" in cfg
    assert cfg["model"]["d"] == 128


def test_load_method_config():
    cfg = load_config("icarl")
    flat = flatten_config(cfg)
    assert flat["memory_size"] == 2000
    assert flat["T"] == 2.0
    assert flat["lr"] == 1e-3


def test_cli_override():
    cfg = load_config("ewc", cli_overrides=["method.ewc_lambda=999", "training.lr=0.01"])
    flat = flatten_config(cfg)
    assert flat["ewc_lambda"] == 999
    assert flat["lr"] == 0.01


def test_flatten_config():
    cfg = {
        "model": {"backbone": "simple_encoder", "d": 64},
        "training": {"epochs": 100, "lr": 0.005},
        "method": {"alpha": 0.5},
    }
    flat = flatten_config(cfg)
    assert flat["d"] == 64
    assert flat["epochs"] == 100
    assert flat["alpha"] == 0.5
    assert flat["_backbone"] == "simple_encoder"
