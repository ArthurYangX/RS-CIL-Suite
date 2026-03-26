"""Test protocol definitions and YAML loading."""
import tempfile
import pytest
import yaml
from benchmark.protocols.cil import (
    PROTOCOLS, get_protocol, build_within_scene, build_cross_scene,
    load_protocol_yaml, NUM_CLASSES,
)


def test_builtin_protocols_exist():
    assert "B1" in PROTOCOLS
    assert "A_Trento" in PROTOCOLS
    assert len(PROTOCOLS) >= 15


def test_b1_structure():
    p = PROTOCOLS["B1"]
    assert p.num_tasks == 9
    assert p.total_classes == 32  # 6+15+11
    assert len(p.dataset_order) == 3


def test_within_scene():
    p = build_within_scene("IndianPines", [4, 4, 4, 4], 16)
    assert p.num_tasks == 4
    assert p.total_classes == 16
    # Classes should be sequential
    all_ids = [c for t in p.tasks for c in t.class_ids]
    assert all_ids == list(range(16))


def test_cross_scene():
    p = build_cross_scene(
        ["Trento", "Houston2013"],
        {"Trento": [3, 3], "Houston2013": [5, 5, 5]},
        NUM_CLASSES,
    )
    assert p.num_tasks == 5
    assert p.total_classes == 21


def test_get_protocol_by_name():
    p = get_protocol("B1")
    assert p.name is not None


def test_get_protocol_by_yaml():
    cfg = {
        "name": "TestYAML",
        "type": "within_scene",
        "dataset_order": ["PaviaU"],
        "class_splits": {"PaviaU": [3, 3, 3]},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        path = f.name
    p = get_protocol(path)
    assert p.name == "TestYAML"
    assert p.num_tasks == 3


def test_shuffle_classes():
    cfg = {
        "name": "ShuffleTest",
        "type": "within_scene",
        "dataset_order": ["IndianPines"],
        "class_splits": {"IndianPines": [4, 4, 4, 4]},
        "shuffle_classes": True,
        "class_order_seed": 123,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        path = f.name
    p = load_protocol_yaml(path)
    first_ids = p.tasks[0].class_ids
    # With shuffle, first task should NOT be [0,1,2,3]
    assert first_ids != [0, 1, 2, 3], f"shuffle did not work: {first_ids}"


def test_train_ratio_metadata():
    cfg = {
        "name": "RatioTest",
        "type": "within_scene",
        "dataset_order": ["PaviaU"],
        "class_splits": {"PaviaU": [3, 3, 3]},
        "train_ratio": 0.2,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        path = f.name
    p = load_protocol_yaml(path)
    assert p.train_ratio == 0.2


def test_get_protocol_unknown():
    with pytest.raises(ValueError, match="Unknown protocol"):
        get_protocol("NONEXISTENT_PROTOCOL_XYZ")
