"""Test ExemplarMemory with all 8 strategies."""
import pytest
import torch
import torch.nn as nn

from benchmark.utils.exemplars import ExemplarMemory, list_strategies


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(37, 16)

    def forward(self, xh, xl):
        x = torch.cat([xh, xl], dim=1).flatten(1)
        return self.fc(x[:, :37])


@pytest.fixture
def dummy():
    torch.manual_seed(0)
    model = DummyModel()
    hsi = torch.randn(60, 36, 1, 1)
    lidar = torch.randn(60, 1, 1, 1)
    labels = torch.arange(3).repeat_interleave(20)
    return model, hsi, lidar, labels


def test_list_strategies():
    s = list_strategies()
    assert len(s) == 8
    assert "herding" in s
    assert "random" in s
    assert "entropy" in s
    assert "kmeans" in s


@pytest.mark.parametrize("strategy", list(list_strategies().keys()))
def test_strategy_smoke(strategy, dummy):
    model, hsi, lidar, labels = dummy
    mem = ExemplarMemory(budget=15, strategy=strategy)
    mem.update(model, hsi, lidar, labels, torch.device("cpu"),
               new_class_ids=[0, 1, 2])
    assert mem.n_classes == 3
    assert mem.n_exemplars <= 15
    assert mem.n_exemplars > 0

    # get_data
    h, l, y = mem.get_data()
    assert h.shape[0] == mem.n_exemplars
    assert set(y.tolist()) == {0, 1, 2}

    # get_loader
    loader = mem.get_loader(batch_size=4)
    batch = next(iter(loader))
    assert len(batch) == 3


def test_budget_reduction(dummy):
    model, hsi, lidar, labels = dummy
    mem = ExemplarMemory(budget=12, strategy="random")
    # Add 3 classes → 4 per class
    mem.update(model, hsi, lidar, labels, torch.device("cpu"),
               new_class_ids=[0, 1, 2])
    assert mem.k_per_class(3) == 4

    # Simulate adding 3 more classes → should reduce to 2 per class
    hsi2 = torch.randn(60, 36, 1, 1)
    lidar2 = torch.randn(60, 1, 1, 1)
    labels2 = torch.tensor([3]*20 + [4]*20 + [5]*20)
    mem.update(model, hsi2, lidar2, labels2, torch.device("cpu"),
               new_class_ids=[3, 4, 5])
    assert mem.n_classes == 6
    assert mem.k_per_class(6) == 2


def test_state_dict_roundtrip(dummy):
    model, hsi, lidar, labels = dummy
    mem = ExemplarMemory(budget=15, strategy="random")
    mem.update(model, hsi, lidar, labels, torch.device("cpu"),
               new_class_ids=[0, 1, 2])
    state = mem.state_dict()

    mem2 = ExemplarMemory(budget=100, strategy="herding")
    mem2.load_state_dict(state)
    assert mem2.budget == 15
    assert mem2.strategy == "random"
    assert mem2.n_classes == 3


def test_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        ExemplarMemory(budget=10, strategy="nonexistent")
