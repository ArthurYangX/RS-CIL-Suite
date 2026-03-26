"""Test evaluation metrics and BWT/FWT conventions."""
import numpy as np
import pytest
from benchmark.eval.metrics import (
    overall_accuracy, average_accuracy, cohen_kappa,
    evaluate, BenchmarkResult, TaskResult,
)


def test_overall_accuracy():
    preds = np.array([0, 1, 2, 0, 1])
    targets = np.array([0, 1, 2, 1, 1])
    assert overall_accuracy(preds, targets) == pytest.approx(0.8)


def test_average_accuracy():
    preds = np.array([0, 0, 1, 1, 2, 2])
    targets = np.array([0, 0, 1, 1, 2, 0])
    aa = average_accuracy(preds, targets, [0, 1, 2])
    # class 0: 2/3, class 1: 2/2, class 2: 1/1 → mean=(2/3+1+1)/3
    assert aa == pytest.approx((2/3 + 1.0 + 1.0) / 3, abs=1e-6)


def test_cohen_kappa_perfect():
    preds = np.array([0, 1, 2, 0, 1])
    assert cohen_kappa(preds, preds, [0, 1, 2]) == pytest.approx(1.0, abs=1e-6)


def test_evaluate():
    preds = np.array([0, 0, 1, 1, 2, 2])
    targets = np.array([0, 0, 1, 1, 2, 2])
    result = evaluate(preds, targets, [0, 1, 2],
                      {0: "A", 1: "A", 2: "B"}, ["A", "B"])
    assert result.oa == 1.0
    assert "A" in result.per_dataset
    assert "B" in result.per_dataset


def test_bwt_negative_for_forgetting():
    """BWT must be negative when there is forgetting (standard CL convention)."""
    br = BenchmarkResult(protocol_name="test", method_name="test")
    br.add(TaskResult(task_id=0, per_dataset={"A": 0.9}, avg_aa=0.9, oa=0.9, kappa=0.88))
    br.add(TaskResult(task_id=1, per_dataset={"A": 0.7}, avg_aa=0.7, oa=0.7, kappa=0.68))
    br.compute_cl_metrics()
    assert br.bwt < 0, f"BWT should be negative for forgetting, got {br.bwt}"
    assert br.forgetting["A"] == pytest.approx(0.2)  # peak(0.9) - final(0.7)


def test_bwt_zero_no_forgetting():
    """BWT should be 0 if there is no forgetting."""
    br = BenchmarkResult(protocol_name="test", method_name="test")
    br.add(TaskResult(task_id=0, per_dataset={"A": 0.8}, avg_aa=0.8, oa=0.8, kappa=0.75))
    br.add(TaskResult(task_id=1, per_dataset={"A": 0.8}, avg_aa=0.8, oa=0.8, kappa=0.75))
    br.compute_cl_metrics()
    assert br.bwt == 0.0


def test_fwt_is_first_task_aa():
    """FWT should be the mean first-task AA across datasets."""
    br = BenchmarkResult(protocol_name="test", method_name="test")
    br.add(TaskResult(task_id=0, per_dataset={"A": 0.85}, avg_aa=0.85, oa=0.85, kappa=0.8))
    br.add(TaskResult(task_id=1, per_dataset={"A": 0.8, "B": 0.7}, avg_aa=0.75, oa=0.75, kappa=0.7))
    br.compute_cl_metrics()
    # FWT = mean(first_aa_A=0.85, first_aa_B=0.7) = 0.775
    assert br.fwt == pytest.approx(0.775)
