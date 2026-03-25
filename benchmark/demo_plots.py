"""Generate demo plots with synthetic data to showcase all visualization functions."""
from __future__ import annotations
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    out_dir = Path("demo_figs")
    out_dir.mkdir(exist_ok=True)

    # ── 1. Classification map ────────────────────────────────────
    from benchmark.eval.plots import plot_classification_map
    from benchmark.eval.colors import get_colormap

    np.random.seed(42)
    H, W, nc = 100, 120, 6
    gt_map = np.zeros((H, W), dtype=np.int32)
    # Create some class regions
    for c in range(1, nc + 1):
        r0, c0 = np.random.randint(5, H-25), np.random.randint(5, W-25)
        gt_map[r0:r0+20, c0:c0+25] = c

    # Simulate test pixels and predictions (90% accuracy)
    test_mask = gt_map > 0
    targets = gt_map[test_mask]
    preds = targets.copy()
    n_err = int(len(preds) * 0.1)
    err_idx = np.random.choice(len(preds), n_err, replace=False)
    preds[err_idx] = np.random.randint(1, nc + 1, n_err)

    class_names = ["Apple trees", "Buildings", "Ground", "Woods", "Vineyard", "Roads"]
    fig = plot_classification_map(
        gt_map, preds, targets, class_names,
        dataset_name="Trento",
        title="Classification Map — Trento (demo)",
        save=str(out_dir / "1_classification_map.png"),
    )
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("[1/7] Classification map saved")

    # ── 2. Confusion matrix ──────────────────────────────────────
    from benchmark.eval.plots import plot_confusion_matrix

    # Larger synthetic predictions for better confusion matrix
    n_samples = 2000
    true_labels = np.random.randint(1, nc + 1, n_samples)
    pred_labels = true_labels.copy()
    n_err = int(n_samples * 0.15)
    err_idx = np.random.choice(n_samples, n_err, replace=False)
    pred_labels[err_idx] = np.random.randint(1, nc + 1, n_err)

    fig = plot_confusion_matrix(
        pred_labels, true_labels, class_names,
        normalize=True,
        title="Confusion Matrix — Trento (demo)",
        save=str(out_dir / "2_confusion_matrix.png"),
    )
    plt.close(fig)
    print("[2/7] Confusion matrix saved")

    # ── 3. Per-class accuracy ────────────────────────────────────
    from benchmark.eval.plots import plot_per_class_accuracy

    fig = plot_per_class_accuracy(
        pred_labels, true_labels, class_names,
        title="Per-Class Accuracy — Trento (demo)",
        save=str(out_dir / "3_per_class_accuracy.png"),
    )
    plt.close(fig)
    print("[3/7] Per-class accuracy saved")

    # ── 4. Radar comparison ──────────────────────────────────────
    from benchmark.eval.plots import plot_radar_comparison

    radar_results = {
        "iCaRL":    {"Trento": 82.3, "Houston2013": 71.5, "MUUFL": 68.2, "IndianPines": 75.1, "PaviaU": 88.4},
        "WA":       {"Trento": 84.1, "Houston2013": 73.8, "MUUFL": 70.5, "IndianPines": 76.3, "PaviaU": 87.9},
        "NCM":      {"Trento": 78.5, "Houston2013": 65.2, "MUUFL": 62.1, "IndianPines": 70.8, "PaviaU": 85.3},
        "FineTune": {"Trento": 55.2, "Houston2013": 42.1, "MUUFL": 38.7, "IndianPines": 48.5, "PaviaU": 62.1},
        "EWC":      {"Trento": 72.1, "Houston2013": 58.3, "MUUFL": 55.9, "IndianPines": 64.2, "PaviaU": 80.5},
    }
    fig = plot_radar_comparison(
        radar_results,
        title="Per-Dataset AA (%) — B1 Protocol (demo)",
        save=str(out_dir / "4_radar_comparison.png"),
    )
    plt.close(fig)
    print("[4/7] Radar comparison saved")

    # ── 5. Task progression curves ───────────────────────────────
    from benchmark.eval.plots import plot_task_curves

    class FakeResult:
        def __init__(self):
            self.protocol_name = "B1"
            self.method_name = "iCaRL"
            self.bwt = -0.045
            self.fwt = 0.72
            self.task_results = []
            for i in range(9):
                r = type('R', (), {
                    'task_id': i,
                    'oa': 0.85 - i * 0.02 + np.random.normal(0, 0.005),
                    'avg_aa': 0.82 - i * 0.025 + np.random.normal(0, 0.005),
                    'kappa': 0.83 - i * 0.022 + np.random.normal(0, 0.005),
                })()
                self.task_results.append(r)

    fig = plot_task_curves(
        FakeResult(),
        save=str(out_dir / "5_task_curves.png"),
    )
    plt.close(fig)
    print("[5/7] Task progression curves saved")

    # ── 6. Methods overlay ───────────────────────────────────────
    from benchmark.eval.plots import plot_methods_overlay

    methods_results = {}
    for method_name, base_oa, decay in [
        ("iCaRL", 0.85, 0.018), ("WA", 0.86, 0.016),
        ("NCM", 0.82, 0.012), ("FineTune", 0.80, 0.045),
        ("EWC", 0.83, 0.025), ("LwF", 0.84, 0.022),
    ]:
        r = type('R', (), {
            'protocol_name': 'B1', 'method_name': method_name,
            'task_results': [
                type('T', (), {
                    'task_id': i,
                    'oa': base_oa - i * decay + np.random.normal(0, 0.003),
                    'avg_aa': base_oa - 0.02 - i * decay + np.random.normal(0, 0.003),
                    'kappa': base_oa - 0.01 - i * decay + np.random.normal(0, 0.003),
                })() for i in range(9)
            ]
        })()
        methods_results[method_name] = r

    fig = plot_methods_overlay(
        methods_results, metric="oa",
        save=str(out_dir / "6_methods_overlay.png"),
    )
    plt.close(fig)
    print("[6/7] Methods overlay saved")

    # ── 7. Method comparison bar chart ───────────────────────────
    from benchmark.eval.plots import plot_method_comparison

    bar_results = {}
    for name, r in methods_results.items():
        r.final_oa = r.task_results[-1].oa
        r.final_aa = r.task_results[-1].avg_aa
        r.final_kappa = r.task_results[-1].kappa
        r.bwt = -np.random.uniform(0.02, 0.08)
        r.fwt = np.random.uniform(0.6, 0.8)
        bar_results[name] = r

    fig = plot_method_comparison(
        bar_results, metric="final_oa",
        save=str(out_dir / "7_method_comparison_bar.png"),
    )
    plt.close(fig)
    print("[7/7] Method comparison bar chart saved")

    print(f"\nAll demo figures saved to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
