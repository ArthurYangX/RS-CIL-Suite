"""Standalone inference from a saved checkpoint.

Usage:
    # Predict on all seen datasets after task 5
    python benchmark/infer.py \
        --checkpoint checkpoints/icarl_B1_seed0/task_5.pt \
        --protocol B1 --method icarl --data_root ~/data/rs_cil

    # Generate classification maps
    python benchmark/infer.py \
        --checkpoint checkpoints/icarl_B1_seed0/task_5.pt \
        --protocol B1 --method icarl --data_root ~/data/rs_cil \
        --save_maps --output_dir results/maps/

    # Save metrics to JSON
    python benchmark/infer.py \
        --checkpoint checkpoints/icarl_B1_seed0/task_5.pt \
        --protocol B1 --method icarl --data_root ~/data/rs_cil \
        --output results/icarl_infer.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.datasets.registry import get_dataset
from benchmark.protocols.cil import PROTOCOLS
from benchmark.eval.metrics import evaluate, BenchmarkResult
from benchmark.methods import get_method_registry
from benchmark.config import load_config, flatten_config


def _remap_labels(ds, local_ids, global_ids):
    from benchmark.datasets.base import PatchDataset
    local_to_global = {l: g for l, g in zip(local_ids, global_ids)}
    new_labels = ds.labels.clone()
    for l, g in local_to_global.items():
        new_labels[ds.labels == l] = g
    new_ds = PatchDataset.__new__(PatchDataset)
    new_ds.hsi    = ds.hsi
    new_ds.lidar  = ds.lidar
    new_ds.labels = new_labels
    new_ds.coords = ds.coords
    return new_ds


def main():
    p = argparse.ArgumentParser(description="RS-CIL Suite — Inference")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--protocol",   required=True,
                   help="Protocol key (e.g. B1)")
    p.add_argument("--method",     required=True,
                   help="Method name (e.g. icarl)")
    p.add_argument("--data_root",  required=True,
                   help="Root directory containing dataset sub-folders")
    p.add_argument("--config",     default=None,
                   help="Path to custom YAML config")
    p.add_argument("--opts",       nargs="*", default=None,
                   help="Config overrides (key=value)")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--patch_size", type=int, default=7)
    p.add_argument("--pca_components", type=int, default=36)
    p.add_argument("--output",     default=None,
                   help="Path to save JSON metrics")
    p.add_argument("--output_dir", default=None,
                   help="Directory for classification maps and figures")
    p.add_argument("--save_maps",  action="store_true",
                   help="Generate classification map images")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Protocol ──────────────────────────────────────────────────
    from benchmark.protocols.cil import get_protocol
    protocol = get_protocol(args.protocol)

    # ── Load checkpoint metadata ──────────────────────────────────
    ckpt_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    task_id = ckpt_meta.get("task_id", len(protocol.tasks) - 1)
    seen_classes = ckpt_meta.get("seen_classes", [])
    run_meta = ckpt_meta.get("run_meta", {})
    print(f"Checkpoint: task_id={task_id}, "
          f"seen_classes={len(seen_classes)}, "
          f"method={ckpt_meta.get('method_name', '?')}")

    # ── Resolve params: checkpoint run_meta is the default, CLI overrides ──
    # This makes checkpoints self-describing: if the user doesn't pass
    # --patch_size or --pca_components, we use what the checkpoint was
    # trained with. Explicit CLI args always take precedence.
    _patch_size = args.patch_size
    _pca_components = args.pca_components
    _backbone = None

    # CLI default sentinels (must match _build_parser defaults)
    _cli_defaults = {"patch_size": 7, "pca_components": 36}

    if run_meta:
        # Validate protocol and method match
        if "protocol" in run_meta and run_meta["protocol"] != args.protocol:
            raise ValueError(
                f"CLI --protocol={args.protocol} does not match checkpoint "
                f"protocol={run_meta['protocol']}. Use the same protocol "
                f"that was used during training.")
        if "method" in run_meta and run_meta["method"] != args.method:
            raise ValueError(
                f"CLI --method={args.method} does not match checkpoint "
                f"method={run_meta['method']}. Cannot load weights into "
                f"a different method architecture.")

        # Use checkpoint values as defaults when CLI is at its default
        if args.patch_size == _cli_defaults["patch_size"] and "patch_size" in run_meta:
            _patch_size = run_meta["patch_size"]
        if args.pca_components == _cli_defaults["pca_components"] and "pca_components" in run_meta:
            _pca_components = run_meta["pca_components"]
        if "backbone" in run_meta:
            _backbone = run_meta["backbone"]

        # Warn on mismatch (only when CLI was explicitly set to non-default)
        for key, cli_val, meta_val, default in [
            ("patch_size", args.patch_size, run_meta.get("patch_size"), _cli_defaults["patch_size"]),
            ("pca_components", args.pca_components, run_meta.get("pca_components"), _cli_defaults["pca_components"]),
        ]:
            if meta_val is not None and cli_val != default and cli_val != meta_val:
                print(f"[WARN] CLI --{key}={cli_val} differs from "
                      f"checkpoint run_meta {key}={meta_val}")

    # ── Load datasets ─────────────────────────────────────────────
    _train_ratio = run_meta.get("train_ratio") if run_meta else None
    datasets = {}
    for ds_name in protocol.dataset_order:
        root = Path(args.data_root) / ds_name
        ds_kwargs = dict(root=root, patch_size=_patch_size,
                         pca_components=_pca_components)
        if _train_ratio is not None:
            ds_kwargs["train_ratio"] = _train_ratio
        datasets[ds_name] = get_dataset(ds_name, **ds_kwargs)
        info = datasets[ds_name].info
        print(f"  [{ds_name}] {info.num_classes} classes | "
              f"test={len(datasets[ds_name].test)}")

    # Unify LiDAR channels
    lid_ch_max = max(ds.train.lidar.shape[1] for ds in datasets.values())
    if lid_ch_max > 1:
        for ds_name, ds in datasets.items():
            if ds.train.lidar.shape[1] < lid_ch_max:
                ds._train = ds.train.pad_lidar(lid_ch_max)
                ds._test  = ds.test.pad_lidar(lid_ch_max)

    # ── Build method and load checkpoint ──────────────────────────
    registry = get_method_registry()
    if args.method not in registry:
        raise ValueError(f"Unknown method '{args.method}'. "
                         f"Available: {sorted(registry)}")

    # Prefer checkpoint's saved config; fall back to loading from disk.
    # CLI --opts and --config deep-merge on top (not shallow replace).
    from benchmark.config import _deep_merge
    if run_meta.get("config"):
        cfg = run_meta["config"]
        print(f"[INFO] Using config from checkpoint run_meta")
        # Deep-merge CLI overrides on top of checkpoint config
        if args.opts or args.config:
            cli_cfg = load_config(args.method, config_path=args.config,
                                  cli_overrides=args.opts)
            cfg = _deep_merge(cfg, cli_cfg)
    else:
        cfg = load_config(args.method, config_path=args.config,
                          cli_overrides=args.opts)

    flat_cfg = flatten_config(cfg)

    # Map config key to method kwarg
    if "_backbone" in flat_cfg:
        flat_cfg["backbone"] = flat_cfg.pop("_backbone")
    # Checkpoint backbone overrides config default
    if _backbone:
        if flat_cfg.get("backbone") != _backbone:
            print(f"[INFO] Using backbone '{_backbone}' from checkpoint "
                  f"(config had '{flat_cfg.get('backbone', 'N/A')}')")
        flat_cfg["backbone"] = _backbone

    # Use actual data channels (PCA may have clipped below requested)
    hsi_ch = max(ds.train.hsi.shape[1] for ds in datasets.values())
    lid_ch = lid_ch_max
    kwargs = dict(hsi_channels=hsi_ch, lidar_channels=lid_ch,
                  num_classes=protocol.total_classes, device=device)
    kwargs.update(flat_cfg)

    method = registry[args.method](**kwargs)
    method.load_checkpoint(args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # ── Class-to-dataset mapping ──────────────────────────────────
    class_to_dataset = {}
    for task in protocol.tasks:
        for gid in task.global_class_ids:
            class_to_dataset[gid] = task.dataset_name

    # ── Inference ─────────────────────────────────────────────────
    all_preds, all_targets = [], []
    per_dataset_outputs: dict[str, dict] = {}
    for eval_ds_name in protocol.dataset_order:
        eval_ds = datasets[eval_ds_name]
        ds_seen_local = [c for t in protocol.tasks[:task_id + 1]
                           if t.dataset_name == eval_ds_name
                           for c in t.class_ids]
        ds_seen_global = [c for t in protocol.tasks[:task_id + 1]
                            if t.dataset_name == eval_ds_name
                            for c in t.global_class_ids]
        if not ds_seen_local:
            continue
        test_sub = eval_ds.test.subset(ds_seen_local)
        test_sub = _remap_labels(test_sub, ds_seen_local, ds_seen_global)
        test_loader = DataLoader(test_sub, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)
        preds, targets = method.predict(test_loader)
        all_preds.append(preds)
        all_targets.append(targets)
        per_dataset_outputs[eval_ds_name] = {
            "preds": preds,
            "targets": targets,
            "coords": test_sub.coords.cpu().numpy() if getattr(test_sub, "coords", None) is not None else None,
            "global_to_local": {
                int(global_id): int(local_id)
                for local_id, global_id in zip(ds_seen_local, ds_seen_global)
            },
        }
        # Per-dataset metrics
        from benchmark.eval.metrics import evaluate as eval_fn
        ds_result = eval_fn(preds, targets, ds_seen_global,
                            {g: eval_ds_name for g in ds_seen_global},
                            [eval_ds_name])
        print(f"  [{eval_ds_name}] OA={ds_result.oa*100:.2f}%  "
              f"AA={ds_result.avg_aa*100:.2f}%  κ={ds_result.kappa:.4f}")

    # ── Aggregate metrics ─────────────────────────────────────────
    if all_preds:
        preds_np = np.concatenate(all_preds)
        targets_np = np.concatenate(all_targets)
        result = evaluate(preds_np, targets_np, seen_classes,
                          class_to_dataset, protocol.dataset_order)
        print(f"\nOverall: OA={result.oa*100:.2f}%  "
              f"AA={result.avg_aa*100:.2f}%  κ={result.kappa:.4f}")

        # Save JSON
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "protocol": args.protocol,
                "method": args.method,
                "checkpoint": str(args.checkpoint),
                "task_id": task_id,
                "oa": result.oa,
                "aa": result.avg_aa,
                "kappa": result.kappa,
                "per_dataset": result.per_dataset,
            }
            with open(out, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Results saved → {out}")

    # ── Classification maps ───────────────────────────────────────
    if args.save_maps:
        out_dir = Path(args.output_dir or "maps")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            from benchmark.eval.plots import plot_classification_map
            for ds_name_vis, ds_vis in datasets.items():
                if ds_name_vis in per_dataset_outputs and hasattr(ds_vis, 'gt_map'):
                    try:
                        payload = per_dataset_outputs[ds_name_vis]
                        plot_classification_map(
                            gt_map=ds_vis.gt_map,
                            preds=payload["preds"], targets=payload["targets"],
                            class_names=ds_vis.class_names,
                            dataset_name=ds_name_vis,
                            title=f"{args.method} — {ds_name_vis}",
                            save=str(out_dir / f"map_{ds_name_vis}.pdf"),
                            coords=payload["coords"],
                            global_to_local=payload["global_to_local"],
                        )
                        print(f"  Map saved → {out_dir / f'map_{ds_name_vis}.pdf'}")
                    except Exception as e:
                        print(f"  [WARN] Map for {ds_name_vis}: {e}")
        except ImportError as e:
            print(f"[WARN] Plotting unavailable: {e}")


if __name__ == "__main__":
    main()
