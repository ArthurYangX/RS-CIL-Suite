"""RS-CIL Suite runner.

Usage:
    python benchmark/run.py --protocol B1 --method finetune \
        --data_root ~/data/rs_cil --seed 0

    # With YAML config override
    python benchmark/run.py --protocol B1 --method icarl \
        --data_root ~/data/rs_cil --opts training.lr=0.0005 method.T=3.0

    # With explicit config file
    python benchmark/run.py --protocol B1 --method icarl \
        --config my_config.yaml --data_root ~/data/rs_cil

    # With wandb logging
    python benchmark/run.py --protocol A_Houston2013 --method icarl \
        --data_root ~/data/rs_cil --wandb --wandb_project rs-cil

    # Multi-seed
    python benchmark/run.py --protocol B1 --method ncm --seeds 0,1,2 \
        --data_root ~/data/rs_cil --output results/ncm_B1.json

    # Save checkpoints + classification maps
    python benchmark/run.py --protocol B1 --method icarl \
        --data_root ~/data/rs_cil --save_checkpoints --plot_maps

    # Generate figures after run
    python benchmark/run.py --protocol B1 --method icarl \
        --data_root ~/data/rs_cil --output results/icarl_B1.json --plot
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── local imports ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.datasets.registry import get_dataset
from benchmark.protocols.cil import PROTOCOLS, CILProtocol, get_protocol
from benchmark.eval.metrics import BenchmarkResult, TaskFeedbackResult, evaluate
from benchmark.methods import get_method_registry
from benchmark.config import load_config, flatten_config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_to_dataset(protocol: CILProtocol) -> dict[int, str]:
    mapping = {}
    for task in protocol.tasks:
        for gid in task.global_class_ids:
            mapping[gid] = task.dataset_name
    return mapping


def build_dataset_class_mappings(protocol: CILProtocol, datasets: dict) -> dict[str, dict]:
    """Return dataset-local/global class mappings for saved plotting artifacts."""
    mappings: dict[str, dict] = {}
    for ds_name, ds in datasets.items():
        local_ids: list[int] = []
        global_ids: list[int] = []
        for task in protocol.tasks:
            if task.dataset_name != ds_name:
                continue
            local_ids.extend(task.class_ids)
            global_ids.extend(task.global_class_ids)
        pairs = sorted(zip(global_ids, local_ids), key=lambda x: x[0])
        mappings[ds_name] = {
            "class_names": list(ds.class_names),
            "local_class_ids": [local for _, local in pairs],
            "global_class_ids": [global_id for global_id, _ in pairs],
        }
    return mappings


def _task_class_names(dataset, class_ids: list[int]) -> list[str]:
    return [
        dataset.class_names[c]
        for c in class_ids
        if 0 <= c < len(dataset.class_names)
    ]


def _coords_to_numpy(ds) -> np.ndarray | None:
    coords = getattr(ds, "coords", None)
    if coords is None:
        return None
    return coords.cpu().numpy()


def _make_task_eval_artifact(
    *,
    after_task_id: int,
    eval_task,
    dataset,
    dataset_mapping: dict,
    args,
    preds: np.ndarray,
    targets: np.ndarray,
    coords: np.ndarray | None,
    oa: float,
    aa: float,
    kappa: float,
) -> dict:
    return {
        "after_task_id": after_task_id,
        "eval_task_id": eval_task.task_id,
        "dataset_name": eval_task.dataset_name,
        "method_name": args.method,
        "protocol_name": args.protocol,
        "seed": args.seed,
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "num_samples": int(len(targets)),
        "class_names": list(dataset.class_names),
        "task_class_names": _task_class_names(dataset, eval_task.class_ids),
        "local_class_ids": list(eval_task.class_ids),
        "global_class_ids": list(eval_task.global_class_ids),
        "dataset_local_class_ids": list(dataset_mapping.get("local_class_ids", [])),
        "dataset_global_class_ids": list(dataset_mapping.get("global_class_ids", [])),
        "preds": preds.astype(np.int64, copy=False),
        "targets": targets.astype(np.int64, copy=False),
        "coords": coords.astype(np.int32, copy=False) if coords is not None else None,
    }


def _split_task_eval_artifacts(task_eval_artifacts: list[dict]) -> tuple[list[dict], dict[str, np.ndarray]]:
    metadata: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    for idx, artifact in enumerate(task_eval_artifacts):
        prefix = f"eval_{idx:04d}"
        arrays[f"{prefix}_preds"] = artifact["preds"]
        arrays[f"{prefix}_targets"] = artifact["targets"]
        if artifact["coords"] is not None:
            arrays[f"{prefix}_coords"] = artifact["coords"]

        meta = {
            k: v for k, v in artifact.items()
            if k not in {"preds", "targets", "coords"}
        }
        meta["preds_key"] = f"{prefix}_preds"
        meta["targets_key"] = f"{prefix}_targets"
        if artifact["coords"] is not None:
            meta["coords_key"] = f"{prefix}_coords"
        metadata.append(meta)
    return metadata, arrays


def _default_output_path(args) -> Path:
    """Return the default per-run results path used for saved artifacts."""
    return Path("results") / f"{args.method}_{args.protocol}_seed{args.seed}.json"


# ── wandb helpers ─────────────────────────────────────────────────

def _init_wandb(args, cfg: dict):
    """Initialise wandb run if --wandb is set. Returns run or None."""
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
        wandb_config = {
            "protocol":       args.protocol,
            "method":         args.method,
            "seed":           args.seed,
            "patch_size":     args.patch_size,
            "pca_components": args.pca_components,
            "batch_size":     args.batch_size,
        }
        # Merge in method config for hyperparameter comparison
        wandb_config.update(flatten_config(cfg))
        run = wandb.init(
            project=getattr(args, "wandb_project", "rs-cil-suite"),
            name=f"{args.method}_{args.protocol}_seed{args.seed}",
            config=wandb_config,
            reinit=True,
        )
        return run
    except ImportError:
        print("[WARN] wandb not installed. Run: pip install wandb")
        return None


def _wandb_log_task(wandb_run, task_id: int, task_result,
                    preds_np=None, targets_np=None, seen_classes=None,
                    class_to_dataset=None, datasets=None, protocol=None):
    if wandb_run is None:
        return
    import wandb
    wandb_run.log({
        "task":       task_id,
        "oa":         task_result.oa * 100,
        "aa":         task_result.avg_aa * 100,
        "kappa":      task_result.kappa,
    }, step=task_id)

    # Per-class accuracy
    if preds_np is not None and targets_np is not None and seen_classes:
        for c in sorted(seen_classes):
            mask = targets_np == c
            if mask.sum() > 0:
                acc = (preds_np[mask] == c).mean()
                wandb_run.log({f"class_acc/gid_{c}": acc * 100}, step=task_id)

    # Confusion matrix
    if preds_np is not None and targets_np is not None:
        try:
            class_names = [str(c) for c in sorted(seen_classes)]
            wandb_run.log({
                f"confusion/task_{task_id}": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=targets_np.tolist(),
                    preds=preds_np.tolist(),
                    class_names=class_names,
                )
            }, step=task_id)
        except Exception:
            pass  # graceful fallback


def _wandb_log_final(wandb_run, result):
    if wandb_run is None:
        return
    wandb_run.summary.update({
        "final_oa":    result.final_oa * 100,
        "final_aa":    result.final_aa * 100,
        "final_kappa": result.final_kappa,
        "bwt":         result.bwt * 100,
        "fwt":         result.fwt * 100,
        "forgetting_mean": (
            float(np.mean(list(result.forgetting.values()))) * 100
            if result.forgetting else 0.0
        ),
        "plasticity_mean": (
            float(np.mean(list(result.plasticity.values()))) * 100
            if result.plasticity else 0.0
        ),
    })
    wandb_run.finish()


# ── Main run ──────────────────────────────────────────────────────

def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Config ─────────────────────────────────────────────────
    cfg = load_config(
        args.method,
        config_path=getattr(args, "config", None),
        cli_overrides=getattr(args, "opts", None),
    )
    print(f"Config: {cfg}")

    wandb_run = _init_wandb(args, cfg)

    # ── Protocol ──────────────────────────────────────────────────
    protocol = get_protocol(args.protocol)
    print(protocol.summary())

    # ── Load datasets ─────────────────────────────────────────────
    # Custom train_ratio from protocol YAML (if set)
    train_ratio = getattr(protocol, "train_ratio", None)

    datasets = {}
    for ds_name in protocol.dataset_order:
        root = Path(args.data_root) / ds_name if args.data_root else None
        ds_kwargs = dict(root=root, patch_size=args.patch_size,
                         pca_components=args.pca_components)
        if train_ratio is not None:
            ds_kwargs["train_ratio"] = train_ratio
        datasets[ds_name] = get_dataset(ds_name, **ds_kwargs)
        info = datasets[ds_name].info
        print(f"  [{ds_name}] {info.num_classes} classes | "
              f"train={len(datasets[ds_name].train)} "
              f"test={len(datasets[ds_name].test)}")

    # Unify LiDAR channel count across all datasets (pad to maximum)
    lid_ch_max = max(ds.train.lidar.shape[1] for ds in datasets.values())
    if lid_ch_max > 1:
        for ds_name, ds in datasets.items():
            if ds.train.lidar.shape[1] < lid_ch_max:
                ds._train = ds.train.pad_lidar(lid_ch_max)
                ds._test  = ds.test.pad_lidar(lid_ch_max)

    # ── Build class-to-dataset mapping ────────────────────────────
    class_to_dataset = build_class_to_dataset(protocol)
    dataset_class_mappings = build_dataset_class_mappings(protocol, datasets)

    # ── Method ────────────────────────────────────────────────────
    method = _build_method(args.method, protocol, device, datasets, cfg,
                           pca_components=args.pca_components)

    # Set up wandb logging callback
    if wandb_run:
        _global_step = [0]
        def _wandb_log_fn(metrics: dict, step=None):
            wandb_run.log(metrics, step=step if step is not None else _global_step[0])
            _global_step[0] += 1
        method.log_fn = _wandb_log_fn

    # ── Checkpoint dir + run metadata ────────────────────────────
    run_meta = {
        "protocol": args.protocol,
        "method": args.method,
        "seed": args.seed,
        "patch_size": args.patch_size,
        "pca_components": args.pca_components,
        "train_ratio": train_ratio,
        "backbone": cfg.get("model", {}).get("backbone", "simple_encoder") if cfg else "simple_encoder",
        "config": cfg,
    }
    ckpt_dir = None
    if getattr(args, "save_checkpoints", False):
        ckpt_base = Path(getattr(args, "checkpoint_dir", "checkpoints"))
        ckpt_dir = ckpt_base / f"{args.method}_{args.protocol}_seed{args.seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints → {ckpt_dir}")

    # ── CIL loop ──────────────────────────────────────────────────
    result = BenchmarkResult(protocol_name=args.protocol, method_name=args.method)
    seen_classes: list[int] = []
    task_eval_artifacts: list[dict] = []

    for task in protocol.tasks:
        seen_classes = seen_classes + task.global_class_ids
        ds_name = task.dataset_name
        ds = datasets[ds_name]

        # Training data: current task classes (local IDs)
        train_sub = ds.train.subset(task.class_ids)
        # Re-map labels to global IDs
        train_sub = _remap_labels(train_sub, task.class_ids, task.global_class_ids)
        train_loader = DataLoader(train_sub, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)

        print(f"\n{'='*60}")
        print(f"Task {task.task_id}: [{ds_name}] "
              f"classes={task.class_ids} "
              f"(global={task.global_class_ids})  "
              f"n_train={len(train_sub)}")
        print(f"{'='*60}")

        method.before_task(task)
        method.train_task(task, train_loader)
        method.after_task(task, train_loader)

        # ── Evaluate on each seen task exactly (task-aware) ───────
        all_preds, all_targets = [], []
        for eval_task in protocol.tasks[:task.task_id + 1]:
            eval_ds_name = eval_task.dataset_name
            eval_ds = datasets[eval_ds_name]
            test_sub = eval_ds.test.subset(eval_task.class_ids)
            test_sub = _remap_labels(
                test_sub, eval_task.class_ids, eval_task.global_class_ids
            )
            test_loader = DataLoader(test_sub, batch_size=args.batch_size * 2,
                                     shuffle=False, num_workers=0)
            preds, targets = method.predict(test_loader)
            eval_result = evaluate(
                preds,
                targets,
                eval_task.global_class_ids,
                {g: eval_ds_name for g in eval_task.global_class_ids},
                [eval_ds_name],
            )
            result.add_task_feedback(TaskFeedbackResult(
                after_task_id=task.task_id,
                eval_task_id=eval_task.task_id,
                dataset_name=eval_ds_name,
                oa=eval_result.oa,
                aa=eval_result.avg_aa,
                kappa=eval_result.kappa,
                num_samples=int(len(targets)),
            ))
            task_eval_artifacts.append(_make_task_eval_artifact(
                after_task_id=task.task_id,
                eval_task=eval_task,
                dataset=eval_ds,
                dataset_mapping=dataset_class_mappings.get(eval_ds_name, {}),
                args=args,
                preds=preds,
                targets=targets,
                coords=_coords_to_numpy(test_sub),
                oa=eval_result.oa,
                aa=eval_result.avg_aa,
                kappa=eval_result.kappa,
            ))
            all_preds.append(preds)
            all_targets.append(targets)

        if all_preds:
            preds_np   = np.concatenate(all_preds)
            targets_np = np.concatenate(all_targets)
            task_result = evaluate(preds_np, targets_np, seen_classes,
                                   class_to_dataset, protocol.dataset_order)
            task_result.task_id = task.task_id
            result.add(task_result)

            ds_str = "  ".join(
                f"{k}={v*100:.1f}%" for k, v in task_result.per_dataset.items()
            )
            print(f"  OA={task_result.oa*100:.2f}%  "
                  f"AA={task_result.avg_aa*100:.2f}%  "
                  f"κ={task_result.kappa:.4f}")
            print(f"  {ds_str}")
            _wandb_log_task(wandb_run, task.task_id, task_result,
                            preds_np, targets_np, seen_classes,
                            class_to_dataset, datasets, protocol)

        # ── Save checkpoint ────────────────────────────────────────
        if ckpt_dir is not None:
            method.save_checkpoint(ckpt_dir / f"task_{task.task_id}.pt",
                                   task.task_id, run_meta=run_meta)

    result.compute_cl_metrics()
    print(result.summary())
    _wandb_log_final(wandb_run, result)

    # ── Save ──────────────────────────────────────────────────────
    out = Path(args.output) if args.output else _default_output_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    task_eval_meta, task_eval_arrays = _split_task_eval_artifacts(task_eval_artifacts)
    artifacts_file = None
    dataset_mappings_for_save = {
        ds_name: dict(mapping) for ds_name, mapping in dataset_class_mappings.items()
    }
    for ds_idx, (ds_name, ds) in enumerate(datasets.items()):
        gt_key = f"dataset_{ds_idx:02d}_gt_map"
        task_eval_arrays[gt_key] = ds.gt_map.astype(np.int32, copy=False)
        dataset_mappings_for_save.setdefault(ds_name, {})["gt_map_key"] = gt_key

    if task_eval_arrays:
        artifacts_path = out.with_name(f"{out.stem}_task_artifacts.npz")
        np.savez_compressed(artifacts_path, **task_eval_arrays)
        artifacts_file = artifacts_path.name
    data = {
        "protocol": args.protocol,
        "method":   args.method,
        "seed":     args.seed,
        "artifact_version": 2,
        "final_oa":    result.final_oa,
        "final_aa":    result.final_aa,
        "final_kappa": result.final_kappa,
        "bwt":         result.bwt,
        "fwt":         result.fwt,
        "forgetting":  result.forgetting,
        "plasticity":  result.plasticity,
        "dataset_mappings": dataset_mappings_for_save,
        "tasks": [
            {"task_id": r.task_id, "oa": r.oa, "avg_aa": r.avg_aa,
             "kappa": r.kappa, "per_dataset": r.per_dataset}
            for r in result.task_results
        ],
        "task_feedback": [
            {
                "after_task_id": r.after_task_id,
                "eval_task_id": r.eval_task_id,
                "dataset_name": r.dataset_name,
                "oa": r.oa,
                "aa": r.aa,
                "kappa": r.kappa,
                "num_samples": r.num_samples,
            }
            for r in result.task_feedback
        ],
        "task_evals": task_eval_meta,
    }
    if artifacts_file is not None:
        data["artifacts_file"] = artifacts_file
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved → {out}")

    # ── Plot ──────────────────────────────────────────────────────
    if getattr(args, "plot", False):
        try:
            from benchmark.eval.plots import (
                plot_forgetting_matrix,
                plot_task_accuracy_matrix,
                plot_task_curves,
                plot_task_feedback_curve,
            )
            stem = out.stem
            fig_dir = out.parent / "figs"
            plot_task_curves(result, save=str(fig_dir / f"{stem}_curves.pdf"))
            plot_forgetting_matrix(result, save=str(fig_dir / f"{stem}_forgetting.pdf"))
            if result.task_feedback:
                plot_task_accuracy_matrix(
                    result,
                    metric="oa",
                    save=str(fig_dir / f"{stem}_task_matrix.pdf"),
                )
                plot_task_feedback_curve(
                    result,
                    metric="oa",
                    save=str(fig_dir / f"{stem}_task_feedback_curve.pdf"),
                )
        except ImportError as e:
            print(f"[WARN] Plotting skipped: {e}")

    # ── Classification maps ───────────────────────────────────────
    if getattr(args, "plot_maps", False):
        try:
            from benchmark.eval.plots import plot_classification_maps_per_task
            stem = out.stem
            fig_dir = out.parent / "figs"
            by_dataset: dict[str, list[dict]] = {}
            for artifact in task_eval_artifacts:
                by_dataset.setdefault(artifact["dataset_name"], []).append(artifact)

            for ds_name_vis, ds_artifacts in by_dataset.items():
                ds_vis = datasets.get(ds_name_vis)
                if ds_vis is None:
                    continue
                suffix = "" if len(by_dataset) == 1 else f"_{ds_name_vis}"
                plot_classification_maps_per_task(
                    gt_map=ds_vis.gt_map,
                    predictions_per_task=ds_artifacts,
                    class_names=ds_vis.class_names,
                    dataset_name=ds_name_vis,
                    method_name=args.method,
                    protocol_name=args.protocol,
                    save=str(fig_dir / f"{stem}{suffix}_maps_per_task.pdf"),
                )
        except ImportError as e:
            print(f"[WARN] Map plotting skipped: {e}")

    return result


# ── Label remapping helper ────────────────────────────────────────

def _remap_labels(ds, local_ids: list[int], global_ids: list[int]):
    """Return a copy of PatchDataset with labels mapped from local to global IDs."""
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


# ── Method factory (uses auto-registry + config) ─────────────────

def _build_method(name: str, protocol: CILProtocol, device, datasets,
                  cfg: dict | None = None, pca_components: int = 36):
    registry = get_method_registry()
    if name not in registry:
        raise ValueError(f"Unknown method '{name}'. "
                         f"Available: {sorted(registry)}")

    total  = protocol.total_classes

    # Use actual data channels (PCA may clip to min(requested, C, H*W))
    hsi_ch = max(ds.train.hsi.shape[1] for ds in datasets.values())
    lid_ch = max(ds.train.lidar.shape[1] for ds in datasets.values())

    kwargs = dict(hsi_channels=hsi_ch, lidar_channels=lid_ch,
                  num_classes=total, device=device)

    # Merge flattened config into kwargs (config values override defaults)
    if cfg:
        flat = flatten_config(cfg)
        # Map config key to method kwarg
        if "_backbone" in flat:
            flat["backbone"] = flat.pop("_backbone")
        kwargs.update(flat)

    return registry[name](**kwargs)


# ── CLI ───────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(description="RS-CIL Suite Runner")
    p.add_argument("--protocol",      default="B1",
                   help=f"Protocol key. Available: {list(PROTOCOLS)}")
    p.add_argument("--method",        default="ncm",
                   help="Method name (auto-discovered from benchmark.methods)")
    p.add_argument("--data_root",     default=None,
                   help="Root directory containing dataset sub-folders")
    p.add_argument("--patch_size",    type=int,   default=7)
    p.add_argument("--pca_components",type=int,   default=36)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--seeds",         type=str,   default=None,
                   help="Comma-separated seeds for multi-seed averaging, e.g. '0,1,2'")
    p.add_argument("--output",        default=None,
                   help="Path to save JSON results (default: results/{method}_{protocol}_seed{seed}.json)")
    # Config
    p.add_argument("--config",        default=None,
                   help="Path to custom YAML config (overrides auto-discovered method config)")
    p.add_argument("--opts",          nargs="*",  default=None,
                   help="Config overrides in dotted key=value format, "
                        "e.g. training.lr=0.0005 method.T=3.0")
    # Checkpoints
    p.add_argument("--save_checkpoints", action="store_true",
                   help="Save model checkpoint + predictions after each task")
    p.add_argument("--checkpoint_dir",   default="checkpoints",
                   help="Base directory for checkpoints (default: checkpoints/)")
    # Wandb
    p.add_argument("--wandb",         action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", default="rs-cil-suite",
                   help="W&B project name (default: rs-cil-suite)")
    # Plotting
    p.add_argument("--plot",          action="store_true",
                   help="Generate figures after run (requires matplotlib/seaborn)")
    p.add_argument("--plot_maps",     action="store_true",
                   help="Generate HyperKD-style classification maps after the run")
    return p


def main_cli():
    """Entry point for ``rs-cil-run`` console script."""
    p = _build_parser()
    args = p.parse_args()
    _run_from_args(args)


def _run_from_args(args):
    if args.seeds:
        # Multi-seed mode: run once per seed and average final metrics
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        all_results = []
        for seed in seeds:
            print(f"\n{'#'*60}")
            print(f"# Seed {seed}")
            print(f"{'#'*60}")
            args.seed = seed
            orig_output = args.output
            if orig_output:
                p2 = Path(orig_output)
                args.output = str(p2.parent / f"{p2.stem}_seed{seed}{p2.suffix}")
            all_results.append(run(args))
            args.output = orig_output
        # Print averaged summary
        import statistics
        oa  = statistics.mean(r.final_oa    for r in all_results)
        aa  = statistics.mean(r.final_aa    for r in all_results)
        kap = statistics.mean(r.final_kappa for r in all_results)
        bwt = statistics.mean(r.bwt         for r in all_results)
        fwt = statistics.mean(r.fwt         for r in all_results)
        oa_std  = statistics.stdev(r.final_oa    for r in all_results) if len(seeds) > 1 else 0
        aa_std  = statistics.stdev(r.final_aa    for r in all_results) if len(seeds) > 1 else 0
        print(f"\n{'='*60}")
        print(f"Multi-seed average  ({len(seeds)} seeds: {seeds})")
        print(f"  OA:  {oa*100:.2f} ± {oa_std*100:.2f}%")
        print(f"  AA:  {aa*100:.2f} ± {aa_std*100:.2f}%")
        print(f"  κ:   {kap:.4f}")
        print(f"  BWT: {bwt*100:.2f}pp")
        print(f"  Plasticity: {fwt*100:.2f}%")
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump({
                    "protocol": args.protocol, "method": args.method,
                    "seeds": seeds,
                    "oa_mean": oa, "oa_std": oa_std,
                    "aa_mean": aa, "aa_std": aa_std,
                    "kappa_mean": kap, "bwt_mean": bwt, "fwt_mean": fwt,
                }, f, indent=2)
    else:
        run(args)


if __name__ == "__main__":
    main_cli()
