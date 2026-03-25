"""RS-CIL Benchmark runner.

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
from benchmark.eval.metrics import evaluate, BenchmarkResult
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
            project=getattr(args, "wandb_project", "rs-cil-benchmark"),
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
        "forgetting":  result.forgetting * 100,
        "plasticity":  result.plasticity * 100,
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
    datasets = {}
    for ds_name in protocol.dataset_order:
        root = Path(args.data_root) / ds_name if args.data_root else None
        datasets[ds_name] = get_dataset(ds_name, root=root,
                                        patch_size=args.patch_size,
                                        pca_components=args.pca_components)
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

    # ── Method ────────────────────────────────────────────────────
    method = _build_method(args.method, protocol, device, datasets, cfg)

    # Set up wandb logging callback
    if wandb_run:
        _global_step = [0]
        def _wandb_log_fn(metrics: dict, step=None):
            wandb_run.log(metrics, step=step if step is not None else _global_step[0])
            _global_step[0] += 1
        method.log_fn = _wandb_log_fn

    # ── Checkpoint dir ────────────────────────────────────────────
    ckpt_dir = None
    if getattr(args, "save_checkpoints", False):
        ckpt_base = Path(getattr(args, "checkpoint_dir", "checkpoints"))
        ckpt_dir = ckpt_base / f"{args.method}_{args.protocol}_seed{args.seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints → {ckpt_dir}")

    # ── CIL loop ──────────────────────────────────────────────────
    result = BenchmarkResult(protocol_name=args.protocol, method_name=args.method)
    seen_classes: list[int] = []

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

        # ── Evaluate on all seen classes ──────────────────────────
        all_preds, all_targets = [], []
        for eval_ds_name in protocol.dataset_order:
            eval_ds = datasets[eval_ds_name]
            # Select seen classes that belong to this dataset
            ds_seen_local = [c for t in protocol.tasks[:task.task_id + 1]
                               if t.dataset_name == eval_ds_name
                               for c in t.class_ids]
            ds_seen_global = [c for t in protocol.tasks[:task.task_id + 1]
                                if t.dataset_name == eval_ds_name
                                for c in t.global_class_ids]
            if not ds_seen_local:
                continue
            test_sub  = eval_ds.test.subset(ds_seen_local)
            test_sub  = _remap_labels(test_sub, ds_seen_local, ds_seen_global)
            test_loader = DataLoader(test_sub, batch_size=args.batch_size * 2,
                                     shuffle=False, num_workers=0)
            preds, targets = method.predict(test_loader)
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

        # ── Save checkpoint + predictions ─────────────────────────
        if ckpt_dir is not None:
            method.save_checkpoint(ckpt_dir / f"task_{task.task_id}.pt",
                                   task.task_id)
            if all_preds:
                np.savez_compressed(
                    ckpt_dir / f"preds_task_{task.task_id}.npz",
                    preds=preds_np, targets=targets_np,
                    seen_classes=np.array(seen_classes),
                )

    result.compute_cl_metrics()
    print(result.summary())
    _wandb_log_final(wandb_run, result)

    # ── Save ──────────────────────────────────────────────────────
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "protocol": args.protocol,
            "method":   args.method,
            "seed":     args.seed,
            "final_oa":    result.final_oa,
            "final_aa":    result.final_aa,
            "final_kappa": result.final_kappa,
            "bwt":         result.bwt,
            "fwt":         result.fwt,
            "forgetting":  result.forgetting,
            "plasticity":  result.plasticity,
            "tasks": [
                {"task_id": r.task_id, "oa": r.oa, "avg_aa": r.avg_aa,
                 "kappa": r.kappa, "per_dataset": r.per_dataset}
                for r in result.task_results
            ],
        }
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved → {out}")

    # ── Plot ──────────────────────────────────────────────────────
    if getattr(args, "plot", False):
        try:
            from benchmark.eval.plots import plot_task_curves, plot_forgetting_matrix
            stem = Path(args.output).stem if args.output else f"{args.method}_{args.protocol}_seed{args.seed}"
            fig_dir = Path(args.output).parent / "figs" if args.output else Path("figs")
            plot_task_curves(result, save=str(fig_dir / f"{stem}_curves.pdf"))
            plot_forgetting_matrix(result, save=str(fig_dir / f"{stem}_forgetting.pdf"))
        except ImportError as e:
            print(f"[WARN] Plotting skipped: {e}")

    # ── Classification maps ───────────────────────────────────────
    if getattr(args, "plot_maps", False) and ckpt_dir is not None:
        try:
            from benchmark.eval.plots import plot_classification_map
            fig_dir = ckpt_dir / "maps"
            fig_dir.mkdir(parents=True, exist_ok=True)
            # Generate maps from saved predictions
            for npz_file in sorted(ckpt_dir.glob("preds_task_*.npz")):
                d = np.load(npz_file)
                tid = int(npz_file.stem.split("_")[-1])
                for ds_name_vis, ds_vis in datasets.items():
                    if hasattr(ds_vis, 'gt_map'):
                        try:
                            plot_classification_map(
                                gt_map=ds_vis.gt_map,
                                preds=d["preds"], targets=d["targets"],
                                class_names=ds_vis.class_names,
                                title=f"{args.method} — {ds_name_vis} (task {tid})",
                                save=str(fig_dir / f"map_{ds_name_vis}_task{tid}.pdf"),
                            )
                        except Exception as e:
                            print(f"[WARN] Map generation failed for {ds_name_vis}: {e}")
        except ImportError as e:
            print(f"[WARN] Map plotting skipped: {e}")

    return result


# ── Label remapping helper ────────────────────────────────────────

def _remap_labels(ds, local_ids: list[int], global_ids: list[int]):
    """Return a copy of PatchDataset with labels mapped from local to global IDs."""
    from benchmark.datasets.base import PatchDataset
    import torch
    local_to_global = {l: g for l, g in zip(local_ids, global_ids)}
    new_labels = ds.labels.clone()
    for l, g in local_to_global.items():
        new_labels[ds.labels == l] = g
    new_ds = PatchDataset.__new__(PatchDataset)
    new_ds.hsi    = ds.hsi
    new_ds.lidar  = ds.lidar
    new_ds.labels = new_labels
    return new_ds


# ── Method factory (uses auto-registry + config) ─────────────────

def _build_method(name: str, protocol: CILProtocol, device, datasets,
                  cfg: dict | None = None):
    registry = get_method_registry()
    if name not in registry:
        raise ValueError(f"Unknown method '{name}'. "
                         f"Available: {sorted(registry)}")

    total  = protocol.total_classes
    hsi_ch = 36   # always 36 after PCA

    # Determine LiDAR channel count from actual data
    lid_ch = max(
        ds.train.lidar.shape[1]
        for ds in datasets.values()
    )

    kwargs = dict(hsi_channels=hsi_ch, lidar_channels=lid_ch,
                  num_classes=total, device=device)

    # Merge flattened config into kwargs (config values override defaults)
    if cfg:
        kwargs.update(flatten_config(cfg))
        # Remove internal keys
        kwargs.pop("_backbone", None)

    return registry[name](**kwargs)


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RS-CIL Benchmark Runner")
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
                   help="Path to save JSON results")
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
    p.add_argument("--wandb_project", default="rs-cil-benchmark",
                   help="W&B project name (default: rs-cil-benchmark)")
    # Plotting
    p.add_argument("--plot",          action="store_true",
                   help="Generate figures after run (requires matplotlib/seaborn)")
    p.add_argument("--plot_maps",     action="store_true",
                   help="Generate classification maps (requires --save_checkpoints)")
    args = p.parse_args()

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
        print(f"  FWT: {fwt*100:.2f}%")
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
