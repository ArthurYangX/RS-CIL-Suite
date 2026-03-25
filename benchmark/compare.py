"""Results comparison and aggregation tool for RS-CIL benchmark.

Usage:
    # Compare multiple methods on one protocol
    python benchmark/compare.py results/B1_*.json

    # Generate a LaTeX table
    python benchmark/compare.py results/B1_*.json --latex

    # Compare across protocols
    python benchmark/compare.py results/*.json --group-by protocol
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path
from collections import defaultdict


def load_results(paths: list[str]) -> list[dict]:
    results = []
    for p in paths:
        for f in glob.glob(p):
            with open(f) as fp:
                d = json.load(fp)
            d["_file"] = f
            results.append(d)
    return results


def fmt(val: float, pct: bool = True, decimals: int = 2) -> str:
    if pct:
        return f"{val*100:.{decimals}f}"
    return f"{val:.{decimals+2}f}"


def print_table(results: list[dict], group_by: str = "method"):
    """Print a comparison table sorted by final OA."""
    if not results:
        print("No results found.")
        return

    # Collect all datasets seen across results
    all_ds = sorted({ds for r in results for ds in r.get("tasks", [{}])[-1].get("per_dataset", {}).keys()})

    # Group
    groups = defaultdict(list)
    for r in results:
        key = r.get(group_by, "unknown")
        groups[key].append(r)

    # Average across seeds if multiple runs per group
    rows = []
    for key, rs in sorted(groups.items()):
        oa  = sum(r.get("final_oa",    r.get("oa_mean",    0)) for r in rs) / len(rs)
        aa  = sum(r.get("final_aa",    r.get("aa_mean",    0)) for r in rs) / len(rs)
        kap = sum(r.get("final_kappa", r.get("kappa_mean", 0)) for r in rs) / len(rs)
        bwt = sum(r.get("bwt",         r.get("bwt_mean",   0)) for r in rs) / len(rs)
        fwt = sum(r.get("fwt",         r.get("fwt_mean",   0)) for r in rs) / len(rs)

        # Per-dataset from last task
        per_ds = {}
        for r in rs:
            tasks = r.get("tasks", [])
            if tasks:
                for ds, v in tasks[-1].get("per_dataset", {}).items():
                    per_ds.setdefault(ds, []).append(v)
        per_ds_avg = {ds: sum(vs)/len(vs) for ds, vs in per_ds.items()}

        rows.append({
            "key": key, "oa": oa, "aa": aa, "kappa": kap, "bwt": bwt, "fwt": fwt,
            "per_ds": per_ds_avg, "n": len(rs),
            "protocol": rs[0].get("protocol", "?"),
        })

    rows.sort(key=lambda r: r["oa"], reverse=True)

    # Header
    col_w = 14
    ds_cols = all_ds if len(all_ds) <= 5 else []
    header = f"{'Method':<{col_w}}{'OA%':>7}{'AA%':>7}{'κ':>7}{'BWT%':>7}{'FWT%':>7}"
    for ds in ds_cols:
        header += f"  {ds[:6]:>6}"
    header += f"  {'n':>3}"
    print(f"\n{'='*len(header)}")
    print(f"Protocol: {rows[0]['protocol'] if rows else '?'}")
    print('='*len(header))
    print(header)
    print('-'*len(header))

    for r in rows:
        line = (f"{r['key']:<{col_w}}"
                f"{fmt(r['oa']):>7}"
                f"{fmt(r['aa']):>7}"
                f"{fmt(r['kappa'], pct=False, decimals=3):>7}"
                f"{fmt(r['bwt']):>7}"
                f"{fmt(r['fwt']):>7}")
        for ds in ds_cols:
            v = r["per_ds"].get(ds, float("nan"))
            line += f"  {fmt(v):>6}" if v == v else f"  {'N/A':>6}"
        line += f"  {r['n']:>3}"
        print(line)
    print('='*len(header))


def print_latex(results: list[dict]):
    """Print a LaTeX booktabs table."""
    groups = defaultdict(list)
    for r in results:
        groups[r.get("method", "?")].append(r)

    rows = []
    for method, rs in sorted(groups.items()):
        oa  = sum(r.get("final_oa",    r.get("oa_mean",    0)) for r in rs) / len(rs)
        aa  = sum(r.get("final_aa",    r.get("aa_mean",    0)) for r in rs) / len(rs)
        kap = sum(r.get("final_kappa", r.get("kappa_mean", 0)) for r in rs) / len(rs)
        bwt = sum(r.get("bwt",         r.get("bwt_mean",   0)) for r in rs) / len(rs)
        rows.append((method, oa, aa, kap, bwt))

    rows.sort(key=lambda r: r[1], reverse=True)

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Method & OA (\%) & AA (\%) & $\kappa$ & BWT (pp) \\")
    print(r"\midrule")
    for method, oa, aa, kap, bwt in rows:
        print(f"{method} & {oa*100:.2f} & {aa*100:.2f} & {kap:.4f} & {bwt*100:.2f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{RS-CIL benchmark results.}")
    print(r"\end{table}")


def main():
    p = argparse.ArgumentParser(description="RS-CIL results comparison tool")
    p.add_argument("files", nargs="+", help="JSON result files (glob patterns ok)")
    p.add_argument("--latex", action="store_true", help="Output LaTeX table")
    p.add_argument("--group-by", default="method", choices=["method", "protocol", "seed"],
                   help="Column to group results by")
    args = p.parse_args()

    results = load_results(args.files)
    print(f"Loaded {len(results)} result(s) from {len(args.files)} pattern(s).")

    if args.latex:
        print_latex(results)
    else:
        print_table(results, group_by=args.group_by)


if __name__ == "__main__":
    main()
