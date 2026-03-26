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
        expanded = glob.glob(p)
        if not expanded:
            # If it's a directory, glob for JSON files inside
            from pathlib import Path
            pp = Path(p)
            if pp.is_dir():
                expanded = [str(f) for f in sorted(pp.glob("**/*.json"))]
        for f in expanded:
            from pathlib import Path
            if Path(f).is_dir():
                # Recursively find JSON files in directory
                for jf in sorted(Path(f).glob("**/*.json")):
                    try:
                        with open(jf) as fp:
                            d = json.load(fp)
                        d["_file"] = str(jf)
                        results.append(d)
                    except (json.JSONDecodeError, KeyError):
                        continue
            else:
                try:
                    with open(f) as fp:
                        d = json.load(fp)
                    d["_file"] = f
                    results.append(d)
                except (json.JSONDecodeError, KeyError):
                    continue
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


def print_markdown(results: list[dict]):
    """Print a GitHub-flavored Markdown leaderboard table."""
    groups = defaultdict(list)
    for r in results:
        groups[r.get("method", "?")].append(r)

    # Collect all datasets
    all_ds = sorted({ds for r in results for ds in r.get("tasks", [{}])[-1].get("per_dataset", {}).keys()})

    rows = []
    for method, rs in sorted(groups.items()):
        oa  = sum(r.get("final_oa",    r.get("oa_mean",    0)) for r in rs) / len(rs)
        aa  = sum(r.get("final_aa",    r.get("aa_mean",    0)) for r in rs) / len(rs)
        kap = sum(r.get("final_kappa", r.get("kappa_mean", 0)) for r in rs) / len(rs)
        bwt = sum(r.get("bwt",         r.get("bwt_mean",   0)) for r in rs) / len(rs)
        fwt = sum(r.get("fwt",         r.get("fwt_mean",   0)) for r in rs) / len(rs)
        n   = len(rs)

        # Std if multiple seeds
        oa_std = 0
        if n > 1:
            import statistics
            oa_std = statistics.stdev(r.get("final_oa", r.get("oa_mean", 0)) for r in rs)

        per_ds = {}
        for r in rs:
            tasks = r.get("tasks", [])
            if tasks:
                for ds, v in tasks[-1].get("per_dataset", {}).items():
                    per_ds.setdefault(ds, []).append(v)
        per_ds_avg = {ds: sum(vs)/len(vs) for ds, vs in per_ds.items()}

        rows.append({
            "method": method, "oa": oa, "oa_std": oa_std,
            "aa": aa, "kappa": kap, "bwt": bwt, "fwt": fwt,
            "per_ds": per_ds_avg, "n": n,
            "protocol": rs[0].get("protocol", "?"),
        })

    rows.sort(key=lambda r: r["oa"], reverse=True)

    # Build markdown table
    protocol = rows[0]["protocol"] if rows else "?"
    ds_cols = all_ds if len(all_ds) <= 5 else []

    print(f"\n## Leaderboard — Protocol {protocol}\n")

    # Header
    header = "| Rank | Method | OA (%) | AA (%) | BWT (pp) | FWT (%) |"
    separator = "|:----:|:-------|-------:|-------:|---------:|--------:|"
    for ds in ds_cols:
        header += f" {ds} |"
        separator += " ------:|"

    print(header)
    print(separator)

    # Rows
    for rank, r in enumerate(rows, 1):
        oa_str = f"{r['oa']*100:.2f}"
        if r["oa_std"] > 0:
            oa_str += f" +/- {r['oa_std']*100:.2f}"

        line = (f"| {rank} | **{r['method']}** | {oa_str} | "
                f"{r['aa']*100:.2f} | {r['bwt']*100:.2f} | {r['fwt']*100:.2f} |")
        for ds in ds_cols:
            v = r["per_ds"].get(ds)
            line += f" {v*100:.1f} |" if v is not None else " - |"
        print(line)

    print(f"\n*{sum(r['n'] for r in rows)} runs across {len(rows)} methods.*\n")


def generate_leaderboard_file(results_dir: str, output: str = "LEADERBOARD.md"):
    """Scan results directory and generate a leaderboard markdown file.

    Usage:
        python benchmark/compare.py results/ --leaderboard --output LEADERBOARD.md
    """
    from pathlib import Path
    import sys
    from io import StringIO

    results_dir = Path(results_dir)
    all_results = []
    for f in sorted(results_dir.glob("**/*.json")):
        try:
            with open(f) as fp:
                d = json.load(fp)
            d["_file"] = str(f)
            all_results.append(d)
        except Exception:
            continue

    if not all_results:
        print(f"No JSON results found in {results_dir}")
        return

    # Group by protocol
    by_proto = defaultdict(list)
    for r in all_results:
        proto = r.get("protocol", "unknown")
        by_proto[proto].append(r)

    # Capture markdown output
    old_stdout = sys.stdout
    buf = StringIO()
    sys.stdout = buf

    print("# RS-CIL-Bench Leaderboard\n")
    print("Auto-generated from experiment results.\n")

    for proto in sorted(by_proto):
        print_markdown(by_proto[proto])

    sys.stdout = old_stdout
    content = buf.getvalue()

    out_path = Path(output)
    out_path.write_text(content)
    print(f"Leaderboard saved → {out_path}")
    print(content)


def main():
    p = argparse.ArgumentParser(description="RS-CIL results comparison tool")
    p.add_argument("files", nargs="+", help="JSON result files or directory (glob patterns ok)")
    p.add_argument("--latex", action="store_true", help="Output LaTeX table")
    p.add_argument("--markdown", action="store_true", help="Output GitHub Markdown leaderboard")
    p.add_argument("--leaderboard", action="store_true",
                   help="Generate leaderboard file from results directory")
    p.add_argument("--output", default="LEADERBOARD.md",
                   help="Output file for --leaderboard mode")
    p.add_argument("--group-by", default="method", choices=["method", "protocol", "seed"],
                   help="Column to group results by")
    args = p.parse_args()

    if args.leaderboard:
        generate_leaderboard_file(args.files[0], args.output)
        return

    results = load_results(args.files)
    print(f"Loaded {len(results)} result(s) from {len(args.files)} pattern(s).")

    if args.latex:
        print_latex(results)
    elif args.markdown:
        print_markdown(results)
    else:
        print_table(results, group_by=args.group_by)


if __name__ == "__main__":
    main()
