"""
compare.py  –  Load all experiment results and print a ranked comparison table.

Usage:
    python compare.py                       # rank by best fitness (default)
    python compare.py --sort duration       # sort by training time
    python compare.py --sort name           # sort alphabetically
    python compare.py --curves              # also print ASCII training curves
    python compare.py --detail default_baseline   # full detail for one experiment
"""
import glob
import json
import os
import sys
from pathlib import Path

RESULTS_DIR = "results"


# ── loaders ───────────────────────────────────────

def load_results(results_dir: str = RESULTS_DIR) -> list:
    """Load all *.json result files.  Returns list sorted by best_fitness desc."""
    paths = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    results = []
    for p in paths:
        try:
            with open(p) as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"  [warn] Could not read {p}: {e}")
    return results


def _sort_key(r: dict, sort_by: str):
    if sort_by == "duration":
        return r.get("duration_seconds", 0)
    if sort_by == "name":
        return r["experiment"]["name"]
    return -r["final_metrics"]["best_fitness"]   # default: best fitness descending


# ── main table ────────────────────────────────────

COL = {
    "rank":      4,
    "name":      26,
    "arch":      9,
    "fitness":   9,
    "gens":      5,
    "cnn_w":     6,
    "score_w":   7,
    "w_dist":    6,
    "w_open":    6,
    "duration":  9,
    "desc":      35,
}

def _row(*cols):
    widths = list(COL.values())
    parts  = [str(c).ljust(w) for c, w in zip(cols, widths)]
    return "  ".join(parts)


def print_table(results: list, sort_by: str = "best_fitness") -> None:
    if not results:
        print("No results found.")
        return

    results = sorted(results, key=lambda r: _sort_key(r, sort_by))

    header = _row("Rank", "Name", "Arch", "Fitness", "Gens",
                  "cnn_w", "score_w", "w_dist", "w_open",
                  "Duration", "Description")
    sep = "-" * len(header)

    print()
    print(header)
    print(sep)

    for rank, r in enumerate(results, 1):
        exp  = r["experiment"]
        fm   = r["final_metrics"]
        row  = _row(
            rank,
            exp["name"],
            exp["arch_name"],
            f"{fm['best_fitness']:.2f}",
            fm["final_generation"],
            exp["cnn_w"],
            exp["score_w"],
            exp["w_dist"],
            exp["w_open"],
            f"{r['duration_seconds']:.1f}s",
            exp["description"][:35],
        )
        print(row)

    print(sep)
    print(f"  {len(results)} experiments  |  sorted by: {sort_by}")
    print()


# ── ASCII training curves ─────────────────────────

def print_curves(results: list, width: int = 50) -> None:
    """
    Print one ASCII curve per experiment showing fitness over generations.
    """
    print("\n  Training curves  (each * = one checkpoint)\n")

    for r in results:
        name   = r["experiment"]["name"]
        curve  = r["training_curve"]
        if len(curve) < 2:
            continue

        fits = [p["best_fitness"] for p in curve]
        lo   = min(fits)
        hi   = max(fits)
        span = hi - lo or 1.0

        print(f"  {name}")
        print(f"  min={lo:.1f}  max={hi:.1f}")
        for p in curve:
            gen = p["generation"]
            val = p["best_fitness"]
            bar = int((val - lo) / span * width)
            print(f"  {gen:5d} | {'#' * bar}")
        print()


# ── detail view for one experiment ───────────────

def print_detail(results: list, name: str) -> None:
    """Print full config + metrics + training curve for a single experiment."""
    matches = [r for r in results if r["experiment"]["name"] == name]
    if not matches:
        print(f"No result found for '{name}'")
        return
    r   = matches[0]
    exp = r["experiment"]
    fm  = r["final_metrics"]

    print(f"\n  {'='*55}")
    print(f"  Experiment : {exp['name']}")
    print(f"  {'='*55}")
    print(f"  Description  : {exp['description']}")
    print(f"  Timestamp    : {r['timestamp']}")
    print(f"  Duration     : {r['duration_seconds']:.1f}s")
    print()
    print("  -- Architecture --")
    print(f"    arch_name  : {exp['arch_name']}")
    print()
    print("  -- Decision weights --")
    print(f"    cnn_w      : {exp['cnn_w']}")
    print(f"    score_w    : {exp['score_w']}")
    print(f"    w_dist     : {exp['w_dist']}")
    print(f"    w_open     : {exp['w_open']}")
    print()
    print("  -- Training params --")
    print(f"    generations      : {exp['generations']}")
    print(f"    sigma            : {exp['sigma']}")
    print(f"    episodes_per_eval: {exp['episodes_per_eval']}")
    print(f"    max_steps        : {exp['max_steps']}")
    print()
    print("  -- Final metrics --")
    print(f"    best_fitness     : {fm['best_fitness']:.4f}")
    print(f"    total_episodes   : {fm['total_episodes_approx']}")
    print()
    print("  -- Training curve --")
    curve = r["training_curve"]
    fits  = [p["best_fitness"] for p in curve]
    lo, hi = min(fits), max(fits)
    span  = hi - lo or 1.0
    width = 40
    print(f"  {'gen':>5}  {'fitness':>8}  chart")
    print(f"  {'---':>5}  {'-------':>8}  {'-'*width}")
    for p in curve:
        bar = int((p["best_fitness"] - lo) / span * width)
        print(f"  {p['generation']:5d}  {p['best_fitness']:8.3f}  {'#' * bar}")
    print()


# ── summary stats ─────────────────────────────────

def print_summary(results: list) -> None:
    """Print aggregate stats across all experiments."""
    if not results:
        return
    fits     = [r["final_metrics"]["best_fitness"] for r in results]
    durations = [r["duration_seconds"] for r in results]
    best_r   = results[0]

    print(f"  Total experiments  : {len(results)}")
    print(f"  Best fitness       : {max(fits):.2f}  ({best_r['experiment']['name']})")
    print(f"  Worst fitness      : {min(fits):.2f}")
    print(f"  Avg fitness        : {sum(fits)/len(fits):.2f}")
    print(f"  Total train time   : {sum(durations)/60:.1f} min")
    print()


# ── CLI ───────────────────────────────────────────

def main():
    args     = sys.argv[1:]
    sort_by  = "best_fitness"
    detail   = None
    curves   = "--curves" in args

    if "--sort" in args:
        idx = args.index("--sort")
        if idx + 1 < len(args):
            sort_by = args[idx + 1]

    if "--detail" in args:
        idx = args.index("--detail")
        if idx + 1 < len(args):
            detail = args[idx + 1]

    results = load_results()

    if not results:
        print(f"\n  No results found in '{RESULTS_DIR}/'.")
        print("  Run  python experiment.py  first to generate results.\n")
        return

    if detail:
        print_detail(results, detail)
        return

    results_sorted = sorted(results, key=lambda r: _sort_key(r, sort_by))
    print_summary(results_sorted)
    print_table(results_sorted, sort_by)

    if curves:
        print_curves(results_sorted)


if __name__ == "__main__":
    main()
