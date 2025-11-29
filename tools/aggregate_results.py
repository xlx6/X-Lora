import json
import os
from glob import glob
from typing import List, Dict


def collect_summaries(root: str) -> List[Dict]:
    paths = glob(os.path.join(root, "**", "run_summary.json"), recursive=True)
    runs = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                runs.append(json.load(f))
        except Exception:
            pass
    return runs


def to_csv(runs: List[Dict], out_csv: str):
    import csv
    keys = [
        "method", "rank", "trainable_params",
        "orthogonal_lambda", "structure_type", "svd_auto_rank", "svd_energy_threshold",
        "metrics.eval_loss", "metrics.eval_accuracy", "metrics.eval_f1",
        "output_dir",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in runs:
            def get(m, k, default=""):
                cur = m
                for part in k.split('.'):
                    if isinstance(cur, dict) and part in cur:
                        cur = cur[part]
                    else:
                        return default
                return cur
            row = [get(r, k, "") for k in keys]
            w.writerow(row)
    print(f"Saved {len(runs)} rows to {out_csv}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs")
    ap.add_argument("--out", type=str, default="outputs/summary.csv")
    args = ap.parse_args()
    runs = collect_summaries(args.root)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    to_csv(runs, args.out)

