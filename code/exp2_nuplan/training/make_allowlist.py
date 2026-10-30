"""Common training samples of several nuPlan feature caches, and their log-level train/val split.

A nuPlan cache stores one folder per sample: <cache>/<log_name>/<scenario_type>/<token>/<feature>.gz. A sample is
usable for a model when its folder holds all the model's features and targets. This script writes the tokens usable
by ALL given caches (so the three baselines train and validate on exactly the same samples) and a JSON summary with
the counts per cache and per split (same hash rule as LogHashSplitter).

  python make_allowlist.py --out tokens.txt --summary split.json \
      --cache mlp=/data/exp2/baselines/cache/mlp:agents,trajectory \
      --cache vector=/data/exp2/baselines/cache/vector:vector_map,agents,trajectory \
      --cache raster=/data/exp2/baselines/cache/raster:raster,trajectory
"""
import argparse
import json
import os
import sys
from typing import Dict, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from exp2_nuplan.training.splitter import is_val_log  # noqa: E402



def scan(cache: str, required: Set[str]) -> Dict[str, Tuple[str, str]]:
    """token -> (log_name, scenario_type) for every sample folder that holds all required features."""
    out: Dict[str, Tuple[str, str]] = {}
    for log in os.scandir(cache):
        if not log.is_dir():
            continue
        for stype in os.scandir(log.path):
            if not stype.is_dir():
                continue
            for tok in os.scandir(stype.path):
                if not tok.is_dir():
                    continue
                names = {n[:-3] for n in os.listdir(tok.path) if n.endswith(".gz")}
                if required <= names:
                    out[tok.name] = (log.name, stype.name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="append", required=True, help="name=path:feature1,feature2,...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--salt", default="exp2-train-pittsburgh")
    ap.add_argument("--forbidden-logs-dir", action="append", default=[],
                    help="folder(s) of DBs that must not appear in the training pool (e.g. splits/val, splits/test)")
    a = ap.parse_args()

    per_cache: Dict[str, Dict[str, Tuple[str, str]]] = {}
    for spec in a.cache:
        name, rest = spec.split("=", 1)
        path, feats = rest.rsplit(":", 1)
        per_cache[name] = scan(path, set(feats.split(",")))
        print(f"{name}: {len(per_cache[name])} usable samples in {path}", flush=True)

    common = set.intersection(*(set(d) for d in per_cache.values()))
    first = next(iter(per_cache.values()))
    rows = sorted((first[t][0], first[t][1], t) for t in common)

    forbidden = set()
    for d in a.forbidden_logs_dir:
        if os.path.isdir(d):
            forbidden |= {n[:-3] for n in os.listdir(d) if n.endswith(".db")}
    leaked = sorted({log for log, _, _ in rows if log in forbidden})
    if leaked:
        raise SystemExit(f"ERROR: {len(leaked)} training-pool logs are val/test logs, e.g. {leaked[:3]}")

    with open(a.out, "w") as f:
        f.write("# tokens usable by all caches: " + ", ".join(per_cache) + "\n")
        for _, _, tok in rows:
            f.write(tok + "\n")

    val = [r for r in rows if is_val_log(r[0], a.val_fraction, a.salt)]
    train = [r for r in rows if not is_val_log(r[0], a.val_fraction, a.salt)]
    types: Dict[str, int] = {}
    for _, stype, _ in rows:
        types[stype] = types.get(stype, 0) + 1
    summary = {
        "caches": {k: len(v) for k, v in per_cache.items()},
        "common_samples": len(rows),
        "train_samples": len(train), "val_samples": len(val),
        "train_logs": len({r[0] for r in train}), "val_logs": len({r[0] for r in val}),
        "val_fraction_of_logs": a.val_fraction, "salt": a.salt,
        "checked_against_forbidden_dirs": a.forbidden_logs_dir, "forbidden_logs_found": len(leaked),
        "scenario_types": dict(sorted(types.items(), key=lambda kv: -kv[1])),
        "allowlist": os.path.abspath(a.out),
    }
    with open(a.summary, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "scenario_types"}), flush=True)


if __name__ == "__main__":
    main()
