#!/usr/bin/env python3
"""
Aggregate OASIS ablation JSONs: mean/std over scenes for 9 (Clustering × Selection)
modes — Instance Coverage and Frame Count.

Examples:
  # Default: output/map_results/oasis_ablation/*.json
  python scripts/aggregate_oasis_ablation_results.py

  # Custom input directory
  python scripts/aggregate_oasis_ablation_results.py --input_dir ./output/map_results/oasis_ablation

  # Restrict to listed scene ids
  python scripts/aggregate_oasis_ablation_results.py --scenes room_1 room_2 office_0 office_1 office_2 office_3 office_4
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_results_from_dir(input_dir: str, scene_ids: list = None) -> list:
    """Load <scene_id>.json from input_dir; optionally filter by scene_ids. Returns [(scene_id, results_list), ...]."""
    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        return []
    out = []
    if scene_ids is not None:
        for sid in scene_ids:
            p = os.path.join(input_dir, f"{sid}.json")
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                out.append((data.get("scene", sid), data.get("results", [])))
            else:
                print(f"[Aggregate] missing file: {p}", flush=True)
    else:
        for name in sorted(os.listdir(input_dir)):
            if not name.endswith(".json"):
                continue
            p = os.path.join(input_dir, name)
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            out.append((data.get("scene", name.replace(".json", "")), data.get("results", [])))
    return out


def aggregate(loaded: list) -> dict:
    """
    Aggregate over 9 (Clustering × Selection) combinations.
    Collects instance_coverage and frame_count per mode.
    Expects each results list to have 9 entries (Spatial+Global, Spatial+Local, ..., Hybrid+Adaptive).
    """
    if not loaded:
        return {}
    # Keys = (clustering, selection) from first scene
    first_results = loaded[0][1]
    keys = [(r["clustering"], r["selection"]) for r in first_results]
    n_scenes = len(loaded)
    by_key = {k: {"coverage": [], "frame_count": []} for k in keys}
    for scene_id, results in loaded:
        if len(results) != len(keys):
            print(f"[Aggregate] scene {scene_id}: len(results)={len(results)} != {len(keys)}, skipped.", flush=True)
            continue
        for i, r in enumerate(results):
            k = (r["clustering"], r["selection"])
            if k in by_key:
                by_key[k]["coverage"].append(r["instance_coverage"])
                by_key[k]["frame_count"].append(r["frame_count"])
    import math
    summary = []
    for k in keys:
        cov = by_key[k]["coverage"]
        fc = by_key[k]["frame_count"]
        n = len(cov)
        if n == 0:
            summary.append({
                "clustering": k[0],
                "selection": k[1],
                "instance_coverage_mean": 0.0,
                "instance_coverage_std": 0.0,
                "frame_count_mean": 0.0,
                "frame_count_std": 0.0,
                "num_scenes": 0,
            })
            continue
        mean_cov = sum(cov) / n
        mean_fc = sum(fc) / n
        std_cov = math.sqrt(sum((x - mean_cov) ** 2 for x in cov) / n) if n > 1 else 0.0
        std_fc = math.sqrt(sum((x - mean_fc) ** 2 for x in fc) / n) if n > 1 else 0.0
        summary.append({
            "clustering": k[0],
            "selection": k[1],
            "instance_coverage_mean": float(mean_cov),
            "instance_coverage_std": float(std_cov),
            "frame_count_mean": float(mean_fc),
            "frame_count_std": float(std_fc),
            "num_scenes": n,
        })
    return {"summary": summary, "num_scenes": n_scenes, "scenes": [s[0] for s in loaded]}


def main():
    parser = argparse.ArgumentParser(description="Aggregate OASIS ablation JSON results")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory of per-scene xxx.json files; default: output_path/oasis_ablation, e.g. ./output/map_results/oasis_ablation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Directory to write aggregated JSON; default: parent of input_dir",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Scene ids to include, e.g. room_1 room_2 office_0 ...; must exist as <id>.json under input_dir",
    )
    args = parser.parse_args()

    default_dir = os.path.join(PROJECT_ROOT, "output", "map_results", "oasis_ablation")
    input_dir = args.input_dir or default_dir
    output_path = args.output_path or os.path.dirname(input_dir)

    loaded = load_results_from_dir(input_dir, scene_ids=args.scenes)
    if not loaded:
        print(f"[Aggregate] No JSON files loaded from: {input_dir}", flush=True)
        if args.scenes:
            print(f"[Aggregate] Requested scenes: {args.scenes}", flush=True)
        sys.exit(1)

    print(f"[Aggregate] Loaded {len(loaded)} scene(s): {[s[0] for s in loaded]}", flush=True)
    agg = aggregate(loaded)
    if not agg:
        sys.exit(1)

    summary = agg["summary"]
    num_scenes = agg["num_scenes"]
    scenes = agg["scenes"]

    os.makedirs(output_path, exist_ok=True)
    out_json = os.path.join(output_path, "oasis_ablation_aggregated.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_scenes": num_scenes,
                "scenes": scenes,
                "summary": summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[Aggregate] Wrote: {out_json}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("OASIS Ablation — summary (Instance Coverage & Frame Count)", flush=True)
    print("=" * 90, flush=True)
    print(f"Scenes ({num_scenes}): {', '.join(scenes)}", flush=True)
    print("-" * 90, flush=True)
    print(
        f"{'Clustering':<14} {'Selection':<16} {'Coverage (mean±std)':<24} {'Frame Count (mean±std)':<24}",
        flush=True,
    )
    print("-" * 90, flush=True)
    for s in summary:
        cov_m, cov_s = s["instance_coverage_mean"], s["instance_coverage_std"]
        fc_m, fc_s = s["frame_count_mean"], s["frame_count_std"]
        cov_str = f"{cov_m:.2%} ± {cov_s:.2%}" if cov_s > 0 else f"{cov_m:.2%}"
        fc_str = f"{fc_m:.1f} ± {fc_s:.1f}" if fc_s > 0 else f"{fc_m:.1f}"
        print(f"{s['clustering']:<14} {s['selection']:<16} {cov_str:<24} {fc_str:<24}", flush=True)
    print("=" * 90 + "\n", flush=True)


if __name__ == "__main__":
    main()
