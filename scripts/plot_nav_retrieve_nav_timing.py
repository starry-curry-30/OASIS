#!/usr/bin/env python3
"""
Print a table from nav_retrieve_nav_timing.jsonl; optionally plot the last row as a bar chart.

Examples:
  python scripts/plot_nav_retrieve_nav_timing.py /path/to/output/nav_retrieve_nav_timing.jsonl
  python scripts/plot_nav_retrieve_nav_timing.py /path/to/output/nav_retrieve_nav_timing.jsonl --plot out.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


LABELS = [
    ("local_scene_graph_ms", "local scene graph (+I/O+CLIP)"),
    ("llm_anchor_proposal_ms", "LLM anchor proposal"),
    ("vlm_verification_ms", "VLM verification"),
    ("navigation_path_planning_ms", "navigation path planning"),
]


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("(empty)", file=sys.stderr)
        return
    headers = ["#", "inquiry"] + [lbl for _, lbl in LABELS] + ["retrieve+plan (ms)", "wall (ms)"]
    col_keys = [None, None] + [k for k, _ in LABELS] + ["retrieve_plus_plan_ms", "subprocess_wall_ms"]
    widths = [len(h) for h in headers]
    str_rows: List[List[str]] = []
    for i, r in enumerate(rows):
        cells: List[str] = []
        for j, key in enumerate(col_keys):
            if j == 0:
                cells.append(str(i + 1))
            elif j == 1:
                cells.append(str(r.get("inquiry", ""))[:48])
            else:
                v = r.get(key)
                if v is None:
                    cells.append("-")
                elif isinstance(v, float):
                    cells.append(f"{v:.1f}")
                else:
                    cells.append(str(v))
        str_rows.append(cells)
        for j, c in enumerate(cells):
            widths[j] = max(widths[j], len(c))
    def _row(cells: List[str]) -> str:
        return " | ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    print(_row(list(headers)))
    print("-+-".join("-" * w for w in widths))
    for cells in str_rows:
        print(_row(cells))


def plot_last_bar(rows: List[Dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    r = rows[-1]
    keys = [k for k, _ in LABELS]
    vals = [float(r.get(k, 0.0) or 0.0) for k in keys]
    labels_en = [lbl for _, lbl in LABELS]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels_en, vals, color=["#2ecc71", "#3498db", "#9b59b6", "#e67e22"])
    ax.set_ylabel("Time (ms)")
    ax.set_title("Last row — stage breakdown")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize nav_retrieve_nav_timing.jsonl")
    parser.add_argument("jsonl", type=Path, help="Path to nav_retrieve_nav_timing.jsonl")
    parser.add_argument("--plot", type=Path, default=None, help="Optional PNG output path")
    args = parser.parse_args()
    if not args.jsonl.is_file():
        print(f"File not found: {args.jsonl}", file=sys.stderr)
        return 1
    rows = load_rows(args.jsonl)
    print_table(rows)
    if args.plot is not None:
        try:
            plot_last_bar(rows, args.plot)
        except ImportError:
            print("matplotlib required for --plot", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
