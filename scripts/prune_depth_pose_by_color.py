#!/usr/bin/env python3
"""
Remove depth/ pose files whose basename stem no longer exists under color/.

Usage:
  # Dry-run (list only, default)
  python scripts/prune_depth_pose_by_color.py --root /path/to/data

  # Actually delete
  python scripts/prune_depth_pose_by_color.py --root /path/to/data --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def collect_stems(color_dir: Path) -> set[str]:
    stems: set[str] = set()
    for p in color_dir.iterdir():
        if p.is_file():
            stems.add(p.stem)
    return stems


def prune_subdir(subdir: Path, keep_stems: set[str], apply: bool) -> list[Path]:
    to_remove: list[Path] = []
    if not subdir.is_dir():
        return to_remove
    for p in subdir.iterdir():
        if not p.is_file():
            continue
        if p.stem not in keep_stems:
            to_remove.append(p)
    for p in sorted(to_remove):
        if apply:
            p.unlink()
        print(f"{'DEL' if apply else 'DRY'} {p}")
    return to_remove


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prune depth/pose to match color stems"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/path/to/data"),
        help="Data root containing color/, depth/, pose/",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete files; omit for dry-run only",
    )
    args = parser.parse_args()

    root: Path = args.root.expanduser().resolve()
    color_dir = root / "color"
    depth_dir = root / "depth"
    pose_dir = root / "pose"

    if not color_dir.is_dir():
        print(f"error: missing or not a directory: {color_dir}", file=sys.stderr)
        return 1

    keep = collect_stems(color_dir)
    if not keep:
        print(
            f"warning: no files in {color_dir}; depth/pose would all be removed with --apply",
            file=sys.stderr,
        )

    removed_depth = prune_subdir(depth_dir, keep, args.apply)
    removed_pose = prune_subdir(pose_dir, keep, args.apply)

    n = len(removed_depth) + len(removed_pose)
    if not args.apply:
        print(f"\n{n} file(s) would be deleted. Re-run with --apply to delete.")
    else:
        print(f"\nDeleted {n} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
