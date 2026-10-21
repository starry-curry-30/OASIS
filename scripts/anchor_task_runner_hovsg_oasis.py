#!/usr/bin/env python3
from __future__ import annotations
"""
CLIP-based anchor task runner (no VLM).

Workflow:
1. Use CLIP to directly match task description against all objects in the map.
2. Return top-5 candidates and save to predictions.json.
3. Use eval_iou.py to evaluate accuracy.

Usage:
    python scripts/anchor_task_runner_hovsg_dualmap.py --tasks-file tasks.txt
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def dualmap_root() -> Path:
    """Root of the DualMap repo: env DUALMAP_ROOT, else ../DualMap sibling to OASIS."""
    env = os.environ.get("DUALMAP_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (project_root.parent / "DualMap").resolve()


from omegaconf import OmegaConf
import numpy as np
import torch
import open_clip

from utils.object import BaseObject


def load_anchor_file(path: Path) -> List[Dict]:
    """anchor"""
    if not path.exists():
        raise FileNotFoundError(f"Anchor file not found: {path}")
    anchors: List[Dict] = []
    current = None
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Anchor id:"):
            if current is not None:
                anchors.append(current)
            current = {"id": line.replace("Anchor id:", "").strip(), "category": "", "attached_objects": [], "nearby_objects": []}
        elif line.startswith("Category:") and current is not None:
            current["category"] = line.replace("Category:", "").strip()
        elif line == "Attached objects:" and current is not None:
            i += 1
            while i < len(lines):
                obj_line = lines[i].strip()
                if not obj_line or obj_line.startswith("Anchor id:") or obj_line == "Nearby objects:" or obj_line == "Others:":
                    i -= 1
                    break
                if obj_line == "(none)":
                    i += 1
                    continue
                if " x " in obj_line:
                    parts = obj_line.split(" x ", 1)
                    if len(parts) == 2:
                        try:
                            count_part = parts[1].strip()
                            count_str = count_part.split('(')[0].strip()
                            count = int(count_str)
                            current["attached_objects"].append({
                                "category": parts[0].strip(), 
                                "count": count,
                            })
                        except ValueError:
                            pass
                i += 1
        elif line == "Nearby objects:" and current is not None:
            i += 1
            while i < len(lines):
                obj_line = lines[i].strip()
                if not obj_line or obj_line.startswith("Anchor id:") or obj_line == "Others:":
                    i -= 1
                    break
                if obj_line == "(none)":
                    i += 1
                    continue
                if " x " in obj_line:
                    parts = obj_line.split(" x ", 1)
                    if len(parts) == 2:
                        try:
                            count_part = parts[1].strip()
                            count_str = count_part.split('(')[0].strip()
                            count = int(count_str)
                            current["nearby_objects"].append({
                                "category": parts[0].strip(), 
                                "count": count,
                            })
                        except ValueError:
                            pass
                i += 1
        i += 1
    if current is not None:
        anchors.append(current)
    return anchors


def load_map_from_disk(map_dir: str) -> List[BaseObject]:
    """"""
    if not os.path.exists(map_dir):
        print(f"[Runner] Map directory not found: {map_dir}", file=sys.stderr)
        return []
    
    obj_map = []
    pkl_files = [f for f in os.listdir(map_dir) if f.endswith(".pkl")]
    
    if not pkl_files:
        print(f"[Runner] No .pkl files found in {map_dir}", file=sys.stderr)
        return []
    
    print(f"[Runner] Loading {len(pkl_files)} objects from map...")
    for pkl_file in pkl_files:
        try:
            obj_path = os.path.join(map_dir, pkl_file)
            obj = BaseObject.load_from_disk(obj_path)
            if hasattr(obj, 'clip_ft') and obj.clip_ft.size > 0:
                obj_map.append(obj)
        except Exception as e:
            print(f"[Runner] Failed to load {pkl_file}: {e}", file=sys.stderr)
            continue
    
    print(f"[Runner] Successfully loaded {len(obj_map)} objects with CLIP features")
    return obj_map


def find_objects_in_map_by_clip_description(
    clip_description: str,
    obj_map: List[BaseObject],
    clip_model,
    clip_tokenizer,
    device: str = "cuda",
    similarity_threshold: float = 0.0,
) -> List[Tuple[BaseObject, float]]:
    """CLIP"""
    if not obj_map:
        return []
    
    text_tokens = clip_tokenizer([clip_description]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features.float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    matches = []
    for obj in obj_map:
        if not hasattr(obj, 'clip_ft') or obj.clip_ft.size == 0:
            continue
        
        obj_feat = torch.from_numpy(obj.clip_ft).float().to(device).unsqueeze(0)
        obj_feat = obj_feat / obj_feat.norm(dim=-1, keepdim=True)
        
        similarity = (text_features @ obj_feat.T).item()
        
        if similarity >= similarity_threshold:
            matches.append((obj, similarity))
    
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def load_clip(cfg_path: Optional[str] = None):
    """CLIP"""
    if cfg_path is None:
        cfg_path = str(dualmap_root() / "config" / "system_config.yaml")
    cfg = OmegaConf.load(cfg_path)
    device = cfg.get("device", "cuda")
    
    clip_model_name = cfg.clip.model_name
    clip_pretrained = cfg.clip.pretrained
    print(f"[Runner] Loading CLIP model: {clip_model_name} with pretrained '{clip_pretrained}'")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    if "MobileCLIP" in clip_model_name:
        from mobileclip.modules.common.mobileone import reparameterize_model
        clip_model = reparameterize_model(clip_model)
    
    clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
    
    return clip_model, clip_tokenizer, device


def process_single_task(
    task: str,
    obj_map: List[BaseObject],
    clip_model,
    clip_tokenizer,
    device: str = "cuda",
    task_index: int = 0,
) -> Dict:
    """Run one task: CLIP retrieval, top-5 candidates, timing stats."""
    import time
    task_start_time = time.time()
    
    stats = {
        "task": task,
        "task_index": task_index,
        "duration": 0.0,
        "success": False,
        "top_5_candidates": []
    }
    
    print(f"\n{'='*80}")
    print(f"[Runner] Processing task {task_index}: {task}")
    print(f"{'='*80}")
    
    # CLIP
    matches = find_objects_in_map_by_clip_description(
        task, obj_map, clip_model, clip_tokenizer, device, similarity_threshold=0.0
    )
    
    top_5 = matches[:5]
    
    for obj, score in top_5:
        candidate_info = {
            "uid": str(getattr(obj, "uid", "")),
            "score": float(score),
            "class_id": getattr(obj, "class_id", -1),
            "class_name": "unknown"
        }
        
        bbox = getattr(obj, "bbox", None)
        if bbox:
            candidate_info["bbox"] = {
                "center": bbox.get_center().tolist(),
                "sizes": bbox.get_extent().tolist()
            }
        
        stats["top_5_candidates"].append(candidate_info)
    
    if top_5:
        best_obj, best_score = top_5[0]
        print(f"[Runner] Best match: UID {getattr(best_obj, 'uid', 'unknown')} (score {best_score:.4f})")
        stats["success"] = True
    
    stats["duration"] = time.time() - task_start_time
    return stats


def main() -> int:
    # 
    dm = dualmap_root()
    base_cfg = OmegaConf.load(str(dm / "config" / "base_config.yaml"))
    system_cfg = OmegaConf.load(str(dm / "config" / "system_config.yaml"))
    merged_cfg = OmegaConf.merge(base_cfg, system_cfg)
    
    BaseObject.initialize_config(merged_cfg)
    
    # anchor key
    if "representative_frames_path" in merged_cfg:
        anchor_key_path = os.path.join(merged_cfg.representative_frames_path, "anchor_key")
    else:
        anchor_key_path = str(
            dm / "dataset" / "HM3D_hovsg" / "00824-Dd4bFSTQ8gi" / "representative_frames" / "anchor_key"
        )
    
    default_anchor_file = os.path.join(anchor_key_path, "anchor_attached_objects.txt")
    
    parser = argparse.ArgumentParser(description="CLIP-based anchor task runner")
    parser.add_argument("task", nargs="?", help="User task (or use --tasks-file for batch)")
    parser.add_argument("--tasks-file", type=Path, help="File containing tasks (one per line)")
    parser.add_argument("--anchor-file", type=Path, default=Path(default_anchor_file), help="Path to anchor file")
    args = parser.parse_args()
    
    # 
    tasks = []
    if args.tasks_file:
        if not args.tasks_file.exists():
            print(f"Error: Tasks file not found: {args.tasks_file}", file=sys.stderr)
            return 1
        with args.tasks_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ")" in line and line[0].isdigit():
                    idx = line.find(")")
                    if idx >= 0:
                        task = line[idx + 1:].strip()
                        if task:
                            tasks.append(task)
                else:
                    tasks.append(line)
    elif args.task:
        tasks = [args.task]
    else:
        print("Error: Either provide a task or use --tasks-file", file=sys.stderr)
        parser.print_help()
        return 1
    
    if not tasks:
        print("Error: No tasks found", file=sys.stderr)
        return 1
    
    print(f"[Runner] Processing {len(tasks)} task(s)")
    
    # 
    map_dir = merged_cfg.map_save_path
    print(f"[Runner] Loading map from: {map_dir}")
    obj_map = load_map_from_disk(map_dir)
    
    if not obj_map:
        print("[Runner] ERROR: No objects loaded from map.", file=sys.stderr)
        return 1
    
    # CLIP
    print("[Runner] Loading CLIP model...")
    try:
        clip_model, clip_tokenizer, device = load_clip()
        print("[Runner] CLIP model loaded successfully.")
    except Exception as e:
        print(f"[Runner] Failed to load CLIP model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    # Output dir for eval_iou.py compatibility
    output_dir = dm / "output" / "query_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 
    all_stats = []
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    predictions_path = output_dir / "predictions.json"
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'#'*80}")
        print(f"# Task {i}/{len(tasks)}")
        print(f"{'#'*80}")
        
        stats = process_single_task(
            task,
            obj_map,
            clip_model,
            clip_tokenizer,
            device,
            task_index=i,
        )
        
        all_stats.append(stats)
        
        # 
        try:
            with open(predictions_path, "w") as f:
                json.dump(all_stats, f, indent=4, cls=NumpyEncoder)
        except Exception as e:
            print(f"[Runner] Failed to save predictions.json after task {i}: {e}", file=sys.stderr)
    
    # 
    try:
        with open(predictions_path, "w") as f:
            json.dump(all_stats, f, indent=4, cls=NumpyEncoder)
        print(f"[Runner] Saved predictions to: {predictions_path}")
    except Exception as e:
        print(f"[Runner] Failed to save predictions.json: {e}", file=sys.stderr)
    
    print(f"\n[Runner] Completed processing {len(all_stats)} tasks")
    print(f"[Runner] Use eval_iou.py to evaluate accuracy")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

