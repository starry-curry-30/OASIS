#!/usr/bin/env python3
"""
Convert anchor predictions to OpenLex3D-style layout (point cloud, masks, embeddings).

Inputs:
- predictions.json from anchor_task_runner.py
- Object map from .pkl
- layout.pcd (optional; merged from objects if missing)

Outputs under predictions_path/scene_name/:
- point_cloud.pcd: scene geometry
- index.npy: per-point instance IDs (n_points,)
- embeddings.npy: CLIP features (n_instances, D)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import open3d as o3d
from omegaconf import OmegaConf

# sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def dualmap_root() -> Path:
    """Use this checkout by default; retain the legacy root override."""
    env = os.environ.get("OASIS_ROOT") or os.environ.get("DUALMAP_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return project_root.resolve()


_dm_default = dualmap_root()

from utils.object import BaseObject


def load_predictions_json(predictions_path: str) -> List[Dict]:
    """ predictions.json """
    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def collect_unique_candidate_uids(predictions: List[Dict]) -> Set[str]:
    """ top_5_candidates  UID"""
    uids = set()
    for pred in predictions:
        top_5 = pred.get("top_5_candidates", [])
        for candidate in top_5:
            uid = candidate.get("uid", "")
            if uid:
                uids.add(str(uid))
    return uids


def load_objects_from_map(map_dir: str, target_uids: Set[str]) -> Dict[str, BaseObject]:
    """"""
    obj_dict = {}
    if not os.path.exists(map_dir):
        print(f"[Converter] Map directory not found: {map_dir}", file=sys.stderr)
        return obj_dict
    
    pkl_files = [f for f in os.listdir(map_dir) if f.endswith(".pkl")]
    print(f"[Converter] Loading objects from {len(pkl_files)} .pkl files...")
    
    loaded_count = 0
    for pkl_file in pkl_files:
        try:
            obj_path = os.path.join(map_dir, pkl_file)
            obj = BaseObject.load_from_disk(obj_path)
            obj_uid = str(getattr(obj, 'uid', ''))
            
            #  UID 
            if obj_uid in target_uids:
                if hasattr(obj, 'clip_ft') and obj.clip_ft.size > 0:
                    obj_dict[obj_uid] = obj
                    loaded_count += 1
        except Exception as e:
            print(f"[Converter] Failed to load {pkl_file}: {e}", file=sys.stderr)
            continue
    
    print(f"[Converter] Successfully loaded {loaded_count} objects from map")
    return obj_dict


def load_layout_pointcloud(cfg) -> Optional[o3d.geometry.PointCloud]:
    """"""
    layout_pcd_path = None
    
    #  preload_path
    if hasattr(cfg, 'preload_path') and cfg.preload_path:
        layout_pcd_path = os.path.join(cfg.preload_path, "layout.pcd")
        if os.path.exists(layout_pcd_path):
            print(f"[Converter] Using layout from preload_path: {layout_pcd_path}")
            return o3d.io.read_point_cloud(layout_pcd_path)
    
    #  map_save_path
    if hasattr(cfg, 'map_save_path') and cfg.map_save_path:
        layout_pcd_path = os.path.join(cfg.map_save_path, "layout.pcd")
        if os.path.exists(layout_pcd_path):
            print(f"[Converter] Using layout from map_save_path: {layout_pcd_path}")
            return o3d.io.read_point_cloud(layout_pcd_path)
    
    print(f"[Converter] Layout point cloud not found, will merge object point clouds")
    return None


def merge_object_pointclouds(obj_dict: Dict[str, BaseObject]) -> o3d.geometry.PointCloud:
    """"""
    merged_pcd = o3d.geometry.PointCloud()
    
    for uid, obj in obj_dict.items():
        obj_pcd = getattr(obj, 'pcd', None)
        if obj_pcd is not None and len(obj_pcd.points) > 0:
            merged_pcd += obj_pcd
    
    print(f"[Converter] Merged point cloud has {len(merged_pcd.points)} points")
    return merged_pcd


def create_instance_masks(
    scene_pcd: o3d.geometry.PointCloud,
    obj_dict: Dict[str, BaseObject],
    uid_to_instance_id: Dict[str, int]
) -> np.ndarray:
    """
    Build per-point instance IDs for the scene (nearest-neighbor to object geometry).

    Args:
        scene_pcd: Full scene point cloud
        obj_dict: {uid: BaseObject}
        uid_to_instance_id: UID -> instance id (1-based; 0 = background)

    Returns:
        Array shaped (n_points,) with per-point instance IDs (index.npy format).
    """
    scene_points = np.asarray(scene_pcd.points)
    n_points = len(scene_points)
    instance_indices = np.zeros(n_points, dtype=np.int32)  # 0 = background
    
    print(f"[Converter] Assigning instance IDs to {n_points} points...")
    
    # Nearest-neighbor assignment per object
    from sklearn.neighbors import NearestNeighbors
    
    assigned_count = 0
    for uid, instance_id in uid_to_instance_id.items():
        if uid not in obj_dict:
            continue
        
        obj = obj_dict[uid]
        obj_pcd = getattr(obj, 'pcd', None)
        
        if obj_pcd is None or len(obj_pcd.points) == 0:
            # Fallback: sample inside bbox
            bbox = getattr(obj, 'bbox', None)
            if bbox is not None:
                min_bound = bbox.get_min_bound()
                max_bound = bbox.get_max_bound()
                # Random samples inside bbox
                num_samples = 100
                bbox_points = np.random.uniform(min_bound, max_bound, (num_samples, 3))
            else:
                continue
        else:
            bbox_points = np.asarray(obj_pcd.points)
        
        if len(bbox_points) == 0:
            continue
        
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(bbox_points)
        distances, indices = nbrs.kneighbors(scene_points)
        distances = distances.ravel()
        
        threshold = 0.1  # meters
        mask = distances < threshold
        
        # Do not overwrite already labeled points
        mask = mask & (instance_indices == 0)
        instance_indices[mask] = instance_id
        assigned_count += np.sum(mask)
    
    print(f"[Converter] Assigned {assigned_count} points to instances (background: {n_points - assigned_count})")
    return instance_indices


def create_embeddings(
    obj_dict: Dict[str, BaseObject],
    uid_to_instance_id: Dict[str, int]
) -> np.ndarray:
    """
    Stack normalized CLIP features per instance row.

    Args:
        obj_dict: {uid: BaseObject}
        uid_to_instance_id: UID -> instance id

    Returns:
        Matrix (n_instances, D) for embeddings.npy
    """
    max_instance_id = max(uid_to_instance_id.values()) if uid_to_instance_id else 0
    n_instances = max_instance_id
    
    # Infer feature dim from first valid object
    feature_dim = None
    for uid, obj in obj_dict.items():
        if uid in uid_to_instance_id:
            clip_ft = getattr(obj, 'clip_ft', None)
            if clip_ft is not None and clip_ft.size > 0:
                feature_dim = clip_ft.shape[0] if clip_ft.ndim > 0 else len(clip_ft)
                break
    
    if feature_dim is None:
        raise ValueError("No valid CLIP features found in objects")
    
    print(f"[Converter] Creating embeddings matrix: ({n_instances}, {feature_dim})")
    
    embeddings = np.zeros((n_instances, feature_dim), dtype=np.float32)
    
    for uid, instance_id in uid_to_instance_id.items():
        if uid in obj_dict:
            obj = obj_dict[uid]
            clip_ft = getattr(obj, 'clip_ft', None)
            if clip_ft is not None and clip_ft.size > 0:
                if clip_ft.ndim > 1:
                    clip_ft = clip_ft.flatten()
                clip_ft = clip_ft / (np.linalg.norm(clip_ft) + 1e-8)
                embeddings[instance_id - 1] = clip_ft  # rows are 1-based ids; array is 0-based
    
    return embeddings


def convert_predictions(
    predictions_json_path: str,
    map_dir: str,
    output_path: str,
    scene_name: str,
    cfg,
    use_layout_pcd: bool = True
):
    """
    Main conversion: predictions + map -> OpenLex3D-style folder layout.

    Args:
        predictions_json_path: Path to predictions.json
        map_dir: Serialized map directory
        output_path: Base output directory
        scene_name: Subfolder name
        cfg: OmegaConf with paths
        use_layout_pcd: If True, try layout.pcd from cfg first
    """
    print(f"[Converter] Converting predictions from {predictions_json_path}")
    print(f"[Converter] Output directory: {output_path}/{scene_name}")
    
    predictions = load_predictions_json(predictions_json_path)
    print(f"[Converter] Loaded {len(predictions)} task predictions")
    
    target_uids = collect_unique_candidate_uids(predictions)
    print(f"[Converter] Found {len(target_uids)} unique candidate UIDs")
    
    if not target_uids:
        print("[Converter] ERROR: No candidate UIDs found in predictions", file=sys.stderr)
        return
    
    obj_dict = load_objects_from_map(map_dir, target_uids)
    
    if not obj_dict:
        print("[Converter] ERROR: No objects loaded from map", file=sys.stderr)
        return
    
    uid_to_instance_id = {uid: idx + 1 for idx, uid in enumerate(sorted(obj_dict.keys()))}
    print(f"[Converter] Created {len(uid_to_instance_id)} instance mappings")
    
    scene_pcd = None
    if use_layout_pcd:
        scene_pcd = load_layout_pointcloud(cfg)
    
    if scene_pcd is None or len(scene_pcd.points) == 0:
        print("[Converter] Merging object point clouds as scene point cloud...")
        scene_pcd = merge_object_pointclouds(obj_dict)
    
    if scene_pcd is None or len(scene_pcd.points) == 0:
        print("[Converter] ERROR: Failed to create scene point cloud", file=sys.stderr)
        return
    
    instance_indices = create_instance_masks(scene_pcd, obj_dict, uid_to_instance_id)
    
    embeddings = create_embeddings(obj_dict, uid_to_instance_id)
    
    output_dir = Path(output_path) / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 
    pcd_path = output_dir / "point_cloud.pcd"
    o3d.io.write_point_cloud(str(pcd_path), scene_pcd)
    print(f"[Converter] Saved point cloud to: {pcd_path}")
    
    # 
    index_path = output_dir / "index.npy"
    np.save(index_path, instance_indices)
    print(f"[Converter] Saved instance indices to: {index_path}")
    
    # 
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"[Converter] Saved embeddings to: {embeddings_path}")
    
    print(f"[Converter] Conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert anchor_task_runner predictions to OpenLex3D format"
    )
    parser.add_argument(
        "--predictions-json",
        type=str,
        required=True,
        help="Path to predictions.json from anchor_task_runner.py"
    )
    parser.add_argument(
        "--map-dir",
        type=str,
        required=True,
        help="Path to map directory containing .pkl files"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output directory path (will create scene_name subdirectory)"
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        required=True,
        help="Scene name (e.g., '00824-Dd4bFSTQ8gi')"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_dm_default / "config" / "base_config.yaml"),
        help="Path to base config file (default: DUALMAP_ROOT or ../DualMap)"
    )
    parser.add_argument(
        "--system-config",
        type=str,
        default=str(_dm_default / "config" / "system_config.yaml"),
        help="Path to system config file (default: DUALMAP_ROOT or ../DualMap)"
    )
    parser.add_argument(
        "--no-layout-pcd",
        action="store_true",
        help="Don't use layout.pcd, always merge object point clouds"
    )
    
    args = parser.parse_args()
    
    # 
    base_cfg = OmegaConf.load(args.config)
    system_cfg = OmegaConf.load(args.system_config)
    merged_cfg = OmegaConf.merge(base_cfg, system_cfg)
    
    # BaseObject
    BaseObject.initialize_config(merged_cfg)
    
    # 
    convert_predictions(
        args.predictions_json,
        args.map_dir,
        args.output_path,
        args.scene_name,
        merged_cfg,
        use_layout_pcd=not args.no_layout_pcd
    )


if __name__ == "__main__":
    main()











