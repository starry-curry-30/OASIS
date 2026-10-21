import json
import numpy as np
import os
import argparse
from pathlib import Path
from collections import defaultdict

from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import open3d as o3d

_REPO_ROOT = Path(__file__).resolve().parent.parent

def get_bbox_min_max(center, sizes):
    """
    Convert center, sizes to min_bound, max_bound
    center: [x, y, z]
    sizes: [w, h, d]
    """
    cx, cy, cz = center
    w, h, d = sizes
    min_bound = np.array([cx - w/2, cy - h/2, cz - d/2])
    max_bound = np.array([cx + w/2, cy + h/2, cz + d/2])
    return min_bound, max_bound

def compute_iou_3d(box1, box2):
    """
    Compute 3D IoU between two AABBs.
    box: {'center': [x,y,z], 'sizes': [w,h,d]}
    """
    min1, max1 = get_bbox_min_max(box1['center'], box1['sizes'])
    min2, max2 = get_bbox_min_max(box2['center'], box2['sizes'])

    # Intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(0, inter_max - inter_min)
    inter_vol = np.prod(inter_dims)

    # Volumes
    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)

    # Union
    union_vol = vol1 + vol2 - inter_vol

    if union_vol <= 0:
        return 0.0

    return inter_vol / union_vol

def load_gt_objects(path):
    """Load GT objects dict (id -> object dict) from scene_info.json"""
    with open(path, 'r') as f:
        data = json.load(f)
    
    gt_objects = {}
    
    # Helper
    def process_obj_list(obj_list):
        for obj in obj_list:
            if isinstance(obj, dict) and 'id' in obj:
                gt_objects[obj['id']] = obj
    
    if 'objects' in data:
        process_obj_list(data['objects'])
        
    if 'levels' in data:
        for level in data['levels']:
            if 'objects' in level:
                process_obj_list(level['objects'])
                
    return gt_objects

def visualize_task(task_id, task_name, gt_bboxes, pred_candidates, pcd_points, pcd_colors, output_dir):
    """
    Visualize GT and Predicted BBoxes on the point cloud (Top-down view).
    Assuming Y-up coordinate system (XZ plane is ground).
    gt_bboxes: list of dicts, each with 'center' and 'sizes'
    """
    if pcd_points is None:
        return

    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot point cloud (XZ plane)
    # Subsample for speed/clarity
    if len(pcd_points) > 50000:
        indices = np.random.choice(len(pcd_points), 50000, replace=False)
        pts = pcd_points[indices]
        clrs = pcd_colors[indices] if pcd_colors is not None else None
    else:
        pts = pcd_points
        clrs = pcd_colors

    # Plot points: X horizontal, Z vertical
    # Usually X is right, -Z is forward. Let's just plot X and Z.
    # Note: matplotlib scatter c argument takes RGB in [0,1] or color strings.
    if clrs is not None:
        ax.scatter(pts[:, 0], pts[:, 2], c=clrs, s=0.5, alpha=0.3)
    else:
        ax.scatter(pts[:, 0], pts[:, 2], c='gray', s=0.5, alpha=0.3)
    
    # Draw GT BBoxes (Green)
    if gt_bboxes:
        # Handle single bbox case if passed by mistake (though loop below handles list)
        if isinstance(gt_bboxes, dict):
            gt_bboxes = [gt_bboxes]
            
        for i, gt_bbox in enumerate(gt_bboxes):
            cx, cy, cz = gt_bbox['center']
            w, h, d = gt_bbox['sizes']
            # XZ plane projection
            # Rect: (x, z) width=w, height=d
            x0 = cx - w/2
            z0 = cz - d/2
            
            # Only label the first GT to avoid legend clutter
            label = 'GT' if i == 0 else None
            
            rect = patches.Rectangle((x0, z0), w, d, linewidth=2.5, edgecolor='lime', facecolor='none', label=label)
            ax.add_patch(rect)
            if i == 0:
                ax.text(x0, z0, "GT", color='lime', fontsize=10, fontweight='bold', ha='right', va='bottom')

    # Draw Pred BBoxes
    # Use colormap or specific colors
    colors = ['red', 'orange', 'gold', 'magenta', 'cyan']
    
    for i, cand in enumerate(pred_candidates):
        bbox = cand.get('bbox')
        if not bbox: continue
        
        cx, cy, cz = bbox['center']
        w, h, d = bbox['sizes']
        
        x0 = cx - w/2
        z0 = cz - d/2
        
        color = colors[i % len(colors)]
        label = f"P{i+1}"
        
        rect = patches.Rectangle((x0, z0), w, d, linewidth=1.5, edgecolor=color, facecolor='none', linestyle='--', label=f'Pred {i+1}')
        ax.add_patch(rect)
        
        # Add label near the box
        ax.text(x0, z0, label, color=color, fontsize=9, ha='left', va='top')

    ax.set_aspect('equal')
    ax.set_title(f"Task {task_id}: {task_name}", fontsize=12)
    
    # Create custom legend handles if needed, or rely on automatic
    # Only adding unique labels to legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Save
    safe_name = "".join(c if c.isalnum() else "_" for c in task_name[:30])
    out_path = os.path.join(output_dir, f"vis_{task_id:02d}_{safe_name}.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {out_path}")

def evaluate_recall_at_k(scene_info_path, prompts_path, predictions_path, pcd_path=None, vis_dir=None, iou_threshold=0.1, k=5):
    print(f"Loading Scene Info from {scene_info_path}")
    gt_objects_map = load_gt_objects(scene_info_path)
    
    print(f"Loading Prompts from {prompts_path}")
    with open(prompts_path, 'r') as f:
        prompts_data = json.load(f)
        
    print(f"Loading Predictions from {predictions_path}")
    with open(predictions_path, 'r') as f:
        preds_data = json.load(f)
        
    # Load PCD if visualization is requested
    pcd_points = None
    pcd_colors = None
    if pcd_path and vis_dir:
        print(f"Loading Point Cloud from {pcd_path}")
        if os.path.exists(pcd_path):
            pcd = o3d.io.read_point_cloud(pcd_path)
            if not pcd.is_empty():
                pcd_points = np.asarray(pcd.points)
                if pcd.has_colors():
                    pcd_colors = np.asarray(pcd.colors)
                print(f"Loaded {len(pcd_points)} points.")
            else:
                print("Point cloud is empty.")
        else:
            print(f"Point cloud file not found: {pcd_path}")
            
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)

    # Map predictions by task string for easy lookup
    preds_map = {p['task']: p for p in preds_data}
    
    success_count = 0
    total_count = 0
    
    print(f"\nEvaluating Recall@{k} with IoU threshold: {iou_threshold}")
    print("-" * 120)
    print(f"{'ID':<4} | {'Status':<8} | {'Max IoU':<8} | {'Task':<50} | {'GT Category'}")
    print("-" * 120)
    
    for i, prompt_item in enumerate(prompts_data, 1):
        task = prompt_item['prompt']
        
        # Support both 'gt_id' (single) and 'gt_ids' (multiple)
        gt_ids = prompt_item.get('gt_ids', [])
        if not gt_ids and 'gt_id' in prompt_item:
            gt_ids = [prompt_item['gt_id']]
            
        gt_category = prompt_item.get('gt_category', 'unknown')
        if not gt_category:
            gt_category = prompt_item.get('target_object', 'unknown')
        
        # Get All GT BBoxes for this task
        gt_bboxes = []
        for gid in gt_ids:
            gt_obj = gt_objects_map.get(gid)
            if gt_obj and gt_obj.get('aabb_center') and gt_obj.get('aabb_dims'):
                gt_bboxes.append({
                    'center': gt_obj.get('aabb_center'),
                    'sizes': gt_obj.get('aabb_dims')
                })
        
        # Get Prediction
        pred_item = preds_map.get(task)
        candidates = []
        if pred_item:
            candidates = pred_item.get('top_5_candidates', [])[:k]
        
        # Evaluate
        # We consider success if ANY candidate matches ANY valid GT bbox
        max_iou = 0.0
        
        if gt_bboxes and candidates:
            for cand in candidates:
                cand_bbox = cand.get('bbox')
                if not cand_bbox: continue
                
                # Check against all valid GTs
                for gt_bbox in gt_bboxes:
                    iou = compute_iou_3d(gt_bbox, cand_bbox)
                    if iou > max_iou:
                        max_iou = iou
        
        is_success = max_iou >= iou_threshold
        status = "SUCCESS" if is_success else "FAIL"
        if not gt_bboxes: status = "SKIP (NoGT)"
        if not pred_item: status = "MISSING"
        
        if is_success:
            success_count += 1
            
        if gt_bboxes: # Only count tasks where GT exists
            total_count += 1
        
        print(f"{i:<4} | {status:<8} | {max_iou:.4f}   | {task:<50} | {gt_category}")
        
        # Visualize (Use the first GT bbox for visualization if multiple exist, or draw all?)
        # Let's modify visualization to draw all GTs
        if pcd_points is not None and vis_dir:
            visualize_task(i, task, gt_bboxes, candidates, pcd_points, pcd_colors, vis_dir)

    print("-" * 120)
    
    accuracy = success_count / total_count if total_count > 0 else 0.0
    print(f"\nTotal Valid Tasks: {total_count}")
    print(f"Successes (IoU >= {iou_threshold}): {success_count}")
    print(f"Recall@{k}: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    # Load defaults from config
    try:
        base_cfg = OmegaConf.load(str(_REPO_ROOT / "config" / "base_config.yaml"))
        scene_id = base_cfg.scene_id
        dataset_name = base_cfg.dataset_name
        dataset_path = base_cfg.get("dataset_path", "dataset/HM3D_hovsg")
        
        default_gt = os.path.join(dataset_path, scene_id, "scene_info.json")
        default_prompts = "nav_prompts.json"
        default_pred = "output/query_results/predictions.json"
        default_pcd = f"output/map_results/{dataset_name}_{scene_id}/map/layout.pcd"
        default_vis_dir = "output/query_results/visualization"
        print(f"Loaded config for scene: {scene_id}")
    except Exception as e:
        print(f"Warning: Failed to load config/base_config.yaml: {e}")
        default_gt = None
        default_prompts = "nav_prompts.json"
        default_pred = "output/query_results/predictions.json"
        default_pcd = None
        default_vis_dir = "output/query_results/visualization"

    parser = argparse.ArgumentParser(description="Evaluate 3D Object Retrieval Recall@K")
    parser.add_argument("--gt", type=str, default=default_gt, help="Path to scene_info.json")
    parser.add_argument("--prompts", type=str, default=default_prompts, help="Path to nav_prompts.json")
    parser.add_argument("--pred", type=str, default=default_pred, help="Path to predictions.json")
    parser.add_argument("--iou", type=float, default=0.1, help="IoU threshold (default 0.1)")
    parser.add_argument("--k", type=int, default=5, help="Top K candidates to check (default 5)")
    parser.add_argument("--pcd", type=str, default=default_pcd, help="Path to layout.pcd for visualization (optional)")
    parser.add_argument("--vis_dir", type=str, default=default_vis_dir, help="Directory to save visualizations (optional)")
    
    args = parser.parse_args()
    
    if not args.gt:
        parser.error("--gt is required (failed to load from config)")
    
    if not os.path.exists(args.gt):
        print(f"Error: GT file not found at {args.gt}")
        exit(1)
    if not os.path.exists(args.prompts):
        print(f"Error: Prompts file not found at {args.prompts}")
        exit(1)
    if not os.path.exists(args.pred):
        print(f"Error: Pred file not found at {args.pred}")
        exit(1)

    evaluate_recall_at_k(args.gt, args.prompts, args.pred, args.pcd, args.vis_dir, args.iou, args.k)
