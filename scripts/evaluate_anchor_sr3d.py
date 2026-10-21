#!/usr/bin/env python3
"""
Evaluate anchor_task_runner on Sr3D+ dataset using BeyondBareQueries evaluation protocol.

This script:
1. Loads Sr3D+ queries from annotation files
2. Runs anchor_task_runner for each query
3. Maps UIDs to BBQ format object IDs using bbox matching
4. Evaluates using the same metrics as BeyondBareQueries (IoU-based accuracy)

Similar structure to evaluate_anchor_queries.py but adapted for Sr3D+ dataset.
"""

import os
import json
import random
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import tempfile
import shutil
import open3d as o3d

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DEFAULT_VLM_MODEL = os.environ.get("GEMINI_VLM_MODEL", "gemini-3-pro-preview")

# Define BeyondBareQueries evaluation functions directly
# (We don't import from compute_accuracy_bbq_objects_sr3d.py because it has module-level file I/O)

def get_box3d_min_max(corner):
    """Compute min and max coordinates for 3D bounding box"""
    min_coord = corner.min(axis=0)
    max_coord = corner.max(axis=0)
    x_min, x_max = min_coord[0], max_coord[0]
    y_min, y_max = min_coord[1], max_coord[1]
    z_min, z_max = min_coord[2], max_coord[2]
    return x_min, x_max, y_min, y_max, z_min, z_max


def construct_bbox_corners(center, box_size):
    """Construct 8 corners of a 3D bounding box"""
    sx, sy, sz = box_size
    x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
    y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
    z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def box3d_iou(corners1, corners2):
    """Compute 3D bounding box IoU"""
    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max(corners1)
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)
    return iou

# Import anchor_task_runner components
from scripts.anchor_task_runner import (
    process_single_task,
    load_anchor_file,
    load_manifest,
    find_manifest_entry,
    collect_image_paths,
    load_map_from_disk,
    load_clip_and_fastsam,
    _short_anchor_id,
)
from utils.object import BaseObject
from omegaconf import OmegaConf

# Setup logging
def setup_logging(log_file=None):
    """Setup logging to both console and file"""
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # File handler (if log_file is provided)
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# Default paths
BBQ_DATA_ROOT = project_root / "BeyondBareQueries-main" / "evaluation" / "object_grounding" / "data"
GT_DIRECTORY = BBQ_DATA_ROOT / "scene_description_unaligned_gt"
ANNOTATION_DIRECTORY = BBQ_DATA_ROOT / "scannet" / "sr3d+"
INST_SEM_DIRECTORY = BBQ_DATA_ROOT / "bbq"

# All scenes (you can modify this list)
SCENES = [
    # "scene0011_00",
    # "scene0030_00",
    # "scene0046_00",
    # "scene0086_00",
    "scene0222_00",
    # "scene0378_00",
    # "scene0389_00",
    # "scene0435_00"
]

RELATIONS = [
    "above", "below", "supported", "supporting",
    "left", "right", "back", "front",
    "between", "farthest", "closest"
]


def find_bbq_id_by_bbox(
    pred_bbox_center: List[float],
    pred_bbox_extent: List[float],
    bbq_objects: List[Dict],
    iou_threshold: float = 0.1
) -> Optional[int]:
    """
    Find BBQ object ID by matching bbox with IoU threshold.
    
    Args:
        pred_bbox_center: [x, y, z] center of predicted bbox
        pred_bbox_extent: [width, height, depth] extent of predicted bbox
        bbq_objects: List of BBQ format objects with id, bbox_center, bbox_extent
        iou_threshold: Minimum IoU to consider a match
    
    Returns:
        BBQ object ID if match found, None otherwise
    """
    pred_corners = construct_bbox_corners(pred_bbox_center, pred_bbox_extent)
    
    best_iou = 0.0
    best_id = None
    
    for bbq_obj in bbq_objects:
        bbq_corners = construct_bbox_corners(bbq_obj['bbox_center'], bbq_obj['bbox_extent'])
        iou = box3d_iou(pred_corners, bbq_corners)
        
        if iou > best_iou:
            best_iou = iou
            best_id = bbq_obj['id']
    
    if best_iou >= iou_threshold:
        return best_id
    return None


def load_sr3d_queries(
    scene: str,
    bbq_data_root: Path,
    relations: List[str] = None
) -> List[Dict]:
    """
    Load Sr3D+ queries for a scene.
    
    Args:
        scene: Scene ID (e.g., "scene0011_00")
        bbq_data_root: Path to BBQ data root
        relations: List of relations to include (None = all)
    
    Returns:
        List of query dicts with keys: scene, relation, ann_id, utterance, target_id, is_easy, is_view_dep
    """
    queries = []
    
    if relations is None:
        relations = RELATIONS
    
    annotation_file = bbq_data_root / "scannet" / "sr3d+_all_queries" / f"{scene}_annotation.json"
    
    if not annotation_file.exists():
        logger.warning(f"Annotation file not found: {annotation_file}")
        return queries
    
    with open(annotation_file, 'r') as f:
        all_anns = json.load(f)
    
    # Load relation-specific annotations
    for relation in relations:
        rel_annotation_file = bbq_data_root / "scannet" / f"sr3d+_{relation}" / f"{scene}_annotation.json"
        
        if not rel_annotation_file.exists():
            continue
        
        with open(rel_annotation_file, 'r') as f:
            rel_anns = json.load(f)
        
        # Match with all_anns to get is_easy
        for ann_id, rel_ann in enumerate(rel_anns):
            utterance = rel_ann["utterance"]
            target_id = rel_ann["target_id"]
            
            # Find matching annotation in all_anns to get is_easy
            is_easy = False
            is_view_dep = relation in ["left", "right", "front", "back"]
            
            for ann in all_anns:
                if ann["target_id"] == target_id and ann["utterance"] == utterance:
                    is_easy = ann.get("is_easy", False)
                    break
            
            queries.append({
                "scene": scene,
                "relation": relation,
                "ann_id": ann_id,
                "utterance": utterance,
                "target_id": target_id,
                "is_easy": is_easy,
                "is_view_dep": is_view_dep,
                "query_id": f"{scene}_{relation}_{ann_id}_{target_id}"
            })
    
    return queries


def visualize_gt_and_pred_bboxes(
    gt_pcd_points: np.ndarray,
    gt_bbox: dict,
    pred_bbox: dict,
    output_dir: Path,
    query_text: str,
    query_id: str,
    query_index: int = 0,
    iou: float = 0.0,
    acc_01: float = 0.0,
    acc_025: float = 0.0,
) -> Optional[str]:
    """
    Render GT vs predicted bbox on a top-down XZ view of the GT point cloud.

    Args:
        gt_pcd_points: GT point cloud (N, 3)
        gt_bbox: GT box dict with 'center' and 'sizes'
        pred_bbox: Predicted box dict with 'center' and 'sizes'
        output_dir: Directory for PNG output
        query_text: Query string for title
        query_id: Query id
        query_index: Index in batch
        iou: IoU value for annotation
        acc_01: Accuracy@0.1
        acc_025: Accuracy@0.25

    Returns:
        Path to saved image, or None on failure
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import matplotlib.patches as patches
        from PIL import Image, ImageDraw, ImageFont
        
        # Matplotlib top-down: X horizontal, Z vertical (Y-up world)
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.invert_yaxis()
        
        num_points = len(gt_pcd_points)
        scene_x = gt_pcd_points[:, 0]  # X on plot
        scene_z = gt_pcd_points[:, 2]  # Z on plot (height axis Y unused in 2D)
        
        # 
        sample_step = max(1, num_points // 50000)
        scene_x_sampled = scene_x[::sample_step]
        scene_z_sampled = scene_z[::sample_step]
        ax.scatter(scene_x_sampled, scene_z_sampled, c='gray', s=20.0, alpha=0.5, marker='.', edgecolors='none', zorder=1)
        
        # Draw GT bbox
        cx, cy, cz = gt_bbox['center']
        w, h, d = gt_bbox['sizes']
        x0 = cx - w/2
        y0 = cz - d/2
        rect_gt = patches.Rectangle(
            (x0, y0), w, d,
            linewidth=2.5, edgecolor='lime', facecolor='none',
            label='GT', zorder=3
        )
        ax.add_patch(rect_gt)
        ax.text(x0, y0, "GT", color='lime', fontsize=12, fontweight='bold',
               ha='right', va='bottom', zorder=4)
        
        # Draw predicted bbox
        cx, cy, cz = pred_bbox['center']
        w, h, d = pred_bbox['sizes']
        x0 = cx - w/2
        y0 = cz - d/2
        rect_pred = patches.Rectangle(
            (x0, y0), w, d,
            linewidth=2.5, edgecolor='red', facecolor='none',
            linestyle='--', label='Pred', zorder=2
        )
        ax.add_patch(rect_pred)
        ax.text(x0, y0, f"Pred\nIoU:{iou:.3f}", color='red', fontsize=10,
               ha='left', va='top', zorder=4)
        
        # 
        all_x = np.concatenate([scene_x, [cx - w/2, cx + w/2, gt_bbox['center'][0] - gt_bbox['sizes'][0]/2, gt_bbox['center'][0] + gt_bbox['sizes'][0]/2]])
        all_z = np.concatenate([scene_z, [cz - d/2, cz + d/2, gt_bbox['center'][2] - gt_bbox['sizes'][2]/2, gt_bbox['center'][2] + gt_bbox['sizes'][2]/2]])
        margin_x = (all_x.max() - all_x.min()) * 0.1
        margin_z = (all_z.max() - all_z.min()) * 0.1
        ax.set_xlim(all_x.min() - margin_x, all_x.max() + margin_x)
        ax.set_ylim(all_z.min() - margin_z, all_z.max() + margin_z)
        
        # 
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # 
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image_np = np.asarray(buf)
        plt.close(fig)
        
        # RGB
        image_np = image_np[:, :, :3]
        
        # PIL
        img = Image.fromarray(image_np)
        draw = ImageDraw.Draw(img)
        
        # 
        try:
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except:
            try:
                font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
            except:
                font_small = ImageFont.load_default()
        
        query_short = query_text[:50] + "..." if len(query_text) > 50 else query_text
        info_text = f"Q: {query_short} | IoU: {iou:.3f} | Acc@0.1: {acc_01:.3f} | Acc@0.25: {acc_025:.3f}"
        
        if info_text:
            try:
                if hasattr(draw, 'textbbox'):
                    text_bbox = draw.textbbox((0, 0), info_text, font=font_small)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                else:
                    text_width, text_height = draw.textsize(info_text, font=font_small)
                
                padding = 10
                text_x = padding
                text_y = padding
                text_bg_width = text_width + 20
                text_bg_height = text_height + 10
                
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(
                    [(text_x - 5, text_y - 5), (text_x + text_bg_width, text_y + text_bg_height)],
                    fill=(0, 0, 0, 180)
                )
                img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(img)
                draw.text((text_x, text_y), info_text, fill="white", font=font_small)
            except Exception as e:
                logger.warning(f"Failed to add text to visualization: {e}")
        
        # 
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_task = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in query_text[:50])
        safe_task = safe_task.replace(' ', '_')
        prefix = f"query_{query_index:03d}_"
        output_filename = f"{prefix}visualization_gt_pred_{safe_task}.png"
        output_path = output_dir / output_filename
        img.save(output_path, "PNG")
        
        logger.info(f"Saved GT and predictions visualization to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error visualizing GT and predicted bboxes: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_anchor_predictions_for_sr3d_query(
    query: Dict,
    anchors_with_frames: list,
    manifest: dict,
    obj_map: list,
    anchor_id_map: dict,
    clip_model,
    clip_tokenizer,
    clip_preprocess,
    fastsam,
    device: str,
    merged_cfg,
    llm_model: str = "gpt-5.2",
    vlm_model: str = None,
    base_url: str = None,
    api_key: str = None,
    gemini_api_key: str = None,
    top_k: int = 10,
    output_dir: Optional[Path] = None,
    ablation_mode: str = "full",
    ablation_max_anchors: Optional[int] = 15,
) -> tuple[list, dict]:
    """
    Run anchor_task_runner for a single Sr3D+ query and extract top-k candidates.
    
    Returns:
        (candidates, stats): List of prediction candidates and stats dict
    """
    from pathlib import Path
    
    # Use provided output_dir or create temporary one
    if output_dir is None:
        temp_output_dir = Path(tempfile.mkdtemp(prefix="anchor_sr3d_eval_"))
        cleanup_temp = True
    else:
        temp_output_dir = output_dir
        cleanup_temp = False
    
    if vlm_model is None:
        vlm_model = DEFAULT_VLM_MODEL
    try:
        success, saved_path, stats = process_single_task(
            query["utterance"],
            anchors_with_frames,
            manifest,
            temp_output_dir,
            llm_model,
            vlm_model,
            base_url,
            api_key,
            clip_model,
            clip_tokenizer,
            clip_preprocess,
            fastsam,
            device,
            obj_map,
            anchor_id_map,
            None,
            merged_cfg,
            task_index=0,
            ablation_mode=ablation_mode,
            ablation_max_anchors=ablation_max_anchors,
            gemini_api_key=gemini_api_key,
        )
        
        # Get top candidates from stats
        candidates = stats.get("top_10_candidates", [])[:top_k]
        
        # Add query_id to each candidate
        for c in candidates:
            c["query_id"] = query["query_id"]
        
        # Update stats
        stats["query_id"] = query["query_id"]
        stats["query"] = query
        
        return candidates, stats
        
    finally:
        # Clean up temporary directory only if we created it
        if cleanup_temp:
            try:
                shutil.rmtree(temp_output_dir)
            except:
                pass


def evaluate_scene(
    scene: str,
    bbq_data_root: Path,
    anchors_with_frames: list,
    manifest: dict,
    obj_map: list,
    anchor_id_map: dict,
    clip_model,
    clip_tokenizer,
    clip_preprocess,
    fastsam,
    device: str,
    merged_cfg,
    llm_model: str = "gpt-5.2",
    vlm_model: str = None,
    base_url: str = None,
    api_key: str = None,
    gemini_api_key: str = None,
    relations: List[str] = None,
    map_save_path: str = None,
    ablation_mode: str = "full",
    max_queries_per_scene: Optional[int] = None,
    query_sample_seed: Optional[int] = 42,
    ablation_max_anchors: Optional[int] = 15,
) -> Dict:
    """
    Evaluate a single scene.
    
    Returns:
        Dictionary with evaluation results and statistics
    """
    scene_start_time = time.time()
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing scene: {scene}")
    logger.info(f"{'='*80}")
    
    # Load GT and BBQ objects
    gt_scene_file = bbq_data_root / "scene_description_unaligned_gt" / f"{scene}_scene_description_unaligned.json"
    inst_segm_scene_file = bbq_data_root / "bbq" / f"09.06.2024_scannet_{scene}.json"
    
    if not gt_scene_file.exists():
        logger.warning(f"GT file not found: {gt_scene_file}, skipping scene")
        return {}
    
    if not inst_segm_scene_file.exists():
        logger.warning(f"BBQ scene file not found: {inst_segm_scene_file}, skipping scene")
        return {}
    
    with open(gt_scene_file, 'r') as f:
        gt_objects = json.load(f)
    
    with open(inst_segm_scene_file, 'r') as f:
        bbq_objects = json.load(f)
    
    # Load queries
    queries = load_sr3d_queries(scene, bbq_data_root, relations)
    logger.info(f"Loaded {len(queries)} queries for scene {scene}")
    if max_queries_per_scene is not None and len(queries) > max_queries_per_scene:
        if query_sample_seed is not None:
            random.seed(query_sample_seed)
        queries = random.sample(queries, max_queries_per_scene)
        logger.info(f"Subsampled to {len(queries)} queries (max_queries_per_scene={max_queries_per_scene}, seed={query_sample_seed})")
    if not queries:
        logger.warning(f"No queries found for scene {scene}")
        return {}
    
    # Initialize metrics
    keys = ["is_easy", "is_view_dep"]
    tp = {
        k: {
            "True": {"0.1": 0, "0.25": 0},
            "False": {"0.1": 0, "0.25": 0}
        }
        for k in keys
    }
    tp["overall"] = {"0.1": 0, "0.25": 0}
    
    total_number = {
        k: {
            "True": 0,
            "False": 0
        }
        for k in keys
    }
    total_number["overall"] = 0
    
    # Statistics
    matched_queries = 0
    unmatched_queries = 0
    all_query_stats = []
    
    # Setup visualization output directory (remove existing if present to avoid accumulation)
    scene_output_base = project_root / "output" / "query_results" / "sr3d_eval" / scene
    if scene_output_base.exists():
        logger.info(f"Removing existing output directory for scene {scene}: {scene_output_base}")
        shutil.rmtree(scene_output_base)
    
    vis_output_dir = scene_output_base / "visualizations"
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup best_frame output directory
    best_frame_output_dir = scene_output_base / "best_frames"
    best_frame_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load point cloud for visualization
    layout_pcd_path = None
    gt_pcd_points = None
    if map_save_path:
        layout_pcd_path = os.path.join(map_save_path, "layout.pcd")
        if not os.path.exists(layout_pcd_path):
            if hasattr(merged_cfg, 'preload_path') and merged_cfg.preload_path:
                layout_pcd_path = os.path.join(merged_cfg.preload_path, "layout.pcd")
    
    if layout_pcd_path and os.path.exists(layout_pcd_path):
        try:
            scene_pcd = o3d.io.read_point_cloud(layout_pcd_path)
            if len(scene_pcd.points) > 0:
                gt_pcd_points = np.asarray(scene_pcd.points)
                logger.info(f"Loaded point cloud with {len(gt_pcd_points)} points for visualization")
            else:
                logger.warning(f"Point cloud is empty: {layout_pcd_path}")
        except Exception as e:
            logger.warning(f"Failed to load point cloud from {layout_pcd_path}: {e}")
    else:
        logger.warning(f"Point cloud not found: {layout_pcd_path}")

    if vlm_model is None:
        vlm_model = DEFAULT_VLM_MODEL

    # Process each query
    for query_idx, query in enumerate(queries, 1):
        logger.info(f"  Processing query {query_idx}/{len(queries)}: {query['utterance'][:50]}...")
        
        query_start = time.time()
        
        # Create query-specific output directory
        query_output_dir = vis_output_dir / f"query_{query_idx:03d}"
        query_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get predictions from anchor_task_runner
        try:
            candidates, query_stats = get_anchor_predictions_for_sr3d_query(
                query,
                anchors_with_frames,
                manifest,
                obj_map,
                anchor_id_map,
                clip_model,
                clip_tokenizer,
                clip_preprocess,
                fastsam,
                device,
                merged_cfg,
                llm_model=llm_model,
                vlm_model=vlm_model,
                base_url=base_url,
                api_key=api_key,
                gemini_api_key=gemini_api_key,
                top_k=1,
                output_dir=query_output_dir,
                ablation_mode=ablation_mode,
                ablation_max_anchors=ablation_max_anchors,
            )
        except Exception as e:
            reason = f"Failed to get predictions: {e}"
            logger.error(f"    {reason}")
            unmatched_queries += 1
            all_query_stats.append({
                "query": query,
                "matched": False,
                "reason": reason
            })
            continue
        
        # Save best_frame if available
        best_frame_path = query_stats.get("best_frame_path", "")
        if best_frame_path and os.path.exists(best_frame_path):
            try:
                import shutil as shutil_module
                best_frame_filename = f"query_{query_idx:03d}_best_frame_{os.path.basename(best_frame_path)}"
                saved_best_frame_path = best_frame_output_dir / best_frame_filename
                shutil_module.copy2(best_frame_path, saved_best_frame_path)
                logger.info(f"    Saved best_frame to: {saved_best_frame_path}")
            except Exception as e:
                logger.warning(f"    Failed to save best_frame: {e}")
        
        query_time = time.time() - query_start
        logger.info(f"    Got {len(candidates)} candidates in {query_time:.2f}s")
        
        # Track if this query matched successfully
        query_matched = False
        iou = 0.0
        
        if not candidates:
            reason = "No candidates returned from anchor_task_runner"
            logger.warning(f"    UNMATCHED: {reason}")
            unmatched_queries += 1
            all_query_stats.append({
                "query": query,
                "stats": query_stats,
                "matched": False,
                "reason": reason,
                "iou": 0.0
            })
        else:
            # Get top-1 candidate
            top_candidate = candidates[0]
            pred_uid = top_candidate.get("uid", "")
            pred_bbox = top_candidate.get("bbox", {})
            
            if not pred_bbox or "center" not in pred_bbox or "sizes" not in pred_bbox:
                reason = "No bbox in prediction (candidate missing bbox data)"
                logger.warning(f"    UNMATCHED: {reason}")
                unmatched_queries += 1
                all_query_stats.append({
                    "query": query,
                    "stats": query_stats,
                    "matched": False,
                    "reason": reason,
                    "iou": 0.0
                })
            else:
                # Map UID bbox to BBQ object ID
                pred_bbox_center = pred_bbox["center"]
                pred_bbox_extent = pred_bbox["sizes"]
                
                pred_bbq_id = find_bbq_id_by_bbox(pred_bbox_center, pred_bbox_extent, bbq_objects)
                
                if pred_bbq_id is None:
                    reason = f"Could not match bbox to BBQ object (pred_bbox center={pred_bbox_center}, extent={pred_bbox_extent}, IoU threshold=0.1)"
                    logger.warning(f"    UNMATCHED: {reason}")
                    unmatched_queries += 1
                    all_query_stats.append({
                        "query": query,
                        "stats": query_stats,
                        "matched": False,
                        "reason": reason,
                        "iou": 0.0
                    })
                else:
                    # Get GT bbox
                    gt_obj = next((ob for ob in gt_objects if ob['id'] == query['target_id']), None)
                    if not gt_obj:
                        reason = f"GT object not found (target_id={query['target_id']})"
                        logger.warning(f"    UNMATCHED: {reason}")
                        unmatched_queries += 1
                        all_query_stats.append({
                            "query": query,
                            "stats": query_stats,
                            "matched": False,
                            "reason": reason,
                            "iou": 0.0
                        })
                    else:
                        gt_corners = construct_bbox_corners(gt_obj['bbox_center'], gt_obj['bbox_extent'])
                        
                        # Get predicted bbox from BBQ objects
                        pred_bbq_obj = next((ob for ob in bbq_objects if ob['id'] == pred_bbq_id), None)
                        if not pred_bbq_obj:
                            reason = f"Predicted BBQ object not found (pred_bbq_id={pred_bbq_id})"
                            logger.warning(f"    UNMATCHED: {reason}")
                            unmatched_queries += 1
                            all_query_stats.append({
                                "query": query,
                                "stats": query_stats,
                                "matched": False,
                                "reason": reason,
                                "iou": 0.0
                            })
                        else:
                            # Successfully matched, compute IoU
                            query_matched = True
                            pred_corners = construct_bbox_corners(pred_bbq_obj['bbox_center'], pred_bbq_obj['bbox_extent'])
                            gt_corners = construct_bbox_corners(gt_obj['bbox_center'], gt_obj['bbox_extent'])
                            iou = box3d_iou(pred_corners, gt_corners)
                            # Store for visualization
                            pred_bbq_id_for_viz = pred_bbq_id
                            pred_bbq_obj_for_viz = pred_bbq_obj
                            gt_obj_for_viz = gt_obj
        
        # Update metrics for ALL queries (matched or unmatched)
        # Unmatched queries count as failures (IoU = 0.0)
        if query_matched:
            matched_queries += 1
        else:
            unmatched_queries += 1
        
        # Update total_number for all queries
        # is_view_dep
        if query["is_view_dep"]:
            if query_matched and iou >= 0.25:
                tp["is_view_dep"]["True"]["0.25"] += 1
            if query_matched and iou >= 0.1:
                tp["is_view_dep"]["True"]["0.1"] += 1
            total_number["is_view_dep"]["True"] += 1
        else:
            if query_matched and iou >= 0.25:
                tp["is_view_dep"]["False"]["0.25"] += 1
            if query_matched and iou >= 0.1:
                tp["is_view_dep"]["False"]["0.1"] += 1
            total_number["is_view_dep"]["False"] += 1
        
        # is_easy
        if query["is_easy"]:
            if query_matched and iou >= 0.25:
                tp["is_easy"]["True"]["0.25"] += 1
            if query_matched and iou >= 0.1:
                tp["is_easy"]["True"]["0.1"] += 1
            total_number["is_easy"]["True"] += 1
        else:
            if query_matched and iou >= 0.25:
                tp["is_easy"]["False"]["0.25"] += 1
            if query_matched and iou >= 0.1:
                tp["is_easy"]["False"]["0.1"] += 1
            total_number["is_easy"]["False"] += 1
        
        # Overall
        if query_matched and iou >= 0.25:
            tp["overall"]["0.25"] += 1
        if query_matched and iou >= 0.1:
            tp["overall"]["0.1"] += 1
        total_number["overall"] += 1
        
        # Calculate current accuracies
        # Use total processed queries (matched + unmatched) for denominator
        total_processed = matched_queries + unmatched_queries
        acc_01 = 100 * tp["overall"]["0.1"] / total_processed if total_processed > 0 else 0
        acc_025 = 100 * tp["overall"]["0.25"] / total_processed if total_processed > 0 else 0
        
        # Print accuracy after each query
        logger.info(f"    IoU: {iou:.4f} | Acc@0.1: {acc_01:.2f}% ({tp['overall']['0.1']}/{total_processed}) | Acc@0.25: {acc_025:.2f}% ({tp['overall']['0.25']}/{total_processed})")
        
        # Only visualize if matched
        if query_matched and gt_pcd_points is not None:
            try:
                gt_bbox_dict = {
                    "center": gt_obj_for_viz['bbox_center'],
                    "sizes": gt_obj_for_viz['bbox_extent']
                }
                pred_bbox_dict = {
                    "center": pred_bbq_obj_for_viz['bbox_center'],
                    "sizes": pred_bbq_obj_for_viz['bbox_extent']
                }
                
                vis_path = visualize_gt_and_pred_bboxes(
                    gt_pcd_points,
                    gt_bbox_dict,
                    pred_bbox_dict,
                    query_output_dir,
                    query['utterance'],
                    query['query_id'],
                    query_idx,
                    iou,
                    acc_01 / 100.0,  # Convert to fraction
                    acc_025 / 100.0,  # Convert to fraction
                )
                if vis_path:
                    logger.info(f"    Saved visualization to: {vis_path}")
            except Exception as e:
                logger.warning(f"    Failed to create visualization: {e}")
        
        # Update all_query_stats
        if query_matched:
            all_query_stats.append({
                "query": query,
                "stats": query_stats,
                "matched": True,
                "pred_bbq_id": pred_bbq_id_for_viz,
                "iou": float(iou),
                "acc_01": acc_01,
                "acc_025": acc_025
            })
        
        all_query_stats.append({
            "query": query,
            "stats": query_stats,
            "matched": True,
            "pred_bbq_id": pred_bbq_id,
            "iou": float(iou),
            "acc_01": acc_01,
            "acc_025": acc_025
        })
    
    scene_time = time.time() - scene_start_time
    logger.info(f"Scene {scene} completed in {scene_time:.2f}s")
    logger.info(f"  Matched: {matched_queries}, Unmatched: {unmatched_queries}")
    
    return {
        "scene": scene,
        "ablation_mode": ablation_mode,
        "max_queries_per_scene": max_queries_per_scene,
        "query_sample_seed": query_sample_seed,
        "metrics": {
            k: {
                v: {
                    iou: {
                        "accuracy": 100 * tp[k][v][iou] / total_number[k][v] if total_number[k][v] > 0 else 0,
                        "count": tp[k][v][iou],
                        "total": total_number[k][v]
                    }
                    for iou in ["0.1", "0.25"]
                }
                for v in ["False", "True"]
            }
            for k in keys
        },
        "overall": {
            iou: {
                "accuracy": 100 * tp["overall"][iou] / total_number["overall"] if total_number["overall"] > 0 else 0,
                "count": tp["overall"][iou],
                "total": total_number["overall"]
            }
            for iou in ["0.1", "0.25"]
        },
        "statistics": {
            "matched_queries": matched_queries,
            "unmatched_queries": unmatched_queries,
            "total_queries": len(queries)
        },
        "query_stats": all_query_stats
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate anchor_task_runner on Sr3D+")
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        default=SCENES,
        help="List of scenes to evaluate (default: all scenes)"
    )
    parser.add_argument(
        "--bbq_data_root",
        type=str,
        default=str(BBQ_DATA_ROOT),
        help="Path to BeyondBareQueries evaluation data root"
    )
    parser.add_argument(
        "--relations",
        type=str,
        nargs="+",
        default=RELATIONS,
        help="List of relations to evaluate (default: all relations)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        dest="llm_model",
        help="OpenAI LLM model name",
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default=None,
        dest="vlm_model",
        help="Gemini VLM model (default: env GEMINI_VLM_MODEL or gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        dest="gemini_api_key",
        help="Gemini API key",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results JSON (optional, deprecated - results are now saved to log file)"
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to save log file (optional, defaults to output_dir/evaluation.log if output is specified)"
    )
    
    args = parser.parse_args()
    
    # Setup log file
    log_file = args.log
    if not log_file:
        # Default log file in output directory
        log_file = str(project_root / "output" / "query_results" / "sr3d_eval" / "evaluation.log")
    
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup logging with file output
    global logger
    logger = setup_logging(log_file=log_file)
    logger.info(f"="*80)
    logger.info(f"Starting Sr3D+ Evaluation")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Scenes: {args.scenes}")
    logger.info(f"="*80)
    
    bbq_data_root = Path(args.bbq_data_root)
    
    # Load DualMap configuration (will be updated per scene)
    base_cfg_template = OmegaConf.load(project_root / "config" / "base_config.yaml")
    system_cfg = OmegaConf.load(project_root / "config" / "system_config.yaml")
    # Create initial merged_cfg for initialization (will be updated per scene)
    merged_cfg = OmegaConf.merge(base_cfg_template, system_cfg)
    
    # Initialize BaseObject config
    BaseObject.initialize_config(merged_cfg)
    
    # Load CLIP and FastSAM
    logger.info("Loading CLIP and FastSAM models...")
    try:
        clip_model, clip_tokenizer, clip_preprocess, fastsam, device = load_clip_and_fastsam()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return 1
    
    # Aggregate results across all scenes
    all_scene_results = []
    all_metrics = {
        "is_easy": {
            "True": {"0.1": 0, "0.25": 0},
            "False": {"0.1": 0, "0.25": 0}
        },
        "is_view_dep": {
            "True": {"0.1": 0, "0.25": 0},
            "False": {"0.1": 0, "0.25": 0}
        },
        "overall": {"0.1": 0, "0.25": 0}
    }
    all_totals = {
        "is_easy": {"True": 0, "False": 0},
        "is_view_dep": {"True": 0, "False": 0},
        "overall": 0
    }
    
    # Process each scene
    for scene in args.scenes:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing scene: {scene}")
        logger.info(f"{'='*80}")
        
        # Update config for this scene
        base_cfg = OmegaConf.create(dict(base_cfg_template))  # Create a copy
        base_cfg.scene_id = scene
        # Update representative_frames_path based on scene_id
        if "representative_frames_path" in base_cfg:
            # Replace scene_id in the path
            base_cfg.representative_frames_path = str(base_cfg.representative_frames_path).replace(
                base_cfg_template.scene_id, scene
            )
        else:
            # Construct default path
            dataset_path = base_cfg.get("dataset_path", "./dataset/scannet")
            base_cfg.representative_frames_path = f"{dataset_path}/exported/{scene}/representative_frames"
        
        merged_cfg = OmegaConf.merge(base_cfg, system_cfg)
        logger.info(f"Updated config: scene_id={merged_cfg.scene_id}, representative_frames_path={merged_cfg.representative_frames_path}")
        
        # Resolve paths for this scene
        # Try to get representative_frames_path from config first
        representative_frames_path = merged_cfg.get("representative_frames_path", None)
        
        if representative_frames_path:
            # If config has a template with ${scene_id}, replace it
            if "${scene_id}" in str(representative_frames_path):
                representative_frames_path = str(representative_frames_path).replace("${scene_id}", scene)
            if not os.path.isabs(representative_frames_path):
                representative_frames_path = str(project_root / representative_frames_path.lstrip("./"))
        else:
            # Fallback: try to find representative_frames
            dataset_path = merged_cfg.get("dataset_path", "./dataset")
            if not os.path.isabs(dataset_path):
                dataset_path = str(project_root / dataset_path.lstrip("./"))
            
            # Try different scene ID formats and locations
            scene_variations = [scene]
            if "_" in scene:
                scene_variations.append(scene.replace("_", ""))
            
            for scene_var in scene_variations:
                # Try direct path
                candidate = os.path.join(dataset_path, scene_var, "representative_frames")
                if os.path.exists(candidate):
                    representative_frames_path = candidate
                    break
                
                # Try in scannet subdirectory
                candidate = os.path.join(dataset_path, "scannet", scene_var, "representative_frames")
                if os.path.exists(candidate):
                    representative_frames_path = candidate
                    break
                
                # Try exported subdirectory (common for ScanNet)
                candidate = os.path.join(dataset_path, "scannet", "exported", scene_var, "representative_frames")
                if os.path.exists(candidate):
                    representative_frames_path = candidate
                    break
        
        if representative_frames_path is None or not os.path.exists(representative_frames_path):
            logger.warning(f"Could not find representative_frames for scene {scene}, skipping")
            logger.warning(f"  Tried variations: {scene_variations if 'scene_variations' in locals() else [scene]}")
            continue
        
        logger.info(f"Using representative_frames_path: {representative_frames_path}")
        
        # Load anchor data
        anchor_key_path = os.path.join(representative_frames_path, "anchor_key")
        default_anchor_file = os.path.join(anchor_key_path, "anchor_attached_objects.txt")
        default_manifest = os.path.join(anchor_key_path, "manifest.json")
        
        if not os.path.exists(default_anchor_file) or not os.path.exists(default_manifest):
            logger.warning(f"Anchor files not found for scene {scene} at {anchor_key_path}, skipping")
            continue
        
        manifest, saved_anchor_id_map = load_manifest(Path(default_manifest))
        anchors = load_anchor_file(Path(default_anchor_file))
        
        # Filter anchors with frames
        anchors_with_frames = []
        for anchor in anchors:
            anchor_id = anchor.get("id", "")
            entry = find_manifest_entry(manifest, anchor_id)
            if entry is not None:
                image_paths = collect_image_paths(entry)
                if image_paths:
                    anchors_with_frames.append(anchor)
        
        # Create anchor ID mapping
        anchor_id_map = saved_anchor_id_map.copy() if saved_anchor_id_map else {}
        
        logger.info(f"Loaded {len(anchors_with_frames)} anchors with frames for scene {scene}")
        
        # Resolve map_save_path for this scene
        map_save_path_template = merged_cfg.get("map_save_path", "")
        if not map_save_path_template:
            # Fallback: construct standard path
            output_path = merged_cfg.get("output_path", "./output/map_results")
            dataset_name = "scannet"  # Assuming ScanNet for Sr3D+
            map_save_path_template = f"${{output_path}}/${{dataset_name}}_{{scene_id}}/map"
        
        # Replace variables
        output_path = merged_cfg.get("output_path", "./output/map_results")
        dataset_name = "scannet"  # Assuming ScanNet for Sr3D+
        map_save_path = str(map_save_path_template).replace("${output_path}", output_path)
        map_save_path = map_save_path.replace("${dataset_name}", dataset_name)
        map_save_path = map_save_path.replace("${scene_id}", scene)
        
        if not os.path.isabs(map_save_path):
            map_save_path = str(project_root / map_save_path.lstrip("./"))
        
        # Load map
        logger.info(f"Loading map from: {map_save_path}")
        obj_map = load_map_from_disk(map_save_path)
        
        if not obj_map:
            logger.warning(f"No objects loaded from map for scene {scene}!")
            continue
        
        # Evaluate scene
        scene_results = evaluate_scene(
            scene,
            bbq_data_root,
            anchors_with_frames,
            manifest,
            obj_map,
            anchor_id_map,
            clip_model,
            clip_tokenizer,
            clip_preprocess,
            fastsam,
            device,
            merged_cfg,
            llm_model=args.llm_model,
            vlm_model=args.vlm_model,
            base_url=args.base_url,
            api_key=args.api_key,
            gemini_api_key=args.gemini_api_key,
            relations=args.relations,
            map_save_path=map_save_path,
        )
        
        if scene_results:
            # Print scene-level results
            print(f"\n{'='*80}")
            print(f"SCENE {scene} EVALUATION RESULTS")
            print(f"{'='*80}")
            
            scene_metrics = scene_results.get("metrics", {})
            scene_overall = scene_results.get("overall", {})
            scene_stats = scene_results.get("statistics", {})
            
            # Print and log by category
            for k in ["is_easy", "is_view_dep"]:
                for v in ["False", "True"]:
                    if k in scene_metrics and v in scene_metrics[k]:
                        for iou in ["0.1", "0.25"]:
                            if iou in scene_metrics[k][v]:
                                metric = scene_metrics[k][v][iou]
                                if metric.get("total", 0) > 0:
                                    acc = metric.get("accuracy", 0)
                                    count = metric.get("count", 0)
                                    total = metric.get("total", 0)
                                    result_line = f"{k} {v} IoU@{iou}: {acc:.1f}% ({count}/{total})"
                                    print(result_line)
                                    logger.info(result_line)
            
            # Print and log overall
            for iou in ["0.1", "0.25"]:
                if iou in scene_overall:
                    metric = scene_overall[iou]
                    if metric.get("total", 0) > 0:
                        acc = metric.get("accuracy", 0)
                        count = metric.get("count", 0)
                        total = metric.get("total", 0)
                        result_line = f"Overall IoU@{iou}: {acc:.1f}% ({count}/{total})"
                        print(result_line)
                        logger.info(result_line)
            
            # Print and log statistics
            if scene_stats:
                matched = scene_stats.get("matched_queries", 0)
                unmatched = scene_stats.get("unmatched_queries", 0)
                total_queries = scene_stats.get("total_queries", 0)
                stats_line = f"Matched: {matched}, Unmatched: {unmatched}, Total: {total_queries}"
                print(stats_line)
                logger.info(stats_line)
            
            print(f"{'='*80}\n")
            logger.info(f"{'='*80}\n")
            all_scene_results.append(scene_results)
            
            # Aggregate metrics
            # Note: total is the same for both IoU thresholds, so only add once
            for k in ["is_easy", "is_view_dep"]:
                for v in ["True", "False"]:
                    # Add total only once (using 0.1 as reference, both thresholds have same total)
                    if "0.1" in scene_results["metrics"][k][v]:
                        all_totals[k][v] += scene_results["metrics"][k][v]["0.1"]["total"]
                    for iou in ["0.1", "0.25"]:
                        all_metrics[k][v][iou] += scene_results["metrics"][k][v][iou]["count"]
            
            # Add overall total only once
            if "0.1" in scene_results["overall"]:
                all_totals["overall"] += scene_results["overall"]["0.1"]["total"]
            for iou in ["0.1", "0.25"]:
                all_metrics["overall"][iou] += scene_results["overall"][iou]["count"]
    
    # Print and log overall results
    print("\n" + "="*80)
    print("OVERALL EVALUATION RESULTS")
    print("="*80)
    logger.info("\n" + "="*80)
    logger.info("OVERALL EVALUATION RESULTS")
    logger.info("="*80)
    
    for k in ["is_easy", "is_view_dep"]:
        for v in ["False", "True"]:
            if all_totals[k][v] > 0:
                for iou in ["0.1", "0.25"]:
                    acc = 100 * all_metrics[k][v][iou] / all_totals[k][v]
                    result_line = f"{k} {v} IoU@{iou}: {acc:.1f}% ({all_metrics[k][v][iou]}/{all_totals[k][v]})"
                    print(result_line)
                    logger.info(result_line)
    
    if all_totals["overall"] > 0:
        for iou in ["0.1", "0.25"]:
            acc = 100 * all_metrics["overall"][iou] / all_totals["overall"]
            result_line = f"Overall IoU@{iou}: {acc:.1f}% ({all_metrics['overall'][iou]}/{all_totals['overall']})"
            print(result_line)
            logger.info(result_line)
    
    # Log overall statistics
    total_matched = sum(sr.get("statistics", {}).get("matched_queries", 0) for sr in all_scene_results)
    total_unmatched = sum(sr.get("statistics", {}).get("unmatched_queries", 0) for sr in all_scene_results)
    total_queries = sum(sr.get("statistics", {}).get("total_queries", 0) for sr in all_scene_results)
    stats_line = f"\nOverall Statistics: Matched: {total_matched}, Unmatched: {total_unmatched}, Total: {total_queries}"
    print(stats_line)
    logger.info(stats_line)


if __name__ == "__main__":
    main()
