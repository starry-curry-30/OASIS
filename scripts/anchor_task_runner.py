#!/usr/bin/env python3
from __future__ import annotations
"""
End-to-end anchor task runner.

Workflow:
1. Ask GPT-5 (text-only) which anchors should be inspected next.
2. Immediately feed those anchors' representative frames (macro+micro) to GPT-5 with vision until the task is solved.

Usage:
    python scripts/anchor_task_runner.py "on which bed can I watch TV?"
"""

import argparse
import base64
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set matplotlib backend to Agg to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

# sys.path and utils
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def dualmap_root() -> Path:
    """Root of the DualMap repo: env DUALMAP_ROOT, else ../DualMap sibling to OASIS."""
    env = os.environ.get("DUALMAP_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (project_root.parent / "DualMap").resolve()


from omegaconf import OmegaConf
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import open_clip
from ultralytics import FastSAM
from sklearn.metrics.pairwise import cosine_similarity
import open3d as o3d

from utils.object import BaseObject
from utils.local_observation_cache import LocalObservationCache

DEFAULT_LLM_MODEL = "gpt-5.2"
DEFAULT_VLM_MODEL = os.environ.get("GEMINI_VLM_MODEL", "gemini-3-pro-preview")
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")


SELECTION_SYSTEM_PROMPT = """You are an assistant that selects the most relevant anchor representative frames for visual inspection.

Inputs:

A list of anchors in the scene. Each anchor has:
an id,
a semantic category (e.g., desk, bed, cabinet),
a list of attached movable objects (each with category and count),
a list of nearby objects (each with category and count).

A user task / question in natural language.

Your goal:

Based on the user's task and the structured anchor information, select the most relevant anchor representative frames to inspect next.

Output format:

You MUST respond with a single JSON object, with exactly the following fields:

{
"reasoning": "brief explanation of your reasoning and how you mapped the user task to relevant anchors",
"anchors_to_view": ["anchor_id_1", "anchor_id_2", ...]  // sorted from highest to lowest priority
}

Constraints:

Always provide anchor ids in "anchors_to_view" (sorted by priority).
"anchors_to_view" must ONLY contain ids that appear in the provided anchor list. Do NOT invent new anchor ids.
You must select AT MOST 3 anchors. Prioritize the most promising candidates.
If no anchor seems directly relevant, select the anchor with the most objects (attached + nearby) as it may contain the target.
"reasoning" should be concise (max 2 sentences) but clearly show how you mapped the user task → relevant anchor categories and attached/nearby objects → candidate anchors.
Return ONLY one JSON object. Do not include any duplicate keys, explanations, or extra text before/after the JSON."""


INSPECTION_SYSTEM_PROMPT = """You are an assistant that inspects images to decide whether the user's task is satisfied.

For each call you receive:
- The task description (e.g., "find the picture in the bathroom")
- Multiple images with bounding boxes drawn for all stable anchors, labeled with simple IDs (1, 2, 3, etc.)

You must answer with a single JSON object in this exact schema:
{
  "task_completed": true/false,
  "reasoning": "CRITICAL: Explain exactly what you see. If completed, explicitly state what target object you found and its location. If you cannot clearly see the target object, you MUST fail.",
  "best_frame_path": "The filename of the chosen frame image (e.g., 'frame_000254_representative.png') if task_completed=true, otherwise empty string. Only provide the filename, not the full path.",
  "best_frame_reason": "Why this frame is the best viewpoint if task_completed=true, otherwise empty string.",
  "target_anchor_id": "The simple anchor ID (1, 2, 3, etc.) of the target object if task_completed=true. This is the ID of the anchor that IS the target or contains the target. If task_completed=false, set to empty string.",
  "target_relation": "The spatial relation of the target object. Must be one of: 'self', 'on', 'nearby'. Use 'self' if the target IS an anchor (the anchor itself is the target). Use 'on' if the target is attached to/on an anchor. Use 'nearby' if the target is near an anchor.",
  "target_object_class": "The general class name of the target object (e.g. 'book', 'pillow', 'table', 'window'). ALWAYS extract this from the task description, even if task_completed=false. For example, if task is 'find the window near the table', extract 'window' as the target_object_class. If relation is 'self', this should be the anchor's category.",
  "target_visual_description": "A very simple, short phrase for CLIP matching. Include ONLY the core object name and its most obvious color/material. NO complex details. NO relations. Good: 'silver fish', 'metallic ornament', 'red pillow'. Bad: 'fish with curved tail', 'ornament on a stand'. ALWAYS provide this based on the target_object_class, even if task_completed=false."
}

STRICT RULES TO AVOID HALLUCINATION:
1. **Target Existence**: You must clearly SEE the specific object described in the user's task.
2. **Self Relation**: If the target object IS one of the anchors (e.g., task asks for "the table" and you see anchor ID 15 is a table), use relation='self' and target_anchor_id=15. Do NOT use 'nearby' just because you're viewing from a different anchor's perspective.
3. **Room Verification**: If the user specifies a room (e.g., 'bathroom'), verification of the background is required. Finding the correct object in the WRONG room counts as FAILURE.
4. **Success Criteria**: Set task_completed=true ONLY if you have visually confirmed the target object.
5. **Output**: When task_completed=true, you MUST provide target_anchor_id and best_frame_path (filename only).
6. **Image Selection**: Choose the best frame filename from the provided list. Only provide the filename, not the full path.
7. **Semantic Similarity**: Objects with semantically similar meanings should be considered as the same type of object (e.g., "picture" and "framed photo"), but distinct objects must NOT be confused.
8. **ALWAYS Extract Target Class**: Even when task_completed=false, you MUST extract the target_object_class from the task description. For example, "find the window near the table" → target_object_class="window". This helps with fallback matching."""


def _short_anchor_id(full_id: str) -> str:
    """Return the short display id (first segment, up to 8 characters), safe for filenames."""
    if not full_id:
        return ""
    first_segment = full_id.split("-")[0] if "-" in full_id else full_id
    safe_uid = "".join(ch if ch.isalnum() else "_" for ch in first_segment) or "unknown"
    return safe_uid[:8]


def load_anchor_file(path: Path) -> List[Dict]:
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
                            #  (: "1 (Frames: 88, UID: ...)")
                            count_part = parts[1].strip()
                            count_str = count_part.split('(')[0].strip()
                            count = int(count_str)
                            
                            #  UID ()
                            uids = []
                            if "(UID: " in count_part:
                                uid_part = count_part.split("(UID: ")[1].split(")")[0]
                                uids = [u.strip() for u in uid_part.split(",")]
                            
                            current["attached_objects"].append({
                                "category": parts[0].strip(), 
                                "count": count,
                                "uids": uids
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
                            # 
                            count_part = parts[1].strip()
                            count_str = count_part.split('(')[0].strip()
                            count = int(count_str)
                            
                            #  UID ()
                            uids = []
                            if "UID: " in count_part:
                                uid_part = count_part.split("UID: ")[1].split(")")[0]
                                # nearby line: split after "Frames:" ...
                                # (Frames: ..., UID: ...) patterns
                                # then regex / split
                                # split on UID: ... )
                                #  "Frames: ..., UID: ..."
                                pass 
                            
                            # 
                            import re
                            uid_match = re.search(r'UID: ([^)]+)', count_part)
                            if uid_match:
                                uids = [u.strip() for u in uid_match.group(1).split(",")]

                            current["nearby_objects"].append({
                                "category": parts[0].strip(), 
                                "count": count,
                                "uids": uids
                            })
                        except ValueError:
                            pass
                i += 1
        i += 1
    if current is not None:
        anchors.append(current)
    for anchor in anchors:
        anchor["short_id"] = _short_anchor_id(anchor.get("id", ""))
    return anchors


def build_selection_prompt(anchors: List[Dict], task: str) -> str:
    lines = ["Anchors in the scene:", ""]
    for anchor in anchors:
        display_id = anchor.get("short_id") or anchor.get("id", "")
        lines.append(f"Anchor id: {display_id}")
        lines.append(f"Category: {anchor.get('category', '')}")
        lines.append("Attached objects:")
        objs = anchor.get("attached_objects") or []
        if objs:
            for obj in objs:
                lines.append(f"{obj['category']} x {obj['count']}")
        else:
            lines.append("(none)")
        lines.append("Nearby objects:")
        nearby_objs = anchor.get("nearby_objects") or []
        if nearby_objs:
            for obj in nearby_objs:
                lines.append(f"{obj['category']} x {obj['count']}")
        else:
            lines.append("(none)")
        lines.append("")
    lines.append(f"User task / question: {task}")
    return "\n".join(lines)


def call_chat_json(model: str, system_prompt: str, user_prompt: str, *, base_url: Optional[str], api_key: Optional[str]) -> str:
    if base_url is None:
        base_url = DEFAULT_BASE_URL
    if api_key is None:
        api_key = DEFAULT_API_KEY
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Empty response from GPT")
    return content.strip()


def load_manifest(path: Path) -> Tuple[Dict[str, Dict], Optional[Dict[str, int]]]:
    """
    Load manifest; returns (anchors, anchor_id_map).
    If manifest embeds anchor_id_map, return it; else None for map.
    """
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Wrapped format with anchor_id_map
    if isinstance(data, dict) and "anchor_id_map" in data:
        anchor_id_map = data.get("anchor_id_map", {})
        anchors = data.get("anchors", {})
        return anchors, anchor_id_map
    else:
        # Legacy: anchors-only dict
        return data, None


def find_manifest_entry(manifest: Dict[str, Dict], token: str) -> Optional[Dict]:
    """
    Look up token in manifest.
    Resolve "anchors" wrapper to inner anchors dict.
    """
    # Normalize anchors dict
    anchors = manifest.get("anchors", manifest) if isinstance(manifest, dict) and "anchors" in manifest else manifest
    
    if token in anchors:
        return anchors[token]
    for uid, entry in anchors.items():
        if _short_anchor_id(uid) == token:
            return entry
    return None


def collect_image_paths(entry: Dict) -> List[str]:
    saved_images = entry.get("saved_images") or []
    # Prefer "representative" type images (macro/micro)
    representative = [img for img in saved_images if img.get("type") == "representative"]
    paths = []
    for item in representative:
        path = item.get("path")
        if path and os.path.exists(path):
            paths.append(path)
    return paths


def encode_image(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_size(image_path: str) -> tuple[int, int]:
    """ (width, height)"""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return (0, 0)


def visualize_anchor_in_pointcloud(
    layout_pcd_path: str,
    target_anchor_uid: str,
    obj_map: List[BaseObject],
    output_dir: Path,
    task: str,
    anchor_id_map: Optional[Dict[str, int]] = None,
    suffix: str = "",
    task_index: int = 0,
) -> Optional[str]:
    """
    Top-down matplotlib projection of layout + anchor.
    
    Args:
        layout_pcd_path: layout.pcd
        target_anchor_uid: UID
        obj_map: 
        output_dir: 
        task: 
        anchor_id_map: optional UID -> simple id
    
    Returns:
        or None
    """
    try:
        # 
        if not os.path.exists(layout_pcd_path):
            print(f"[Runner] Layout point cloud not found: {layout_pcd_path}", file=sys.stderr)
            return None
        
        scene_pcd = o3d.io.read_point_cloud(layout_pcd_path)
        if len(scene_pcd.points) == 0:
            print(f"[Runner] Empty point cloud loaded from {layout_pcd_path}", file=sys.stderr)
            return None
        
        # 
        num_points = len(scene_pcd.points)
        has_colors = len(scene_pcd.colors) > 0
        has_normals = len(scene_pcd.normals) > 0
        
        print(f"[Runner] Point cloud information:")
        print(f"  - Total points: {num_points}")
        print(f"  - Has colors: {has_colors} ({len(scene_pcd.colors) if has_colors else 0} color values)")
        print(f"  - Has normals: {has_normals} ({len(scene_pcd.normals) if has_normals else 0} normal vectors)")
        
        if num_points > 0:
            points_array = np.asarray(scene_pcd.points)
            print(f"  - Point range: X[{points_array[:, 0].min():.2f}, {points_array[:, 0].max():.2f}], "
                  f"Y[{points_array[:, 1].min():.2f}, {points_array[:, 1].max():.2f}], "
                  f"Z[{points_array[:, 2].min():.2f}, {points_array[:, 2].max():.2f}]")
        
        if has_colors:
            colors_array = np.asarray(scene_pcd.colors)
            print(f"  - Color range: R[{colors_array[:, 0].min():.2f}, {colors_array[:, 0].max():.2f}], "
                  f"G[{colors_array[:, 1].min():.2f}, {colors_array[:, 1].max():.2f}], "
                  f"B[{colors_array[:, 2].min():.2f}, {colors_array[:, 2].max():.2f}]")
        
        # 
        if not has_colors:
            print(f"[Runner] Warning: layout.pcd has no colors, using gray for scene", file=sys.stderr)
            # Uniform gray when layout has no vertex colors
            scene_colors = np.ones((len(scene_pcd.points), 3)) * 0.5
            scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors)
        else:
            print(f"[Runner] Using original colors from layout.pcd")
        
        # 
        print(f"[Runner] Searching for anchor object with UID: {target_anchor_uid}")
        target_obj = None
        matched_objs = []
        for obj in obj_map:
            obj_uid = str(getattr(obj, 'uid', None))
            if obj_uid == target_anchor_uid:
                target_obj = obj
                matched_objs.append((obj_uid, "exact match"))
                break
            # Match via simple_id when map exists
            if anchor_id_map:
                obj_simple_id = anchor_id_map.get(obj_uid, -1)
                target_simple_id = anchor_id_map.get(target_anchor_uid, -1) if target_anchor_uid in anchor_id_map else -1
                if obj_simple_id > 0 and target_simple_id > 0 and obj_simple_id == target_simple_id:
                    matched_objs.append((obj_uid, f"simple_id match ({obj_simple_id})"))
        
        if target_obj is None:
            print(f"[Runner] ERROR: Target anchor object not found: {target_anchor_uid}", file=sys.stderr)
            if matched_objs:
                print(f"[Runner] Found {len(matched_objs)} potential matches: {matched_objs}", file=sys.stderr)
            # obj_mapUID
            print(f"[Runner] Available object UIDs in obj_map:", file=sys.stderr)
            for i, obj in enumerate(obj_map[:10]):  # 10
                obj_uid = str(getattr(obj, 'uid', None))
                obj_simple_id = anchor_id_map.get(obj_uid, -1) if anchor_id_map else -1
                print(f"  [{i}] UID={obj_uid}, simple_id={obj_simple_id}", file=sys.stderr)
            return None
        
        print(f"[Runner] Found target anchor object: UID={target_anchor_uid}")
        
        # 
        anchor_pcd = getattr(target_obj, 'pcd', None)
        if anchor_pcd is None or len(anchor_pcd.points) == 0:
            print(f"[Runner] Anchor point cloud is empty", file=sys.stderr)
            # Synthesize points inside bbox when anchor has no PCD
            bbox = getattr(target_obj, 'bbox', None)
            if bbox is not None:
                # Sample uniformly in bbox
                min_bound = bbox.get_min_bound()
                max_bound = bbox.get_max_bound()
                # Random points in bbox
                num_points = 1000
                points = np.random.uniform(min_bound, max_bound, (num_points, 3))
                anchor_pcd = o3d.geometry.PointCloud()
                anchor_pcd.points = o3d.utility.Vector3dVector(points)
            else:
                return None
        
        anchor_colors = np.ones((len(anchor_pcd.points), 3)) * [1.0, 0.0, 0.0]
        anchor_pcd.colors = o3d.utility.Vector3dVector(anchor_colors)
        
        combined_pcd = scene_pcd + anchor_pcd
        
        # Matplotlib uses XY; we map XZ -> XY
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        # 
        points = np.asarray(combined_pcd.points)
        colors = np.asarray(combined_pcd.colors) if len(combined_pcd.colors) > 0 else None
        
        # Plot X vs Z as 2D axes
        # World Y-up: use XZ as horizontal plane; matplotlib Y is inverted.
        # Scatter X and Z columns onto XY figure axes.
        x = points[:, 0]
        y = points[:, 2]
        
        # 
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.axis('off')  # 
        ax.invert_yaxis()  # screen Y vs world Z
        
        # Scene vs anchor layers
        scene_points = np.asarray(scene_pcd.points)
        anchor_points = np.asarray(anchor_pcd.points)
        scene_colors = np.asarray(scene_pcd.colors) if len(scene_pcd.colors) > 0 else None
        anchor_colors = np.asarray(anchor_pcd.colors) if len(anchor_pcd.colors) > 0 else None
        
        # Scene points (colored)
        if scene_colors is not None:
            scene_x = scene_points[:, 0]
            scene_y = scene_points[:, 2]  # Use Z
            ax.scatter(scene_x, scene_y, c=scene_colors, s=15.0, alpha=1.0, marker='.', edgecolors='none')
        else:
            scene_x = scene_points[:, 0]
            scene_y = scene_points[:, 2]  # Use Z
            ax.scatter(scene_x, scene_y, c='gray', s=15.0, alpha=0.8, marker='.', edgecolors='none')
        
        # Anchor highlight
        anchor_x = anchor_points[:, 0]
        anchor_y = anchor_points[:, 2]  # Use Z
        ax.scatter(anchor_x, anchor_y, c='red', s=40.0, alpha=1.0, marker='o', edgecolors='darkred', linewidths=0.5)
        
        # 
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # 
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image_np = np.asarray(buf)
        plt.close(fig)
        
        # Drop alpha; RGB only
        image_np = image_np[:, :, :3]
        
        # PIL
        img = Image.fromarray(image_np)
        draw = ImageDraw.Draw(img)
        
        # 
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        except:
            try:
                font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
            except:
                font_large = ImageFont.load_default()
        
        if task:
            try:
                # 
                if hasattr(draw, 'textbbox'):
                    task_bbox = draw.textbbox((0, 0), task, font=font_large)
                    task_text_width = task_bbox[2] - task_bbox[0]
                    task_text_height = task_bbox[3] - task_bbox[1]
                else:
                    task_text_width, task_text_height = draw.textsize(task, font=font_large)
                
                # 
                max_width = img.width - 40
                task_lines = [task]
                if task_text_width > max_width:
                    words = task.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        test_line = current_line + (" " if current_line else "") + word
                        if hasattr(draw, 'textbbox'):
                            test_bbox = draw.textbbox((0, 0), test_line, font=font_large)
                            test_width = test_bbox[2] - test_bbox[0]
                        else:
                            test_width, _ = draw.textsize(test_line, font=font_large)
                        if test_width <= max_width:
                            current_line = test_line
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)
                    task_lines = lines
                
                # 
                line_height = task_text_height + 10
                total_height = len(task_lines) * line_height + 20
                
                # 
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(
                    [(0, 0), (img.width, total_height)],
                    fill=(0, 0, 0, 200)
                )
                img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(img)
                
                # 
                y_offset = 10
                for line in task_lines:
                    if hasattr(draw, 'textbbox'):
                        line_bbox = draw.textbbox((0, 0), line, font=font_large)
                        line_width = line_bbox[2] - line_bbox[0]
                    else:
                        line_width, _ = draw.textsize(line, font=font_large)
                    x_center = (img.width - line_width) // 2
                    draw.text((x_center, y_offset), line, fill="white", font=font_large)
                    y_offset += line_height
                
                # Optional simple id label
                if anchor_id_map:
                    simple_id = anchor_id_map.get(target_anchor_uid, -1)
                    if simple_id > 0:
                        label = f"Anchor ID: {simple_id}"
                        if hasattr(draw, 'textbbox'):
                            label_bbox = draw.textbbox((0, 0), label, font=font_large)
                            label_width = label_bbox[2] - label_bbox[0]
                        else:
                            label_width, _ = draw.textsize(label, font=font_large)
                        x_label = (img.width - label_width) // 2
                        y_label = total_height + 10
                        draw.rectangle(
                            [(x_label - 10, y_label - 5), (x_label + label_width + 10, y_label + task_text_height + 5)],
                            fill=(255, 0, 0, 200)
                        )
                        draw.text((x_label, y_label), label, fill="white", font=font_large)
            except Exception as e:
                print(f"[Runner] Failed to add text to point cloud image: {e}", file=sys.stderr)
        
        # 
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_task = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in task[:50])
        safe_task = safe_task.replace(' ', '_')
        simple_id = anchor_id_map.get(target_anchor_uid, -1) if anchor_id_map else -1
        
        prefix = f"task_{task_index:03d}_" if task_index > 0 else ""
        
        if simple_id > 0:
            output_filename = f"{prefix}pointcloud_anchor_{simple_id}_{safe_task}{suffix}.png"
        else:
            output_filename = f"{prefix}pointcloud_anchor_{safe_task}{suffix}.png"
        output_path = output_dir / output_filename
        img.save(output_path, "PNG")
        
        print(f"[Runner] Saved point cloud projection to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"[Runner] Error visualizing anchor in point cloud: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def add_task_text_to_image(
    image_path: str,
    task: str,
) -> Optional[str]:
    """
    Overlay task text on anchor_key image.
    
    Args:
        image_path: path under anchor_key
        task: 
    
    Returns:
        or None
    """
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # 
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except:
            try:
                font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
            except:
                font_large = ImageFont.load_default()
        
        # 
        if task:
            try:
                # 
                if hasattr(draw, 'textbbox'):
                    task_bbox = draw.textbbox((0, 0), task, font=font_large)
                    task_text_width = task_bbox[2] - task_bbox[0]
                    task_text_height = task_bbox[3] - task_bbox[1]
                else:
                    task_text_width, task_text_height = draw.textsize(task, font=font_large)
                
                # Word-wrap task text to image width
                max_width = img.width - 20
                task_lines = [task]
                if task_text_width > max_width:
                    words = task.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        test_line = current_line + (" " if current_line else "") + word
                        if hasattr(draw, 'textbbox'):
                            test_bbox = draw.textbbox((0, 0), test_line, font=font_large)
                            test_width = test_bbox[2] - test_bbox[0]
                        else:
                            test_width, _ = draw.textsize(test_line, font=font_large)
                        if test_width <= max_width:
                            current_line = test_line
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)
                    task_lines = lines
                
                # 
                line_height = task_text_height + 8
                total_height = len(task_lines) * line_height + 16
                
                # 
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(
                    [(0, 0), (img.width, total_height)],
                    fill=(0, 0, 0, 200)  # 
                )
                img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(img)
                
                y_offset = 8
                for line in task_lines:
                    if hasattr(draw, 'textbbox'):
                        line_bbox = draw.textbbox((0, 0), line, font=font_large)
                        line_width = line_bbox[2] - line_bbox[0]
                    else:
                        line_width, _ = draw.textsize(line, font=font_large)
                    x_center = (img.width - line_width) // 2
                    draw.text((x_center, y_offset), line, fill="white", font=font_large)
                    y_offset += line_height
            except Exception as e:
                print(f"[Runner] Failed to draw task text: {e}", file=sys.stderr)
        
        # 
        output_dir = os.path.join(os.path.dirname(image_path), "annotated")
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_with_task{ext}")
        img.save(output_path)
        
        return output_path
        
    except Exception as e:
        print(f"[Runner] Error adding task text: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def extract_frame_index_from_path(image_path: str) -> Optional[int]:
    """"""
    #  frame_XXXXX 
    match = re.search(r'frame_(\d+)', image_path)
    if match:
        return int(match.group(1))
    # or _XXXXXX_ six-digit frame id
    match = re.search(r'_(\d{6})_', image_path)
    if match:
        return int(match.group(1))
    return None


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
            # CLIP
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
    similarity_threshold: float = 0.2,
) -> List[Tuple[BaseObject, float]]:
    """
    Objects must have CLIP features.
    
    Returns:
        List of (object, similarity) tuples, sorted by similarity descending
    """
    if not obj_map:
        return []
    
    # CLIP
    text_tokens = clip_tokenizer([clip_description]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features.float()  # float32
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 
    matches = []
    for obj in obj_map:
        if not hasattr(obj, 'clip_ft') or obj.clip_ft.size == 0:
            continue
        
        obj_feat = torch.from_numpy(obj.clip_ft).float().to(device).unsqueeze(0)  # float32
        obj_feat = obj_feat / obj_feat.norm(dim=-1, keepdim=True)
        
        similarity = (text_features @ obj_feat.T).item()
        
        if similarity >= similarity_threshold:
            matches.append((obj, similarity))
    
    # 
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def find_object_observations_in_frame(
    obj: BaseObject,
    frame_idx: int,
) -> List:
    """"""
    observations = []
    for obs in getattr(obj, 'observations', []):
        obs_frame_idx = getattr(obs, 'idx', getattr(obs, 'raw_image_idx', None))
        if obs_frame_idx is not None and int(obs_frame_idx) == frame_idx:
            observations.append(obs)
    return observations


def load_clip_and_fastsam(cfg_path: Optional[str] = None):
    """CLIPFastSAM"""
    if cfg_path is None:
        cfg_path = str(dualmap_root() / "config" / "system_config.yaml")
    cfg = OmegaConf.load(cfg_path)
    device = cfg.get("device", "cuda")
    
    # CLIP
    clip_model_name = cfg.clip.model_name
    clip_pretrained = cfg.clip.pretrained
    print(f"[Runner] Loading CLIP model: {clip_model_name} with pretrained '{clip_pretrained}'")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    # MobileCLIP
    if "MobileCLIP" in clip_model_name:
        from mobileclip.modules.common.mobileone import reparameterize_model
        clip_model = reparameterize_model(clip_model)
    
    clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
    
    # FastSAM
    fastsam_path = cfg.fastsam.model_path
    print(f"[Runner] Loading FastSAM model from {fastsam_path}")
    fastsam = FastSAM(fastsam_path)
    
    return clip_model, clip_tokenizer, clip_preprocess, fastsam, device


def generate_mask_from_map(
    image_path: str,
    clip_description: str,
    obj_map: List[BaseObject],
    clip_model,
    clip_tokenizer,
    clip_preprocess=None,
    fastsam=None,
    device: str = "cuda",
    similarity_threshold: float = 0.2,
) -> Optional[tuple[np.ndarray, np.ndarray, List[int]]]:
    """
    For each detection, run mask pipeline.
    Prefer FastSAM masks when available.
    
    Args:
        image_path: 
        clip_description: CLIP
        obj_map: 
        clip_model: CLIP
        clip_tokenizer: CLIP tokenizer
        clip_preprocess: CLIP preprocess (fallback)
        fastsam: FastSAM (fallback)
        device: 
        similarity_threshold: 
    
    Returns:
        (masks, similarities, matched_obj_indices): per-detection outputs
    """
    # 
    frame_idx = extract_frame_index_from_path(image_path)
    if frame_idx is None:
        print(f"[Runner] Cannot extract frame index from path: {image_path}", file=sys.stderr)
        return None
    
    # 
    matches = find_objects_in_map_by_clip_description(
        clip_description, obj_map, clip_model, clip_tokenizer, device, similarity_threshold
    )
    
    if not matches:
        print(f"[Runner] No objects found matching description: {clip_description}")
        return None
    
    # 
    all_masks = []
    all_similarities = []
    matched_obj_indices = []
    
    for obj_idx, (obj, similarity) in enumerate(matches):
        observations = find_object_observations_in_frame(obj, frame_idx)
        
        for obs in observations:
            # maskbbox
            mask = getattr(obs, 'mask', None)
            if mask is not None:
                mask_np = np.asarray(mask)
                if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                    mask_np = mask_np[0]
                if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
                    mask_np = mask_np[..., 0]
                mask_bool = mask_np.astype(bool) if mask_np.dtype != bool else mask_np
                all_masks.append(mask_bool)
                all_similarities.append(similarity)
                matched_obj_indices.append(obj_idx)
            else:
                # Crop mask by bbox
                xyxy = getattr(obs, 'xyxy', None)
                if xyxy is not None:
                    # mask
                    img = Image.open(image_path)
                    img_width, img_height = img.size
                    xyxy_arr = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
                    if len(xyxy_arr) > 0:
                        x1, y1, x2, y2 = map(int, xyxy_arr[0])
                        mask_bool = np.zeros((img_height, img_width), dtype=bool)
                        mask_bool[y1:y2, x1:x2] = True
                        all_masks.append(mask_bool)
                        all_similarities.append(similarity)
                        matched_obj_indices.append(obj_idx)
    
    if not all_masks:
        print(f"[Runner] No observations found for matched objects in frame {frame_idx}")
        # Fallback: FastSAMmask
        if fastsam is not None and clip_preprocess is not None:
            print(f"[Runner] Falling back to FastSAM for mask generation...")
            result = generate_mask_with_clip_fastsam(
                image_path, clip_description, clip_model, clip_tokenizer, 
                clip_preprocess, fastsam, device
            )
            if result is not None:
                masks, similarities, best_mask_idx = result
                # 
                matched_obj_indices = [0] * len(masks)  # 0
                return (masks, similarities, matched_obj_indices)
        print(f"[Runner] WARNING: Failed to find objects in map.")
        return None
    
    # numpy
    # mask
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    masks_array = []
    for mask in all_masks:
        if mask.shape[:2] != (img_height, img_width):
            mask_resized = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((img_width, img_height))) > 127
            masks_array.append(mask_resized)
        else:
            masks_array.append(mask)
    
    masks = np.array(masks_array)
    similarities = np.array(all_similarities)
    
    print(f"[Runner] Found {len(masks)} observations from {len(set(matched_obj_indices))} objects")
    return (masks, similarities, matched_obj_indices)


def generate_mask_with_clip_fastsam(
    image_path: str,
    clip_description: str,
    clip_model,
    clip_tokenizer,
    clip_preprocess,
    fastsam,
    device: str = "cuda",
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    CLIP ranking + FastSAM masks.
    
    Args:
        image_path: 
        clip_description: CLIP
        clip_model: CLIP
        clip_tokenizer: CLIP tokenizer
        clip_preprocess: CLIP
        fastsam: FastSAM
        device: 
    
    Returns:
        (masks, similarities, best_mask_idx): masks list; best idx or None
    """
    try:
        # 
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # FastSAMmask
        results = fastsam(
            image_np,
            device=device,
            retina_masks=True,
            imgsz=1024,
            conf=0.8,
            iou=0.9,
            verbose=False,
        )
        
        if not results or len(results) == 0 or results[0].masks is None:
            print(f"[Runner] FastSAM returned no masks for {image_path}")
            return None
        
        masks = results[0].masks.data.cpu().numpy()  # [N, H, W]
        num_masks = masks.shape[0]
        
        if num_masks == 0:
            print(f"[Runner] No masks found in {image_path}")
            return None
        
        # CLIP
        text_tokens = clip_tokenizer([clip_description]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # maskCLIP
        similarities = []
        best_mask_idx = -1
        best_similarity = -1.0
        
        for i in range(num_masks):
            mask = masks[i]  # [H, W]
            mask_bool = mask > 0.5
            
            # mask
            masked_image = image_np.copy()
            masked_image[~mask_bool] = 0  # mask
            
            # CLIP
            mask_pil = Image.fromarray(masked_image.astype(np.uint8))
            image_tensor = clip_preprocess(mask_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = clip_model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # 
            similarity = (text_features @ image_features.T).item()
            similarities.append(similarity)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_mask_idx = i
        
        if best_mask_idx >= 0:
            print(f"[Runner] Selected mask {best_mask_idx} with similarity {best_similarity:.3f}")
            similarities = np.array(similarities)
            return (masks, similarities, best_mask_idx)
        else:
            print(f"[Runner] No suitable mask found")
            return None
            
    except Exception as e:
        print(f"[Runner] Error generating mask: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def build_inspection_prompt(task: str, entry: Dict, image_paths: List[str], anchor_id_map: Dict[str, int]) -> str:
    """
    Build inspection prompt without revealing current anchor ID/category to avoid bias.
    Only provide image filenames, not full paths.
    """
    lines = [
        f"Task: {task}",
        "",
        "Provided images:",
    ]
    # Only provide image filenames, not full paths
    for p in image_paths:
        lines.append(f"- {os.path.basename(p)}")
    lines.append("")
    lines.append("Each image has bounding boxes drawn for all stable anchors, labeled with simple IDs (1, 2, 3, etc.).")
    lines.append("When task_completed=true, provide target_anchor_id and also:")
    lines.append("- target_relation: 'self', 'on', or 'nearby'")
    lines.append("- target_object_class: the class name of the target object")
    lines.append("- target_visual_description: SIMPLE phrase (Color + Object) only. NO complex details.")
    lines.append("")
    lines.append("Respond using the required JSON schema only.")
    lines.append("Note: For best_frame_path, provide only the image filename (e.g., 'frame_000254_representative_with_task.png'), not the full path.")
    return "\n".join(lines)


def call_inspection(
    vlm_model: str,
    user_prompt: str,
    image_paths: List[str],
    *,
    gemini_api_key: Optional[str] = None,
) -> str:
    from utils.gemini_vlm import generate_content_with_images

    text = generate_content_with_images(
        system_instruction=INSPECTION_SYSTEM_PROMPT,
        user_text=user_prompt,
        image_paths=list(image_paths),
        model=vlm_model,
        api_key=gemini_api_key,
    )
    if not text:
        raise RuntimeError("Empty response from Gemini VLM (inspection)")
    return text


def _extract_text_from_response(response) -> str:
    """
    Extract textual content from an OpenAI responses API object, matching the
    flexible parsing used in query_anchor_vlm.py.
    """
    text = None

    try:
        if hasattr(response, "model_dump_json"):
            resp_dict = json.loads(response.model_dump_json())
        elif hasattr(response, "model_dump"):
            resp_dict = response.model_dump()
        elif hasattr(response, "dict"):
            resp_dict = response.dict()
        elif hasattr(response, "__dict__"):
            resp_dict = response.__dict__
        elif isinstance(response, dict):
            resp_dict = response
        else:
            resp_dict = None

        if resp_dict and "output" in resp_dict and isinstance(resp_dict["output"], list):
            for output_item in resp_dict["output"]:
                if isinstance(output_item, dict) and "content" in output_item:
                    content = output_item["content"]
                    if isinstance(content, list):
                        for content_item in content:
                            if isinstance(content_item, dict) and "text" in content_item:
                                text_candidate = content_item["text"]
                                if isinstance(text_candidate, str) and text_candidate.strip():
                                    return text_candidate.strip()
                elif hasattr(output_item, "content") and output_item.content:
                    for content_item in output_item.content:
                        text_candidate = getattr(content_item, "text", None)
                        if text_candidate and str(text_candidate).strip():
                            return str(text_candidate).strip()
    except Exception:
        pass

    if hasattr(response, "output") and response.output:
        for output_item in response.output:
            content = getattr(output_item, "content", None)
            if isinstance(content, list):
                for content_item in content:
                    text_candidate = getattr(content_item, "text", None)
                    if text_candidate and str(text_candidate).strip():
                        return str(text_candidate).strip()
                    if isinstance(content_item, dict) and "text" in content_item:
                        text_candidate = content_item["text"]
                        if isinstance(text_candidate, str) and text_candidate.strip():
                            return text_candidate.strip()
    return ""


def draw_mask_and_save(
    image_path: str,
    masks: np.ndarray,
    similarities: np.ndarray,
    matched_obj_indices: List[int],
    output_dir: Path,
    task: str,
) -> Optional[str]:
    """
    Overlay mask on RGB.
    Heatmap from similarities.
    
    Args:
        image_path: 
        masks: mask (N, H, W)
        similarities:  (N,)
        matched_obj_indices: 
        output_dir: 
        task: query string
    
    Returns:
        or None
    """
    try:
        # 
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        img_width, img_height = img.size
        
        # 0-1
        if similarities.max() > similarities.min():
            normalized_sims = (similarities - similarities.min()) / (similarities.max() - similarities.min())
        else:
            normalized_sims = np.ones_like(similarities)

        # overlay
        heatmap = np.zeros((img_height, img_width), dtype=np.float32)
        
        # Resize mask
        for i in range(len(masks)):
            mask = masks[i]  # [H, W]
            mask_bool = mask > 0.5
            
            # mask
            if mask_bool.shape[:2] != (img_height, img_width):
                mask_bool = np.array(Image.fromarray(mask_bool.astype(np.uint8) * 255).resize((img_width, img_height))) > 127
            
            similarity_weight = normalized_sims[i]
            heatmap[mask_bool] = np.maximum(heatmap[mask_bool], similarity_weight)
        
        # RGBA composite
        overlay = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        overlay_np = np.array(overlay)
        
        # Heatmap to red channel; alpha from similarity (0-255 scale)
        alpha = (heatmap * 200).astype(np.uint8)  # cap ~200
        overlay_np[:, :, 0] = (heatmap * 255).astype(np.uint8)  # 
        overlay_np[:, :, 3] = alpha  # alpha
        
        overlay = Image.fromarray(overlay_np)
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        
        # 
        draw = ImageDraw.Draw(img)
        
        # Scene vs anchor layers
        font = None
        font_large = None
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, 20)
                    font_large = ImageFont.truetype(font_path, 24)
                    break
            except:
                continue
        if font is None:
            try:
                font = ImageFont.load_default()
                font_large = ImageFont.load_default()
            except:
                font = None
                font_large = None
        
        # 
        if font_large is not None and task:
            try:
                # 
                if hasattr(draw, 'textbbox'):
                    task_bbox = draw.textbbox((0, 0), task, font=font_large)
                    task_text_width = task_bbox[2] - task_bbox[0]
                    task_text_height = task_bbox[3] - task_bbox[1]
                else:
                    task_text_width, task_text_height = draw.textsize(task, font=font_large)
                
                max_width = img_width - 20
                if task_text_width > max_width:
                    # 
                    words = task.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        test_line = current_line + (" " if current_line else "") + word
                        if hasattr(draw, 'textbbox'):
                            test_bbox = draw.textbbox((0, 0), test_line, font=font_large)
                            test_width = test_bbox[2] - test_bbox[0]
                        else:
                            test_width, _ = draw.textsize(test_line, font=font_large)
                        if test_width <= max_width:
                            current_line = test_line
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)
                    task_lines = lines
                else:
                    task_lines = [task]
                
                line_height = task_text_height + 5
                total_height = len(task_lines) * line_height + 10
                
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(
                    [(0, 0), (img_width, total_height)],
                    fill=(0, 0, 0, 200)  # 
                )
                img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(img)
                
                y_offset = 5
                for line in task_lines:
                    if hasattr(draw, 'textbbox'):
                        line_bbox = draw.textbbox((0, 0), line, font=font_large)
                        line_width = line_bbox[2] - line_bbox[0]
                    else:
                        line_width, _ = draw.textsize(line, font=font_large)
                    x_center = (img_width - line_width) // 2
                    draw.text((x_center, y_offset), line, fill="white", font=font_large)
                    y_offset += line_height
            except Exception as e:
                print(f"[Runner] Failed to draw task text: {e}", file=sys.stderr)
        
        # 
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 
        image_name = Path(image_path).stem
        # 
        safe_task = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in task[:50])
        safe_task = safe_task.replace(' ', '_')
        output_filename = f"{image_name}_mask_{safe_task}.png"
        output_path = output_dir / output_filename
        
        # 
        img.save(output_path, "PNG")
        
        return str(output_path)
    except Exception as exc:
        print(f"[Runner] Failed to draw bbox and save: {exc}", file=sys.stderr)
        return None


def process_single_task(
    task: str,
    anchors_with_frames: List[Dict],
    manifest: Dict[str, Dict],
    output_dir: Path,
    llm_model: str,
    vlm_model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    clip_model=None,
    clip_tokenizer=None,
    clip_preprocess=None,
    fastsam=None,
    device: str = "cuda",
    obj_map: Optional[List[BaseObject]] = None,
    anchor_id_map: Optional[Dict[str, int]] = None,
    observation_cache: Optional[LocalObservationCache] = None,
    merged_cfg=None,
    task_index: int = 0,
    ablation_mode: str = "full",
    ablation_max_anchors: Optional[int] = 15,
    gemini_api_key: Optional[str] = None,
) -> tuple[bool, Optional[str], Dict]:
    """
    Top-down matplotlib projection of layout + anchor.
    
    Returns:
        (success, saved_image_path, stats): 
        - success: True
        - saved_image_path: 
        - stats: 
    """
    import time
    task_start_time = time.time()
    
    # 
    stats = {
        "task": task,
        "anchors_viewed": 0,
        "frames_viewed": 0,
        "completed_by": "None",  # "VLM", "CLIP", or "None"
        "duration": 0.0,
        "success": False
    }

    # --- Ablation: w/o LLM Proposal = exhaustive VLM over all anchors ---
    if ablation_mode == "w/o_llm":
        # No LLM call; verify all anchors with VLM (exhaustive, optionally capped for efficiency)
        anchors_to_view = []
        for a in anchors_with_frames:
            sid = a.get("short_id") or _short_anchor_id(a.get("id", ""))
            if sid and sid not in anchors_to_view:
                anchors_to_view.append(sid)
        total_anchors = len(anchors_to_view)
        if ablation_max_anchors is not None and total_anchors > ablation_max_anchors:
            anchors_to_view = anchors_to_view[:ablation_max_anchors]
            print(f"[Runner] [Ablation w/o LLM] Exhaustive VLM: viewing first {len(anchors_to_view)}/{total_anchors} anchors (capped).", file=sys.stderr)
        else:
            print(f"[Runner] [Ablation w/o LLM] Exhaustive VLM: viewing all {len(anchors_to_view)} anchors.", file=sys.stderr)
        stats["ablation_mode"] = "w/o_llm"
    else:
        selection_prompt = build_selection_prompt(anchors_with_frames, task)

        # --- Stage 1: Selection ---
        stage1_start = time.time()
        selection_response = call_chat_json(llm_model, SELECTION_SYSTEM_PROMPT, selection_prompt, base_url=base_url, api_key=api_key)
        stage1_end = time.time()
        stage1_duration = stage1_end - stage1_start

        try:
            selection_json = json.loads(selection_response)
        except json.JSONDecodeError as exc:
            print(f"[Runner] Invalid JSON from selection stage: {exc}", file=sys.stderr)
            stats["duration"] = time.time() - task_start_time
            return (False, None, stats)

        anchors_to_view = selection_json.get("anchors_to_view") or []
        if not anchors_to_view:
            print("[Runner] Selection stage returned no anchors_to_view.")
            print("[Runner] Falling back to anchor with most objects (attached + nearby).")
            max_objects = -1
            best_anchor_id = None
            for anchor in anchors_with_frames:
                attached_count = sum(obj.get("count", 0) for obj in anchor.get("attached_objects", []))
                nearby_count = sum(obj.get("count", 0) for obj in anchor.get("nearby_objects", []))
                total_objects = attached_count + nearby_count
                if total_objects > max_objects:
                    max_objects = total_objects
                    best_anchor_id = anchor.get("short_id") or anchor.get("id", "")
            
            if best_anchor_id:
                anchors_to_view = [best_anchor_id]
                print(f"[Runner] Selected anchor {best_anchor_id} with {max_objects} total objects.")
            else:
                print("[Runner] No anchor found with objects; nothing to inspect.")
                if not obj_map:
                    stats["duration"] = time.time() - task_start_time
                    return (False, None, stats)

        # Top-3 anchors (full detail)
        if ablation_mode == "full":
            max_anchors_to_view = 3
            if len(anchors_to_view) > max_anchors_to_view:
                print(f"[Runner] Limiting anchors_to_view to top {max_anchors_to_view} (from {len(anchors_to_view)}).")
                anchors_to_view = anchors_to_view[:max_anchors_to_view]
        stats["ablation_mode"] = ablation_mode

    # --- Ablation: w/o VLM Verification = LLM top-1 only, no inspection ---
    if ablation_mode == "w/o_vlm":
        if not anchors_to_view:
            stats["duration"] = time.time() - task_start_time
            if obj_map and clip_model:
                clip_query = task
                matches = find_objects_in_map_by_clip_description(
                    clip_query, obj_map, clip_model, clip_tokenizer, device, similarity_threshold=0.0
                )
                stats["top_10_candidates"] = []
                for obj, score in matches[:10]:
                    candidate_info = {
                        "uid": str(getattr(obj, "uid", "")),
                        "score": float(score),
                        "class_id": getattr(obj, "class_id", -1),
                        "class_name": "unknown"
                    }
                    bbox = getattr(obj, "bbox", None)
                    if bbox:
                        candidate_info["bbox"] = {"center": bbox.get_center().tolist(), "sizes": bbox.get_extent().tolist()}
                    stats["top_10_candidates"].append(candidate_info)
            return (False, None, stats)
        first_token = anchors_to_view[0]
        entry = find_manifest_entry(manifest, first_token)
        if entry is None:
            stats["duration"] = time.time() - task_start_time
            if obj_map and clip_model:
                matches = find_objects_in_map_by_clip_description(
                    task, obj_map, clip_model, clip_tokenizer, device, similarity_threshold=0.0
                )
                stats["top_10_candidates"] = []
                for obj, score in matches[:10]:
                    candidate_info = {
                        "uid": str(getattr(obj, "uid", "")),
                        "score": float(score),
                        "class_id": getattr(obj, "class_id", -1),
                        "class_name": "unknown"
                    }
                    bbox = getattr(obj, "bbox", None)
                    if bbox:
                        candidate_info["bbox"] = {"center": bbox.get_center().tolist(), "sizes": bbox.get_extent().tolist()}
                    stats["top_10_candidates"].append(candidate_info)
            return (False, None, stats)
        target_anchor_uid = entry.get("uid", "")
        stats["top_10_candidates"] = []
        anchor_obj = next((o for o in obj_map if str(getattr(o, "uid", "")) == target_anchor_uid), None) if obj_map else None
        if anchor_obj:
            candidate_info = {
                "uid": target_anchor_uid,
                "score": 1.0,
                "class_id": getattr(anchor_obj, "class_id", -1),
                "class_name": "unknown"
            }
            bbox = getattr(anchor_obj, "bbox", None)
            if bbox:
                candidate_info["bbox"] = {"center": bbox.get_center().tolist(), "sizes": bbox.get_extent().tolist()}
            stats["top_10_candidates"].append(candidate_info)
        if obj_map and clip_model:
            existing_uids = {c["uid"] for c in stats["top_10_candidates"]}
            matches = find_objects_in_map_by_clip_description(
                task, obj_map, clip_model, clip_tokenizer, device, similarity_threshold=0.0
            )
            for obj, score in matches:
                uid = str(getattr(obj, "uid", ""))
                if uid not in existing_uids:
                    candidate_info = {
                        "uid": uid,
                        "score": float(score),
                        "class_id": getattr(obj, "class_id", -1),
                        "class_name": "unknown"
                    }
                    bbox = getattr(obj, "bbox", None)
                    if bbox:
                        candidate_info["bbox"] = {"center": bbox.get_center().tolist(), "sizes": bbox.get_extent().tolist()}
                    stats["top_10_candidates"].append(candidate_info)
                    existing_uids.add(uid)
                    if len(stats["top_10_candidates"]) >= 10:
                        break
        stats["completed_by"] = "LLM_only"
        stats["duration"] = time.time() - task_start_time
        return (False, None, stats)

    # Initialize set to track objects viewed by VLM to avoid redundant CLIP search
    viewed_uids = set()

    # --- Stage 2: Inspection ---
    stage2_start = time.time()
    # Setup logger for Stage 2 (inspection) - print to console AND log
    import logging
    logger_stage2 = logging.getLogger(__name__)
    
    # Stage 2: inspection loop
    for anchor_token in anchors_to_view:
        stats["anchors_viewed"] += 1
        entry = find_manifest_entry(manifest, anchor_token)
        if entry is None:
            print(f"[Runner] Anchor {anchor_token} not found in manifest, skipping.", file=sys.stderr)
            continue
        image_paths = collect_image_paths(entry)
        if not image_paths:
            print(f"[Runner] Anchor {anchor_token} has no representative frames, skipping.", file=sys.stderr)
            continue
        
        # Collect UIDs of objects associated with this anchor (attached + nearby) + anchor itself
        # These objects are visible in the representative frames, so if VLM fails, we shouldn't CLIP-search them again.
        full_anchor_uid = entry.get("uid", "")
        if full_anchor_uid:
            viewed_uids.add(full_anchor_uid)
            
        current_anchor_detail = next((a for a in anchors_with_frames if a.get("short_id") == anchor_token or a.get("id") == anchor_token), None)
        if current_anchor_detail:
            for item in current_anchor_detail.get("attached_objects", []):
                for uid in item.get("uids", []):
                    viewed_uids.add(uid)
            for item in current_anchor_detail.get("nearby_objects", []):
                for uid in item.get("uids", []):
                    viewed_uids.add(uid)
        
        stats["frames_viewed"] += len(image_paths)

        print("\n" + "=" * 80)
        print(f"[Runner] Inspecting anchor {anchor_token} ({entry.get('class_name', 'unknown')})")
        print("[Runner] Frames provided:")
        
        # Use original images without task text annotation (to avoid covering image content)
        # Images already have bounding boxes drawn from anchor_key saving
        for p in image_paths:
            print(f"  - {os.path.basename(p)}")

        inspection_prompt = build_inspection_prompt(task, entry, image_paths, anchor_id_map or {})
        # Stage 2: Print to console AND log
        print("\n=== Inspection Prompt ===\n", inspection_prompt)
        logger_stage2.info("\n=== Inspection Prompt ===\n" + str(inspection_prompt))

        try:
            inspect_call_start = time.time()
            inspection_response = call_inspection(
                vlm_model,
                inspection_prompt,
                image_paths,
                gemini_api_key=gemini_api_key,
            )
            inspect_call_end = time.time()
            timer_msg = f"[Timer] Inspection call for {anchor_token} took: {inspect_call_end - inspect_call_start:.2f}s"
            print(timer_msg)
            logger_stage2.info(timer_msg)
            
        except Exception as exc:
            error_msg = f"[Runner] Inspection failed for {anchor_token}: {exc}"
            print(error_msg, file=sys.stderr)
            logger_stage2.error(error_msg)
            continue

        print("\n[Runner] Inspection Response:")
        print(inspection_response)
        logger_stage2.info("\n[Runner] Inspection Response:")
        logger_stage2.info(inspection_response)

        # Parse JSON response
        inspection_json = None
        try:
            inspection_json = json.loads(inspection_response)
        except json.JSONDecodeError as exc:
            print(f"[Runner] Invalid JSON from inspection stage: {exc}", file=sys.stderr)
            print(f"[Runner] Response length: {len(inspection_response)} characters", file=sys.stderr)
            print(f"[Runner] First 500 chars: {inspection_response[:500]}", file=sys.stderr)
            print(f"[Runner] Last 500 chars: {inspection_response[-500:]}", file=sys.stderr)
            
            # Strip markdown fences
            #  ```json ... ```  ``` ... ```
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', inspection_response, re.DOTALL)
            if json_match:
                try:
                    inspection_json = json.loads(json_match.group(1))
                    print("[Runner] Successfully extracted JSON from code block", file=sys.stderr)
                except json.JSONDecodeError:
                    pass
            
            # Fallback: substring between first/last brace
            if inspection_json is None:
                first_brace = inspection_response.find('{')
                last_brace = inspection_response.rfind('}')
                if first_brace >= 0 and last_brace > first_brace:
                    json_candidate = inspection_response[first_brace:last_brace+1]
                    try:
                        inspection_json = json.loads(json_candidate)
                        print("[Runner] Successfully extracted JSON by finding braces", file=sys.stderr)
                    except json.JSONDecodeError:
                        pass
            
            if inspection_json is None:
                print("[Runner] Failed to parse JSON, moving on.", file=sys.stderr)
                continue

        if inspection_json.get("task_completed"):
            stage2_end = time.time()
            timer_msg1 = f"\n[Timer] Stage 2 (Inspection) finished. Total Duration: {stage2_end - stage2_start:.2f}s"
            timer_msg2 = f"[Timer] Total Task Processing Time: {stage2_end - task_start_time:.2f}s"
            print(timer_msg1)
            print(timer_msg2)
            logger_stage2.info(timer_msg1)
            logger_stage2.info(timer_msg2)
            
            best_frame_filename = inspection_json.get("best_frame_path", "")
            # VLM now returns only filename, need to find full path
            best_frame = None
            if best_frame_filename:
                # Try to find the full path by matching filename
                for img_path in image_paths:
                    if os.path.basename(img_path) == best_frame_filename or img_path.endswith(best_frame_filename):
                        best_frame = img_path
                        break
                # If not found, try to construct path from original image paths
                if not best_frame and image_paths:
                    # Use the directory of first image and append filename
                    base_dir = os.path.dirname(image_paths[0])
                    candidate_path = os.path.join(base_dir, best_frame_filename)
                    if os.path.exists(candidate_path):
                        best_frame = candidate_path
                    else:
                        # Try without _with_task suffix (in case VLM returns original name)
                        base_name = best_frame_filename.replace("_with_task", "")
                        candidate_path = os.path.join(base_dir, base_name)
                        if os.path.exists(candidate_path):
                            best_frame = candidate_path
                        else:
                            # Fallback: use first image path
                            best_frame = image_paths[0]
                            print(f"[Runner] WARNING: Could not find best_frame '{best_frame_filename}', using first image: {best_frame}")
            
            target_anchor_id = inspection_json.get("target_anchor_id", "")
            target_relation = inspection_json.get("target_relation", "self")
            target_object_class = inspection_json.get("target_object_class", "")
            target_visual_description = inspection_json.get("target_visual_description", "")
            
            print("\n[Runner] Task completed by this anchor!")
            print(f"[Runner] Relation: {target_relation}, Target Class: {target_object_class}")
            logger_stage2.info("\n[Runner] Task completed by this anchor!")
            logger_stage2.info(f"[Runner] Relation: {target_relation}, Target Class: {target_object_class}")
            if target_visual_description:
                print(f"[Runner] Visual Description: {target_visual_description}")
                logger_stage2.info(f"[Runner] Visual Description: {target_visual_description}")
            
            if best_frame and target_anchor_id:
                print(f"[Runner] Navigate to frame: {best_frame}")
                print(f"[Runner] Target anchor ID: {target_anchor_id}")
                
                # 1. UID
                target_anchor_uid = None
                target_anchor_uid_candidates = []
                for uid, simple_id in sorted(anchor_id_map.items(), key=lambda x: x[1]):
                    if str(simple_id) == str(target_anchor_id):
                        target_anchor_uid_candidates.append(uid)
                
                if target_anchor_uid_candidates:
                    target_anchor_uid = target_anchor_uid_candidates[0]
                else:
                    print(f"[Runner] ERROR: No UID found for anchor ID {target_anchor_id}")
                    return (True, None)

                # Step 2: resolved object UID (starts as anchor; CLIP may change for on/nearby)
                target_final_uid = target_anchor_uid 
                
                if target_relation in ["on", "nearby"] and target_object_class:
                    print(f"[Runner] Refining target object ({target_relation} {target_object_class})...")
                    
                    current_anchor_dict = None
                    for a in anchors_with_frames:
                        if a.get("short_id") == _short_anchor_id(target_anchor_uid) or a.get("id") == target_anchor_uid:
                            current_anchor_dict = a
                            break
                    
                    if current_anchor_dict:
                        candidates = []
                        source_list = []
                        if target_relation == "on":
                            source_list = current_anchor_dict.get("attached_objects", [])
                        elif target_relation == "nearby":
                            source_list = current_anchor_dict.get("nearby_objects", [])
                        
                        candidate_uids = []
                        for item in source_list:
                            if target_object_class.lower() in item.get("category", "").lower():
                                candidate_uids.extend(item.get("uids", []))
                        
                        # If no name match, fall back to all attached+nearby for CLIP
                        if not candidate_uids:
                            print(f"[Runner] No name match for '{target_object_class}' in {target_relation} list.")
                            print(f"[Runner] Falling back: using ALL attached and nearby objects for CLIP matching.")
                            
                            fallback_sources = []
                            fallback_sources.extend(current_anchor_dict.get("attached_objects", []))
                            fallback_sources.extend(current_anchor_dict.get("nearby_objects", []))
                            
                            for item in fallback_sources:
                                candidate_uids.extend(item.get("uids", []))

                        if candidate_uids:
                            print(f"[Runner] Found {len(candidate_uids)} candidate UIDs: {candidate_uids}")
                            
                            # CLIP Re-ranking of candidate UIDs
                            clip_query = target_visual_description if target_visual_description else target_object_class
                            print(f"[Runner] Scoring candidates with CLIP using query: '{clip_query}'")
                            
                            # Encode text
                            text_tokens = clip_tokenizer([clip_query]).to(device)
                            with torch.no_grad():
                                text_features = clip_model.encode_text(text_tokens)
                                text_features /= text_features.norm(dim=-1, keepdim=True)
                            
                            local_candidates = []
                            for uid in candidate_uids:
                                obj = next((o for o in obj_map if str(getattr(o, 'uid', '')) == uid), None)
                                if obj and hasattr(obj, 'clip_ft') and obj.clip_ft is not None:
                                    obj_feat = torch.from_numpy(obj.clip_ft).to(device).float().unsqueeze(0)
                                    obj_feat /= obj_feat.norm(dim=-1, keepdim=True)
                                    sim = (text_features @ obj_feat.T).item()
                                    local_candidates.append((obj, sim))
                                else:
                                    # If object not found or no features, skip or assign low score
                                    print(f"  - UID {uid}: features missing, skipping.")
                                    
                            # Sort by similarity descending
                            local_candidates.sort(key=lambda x: x[1], reverse=True)
                            
                            if local_candidates:
                                # Best candidate for VLM visualization target
                                target_final_uid = str(getattr(local_candidates[0][0], 'uid', ''))
                                best_sim = local_candidates[0][1]
                                print(f"[Runner] CLIP selected best candidate: {target_final_uid} (score {best_sim:.4f})")
                                
                                # Populate stats["top_10_candidates"]
                                stats["top_10_candidates"] = []
                                
                                # Add up to top 10 from local candidates
                                for obj, score in local_candidates[:10]:
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
                                    stats["top_10_candidates"].append(candidate_info)
                                
                                # If we have fewer than 10, fill from global search
                                if len(stats["top_10_candidates"]) < 10:
                                    needed = 10 - len(stats["top_10_candidates"])
                                    print(f"[Runner] Only {len(stats['top_10_candidates'])} local candidates. Filling {needed} from global search.")
                                    
                                    # Exclude UIDs we already have
                                    existing_uids = {c["uid"] for c in stats["top_10_candidates"]}
                                    
                                    # Global Search
                                    global_matches = find_objects_in_map_by_clip_description(
                                        clip_query, obj_map, clip_model, clip_tokenizer, device, similarity_threshold=0.0
                                    )
                                    
                                    for obj, score in global_matches:
                                        uid = str(getattr(obj, "uid", ""))
                                        if uid not in existing_uids:
                                            candidate_info = {
                                                "uid": uid,
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
                                            stats["top_10_candidates"].append(candidate_info)
                                            existing_uids.add(uid)
                                            if len(stats["top_10_candidates"]) >= 10:
                                                break
                                    
                            else:
                                print("[Runner] Failed to compute similarities for any candidate, falling back to first from list.")
                                target_final_uid = candidate_uids[0]
                                
                        else:
                            print(f"[Runner] No candidates found for class '{target_object_class}' in {target_relation} list.")
                    else:
                        print("[Runner] Warning: Could not find anchor info in loaded list to look up relations.")
                
                # Handle 'self' relation: anchor itself is the target
                elif target_relation == "self" and target_anchor_uid:
                     print(f"[Runner] Relation is 'self'. Using anchor {target_anchor_uid} as primary target.")
                     target_final_uid = target_anchor_uid
                     
                     # Construct Top-10: 1st is the anchor itself
                     # 2-10 are top global matches
                     
                     stats["top_10_candidates"] = []
                     
                     # 1. Add self (the anchor)
                     anchor_obj = next((o for o in obj_map if str(getattr(o, 'uid', '')) == target_anchor_uid), None)
                     if anchor_obj:
                         clip_query = target_visual_description if target_visual_description else target_object_class
                         # Compute score for self just for consistency (optional)
                         self_score = 1.0 # Or compute real score
                         
                         candidate_info = {
                             "uid": target_anchor_uid,
                             "score": float(self_score),
                             "class_id": getattr(anchor_obj, "class_id", -1),
                             "class_name": "unknown"
                         }
                         bbox = getattr(anchor_obj, "bbox", None)
                         if bbox:
                             candidate_info["bbox"] = {
                                 "center": bbox.get_center().tolist(),
                                 "sizes": bbox.get_extent().tolist()
                             }
                         stats["top_10_candidates"].append(candidate_info)
                     
                     # 2. Fill remaining from global search
                     needed = 10 - len(stats["top_10_candidates"])
                     if needed > 0:
                         clip_query = target_visual_description if target_visual_description else target_object_class
                         print(f"[Runner] Filling {needed} additional candidates from global search for query '{clip_query}'")
                         
                         global_matches = find_objects_in_map_by_clip_description(
                            clip_query, obj_map, clip_model, clip_tokenizer, device, similarity_threshold=0.0
                         )
                         
                         existing_uids = {c["uid"] for c in stats["top_10_candidates"]}
                         
                         for obj, score in global_matches:
                             uid = str(getattr(obj, "uid", ""))
                             if uid not in existing_uids:
                                 candidate_info = {
                                     "uid": uid,
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
                                 stats["top_10_candidates"].append(candidate_info)
                                 existing_uids.add(uid)
                                 if len(stats["top_10_candidates"]) >= 10:
                                     break

                if target_final_uid:
                    print(f"[Runner] Final visualization target UID: {target_final_uid}")
                    # ... then visualize_anchor_in_pointcloud(target_final_uid)
                    
                    if best_frame and os.path.exists(best_frame):
                        safe_task = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in task[:50])
                        safe_task = safe_task.replace(' ', '_')
                        prefix = f"task_{task_index:03d}_" if task_index > 0 else ""
                        output_filename = f"{prefix}task_result_anchor_{target_anchor_id}_{safe_task}.png"
                        output_path = output_dir / output_filename
                        import shutil
                        shutil.copy2(best_frame, output_path)
                        print(f"[Runner] Saved result image to: {output_path}")
                        
                        if merged_cfg:
                            layout_pcd_path = None
                            if hasattr(merged_cfg, 'map_save_path'):
                                layout_pcd_path = os.path.join(merged_cfg.map_save_path, "layout.pcd")
                                if not os.path.exists(layout_pcd_path):
                                    # preload_path
                                    if hasattr(merged_cfg, 'preload_path') and merged_cfg.preload_path:
                                        layout_pcd_path = os.path.join(merged_cfg.preload_path, "layout.pcd")
                            
                            if layout_pcd_path and os.path.exists(layout_pcd_path) and obj_map:
                                pcd_output_path = visualize_anchor_in_pointcloud(
                                    layout_pcd_path,
                                    target_final_uid,  # refined object, not always anchor
                                    obj_map,
                                    output_dir,
                                    task,
                                    anchor_id_map,
                                    task_index=task_index,
                                )
                                if pcd_output_path:
                                    print(f"[Runner] Saved point cloud visualization to: {pcd_output_path}")
                            else:
                                print(f"[Runner] Skipping point cloud visualization (layout.pcd not found or obj_map empty)")
                        
                        stats["success"] = True
                        stats["completed_by"] = "VLM"
                        stats["duration"] = time.time() - task_start_time
                        stats["best_frame_path"] = best_frame  # Save original best_frame_path
                        stats["saved_image_path"] = str(output_path)  # Save saved image path
                        # Save VLM response info for candidate re-ranking in evaluation
                        stats["target_relation"] = target_relation
                        stats["target_anchor_id"] = target_anchor_id
                        stats["target_anchor_uid"] = target_anchor_uid
                        stats["target_object_class"] = target_object_class
                        stats["target_visual_description"] = target_visual_description
                        # Save viewed_uids to exclude them in global CLIP search
                        stats["viewed_uids"] = list(viewed_uids)
                        return (True, str(output_path), stats)
                    else:
                        print(f"[Runner] WARNING: Best frame {best_frame} not found.")
                else:
                    print(f"[Runner] ERROR: Could not find UID for anchor ID {target_anchor_id}")
            else:
                if not best_frame:
                    print("[Runner] WARNING: task_completed true but no best_frame_path provided.")
                if not target_anchor_id:
                    print("[Runner] WARNING: task_completed true but no target_anchor_id provided.")
                # Even if task_completed but no best_frame, still save best_frame_path if available
                if best_frame:
                    stats["best_frame_path"] = best_frame
                # Save VLM response info even if no best_frame
                stats["target_relation"] = target_relation
                stats["target_anchor_id"] = target_anchor_id
                if target_anchor_uid:
                    stats["target_anchor_uid"] = target_anchor_uid
                stats["target_object_class"] = target_object_class
                stats["target_visual_description"] = target_visual_description
                # Save viewed_uids to exclude them in global CLIP search
                stats["viewed_uids"] = list(viewed_uids)
            return (True, None, stats)

        # Even if task_completed=false, save target_object_class if VLM extracted it
        # This helps with CLIP fallback matching
        if not inspection_json.get("task_completed") and inspection_json:
            target_object_class = inspection_json.get("target_object_class", "")
            target_visual_description = inspection_json.get("target_visual_description", "")
            if target_object_class:
                stats["target_object_class"] = target_object_class
                if target_visual_description:
                    stats["target_visual_description"] = target_visual_description
                print(f"[Runner] VLM extracted target class: {target_object_class} (task not completed)")

        print("[Runner] Task not completed with this anchor, continuing...\n")

    print("[Runner] All suggested anchors processed, but task remains unresolved.")
    
    # --- Stage 3: Global CLIP Fallback ---
    # Even if VLM succeeded or failed, we compute Top-10 Global CLIP candidates for evaluation
    # as requested by the user for "Recall@10" evaluation.
    # Logic Update: Only compute global top 10 if we haven't already populated stats["top_10_candidates"]
    # (i.e. if VLM failed or didn't select candidates).
    
    if obj_map and clip_model and "top_10_candidates" not in stats:
        # 
        # Prefer using target_object_class if available, otherwise use task description
        target_class = stats.get("target_object_class", "")
        target_visual_desc = stats.get("target_visual_description", "")
        clip_query_for_top10 = target_visual_desc if target_visual_desc else (target_class if target_class else task)
        
        matches = find_objects_in_map_by_clip_description(
            clip_query_for_top10, obj_map, clip_model, clip_tokenizer, device, similarity_threshold=0.0
        )
        
        top_10 = matches[:10]
        stats["top_10_candidates"] = []
        
        for obj, score in top_10:
            candidate_info = {
                "uid": str(getattr(obj, "uid", "")),
                "score": float(score),
                "class_id": getattr(obj, "class_id", -1),
                "class_name": "unknown"
            }
            
            # Get class name if available
            if hasattr(obj, "class_id") and hasattr(BaseObject, "obj_classes"):
                 # BaseObject might not have obj_classes directly accessible like this
                 # Usually visualizer has it. Let's try to get it if possible, or skip.
                 pass

            # Add BBox info
            bbox = getattr(obj, "bbox", None)
            if bbox:
                candidate_info["bbox"] = {
                    "center": bbox.get_center().tolist(),
                    "sizes": bbox.get_extent().tolist()
                }
            
            stats["top_10_candidates"].append(candidate_info)
            
        print(f"[Runner] Computed Top-10 Global CLIP candidates for evaluation (VLM failed or no candidates).")

    if not stats["success"] and obj_map and clip_model:
        print(f"\n{'='*80}")
        print("[Runner] Entering Global CLIP Fallback Stage")
        print(f"{'='*80}")
        print("[Runner] VLM failed to find target in selected anchors.")
        
        # Prefer using target_object_class if VLM extracted it, otherwise use task description
        target_class = stats.get("target_object_class", "")
        target_visual_desc = stats.get("target_visual_description", "")
        
        # Build CLIP query: prefer visual description, then class, then full task
        if target_visual_desc:
            clip_query = target_visual_desc
            print(f"[Runner] Using VLM-extracted visual description for CLIP: '{clip_query}'")
        elif target_class:
            clip_query = target_class
            print(f"[Runner] Using VLM-extracted target class for CLIP: '{clip_query}'")
        else:
            clip_query = task
            print(f"[Runner] Using full task description for CLIP: '{clip_query}'")
        
        print(f"[Runner] Attempting to match against ALL objects using CLIP...")
        
        try:
            #  ()
            text_tokens = clip_tokenizer([clip_query]).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            
            best_sim = -1.0
            best_uid = None
            
            # 
            skipped_count = 0
            for obj in obj_map:
                # Filter out objects already viewed by VLM
                if str(getattr(obj, 'uid', '')) in viewed_uids:
                    skipped_count += 1
                    continue

                if hasattr(obj, 'clip_ft') and obj.clip_ft is not None:
                    # 
                    obj_feat = torch.from_numpy(obj.clip_ft).to(device).float().unsqueeze(0)
                    obj_feat /= obj_feat.norm(dim=-1, keepdim=True)
                    sim = (text_features @ obj_feat.T).item()
                    
                    if sim > best_sim:
                        best_sim = sim
                        best_uid = str(getattr(obj, 'uid', ''))
            
            # Always build top_10_candidates from global CLIP search, even if score is low
            # This ensures evaluation always has candidates to work with
            if best_uid:
                print(f"[Runner] Global CLIP found best match: UID {best_uid} (score {best_sim:.4f})")
                
                # Build top_10_candidates from global CLIP search
                if "top_10_candidates" not in stats:
                    stats["top_10_candidates"] = []
                
                # Get top 10 matches from global search (use same query as above)
                global_matches = find_objects_in_map_by_clip_description(
                    clip_query, obj_map, clip_model, clip_tokenizer, device, similarity_threshold=0.0
                )
                
                existing_uids = {c["uid"] for c in stats["top_10_candidates"]}
                for obj, score in global_matches:
                    uid = str(getattr(obj, "uid", ""))
                    if uid not in existing_uids:
                        candidate_info = {
                            "uid": uid,
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
                        stats["top_10_candidates"].append(candidate_info)
                        if len(stats["top_10_candidates"]) >= 10:
                            break
                
                # If we still have no candidates, at least add the best match
                if not stats["top_10_candidates"] and best_uid:
                    best_obj = next((o for o in obj_map if str(getattr(o, 'uid', '')) == best_uid), None)
                    if best_obj:
                        candidate_info = {
                            "uid": best_uid,
                            "score": float(best_sim),
                            "class_id": getattr(best_obj, "class_id", -1),
                            "class_name": "unknown"
                        }
                        bbox = getattr(best_obj, "bbox", None)
                        if bbox:
                            candidate_info["bbox"] = {
                                "center": bbox.get_center().tolist(),
                                "sizes": bbox.get_extent().tolist()
                            }
                        stats["top_10_candidates"].append(candidate_info)
                
                # Only mark as success if score is above threshold
                if best_sim > 0.25:
                    # 
                    if merged_cfg:
                        layout_pcd_path = None
                        if hasattr(merged_cfg, 'map_save_path'):
                            layout_pcd_path = os.path.join(merged_cfg.map_save_path, "layout.pcd")
                            if not os.path.exists(layout_pcd_path):
                                if hasattr(merged_cfg, 'preload_path') and merged_cfg.preload_path:
                                    layout_pcd_path = os.path.join(merged_cfg.preload_path, "layout.pcd")
                        
                        if layout_pcd_path and os.path.exists(layout_pcd_path):
                            pcd_output_path = visualize_anchor_in_pointcloud(
                                layout_pcd_path,
                                best_uid,
                                obj_map,
                            output_dir,
                            task,
                            anchor_id_map,
                            suffix="_fallback_clip",
                            task_index=task_index,
                        )
                        if pcd_output_path:
                            print(f"[Runner] Saved fallback visualization to: {pcd_output_path}")
                            stats["success"] = True
                            stats["completed_by"] = "CLIP"
                            stats["duration"] = time.time() - task_start_time
                            return (True, str(pcd_output_path), stats)
                else:
                    print(f"[Runner] Global CLIP search found match but score {best_sim:.4f} below threshold (0.25). Still returning candidates for evaluation.")
            else:
                print(f"[Runner] Global CLIP search found no objects.")
                
        except Exception as e:
            print(f"[Runner] Global CLIP fallback failed with error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    stats["duration"] = time.time() - task_start_time
    # Even if task failed, save best_frame_path if it was found during inspection
    # (This can happen if task_completed=false but VLM still selected a best frame)
    # Note: best_frame is only set if task_completed=true, so this is mainly for consistency
    # Save viewed_uids to exclude them in global CLIP search
    stats["viewed_uids"] = list(viewed_uids)
    return (False, None, stats)


class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def main() -> int:
    # Load configuration
    dm = dualmap_root()
    cfg = OmegaConf.load(str(dm / "config" / "base_config.yaml"))

    # Get anchor key path from config
    if "representative_frames_path" in cfg:
        anchor_key_path = os.path.join(cfg.representative_frames_path, "anchor_key")
        print(f"[Runner] Using anchor_key_path from config: {anchor_key_path}")
    else:
        # Fallback if config is missing (for safety during dev)
        anchor_key_path = str(
            dm / "dataset" / "HM3D_hovsg" / "00824-Dd4bFSTQ8gi" / "representative_frames" / "anchor_key"
        )
        print(f"[Runner] WARNING: representative_frames_path not found in config, using fallback: {anchor_key_path}")
    default_anchor_file = os.path.join(anchor_key_path, "anchor_attached_objects.txt")
    default_manifest = os.path.join(anchor_key_path, "manifest.json")

    parser = argparse.ArgumentParser(description="Run anchor selection and inspection in one script.")
    parser.add_argument("task", nargs="?", help="User task / question (or use --tasks-file for batch processing)")
    parser.add_argument("--tasks-file", type=Path, help="File containing tasks (one per line, numbered lines like '1) task' are supported)")
    parser.add_argument("--anchor-file", type=Path, default=Path(default_anchor_file), help="Path to anchor_attached_objects.txt")
    parser.add_argument("--manifest", type=Path, default=Path(default_manifest), help="Path to anchor manifest JSON")
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
        dest="llm_model",
        help="OpenAI LLM for anchor selection (JSON)",
    )
    parser.add_argument(
        "--vlm-model",
        default=DEFAULT_VLM_MODEL,
        dest="vlm_model",
        help="Gemini model for visual inspection",
    )
    parser.add_argument(
        "--gemini-api-key",
        dest="gemini_api_key",
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY / GOOGLE_API_KEY)",
    )
    parser.add_argument("--api-key", dest="api_key", default=None, help="API key override")
    parser.add_argument("--base-url", dest="base_url", default=None, help=f"API base URL (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()

    # 
    tasks = []
    if args.tasks_file:
        # 
        if not args.tasks_file.exists():
            print(f"Error: Tasks file not found: {args.tasks_file}", file=sys.stderr)
            return 1
        with args.tasks_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Numbered task lines
                if ")" in line and line[0].isdigit():
                    #  ")" 
                    idx = line.find(")")
                    if idx >= 0:
                        task = line[idx + 1:].strip()
                        if task:
                            tasks.append(task)
                else:
                    tasks.append(line)
    elif args.task:
        # 
        tasks = [args.task]
    else:
        print("Error: Either provide a task or use --tasks-file", file=sys.stderr)
        parser.print_help()
        return 1

    if not tasks:
        print("Error: No tasks found", file=sys.stderr)
        return 1

    print(f"[Runner] Processing {len(tasks)} task(s)")
    
    # Merge base_config + system_config; output under output_path
    base_cfg = OmegaConf.load(str(project_root / "config" / "base_config.yaml"))
    system_cfg = OmegaConf.load(str(project_root / "config" / "system_config.yaml"))
    # Defaults from base_cfg; overrides from system_cfg
    merged_cfg = OmegaConf.merge(base_cfg, system_cfg)
    
    # Init BaseObject from merged cfg
    BaseObject.initialize_config(merged_cfg)
    
    # 
    map_dir = merged_cfg.map_save_path
    print(f"[Runner] Loading map from: {map_dir}")
    obj_map = load_map_from_disk(map_dir)
    
    if not obj_map:
        print("[Runner] WARNING: No objects loaded from map. Will use FastSAM fallback.", file=sys.stderr)
    
    # Load CLIP (+ FastSAM if configured)
    print("[Runner] Loading CLIP model...")
    try:
        clip_model, clip_tokenizer, clip_preprocess, fastsam, device = load_clip_and_fastsam()
        print("[Runner] CLIP model loaded successfully.")
    except Exception as e:
        print(f"[Runner] Failed to load CLIP model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    #  manifest 
    manifest, saved_anchor_id_map = load_manifest(args.manifest)

    # Stage 1: anchor selection
    anchors = load_anchor_file(args.anchor_file)
    
    # 
    anchors_with_frames = []
    for anchor in anchors:
        anchor_id = anchor.get("id", "")
        entry = find_manifest_entry(manifest, anchor_id)
        if entry is not None:
            image_paths = collect_image_paths(entry)
            if image_paths:  # 
                anchors_with_frames.append(anchor)
            else:
                print(f"[Runner] Filtering out anchor {anchor_id} (no representative frames)", file=sys.stderr)
        else:
            print(f"[Runner] Filtering out anchor {anchor_id} (not found in manifest)", file=sys.stderr)
    
    if not anchors_with_frames:
        print("[Runner] No anchors with representative frames found.", file=sys.stderr)
        return 1

    # Short ids (1,2,3) for prompts
    # manifest anchor_id_map -> short ids
    anchor_id_map = {}  # UID -> simple_id
    anchor_id_map_short = {}  # UID -> simple_id (display)
    
    if saved_anchor_id_map:
        print(f"[Runner] Using saved anchor_id_map from manifest (ensures consistency with saved images)")
        print(f"[Runner] Loaded {len(saved_anchor_id_map)} anchor ID mappings from manifest")
        anchor_id_map = saved_anchor_id_map.copy()
        # ID
        for uid, simple_id in anchor_id_map.items():
            short_id = _short_anchor_id(uid)
            if short_id and short_id != uid:
                anchor_id_map_short[short_id] = simple_id
    else:
        # For each manifest uid
        print("[Runner] ERROR: No saved anchor_id_map found in manifest!")
        print("[Runner] This means the map was built with an old version that didn't save anchor_id_map.")
        print("[Runner] Please rebuild the map to generate a new manifest with anchor_id_map.")
        print("[Runner] Falling back to UID-sorted mapping (IDs may not match images):")
        manifest_anchors = []
        for uid, entry in manifest.items():
            image_paths = collect_image_paths(entry)
            if image_paths:  # 
                manifest_anchors.append((uid, entry))
        
        # Assign uid ->
        manifest_anchors.sort(key=lambda x: x[0])  # UID
        
        # stable simple_id
        for idx, (uid, entry) in enumerate(manifest_anchors, 1):
            anchor_id_map[uid] = idx
            # ID
            short_id = _short_anchor_id(uid)
            if short_id and short_id != uid:
                anchor_id_map_short[short_id] = idx
    
    print(f"[Runner] Created anchor ID mapping: {len(anchor_id_map)} anchors (by UID)")
    print(f"[Runner] Short ID mappings: {len(anchor_id_map_short)}")
    
    # 
    print("\n[Runner] Full anchor ID mapping:")
    for uid, simple_id in sorted(anchor_id_map.items(), key=lambda x: x[1]):
        short_id = _short_anchor_id(uid)
        print(f"  Simple ID {simple_id}: UID={uid}, short_id={short_id}")
    
    # 
    print("\n[Runner] Verifying mapping consistency...")
    for idx, anchor in enumerate(anchors_with_frames, 1):
        anchor_id = anchor.get("id", "")
        entry = find_manifest_entry(manifest, anchor_id)
        if entry:
            uid = entry.get("uid", anchor_id)
            mapped_id = anchor_id_map.get(uid, -1)
            if mapped_id != idx:
                print(f"[Runner] ERROR: Mapping inconsistency! Anchor {idx} -> UID {uid} -> mapped ID {mapped_id}")
    
    # Build obj_map
    observation_cache = None
    if merged_cfg.get("observation_cache", {}).get("enabled", False):
        cache_cfg = merged_cfg.observation_cache
        if cache_cfg.get("mode", "").lower() in ("load", "both"):
            observation_cache = LocalObservationCache(cache_cfg)
            print(f"[Runner] Observation cache enabled: {cache_cfg.get('output_dir', '')}")

    # 
    output_dir = dm / "output" / "query_results"
    if output_dir.exists():
        import shutil
        print(f"[Runner] Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 
    log_path = output_dir / "run.log"
    print(f"[Runner] Logging to: {log_path}")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

       # 
    success_count = 0
    failed_tasks = []
    all_stats = []
    
    import time

    # JSON: NumpyEncoder
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
        task_start_time = time.time()
        print(f"\n{'#'*80}")
        print(f"# Task {i}/{len(tasks)}")
        print(f"{'#'*80}")
        
        success, saved_path, stats = process_single_task(
            task,
            anchors_with_frames,
            manifest,
            output_dir,
            args.llm_model,
            args.vlm_model,
            args.base_url,
            args.api_key,
            clip_model,
            clip_tokenizer,
            clip_preprocess,
            fastsam,
            device,
            obj_map,
            anchor_id_map,
            observation_cache,
            merged_cfg,
            task_index=i,
            gemini_api_key=args.gemini_api_key,
        )
        
        all_stats.append(stats)
        
        task_end_time = time.time()
        duration = task_end_time - task_start_time
        print(f"[Runner] Task {i} processing time: {duration:.2f} seconds")
        
        if success:
            success_count += 1
            if saved_path:
                print(f"[Runner] Task {i} completed successfully. Image saved: {saved_path}")
            else:
                print(f"[Runner] Task {i} completed but no image was saved.")
        else:
            failed_tasks.append((i, task))
            print(f"[Runner] Task {i} failed: {task}")

        # === write predictions.json ===
        try:
            with open(predictions_path, "w") as f:
                json.dump(all_stats, f, indent=4, cls=NumpyEncoder)
            print(f"[Runner] Incrementally saved predictions and stats to: {predictions_path}")
        except Exception as e:
            print(f"[Runner] Failed to save predictions.json after task {i}: {e}", file=sys.stderr)

    # 
    print(f"\n{'='*80}")
    print(f"[Runner] Batch processing complete!")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(failed_tasks)}")
    
    # Save all stats (including Top-10 predictions) to JSON
    try:
        with open(predictions_path, "w") as f:
            json.dump(all_stats, f, indent=4, cls=NumpyEncoder)
        print(f"[Runner] Saved predictions and stats to: {predictions_path}")
    except Exception as e:
        print(f"[Runner] Failed to save predictions.json: {e}", file=sys.stderr)
    
    # 
    print(f"\n{'='*130}")
    print(f"{'ID':<4} | {'Status':<10} | {'Method':<15} | {'Anchors':<8} | {'Frames':<8} | {'Time (s)':<10} | {'Task':<40}")
    print(f"{'-'*130}")
    
    total_time = 0.0
    total_frames = 0
    
    for i, stats in enumerate(all_stats, 1):
        status = "Success" if stats["success"] else "Failed"
        method = stats["completed_by"]
        anchors = stats["anchors_viewed"]
        frames = stats["frames_viewed"]
        duration = stats["duration"]
        task_raw = stats["task"]
        task_short = task_raw[:37] + "..." if len(task_raw) > 37 else task_raw
        
        total_time += duration
        total_frames += frames
        
        print(f"{i:<4} | {status:<10} | {method:<15} | {anchors:<8} | {frames:<8} | {duration:<10.2f} | {task_short:<40}")
    
    print(f"{'='*130}")
    
    avg_time = total_time / len(tasks) if tasks else 0
    avg_frames = total_frames / len(tasks) if tasks else 0
    
    print(f"\nSummary Statistics:")
    print(f"  Total Time:     {total_time:.2f}s")
    print(f"  Average Time:   {avg_time:.2f}s/task")
    print(f"  Average Frames: {avg_frames:.1f} frames/task")
    
    if failed_tasks:
        print(f"\nFailed tasks:")
        for idx, task in failed_tasks:
            print(f"  {idx}. {task}")
    print(f"{'='*80}")
    
    return 0 if success_count > 0 else 1


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

