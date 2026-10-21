#!/usr/bin/env python3
from __future__ import annotations
"""
Navigation-focused anchor task runner (top-1 retrieval).

Variant of anchor_task_runner.py tuned for navigation-style queries.

Usage:
    python scripts/anchor_task_runner_nav.py "find the red book"
"""

import argparse
import base64
import json
import os
import time
from datetime import datetime
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set matplotlib backend to Agg to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

# sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def dualmap_root() -> Path:
    """Use this checkout by default; retain the legacy root override."""
    env = os.environ.get("OASIS_ROOT") or os.environ.get("DUALMAP_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return project_root.resolve()


from omegaconf import OmegaConf
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import open_clip
import open3d as o3d

from utils.object import BaseObject
from utils.visualizer_ui import show_visualization_window
from utils.vlm_prompt_transcript import (
    DEFAULT_TASK_PROMPT_DIR,
    append_transcript,
    get_path_from_env,
    init_standalone_nav_file,
)

DEFAULT_MODEL = "gpt-5.2"
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# prompt
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
You must select AT MOST {top_k} anchors. Prioritize the most promising candidates.
**CRITICAL PRIORITY: If the target object (or a semantically similar object) appears in an anchor's "Attached objects" list, you MUST select and prioritize that anchor above others.**
If no anchor seems directly relevant, use common sense to select the anchor that is SEMANTICALLY most likely to contain or be near the target object (e.g., look for 'toothbrush' near 'sink', 'pillow' on 'bed'). Do NOT simply select the anchor with the most objects.
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

STRICT RULES:

1. **TARGET IDENTIFICATION (Prioritize Recall)**:
   - **Visual Confirmation**: You must see the object, but for **Small/Blurry Objects**, do NOT require HD clarity.
   - **Broad Queries**: If the task only gives a class name (e.g., "find the box") without detailed attributes, ANY object matching that class/shape should be considered a match.
   - **Semantic Consistency**: "Picture" = "Framed Photo". Do not fail just because of synonym differences. If it looks like the target, mark `task_completed=true`.

2. **SPATIAL & CONTEXT**:
   - **Self Relation**: If the anchor ITSELF is the target (e.g. task "find table", and anchor is a table), use `relation='self'` and `target_anchor_id=anchor_id`.
   - **Room Verification**: If task specifies a room (e.g. "bathroom"), finding the object in the wrong room is a FAILURE.

3. **OUTPUT INTEGRITY**:
   - **Success**: If true, MUST provide `target_anchor_id` and `best_frame_path` (filename only).
   - **Failure**: If false, do NOT invent visual details. Leave `target_visual_description` empty or just use the class name.
   - **Always Extract**: Always fill `target_object_class` based on the text, even if visual search failed."""

# 
def _short_anchor_id(full_id: str) -> str:
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
                            count_part = parts[1].strip()
                            count_str = count_part.split('(')[0].strip()
                            count = int(count_str)
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
                            count_part = parts[1].strip()
                            count_str = count_part.split('(')[0].strip()
                            count = int(count_str)
                            uids = []
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
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "anchor_id_map" in data:
        anchor_id_map = data.get("anchor_id_map", {})
        anchors = data.get("anchors", {})
        return anchors, anchor_id_map
    else:
        return data, None

def find_manifest_entry(manifest: Dict[str, Dict], token: str) -> Optional[Dict]:
    anchors = manifest.get("anchors", manifest) if isinstance(manifest, dict) and "anchors" in manifest else manifest
    if token in anchors:
        return anchors[token]
    for uid, entry in anchors.items():
        if _short_anchor_id(uid) == token:
            return entry
    return None

def collect_image_paths(entry: Dict) -> List[str]:
    saved_images = entry.get("saved_images") or []
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

def build_inspection_prompt(task: str, entry: Dict, image_paths: List[str], anchor_id_map: Dict[str, int]) -> str:
    lines = [
        f"Task: {task}",
        "",
        "Provided images:",
    ]
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

def call_inspection(model: str, user_prompt: str, image_paths: List[str], *, base_url: Optional[str], api_key: Optional[str]) -> str:
    if base_url is None:
        base_url = DEFAULT_BASE_URL
    if api_key is None:
        api_key = DEFAULT_API_KEY
    client = OpenAI(base_url=base_url, api_key=api_key)
    content = [{"type": "input_text", "text": user_prompt}]
    for path in image_paths:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{encode_image(Path(path))}",
        })
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": INSPECTION_SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ],
    )
    text = _extract_text_from_response(response)
    if not text:
        raise RuntimeError("Empty response from GPT (inspection)")
    return text

def _extract_text_from_response(response) -> str:
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


def find_objects_in_map_by_clip_description(
    clip_description: str,
    obj_map: List[BaseObject],
    clip_model,
    clip_tokenizer,
    device: str = "cuda",
    similarity_threshold: float = 0.2,
) -> List[Tuple[BaseObject, float]]:
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

def process_single_task_nav(
    task: str,
    anchors_with_frames: List[Dict],
    manifest: Dict[str, Dict],
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    clip_model=None,
    clip_tokenizer=None,
    device: str = "cuda",
    obj_map: Optional[List[BaseObject]] = None,
    anchor_id_map: Optional[Dict[str, int]] = None,
    visualize: bool = False,
    viz_mode: str = "auto",
    top_k: int = 3,
    prompt_transcript_path: Optional[str] = None,
    scene_graph_io_ms: float = 0.0,
) -> Dict:
    """
    Navigation retrieval pipeline: LLM/VLM/CLIP stages, top-1 target.

    Returns:
        Dict:
        - success: bool
        - target_anchor_uid: str (anchor UID)
        - target_final_uid: str (resolved object UID)
        - best_frame_path: str (representative frame path)
        - frame_idx: int or None
        - method: str ("VLM" or "CLIP")
        - nav_retrieval_timing_ms: dict — per-stage timings in ms
    """
    local_prep_ms = 0.0
    timing_llm_ms = 0.0
    timing_vlm_ms = 0.0
    timing_clip_local_ms = 0.0

    def _attach_nav_timing() -> None:
        result["nav_retrieval_timing_ms"] = {
            "local_scene_graph_ms": float(scene_graph_io_ms + local_prep_ms + timing_clip_local_ms),
            "llm_anchor_proposal_ms": float(timing_llm_ms),
            "vlm_verification_ms": float(timing_vlm_ms),
            "local_clip_refinement_ms": float(timing_clip_local_ms),
            "compact_anchor_io_ms": float(scene_graph_io_ms),
            "prompt_build_ms": float(local_prep_ms),
            "runner_subtotal_ms": float(
                scene_graph_io_ms + local_prep_ms + timing_llm_ms + timing_vlm_ms + timing_clip_local_ms
            ),
        }

    result = {
        "success": False,
        "target_anchor_uid": None,
        "target_final_uid": None,
        "best_frame_path": None,
        "frame_idx": None,
        "method": "None"
    }

    if prompt_transcript_path:
        envp = get_path_from_env()
        if envp and os.path.abspath(envp) == os.path.abspath(prompt_transcript_path):
            append_transcript(
                prompt_transcript_path,
                f"========== NavRunner: initial retrieval ==========\n(task: {task}, model: {model})",
            )
        else:
            init_standalone_nav_file(prompt_transcript_path, task, model)

    print(f"[NavRunner] Task: {task}")
    print(f"[NavRunner] Top-K Anchors: {top_k}")
    print("-" * 80)
    print("[NavRunner] Stage 1: Anchor Selection (LLM)")
    print("-" * 80)
    
    t_prompt = time.perf_counter()
    selection_prompt = build_selection_prompt(anchors_with_frames, task)
    formatted_system_prompt = SELECTION_SYSTEM_PROMPT.replace("{top_k}", str(top_k))
    local_prep_ms = (time.perf_counter() - t_prompt) * 1000.0

    print(f"[NavRunner] Sending selection request to LLM (model: {model})...")
    print(f"[NavRunner] Selection prompt preview (first 200 chars): {selection_prompt[:200]}...")
    
    t_llm = time.perf_counter()
    selection_response = call_chat_json(model, formatted_system_prompt, selection_prompt, base_url=base_url, api_key=api_key)
    timing_llm_ms = (time.perf_counter() - t_llm) * 1000.0
    print(f"[NavRunner] Received selection response from LLM")
    # Full LLM response for debugging
    print(f"[NavRunner] Full Selection Response: {selection_response}")

    if prompt_transcript_path:
        append_transcript(
            prompt_transcript_path,
            "========== Stage 1: Anchor Selection (LLM) ==========\n\n"
            "[SYSTEM]\n"
            + formatted_system_prompt
            + "\n\n[USER]\n"
            + selection_prompt
            + "\n\n[ASSISTANT]\n"
            + selection_response,
        )

    try:
        selection_json = json.loads(selection_response)
        
        if visualize:
            simple_prompts = f"TASK: {task}\n\nSTEP: Anchor Selection (LLM)\n\nFinding best candidates from map..."
            
            anchors_view = selection_json.get("anchors_to_view", [])
            reason = selection_json.get("reasoning", "")
            
            simple_resp = f"SELECTED ANCHORS: {anchors_view}\n\nREASONING:\n{reason}"
            
            #  (5)
            viz_items = []
            max_viz_anchors = 5
            display_anchors = anchors_with_frames[:max_viz_anchors]
            
            for anchor in display_anchors:
                short_id = _short_anchor_id(anchor.get("id", "unknown"))
                category = anchor.get("category", "unknown")
                
                #  Attached Objects
                attached = anchor.get("attached_objects", [])
                attached_str = ", ".join([f"{item['category']} x{item['count']}" for item in attached]) if attached else "(none)"
                
                #  Nearby Objects
                nearby = anchor.get("nearby_objects", [])
                nearby_str = ", ".join([f"{item['category']} x{item['count']}" for item in nearby]) if nearby else "(none)"
                
                info_text = f"Category: {category}\nAttached: {attached_str}\nNearby: {nearby_str}"
                
                viz_items.append({
                    "type": "text", 
                    "label": f"Anchor {short_id} ({category})",
                    "text_content": info_text
                })
            
            if len(anchors_with_frames) > max_viz_anchors:
                remaining = len(anchors_with_frames) - max_viz_anchors
                viz_items.append({
                    "type": "text",
                    "label": "...",
                    "text_content": f"... and {remaining} more anchors available in the map."
                })
            
            show_visualization_window(
                title=f"Stage 1: Anchor Selection - {task}",
                images=viz_items, # Pass anchor text cards
                prompts=simple_prompts,
                response=simple_resp,
                mode=viz_mode
            )
            
        reasoning = selection_json.get("reasoning", "")
        if reasoning:
            print(f"[NavRunner] VLM Reasoning: {reasoning}")
    except json.JSONDecodeError:
        print(f"[NavRunner] Failed to parse selection response as JSON")
        if prompt_transcript_path:
            append_transcript(
                prompt_transcript_path,
                "[NOTE] Selection response could not be parsed as JSON (see Stage 1 [ASSISTANT] above).",
            )
        _attach_nav_timing()
        return result

    anchors_to_view = selection_json.get("anchors_to_view") or []
    print(f"[NavRunner] VLM selected anchors to view: {anchors_to_view}")
    if not anchors_to_view:
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

    max_anchors_to_view = top_k
    if len(anchors_to_view) > max_anchors_to_view:
        anchors_to_view = anchors_to_view[:max_anchors_to_view]

    viewed_uids = set()

    # Stage 2: Inspection
    print()
    print("-" * 80)
    print("[NavRunner] Stage 2: Image Inspection (VLM)")
    print("-" * 80)
    
    for anchor_idx, anchor_token in enumerate(anchors_to_view):
        print(f"\n[NavRunner] Inspecting anchor {anchor_idx + 1}/{len(anchors_to_view)}: {anchor_token}")
        
        entry = find_manifest_entry(manifest, anchor_token)
        if entry is None:
            print(f"[NavRunner]   Warning: Manifest entry not found for anchor {anchor_token}")
            continue
        image_paths = collect_image_paths(entry)
        if not image_paths:
            print(f"[NavRunner]   Warning: No representative frames found for anchor {anchor_token}")
            continue
        
        print(f"[NavRunner]   Found {len(image_paths)} representative frame(s):")
        for img_path in image_paths[:3]:  # 3
            print(f"[NavRunner]     - {img_path}")
        if len(image_paths) > 3:
            print(f"[NavRunner]     ... and {len(image_paths) - 3} more")
        
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

        inspection_prompt = build_inspection_prompt(task, entry, image_paths, anchor_id_map or {})
        print(f"[NavRunner]   Sending inspection request to VLM with {len(image_paths)} image(s)...")
        print(f"[NavRunner]   Inspection prompt preview: {inspection_prompt[:200]}...")

        try:
            t_vlm = time.perf_counter()
            inspection_response = call_inspection(
                model,
                inspection_prompt,
                image_paths,
                base_url=base_url,
                api_key=api_key,
            )
            timing_vlm_ms += (time.perf_counter() - t_vlm) * 1000.0
            print(f"[NavRunner]   Received inspection response from VLM")
            print(f"[NavRunner]   Full Inspection Response: {inspection_response}")

            if prompt_transcript_path:
                img_lines = "\n".join(f"  - {os.path.basename(p)}" for p in image_paths)
                append_transcript(
                    prompt_transcript_path,
                    "========== Stage 2: Image Inspection (VLM) — anchor "
                    + str(anchor_token)
                    + " ==========\n"
                    "(API request included the following images; image bytes are NOT stored in this file.)\n"
                    + img_lines
                    + "\n\n[SYSTEM]\n"
                    + INSPECTION_SYSTEM_PROMPT
                    + "\n\n[USER]\n"
                    + inspection_prompt
                    + "\n\n[ASSISTANT]\n"
                    + inspection_response,
                )

            inspection_json_temp = None
            try:
                inspection_json_temp = json.loads(inspection_response)
            except:
                pass # Will handle later

            if visualize:
                viz_images = []
                for p in image_paths:
                    if os.path.exists(p):
                        viz_images.append({"path": p, "label": os.path.basename(p)})
                
                simple_prompts = f"TASK: {task}\n\nSTEP: Inspecting Anchor {anchor_token}\n\nChecking representative frames..."
                
                simple_resp = ""
                if inspection_json_temp:
                    completed = inspection_json_temp.get("task_completed", False)
                    icon = "✅" if completed else "❌"
                    simple_resp += f"TASK COMPLETED: {completed} {icon}\n"
                    
                    if completed:
                        simple_resp += f"TARGET ANCHOR: {inspection_json_temp.get('target_anchor_id')}\n"
                        simple_resp += f"RELATION: {inspection_json_temp.get('target_relation')}\n"
                        simple_resp += f"CLASS: {inspection_json_temp.get('target_object_class')}\n"
                    
                    simple_resp += f"\nREASONING:\n{inspection_json_temp.get('reasoning', '')}"
                else:
                    simple_resp = inspection_response # Fallback

                show_visualization_window(
                    title=f"Stage 2: Inspection (Anchor {anchor_idx+1}/{len(anchors_to_view)}) - {task}",
                    images=viz_images,
                    prompts=simple_prompts,
                    response=simple_resp,
                    mode=viz_mode
                )

        except Exception as e:
            print(f"[NavRunner]   Error during inspection: {e}")
            if prompt_transcript_path:
                append_transcript(
                    prompt_transcript_path,
                    "========== Stage 2: Image Inspection (VLM) — anchor "
                    + str(anchor_token)
                    + " ==========\n[ERROR]\n"
                    + repr(e),
                )
            continue

        inspection_json = None
        try:
            inspection_json = json.loads(inspection_response)
        except json.JSONDecodeError:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', inspection_response, re.DOTALL)
            if json_match:
                try:
                    inspection_json = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            if inspection_json is None:
                first_brace = inspection_response.find('{')
                last_brace = inspection_response.rfind('}')
                if first_brace >= 0 and last_brace > first_brace:
                    json_candidate = inspection_response[first_brace:last_brace+1]
                    try:
                        inspection_json = json.loads(json_candidate)
                    except json.JSONDecodeError:
                        pass
            
            if inspection_json is None:
                continue

        if inspection_json.get("task_completed"):
            print(f"[NavRunner]   ✓ Task completed! VLM found the target.")
            reasoning = inspection_json.get("reasoning", "")
            if reasoning:
                print(f"[NavRunner]   VLM Reasoning: {reasoning}")
            
            best_frame_filename = inspection_json.get("best_frame_path", "")
            best_frame = None
            if best_frame_filename:
                for img_path in image_paths:
                    if os.path.basename(img_path) == best_frame_filename or img_path.endswith(best_frame_filename):
                        best_frame = img_path
                        break
                if not best_frame and image_paths:
                    base_dir = os.path.dirname(image_paths[0])
                    candidate_path = os.path.join(base_dir, best_frame_filename)
                    if os.path.exists(candidate_path):
                        best_frame = candidate_path
                    else:
                        base_name = best_frame_filename.replace("_with_task", "")
                        candidate_path = os.path.join(base_dir, base_name)
                        if os.path.exists(candidate_path):
                            best_frame = candidate_path
                        else:
                            best_frame = image_paths[0]
            
            target_anchor_id = inspection_json.get("target_anchor_id", "")
            target_relation = inspection_json.get("target_relation", "self")
            target_object_class = inspection_json.get("target_object_class", "")
            target_visual_description = inspection_json.get("target_visual_description", "")
            
            if best_frame and target_anchor_id:
                # UID
                target_anchor_uid = None
                for uid, simple_id in sorted(anchor_id_map.items(), key=lambda x: x[1]) if anchor_id_map else []:
                    if str(simple_id) == str(target_anchor_id):
                        target_anchor_uid = uid
                        break
                
                if not target_anchor_uid:
                    if prompt_transcript_path:
                        append_transcript(
                            prompt_transcript_path,
                            "[NOTE] task_completed but target_anchor_id did not match manifest anchor_id_map; aborting.",
                        )
                    _attach_nav_timing()
                    return result

                target_final_uid = target_anchor_uid
                clip_match_info = None  # filled if CLIP refines target
                
                if target_relation in ["on", "nearby"] and target_object_class and obj_map and clip_model:
                    print(f"[NavRunner]   Processing relation='{target_relation}', target_object_class='{target_object_class}', target_visual_description='{target_visual_description}'")
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
                            print(f"[NavRunner]   Searching in attached_objects: {len(source_list)} categories")
                        elif target_relation == "nearby":
                            source_list = current_anchor_dict.get("nearby_objects", [])
                            print(f"[NavRunner]   Searching in nearby_objects: {len(source_list)} categories")
                        
                        candidate_uids = []
                        for item in source_list:
                            if target_object_class.lower() in item.get("category", "").lower():
                                candidate_uids.extend(item.get("uids", []))
                        
                        if not candidate_uids:
                            print(f"[NavRunner]   No exact class match found, using fallback: all attached+nearby objects")
                            fallback_sources = []
                            fallback_sources.extend(current_anchor_dict.get("attached_objects", []))
                            fallback_sources.extend(current_anchor_dict.get("nearby_objects", []))
                            for item in fallback_sources:
                                candidate_uids.extend(item.get("uids", []))

                        print(f"[NavRunner]   Found {len(candidate_uids)} candidate UIDs for CLIP matching")

                        if candidate_uids:
                            clip_query = target_visual_description if target_visual_description else target_object_class
                            print(f"[NavRunner]   CLIP query: '{clip_query}'")
                            t_clip = time.perf_counter()
                            text_tokens = clip_tokenizer([clip_query]).to(device)
                            with torch.no_grad():
                                text_features = clip_model.encode_text(text_tokens)
                                text_features /= text_features.norm(dim=-1, keepdim=True)
                            
                            best_candidate = None
                            best_sim = -1.0
                            for uid in candidate_uids:
                                obj = next((o for o in obj_map if str(getattr(o, 'uid', '')) == uid), None)
                                
                                if not obj:
                                    print(f"[NavRunner]   Debug: UID {uid} not found in obj_map")
                                    continue
                                
                                has_feat = hasattr(obj, 'clip_ft')
                                clip_ft_val = getattr(obj, 'clip_ft', None)
                                is_valid_feat = clip_ft_val is not None
                                
                                if is_valid_feat and hasattr(clip_ft_val, 'size'):
                                    is_valid_feat = clip_ft_val.size > 0
                                elif is_valid_feat and isinstance(clip_ft_val, np.ndarray):
                                    is_valid_feat = clip_ft_val.size > 0
                                
                                if not (has_feat and is_valid_feat):
                                    print(f"[NavRunner]   Debug: UID {uid} has no valid clip_ft. has_feat={has_feat}, type={type(clip_ft_val)}, size={getattr(clip_ft_val, 'size', 'N/A') if clip_ft_val is not None else 'N/A'}")
                                    continue

                                try:
                                    obj_feat = torch.from_numpy(clip_ft_val).to(device).float().unsqueeze(0)
                                    obj_feat /= obj_feat.norm(dim=-1, keepdim=True)
                                    sim = (text_features @ obj_feat.T).item()
                                    
                                    # Get class name for logging
                                    curr_class_id = getattr(obj, 'class_id', None)
                                    curr_class_name = "unknown"
                                    try:
                                        if curr_class_id is not None and hasattr(BaseObject, "obj_classes"):
                                            curr_class_name = BaseObject.obj_classes[curr_class_id]
                                    except:
                                        pass
                                    print(f"[NavRunner]   Candidate: UID={uid}, class={curr_class_name}, similarity={sim:.4f}")
                                    
                                    if sim > best_sim:
                                        best_sim = sim
                                        best_candidate = obj
                                except Exception as e:
                                    print(f"[NavRunner]   Debug: Error computing similarity for UID {uid}: {e}")
                            timing_clip_local_ms += (time.perf_counter() - t_clip) * 1000.0
                            
                            if best_candidate:
                                target_final_uid = str(getattr(best_candidate, 'uid', ''))
                                best_class_id = getattr(best_candidate, 'class_id', None)
                                best_class_name = None
                                try:
                                    if best_class_id is not None and hasattr(BaseObject, "obj_classes"):
                                        best_class_name = BaseObject.obj_classes[best_class_id]
                                except Exception:
                                    best_class_name = None
                                print(f"[NavRunner]   VLM+CLIP matched target object: UID={target_final_uid}, class_id={best_class_id}, class_name={best_class_name}, similarity={best_sim:.4f}")
                                clip_match_info = {"uid": target_final_uid, "class_id": best_class_id, "class_name": best_class_name, "similarity": best_sim}
                            else:
                                print(f"[NavRunner]   ⚠️ CLIP matching failed: no valid candidate found (checked {len(candidate_uids)} UIDs)")
                                clip_match_info = {"status": "failed", "reason": "no_valid_candidate", "candidates_checked": len(candidate_uids)}
                        else:
                            print(f"[NavRunner]   ⚠️ No candidate UIDs found for relation '{target_relation}'")
                            clip_match_info = {"status": "failed", "reason": "no_candidates"}
                    else:
                        print(f"[NavRunner]   ⚠️ Could not find anchor dict for UID {target_anchor_uid}")
                        clip_match_info = {"status": "failed", "reason": "anchor_not_found"}
                elif target_relation == "self":
                    print(f"[NavRunner]   Relation is 'self': target is the anchor itself (UID={target_anchor_uid})")
                else:
                    print(f"[NavRunner]   ⚠️ Skipping CLIP matching: relation='{target_relation}', target_object_class={target_object_class}, obj_map={obj_map is not None}, clip_model={clip_model is not None}")
                
                # 
                frame_idx = None
                if best_frame:
                    match = re.search(r'frame_(\d+)', best_frame) or re.search(r'_(\d{6})_', best_frame)
                    if match:
                        frame_idx = int(match.group(1))
                
                # Resolve class names for logging
                anchor_class_name = None
                try:
                    current_anchor_entry = next(
                        (a for a in anchors_with_frames if a.get("short_id") == _short_anchor_id(target_anchor_uid) or a.get("id") == target_anchor_uid),
                        None
                    )
                    if current_anchor_entry is not None:
                        anchor_class_name = current_anchor_entry.get("category", None)
                except Exception:
                    anchor_class_name = None
                
                instance_class_id = None
                instance_class_name = None
                if clip_match_info and clip_match_info.get("class_id") is not None:
                    instance_class_id = clip_match_info.get("class_id")
                    instance_class_name = clip_match_info.get("class_name")
                else:
                    try:
                        if target_final_uid != target_anchor_uid:
                            best_obj = next((o for o in obj_map if str(getattr(o, 'uid', '')) == target_final_uid), None)
                            if best_obj is not None:
                                instance_class_id = getattr(best_obj, "class_id", None)
                                if instance_class_id is not None and hasattr(BaseObject, "obj_classes"):
                                    instance_class_name = BaseObject.obj_classes[instance_class_id]
                    except Exception:
                        instance_class_name = None

                # 
                summary_parts = [
                    f"relation={target_relation}",
                    f"target_object_class='{target_object_class}'",
                    f"target_visual_description='{target_visual_description}'",
                    f"anchor_uid={target_anchor_uid}",
                    f"anchor_class='{anchor_class_name}'",
                    f"final_uid={target_final_uid}",
                ]
                
                if instance_class_id is not None:
                    summary_parts.append(f"final_class_id={instance_class_id}")
                if instance_class_name:
                    summary_parts.append(f"final_class_name='{instance_class_name}'")
                elif target_relation in ["on", "nearby"]:
                    summary_parts.append("final_class_name=None (CLIP matching may have failed)")
                
                if clip_match_info and clip_match_info.get("similarity") is not None:
                    summary_parts.append(f"clip_similarity={clip_match_info.get('similarity'):.4f}")

                print(f"[NavRunner]   Final target summary: {', '.join(summary_parts)}")

                result["success"] = True
                result["target_anchor_uid"] = target_anchor_uid
                result["target_final_uid"] = target_final_uid
                result["best_frame_path"] = best_frame
                result["frame_idx"] = frame_idx
                result["method"] = "VLM"
                _attach_nav_timing()
                return result

    # Stage 3: Global CLIP Fallback (top-1)
    if not result["success"] and obj_map and clip_model:
        print()
        print("-" * 80)
        print("[NavRunner] Stage 3: CLIP Fallback")
        print("-" * 80)
        print(f"[NavRunner] VLM did not find target, using CLIP for global search...")
        
        target_class = None
        target_visual_desc = None
        
        # Use full task text as CLIP query in fallback stage
        clip_query = task
        print(f"[NavRunner] CLIP query: {clip_query}")
        
        t_clip3 = time.perf_counter()
        text_tokens = clip_tokenizer([clip_query]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        best_sim = -1.0
        best_obj = None
        
        for obj in obj_map:
            if str(getattr(obj, 'uid', '')) in viewed_uids:
                continue
            if hasattr(obj, 'clip_ft') and obj.clip_ft is not None:
                obj_feat = torch.from_numpy(obj.clip_ft).to(device).float().unsqueeze(0)
                obj_feat /= obj_feat.norm(dim=-1, keepdim=True)
                sim = (text_features @ obj_feat.T).item()
                if sim > best_sim:
                    best_sim = sim
                    best_obj = obj
        timing_clip_local_ms += (time.perf_counter() - t_clip3) * 1000.0
        
        if best_obj and best_sim > 0.25:
            best_uid = str(getattr(best_obj, 'uid', ''))
            best_class_id = getattr(best_obj, 'class_id', None)
            best_class_name = None
            try:
                if best_class_id is not None and hasattr(BaseObject, "obj_classes"):
                    best_class_name = BaseObject.obj_classes[best_class_id]
            except Exception:
                best_class_name = None
            print(f"[NavRunner] CLIP found best match with similarity: {best_sim:.4f}")
            print(f"[NavRunner] Global CLIP matched target object: UID={best_uid}, class_id={best_class_id}, class_name={best_class_name}")
            result["success"] = True
            result["target_anchor_uid"] = best_uid
            result["target_final_uid"] = best_uid
            result["method"] = "CLIP"
        else:
            print(f"[NavRunner] CLIP did not find a good match (best similarity: {best_sim:.4f}, threshold: 0.25)")

        if prompt_transcript_path:
            clip_lines = [
                "========== Stage 3: CLIP Fallback (no LLM text exchange) ==========",
                f"CLIP text query: {clip_query}",
            ]
            if result.get("success"):
                clip_lines.append(
                    f"result: matched UID={result.get('target_final_uid')}, method=CLIP"
                )
            else:
                clip_lines.append(
                    f"result: no match above threshold (best_similarity={best_sim:.4f})"
                )
            append_transcript(prompt_transcript_path, "\n".join(clip_lines))
    
    print()
    _attach_nav_timing()
    return result

def load_map_from_disk(map_dir: str) -> List[BaseObject]:
    if not os.path.exists(map_dir):
        return []
    obj_map = []
    pkl_files = [f for f in os.listdir(map_dir) if f.endswith(".pkl")]
    for pkl_file in pkl_files:
        try:
            obj_path = os.path.join(map_dir, pkl_file)
            obj = BaseObject.load_from_disk(obj_path)
            if hasattr(obj, 'clip_ft') and obj.clip_ft.size > 0:
                obj_map.append(obj)
        except Exception:
            continue
    return obj_map

def load_clip_model(cfg_path: Optional[str] = None):
    if cfg_path is None:
        cfg_path = str(dualmap_root() / "config" / "system_config.yaml")
    cfg = OmegaConf.load(cfg_path)
    device = cfg.get("device", "cuda")
    clip_model_name = cfg.clip.model_name
    clip_pretrained = cfg.clip.pretrained
    clip_model, _, _ = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    if "MobileCLIP" in clip_model_name:
        from mobileclip.modules.common.mobileone import reparameterize_model
        clip_model = reparameterize_model(clip_model)
    clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
    return clip_model, clip_tokenizer, device

def save_best_frame_image(best_frame_path: str, output_dir: Path, task: str, task_index: int) -> Optional[str]:
    """
    Copy the VLM-selected best frame into the output directory.

    Args:
        best_frame_path: Source image from VLM stage
        output_dir: Destination directory
        task: Task string (for filename)
        task_index: Task index in batch

    Returns:
        Saved path, or None on error
    """
    try:
        if not best_frame_path or not os.path.exists(best_frame_path):
            print(f"[NavRunner] Best frame image not found: {best_frame_path}", file=sys.stderr)
            return None
        
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_task = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in task[:50])
        safe_task = safe_task.replace(' ', '_')
        
        output_filename = f"best_frame_task{task_index:03d}_{safe_task}.png"
        output_path = output_dir / output_filename
        
        # 
        import shutil
        shutil.copy2(best_frame_path, output_path)
        
        print(f"[NavRunner] Saved best frame image to: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"[NavRunner] Error saving best frame image: {e}", file=sys.stderr)
        return None

def visualize_instance_pointcloud(
    target_uid: str,
    obj_map: List[BaseObject],
    output_dir: Path,
    task: str,
    task_index: int,
    suffix: str = "",
) -> Optional[str]:
    """
    Render instance point cloud off-screen and save RGB screenshot.

    Args:
        target_uid: Object UID
        obj_map: Loaded objects
        output_dir: Output directory
        task: Task string (for filename)
        task_index: Task index
        suffix: Optional filename suffix

    Returns:
        Path to PNG, or None on error
    """
    try:
        # 
        target_obj = None
        for obj in obj_map:
            obj_uid = str(getattr(obj, 'uid', None))
            if obj_uid == target_uid:
                target_obj = obj
                break
        
        if target_obj is None:
            print(f"[NavRunner] Target object not found: {target_uid}", file=sys.stderr)
            return None
        
        # 
        instance_pcd = getattr(target_obj, 'pcd', None)
        if instance_pcd is None or len(instance_pcd.points) == 0:
            print(f"[NavRunner] Instance point cloud is empty", file=sys.stderr)
            return None
        
        # 
        if len(instance_pcd.colors) == 0:
            print(f"[NavRunner] Warning: Instance point cloud has no colors, using uniform color", file=sys.stderr)
            instance_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        
        # 
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1920, height=1080)
        vis.add_geometry(instance_pcd)
        
        view_ctrl = vis.get_view_control()
        
        bbox = instance_pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        max_extent = np.max(extent)
        
        # Oblique view (~45 deg)
        view_ctrl.set_front([0.5, -0.5, -0.7])
        view_ctrl.set_lookat(center)
        view_ctrl.set_up([0, 0, 1])
        view_ctrl.set_zoom(0.7)
        
        # 
        vis.poll_events()
        vis.update_renderer()
        
        # 
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_task = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in task[:50])
        safe_task = safe_task.replace(' ', '_')
        
        output_filename = f"instance_pcd_task{task_index:03d}_{safe_task}{suffix}.png"
        output_path = output_dir / output_filename
        
        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = np.asarray(image)
        image_np = (image_np * 255).astype(np.uint8)
        img = Image.fromarray(image_np)
        img.save(output_path, "PNG")
        vis.destroy_window()
        
        print(f"[NavRunner] Saved instance point cloud visualization to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"[NavRunner] Error visualizing instance point cloud: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None

def save_best_frame_image(best_frame_path: str, output_dir: Path, task: str, task_index: int) -> Optional[str]:
    """
    Copy the VLM-selected best frame into the output directory.

    Args:
        best_frame_path: Source image from VLM stage
        output_dir: Destination directory
        task: Task string (for filename)
        task_index: Task index in batch

    Returns:
        Saved path, or None on error
    """
    try:
        if not best_frame_path or not os.path.exists(best_frame_path):
            print(f"[NavRunner] Best frame image not found: {best_frame_path}", file=sys.stderr)
            return None
        
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_task = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in task[:50])
        safe_task = safe_task.replace(' ', '_')
        
        output_filename = f"best_frame_task{task_index:03d}_{safe_task}.png"
        output_path = output_dir / output_filename
        
        # 
        import shutil
        shutil.copy2(best_frame_path, output_path)
        
        print(f"[NavRunner] Saved best frame image to: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"[NavRunner] Error saving best frame image: {e}", file=sys.stderr)
        return None

def visualize_instance_pointcloud(
    target_uid: str,
    obj_map: List[BaseObject],
    output_dir: Path,
    task: str,
    task_index: int,
    suffix: str = "",
) -> Optional[str]:
    """
    Render instance point cloud off-screen and save RGB screenshot.

    Args:
        target_uid: Object UID
        obj_map: Loaded objects
        output_dir: Output directory
        task: Task string (for filename)
        task_index: Task index
        suffix: Optional filename suffix

    Returns:
        Path to PNG, or None on error
    """
    try:
        # 
        target_obj = None
        for obj in obj_map:
            obj_uid = str(getattr(obj, 'uid', None))
            if obj_uid == target_uid:
                target_obj = obj
                break
        
        if target_obj is None:
            print(f"[NavRunner] Target object not found: {target_uid}", file=sys.stderr)
            return None
        
        # 
        instance_pcd = getattr(target_obj, 'pcd', None)
        if instance_pcd is None or len(instance_pcd.points) == 0:
            print(f"[NavRunner] Instance point cloud is empty", file=sys.stderr)
            return None
        
        # 
        if len(instance_pcd.colors) == 0:
            print(f"[NavRunner] Warning: Instance point cloud has no colors, using uniform color", file=sys.stderr)
            instance_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        
        # 
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1920, height=1080)
        vis.add_geometry(instance_pcd)
        
        view_ctrl = vis.get_view_control()
        
        bbox = instance_pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        max_extent = np.max(extent)
        
        # Oblique view (~45 deg)
        view_ctrl.set_front([0.5, -0.5, -0.7])
        view_ctrl.set_lookat(center)
        view_ctrl.set_up([0, 0, 1])
        view_ctrl.set_zoom(0.7)
        
        # 
        vis.poll_events()
        vis.update_renderer()
        
        # 
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_task = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in task[:50])
        safe_task = safe_task.replace(' ', '_')
        
        output_filename = f"instance_pcd_task{task_index:03d}_{safe_task}{suffix}.png"
        output_path = output_dir / output_filename
        
        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = np.asarray(image)
        image_np = (image_np * 255).astype(np.uint8)
        img = Image.fromarray(image_np)
        img.save(output_path, "PNG")
        vis.destroy_window()
        
        print(f"[NavRunner] Saved instance point cloud visualization to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"[NavRunner] Error visualizing instance point cloud: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None

def visualize_anchor_in_pointcloud(
    layout_pcd_path: str,
    target_anchor_uid: str,
    obj_map: List[BaseObject],
    output_dir: Path,
    task: str,
    anchor_id_map: Optional[Dict[str, int]] = None,
    suffix: str = "",
) -> Optional[str]:
    """
    Project layout + anchor highlight to a 2D matplotlib image (top-down).

    Args:
        layout_pcd_path: Path to layout.pcd
        target_anchor_uid: Anchor object UID
        obj_map: Loaded objects
        output_dir: Output directory
        task: Task string for overlay
        anchor_id_map: Optional UID -> simple id map
        suffix: Optional filename suffix

    Returns:
        Path to PNG, or None on error
    """
    try:
        if not os.path.exists(layout_pcd_path):
            print(f"[NavRunner] Layout point cloud not found: {layout_pcd_path}", file=sys.stderr)
            return None
        
        scene_pcd = o3d.io.read_point_cloud(layout_pcd_path)
        if len(scene_pcd.points) == 0:
            print(f"[NavRunner] Empty point cloud loaded from {layout_pcd_path}", file=sys.stderr)
            return None
        
        has_colors = len(scene_pcd.colors) > 0
        if not has_colors:
            scene_colors = np.ones((len(scene_pcd.points), 3)) * 0.5
            scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors)
        
        target_obj = None
        for obj in obj_map:
            obj_uid = str(getattr(obj, 'uid', None))
            if obj_uid == target_anchor_uid:
                target_obj = obj
                break
        
        if target_obj is None:
            print(f"[NavRunner] Target anchor object not found: {target_anchor_uid}", file=sys.stderr)
            return None
        
        anchor_pcd = getattr(target_obj, 'pcd', None)
        if anchor_pcd is None or len(anchor_pcd.points) == 0:
            bbox = getattr(target_obj, 'bbox', None)
            if bbox is not None:
                min_bound = bbox.get_min_bound()
                max_bound = bbox.get_max_bound()
                num_points = 1000
                points = np.random.uniform(min_bound, max_bound, (num_points, 3))
                anchor_pcd = o3d.geometry.PointCloud()
                anchor_pcd.points = o3d.utility.Vector3dVector(points)
            else:
                return None
        
        anchor_colors = np.ones((len(anchor_pcd.points), 3)) * [1.0, 0.0, 0.0]
        anchor_pcd.colors = o3d.utility.Vector3dVector(anchor_colors)
        
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        scene_points = np.asarray(scene_pcd.points)
        anchor_points = np.asarray(anchor_pcd.points)
        scene_colors = np.asarray(scene_pcd.colors) if len(scene_pcd.colors) > 0 else None
        
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.invert_yaxis()
        
        if scene_colors is not None:
            scene_x = scene_points[:, 0]
            scene_y = scene_points[:, 2]
            ax.scatter(scene_x, scene_y, c=scene_colors, s=15.0, alpha=1.0, marker='.', edgecolors='none')
        else:
            scene_x = scene_points[:, 0]
            scene_y = scene_points[:, 2]
            ax.scatter(scene_x, scene_y, c='gray', s=15.0, alpha=0.8, marker='.', edgecolors='none')
        
        anchor_x = anchor_points[:, 0]
        anchor_y = anchor_points[:, 2]
        ax.scatter(anchor_x, anchor_y, c='red', s=40.0, alpha=1.0, marker='o', edgecolors='darkred', linewidths=0.5)
        
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image_np = np.asarray(buf)
        plt.close(fig)
        
        image_np = image_np[:, :, :3]
        img = Image.fromarray(image_np)
        draw = ImageDraw.Draw(img)
        
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        except:
            try:
                font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
            except:
                font_large = ImageFont.load_default()
        
        if task:
            try:
                if hasattr(draw, 'textbbox'):
                    task_bbox = draw.textbbox((0, 0), task, font=font_large)
                    task_text_width = task_bbox[2] - task_bbox[0]
                    task_text_height = task_bbox[3] - task_bbox[1]
                else:
                    task_text_width, task_text_height = draw.textsize(task, font=font_large)
                
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
                
                line_height = task_text_height + 10
                total_height = len(task_lines) * line_height + 20
                
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(
                    [(0, 0), (img.width, total_height)],
                    fill=(0, 0, 0, 200)
                )
                img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(img)
                
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
                print(f"[NavRunner] Failed to add text to point cloud image: {e}", file=sys.stderr)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_task = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in task[:50])
        safe_task = safe_task.replace(' ', '_')
        simple_id = anchor_id_map.get(target_anchor_uid, -1) if anchor_id_map else -1
        
        if simple_id > 0:
            output_filename = f"pointcloud_anchor_{simple_id}_{safe_task}{suffix}.png"
        else:
            output_filename = f"pointcloud_anchor_{safe_task}{suffix}.png"
        output_path = output_dir / output_filename
        img.save(output_path, "PNG")
        
        print(f"[NavRunner] Saved point cloud visualization to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"[NavRunner] Error visualizing anchor in point cloud: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None

def main() -> int:
    dm = dualmap_root()
    cfg = OmegaConf.load(str(dm / "config" / "base_config.yaml"))
    
    if "representative_frames_path" in cfg:
        anchor_key_path = os.path.join(cfg.representative_frames_path, "anchor_key")
    else:
        anchor_key_path = str(
            dm / "dataset" / "HM3D_hovsg" / "00824-Dd4bFSTQ8gi" / "representative_frames" / "anchor_key"
        )
    default_anchor_file = os.path.join(anchor_key_path, "anchor_attached_objects.txt")
    default_manifest = os.path.join(anchor_key_path, "manifest.json")

    parser = argparse.ArgumentParser(description="Navigation mode anchor task runner (top-1 only)")
    parser.add_argument("task", nargs="?", help="User task / question (or use --tasks-file)")
    parser.add_argument("--tasks-file", type=Path, default=None, help="Path to file containing tasks (one per line)")
    parser.add_argument("--anchor-file", type=Path, default=Path(default_anchor_file), help="Path to anchor_attached_objects.txt")
    parser.add_argument("--manifest", type=Path, default=Path(default_manifest), help="Path to anchor manifest JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GPT model")
    parser.add_argument("--api-key", dest="api_key", default=None, help="API key override")
    parser.add_argument("--base-url", dest="base_url", default=None, help=f"API base URL")
    parser.add_argument("--visualize", action="store_true", help="Show visualization windows")
    parser.add_argument("--viz-mode", default="auto", choices=["auto", "manual"], help="Visualization mode")
    parser.add_argument("--top-k", type=int, default=3, help="Number of candidate anchors to retrieve (Top-K)")
    parser.add_argument(
        "--no-prompt-transcript",
        action="store_true",
        help="Disable streaming prompt/response log (default: write to task_prompt/ like a log file)",
    )
    parser.add_argument(
        "--prompt-transcript",
        type=Path,
        default=None,
        help="Explicit transcript .txt path (single task), or directory / multi-task naming; overrides default task_prompt/",
    )
    args = parser.parse_args()

    base_cfg = OmegaConf.load(str(dm / "config" / "base_config.yaml"))
    system_cfg = OmegaConf.load(str(dm / "config" / "system_config.yaml"))
    merged_cfg = OmegaConf.merge(base_cfg, system_cfg)
    
    # 
    print("=" * 80)
    print("[NavRunner] Configuration Information:")
    print("=" * 80)
    dataset_name = merged_cfg.get("dataset_name", "N/A")
    scene_id = merged_cfg.get("scene_id", "N/A")
    print(f"  Dataset: {dataset_name}")
    print(f"  Scene ID: {scene_id}")
    
    representative_frames_path = merged_cfg.get("representative_frames_path", "N/A")
    print(f"  Representative Frames Path: {representative_frames_path}")
    
    map_save_path = merged_cfg.get("map_save_path", "N/A")
    print(f"  Map Save Path: {map_save_path}")
    
    preload_path = merged_cfg.get("preload_path", None)
    if preload_path:
        print(f"  Preload Path: {preload_path}")
    print("=" * 80)
    print()
    
    BaseObject.initialize_config(merged_cfg)
    
    map_dir = merged_cfg.map_save_path
    print(f"[NavRunner] Loading map from: {map_dir}")
    obj_map = load_map_from_disk(map_dir)
    print(f"[NavRunner] Loaded {len(obj_map)} objects from map")
    print()
    
    print("[NavRunner] Loading CLIP model...")
    try:
        clip_model, clip_tokenizer, device = load_clip_model()
        print(f"[NavRunner] CLIP model loaded on device: {device}")
    except Exception as e:
        print(f"[NavRunner] Failed to load CLIP model: {e}", file=sys.stderr)
        return 1
    print()
    
    print(f"[NavRunner] Loading anchor data from: {args.anchor_file}")
    t_compact_sg = time.perf_counter()
    manifest, saved_anchor_id_map = load_manifest(args.manifest)
    anchors = load_anchor_file(args.anchor_file)
    anchors_with_frames = []
    for anchor in anchors:
        anchor_id = anchor.get("id", "")
        entry = find_manifest_entry(manifest, anchor_id)
        if entry is not None:
            image_paths = collect_image_paths(entry)
            if image_paths:
                anchors_with_frames.append(anchor)
    scene_graph_io_ms = (time.perf_counter() - t_compact_sg) * 1000.0
    print(f"[NavRunner] Loaded {len(anchors)} anchors (compact scene graph I/O: {scene_graph_io_ms:.2f} ms)")
    print()
    
    if not anchors_with_frames:
        print("[NavRunner] No anchors with representative frames found.", file=sys.stderr)
        return 1
    
    anchor_id_map = saved_anchor_id_map.copy() if saved_anchor_id_map else {}
    
    # 
    tasks = []
    if args.tasks_file:
        if not args.tasks_file.exists():
            print(f"[NavRunner] Error: Tasks file not found: {args.tasks_file}", file=sys.stderr)
            return 1
        with args.tasks_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Lines like "1) task text"
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
        print("[NavRunner] Error: Either provide a task or use --tasks-file", file=sys.stderr)
        parser.print_help()
        return 1
    
    if not tasks:
        print("[NavRunner] Error: No tasks found", file=sys.stderr)
        return 1
    
    print(f"[NavRunner] Processing {len(tasks)} task(s)")
    
    # 
    layout_pcd_path = None
    if hasattr(merged_cfg, 'map_save_path'):
        layout_pcd_path = os.path.join(merged_cfg.map_save_path, "layout.pcd")
        if not os.path.exists(layout_pcd_path):
            if hasattr(merged_cfg, 'preload_path') and merged_cfg.preload_path:
                layout_pcd_path = os.path.join(merged_cfg.preload_path, "layout.pcd")
    
    if hasattr(merged_cfg, 'map_save_path') and merged_cfg.map_save_path:
        output_dir = Path(merged_cfg.map_save_path).parent / "nav_visualizations"
    else:
        output_dir = Path("./nav_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # 
    for task_index, task in enumerate(tasks):
        print(f"\n[NavRunner] Processing task {task_index + 1}/{len(tasks)}: {task}")

        prompt_transcript_path: Optional[str] = None
        if args.prompt_transcript is not None:
            pt = Path(args.prompt_transcript)
            safe_task = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in task[:60])
            safe_task = safe_task.replace(" ", "_").strip("_") or "task"
            if len(tasks) == 1:
                prompt_transcript_path = str(pt)
            else:
                if pt.suffix.lower() == ".txt" and not pt.is_dir():
                    pt.parent.mkdir(parents=True, exist_ok=True)
                    prompt_transcript_path = str(pt.parent / f"{pt.stem}_{task_index:03d}{pt.suffix}")
                else:
                    base_dir = pt if pt.is_dir() else pt
                    base_dir.mkdir(parents=True, exist_ok=True)
                    prompt_transcript_path = str(base_dir / f"task_{task_index:03d}_{safe_task}.txt")
        elif not args.no_prompt_transcript:
            envp = get_path_from_env()
            if envp:
                prompt_transcript_path = envp
            else:
                log_dir = DEFAULT_TASK_PROMPT_DIR
                log_dir.mkdir(parents=True, exist_ok=True)
                safe_task = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in task[:60])
                safe_task = safe_task.replace(" ", "_").strip("_") or "task"
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                prompt_transcript_path = str(log_dir / f"task_{task_index:03d}_{safe_task}_{ts}.txt")

        result = process_single_task_nav(
            task,
            anchors_with_frames,
            manifest,
            args.model,
            args.base_url,
            args.api_key,
            clip_model,
            clip_tokenizer,
            device,
            obj_map,
            anchor_id_map,
            visualize=args.visualize,
            viz_mode=args.viz_mode,
            top_k=args.top_k,
            prompt_transcript_path=prompt_transcript_path,
            scene_graph_io_ms=scene_graph_io_ms,
        )
        
        result["task"] = task
        result["task_index"] = task_index
        if result.get("target_final_uid") is not None:
            result["final_uid"] = result["target_final_uid"]

        if result["success"] and obj_map:
            if result.get("best_frame_path"):
                best_frame_img_path = save_best_frame_image(
                    result["best_frame_path"],
                    output_dir,
                    task,
                    task_index
                )
                if best_frame_img_path:
                    result["best_frame_image_path"] = best_frame_img_path
            
            if result.get("target_final_uid"):
                instance_pcd_path = visualize_instance_pointcloud(
                    result["target_final_uid"],
                    obj_map,
                    output_dir,
                    task,
                    task_index,
                    suffix="_top1_instance"
                )
                if instance_pcd_path:
                    result["instance_pcd_path"] = instance_pcd_path
        
        all_results.append(result)
        
        # 
        print(f"[NavRunner] Task {task_index + 1} Success: {result['success']}")
        if result["success"]:
            print(f"[NavRunner] Navigate to frame: {result['best_frame_path']}")
            print(f"[NavRunner] Selected target anchor UID: {result['target_anchor_uid']} (anchor ID: {anchor_id_map.get(result['target_anchor_uid'], 'N/A')})")
            if result["frame_idx"] is not None:
                print(f"[NavRunner] Frame index: {result['frame_idx']}")
            print(f"[NavRunner] Method: {result['method']}")
        else:
            print("[NavRunner] Task failed: No target found")

        # Print JSON to stdout (DualMap may parse; ensure Path/nav_retrieval_timing_ms are JSON-safe)
        def _json_safe(o):
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, dict):
                return {k: _json_safe(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_json_safe(v) for v in o]
            return o

        print("\n[NavRunner] Result JSON:", flush=True)
        print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False), flush=True)
    
    # JSON
    results_file = output_dir / "results.json"
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[NavRunner] All results saved to: {results_file}")
    
    # 
    success_count = sum(1 for r in all_results if r["success"])
    print(f"\n[NavRunner] Summary: {success_count}/{len(tasks)} tasks succeeded")
    
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

