#!/usr/bin/env python3
"""
VLM helper to locate a target among new object instances when it left the original view.
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.gemini_vlm import generate_content_with_images
from utils.visualizer_ui import show_visualization_window
from utils.vlm_prompt_transcript import append_transcript, get_path_from_env

DEFAULT_VLM_MODEL = os.environ.get("GEMINI_VLM_MODEL", "gemini-3-pro-preview")

SYSTEM_PROMPT = """You are an intelligent assistant helping a robot find a missing object.
The robot went to the object's original location, but it was gone.
Now, the robot has found several candidate objects (New Instances) elsewhere that look similar.

You will receive:
1. A task description (e.g., "red plate").
2. The REFERENCE image (Image 0): The original view of the target object (from the map).
3. A set of CANDIDATE images (Image 1 to N): New instances found by the robot.

Your task:
1. Compare each CANDIDATE image with the REFERENCE image and the Task Description.
2. Determine if any of the candidates IS the missing target object (or a very likely new instance of it).
3. Select the best match.

You must answer with a single JSON object:
{
  "found": true/false,
  "selected_index": int (1-based index of the selected image, e.g., 1 for Image 1. If found=false, set to -1),
  "confidence": "high/medium/low",
  "reasoning": "Explain why you selected this candidate or why none matched. Compare visual features like color, shape, texture, and context."
}

STRICT RULES:
1. **SMALL OBJECT LENIENCY**: For small objects (e.g., boxes, cups, keys), do NOT require HD clarity. If a candidate looks **similar in color/shape** and is in a plausible location, mark it as `found=true`. Better to check a "maybe" than miss a target.
2. **Weak Reference Handling**: If the REFERENCE image (Image 0) does not clearly show the target object (e.g., it's empty, blurry, or wrong), IGNORE Image 0. Evaluate the Candidate images based solely on the Task Description (text). Do NOT reject good candidates just because Image 0 is bad.
"""

def encode_image(path: str) -> str:
    """Encode image to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_image_mime_type(path: str) -> str:
    """Get MIME type based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.png']:
        return 'image/png'
    elif ext in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    else:
        return 'image/jpeg'

def call_vlm_find(
    task: str,
    ref_image_path: str,
    candidate_paths: List[str],
    model: str = DEFAULT_VLM_MODEL,
    gemini_api_key: Optional[str] = None,
    visualize: bool = False,
    viz_mode: str = "auto",
    prompt_transcript_path: Optional[str] = None,
) -> Dict:
    if not os.path.exists(ref_image_path):
        raise FileNotFoundError(f"Reference image not found: {ref_image_path}")
    for p in candidate_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Candidate image not found: {p}")

    print(f"[FindTarget] Task: {task}", flush=True)
    print(f"[FindTarget] Candidates: {len(candidate_paths)} images", flush=True)

    user_prompt = f"Task: Find '{task}'.\n\nImage 0 is the REFERENCE (original target).\nImages 1-{len(candidate_paths)} are CANDIDATES.\n\nWhich candidate shows the '{task}'?"

    image_paths = [ref_image_path] + list(candidate_paths)
    print(f"[FindTarget] Sending request to Gemini VLM with {len(image_paths)} images...", flush=True)

    try:
        text = generate_content_with_images(
            system_instruction=SYSTEM_PROMPT,
            user_text=user_prompt,
            image_paths=image_paths,
            model=model,
            api_key=gemini_api_key,
        )
        if not text:
            raise RuntimeError("Empty response from Gemini VLM")

        print(f"[FindTarget] Raw response: {text}", flush=True)

        tpath = prompt_transcript_path or get_path_from_env()
        if tpath:
            names = [os.path.basename(ref_image_path)] + [os.path.basename(p) for p in candidate_paths]
            img_note = "(API included images; filenames only, no pixels in this log)\n  " + "\n  ".join(
                f"Image {i}: {n}" for i, n in enumerate(names)
            )
            append_transcript(
                tpath,
                "========== Find target in new instances (VLM) ==========\n"
                f"{img_note}\n\n[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\n{user_prompt}\n\n[ASSISTANT]\n{text}",
            )

        # Parse JSON first to get simplified response
        result = None
        try:
            # Clean up code blocks if present
            clean_text = text
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()
            result = json.loads(clean_text)
        except:
            pass

        if visualize:
            images = [{"path": ref_image_path, "label": "Image 0 (Reference)"}]
            for i, p in enumerate(candidate_paths):
                images.append({"path": p, "label": f"Image {i+1} (Candidate)"})
            
            simple_prompts = f"TASK: {task}\n\nSTEP: Searching Target in New Instances\n\nComparing candidates with reference..."
            
            simple_resp = ""
            if result:
                found = result.get("found", False)
                idx = result.get("selected_index", -1)
                reason = result.get("reasoning", "")
                
                icon = "✅" if found else "❌"
                simple_resp += f"TARGET FOUND: {found} {icon}\n"
                
                if found:
                    simple_resp += f"SELECTED IMAGE INDEX: {idx}\n"
                
                simple_resp += f"\nREASONING:\n{reason}"
            else:
                simple_resp = text # Fallback

            show_visualization_window(
                title=f"Find Target in New Instances - {task}",
                images=images,
                prompts=simple_prompts,
                response=simple_resp,
                mode=viz_mode
            )

        if result is None:
             # Parse again if not parsed
             try:
                # Clean up code blocks if present
                clean_text = text
                if "```json" in clean_text:
                    clean_text = clean_text.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_text:
                    clean_text = clean_text.split("```")[1].split("```")[0].strip()
                result = json.loads(clean_text)
             except:
                 raise

        return result

    except Exception as e:
        print(f"[FindTarget] Error: {e}", flush=True)
        # import traceback
        # traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--ref-image", required=True)
    parser.add_argument("--candidates", required=True, nargs="+", help="List of candidate image paths")
    parser.add_argument("--model", default=DEFAULT_VLM_MODEL, help="Gemini VLM model")
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Gemini API key (or GEMINI_API_KEY / GOOGLE_API_KEY)",
    )
    parser.add_argument("--visualize", action="store_true", help="Show visualization window")
    parser.add_argument("--viz-mode", default="auto", choices=["auto", "manual"], help="Visualization mode")
    parser.add_argument(
        "--prompt-transcript",
        type=str,
        default=None,
        help="Append prompts/responses to this transcript file (same session as NavRunner)",
    )

    args = parser.parse_args()
    
    try:
        result = call_vlm_find(
            task=args.task,
            ref_image_path=args.ref_image,
            candidate_paths=args.candidates,
            model=args.model,
            gemini_api_key=args.gemini_api_key,
            visualize=args.visualize,
            viz_mode=args.viz_mode,
            prompt_transcript_path=args.prompt_transcript,
        )
        print("\n[FindTarget] Result JSON:")
        print(json.dumps(result))
        return 0
    except Exception:
        return 1

if __name__ == "__main__":
    sys.exit(main())
