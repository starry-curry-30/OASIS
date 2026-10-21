#!/usr/bin/env python3
"""
Post-rotation VLM check (navigation mode).

After the robot reaches the goal and finishes rotating, run this to verify the target is visible.
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.gemini_vlm import generate_content_with_images
from utils.visualizer_ui import show_visualization_window
from utils.vlm_prompt_transcript import append_transcript, get_path_from_env

DEFAULT_VLM_MODEL = os.environ.get("GEMINI_VLM_MODEL", "gemini-3-pro-preview")

POST_ROTATION_VLM_SYSTEM_PROMPT = """You are an assistant that checks if a target object is found in the current view after navigation.

You will receive:
1. A task description (e.g., "find the bed")
2. Two images:
   - Image 1: The best representative frame from the original view (when the target was identified)
   - Image 2: The current frame after navigation and rotation to the target position

Your task:
1. First, check if the target object described in the task is visible in Image 2 (current frame).
2. If the target is NOT found in Image 2:
   - Compare Image 1 and Image 2 carefully
   - Determine if the target object has disappeared from its original location
   - Analyze what might have happened (e.g., object moved, occluded, lighting changed, etc.)

You must answer with a single JSON object in this exact schema:
{
  "target_found": true/false,
  "reasoning": "Detailed explanation of what you see. If target_found=true, describe where you see the target in Image 2. If target_found=false, explain why you cannot see it and compare with Image 1.",
  "target_disappeared": true/false (only meaningful if target_found=false),
  "disappearance_analysis": "If target_disappeared=true, explain what happened to the target object by comparing Image 1 and Image 2. Describe changes in position, visibility, occlusion, lighting, etc. If target_disappeared=false, explain why the target might not be visible (e.g., wrong viewpoint, different angle, etc.)",
  "confidence": "high/medium/low"
}

STRICT RULES:
1. **Target Found**: Set target_found=true ONLY if you can clearly see the specific object described in the task in Image 2.
2. **Comparison**: When target_found=false, you MUST compare Image 1 and Image 2 to determine if the target disappeared.
3. **Disappearance**: Set target_disappeared=true if the target object is NOT visible in Image 2.
4. **Weak Reference Handling**: If the target object is NOT clearly visible or is ambiguous in Image 1 (Reference), IGNORE Image 1's visual features and verify Image 2 solely based on the text description. Do NOT conclude "disappeared" just because Image 1 was bad.
5. **Be Precise**: Do not hallucinate. Only report what you can actually see in the images."""

POST_ROTATION_RELOCALIZATION_SYSTEM_PROMPT = """You are an assistant that verifies if a target object has successfully moved to a new location.

Context:
- The target object was originally at Location A (Image 1).
- We detected a potential new location for this object at Location B.
- The robot has navigated to Location B and is now looking at it (Image 2).

You will receive:
1. A task description (e.g., "find the red plate")
2. Two images:
   - Image 1: The original location where the object used to be (Reference Appearance).
   - Image 2: The NEW current location where we suspect the object has moved to.

Your Task:
Check if the target object from Image 1 has moved to the location shown in Image 2.
Crucially, you must verify that the object in Image 2 is the SAME INSTANCE as the one in Image 1 (matching color, shape, texture, size, etc.).

You must answer with a single JSON object in this exact schema:
{
  "target_found": true/false,
  "is_same_instance": true/false,
  "reasoning": "Detailed comparison. 1) Do you see the candidate object in Image 2? 2) Compare its visual features (color, style, wear) with the object in Image 1. 3) Conclude if it is the EXACT SAME object that has moved.",
  "confidence": "high/medium/low"
}

STRICT RULES:
1. **Target Found**: Set `target_found=true` ONLY if you see the object in Image 2.
2. **Same Instance Verification**: Set `is_same_instance=true` ONLY if the object in Image 2 visually matches the one in Image 1 detailedly. It must look like the same physical item.
3. **Weak Reference Exception**: If the target object is NOT clearly visible or is ambiguous in Image 1 (Reference), IGNORE Image 1. Set `is_same_instance=true` if the object in Image 2 matches the TEXT description perfectly. Do NOT fail the verification just because Image 1 was empty or blurry.
4. **Conclusion**: If `target_found` is true AND `is_same_instance` is true, then we consider the relocation successful.
5. **Ignore Disappearance**: Do not analyze if it disappeared from Image 1 (we already know that). Focus entirely on identifying it in Image 2.
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
        return 'image/jpeg'  # default


def call_vlm_check(
    task: str,
    best_frame_path: str,
    current_frame_path: str,
    check_mode: str = "initial",
    model: str = DEFAULT_VLM_MODEL,
    gemini_api_key: Optional[str] = None,
    visualize: bool = False,
    viz_mode: str = "auto",
    prompt_transcript_path: Optional[str] = None,
) -> Dict:
    """
    Run Gemini VLM to check whether the target is visible after rotation.

    Images are sent in order: Image 1 = best reference frame, Image 2 = current frame.
    """
    best_frame_path_abs = os.path.abspath(best_frame_path)
    current_frame_path_abs = os.path.abspath(current_frame_path)

    if not os.path.exists(best_frame_path_abs):
        raise FileNotFoundError(f"Best frame path does not exist: {best_frame_path_abs}")
    if not os.path.exists(current_frame_path_abs):
        raise FileNotFoundError(f"Current frame path does not exist: {current_frame_path_abs}")

    print(f"[PostRotationVLM] Task: {task}", flush=True)
    print(f"[PostRotationVLM] Check Mode: {check_mode}", flush=True)
    print(f"[PostRotationVLM] Best frame: {best_frame_path_abs}", flush=True)
    print(f"[PostRotationVLM] Current frame: {current_frame_path_abs}", flush=True)
    print(f"[PostRotationVLM] Model: {model}", flush=True)
    print("-" * 80, flush=True)

    if check_mode == "relocalization":
        system_prompt = POST_ROTATION_RELOCALIZATION_SYSTEM_PROMPT
        user_prompt = f"""Task: {task}

Please verify Relocalization:
1. Look at Image 1 (Reference): Identify the visual details of the target object.
2. Look at Image 2 (Current View): Is that EXACT SAME object present here?
3. Confirm: Has the target moved to this new location?

Image 1: Original/Reference Appearance.
Image 2: New Location candidate view."""
    else:
        system_prompt = POST_ROTATION_VLM_SYSTEM_PROMPT
        user_prompt = f"""Task: {task}

Please check:
1. Is the target object visible in the current frame (Image 2)?
2. If not, set "target_disappeared": true.
   - If visible in Image 1 but not Image 2 -> It disappeared.
   - If NOT visible in Image 1 AND NOT visible in Image 2 -> Treat as disappeared (so we can search again).

Image 1 (original best frame): This is the representative frame where the target was originally identified.
Image 2 (current frame): This is the current view after navigation and rotation to the target position."""

    print(f"[PostRotationVLM] User Prompt:\n{user_prompt}\n", flush=True)
    print("-" * 80, flush=True)

    print("[PostRotationVLM] Encoding images (Gemini)...", flush=True)
    print(f"[PostRotationVLM]   Image 1 (best): {best_frame_path_abs}", flush=True)
    print(f"[PostRotationVLM]   Image 2 (current): {current_frame_path_abs}", flush=True)
    print("[PostRotationVLM] Sending request to VLM...", flush=True)
    print("-" * 80, flush=True)

    try:
        text = generate_content_with_images(
            system_instruction=system_prompt,
            user_text=user_prompt,
            image_paths=[best_frame_path_abs, current_frame_path_abs],
            model=model,
            api_key=gemini_api_key,
        )
        if not text:
            raise RuntimeError("Empty response from Gemini VLM")

        print("[PostRotationVLM] Received response from VLM", flush=True)
        
        print(f"[PostRotationVLM] Raw response:\n{text}\n", flush=True)
        print("-" * 80, flush=True)

        tpath = prompt_transcript_path or get_path_from_env()
        if tpath:
            img_note = (
                "(API included 2 images; filenames only, no pixels in this log)\n"
                f"  Image 1 (reference): {os.path.basename(best_frame_path_abs)}\n"
                f"  Image 2 (current): {os.path.basename(current_frame_path_abs)}"
            )
            append_transcript(
                tpath,
                f"========== Post-rotation VLM ({check_mode}) ==========\n"
                f"{img_note}\n\n[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}\n\n[ASSISTANT]\n{text}",
            )
        
        # Parse JSON
        try:
            result_json = json.loads(text)
            print(f"[PostRotationVLM] Parsed JSON result:", flush=True)
            print(json.dumps(result_json, indent=2), flush=True)
            print("-" * 80, flush=True)
            
            if visualize:
                # Short display fields
                simple_prompts = f"TASK: {task}\n\nMODE: {check_mode.upper()} CHECK\n\nComparing Image 1 (Original) vs Image 2 (Current)"
                
                simple_response = ""
                target_found = result_json.get("target_found", False)
                reasoning = result_json.get("reasoning", "")
                
                status_icon = "✅" if target_found else "❌"
                simple_response += f"TARGET FOUND: {target_found} {status_icon}\n"
                
                if not target_found:
                    disappeared = result_json.get("target_disappeared", False)
                    simple_response += f"DISAPPEARED: {disappeared}\n"
                
                if check_mode == "relocalization":
                    same_instance = result_json.get("is_same_instance", False)
                    same_icon = "✅" if same_instance else "❌"
                    simple_response += f"SAME INSTANCE: {same_instance} {same_icon}\n"
                
                simple_response += f"\nREASONING:\n{reasoning}"

                show_visualization_window(
                    title=f"Post-Rotation Check ({check_mode}) - {task}",
                    images=[
                        {"path": best_frame_path_abs, "label": "Image 1 (Reference)"},
                        {"path": current_frame_path_abs, "label": "Image 2 (Current)"}
                    ],
                    prompts=simple_prompts,
                    response=simple_response,
                    mode=viz_mode
                )
            
            return result_json
        except json.JSONDecodeError as e:
            print(f"[PostRotationVLM] ERROR: Failed to parse JSON response: {e}", flush=True)
            print(f"[PostRotationVLM] Raw response was: {text}", flush=True)
            raise
    
    except Exception as e:
        print(f"[PostRotationVLM] ERROR: Exception during VLM call: {e}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Post-rotation VLM check")
    parser.add_argument("--task", type=str, required=True, help="Task description (target query)")
    parser.add_argument("--best-frame", type=str, required=True, help="Path to best representative frame")
    parser.add_argument("--current-frame", type=str, required=True, help="Path to current frame")
    parser.add_argument("--check-mode", type=str, default="initial", choices=["initial", "relocalization"], help="Check mode: initial or relocalization")
    parser.add_argument("--model", type=str, default=DEFAULT_VLM_MODEL, help="Gemini VLM model name")
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key (or use GEMINI_API_KEY / GOOGLE_API_KEY)",
    )
    parser.add_argument("--visualize", action="store_true", help="Show visualization window")
    parser.add_argument("--viz-mode", type=str, default="auto", choices=["auto", "manual"], help="Visualization mode")
    parser.add_argument(
        "--prompt-transcript",
        type=str,
        default=None,
        help="Append prompts/responses to this transcript file (same session as NavRunner)",
    )
    
    args = parser.parse_args()
    
    try:
        result = call_vlm_check(
            task=args.task,
            best_frame_path=args.best_frame,
            current_frame_path=args.current_frame,
            check_mode=args.check_mode,
            model=args.model,
            gemini_api_key=args.gemini_api_key,
            visualize=args.visualize,
            viz_mode=args.viz_mode,
            prompt_transcript_path=args.prompt_transcript,
        )
        
        # Emit JSON to stdout for the parent process
        print("\n[PostRotationVLM] Result JSON:", flush=True)
        print(json.dumps(result), flush=True)
        
        return 0
    except Exception as e:
        print(f"[PostRotationVLM] Failed: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())