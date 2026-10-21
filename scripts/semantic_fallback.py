#!/usr/bin/env python3
"""
Semantic Fallback Script

When visual retrieval fails, use an LLM to pick the next best anchor from remaining candidates.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Set, Optional

# Add project root for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openai import OpenAI
from utils.vlm_prompt_transcript import append_transcript, get_path_from_env

DEFAULT_MODEL = "gpt-5.2"
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")

SYSTEM_PROMPT = """You are a robotic navigation assistant.
The robot is trying to find a target object but failed at the initial candidate locations.
Your job is to use COMMON SENSE REASONING to suggest the next best location (Anchor) to search.

Inputs:
1. User Task: The object the robot is looking for.
2. Visited Anchors: IDs of anchors the robot has already checked (and found nothing).
3. Remaining Anchors: A list of available anchors with their semantic categories and nearby objects.

Instructions:
- Analyze the semantic relationship between the target object and the remaining anchors.
- Think about where the object typically appears (e.g., 'cup' on 'table', 'pillow' on 'bed').
- Select ONE anchor that is most likely to contain the target.
- If no anchor is a strong match, select the one with the most objects (attached + nearby) as a hub.

Output (JSON only):
{
  "target_anchor_id": "string",  // The ID of the selected anchor
  "reasoning": "string"          // Brief explanation of your choice
}
"""

def load_manifest(path: Path) -> Dict[str, Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    anchors = data.get("anchors", data)
    return anchors

def load_anchor_file(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    anchors = []
    current = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith("Anchor id:"):
                if current: anchors.append(current)
                current = {"id": line.replace("Anchor id:", "").strip(), "category": "", "info": ""}
            elif line.startswith("Category:") and current:
                current["category"] = line.replace("Category:", "").strip()
            elif current:
                current["info"] += line + " "
        if current: anchors.append(current)
    except Exception as e:
        print(f"Error loading anchor file: {e}", file=sys.stderr)
    return anchors

def call_llm_reasoning(
    task: str,
    visited_ids: List[str],
    all_anchors: List[Dict],
    model: str,
    base_url: str,
    api_key: str,
    prompt_transcript_path: Optional[str] = None,
) -> Dict:
    
    # Drop anchors already visited
    candidates = []
    for anc in all_anchors:
        # Loose ID match (formats may differ)
        is_visited = False
        anc_id = anc.get("id", "")
        for v_id in visited_ids:
            if v_id in anc_id or anc_id in v_id:
                is_visited = True
                break
        
        if not is_visited:
            # One-line summary per candidate
            desc = f"ID: {anc_id}, Category: {anc.get('category', 'unknown')}, Context: {anc.get('info', '')[:200]}"
            candidates.append(desc)
    
    if not candidates:
        return {"target_anchor_id": None, "reasoning": "No remaining anchors to check."}

    user_prompt = f"""Task: Find '{task}'
    
Visited Anchors (IGNORE these): {visited_ids}

Remaining Candidates:
{chr(10).join(candidates[:50])}  // Limit to 50 to avoid context overflow

Select the best anchor ID from the candidates."""

    client = OpenAI(base_url=base_url, api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        out = json.loads(content)
        tpath = prompt_transcript_path or get_path_from_env()
        if tpath:
            append_transcript(
                tpath,
                "========== Semantic fallback (LLM anchor choice) ==========\n\n"
                f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\n{user_prompt}\n\n[ASSISTANT]\n{content}",
            )
        return out
    except Exception as e:
        print(f"LLM Error: {e}", file=sys.stderr)
        tpath = prompt_transcript_path or get_path_from_env()
        if tpath:
            append_transcript(
                tpath,
                f"========== Semantic fallback (LLM) [ERROR] ==========\n{repr(e)}",
            )
        return {"target_anchor_id": None, "reasoning": str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--anchor-file", required=True)
    parser.add_argument("--visited", nargs="*", default=[])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--visualize", action="store_true", help="Show visualization window")
    parser.add_argument(
        "--prompt-transcript",
        type=str,
        default=None,
        help="Append prompts/responses to this transcript file (same session as NavRunner)",
    )
    args = parser.parse_args()

    anchors = load_anchor_file(Path(args.anchor_file))
    
    result = call_llm_reasoning(
        args.task,
        args.visited,
        anchors,
        args.model,
        args.base_url,
        args.api_key,
        prompt_transcript_path=args.prompt_transcript,
    )
    
    # Visualization
    if args.visualize:
        try:
            from utils.visualizer_ui import show_visualization_window
            
            # Construct content for the left side (simulating images with text cards)
            images_data = []
            
            # Card 1: Task Info
            images_data.append({
                "label": "Current Task",
                "text_content": f"Looking for: {args.task}\n\nStatus: Visual Search Failed.\nInitiating Semantic Reasoning..."
            })
            
            # Card 2: Visited Locations
            visited_str = ", ".join(args.visited) if args.visited else "None"
            images_data.append({
                "label": "Already Visited (Excluded)",
                "text_content": f"{visited_str}\n\n(These anchors have been verified as empty)"
            })
            
            # Card 3: Reasoning Output
            reasoning = result.get("reasoning", "No reasoning provided.")
            target = result.get("target_anchor_id", "None")
            images_data.append({
                "label": "LLM Reasoning Result",
                "text_content": f"Selected Target: {target}\n\nReasoning:\n{reasoning}"
            })

            # Show Window
            # Use 'manual' mode by default for fallback reasoning so user can read it
            show_visualization_window(
                title=f"Semantic Fallback - {args.task}",
                images=images_data,
                prompts=SYSTEM_PROMPT,
                response=json.dumps(result, indent=2),
                mode="manual", 
                timeout=5000
            )
        except Exception as e:
            # Don't fail the script if viz fails
            print(f"Visualization error: {e}", file=sys.stderr)

    print(json.dumps(result))

if __name__ == "__main__":
    main()
