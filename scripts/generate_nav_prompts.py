import json
import random
import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent

def generate_prompts(scene_info_path, target_categories, count=33):
    if not os.path.exists(scene_info_path):
        print(f"Error: File not found at {scene_info_path}")
        return

    with open(scene_info_path, 'r') as f:
        data = json.load(f)

    # 1. Build Region Map (id -> category)
    # Regions can be at top level or inside levels
    regions_map = {} # id -> category label
    
    # Check top level regions
    if 'regions' in data:
        for r in data['regions']:
            if 'id' in r and 'category' in r:
                regions_map[r['id']] = r['category']
    
    # Check levels for regions if not found or to add more
    if 'levels' in data:
        for level in data['levels']:
            if 'regions' in level:
                # 'regions' in level might be a list of IDs or list of dicts?
                # Based on previous grep, it seemed like "regions": [0, 1, ...] in levels
                # But there was also "regions": [{"id": 0...}] somewhere.
                # Let's handle list of dicts.
                for r in level['regions']:
                    if isinstance(r, dict) and 'id' in r and 'category' in r:
                        regions_map[r['id']] = r['category']

    # 2. Collect Valid Objects and Group by Prompt
    valid_objects_map = {} # prompt -> {info, ids: []}
    
    # Helper to process an object list
    def process_obj_list(obj_list):
        for obj in obj_list:
            if not isinstance(obj, dict): continue
            
            cat = obj.get('category')
            if not cat: continue
            
            matched_target = None
            for target in target_categories:
                if target.lower() in cat.lower():
                    matched_target = target
                    break
            
            if matched_target:
                region_id = obj.get('region_id')
                if region_id is not None and region_id in regions_map:
                    room_name = regions_map[region_id]
                    prompt = f"{matched_target} in the {room_name}"
                    
                    if prompt not in valid_objects_map:
                        valid_objects_map[prompt] = {
                            "prompt": prompt,
                            "target_object": matched_target,
                            "target_room": room_name,
                            "gt_ids": []
                        }
                    valid_objects_map[prompt]["gt_ids"].append(obj.get('id'))

    # Objects can be at top level or inside levels
    if 'objects' in data:
        process_obj_list(data['objects'])
    
    if 'levels' in data:
        for level in data['levels']:
            if 'objects' in level:
                process_obj_list(level['objects'])

    if not valid_objects_map:
        print("No objects found matching the target categories in the scene info.")
        return

    # 3. Random Select
    unique_candidates = list(valid_objects_map.values())
    
    # Constraint: Max 2 doors
    doors = [c for c in unique_candidates if c['target_object'].lower() == 'door']
    others = [c for c in unique_candidates if c['target_object'].lower() != 'door']
    
    if len(doors) > 2:
        doors = random.sample(doors, 2)
        
    filtered_candidates = doors + others
            
    selected = []
    if len(filtered_candidates) <= count:
        selected = filtered_candidates
    else:
        selected = random.sample(filtered_candidates, count)

    # 4. Print and Save
    print(f"Found {len(unique_candidates)} unique prompts. Selected {len(selected)}:\n")
    
    output_data = []
    for item in selected:
        print(f"{item['prompt']} (Matches: {len(item['gt_ids'])})")
        output_data.append(item)
        
    # Save to JSON for downstream tasks
    output_path = "nav_prompts.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"\nSaved {len(output_data)} prompts with GT info to {output_path}")

    # Save to TXT for anchor_task_runner input
    txt_path = "nav_prompts.txt"
    with open(txt_path, 'w') as f:
        for item in output_data:
            f.write(f"{item['prompt']}\n")
    print(f"Saved prompts text to {txt_path}")

if __name__ == "__main__":
    target_cats = [
        "picture", "pillow", "door", "lamp", "cabinet", "book", "chair", "table", 
        "towel", "plant", "sink", "stairs", "bed", "toilet", "tv", "desk", 
        "couch", "flowerpot", "nightstand", "faucet"
    ]
    
    # Path to the scene_info.json
    # Load default path from config
    default_path = "dataset/HM3D_hovsg/00829-QaLdnwvtxbs/scene_info.json" # Fallback
    try:
        base_cfg = OmegaConf.load(str(_REPO_ROOT / "config" / "base_config.yaml"))
        scene_id = base_cfg.scene_id
        dataset_path = base_cfg.get("dataset_path", "dataset/HM3D_hovsg")
        default_path = os.path.join(dataset_path, scene_id, "scene_info.json")
        print(f"Loaded config for scene: {scene_id}")
    except Exception as e:
        print(f"Warning: Failed to load config/base_config.yaml: {e}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_info", type=str, default=default_path)
    parser.add_argument("--count", type=int, default=33)
    args = parser.parse_args()
    
    generate_prompts(args.scene_info, target_cats, args.count)

