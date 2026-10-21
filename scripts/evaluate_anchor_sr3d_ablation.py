#!/usr/bin/env python3
"""
Sr3D+ ablation: LLM proposal + VLM verification vs variants.

Modes:
  - Full: LLM proposal + VLM verification (coarse-to-fine)
  - w/o LLM Proposal: exhaustive VLM over anchors (no LLM shortlist)
  - w/o VLM Verification: LLM text-only top-1, no VLM

Cost tips:
  - One scene + 50 queries/scene: --scenes scene0222_00 --max_queries_per_scene 50 (default)
  - Cap w/o_llm VLM calls: --max_anchors_w_o_llm 15 (default)
  - Repro: --query_seed 42 (default)
  Paper line: We report ablation on 50 queries per scene (seed 42) and cap exhaustive VLM at 15 anchors/query.

Examples:
  # Light run (default caps)
  python scripts/evaluate_anchor_sr3d_ablation.py --scenes scene0222_00
  # Full queries (slow first run)
  python scripts/evaluate_anchor_sr3d_ablation.py --scenes scene0222_00 --max_queries_per_scene 0 --max_anchors_w_o_llm 0
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from utils.object import BaseObject

# Reused from evaluate_anchor_sr3d
from scripts.evaluate_anchor_sr3d import (
    BBQ_DATA_ROOT,
    DEFAULT_VLM_MODEL,
    SCENES,
    RELATIONS,
    setup_logging,
    evaluate_scene,
)
from scripts.anchor_task_runner import (
    load_anchor_file,
    load_manifest,
    find_manifest_entry,
    collect_image_paths,
    load_map_from_disk,
    load_clip_and_fastsam,
)

ABLATION_MODES = [
    "full",       # Full method (LLM + VLM)
    "w/o_llm",    # w/o LLM Proposal: exhaustive VLM
    "w/o_vlm",    # w/o VLM Verification: LLM top-1 only
]
MODE_DISPLAY_NAMES = {
    "full": "Full (LLM + VLM)",
    "w/o_llm": "w/o LLM Proposal",
    "w/o_vlm": "w/o VLM Verification",
}


def run_ablation_for_scene(
    scene: str,
    bbq_data_root: Path,
    merged_cfg,
    clip_model,
    clip_tokenizer,
    clip_preprocess,
    fastsam,
    device: str,
    llm_model: str,
    vlm_model: Optional[str],
    base_url: str,
    api_key: str,
    gemini_api_key: Optional[str],
    relations: list,
    ablation_mode: str,
    logger: logging.Logger,
    max_queries_per_scene: Optional[int] = None,
    query_sample_seed: Optional[int] = 42,
    ablation_max_anchors: Optional[int] = 15,
):
    """Run evaluate_scene for one scene and mode; returns scene_results or None."""
    base_cfg_template = OmegaConf.load(project_root / "config" / "base_config.yaml")
    base_cfg = OmegaConf.create(dict(base_cfg_template))
    base_cfg.scene_id = scene
    if "representative_frames_path" in base_cfg:
        base_cfg.representative_frames_path = str(base_cfg.representative_frames_path).replace(
            base_cfg_template.scene_id, scene
        )
    else:
        dataset_path = base_cfg.get("dataset_path", "./dataset/scannet")
        base_cfg.representative_frames_path = f"{dataset_path}/exported/{scene}/representative_frames"

    merged = OmegaConf.merge(base_cfg, OmegaConf.load(project_root / "config" / "system_config.yaml"))

    representative_frames_path = merged.get("representative_frames_path", None)
    if representative_frames_path:
        if "${scene_id}" in str(representative_frames_path):
            representative_frames_path = str(representative_frames_path).replace("${scene_id}", scene)
        if not Path(representative_frames_path).is_absolute():
            representative_frames_path = str(project_root / representative_frames_path.lstrip("./"))
    else:
        dataset_path = merged.get("dataset_path", "./dataset")
        if not Path(dataset_path).is_absolute():
            dataset_path = str(project_root / dataset_path.lstrip("./"))
        for scene_var in [scene, scene.replace("_", "")]:
            for sub in ["", "scannet/", "scannet/exported/"]:
                candidate = Path(dataset_path) / sub / scene_var / "representative_frames"
                if candidate.exists():
                    representative_frames_path = str(candidate)
                    break
            if representative_frames_path:
                break

    if not representative_frames_path or not Path(representative_frames_path).exists():
        logger.warning(f"Could not find representative_frames for scene {scene}, skipping")
        return None

    anchor_key_path = Path(representative_frames_path) / "anchor_key"
    default_anchor_file = anchor_key_path / "anchor_attached_objects.txt"
    default_manifest = anchor_key_path / "manifest.json"
    if not default_anchor_file.exists() or not default_manifest.exists():
        logger.warning(f"Anchor files not found for scene {scene} at {anchor_key_path}, skipping")
        return None

    manifest, saved_anchor_id_map = load_manifest(default_manifest)
    anchors = load_anchor_file(default_anchor_file)
    anchors_with_frames = []
    for anchor in anchors:
        anchor_id = anchor.get("id", "")
        entry = find_manifest_entry(manifest, anchor_id)
        if entry is not None:
            image_paths = collect_image_paths(entry)
            if image_paths:
                anchors_with_frames.append(anchor)
    anchor_id_map = saved_anchor_id_map.copy() if saved_anchor_id_map else {}

    output_path = merged.get("output_path", "./output/map_results")
    dataset_name = "scannet"
    map_save_path = f"{output_path}/{dataset_name}_{scene}/map"
    if not Path(map_save_path).is_absolute():
        map_save_path = str(project_root / map_save_path.lstrip("./"))

    obj_map = load_map_from_disk(map_save_path)
    if not obj_map:
        logger.warning(f"No objects loaded from map for scene {scene}, skipping")
        return None

    return evaluate_scene(
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
        merged,
        llm_model=llm_model,
        vlm_model=vlm_model,
        base_url=base_url,
        api_key=api_key,
        gemini_api_key=gemini_api_key,
        relations=relations,
        map_save_path=map_save_path,
        ablation_mode=ablation_mode,
        max_queries_per_scene=max_queries_per_scene,
        query_sample_seed=query_sample_seed,
        ablation_max_anchors=ablation_max_anchors,
    )


def aggregate_metrics(scene_results_list):
    """Aggregate metrics across scenes; returns counts and totals per IoU."""
    all_metrics = {"0.1": 0, "0.25": 0}
    all_totals = {"0.1": 0, "0.25": 0}
    for sr in scene_results_list:
        if not sr:
            continue
        ov = sr.get("overall", {})
        for iou in ["0.1", "0.25"]:
            if iou in ov:
                all_metrics[iou] += ov[iou].get("count", 0)
                all_totals[iou] += ov[iou].get("total", 0)
    return all_metrics, all_totals


def main():
    parser = argparse.ArgumentParser(
        description="Sr3D+ LLM+VLM retrieval ablation: Full vs w/o LLM Proposal vs w/o VLM Verification"
    )
    parser.add_argument("--scenes", type=str, nargs="+", default=SCENES, help="Scenes to evaluate")
    parser.add_argument("--bbq_data_root", type=str, default=str(BBQ_DATA_ROOT), help="BBQ data root")
    parser.add_argument("--relations", type=str, nargs="+", default=RELATIONS, help="Relation types")
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
        help=f"Gemini VLM model (default: env GEMINI_VLM_MODEL or {DEFAULT_VLM_MODEL})",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        dest="gemini_api_key",
        help="Gemini API key",
    )
    parser.add_argument("--base-url", type=str, default=None, help="API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path for ablation summary (default: output/query_results/sr3d_ablation/ablation_results.json)",
    )
    parser.add_argument("--log", type=str, default=None, help="Log file path")
    parser.add_argument(
        "--max_queries_per_scene",
        type=int,
        default=50,
        help="Max queries per scene (subsample with seed for efficiency; default 50). Use 0 or omit for all queries.",
    )
    parser.add_argument(
        "--max_anchors_w_o_llm",
        type=int,
        default=15,
        help="For w/o LLM Proposal: max anchors to verify per query (default 15). Reduces VLM calls.",
    )
    parser.add_argument(
        "--query_seed",
        type=int,
        default=42,
        help="Random seed for query subsampling (default 42) for reproducibility.",
    )
    args = parser.parse_args()

    log_file = args.log or str(project_root / "output" / "query_results" / "sr3d_ablation" / "ablation.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file=log_file)
    max_queries = args.max_queries_per_scene if args.max_queries_per_scene > 0 else None
    ablation_max_anchors = args.max_anchors_w_o_llm if args.max_anchors_w_o_llm > 0 else None
    logger.info("Sr3D+ Ablation: Full | w/o LLM Proposal | w/o VLM Verification")
    logger.info(f"Scenes: {args.scenes}, max_queries_per_scene: {max_queries or 'all'}, max_anchors(w/o_llm): {ablation_max_anchors or 'all'}, seed: {args.query_seed}")

    bbq_data_root = Path(args.bbq_data_root)
    base_cfg_template = OmegaConf.load(project_root / "config" / "base_config.yaml")
    system_cfg = OmegaConf.load(project_root / "config" / "system_config.yaml")
    merged_cfg = OmegaConf.merge(base_cfg_template, system_cfg)
    BaseObject.initialize_config(merged_cfg)

    logger.info("Loading CLIP and FastSAM...")
    clip_model, clip_tokenizer, clip_preprocess, fastsam, device = load_clip_and_fastsam()

    results_by_mode = {}
    for mode in ABLATION_MODES:
        logger.info(f"\n{'='*80}\nAblation mode: {MODE_DISPLAY_NAMES[mode]}\n{'='*80}")
        scene_results_list = []
        for scene in args.scenes:
            logger.info(f"  Scene: {scene} (mode={mode})")
            sr = run_ablation_for_scene(
                scene,
                bbq_data_root,
                merged_cfg,
                clip_model,
                clip_tokenizer,
                clip_preprocess,
                fastsam,
                device,
                llm_model=args.llm_model,
                vlm_model=args.vlm_model,
                base_url=args.base_url,
                api_key=args.api_key,
                gemini_api_key=args.gemini_api_key,
                relations=args.relations,
                ablation_mode=mode,
                logger=logger,
                max_queries_per_scene=max_queries,
                query_sample_seed=args.query_seed,
                ablation_max_anchors=ablation_max_anchors,
            )
            if sr:
                scene_results_list.append(sr)
        all_metrics, all_totals = aggregate_metrics(scene_results_list)
        acc_01 = 100 * all_metrics["0.1"] / all_totals["0.1"] if all_totals["0.1"] > 0 else 0.0
        acc_025 = 100 * all_metrics["0.25"] / all_totals["0.25"] if all_totals["0.25"] > 0 else 0.0
        results_by_mode[mode] = {
            "display_name": MODE_DISPLAY_NAMES[mode],
            "acc_0.1": acc_01,
            "acc_0.25": acc_025,
            "count_0.1": all_metrics["0.1"],
            "count_0.25": all_metrics["0.25"],
            "total": all_totals["0.1"],
            "scene_results": [
                {"scene": r["scene"], "overall": r.get("overall", {})}
                for r in scene_results_list
            ],
        }
        logger.info(f"  {MODE_DISPLAY_NAMES[mode]} -> Acc@0.1: {acc_01:.1f}%, Acc@0.25: {acc_025:.1f}%")

    # Print ablation table
    print("\n" + "=" * 80)
    print("Sr3D+ Retrieval Ablation Results")
    print("=" * 80)
    print(f"{'Method':<30} {'Acc@0.1':>12} {'Acc@0.25':>12}  (total queries: {results_by_mode['full'].get('total', 0)})")
    print("-" * 80)
    for mode in ABLATION_MODES:
        r = results_by_mode[mode]
        print(f"{r['display_name']:<30} {r['acc_0.1']:>10.1f}% {r['acc_0.25']:>10.1f}%")
    print("=" * 80)

    out_path = args.output or str(project_root / "output" / "query_results" / "sr3d_ablation" / "ablation_results.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ablation_modes": ABLATION_MODES,
                "max_queries_per_scene": max_queries,
                "query_sample_seed": args.query_seed,
                "max_anchors_w_o_llm": ablation_max_anchors,
                "results_by_mode": {
                    k: {
                        "display_name": v["display_name"],
                        "acc_0.1": v["acc_0.1"],
                        "acc_0.25": v["acc_0.25"],
                        "count_0.1": v["count_0.1"],
                        "count_0.25": v["count_0.25"],
                        "total": v["total"],
                    }
                    for k, v in results_by_mode.items()
                },
                "scenes": args.scenes,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Ablation summary saved to {out_path}")
    print(f"\nResults saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
