# runner_dataset.py

import json
import logging
import math
import os
import shutil
import time

import hydra
import numpy as np
import psutil
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from dualmap.core import Dualmap
from utils.dataset import dataset_initialization
from utils.eval.eval_utils import (
    get_replica_visibility_data,
    get_visible_instance_ids,
)
from utils.logging_helper import setup_logging
from utils.time_utils import (
    keyframe_timing_memory_cluster_breakdown,
    mean_per_step_time,
    timing_context,
)
from utils.types import DataInput


def _system_used_memory_mb() -> float:
    """Match `free -m` "used" column (MB) for scripts/run_all_replica_resources.sh."""
    return float(psutil.virtual_memory().used) / (1024.0 * 1024.0)


def prepare_keyframe_output_dir(save_cfg, logger):
    if not save_cfg or not getattr(save_cfg, "enabled", False):
        return None

    output_dir = getattr(save_cfg, "output_dir", None)
    if not output_dir:
        logger.warning("[Runner Dataset] keyframe_image_save.output_dir is not set; skipping keyframe export.")
        return None

    output_dir = os.path.abspath(os.path.expanduser(str(output_dir)))

    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            logger.info("[Runner Dataset] Removed existing keyframe dir: %s", output_dir)
        except OSError as exc:
            logger.warning(
                "[Runner Dataset] Failed to remove %s: %s", output_dir, str(exc)
            )
            return None

    os.makedirs(output_dir, exist_ok=True)
    logger.info("[Runner Dataset] Keyframe output directory: %s", output_dir)
    return output_dir


def print_frame_score_details(dualmap, logger):
    """Print per-cluster frame score breakdown (memory cluster / scale-aware score)."""
    if not hasattr(dualmap, "memory_cluster_manager") or dualmap.memory_cluster_manager is None:
        print("[FrameScoreDetails] MemoryClusterManager not available, skipping score details.")
        logger.warning("[FrameScoreDetails] MemoryClusterManager not available, skipping score details.")
        return

    # clusters,all_rep_frames
    clusters = dualmap.memory_cluster_manager.clusters
    if not clusters:
        print("[FrameScoreDetails] No clusters in memory_cluster_manager.")
        logger.warning("[FrameScoreDetails] No clusters in memory_cluster_manager.")
        return

    print("\n" + "=" * 100)
    print("[FrameScoreDetails] Per-representative score details")
    print("=" * 100)
    logger.info("\n" + "=" * 100)
    logger.info("[FrameScoreDetails] Per-representative score details")
    logger.info("=" * 100)

    # clusters
    frame_scores = []
    for cluster in clusters:
        rep = cluster.get("representative")
        if rep is None:
            continue
        
        cluster_id = cluster.get("id", -1)
        saved_path = cluster.get("saved_path", "")
        
        # ,
        frame_idx_from_path = None
        if saved_path:
            import re
            match = re.search(r'frame_(\d+)', saved_path)
            if match:
                frame_idx_from_path = int(match.group(1))
        
        # ,frame_idx
        if frame_idx_from_path is not None:
            frame_idx = frame_idx_from_path
        else:
            frame_idx = rep.frame_idx
            logger.warning(
                "[FrameScoreDetails]  %d: ,frame_idx: %d",
                cluster_id, frame_idx
            )
        
        # 
        if frame_idx_from_path is not None and frame_idx_from_path != rep.frame_idx:
            logger.warning(
                "[FrameScoreDetails]  %d: ! frame_idx: %d, : %d",
                cluster_id, rep.frame_idx, frame_idx_from_path
            )
            logger.warning(
                "[FrameScoreDetails]  %d:  %d ",
                cluster_id, frame_idx_from_path
            )
        
        # ()
        if dualmap.memory_cluster_manager.complex_score_fn is not None:
            actual_score = dualmap.memory_cluster_manager.complex_score_fn(frame_idx)
        elif dualmap.memory_cluster_manager.unified_score_fn is not None:
            actual_score = dualmap.memory_cluster_manager.unified_score_fn(frame_idx)
        else:
            actual_score = getattr(rep, "score", 0.0)
        
        logger.info(
            "[FrameScoreDetails]  %d:  %d (: %s), : %s, : %.6f",
            cluster_id, rep.frame_idx, frame_idx_from_path if frame_idx_from_path is not None else "N/A", saved_path, actual_score
        )
        
        #  compute_frame_score_unified 
        score_details = {}
        if hasattr(dualmap, "compute_frame_score_unified"):
            try:
                score_details = dualmap.compute_frame_score_unified(
                    frame_idx=frame_idx,  # 
                    return_details=True
                )
                logger.debug(
                    "[FrameScoreDetails]  %d:  %d ",
                    cluster_id, frame_idx
                )
            except Exception as e:
                logger.warning(
                    "[FrameScoreDetails]  %d:  %d : %s",
                    cluster_id, frame_idx, str(e)
                )
        
        # 
        details = {
            "frame_idx": frame_idx,  # 
            "rep_frame_idx": rep.frame_idx,  # frame_idx
            "cluster_id": cluster_id,
            "saved_path": saved_path,
            "score": actual_score,  # 
            "anchor_richness": getattr(rep, "anchor_richness", 0.0),
            "object_richness": getattr(rep, "object_richness", 0.0),
            "anchor_count": getattr(rep, "anchor_count", 0),
            "object_count": getattr(rep, "object_count", 0),
            "anchor_conf_sum": getattr(rep, "anchor_conf_sum", 0.0),
            "foreground_ratio": getattr(rep, "foreground_ratio", 0.0),
            "anchor_names": getattr(rep, "anchor_names", []),
            "valid_observations": rep.valid_observations if hasattr(rep, "valid_observations") else [],
        }
        # 
        details.update(score_details)

        frame_scores.append(details)
    # ID()
    frame_scores.sort(key=lambda x: x.get("cluster_id", 0))
    
    # 
    print(f"\n[FrameScoreDetails]  {len(frame_scores)} \n")
    logger.info(f"\n[FrameScoreDetails]  {len(frame_scores)} \n")
    
    for idx, details in enumerate(frame_scores, 1):
        frame_idx = details["frame_idx"]
        cluster_id = details.get("cluster_id", -1)
        saved_path = details.get("saved_path", "")
        score = details.get("score", 0.0)
        M = details.get("M", 0)
        constants = details.get("constants", {})
        
        # 
        logger.info(
            "[FrameScoreDetails]  %d:  %d, : %s, : %.6f",
            cluster_id, frame_idx, saved_path, score
        )
        
        # (frame_idxframe_idx,)
        rep_frame_idx = details.get("rep_frame_idx", frame_idx)
        if rep_frame_idx != frame_idx:
            logger.warning(
                "[FrameScoreDetails] : frame_idx (%d) frame_idx (%d) ",
                rep_frame_idx, frame_idx
            )
            print(f"  ⚠️  : frame_idx ({rep_frame_idx}) frame_idx ({frame_idx}) ")
            print(f"  :  {frame_idx}, {rep_frame_idx}")
        
        print("\n" + "-" * 100)
        print(f"[FrameScoreDetails]  #{idx}:  {frame_idx}, ID: {cluster_id}, : {score:.6f}")
        if saved_path:
            print(f"  : {saved_path}")
        if rep_frame_idx != frame_idx:
            print(f"  : frame_idx {rep_frame_idx}, {frame_idx}")
        print("-" * 100)
        logger.info("\n" + "-" * 100)
        logger.info(f"[FrameScoreDetails]  #{idx}:  {frame_idx}, : {score:.6f}")
        logger.info("-" * 100)

        # 
        M = details.get("M", 0)
        A_sum = details.get("A_sum", 0.0)
        H_anchor = details.get("H_anchor", 0.0)
        H_balance_norm = details.get("H_balance_norm", details.get("H_norm", 0.0))
        r_coverage = details.get("r_coverage", 0.0)
        A_img_detail = details.get("A_img")
        M_capped = details.get("M_capped", 0)
        N_obj_capped = details.get("N_obj_capped", 0)
        GlobalTerm_raw = details.get("GlobalTerm_raw", 0.0)
        GlobalTerm_max = details.get("GlobalTerm_max", 0.0)
        GlobalTerm_norm = details.get("GlobalTerm_norm", 0.0)
        LocalValue_list = details.get("LocalValue_list", [])
        LocalMax = details.get("LocalMax", 0.0)
        LocalMean = details.get("LocalMean", 0.0)
        LocalTerm_raw = details.get("LocalTerm_raw", 0.0)
        LocalTerm_max = details.get("LocalTerm_max", 0.0)
        LocalTerm_norm = details.get("LocalTerm_norm", 0.0)
        w_global = details.get("w_global", 0.0)
        w_local = details.get("w_local", 0.0)
        constants = details.get("constants", {})
        lambda_s = float(details.get("lambda_scale", 0.7))

        # 
        anchor_details = details.get("anchor_details", [])
        other_object_details = details.get("other_object_details", [])
        other_count = details.get("other_object_count", 0)
        valid_observations = details.get("valid_observations", [])
        
        #  valid_observations ()
        # ,
        all_anchor_class_observations = []
        if valid_observations and hasattr(dualmap.memory_cluster_manager, "_is_anchor_class_name"):
            for obs in valid_observations:
                class_name = dualmap.memory_cluster_manager._get_class_name(obs)
                if class_name and dualmap.memory_cluster_manager._is_anchor_class_name(class_name):
                    matched_uid = getattr(obs, "matched_obj_uid", None)
                    conf = getattr(obs, "conf", 0.0)
                    
                    all_anchor_class_observations.append({
                        "class_name": class_name,
                        "uid": str(matched_uid) if matched_uid else "",
                        "conf": conf,
                    })
        
        print(f"\n (M = {M} ):")
        print(f"   A_sum = {A_sum:.2f}")
        
        # ()
        if all_anchor_class_observations:
            print(f"   ( {len(all_anchor_class_observations)} ,YOLO):")
            for i, obs_info in enumerate(all_anchor_class_observations, 1):
                class_name = obs_info.get("class_name", "unknown")
                uid = obs_info.get("uid", "unknown")
                conf = obs_info.get("conf", 0.0)
                obj_info = obs_info.get("obj_info")
                
                #  anchor_details ()
                is_included = False
                for anchor_info in anchor_details:
                    if anchor_info.get("uid") == uid or (uid != "" and anchor_info.get("uid", "").startswith(uid[:8])):
                        is_included = True
                        break
                
                # 
                # ,
                if is_included:
                    status = "✓ "
                else:
                    # ,
                    status = "✗ ()"
                
                print(f"     {i}: {class_name} (UID: {uid[:8] if uid != '' else ''}...), : {conf:.3f}, {status}")
        
        # 
        if anchor_details:
            print(f"\n   ( {len(anchor_details)} ):")
            for i, anchor_info in enumerate(anchor_details, 1):
                uid = anchor_info.get("uid", "unknown")
                class_name = anchor_info.get("class_name", "unknown")
                area = anchor_info.get("area", 0.0)
                completeness = anchor_info.get("completeness", 0.0)
                interaction_count = anchor_info.get("interaction_count", 0)
                print(f"     {i}: {class_name} (UID: {uid[:8]}...), : {area:.2f}, : {completeness:.3f}, : {interaction_count}")
        else:
            print(f"  : ")
        
        # 
        if len(anchor_details) != M:
            logger.warning(
                "[FrameScoreDetails] : ! M = %d, anchor_details = %d",
                M, len(anchor_details)
            )
            print(f"  ⚠️  : ! M = {M}, anchor_details = {len(anchor_details)}")
        
        # (,)
        if all_anchor_class_observations and len(all_anchor_class_observations) != M:
            print(f"\n  :  {len(all_anchor_class_observations)} , {M} ")
            if len(all_anchor_class_observations) > M:
                print(f"  ")
        
        # 
        print(f"\n ( {other_count} ):")
        if other_object_details:
            print(f"   ( {len(other_object_details)} ):")
            for i, obj_info in enumerate(other_object_details, 1):
                uid = obj_info.get("uid", "unknown")
                class_name = obj_info.get("class_name", "unknown")
                is_low_mobility = obj_info.get("is_low_mobility", False)
                is_stable = obj_info.get("is_stable", False)
                print(f"     {i}: {class_name} (UID: {uid[:8]}...), : {is_low_mobility}, : {is_stable}")
        else:
            print(f"  : ")
        
        # 
        if len(other_object_details) != other_count:
            logger.warning(
                "[FrameScoreDetails] : ! other_count = %d, other_object_details = %d",
                other_count, len(other_object_details)
            )
            print(f"  ⚠️  : ! other_count = {other_count}, other_object_details = {len(other_object_details)}")

        print(f"\n,( scale-aware ):")
        if M == 0:
            print(f"  : r; S_global(),w_global=1, w_local=0")
        else:
            print(f"  A_img = {A_img_detail if A_img_detail is not None else '( r=0)'}")
            if A_img_detail is not None:
                print(f"  r(f) = A_sum / A_img = {A_sum:.6f} / {float(A_img_detail):.6f} = {r_coverage:.6f}")
            else:
                print(f"  r(f) = 0( A_img)→ r = {r_coverage:.6f}")
            print(f"  H_anchor = -Σ p_j log(p_j) = {H_anchor:.6f}")
            if M > 1:
                print(f"  H / log(M) = {H_anchor:.6f} / {math.log(M):.6f} = {H_balance_norm:.6f}")
            else:
                print(f"  M ≤ 1 ")

        print(f"\n, GlobalTerm_norm:")
        if M == 0:
            # ,
            other_count = details.get("other_object_count", 0)
            print(f"   (M_capped = 0)")
            print(f"  N_obj_capped = min(, O_max) = min({other_count}, {constants.get('O_max', 10.0):.2f}) = {N_obj_capped}")
            print(f"  GlobalTerm_raw = α_o - N_obj_capped = {constants.get('alpha_o', 0.3):.2f} - {N_obj_capped}")
            print(f"                 = {GlobalTerm_raw:.6f}")
        else:
            # 
            other_count = details.get("other_object_count", 0)
            print(f"  M_capped = min(M, A_max) = min({M}, {constants.get('A_max', 5.0):.2f}) = {M_capped}")
            print(f"  N_obj_capped = min(, O_max) = min({other_count}, {constants.get('O_max', 10.0):.2f}) = {N_obj_capped}")
            print(f"  GlobalTerm_raw = α_a - M_capped + α_o - N_obj_capped")
            print(f"                 = {constants.get('alpha_a', 0.7):.2f} - {M_capped} + {constants.get('alpha_o', 0.3):.2f} - {N_obj_capped}")
            print(f"                 = {GlobalTerm_raw:.6f}")
        print(f"  GlobalTerm_max = α_a - A_max + α_o - O_max")
        print(f"                 = {constants.get('alpha_a', 0.7):.2f} - {constants.get('A_max', 5.0):.2f} + {constants.get('alpha_o', 0.3):.2f} - {constants.get('O_max', 10.0):.2f}")
        print(f"                 = {GlobalTerm_max:.6f}")
        print(f"  GlobalTerm_norm = GlobalTerm_raw / GlobalTerm_max = {GlobalTerm_raw:.6f} / {GlobalTerm_max:.6f} = {GlobalTerm_norm:.6f}")

        print(f"\n, LocalTerm_norm:")
        for i, local_val in enumerate(LocalValue_list):
            print(f"   {i+1} LocalValue = β_e - E_i + β_c - g(C_i) = {local_val:.6f}")

        print(f"  LocalMax = {LocalMax:.6f}")
        print(f"  LocalMean = {LocalMean:.6f}")
        print(f"  LocalTerm_raw = γ_1 - LocalMax + γ_2 - LocalMean")
        print(f"                = {constants.get('gamma_1', 0):.2f} - {LocalMax:.6f} + {constants.get('gamma_2', 0):.2f} - {LocalMean:.6f}")
        print(f"                = {LocalTerm_raw:.6f}")
        local_value_max = constants.get('beta_e', 0) * 1.0 + constants.get('beta_c', 0) * constants.get('C_max', 0)
        print(f"  LocalValue_max = β_e - 1 + β_c - C_max = {constants.get('beta_e', 0):.2f} - 1 + {constants.get('beta_c', 0):.2f} - {constants.get('C_max', 0):.2f} = {local_value_max:.6f}")
        print(f"  LocalTerm_max = (γ_1 + γ_2) - LocalValue_max = ({constants.get('gamma_1', 0):.2f} + {constants.get('gamma_2', 0):.2f}) - {local_value_max:.6f} = {LocalTerm_max:.6f}")
        print(f"  LocalTerm_norm = LocalTerm_raw / LocalTerm_max = {LocalTerm_raw:.6f} / {LocalTerm_max:.6f} = {LocalTerm_norm:.6f}")

        S_global = details.get("S_global", GlobalTerm_norm)
        S_local = details.get("S_local", LocalTerm_norm)
        print(f"\n,Scale-aware :")
        print(f"  λ = {lambda_s:.6f}")
        if M == 0:
            print(f"  w_global = 1.0, w_local = 0.0(,)")
        elif M <= 1:
            print(f"  w_global = 1 - r = 1 - {r_coverage:.6f} = {w_global:.6f}")
            print(f"  w_local = 1 - w_global = {w_local:.6f}")
        else:
            print(f"  w_global = (1-λ)(1-r) + λ-H/log(M)")
            print(f"           = (1-{lambda_s:.4f})(1-{r_coverage:.6f}) + {lambda_s:.4f}-{H_balance_norm:.6f} = {w_global:.6f}")
            print(f"  w_local = 1 - w_global = {w_local:.6f}")
        print(f"  S_global = GlobalTerm_norm = {S_global:.6f}, S_local = LocalTerm_norm = {S_local:.6f}")
        print(f"  Score = w_global-S_global + w_local-S_local")
        print(f"        = {w_global:.6f} - {S_global:.6f} + {w_local:.6f} - {S_local:.6f}")
        print(f"        = {score:.6f}")
        
    
    print("\n" + "=" * 100)
    print("[FrameScoreDetails] End of score details")
    print("=" * 100 + "\n")
    logger.info("\n" + "=" * 100)
    logger.info("[FrameScoreDetails] End of score details")
    logger.info("=" * 100 + "\n")


def save_keyframe_image(output_dir, dataset, image_idx, keyframe_idx, logger, dualmap=None):
    if not output_dir:
        return

    import imageio.v2 as imageio
    import numpy as np
    from utils.visualizer import visualize_result_rgb

    color_paths = getattr(dataset, "color_paths", None)
    src_path = None
    if color_paths and 0 <= image_idx < len(color_paths):
        src_path = color_paths[image_idx]

    if not src_path or not os.path.exists(src_path):
        logger.warning(
            "[Runner Dataset] Missing color image for dataset index %d; skip keyframe save.",
            image_idx,
        )
        return

    # 
    full_image = imageio.imread(str(src_path))
    
    # Resize to match dataset configuration to avoid visualization errors with masks
    if hasattr(dataset, 'desired_height') and hasattr(dataset, 'desired_width'):
        import cv2
        full_image = cv2.resize(full_image, (dataset.desired_width, dataset.desired_height))
    
    # 
    dst_name_original = f"{keyframe_idx:06d}_original_{os.path.basename(src_path)}"
    dst_path_original = os.path.join(output_dir, dst_name_original)
    
    try:
        imageio.imwrite(dst_path_original, full_image)
        logger.info("[Runner Dataset] Saved original frame: %s", dst_path_original)
    except OSError as exc:
        logger.warning(
            "[Runner Dataset] Failed to save original (kf=%d): %s", keyframe_idx, str(exc)
        )
    
    if dualmap is not None and hasattr(dualmap, 'detector'):
        detector = dualmap.detector
        logger.info(
            "[Runner Dataset] detector state: hasattr(curr_detections)=%s, curr_detections set=%s",
            hasattr(detector, "curr_detections"),
            detector.curr_detections is not None if hasattr(detector, "curr_detections") else False,
        )
        if (hasattr(detector, 'curr_detections') and
            detector.curr_detections is not None and
            len(detector.curr_detections.xyxy) > 0):
            logger.info(
                "[Runner Dataset] curr_detections type=%s, num boxes=%s",
                type(detector.curr_detections),
                len(detector.curr_detections.xyxy),
            )
            try:
                class_lookup = detector.obj_classes.get_classes_arr()
                annotated_image, _ = visualize_result_rgb(
                    full_image.copy(),
                    detector.curr_detections,
                    class_lookup,
                    draw_bboxes=True,
                )
                
                dst_name_annotated = f"{keyframe_idx:06d}_yolo_sam_{os.path.basename(src_path)}"
                dst_path_annotated = os.path.join(output_dir, dst_name_annotated)
                imageio.imwrite(dst_path_annotated, annotated_image)
                logger.info("[Runner Dataset] Saved annotated frame: %s", dst_path_annotated)
            except Exception as exc:
                logger.warning(
                    "[Runner Dataset] Annotated save failed (kf=%d): %s", keyframe_idx, str(exc)
                )
                import traceback
                logger.warning("[Runner Dataset] Traceback: %s", traceback.format_exc())
        else:
            logger.info("[Runner Dataset] No detections to annotate for this keyframe.")


@hydra.main(version_base=None, config_path="../config/", config_name="runner_dataset")

def main(cfg: DictConfig):
    print("################dataset_path =", cfg.dataset_path)
    # Set up the logging system
    setup_logging(output_path=cfg.output_path, config_path=cfg.logging_config)
    logger = logging.getLogger(__name__)

    logger.warning("[Runner Dataset]")
    logger.info(OmegaConf.to_yaml(cfg))

    # Initialize Dualmap instance
    dualmap = Dualmap(cfg)

    # Dataset Object Initialization
    dataset = dataset_initialization(cfg)
    dualmap.dataset = dataset
    keyframe_save_cfg = getattr(cfg, "keyframe_image_save", None)
    keyframe_output_dir = prepare_keyframe_output_dir(keyframe_save_cfg, logger)

    keyframe_count = 0
    wall_kf_start = None
    mem_samples_mb: list[float] = []

    # Traverse and process dataset
    for image_idx in trange(len(dataset), desc="Processing"):
        time_stamp = dataset.time_stamps[image_idx]
        curr_pose = dataset.poses[image_idx].cpu().numpy()

        if cfg.use_stride:
            is_keyframe = True
            kf_idx = image_idx
        else:
            is_keyframe = dualmap.check_keyframe(time_stamp, curr_pose)
            kf_idx = dualmap.get_keyframe_idx()

        if not is_keyframe:
            continue

        keyframe_count += 1
        if wall_kf_start is None:
            wall_kf_start = time.perf_counter()

        logger.info(
            "[Main] ============================================================"
        )
        logger.info(
            f"[Main] Keyframe idx: {kf_idx}, image idx: {image_idx}, Time: {time_stamp}"
        )

        color, color_name = dataset.get_color(image_idx)
        depth = dataset.get_depth(image_idx)
        intrinsics = dataset.get_intrinsics(image_idx)

        curr_data = DataInput(
            idx=kf_idx,
            raw_idx=image_idx,
            color=color,
            depth=depth,
            color_name=color_name,
            intrinsics=intrinsics,
            pose=curr_pose,
        )

        # Process data using Dualmap instance
        with timing_context("Time Per Frame", dualmap):
            if cfg.use_parallel:
                dualmap.parallel_process(curr_data)
            else:
                dualmap.sequential_process(curr_data)
        mem_samples_mb.append(_system_used_memory_mb())

        # ()
        save_keyframe_image(keyframe_output_dir, dataset, image_idx, kf_idx, logger, dualmap=dualmap)

    # , end_process; ( end_process)
    if keyframe_count > 0 and wall_kf_start is not None:
        if getattr(cfg, "use_parallel", False):
            dualmap.wait_until_parallel_mapping_idle()
            mem_samples_mb.append(_system_used_memory_mb())

    # End-process stage if configured
    if cfg.use_end_process:
        logger.info("[Main][EndProcess] End Processing Start.")
        dualmap.end_process()
        mem_samples_mb.append(_system_used_memory_mb())
        print("Done")
        
        # 
        print_frame_score_details(dualmap, logger)

    # OASIS : Replica GT  9  Instance Coverage  Frame Count
    if getattr(cfg, "run_oasis_ablation", False):
        managers = getattr(dualmap, "_oasis_ablation_managers", [])
        if managers:
            run_oasis_ablation_replica_report(cfg, dualmap, dataset, logger)
        else:
            print("\n[OASIS Ablation] run_oasis_ablation=True but _oasis_ablation_managers is empty (enable memory_cluster).\n", flush=True)
            logger.warning("[OASIS Ablation] _oasis_ablation_managers is empty; skipping ablation report.")

    wall_kf_sec = None
    if keyframe_count > 0 and wall_kf_start is not None:
        wall_kf_sec = time.perf_counter() - wall_kf_start

    tr = getattr(dualmap, "timing_results", None)
    avg_s, n_frames = mean_per_step_time(tr)
    if n_frames > 0:
        summary_line = (
            f"[Main] Avg per-step time over {n_frames} frames: {avg_s:.4f}s."
        )
    else:
        summary_line = "[Main] No timing_results frames recorded."
    print(summary_line, flush=True)
    logger.info(summary_line)

    if wall_kf_sec is not None and keyframe_count > 0:
        avg_wall = wall_kf_sec / keyframe_count
        if getattr(cfg, "use_parallel", False):
            mode_tag = "(→"
            if cfg.use_end_process:
                mode_tag += "→end_process"
            mode_tag += ")"
        else:
            mode_tag = "(→"
            if cfg.use_end_process:
                mode_tag += "→end_process"
            mode_tag += ")"
        wall_line = (
            f"[Main] Wall-clock {mode_tag}: {wall_kf_sec:.4f}s / {keyframe_count} keyframes "
            f"= {avg_wall:.4f}s per keyframe (includes parallel queue drain if applicable; "
            f"end_process only if use_end_process; memory_cluster per cfg.memory_cluster.enabled)."
        )
        print(wall_line, flush=True)
        logger.info(wall_line)

    bd = keyframe_timing_memory_cluster_breakdown(
        tr, use_parallel=bool(getattr(cfg, "use_parallel", False))
    )
    if bd is not None and n_frames > 0:
        excl = bd.get("avg_time_per_frame_excl_memory_cluster")
        mc_kf = bd.get("memory_cluster_time_per_keyframe")
        if excl is not None:
            line2 = (
                f"[Main] Avg time/frame excluding memory_cluster: {excl:.4f}s "
                f"(+ {mc_kf:.4f}s memory_cluster per keyframe, "
                f"{bd.get('memory_cluster_invocations_per_keyframe', 0)} invocations/keyframe)."
                if mc_kf is not None
                else f"[Main] Avg time/frame excluding memory_cluster: {excl:.4f}s."
            )
            print(line2, flush=True)
            logger.info(line2)
        elif bd.get("note"):
            print(f"[Main] Note: {bd['note']}", flush=True)
            logger.info("[Main] Note: %s", bd["note"])
        if mc_kf is None and bd.get("mean_memory_cluster_invocation") is not None and getattr(
            cfg, "use_parallel", False
        ):
            mmi = bd["mean_memory_cluster_invocation"]
            line3 = f"[Main] Mean memory_cluster invocation time: {mmi:.4f}s (parallel mode)."
            print(line3, flush=True)
            logger.info(line3)

    if mem_samples_mb:
        avg_mem = float(np.mean(mem_samples_mb))
        peak_mem = float(np.max(mem_samples_mb))
        mem_line = (
            f"[Main] Avg. Mem (MB): {avg_mem:.2f}  Peak Mem (MB): {peak_mem:.2f}  "
            f"(compare with run_all_replica_resources.sh / free -m; {len(mem_samples_mb)} samples)"
        )
        print(mem_line, flush=True)
        logger.info(mem_line)


def run_oasis_ablation_replica_report(cfg: DictConfig, dualmap: Dualmap, dataset, logger):
    """Replica GT report: 9 (clustering × frame selection) modes — instance coverage & frame count."""
    replica_gt_path = getattr(cfg, "replica_gt_path", None) or getattr(cfg, "dataset_path", None)
    replica_gt_scene_id = getattr(cfg, "replica_gt_scene_id", None) or getattr(cfg, "scene_id", None)
    if replica_gt_scene_id is not None:
        replica_gt_scene_id = str(replica_gt_scene_id).strip()
    if not replica_gt_path or not replica_gt_scene_id:
        logger.warning(
            "[OASIS Ablation] Need replica_gt_path and replica_gt_scene_id for Replica GT."
        )
        return
    ply_path = os.path.join(replica_gt_path, replica_gt_scene_id, "habitat", "mesh_semantic.ply")
    semantic_path = os.path.join(replica_gt_path, replica_gt_scene_id, "habitat", "info_semantic.json")
    if not os.path.isfile(ply_path) or not os.path.isfile(semantic_path):
        logger.warning(
            "[OASIS Ablation] Replica GT files missing: %s / %s",
            ply_path, semantic_path,
        )
        return
    try:
        vertices_face, object_ids_face, all_instance_ids = get_replica_visibility_data(
            ply_path, semantic_path
        )
    except Exception as e:
        logger.warning("[OASIS Ablation] Failed to load Replica GT: %s", e)
        return
    num_scene_instances = len(all_instance_ids)
    if num_scene_instances == 0:
        logger.warning("[OASIS Ablation] No scene instances in Replica GT.")
        return

    # : runner  dataset (keyframe  frame_idx)
    def get_pose(frame_idx: int) -> np.ndarray:
        p = dataset.poses[frame_idx]
        return p.cpu().numpy() if hasattr(p, "cpu") else np.asarray(p, dtype=np.float64)

    K = dataset.get_intrinsics(0)
    if hasattr(K, "cpu"):
        K = K.cpu().numpy()
    K = np.asarray(K, dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    width = int(getattr(dataset, "desired_width", None) or 640)
    height = int(getattr(dataset, "desired_height", None) or 480)

    clustering_names = ("Spatial-Only", "Visual-Only", "Hybrid")
    selection_names = ("Global", "Local", "Adaptive Fusion")
    results = []
    for idx, manager in enumerate(dualmap._oasis_ablation_managers):
        c_idx = idx // 3
        s_idx = idx % 3
        rep_frames = []
        for cluster in manager.clusters:
            rep = cluster.get("representative")
            if rep is not None:
                rep_frames.append(rep.frame_idx)
        covered = set()
        for fidx in rep_frames:
            if fidx >= len(dataset.poses):
                continue
            c2w = get_pose(fidx)
            visible = get_visible_instance_ids(
                vertices_face, object_ids_face, c2w, fx, fy, cx, cy, width, height, subsample=10
            )
            covered |= visible
        #  GT , mesh  object_id  > 100%
        covered_in_gt = covered & all_instance_ids
        coverage = len(covered_in_gt) / num_scene_instances if num_scene_instances else 0.0
        frame_count = len(rep_frames)
        results.append({
            "clustering": clustering_names[c_idx],
            "selection": selection_names[s_idx],
            "instance_coverage": coverage,
            "frame_count": frame_count,
            "covered_instances": len(covered_in_gt),
            "scene_instances": num_scene_instances,
        })

    # , aggregate_oasis_ablation_results.py 
    out_dir = getattr(cfg, "output_path", "./output/map_results")
    ablation_dir = os.path.join(out_dir, "oasis_ablation")
    os.makedirs(ablation_dir, exist_ok=True)
    out_json = os.path.join(ablation_dir, f"{replica_gt_scene_id}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"scene": replica_gt_scene_id, "results": results}, f, indent=2, ensure_ascii=False)
    logger.info("[OASIS Ablation] : %s", out_json)

    print("\n" + "=" * 80, flush=True)
    print("OASIS Ablation (Replica GT) — Instance Coverage & Frame Count", flush=True)
    print("=" * 80, flush=True)
    print(f"Scene: {replica_gt_scene_id}  |  Scene instances: {num_scene_instances}", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Clustering':<14} {'Selection':<16} {'Coverage':<10} {'Frame Count':<12}", flush=True)
    print("-" * 80, flush=True)
    for r in results:
        print(f"{r['clustering']:<14} {r['selection']:<16} {r['instance_coverage']:.2%}       {r['frame_count']:<12}", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
