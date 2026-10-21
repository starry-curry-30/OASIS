import csv
import glob
import gzip
import json
import logging
import os
import pickle
import shutil
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2
import matplotlib
from open_clip.transform import F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
import supervision as sv


@dataclass
class FrameStats:
    frame_idx: int
    score: float
    anchor_richness: float
    object_richness: float
    anchor_conf_sum: float
    anchor_count: int
    object_count: int
    clip_feature: np.ndarray
    pose: np.ndarray
    image: Optional[np.ndarray] = None
    anchor_names: List[str] = field(default_factory=list)
    foreground_ratio: float = 0.0
    valid_observations: List = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    anchor_uid_map: Dict[str, Dict] = field(default_factory=dict)
    micro_score: float = 0.0
    micro_details: List[Dict] = field(default_factory=list)
    is_micro_frame: bool = False  # True if anchor_count <= 3, False if > 3
    micro_frame_score: float = 0.0  # Score for micro frames (mask edge completeness + contact count)
    micro_best_anchor_uid: Optional[str] = None  # UID of anchor contributing the best micro score


class MemoryClusterManager:
    """
    ,,.
    """

    def __init__(
        self,
        cfg,
        anchor_whitelist: Optional[set],
        class_lookup: Optional[List[str]],
        clip_feature_fn: Optional[Callable[[np.ndarray], Optional[np.ndarray]]],
        unified_score_fn: Optional[Callable[[int], float]] = None,
        complex_score_fn: Optional[Callable[[int], float]] = None,
        reset_dir: bool = True,
        clustering_mode: str = "hybrid",
    ):
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.clustering_mode = str(clustering_mode).lower() if clustering_mode else "hybrid"
        self.anchor_cap = float(cfg.get("anchor_cap", 3.0))
        self.object_cap = float(cfg.get("object_cap", 10.0))
        self.anchor_weight = float(cfg.get("anchor_weight", 0.7))
        self.object_weight = float(cfg.get("object_weight", 0.3))
        self.clip_similarity_threshold = float(cfg.get("clip_similarity_threshold", 0.8))
        self.translation_threshold = float(cfg.get("translation_threshold", 0.5))
        self.rotation_threshold = float(cfg.get("rotation_threshold", 30.0))
        if self.clustering_mode == "visual":
            self.translation_threshold = 1e9
            self.rotation_threshold = 1e9
        self.delta_clip = 1.0 - self.clip_similarity_threshold
        self.min_foreground_ratio = float(cfg.get("min_foreground_ratio", 0.3))
        self.loop_check_enabled = bool(cfg.get("loop_check_enabled", False))
        self.loop_position_threshold = float(cfg.get("loop_position_threshold", 0.5))
        self.loop_angle_threshold_deg = float(cfg.get("loop_angle_threshold_deg", 20.0))
        self.topdown_fov_deg = float(cfg.get("topdown_fov_deg", 80.0))
        self.topdown_view_range = float(cfg.get("topdown_view_range", 5.0))
        self.micro_anchor_weight = float(cfg.get("micro_anchor_weight", 0.5))
        self.micro_movable_weight = float(cfg.get("micro_movable_weight", 0.5))
        self.unified_score_fn = unified_score_fn
        self.complex_score_fn = complex_score_fn
        self.micro_contact_norm = float(cfg.get("micro_contact_norm", 5.0))
        self.topdown_fov_deg = float(cfg.get("topdown_fov_deg", 80.0))
        self.topdown_view_range = float(cfg.get("topdown_view_range", 5.0))
        # :w_j = s_j^alpha
        self.score_alpha = float(cfg.get("score_alpha", 1.0))

        # Candidate frame policy (clustering + representative selection)
        self.candidate_require_anchor = bool(cfg.get("candidate_require_anchor", False))
        self.candidate_use_foreground_filter = bool(
            cfg.get("candidate_use_foreground_filter", True)
        )
        self.candidate_require_clip = bool(cfg.get("candidate_require_clip", True))
        # (,Detector  bbox ); <=0 
        self.max_observation_depth_m = float(cfg.get("max_observation_depth_m", 0.0) or 0.0)

        default_background = {"wall", "floor", "ceiling", "door"}
        background_cfg = cfg.get("background_classes", [])
        self.background_classes = {cls.lower() for cls in background_cfg} if background_cfg else default_background

        self.anchor_whitelist = {cls.lower() for cls in (anchor_whitelist or set())}
        self.class_lookup = class_lookup or []
        self.clip_feature_fn = clip_feature_fn

        #  - representative_frames_path,keyframe_dir
        keyframe_dir = cfg.get("representative_frames_path")
        if keyframe_dir is None:
            keyframe_dir = cfg.get("keyframe_dir", "keyframe")
        self.keyframe_dir = os.path.abspath(os.path.expanduser(str(keyframe_dir)))
        self._keyframe_dir_ready = False
        if reset_dir:
            self._reset_keyframe_dir()

        self.clusters: List[Dict] = []
        self.current_cluster: Optional[Dict] = None

        self.trigger_counters = {
            "clip": 0,
            "translation": 0,
            "rotation": 0,
        }
        self.all_frame_poses: List[Dict] = []
        self._last_candidate_skip: Optional[str] = None

    def _reset_keyframe_dir(self) -> None:
        try:
            if os.path.exists(this_dir := self.keyframe_dir):
                shutil.rmtree(this_dir)
            os.makedirs(this_dir, exist_ok=True)
            self._keyframe_dir_ready = True
        except OSError as exc:
            self.logger.warning("[MemoryCluster]  keyframe : %s", exc)

    def update_class_lookup(self, class_lookup: List[str]) -> None:
        self.class_lookup = class_lookup or []

    def load_clusters(self, load_dir: str) -> None:
        """Load clusters from a directory containing .pkl files."""
        if not os.path.exists(load_dir):
            self.logger.warning(f"[MemoryCluster] Load directory {load_dir} does not exist.")
            return

        pkl_files = sorted([f for f in os.listdir(load_dir) if f.endswith(".pkl")])
        if not pkl_files:
            self.logger.warning(f"[MemoryCluster] No .pkl files found in {load_dir}.")
            return

        self.logger.info(f"[MemoryCluster] Loading clusters from {load_dir} ({len(pkl_files)} files)...")
        
        clusters_map = {}

        for pkl_file in pkl_files:
            try:
                with open(os.path.join(load_dir, pkl_file), "rb") as f:
                    data = pickle.load(f)
                
                cluster_id = data.get("cluster_id")
                if cluster_id is None:
                    continue
                
                # Reconstruct valid_observations from metadata
                valid_obs = []
                metadata = data.get("metadata", {})
                
                # Helper class to mimic observation object
                class SimpleObs:
                    def __init__(self, uid):
                        self.matched_obj_uid = str(uid)
                
                # Extract UIDs from mask data
                for key in ['anchor_mask_data', 'movable_mask_data']:
                    mask_data = metadata.get(key, [])
                    if mask_data:
                        for item in mask_data:
                            uid = item.get('uid')
                            if uid:
                                valid_obs.append(SimpleObs(uid))

                # Reconstruct FrameStats for representative
                stats = FrameStats(
                    frame_idx=data["frame_idx"],
                    score=data["score"],
                    anchor_richness=0.0, # Not strictly needed for comparison if unified score is used
                    object_richness=0.0,
                    anchor_conf_sum=0.0,
                    anchor_count=data.get("anchor_count", 0),
                    object_count=data.get("object_count", 0),
                    clip_feature=data.get("clip_feature"),
                    pose=data.get("pose"),
                    metadata=data.get("metadata", {}),
                    is_micro_frame=data.get("is_micro", False),
                    micro_frame_score=data["score"] if data.get("is_micro", False) else 0.0,
                    valid_observations=valid_obs  # Pass the reconstructed observations
                )
                
                cluster = {
                    "id": cluster_id,
                    "frames": [stats], # Only keep representative for now
                    "representative": stats,
                    "saved_path": None, # Will be set if we save it again
                    "saved_info_path": None,
                    "cluster_type": "micro" if data.get("is_micro", False) else "macro",
                    "micro_frame_count": 1 if data.get("is_micro", False) else 0,
                    "macro_frame_count": 1 if not data.get("is_micro", False) else 0,
                    "novelty_reason": "loaded",
                    "V_sum": data.get("V_sum"),
                    "S_sum": data.get("S_sum"),
                    "cluster_vec_norm": data.get("cluster_vec_norm"),
                }
                
                clusters_map[cluster_id] = cluster
                
            except Exception as e:
                self.logger.warning(f"[MemoryCluster] Failed to load {pkl_file}: {e}")

        # Rebuild self.clusters list (sort by ID)
        self.clusters = [clusters_map[i] for i in sorted(clusters_map.keys())]
        self.logger.info(f"[MemoryCluster] Successfully loaded {len(self.clusters)} clusters.")

    def process_frame(
        self,
        frame_idx: int,
        color_image: np.ndarray,
        pose: np.ndarray,
        observations: Optional[List],
        precomputed_clip_feature: Optional[np.ndarray] = None,
        has_new_instance: bool = False,
    ) -> Optional[Dict]:
        """,."""
        self._last_candidate_skip = None
        if color_image is None or pose is None:
            self._last_candidate_skip = "invalid_image_or_pose"
            return None

        if self.candidate_require_clip:
            if self.clip_feature_fn is None and precomputed_clip_feature is None:
                self.logger.warning("[MemoryCluster] CLIP,.")
                self._last_candidate_skip = "no_clip_pipeline"
                return None
        elif self.clip_feature_fn is None and precomputed_clip_feature is None:
            self.logger.debug(
                "[MemoryCluster] candidate_require_clip=false: proceeding without CLIP pipeline"
            )

        clip_feature = precomputed_clip_feature
        if clip_feature is None and self.clip_feature_fn is not None:
            clip_feature = self.clip_feature_fn(color_image)
        if clip_feature is None:
            if self.candidate_require_clip:
                self.logger.warning("[MemoryCluster] CLIP,.")
                self._last_candidate_skip = "clip_compute_failed"
                return None
            self.logger.debug(
                "[MemoryCluster] candidate_require_clip=false: clip unavailable, pose-only hybrid branch"
            )

        stats = self._compute_frame_stats(frame_idx, color_image, pose, observations or [], clip_feature, has_new_instance)
        if stats is None:
            return None

        self.all_frame_poses.append(
            {
                "frame_idx": frame_idx,
                "pose": stats.pose.copy(),
            }
        )

        if self.current_cluster is None:
            cluster = self._start_new_cluster(stats, reason="first_frame")
            return self._build_result(cluster, stats, is_new_cluster=True, novelty_reason="first_frame")

        # ,
        matched_cluster = None
        matched_novelty = None
        
        # ,
        cluster_distances = []
        for cluster in self.clusters:
            rep: Optional[FrameStats] = cluster.get("representative")
            if rep is None or rep.pose is None:
                continue
            
            # ()
            translation_diff = float(
                np.linalg.norm(stats.pose[:3, 3] - rep.pose[:3, 3])
            )
            cluster_distances.append((cluster, translation_diff))
        
        # ()
        cluster_distances.sort(key=lambda x: x[1])
        
        # ,
        for cluster, distance in cluster_distances:
            novelty = self._check_novelty(stats, cluster)
            if not novelty["is_new_cluster"]:
                # 
                matched_cluster = cluster
                matched_novelty = novelty
                self.logger.info(
                    "[MemoryCluster]  %d  %d (=%.3f)",
                    frame_idx, cluster.get("id", -1), distance
                )
                print(
                    f"[MemoryCluster]  {frame_idx}  {cluster.get('id', -1)} (={distance:.3f})"
                )
                break
        
        # ,
        if matched_cluster is not None:
            self._append_frame(matched_cluster, stats, matched_novelty)
            self.current_cluster = matched_cluster  # 
            return self._build_result(matched_cluster, stats, is_new_cluster=False, novelty_reason="same_cluster", extra=matched_novelty)
        
        # ,
        # novelty()
        if cluster_distances:
            # 
            nearest_cluster, nearest_distance = cluster_distances[0]
            novelty = self._check_novelty(stats, nearest_cluster)
            reason = novelty.get("reason", "no_match")
            self.logger.info(
                "[MemoryCluster]  %d , (=%.3f, =%s)",
                frame_idx, nearest_distance, reason
            )
            print(
                f"[MemoryCluster]  {frame_idx} , (={nearest_distance:.3f}, ={reason})"
            )
        else:
            # ,
            novelty = {"is_new_cluster": True, "reason": "no_match"}
            reason = "no_match"
        
        cluster = self._start_new_cluster(stats, reason=reason, novelty_info=novelty)
        return self._build_result(cluster, stats, is_new_cluster=True, novelty_reason=reason, extra=novelty)

    def _obs_beyond_max_depth(self, obs) -> bool:
        m = self.max_observation_depth_m
        if m is None or m <= 0:
            return False
        d = getattr(obs, "distance", None)
        try:
            df = float(d)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(df):
            return False
        return df > m

    def _compute_frame_stats(
        self,
        frame_idx: int,
        color_image: np.ndarray,
        pose: np.ndarray,
        observations: List,
        clip_feature: Optional[np.ndarray],
        has_new_instance: bool = False,
    ) -> Optional[FrameStats]:
        pose_np = np.asarray(pose, dtype=np.float32)
        if pose_np.shape[0] < 3 or pose_np.shape[1] < 4:
            self.logger.warning("[MemoryCluster] , %d.", frame_idx)
            self._last_candidate_skip = "invalid_pose_shape"
            return None

        image_np = np.asarray(color_image)
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        anchor_conf_sum = 0.0
        anchor_count = 0
        object_count = 0
        anchor_names: List[str] = []
        debug_objects: List[dict] = []
        valid_observations: List = []
        anchor_uid_map: Dict[str, Dict] = {}
        img_height = int(image_np.shape[0])
        img_width = int(image_np.shape[1])
        total_pixels = float(max(img_height * img_width, 1))
        anchor_mask_data: List[Dict] = []
        movable_mask_data: List[Dict] = []

        for obs in observations:
            class_name = self._get_class_name(obs)
            conf = float(getattr(obs, "conf", 0.0))
            if class_name and class_name.lower() in self.background_classes:
                continue
            if self._obs_beyond_max_depth(obs):
                continue
            # ()
            valid_observations.append(obs)
            is_anchor = bool(class_name and self._is_anchor_class_name(class_name))
            debug_objects.append(
                {
                    "class": class_name or "unknown",
                    "conf": conf,
                    "is_anchor": is_anchor,
                }
            )
            matched_uid = getattr(obs, "matched_obj_uid", None)
            if matched_uid:
                uid_str = str(matched_uid)
                entry = anchor_uid_map.setdefault(
                    uid_str, {"class_name": class_name or "unknown", "count": 0}
                )
                entry["count"] += 1

            mask_bool = self._build_binary_mask(obs, img_height, img_width)
            bbox_xyxy = self._get_obs_bbox(obs, img_width, img_height)
            mask_area = float(mask_bool.sum()) if mask_bool is not None else 0.0
            if mask_area <= 0.0:
                mask_area = self._bbox_area(bbox_xyxy)
            entry_dict = {
                "mask": mask_bool,
                "area": mask_area,
                "bbox": bbox_xyxy,
                "class_name": class_name,
                "uid": matched_uid,
            }

            if is_anchor:
                anchor_conf_sum += conf
                anchor_count += 1
                anchor_names.append(class_name)
                anchor_mask_data.append(entry_dict)
            else:
                object_count += 1
                movable_mask_data.append(entry_dict)

        total_bbox_area = self._compute_total_bbox_union_area(valid_observations, img_width, img_height)

        if self.candidate_require_anchor and anchor_count < 1:
            self._last_candidate_skip = "no_anchor_in_frame"
            return None

        foreground_ratio = min(total_bbox_area / max(total_pixels, 1.0), 1.0)
        if self.candidate_use_foreground_filter:
            if foreground_ratio < self.min_foreground_ratio:
                self._last_candidate_skip = "low_foreground_ratio"
                return None

        if self.loop_check_enabled and self._is_duplicate_with_representatives(pose_np):
            self._last_candidate_skip = "loop_duplicate_pose"
            return None

        anchor_richness = min(anchor_conf_sum / self.anchor_cap, 1.0) if self.anchor_cap > 0 else 0.0
        object_richness = min(object_count / self.object_cap, 1.0) if self.object_cap > 0 else 0.0
        score = self.anchor_weight * anchor_richness + self.object_weight * object_richness
        
        # Determine frame type: micro if anchor_count <= 3, macro if > 3
        is_micro_frame = anchor_count <= 3
        
        # Compute micro information (for backward compatibility)
        micro_score, micro_details = self._compute_micro_information(
            anchor_mask_data,
            movable_mask_data,
            float(max(img_height * img_width, 1)),
        )
        
        #  (Novelty Bonus)
        novelty_bonus = 0.3 if has_new_instance else 0.0
        
        #  (Unified Frame Scoring)
        A_max = 5.0
        O_max = 10.0
        alpha_a = 0.7
        alpha_o = 0.3
        
        # 
        norm_anchor = min(float(anchor_count), A_max) / A_max
        norm_object = min(float(object_count), O_max) / O_max
        
        macro_score = alpha_a * norm_anchor + alpha_o * norm_object
        
        #  =  + 
        final_score = macro_score + novelty_bonus
        
        if has_new_instance:
             # 
             new_instance_classes = []
             for obs in observations:
                 if self._obs_beyond_max_depth(obs):
                     continue
                 matched_obj_idx = getattr(obs, "matched_obj_idx", -1)
                 matched_obj_uid = getattr(obs, "matched_obj_uid", None)
                 # 
                 if matched_obj_idx == -1 or matched_obj_uid is None:
                     class_name = self._get_class_name(obs)
                     if class_name and class_name not in new_instance_classes:
                         new_instance_classes.append(class_name)
             
             # 
             class_info = ""
             if new_instance_classes:
                 class_info = f": {', '.join(new_instance_classes)}\n"
             
             msg = (
                 f"\n{'#'*80}\n"
                 f"🔥  🔥\n"
                 f"{class_info}"
                 f" {frame_idx} |  +{novelty_bonus:.1f} | : {macro_score:.3f} | : {final_score:.3f}\n"
                 f"{'#'*80}"
             )
             self.logger.info(msg)
             print(msg)


        # Compute new micro frame score: mask edge completeness + contact count
        micro_frame_score = 0.0
        micro_best_anchor_uid: Optional[str] = None
        if is_micro_frame and anchor_mask_data:
            micro_frame_score, micro_best_anchor_uid = self._compute_micro_frame_score(
                anchor_mask_data,
                movable_mask_data,
                img_height,
                img_width,
            )

        stats = FrameStats(
            frame_idx=frame_idx,
            score=float(final_score),
            anchor_richness=float(anchor_count),
            object_richness=float(object_count),
            anchor_conf_sum=float(anchor_conf_sum),
            anchor_count=anchor_count,
            object_count=object_count,
            clip_feature=clip_feature.copy() if clip_feature is not None else None,
            pose=pose_np.copy(),
            image=np.ascontiguousarray(image_np.copy()),
            anchor_names=anchor_names,
            foreground_ratio=foreground_ratio,
            valid_observations=valid_observations,
            anchor_uid_map=anchor_uid_map,
            micro_score=micro_score,
            micro_details=micro_details,
            is_micro_frame=is_micro_frame,
            micro_frame_score=micro_frame_score,
            micro_best_anchor_uid=micro_best_anchor_uid,
            metadata={
                "total_pixels": float(img_height * img_width),
                "anchor_mask_data": anchor_mask_data,
                "movable_mask_data": movable_mask_data,
                "has_new_instance": has_new_instance
            },
        )
        if anchor_count == 0:
            self._log_anchor_debug(frame_idx, debug_objects)
        return stats

    def _is_anchor_class_name(self, class_name: Optional[str]) -> bool:
        if not class_name:
            return False
        if not self.anchor_whitelist:
            return False
        return class_name.lower() in self.anchor_whitelist

    def _is_duplicate_with_representatives(self, pose: np.ndarray) -> bool:
        """
        Determine whether the given pose is too close to any existing representative frame.
        """
        if pose is None or pose.shape != (4, 4):
            return False

        position = pose[:3, 3]
        rotation = R.from_matrix(pose[:3, :3])

        for cluster in self.clusters:
            rep: Optional[FrameStats] = cluster.get("representative")
            if rep is None or rep.pose is None or rep.pose.shape != (4, 4):
                continue

            rep_position = rep.pose[:3, 3]
            distance = float(np.linalg.norm(position - rep_position))
            if distance > self.loop_position_threshold:
                continue

            rep_rotation = R.from_matrix(rep.pose[:3, :3])
            angle_diff_deg = self._rotation_difference_deg(rotation, rep_rotation)
            if angle_diff_deg <= self.loop_angle_threshold_deg:
                return True

        return False

    @staticmethod
    def _rotation_difference_deg(rot_a: R, rot_b: R) -> float:
        """
        Compute angular difference between two rotations in degrees.
        """
        rel = rot_a.inv() * rot_b
        return float(np.degrees(rel.magnitude()))

    def _get_class_name(self, obs) -> Optional[str]:
        class_name = getattr(obs, "class_name", None)
        if class_name:
            return class_name
        class_id = getattr(obs, "class_id", None)
        if class_id is None:
            return None
        try:
            class_idx = int(class_id)
        except (TypeError, ValueError):
            return None
        if 0 <= class_idx < len(self.class_lookup):
            return self.class_lookup[class_idx]
        return None

    def _compute_total_bbox_union_area(self, observations: List, img_width: int, img_height: int) -> float:
        rects: List[Tuple[float, float, float, float]] = []
        for obs in observations:
            bbox = self._get_obs_bbox(obs, img_width, img_height)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue
            rects.append((float(x1), float(y1), float(x2), float(y2)))

        if not rects:
            return 0.0

        xs = sorted({x for rect in rects for x in (rect[0], rect[2])})
        total_area = 0.0

        for i in range(len(xs) - 1):
            x_start, x_end = xs[i], xs[i + 1]
            width = x_end - x_start
            if width <= 0:
                continue
            intervals: List[Tuple[float, float]] = []
            for x1, y1, x2, y2 in rects:
                if x1 <= x_start and x2 >= x_end:
                    intervals.append((y1, y2))
            if not intervals:
                continue
            intervals.sort()
            merged = []
            cur_start, cur_end = intervals[0]
            for y_start, y_end in intervals[1:]:
                if y_start <= cur_end:
                    cur_end = max(cur_end, y_end)
                else:
                    merged.append((cur_start, cur_end))
                    cur_start, cur_end = y_start, y_end
            merged.append((cur_start, cur_end))
            total_y = sum(max(0.0, y_end - y_start) for y_start, y_end in merged)
            total_area += width * total_y

        return total_area

    def _build_binary_mask(self, obs, height: int, width: int) -> Optional[np.ndarray]:
        mask = getattr(obs, "mask", None)
        if mask is not None:
            mask_np = np.asarray(mask)
            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                mask_np = mask_np[0]
            if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
                mask_np = mask_np[..., 0]
            mask_np = mask_np.astype(bool)
            if mask_np.shape[0] != height or mask_np.shape[1] != width:
                try:
                    mask_np = cv2.resize(
                        mask_np.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                except Exception:
                    return None
            return mask_np

        xyxy = self._get_obs_bbox(obs, width, height)
        if xyxy is None:
            return None
        x1, y1, x2, y2 = xyxy
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return None
        mask_np = np.zeros((height, width), dtype=bool)
        mask_np[y1:y2, x1:x2] = True
        return mask_np

    def _get_obs_bbox(self, obs, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        xyxy = getattr(obs, "xyxy", None)
        if xyxy is not None:
            arr = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)[0]
            x1, y1, x2, y2 = arr
            x1 = int(max(0.0, min(x1, width)))
            x2 = int(max(0.0, min(x2, width)))
            y1 = int(max(0.0, min(y1, height)))
            y2 = int(max(0.0, min(y2, height)))
            if x2 > x1 and y2 > y1:
                return (x1, y1, x2, y2)
        mask = getattr(obs, "mask", None)
        if mask is not None:
            mask_np = np.asarray(mask)
            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                mask_np = mask_np[0]
            if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
                mask_np = mask_np[..., 0]
            mask_bool = mask_np.astype(bool)
            ys, xs = np.where(mask_bool)
            if len(xs) > 0 and len(ys) > 0:
                x1 = int(np.clip(xs.min(), 0, width))
                x2 = int(np.clip(xs.max() + 1, 0, width))
                y1 = int(np.clip(ys.min(), 0, height))
                y2 = int(np.clip(ys.max() + 1, 0, height))
                if x2 > x1 and y2 > y1:
                    return (x1, y1, x2, y2)
        return None

    def _bbox_intersects(
        self, bbox_a: Optional[Tuple[int, int, int, int]], bbox_b: Optional[Tuple[int, int, int, int]]
    ) -> bool:
        if bbox_a is None or bbox_b is None:
            return True
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

    def _bbox_area(self, bbox: Optional[Tuple[int, int, int, int]]) -> float:
        if bbox is None:
            return 0.0
        x1, y1, x2, y2 = bbox
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    def _log_low_foreground(self, frame_idx: int, ratio: float) -> None:
        msg = (
            f"[MemoryCluster][]  {frame_idx}  {ratio * 100:.1f}% "
            f"< {self.min_foreground_ratio * 100:.1f}%,"
        )
        self.logger.info(msg)
        print(msg)

    def _compute_micro_information(
        self,
        anchor_data: List[Dict],
        movable_data: List[Dict],
        total_pixels: float,
    ) -> Tuple[float, List[Dict]]:
        if not anchor_data or total_pixels <= 0:
            return 0.0, []

        weight_sum = self.micro_anchor_weight + self.micro_movable_weight
        if weight_sum <= 0:
            weight_sum = 1.0
        w_anchor = self.micro_anchor_weight / weight_sum
        w_move = self.micro_movable_weight / weight_sum

        best_score = 0.0
        details: List[Dict] = []

        contact_norm = max(self.micro_contact_norm, 1.0)

        for anchor_entry in anchor_data:
            area_anchor = anchor_entry.get("area", 0.0) or 0.0
            if area_anchor <= 0:
                continue
            bbox_anchor = anchor_entry.get("bbox")

            contact_movables = []
            for mv in movable_data:
                if not self._bbox_intersects(bbox_anchor, mv.get("bbox")):
                    continue
                contact_movables.append(mv)

            contact_count = len(contact_movables)
            norm_anchor = area_anchor / total_pixels
            contact_score = min(contact_count / contact_norm, 1.0)
            micro_score = w_anchor * norm_anchor + w_move * contact_score

            details.append(
                {
                    "anchor_uid": anchor_entry.get("uid"),
                    "class_name": anchor_entry.get("class_name"),
                    "area_anchor": area_anchor,
                    "contact_count": contact_count,
                    "score": micro_score,
                    "contact_movables": [
                        {
                            "class_name": mv.get("class_name"),
                            "area": mv.get("area", 0.0),
                            "uid": mv.get("uid"),
                        }
                        for mv in contact_movables
                    ],
                }
            )

            if micro_score > best_score:
                best_score = micro_score

        return best_score, details

    def _compute_micro_frame_score(
        self,
        anchor_data: List[Dict],
        movable_data: List[Dict],
        img_height: int,
        img_width: int,
    ) -> Tuple[float, Optional[str]]:
        """
        Compute micro frame score: mask edge completeness + contact count with movable objects.
        Returns (best_score, best_anchor_uid) where best_anchor_uid corresponds to the anchor instance
        that contributed the best score (if available).
        """
        if not anchor_data:
            return 0.0, None
        
        best_score = 0.0
        best_anchor_uid: Optional[str] = None
        
        for anchor_entry in anchor_data:
            mask = anchor_entry.get("mask")
            bbox = anchor_entry.get("bbox")
            
            # Compute mask edge completeness
            edge_completeness = self._compute_mask_edge_completeness(mask, bbox, img_height, img_width)
            
            # Count contacts with movable objects
            contact_count = 0
            anchor_bbox = anchor_entry.get("bbox")
            for mv in movable_data:
                if self._bbox_intersects(anchor_bbox, mv.get("bbox")):
                    contact_count += 1
            
            # Normalize contact count (assuming max 10 contacts is good)
            contact_score = min(contact_count / 10.0, 1.0)
            
            # Combined score: edge completeness (0.6 weight) + contact score (0.4 weight)
            frame_score = 0.6 * edge_completeness + 0.4 * contact_score
            
            if frame_score > best_score:
                best_score = frame_score
                uid_val = anchor_entry.get("uid")
                best_anchor_uid = str(uid_val) if uid_val is not None else None
        
        return best_score, best_anchor_uid

    def _compute_mask_edge_completeness(
        self,
        mask: Optional[np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]],
        img_height: int,
        img_width: int,
    ) -> float:
        """
        Compute mask edge completeness: ratio of mask pixels that are NOT touching image edges.
        Higher value means more complete mask (less edge contact).
        """
        if mask is None:
            # If no mask, use bbox to estimate
            if bbox is None:
                return 0.0
            x1, y1, x2, y2 = bbox
            # Check if bbox touches image edges
            touches_edge = (
                x1 <= 1 or y1 <= 1 or
                x2 >= img_width - 1 or y2 >= img_height - 1
            )
            return 0.3 if touches_edge else 0.7  # Rough estimate
        
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.ndim == 3:
            mask_bool = mask_bool[0] if mask_bool.shape[0] == 1 else mask_bool[..., 0]
        
        if mask_bool.sum() == 0:
            return 0.0
        
        # Find edge pixels (pixels that are True and adjacent to False or image boundary)
        edge_pixels = 0
        total_pixels = mask_bool.sum()
        
        # Check if mask touches image boundaries
        touches_top = mask_bool[0, :].any()
        touches_bottom = mask_bool[-1, :].any()
        touches_left = mask_bool[:, 0].any()
        touches_right = mask_bool[:, -1].any()
        
        # Count pixels on mask boundary (using erosion to find boundary)
        eroded = ndimage.binary_erosion(mask_bool, structure=np.ones((3, 3)))
        boundary_mask = mask_bool & (~eroded)
        boundary_pixels = boundary_mask.sum()
        
        # Edge completeness: ratio of non-edge-touching boundary pixels
        # If mask touches image edges, reduce score
        edge_penalty = 0.0
        if touches_top or touches_bottom or touches_left or touches_right:
            edge_penalty = 0.3
        
        # Compute completeness: more boundary pixels that don't touch image edge = better
        if boundary_pixels > 0:
            # Normalize by total mask area
            boundary_ratio = boundary_pixels / max(total_pixels, 1.0)
            completeness = (1.0 - boundary_ratio) * (1.0 - edge_penalty)
        else:
            completeness = 0.5 * (1.0 - edge_penalty)
        
        return max(0.0, min(1.0, completeness))

    def _start_new_cluster(self, stats: FrameStats, reason: str, novelty_info: Optional[Dict] = None) -> Dict:
        cluster_id = len(self.clusters)
        
        # (V_sumS_sum)
        V_sum = None
        S_sum = 0.0
        cluster_vec_norm = None
        
        if stats.clip_feature is not None:
            v_i = stats.clip_feature.copy()
            s_i = stats.score
            w_i = self._compute_frame_weight(s_i)
            V_sum = w_i * v_i
            S_sum = w_i
            # ,cluster_vec_norm = v_i()
            norm = np.linalg.norm(v_i)
            if norm > 1e-6:
                cluster_vec_norm = v_i / norm
            else:
                cluster_vec_norm = v_i
        
        cluster = {
            "id": cluster_id,
            "frames": [stats],
            "representative": stats,
            "saved_path": None,
            "cluster_type": None,  # Will be determined later: "macro" or "micro"
            "micro_frame_count": 0,
            "macro_frame_count": 0,
            "novelty_reason": reason,
            "V_sum": V_sum,  # 
            "S_sum": S_sum,  # 
            "cluster_vec_norm": cluster_vec_norm,  # 
        }
        # Initialize frame type counts
        if stats.is_micro_frame:
            cluster["micro_frame_count"] = 1
        else:
            cluster["macro_frame_count"] = 1
        
        self.clusters.append(cluster)
        self.current_cluster = cluster
        
        # Determine cluster type and select representative
        self._update_cluster_representative(cluster, is_new_cluster=True)

        msg = (
            f"\n{'='*80}\n"
            f"🆕  {cluster_id} | : {cluster['representative'].frame_idx} | : {reason}\n"
            f"{'='*80}"
        )
        self.logger.info(msg)
        print(msg)
        self._log_representative_anchors(cluster_id, cluster["representative"], "")
        if novelty_info and novelty_info.get("is_new_cluster"):
            self._update_trigger_counters(novelty_info)
        if novelty_info:
            self._log_novelty_info(stats.frame_idx, cluster_id, novelty_info)
        return cluster

    def _append_frame(self, cluster: Dict, stats: FrameStats, novelty_info: Dict) -> None:
        cluster["frames"].append(stats)
        self._log_novelty_info(stats.frame_idx, cluster["id"], novelty_info)
        
        # :clip
        self._update_cluster_vector(cluster, stats)
        
        # Update frame type counts
        if stats.is_micro_frame:
            cluster["micro_frame_count"] = cluster.get("micro_frame_count", 0) + 1
        else:
            cluster["macro_frame_count"] = cluster.get("macro_frame_count", 0) + 1
        
        # Update cluster representative based on cluster type
        old_rep_idx = cluster["representative"].frame_idx
        # ()
        old_score = self._get_representative_score(cluster["representative"])
        
        self._update_cluster_representative(cluster, is_new_cluster=False)
        new_rep_idx = cluster["representative"].frame_idx
        # ()
        new_score = self._get_representative_score(cluster["representative"])
        
        if old_rep_idx != new_rep_idx:
            score_diff = new_score - old_score
            score_symbol = "↑" if score_diff > 0 else "↓" if score_diff < 0 else "="
            msg = (
                f"\n{'-'*80}\n"
                f"🔄  ( {cluster['id']}) | : {old_rep_idx} ({old_score:.3f}) → : {new_rep_idx} ({new_score:.3f}) | {score_symbol}{abs(score_diff):.3f}\n"
                f"{'-'*80}"
            )
            self.logger.info(msg)
            print(msg)
            self._log_representative_anchors(cluster["id"], cluster["representative"], "")

        if stats.image is not None:
            stats.image = None

    def _get_representative_score(self, frame_stats: FrameStats) -> float:
        """
        ,.
        .
        """
        if self.complex_score_fn is not None:
            return self.complex_score_fn(frame_stats.frame_idx)
        elif self.unified_score_fn is not None:
            return self.unified_score_fn(frame_stats.frame_idx)
        else:
            # Fallback: use the maximum of available scores (macro score or micro frame score)
            macro_score = getattr(frame_stats, "score", 0.0)
            micro_score = getattr(frame_stats, "micro_frame_score", 0.0)
            return max(macro_score, micro_score)
    
    def _update_cluster_representative(self, cluster: Dict, is_new_cluster: bool = False, unified_score_fn=None) -> None:
        """
        Select the best representative frame using unified frame scoring.
        If unified_score_fn is provided, use it for scoring, otherwise fallback to max of macro/micro scores.
        """
        frames = cluster.get("frames", [])
        if not frames:
            return

        cluster_id = cluster.get("id", -1)
        # Select best representative using unified scoring
        best_frame = None
        best_score = -1.0
        all_scores = []

        for frame_stats in frames:
            if unified_score_fn is not None:
                # Use provided unified scoring function
                frame_idx = frame_stats.frame_idx
                unified_score = unified_score_fn(frame_idx)
            elif self.complex_score_fn is not None:
                # Use complex scoring function (entropy-weighted)
                frame_idx = frame_stats.frame_idx
                unified_score = self.complex_score_fn(frame_idx)
            elif self.unified_score_fn is not None:
                # Use stored unified scoring function
                frame_idx = frame_stats.frame_idx
                unified_score = self.unified_score_fn(frame_idx)
            else:
                # Fallback: use the maximum of available scores (macro score or micro frame score)
                macro_score = getattr(frame_stats, "score", 0.0)
                micro_score = getattr(frame_stats, "micro_frame_score", 0.0)
                unified_score = max(macro_score, micro_score)

            all_scores.append((frame_idx, unified_score))
            if unified_score > best_score:
                best_score = unified_score
                best_frame = frame_stats

        if best_frame is not None:
            cluster["representative"] = best_frame
            # 
            # self.logger.info(
            #     "[MemoryCluster][]  %d:  %d  %d (: %.6f)",
            #     cluster_id, len(frames), best_frame.frame_idx, best_score
            # )
            self.logger.debug(
                "[MemoryCluster][]  %d: : %s",
                cluster_id, ", ".join([f"{idx}({score:.3f})" for idx, score in sorted(all_scores, key=lambda x: x[1], reverse=True)[:5]])
            )
            # Save keyframe with unified naming
            self._save_keyframe(cluster, best_frame, is_new_cluster=is_new_cluster)

    def _check_novelty(self, stats: FrameStats, cluster: Dict) -> Dict:
        """
        :(clip).
        clustering_mode: spatial=, visual=CLIP, hybrid=+CLIP.
        """
        representative: FrameStats = cluster["representative"]
        cluster_id = cluster.get("id", -1)
        frame_idx = stats.frame_idx
        rep_frame_idx = representative.frame_idx

        # 1.  ()
        translation_diff = float(
            np.linalg.norm(stats.pose[:3, 3] - representative.pose[:3, 3])
        )
        rotation_diff = float(self._rotation_difference(stats.pose[:3, :3], representative.pose[:3, :3]))

        translation_flag = translation_diff > self.translation_threshold
        rotation_flag = rotation_diff > self.rotation_threshold

        # Spatial-Only: ,
        if self.clustering_mode == "spatial":
            is_new = translation_flag or rotation_flag
            reasons = []
            if translation_flag:
                reasons.append("translation")
            if rotation_flag:
                reasons.append("rotation")
            return {
                "is_new_cluster": is_new,
                "reason": reasons[0] if reasons else "none",
                "clip_similarity": None,
                "clip_distance": None,
                "translation_diff": translation_diff,
                "rotation_diff": rotation_diff,
                "trigger_flags": {"clip": False, "translation": translation_flag, "rotation": rotation_flag},
                "representative_frame_idx": rep_frame_idx,
            }

        # ,,CLIP(Hybrid )
        if translation_flag or rotation_flag:
            reasons = []
            if translation_flag:
                reasons.append("translation")
            if rotation_flag:
                reasons.append("rotation")
            
            # CLIP,None
            clip_similarity = None
            d_clip = None
            clip_flag = False # ,clip_flagFalse,
            
            self.logger.info(
                "[MemoryCluster][]  %d  ( %d vs  %d): T=%.3f, R=%.3f ->  (CLIP)",
                cluster_id, frame_idx, rep_frame_idx, translation_diff, rotation_diff
            )
        else:
            # 2. , CLIP 
            
            # (clip)
            cluster_vec_norm = self._get_cluster_feature_vector(cluster)

            if cluster_vec_norm is not None:
                self.logger.info(
                    "[MemoryCluster][]  %d  ( %d vs  %d)",
                    cluster_id, frame_idx, rep_frame_idx
                )
            else:
                self.logger.warning(
                    "[MemoryCluster][]  %d , ( %d vs  %d)",
                    cluster_id, frame_idx, rep_frame_idx
                )

            # 
            clip_similarity = None
            if stats.clip_feature is not None and cluster_vec_norm is not None:
                # clip
                v_i = stats.clip_feature.copy()
                # 
                if isinstance(v_i, np.ndarray) and not np.all(np.isfinite(v_i)):
                    self.logger.warning(
                        "[MemoryCluster][]  %d clip_feature  NaN  Inf ",
                        frame_idx
                    )
                else:
                    norm_i = np.linalg.norm(v_i)
                    if norm_i > 1e-6:
                        try:
                            v_i = v_i / norm_i
                            # 
                            if isinstance(v_i, np.ndarray) and not np.all(np.isfinite(v_i)):
                                self.logger.warning(
                                    "[MemoryCluster][]  %d clip_feature  NaN  Inf ",
                                    frame_idx
                                )
                            else:
                                # :sim_i = v_i - cluster_vec_norm
                                clip_similarity = float(np.clip(np.dot(v_i, cluster_vec_norm), -1.0, 1.0))
                        except (ZeroDivisionError, ValueError, RuntimeWarning) as e:
                            self.logger.warning(
                                "[MemoryCluster][]  %d clip_feature : %s",
                                frame_idx, str(e)
                            )
                    else:
                        #  norm_i ,()
                        clip_similarity = float(np.clip(np.dot(v_i, cluster_vec_norm), -1.0, 1.0))
                self.logger.info(
                    "[MemoryCluster][] : sim=%.3f",
                    clip_similarity
                )
                print(
                    f"[MemoryCluster][] : sim={clip_similarity:.3f}"
                )
            elif stats.clip_feature is not None and representative.clip_feature is not None:
                # ,()
                clip_similarity = float(np.clip(np.dot(stats.clip_feature, representative.clip_feature), -1.0, 1.0))
                self.logger.info(
                    "[MemoryCluster][] : sim=%.3f",
                    clip_similarity
                )
                print(
                    f"[MemoryCluster][] : sim={clip_similarity:.3f}"
                )
            
            d_clip = None if clip_similarity is None else 1.0 - clip_similarity
            if clip_similarity is None and not self.candidate_require_clip:
                clip_flag = False
            else:
                clip_flag = clip_similarity is None or clip_similarity < self.clip_similarity_threshold

            reasons = []
            if clip_flag:
                reasons.append("clip")

        is_new_cluster = bool(reasons)
        result = {
            "is_new_cluster": is_new_cluster,
            "reason": reasons[0] if reasons else "none",
            "clip_similarity": clip_similarity,
            "clip_distance": d_clip,
            "translation_diff": translation_diff,
            "rotation_diff": rotation_diff,
            "trigger_flags": {
                "clip": clip_flag,
                "translation": translation_flag,
                "rotation": rotation_flag,
            },
            "representative_frame_idx": representative.frame_idx,
        }
        return result

    def _rotation_difference(self, rot_a: np.ndarray, rot_b: np.ndarray) -> float:
        try:
            r_a = R.from_matrix(rot_a[:3, :3])
            r_b = R.from_matrix(rot_b[:3, :3])
            relative = r_a.inv() * r_b
            return np.degrees(relative.magnitude())
        except Exception:
            return 0.0

    def _compute_frame_weight(self, score: float) -> float:
        """
        :w_j = s_j^alpha
        
        Args:
            score:  s_j >= 0
            
        Returns:
             w_j
        """
        if score < 0:
            return 0.0
        if self.score_alpha == 0.0:
            return 1.0  # alpha=0,
        return float(score ** self.score_alpha)

    def _get_cluster_feature_vector(self, cluster: Dict) -> Optional[np.ndarray]:
        """
        ()

        Args:
            cluster: 

        Returns:
            ,None
        """
        cluster_id = cluster.get("id", -1)
        V_sum = cluster.get("V_sum")
        S_sum = cluster.get("S_sum")
        frame_count = len(cluster.get("frames", []))

        if V_sum is None or S_sum is None or S_sum <= 0:
            self.logger.debug(
                "[MemoryCluster][]  %d : V_sum=%s, S_sum=%s, frame_count=%d",
                cluster_id, "None" if V_sum is None else "valid", "None" if S_sum is None else f"{S_sum:.6f}", frame_count
            )
            print(
                f"[MemoryCluster][]  {cluster_id} : "
                f"V_sum={'None' if V_sum is None else 'valid'}, S_sum={'None' if S_sum is None else f'{S_sum:.6f}'}, frame_count={frame_count}"
            )
            return None

        #  S_sum ,
        if S_sum < 1e-10:
            self.logger.warning(
                "[MemoryCluster][]  %d S_sum  (%.6e),",
                cluster_id, S_sum
            )
            return None

        #  V_sum 
        if isinstance(V_sum, np.ndarray):
            if not np.all(np.isfinite(V_sum)):
                self.logger.warning(
                    "[MemoryCluster][]  %d V_sum  NaN  Inf ",
                    cluster_id
                )
                return None

        # cluster_vec = V_sum / S_sum
        try:
            cluster_vec = V_sum / S_sum
            # 
            if isinstance(cluster_vec, np.ndarray) and not np.all(np.isfinite(cluster_vec)):
                self.logger.warning(
                    "[MemoryCluster][]  %d  NaN  Inf ",
                    cluster_id
                )
                return None
        except (ZeroDivisionError, ValueError, RuntimeWarning) as e:
            self.logger.warning(
                "[MemoryCluster][]  %d : %s",
                cluster_id, str(e)
            )
            return None

        # 
        norm = np.linalg.norm(cluster_vec)
        if norm > 1e-6:
            try:
                cluster_vec_norm = cluster_vec / norm
                # 
                if isinstance(cluster_vec_norm, np.ndarray) and not np.all(np.isfinite(cluster_vec_norm)):
                    self.logger.warning(
                        "[MemoryCluster][]  %d  NaN  Inf ",
                        cluster_id
                    )
                    return None
            except (ZeroDivisionError, ValueError, RuntimeWarning) as e:
                self.logger.warning(
                    "[MemoryCluster][]  %d : %s",
                    cluster_id, str(e)
                )
                cluster_vec_norm = cluster_vec
        else:
            cluster_vec_norm = cluster_vec

        self.logger.debug(
            "[MemoryCluster][]  %d : S_sum=%.6f, frame_count=%d",
            cluster_id, S_sum, frame_count
        )
        print(
            f"[MemoryCluster][]  {cluster_id} : S_sum={S_sum:.6f}, frame_count={frame_count}"
        )

        return cluster_vec_norm

    def _update_cluster_vector(self, cluster: Dict, stats: FrameStats) -> None:
        """
        :
        
        Args:
            cluster: 
            stats: 
        """
        if stats.clip_feature is None:
            return
        
        v_i = stats.clip_feature.copy()
        s_i = stats.score
        
        # 
        w_i = self._compute_frame_weight(s_i)
        
        # V_sumS_sum
        V_sum = cluster.get("V_sum")
        S_sum = cluster.get("S_sum")
        
        if V_sum is None or S_sum is None:
            # ,
            V_sum = w_i * v_i
            S_sum_old = 0.0
            S_sum_new = w_i
            self.logger.info(
                "[MemoryCluster][]  %d : S_sum_old=%.6f, w_i=%.6f, S_sum_new=%.6f ( %d, score=%.3f)",
                cluster.get("id", -1), S_sum_old, w_i, S_sum_new, stats.frame_idx, s_i
            )
            print(
                f"[MemoryCluster][]  {cluster.get('id', -1)} : "
                f"S_sum_old={S_sum_old:.6f}, w_i={w_i:.6f}, S_sum_new={S_sum_new:.6f} "
                f"( {stats.frame_idx}, score={s_i:.3f})"
            )
        else:
            # 
            S_sum_old = float(S_sum)
            S_sum_new = S_sum_old + w_i
            V_sum = V_sum + w_i * v_i
            S_sum = S_sum_new
            self.logger.info(
                "[MemoryCluster][]  %d : S_sum_old=%.6f, w_i=%.6f, S_sum_new=%.6f ( %d, score=%.3f)",
                cluster.get("id", -1), S_sum_old, w_i, S_sum_new, stats.frame_idx, s_i
            )
            print(
                f"[MemoryCluster][]  {cluster.get('id', -1)} : "
                f"S_sum_old={S_sum_old:.6f}, w_i={w_i:.6f}, S_sum_new={S_sum_new:.6f} "
                f"( {stats.frame_idx}, score={s_i:.3f})"
            )
        
        # 
        cluster["V_sum"] = V_sum
        cluster["S_sum"] = S_sum
        
        # ()
        #  S_sum ,
        if S_sum < 1e-10:
            self.logger.warning(
                "[MemoryCluster][]  %d S_sum  (%.6e),",
                cluster.get("id", -1), S_sum
            )
            cluster["cluster_vec_norm"] = None
            return
        
        #  V_sum 
        if isinstance(V_sum, np.ndarray) and not np.all(np.isfinite(V_sum)):
            self.logger.warning(
                "[MemoryCluster][]  %d V_sum  NaN  Inf ,",
                cluster.get("id", -1)
            )
            cluster["cluster_vec_norm"] = None
            return
        
        try:
            cluster_vec = V_sum / S_sum
            # 
            if isinstance(cluster_vec, np.ndarray) and not np.all(np.isfinite(cluster_vec)):
                self.logger.warning(
                    "[MemoryCluster][]  %d  NaN  Inf ",
                    cluster.get("id", -1)
                )
                cluster["cluster_vec_norm"] = None
                return
        except (ZeroDivisionError, ValueError, RuntimeWarning) as e:
            self.logger.warning(
                "[MemoryCluster][]  %d : %s",
                cluster.get("id", -1), str(e)
            )
            cluster["cluster_vec_norm"] = None
            return
        
        norm = np.linalg.norm(cluster_vec)
        if norm > 1e-6:
            try:
                cluster_vec_norm = cluster_vec / norm
                # 
                if isinstance(cluster_vec_norm, np.ndarray) and not np.all(np.isfinite(cluster_vec_norm)):
                    self.logger.warning(
                        "[MemoryCluster][]  %d  NaN  Inf ",
                        cluster.get("id", -1)
                    )
                    cluster["cluster_vec_norm"] = None
                    return
                cluster["cluster_vec_norm"] = cluster_vec_norm
            except (ZeroDivisionError, ValueError, RuntimeWarning) as e:
                self.logger.warning(
                    "[MemoryCluster][]  %d : %s",
                    cluster.get("id", -1), str(e)
                )
                cluster["cluster_vec_norm"] = cluster_vec
        else:
            cluster["cluster_vec_norm"] = cluster_vec

    def _save_keyframe(self, cluster: Dict, stats: FrameStats, is_new_cluster: bool) -> None:
        if stats.image is None:
            return

        if not self._keyframe_dir_ready:
            self._reset_keyframe_dir()
        else:
            os.makedirs(self.keyframe_dir, exist_ok=True)

        # 
        if self.complex_score_fn is not None:
            complex_score = self.complex_score_fn(stats.frame_idx)
        else:
            complex_score = getattr(stats, "score", 0.0)

        # supervision(,mask)
        # try:
        #     from utils.visualizer import visualize_result_rgb_bbox_only
            
        #     # valid_observationssv.Detections
        #     if stats.valid_observations and len(stats.valid_observations) > 0:
        #         xyxy_list = []
        #         confidence_list = []
        #         class_id_list = []
                
        #         for obs in stats.valid_observations:
        #             xyxy = getattr(obs, "xyxy", None)
        #             if xyxy is not None and len(xyxy) > 0:
        #                 xyxy_list.append(xyxy)
        #                 confidence_list.append(getattr(obs, "conf", 0.0))
        #                 class_id = getattr(obs, "class_id", -1)
        #                 # class_id
        #                 if class_id < 0 or (self.class_lookup and class_id >= len(self.class_lookup)):
        #                     class_id = 0  # 
        #                 class_id_list.append(class_id)
                
        #         if len(xyxy_list) > 0:
        #             # sv.Detections
        #             detections = sv.Detections(
        #                 xyxy=np.array(xyxy_list, dtype=np.float32),
        #                 confidence=np.array(confidence_list, dtype=np.float32),
        #                 class_id=np.array(class_id_list, dtype=np.int64),
        #                 mask=None,  # mask
        #             )
                    
        #             # ()
        #             # class_lookup
        #             class_lookup = self.class_lookup if self.class_lookup else ["unknown"]
        #             annotated_image, _ = visualize_result_rgb_bbox_only(
        #                 stats.image.copy(),
        #                 detections,
        #                 class_lookup,
        #             )
                    
        #             # BGR
        #             annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        #         else:
        #             # ,
        #             annotated_bgr = cv2.cvtColor(stats.image, cv2.COLOR_RGB2BGR)
        #     else:
        #         # observations,
        #         annotated_bgr = cv2.cvtColor(stats.image, cv2.COLOR_RGB2BGR)
        # except Exception as e:
        #     # ,
        #     self.logger.warning(f"[MemoryCluster] supervision,: {e}")
        #     extra_lines = [
        #         f"Complex Frame Score: {complex_score:.3f}",
        #         f"Anchor Richness: {getattr(stats, 'anchor_richness', 0.0):.3f}",
        #         f"Object Richness: {getattr(stats, 'object_richness', 0.0):.3f}",
        #     ]
        #     annotated = self._annotate_image(cluster["id"], stats, extra_lines=extra_lines)
        #     annotated_bgr = annotated
        
        # ,
        annotated_bgr = cv2.cvtColor(stats.image, cv2.COLOR_RGB2BGR)

        # 
        filename = f"cluster_{cluster['id']:03d}_frame_{stats.frame_idx:06d}_score_{complex_score:.3f}.png"

        save_path = os.path.join(self.keyframe_dir, filename)

        prev_path = cluster.get("saved_path")
        if prev_path and os.path.exists(prev_path):
            try:
                os.remove(prev_path)
            except OSError:
                pass

        cv2.imwrite(save_path, annotated_bgr)
        cluster["saved_path"] = save_path
        
        # ,
        if stats.metadata.get("has_new_instance"):
            new_instance_dir = os.path.join(os.path.dirname(self.keyframe_dir), "new_instances")
            os.makedirs(new_instance_dir, exist_ok=True)
            new_instance_save_path = os.path.join(new_instance_dir, filename)
            cv2.imwrite(new_instance_save_path, annotated_bgr)
            self.logger.info(f"[MemoryCluster] ,: {new_instance_save_path}")

        # 
        info_filename = f"cluster_{cluster['id']:03d}_frame_{stats.frame_idx:06d}_score_{complex_score:.3f}.pkl"
        info_path = os.path.join(self.keyframe_dir, info_filename)

        save_data = {
            "cluster_id": cluster["id"],
            "frame_idx": stats.frame_idx,
            "score": complex_score,
            "pose": stats.pose,
            "clip_feature": stats.clip_feature,
            "V_sum": cluster.get("V_sum"),
            "S_sum": cluster.get("S_sum"),
            "cluster_vec_norm": cluster.get("cluster_vec_norm"),
            "anchor_count": getattr(stats, "anchor_count", 0),
            "object_count": getattr(stats, "object_count", 0),
            "metadata": getattr(stats, "metadata", {}),
        }

        prev_info_path = cluster.get("saved_info_path")
        if prev_info_path and os.path.exists(prev_info_path):
            try:
                os.remove(prev_info_path)
            except OSError:
                pass

        try:
            with open(info_path, "wb") as f:
                pickle.dump(save_data, f)
            cluster["saved_info_path"] = info_path
        except Exception as e:
            self.logger.warning(f"[MemoryCluster] : {e}")

        tag = "" if is_new_cluster else ""
        self.logger.info(
            "[MemoryCluster] %s : %s (: %d, : %.6f)",
            tag, save_path, stats.frame_idx, complex_score
        )
        # 
        self.logger.debug(
            "[MemoryCluster][]  %d:  %d,  %.6f,  %d,  %d, : %s",
            cluster.get("id", -1), stats.frame_idx, complex_score,
            getattr(stats, "anchor_count", 0), getattr(stats, "object_count", 0), save_path
        )

    def _save_micro_keyframe(self, cluster: Dict, stats: FrameStats, is_new_cluster: bool) -> None:
        if stats.image is None:
            return

        if not self._keyframe_dir_ready:
            self._reset_keyframe_dir()
        else:
            os.makedirs(self.keyframe_dir, exist_ok=True)

        best_detail = None
        if stats.micro_details:
            best_detail = max(
                stats.micro_details, key=lambda d: d.get("score", 0.0)
            )
        total_pixels = stats.metadata.get("total_pixels") if stats.metadata else None
        extra_lines = [f"Micro Score: {stats.micro_score:.3f}"]
        if best_detail:
            uid = best_detail.get("anchor_uid")
            cls = best_detail.get("class_name") or "unknown"
            area_anchor = best_detail.get("area_anchor", 0.0)
            norm_anchor = (area_anchor / total_pixels) if total_pixels else 0.0
            contact_count = best_detail.get("contact_count", 0)
            if uid:
                extra_lines.append(f"Anchor UID: {str(uid)[:8]}... ({cls})")
            else:
                extra_lines.append(f"Anchor Class: {cls}")
            extra_lines.append(
                f"A={norm_anchor*100:.1f}% Contact={contact_count}"
            )

        # annotated = self._annotate_image(cluster["id"], stats, extra_lines=extra_lines)
        # ,
        annotated = cv2.cvtColor(stats.image, cv2.COLOR_RGB2BGR)
        # ,
        filename = f"cluster_{cluster['id']:03d}_frame_{stats.frame_idx:06d}_score_{stats.micro_score:.3f}.png"
        save_path = os.path.join(self.keyframe_dir, filename)

        prev_path = cluster.get("saved_path")
        if prev_path and os.path.exists(prev_path):
            try:
                os.remove(prev_path)
            except OSError:
                pass

        cv2.imwrite(save_path, annotated)
        cluster["saved_path"] = save_path
        
        # ,
        if stats.metadata.get("has_new_instance"):
            new_instance_dir = os.path.join(os.path.dirname(self.keyframe_dir), "new_instances")
            os.makedirs(new_instance_dir, exist_ok=True)
            new_instance_save_path = os.path.join(new_instance_dir, filename)
            cv2.imwrite(new_instance_save_path, annotated)
            self.logger.info(f"[MemoryCluster] (Micro),: {new_instance_save_path}")

        # 
        info_filename = f"cluster_{cluster['id']:03d}_frame_{stats.frame_idx:06d}_score_{stats.micro_score:.3f}.pkl"
        info_path = os.path.join(self.keyframe_dir, info_filename)

        save_data = {
            "cluster_id": cluster["id"],
            "frame_idx": stats.frame_idx,
            "score": stats.micro_score,
            "pose": stats.pose,
            "clip_feature": stats.clip_feature,
            "V_sum": cluster.get("V_sum"),
            "S_sum": cluster.get("S_sum"),
            "cluster_vec_norm": cluster.get("cluster_vec_norm"),
            "anchor_count": getattr(stats, "anchor_count", 0),
            "object_count": getattr(stats, "object_count", 0),
            "metadata": getattr(stats, "metadata", {}),
            "is_micro": True
        }

        prev_info_path = cluster.get("saved_info_path")
        if prev_info_path and os.path.exists(prev_info_path):
            try:
                os.remove(prev_info_path)
            except OSError:
                pass

        try:
            with open(info_path, "wb") as f:
                pickle.dump(save_data, f)
            cluster["saved_info_path"] = info_path
        except Exception as e:
            self.logger.warning(f"[MemoryCluster] : {e}")

        tag = "" if is_new_cluster else ""
        self.logger.info("[MemoryCluster] %s: %s", tag, save_path)

    def _annotate_image(
        self, cluster_id: int, stats: FrameStats, extra_lines: Optional[List[str]] = None
    ) -> np.ndarray:
        image = stats.image
        if image is None:
            return None
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        overlay = bgr.copy()
        
        # mask
        height, width = overlay.shape[:2]
        label_info_list = []  # ,

        for obs in stats.valid_observations:
            # xyxy
            xyxy = getattr(obs, "xyxy", None)
            if xyxy is None:
                continue

            # ,
            class_name = self._get_class_name(obs)
            class_id = getattr(obs, "class_id", -1)

            # FastSAM
            is_fastsam_detection = False
            if hasattr(self, 'detector') and hasattr(self.detector, 'unknown_class_id'):
                is_fastsam_detection = (class_id == self.detector.unknown_class_id)
            elif not class_name and class_id >= 0:
                # class_nameNoneclass_id,FastSAM
                is_fastsam_detection = True

            if not class_name:
                # FastSAM,
                class_name = f"fastsam_{class_id}" if class_id >= 0 else "unknown"

            # :
            self.logger.warning(f"[DEBUG] Obs class_name: '{class_name}', class_id: {class_id}, is_fastsam: {is_fastsam_detection}, conf: {getattr(obs, 'conf', 'N/A')}")

            arr = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)[0]
            x1, y1, x2, y2 = int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])

            # 
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            # :
            # - 
            # - FastSAM
            # - YOLO
            if self._is_anchor_class_name(class_name):
                color = (0, 255, 0)  #  - 
            elif is_fastsam_detection or class_name.startswith("fastsam_") or class_name == "unknown":
                color = (255, 0, 255)  #  - FastSAM
            else:
                color = (0, 0, 255)  #  - YOLO

            # 
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # 
            label = f"{class_name}"
            conf = getattr(obs, "conf", 0.0)
            if conf > 0:
                label += f" {conf:.2f}"

            # 
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_x = x1
            label_y = y1 - 5

            # ,
            if label_y - text_height < 0:
                label_y = y1 + text_height + 5

            # 
            cv2.rectangle(overlay, (label_x, label_y - text_height - 2),
                         (label_x + text_width, label_y + 2), color, -1)

            # 
            cv2.putText(overlay, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 
        text_lines = [
            f"Cluster {cluster_id} Representative",
            f"Frame {stats.frame_idx}  Score {stats.score:.3f}",
            f"Foreground Ratio: {stats.foreground_ratio * 100:.1f}%",
            f"AnchorRich {stats.anchor_richness:.2f}  ObjectRich {stats.object_richness:.2f}",
            f"Anchors Σconf={stats.anchor_conf_sum:.2f} (count={stats.anchor_count})  Objects {stats.object_count}",
            f"Anchor Names: {', '.join(stats.anchor_names) if stats.anchor_names else 'none'}",
        ]
        if extra_lines:
            text_lines.extend(extra_lines)

        for idx, text in enumerate(text_lines):
            y = 30 + idx * 28
            cv2.putText(
                overlay,
                text,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return overlay

    def _build_result(
        self,
        cluster: Dict,
        stats: FrameStats,
        is_new_cluster: bool,
        novelty_reason: str,
        extra: Optional[Dict] = None,
    ) -> Dict:
        result = {
            "cluster_id": cluster["id"],
            "frame_idx": stats.frame_idx,
            "score": stats.score,
            "anchor_richness": stats.anchor_richness,
            "object_richness": stats.object_richness,
            "anchor_conf_sum": stats.anchor_conf_sum,
            "anchor_count": stats.anchor_count,
            "object_count": stats.object_count,
            "is_new_cluster": is_new_cluster,
            "novelty_reason": novelty_reason,
        }
        if extra:
            result.update(
                {
                    "clip_similarity": extra.get("clip_similarity"),
                    "clip_distance": extra.get("clip_distance"),
                    "translation_diff": extra.get("translation_diff"),
                    "rotation_diff": extra.get("rotation_diff"),
                    "representative_frame_idx": extra.get("representative_frame_idx"),
                    "trigger_flags": extra.get("trigger_flags"),
                }
            )
        return result

    def _log_novelty_info(self, frame_idx: int, cluster_id: int, info: Dict) -> None:
        clip_similarity = info.get("clip_similarity")
        clip_distance = info.get("clip_distance")
        translation_diff = info.get("translation_diff")
        rotation_diff = info.get("rotation_diff")
        rep_idx = info.get("representative_frame_idx", "N/A")
        msg = (
            f"[MemoryCluster][]  {frame_idx} VS  {rep_idx} ( {cluster_id}): "
            f"S_clip={f'{clip_similarity:.3f}' if clip_similarity is not None else 'N/A'}, "
            f"D_clip={f'{clip_distance:.3f}' if clip_distance is not None else 'N/A'}, "
            f"d_trans={translation_diff:.3f} m, d_rot={rotation_diff:.2f}°, "
            f"={'' if info.get('is_new_cluster') else ''}"
        )
        self.logger.info(msg)
        print(msg)
        if info.get("is_new_cluster"):
            detail_msg = (
                f"[MemoryCluster][]  {frame_idx}  {rep_idx} -> "
                f"D_clip={f'{clip_distance:.3f}' if clip_distance is not None else 'N/A'}, "
                f"d_trans={translation_diff:.3f} m, d_rot={rotation_diff:.2f}°"
            )
            self.logger.info(detail_msg)
            print(detail_msg)

    def _log_representative_anchors(self, cluster_id: int, stats: FrameStats, tag: str) -> None:
        if not stats:
            return
        if stats.anchor_names:
            name_str = ",".join(stats.anchor_names)
        else:
            name_str = ""
        base_msg = f"[MemoryCluster][{tag}]  {cluster_id}  {stats.frame_idx} : {name_str}"
        self.logger.info(base_msg)
        print(base_msg)

        if stats.anchor_uid_map:
            detail_lines = ["  :"]
            for uid, info in stats.anchor_uid_map.items():
                cls = info.get("class_name", "unknown")
                count = info.get("count", 0)
                detail_lines.append(f"    - UID {uid[:8]}... | {cls} |  {count} ")
            detail_msg = "\n".join(detail_lines)
            self.logger.info(detail_msg)
            print(detail_msg)

    def _update_trigger_counters(self, info: Dict) -> None:
        flags = info.get("trigger_flags") or {}
        updated = False
        for key in ("clip", "translation", "rotation"):
            if flags.get(key):
                self.trigger_counters[key] += 1
                updated = True
        if updated:
            self._log_trigger_counters()

    def _log_trigger_counters(self) -> None:
        msg = (
            "[MemoryCluster][]  - "
            f"CLIP:{self.trigger_counters['clip']} "
            f"Trans:{self.trigger_counters['translation']} "
            f"Rot:{self.trigger_counters['rotation']}"
        )
        self.logger.info(msg)
        print(msg)

    def _log_anchor_debug(self, frame_idx: int, objects: List[dict]) -> None:
        if not objects:
            msg = f"[MemoryCluster][]  {frame_idx} ,."
        else:
            formatted = ", ".join(
                f"{item['class']}({item['conf']:.2f}){'*' if item['is_anchor'] else ''}"
                for item in objects
            )
            msg = (
                f"[MemoryCluster][]  {frame_idx} ."
                f" : {formatted}"
            )
        self.logger.info(msg)
        print(msg)

    def log_anchor_instances_summary(self, local_map_manager=None, dataset=None, detector=None) -> None:
        ""","""
        representative_infos = self._collect_representative_pose_info()

        anchor_geometries: List[Dict] = []
        if local_map_manager is not None:
            anchor_geometries, _ = self._collect_stable_anchor_geometries(local_map_manager)

        summary_data: Dict = {}
        if local_map_manager is not None:
            summary_data = self._log_anchor_comparison(local_map_manager)
        else:
            self._log_representative_frame_anchor_instances()

        self._render_topdown_overview(
            local_map_manager,
            representative_infos=representative_infos,
            anchor_geometries=anchor_geometries,
            anchor_rep_frames=summary_data.get("anchor_rep_frames") if summary_data else None,
        )

        if local_map_manager is not None:
            self._export_anchor_key_images(
                anchor_geometries=anchor_geometries,
                summary_data=summary_data,
                local_map_manager=local_map_manager,
                dataset=dataset,
                detector=detector,
            )
            
            # [LifelongNav] , anchor_key  _nav 
            #  self.keyframe_dir  Core  _nav ()
            # _export_anchor_key_images  self.keyframe_dir/anchor_key 
            #  _export_anchor_key_images,
            # () log_anchor_instances_summary 

    def _log_anchor_comparison(self, local_map_manager) -> Dict[str, any]:
        ""","""
        if not hasattr(local_map_manager, 'local_map'):
            msg = "[MemoryCluster][] ,"
            self.logger.info(msg)
            print(msg)
            return {}

        local_map = local_map_manager.local_map
        if not local_map:
            msg = "[MemoryCluster][] ,"
            self.logger.info(msg)
            print(msg)
            return

        # 
        class_lookup = []
        if hasattr(local_map_manager, 'visualizer') and hasattr(local_map_manager.visualizer, 'obj_classes'):
            try:
                class_lookup = local_map_manager.visualizer.obj_classes.get_classes_arr()
            except Exception:
                pass

        # ==========  ==========
        stable_anchor_objects: List[Dict] = []
        stable_anchor_class_counts: Dict[str, int] = {}
        stable_anchor_uids: Set[str] = set()  #  UID ,
        frame_to_anchor_uids: Dict[int, Set[str]] = {}

        for obj in local_map:
            # 
            if not getattr(obj, "is_stable", False):
                continue

            # 
            class_id = getattr(obj, "m_class_id", None)
            if class_id is None:
                class_id = getattr(obj, "class_id", None)
            
            class_name = None
            if class_id is not None and 0 <= int(class_id) < len(class_lookup):
                class_name = class_lookup[int(class_id)]
            elif class_id is not None and 0 <= int(class_id) < len(self.class_lookup):
                class_name = self.class_lookup[int(class_id)]

            if not class_name or not self._is_anchor_class_name(class_name):
                continue

            # 
            obj_uid = str(getattr(obj, "uid", "unknown"))
            observed_num = getattr(obj, "observed_num", 0)
            frame_indices = self._extract_object_frame_indices(obj)
            stable_anchor_objects.append({
                "uid": obj_uid,
                "class_name": class_name,
                "observed_num": observed_num,
                "frame_indices": frame_indices,
            })
            stable_anchor_class_counts[class_name] = stable_anchor_class_counts.get(class_name, 0) + 1
            stable_anchor_uids.add(obj_uid)

            for frame_idx in frame_indices:
                if frame_idx is None:
                    continue
                frame_to_anchor_uids.setdefault(int(frame_idx), set()).add(obj_uid)

        stable_anchor_lookup = {item["uid"]: item for item in stable_anchor_objects}

        # ==========  ==========
        representative_anchor_class_counts: Dict[str, int] = {}
        cluster_anchor_info: List[Dict] = []
        representative_anchor_instances = 0
        anchor_rep_frames: Dict[str, List[List[float]]] = {}

        # ,
        all_rep_frames: Dict[str, List[List[float]]] = {}

        def _record_frame_score(
            container: Dict[str, List[List[float]]],
            uid: Optional[str],
            frame_idx: int,
            score: float,
        ) -> None:
            if not uid:
                return
            entries = container.setdefault(uid, [])
            updated = False
            for entry in entries:
                if entry[0] == frame_idx:
                    if score > entry[1]:
                        entry[1] = score
                    updated = True
                    break
            if not updated:
                entries.append([frame_idx, score])
            entries.sort(key=lambda x: x[1], reverse=True)
            # ,

        if self.clusters:
            for cluster in self.clusters:
                rep: FrameStats = cluster.get("representative")
                cluster_id = cluster.get("id", -1)
                anchor_classes_in_cluster: Dict[str, int] = {}

                if rep is not None:
                    frame_idx = rep.frame_idx
                    # 
                    unified_score_value = float(getattr(rep, "score", 0.0))

                    # 
                    
                    # 1.  frame_to_anchor_uids ( matched_obj_uid)
                    # , FrameStats  matched_obj_uid
                    anchors_in_frame = frame_to_anchor_uids.get(frame_idx, set())
                    for uid_str in anchors_in_frame:
                         _record_frame_score(
                            all_rep_frames,
                            uid_str,
                            frame_idx,
                            unified_score_value,
                        )
                         _record_frame_score(
                            anchor_rep_frames,
                            uid_str,
                            frame_idx,
                            unified_score_value,
                        )

                    # 2. ,
                    for obs in rep.valid_observations:
                        matched_uid = getattr(obs, "matched_obj_uid", None)
                        if matched_uid is not None:
                            uid_str = str(matched_uid)
                            #  matched_obj_uid , anchors_in_frame (),
                            if uid_str not in anchors_in_frame:
                                _record_frame_score(
                                    all_rep_frames,
                                    uid_str,
                                    frame_idx,
                                    unified_score_value,
                                )
                                if uid_str in stable_anchor_uids:
                                    _record_frame_score(
                                        anchor_rep_frames,
                                        uid_str,
                                        frame_idx,
                                        unified_score_value,
                                    )

                        # ()
                        class_name = self._get_class_name(obs)
                        if class_name and self._is_anchor_class_name(class_name):
                            representative_anchor_instances += 1
                            anchor_classes_in_cluster[class_name] = anchor_classes_in_cluster.get(class_name, 0) + 1
                            representative_anchor_class_counts[class_name] = representative_anchor_class_counts.get(class_name, 0) + 1

                if anchor_classes_in_cluster and rep is not None:
                    cluster_anchor_info.append({
                        "cluster_id": cluster_id,
                        "frame_idx": rep.frame_idx,
                        "anchor_classes": anchor_classes_in_cluster,
                    })

        def _filter_frames(frames_dict: Dict[str, List[List[float]]]) -> Dict[str, List[int]]:
            filtered: Dict[str, List[int]] = {}
            for uid, frames in frames_dict.items():
                sorted_frames = sorted(frames, key=lambda x: x[1], reverse=True)
                observed = set(stable_anchor_lookup.get(uid, {}).get("frame_indices") or [])
                valid: List[int] = []
                if observed:
                    # ,5
                    for frame_idx, _score in sorted_frames:
                        if frame_idx in observed:
                            valid.append(frame_idx)
                            if len(valid) >= 5:
                                break
                if not valid:
                    # ,(5)
                    valid = [frame_idx for frame_idx, _score in sorted_frames[:5]]
                filtered[uid] = valid
            return filtered

        filtered_anchor_frames = _filter_frames(anchor_rep_frames)
        filtered_all_frames = _filter_frames(all_rep_frames)

        # ==========  ==========
        all_anchor_classes = set(stable_anchor_class_counts.keys()) | set(representative_anchor_class_counts.keys())
        
        msg_lines = [
            "[MemoryCluster][] ==========  ==========",
            "",
            "[]",
            f": {len(stable_anchor_objects)}",
        ]

        if stable_anchor_class_counts:
            msg_lines.append(":")
            for class_name in sorted(stable_anchor_class_counts.keys()):
                count = stable_anchor_class_counts[class_name]
                msg_lines.append(f"  {class_name}: {count} ")

        msg_lines.extend([
            "",
            "[]",
            f": {len(self.clusters)}",
            f"(): {representative_anchor_instances}",
        ])

        if representative_anchor_class_counts:
            msg_lines.append(":")
            for class_name in sorted(representative_anchor_class_counts.keys()):
                count = representative_anchor_class_counts[class_name]
                msg_lines.append(f"  {class_name}: {count} ")

        # ==========  ==========
        msg_lines.extend([
            "",
            "[]",
        ])

        for class_name in sorted(all_anchor_classes):
            stable_count = stable_anchor_class_counts.get(class_name, 0)
            representative_count = representative_anchor_class_counts.get(class_name, 0)
            
            if stable_count > 0 and representative_count > 0:
                coverage = "✓ " if representative_count >= stable_count else f"⚠  ({representative_count}/{stable_count})"
                msg_lines.append(f"  {class_name}:  {stable_count} ,  {representative_count}  - {coverage}")
            elif stable_count > 0:
                msg_lines.append(f"  {class_name}:  {stable_count} ,  0  - ✗ ")
            else:
                msg_lines.append(f"  {class_name}:  0 ,  {representative_count}  - ⚠ ")

        # ==========  ==========
        if stable_anchor_objects:
            msg_lines.extend([
                "",
                "[]",
            ])
            for info in stable_anchor_objects:
                uid = info['uid']
                frame_list = self._format_frame_indices(info.get("frame_indices", []))
                rep_frames = filtered_anchor_frames.get(uid, [])
                rep_list = self._format_frame_indices(rep_frames)
                has_rep = bool(rep_frames)
                coverage_flag = "✓" if has_rep else "✗"
                msg_lines.append(
                    f"  {info['class_name']} (UID: {uid[:8]}..., : {info['observed_num']}): "
                    f"[{frame_list}] | [{rep_list}] {coverage_flag}"
                )

        if cluster_anchor_info:
            msg_lines.extend([
                "",
                "[]",
            ])
            for info in cluster_anchor_info:
                classes_str = ", ".join(f"{k}({v})" for k, v in info["anchor_classes"].items())
                msg_lines.append(
                    f"   {info['cluster_id']:03d} ( {info['frame_idx']:06d}): {classes_str}"
                )

        msg_lines.append("=" * 50)

        full_msg = "\n".join(msg_lines)
        self.logger.info(full_msg)
        print(full_msg)

        if local_map_manager is not None:
            self._tag_local_objects_representative_frames(
                local_map_manager,
                filtered_all_frames,
            )

        return {
            "anchor_rep_frames": filtered_anchor_frames,
            "all_rep_frames": filtered_all_frames,
            "stable_anchor_objects": stable_anchor_objects,
        }

    def _log_representative_frame_anchor_instances(self) -> None:
        """()"""
        if not self.clusters:
            msg = "[MemoryCluster][] ,"
            self.logger.info(msg)
            print(msg)
            return

        total_anchor_instances = 0
        anchor_class_counts: Dict[str, int] = {}
        cluster_anchor_info: List[Dict] = []

        for cluster in self.clusters:
            rep: FrameStats = cluster.get("representative")
            if rep is None:
                continue

            cluster_id = cluster.get("id", -1)
            frame_idx = rep.frame_idx
            anchor_count = 0
            anchor_classes_in_cluster: Dict[str, int] = {}

            # 
            for obs in rep.valid_observations:
                class_name = self._get_class_name(obs)
                if class_name and self._is_anchor_class_name(class_name):
                    anchor_count += 1
                    total_anchor_instances += 1
                    # 
                    anchor_classes_in_cluster[class_name] = anchor_classes_in_cluster.get(class_name, 0) + 1
                    anchor_class_counts[class_name] = anchor_class_counts.get(class_name, 0) + 1

            cluster_anchor_info.append({
                "cluster_id": cluster_id,
                "frame_idx": frame_idx,
                "anchor_count": anchor_count,
                "anchor_classes": anchor_classes_in_cluster,
            })

        # 
        msg_lines = [
            "[MemoryCluster][] ==========  ==========",
            f": {len(self.clusters)}",
            f": {total_anchor_instances}",
        ]

        if anchor_class_counts:
            msg_lines.append(":")
            for class_name, count in sorted(anchor_class_counts.items(), key=lambda x: x[1], reverse=True):
                msg_lines.append(f"  {class_name}: {count} ")

        msg_lines.append(":")
        for info in cluster_anchor_info:
            if info["anchor_count"] > 0:
                classes_str = ", ".join(f"{k}({v})" for k, v in info["anchor_classes"].items())
                msg_lines.append(
                    f"   {info['cluster_id']:03d} ( {info['frame_idx']:06d}): "
                    f"{info['anchor_count']}  - {classes_str}"
                )
            else:
                msg_lines.append(
                    f"   {info['cluster_id']:03d} ( {info['frame_idx']:06d}): "
                )

        msg_lines.append("=" * 50)

        full_msg = "\n".join(msg_lines)
        self.logger.info(full_msg)
        print(full_msg)

    def _render_topdown_overview(
        self,
        local_map_manager=None,
        representative_infos: Optional[List[Dict]] = None,
        anchor_geometries: Optional[List[Dict]] = None,
        anchor_rep_frames: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        path_points = self._collect_path_points()
        if representative_infos is None:
            representative_infos = self._collect_representative_pose_info()
        if anchor_geometries is None:
            anchor_geometries, _ = self._collect_stable_anchor_geometries(local_map_manager)

        if not (path_points or representative_infos or anchor_geometries):
            msg = "[MemoryCluster][] ,."
            self.logger.info(msg)
            print(msg)
            return

        os.makedirs(self.keyframe_dir, exist_ok=True)
        path_csv = os.path.join(self.keyframe_dir, "frame_path.csv")
        self._log_frame_path_summary(path_points, path_csv)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Top-Down Overview", fontsize=14)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, linestyle="--", alpha=0.3)

        plotted = False

        # 
        if path_points:
            coords = [self._project_to_ground(item["pos"]) for item in path_points]
            xs = [float(c[0]) for c in coords]
            ys = [float(c[1]) for c in coords]
            ax.plot(xs, ys, color="#888888", linewidth=1.5, label="")
            ax.scatter(xs[0], ys[0], color="#1f77b4", marker="o", s=40, label="")
            ax.scatter(xs[-1], ys[-1], color="#d62728", marker="s", s=40, label="")
            plotted = True

        # 
        if anchor_geometries:
            class_names = sorted({item["class_name"] for item in anchor_geometries})
            palette = plt.cm.get_cmap("tab20", max(10, len(class_names) or 1))
            color_map = {}
            for idx, cls in enumerate(class_names):
                color_map[cls] = palette(idx % palette.N)
            rendered_labels = set()

            for anchor in anchor_geometries:
                cls = anchor["class_name"]
                center = anchor["center"]
                corners = anchor["corners_xy"]
                color = color_map.get(cls, (0.2, 0.7, 0.2, 1.0))
                label = None
                if cls not in rendered_labels:
                    label = f"Anchor: {cls}"
                    rendered_labels.add(cls)
                poly = Polygon(
                    corners,
                    closed=True,
                    facecolor=(*color[:3], 0.25 if len(color) == 4 else 0.25),
                    edgecolor=color,
                    linewidth=2.0,
                    label=label,
                    zorder=2,
                )
                ax.add_patch(poly)
                ax.text(
                    float(center[0]),
                    float(center[1]),
                    cls,
                    fontsize=8,
                    color=color,
                    ha="center",
                    va="center",
                    zorder=4,
                )
            plotted = True

        # 
        if representative_infos:
            arrow_len = 0.4
            fov_half = np.deg2rad(self.topdown_fov_deg) / 2.0
            cos_half = np.cos(fov_half)
            view_range = self.topdown_view_range

            for info in representative_infos:
                pos = info.get("pos2d")
                if pos is None:
                    pos = self._project_to_ground(info.get("pos"))
                heading = info["heading"]
                dx = float(heading[0]) * arrow_len
                dy = float(heading[1]) * arrow_len
                ax.quiver(
                    float(pos[0]),
                    float(pos[1]),
                    dx,
                    dy,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color="#ff7f0e",
                    width=0.003,
                )
                # 
                left_dir = self._rotate_2d(heading, fov_half)
                right_dir = self._rotate_2d(heading, -fov_half)
                cone_pts = np.array(
                    [
                        [float(pos[0]), float(pos[1])],
                        [
                            float(pos[0] + left_dir[0] * view_range),
                            float(pos[1] + left_dir[1] * view_range),
                        ],
                        [
                            float(pos[0] + right_dir[0] * view_range),
                            float(pos[1] + right_dir[1] * view_range),
                        ],
                    ]
                )
                cone = Polygon(
                    cone_pts,
                    closed=True,
                    facecolor=(1.0, 0.5, 0.0, 0.15),
                    edgecolor=(1.0, 0.5, 0.0, 0.6),
                    linewidth=1.0,
                    zorder=1,
                )
                ax.add_patch(cone)

                seen_classes = info.get("anchor_classes", set())
                anchor_uid_map = info.get("anchor_uids")
                if anchor_geometries:
                    for anchor in anchor_geometries:
                        cls = anchor["class_name"]
                        anchor_uid = anchor.get("uid")

                        should_link = False
                        if anchor_uid_map:
                            if anchor_uid and anchor_uid in anchor_uid_map:
                                should_link = True
                        elif cls in seen_classes:
                            should_link = True

                        allowed_frames = None
                        if anchor_rep_frames and anchor_uid:
                            frames = anchor_rep_frames.get(anchor_uid)
                            if frames:
                                allowed_frames = set(frames)
                        if allowed_frames and info["frame_idx"] not in allowed_frames:
                            should_link = False

                        if not should_link:
                            continue

                        center = anchor["center"]
                        vector = np.array(
                            [
                                float(center[0] - pos[0]),
                                float(center[1] - pos[1]),
                            ]
                        )
                        dist = np.linalg.norm(vector)
                        if dist < 1e-6 or dist > view_range:
                            continue
                        vector_norm = vector / dist
                        if float(np.dot(vector_norm, heading)) >= cos_half:
                            ax.plot(
                                [float(pos[0]), float(center[0])],
                                [float(pos[1]), float(center[1])],
                                color="#ff7f0e",
                                linewidth=1.0,
                                linestyle="--",
                                zorder=4,
                            )
                            ax.scatter(
                                float(center[0]),
                                float(center[1]),
                                color="#ff7f0e",
                                s=18,
                                zorder=5,
                            )

                ax.text(
                    float(pos[0]),
                    float(pos[1]),
                    f"C{info['cluster_id']}",
                    fontsize=8,
                    color="#ff7f0e",
                    ha="center",
                    va="bottom",
                )
            plotted = True

        if not plotted:
            plt.close(fig)
            msg = "[MemoryCluster][] ,."
            self.logger.info(msg)
            print(msg)
            return

        # 
        all_x = []
        all_y = []
        for item in path_points:
            pos = item.get("pos")
            #  _project_to_ground 
            if pos is not None:
                pos_arr = np.asarray(pos, dtype=np.float32)
                if np.all(np.isfinite(pos_arr)):
                    pos2d = self._project_to_ground(pos)
                    x_val = float(pos2d[0])
                    y_val = float(pos2d[1])
                    #  NaN  Inf 
                    if np.isfinite(x_val) and np.isfinite(y_val):
                        all_x.append(x_val)
                        all_y.append(y_val)
        for item in anchor_geometries:
            for corner in item["corners_xy"]:
                x_val = float(corner[0])
                y_val = float(corner[1])
                #  NaN  Inf 
                if np.isfinite(x_val) and np.isfinite(y_val):
                    all_x.append(x_val)
                    all_y.append(y_val)
        for item in representative_infos:
            pos2d = item.get("pos2d")
            if pos2d is None:
                pos = item.get("pos")
                #  _project_to_ground 
                if pos is not None:
                    pos_arr = np.asarray(pos, dtype=np.float32)
                    if np.all(np.isfinite(pos_arr)):
                        pos2d = self._project_to_ground(pos)
                    else:
                        continue  # 
                else:
                    continue  #  None 
            x_val = float(pos2d[0])
            y_val = float(pos2d[1])
            #  NaN  Inf 
            if np.isfinite(x_val) and np.isfinite(y_val):
                all_x.append(x_val)
                all_y.append(y_val)
        if all_x and all_y:
            margin = 0.5
            #  NaN  Inf ()
            valid_x = [x for x in all_x if np.isfinite(x)]
            valid_y = [y for y in all_y if np.isfinite(y)]
            if valid_x and valid_y:
                try:
                    x_min = min(valid_x)
                    x_max = max(valid_x)
                    y_min = min(valid_y)
                    y_max = max(valid_y)
                    #  min/max 
                    if np.isfinite(x_min) and np.isfinite(x_max) and \
                       np.isfinite(y_min) and np.isfinite(y_max):
                        ax.set_xlim(x_min - margin, x_max + margin)
                        ax.set_ylim(y_min - margin, y_max + margin)
                    else:
                        raise ValueError("min/max ")
                except (ValueError, TypeError) as e:
                    # ,
                    self.logger.warning(
                        f"[MemoryCluster][] : {e},"
                    )
                    ax.set_xlim(-10, 10)
                    ax.set_ylim(-10, 10)
            else:
                # ,
                self.logger.warning(
                    "[MemoryCluster][] ,"
                )
                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)

        ax.legend(loc="upper right", fontsize=8)

        save_path = os.path.join(self.keyframe_dir, "topdown_stable_anchors.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        msg = f"[MemoryCluster][] : {save_path}"
        self.logger.info(msg)
        print(msg)

    def _collect_path_points(self) -> List[Dict]:
        points: List[Dict] = []
        for item in self.all_frame_poses:
            pose = item.get("pose")
            if pose is None:
                continue
            try:
                position = np.asarray(pose[:3, 3], dtype=np.float32)
            except Exception:
                continue
            points.append({"frame_idx": item.get("frame_idx", -1), "pos": position})
        return points

    def _collect_representative_pose_info(self) -> List[Dict]:
        infos: List[Dict] = []
        for cluster in self.clusters:
            rep: FrameStats = cluster.get("representative")
            if rep is None or rep.pose is None:
                continue
            try:
                pos3d = np.asarray(rep.pose[:3, 3], dtype=np.float32)
                pos2d = self._project_to_ground(pos3d)
                heading3d = -np.asarray(rep.pose[:3, 2], dtype=np.float32)
                heading2d = np.array(
                    [float(heading3d[0]), float(heading3d[2])],
                    dtype=np.float32,
                )
                norm = np.linalg.norm(heading2d)
                if norm > 1e-6:
                    heading2d = heading2d / norm
                else:
                    heading2d = np.array([0.0, 0.0], dtype=np.float32)
            except Exception:
                continue
            infos.append(
                {
                    "cluster_id": cluster.get("id", -1),
                    "frame_idx": rep.frame_idx,
                    "pos": pos3d,
                    "pos2d": pos2d,
                    "heading": heading2d,
                    "anchor_classes": self._extract_seen_anchor_classes(rep),
                    "anchor_uids": self._extract_seen_anchor_uids(rep),
                }
            )
        return infos

    def _extract_object_center(self, obj) -> Optional[np.ndarray]:
        bbox = getattr(obj, "bbox", None)
        if bbox is not None:
            try:
                center = bbox.get_center()
                return np.asarray(center, dtype=np.float32)
            except Exception:
                pass
        pcd = getattr(obj, "pcd", None)
        if pcd is not None:
            try:
                points = np.asarray(pcd.points)
                if points.size > 0:
                    return points.mean(axis=0).astype(np.float32)
            except Exception:
                pass
        return None

    def _log_frame_path_summary(self, path_points: List[Dict], csv_path: str) -> None:
        if not path_points:
            msg = "[MemoryCluster][] ,."
            self.logger.info(msg)
            print(msg)
            return

        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_idx", "x", "y", "z"])
                for item in path_points:
                    pos = item["pos"]
                    writer.writerow(
                        [
                            item["frame_idx"],
                            f"{float(pos[0]):.4f}",
                            f"{float(pos[1]):.4f}",
                            f"{float(pos[2]):.4f}",
                        ]
                    )
            msg = f"[MemoryCluster][] : {csv_path}"
            self.logger.info(msg)
            print(msg)
        except OSError as exc:
            self.logger.warning("[MemoryCluster] : %s", exc)

        start = path_points[0]
        end = path_points[-1]
        msg = (
            f"[MemoryCluster][]  {len(path_points)} ,"
            f"({start['pos'][0]:.2f}, {start['pos'][1]:.2f}), "
            f"({end['pos'][0]:.2f}, {end['pos'][1]:.2f})"
        )
        self.logger.info(msg)
        print(msg)

        sample_step = max(1, len(path_points) // 10)
        sample_points = path_points[::sample_step]
        print("[MemoryCluster][] :")
        for item in sample_points:
            pos = item["pos"]
            print(
                f"   {item['frame_idx']:06d}: "
                f"({float(pos[0]):.2f}, {float(pos[1]):.2f}, {float(pos[2]):.2f})"
            )

    def _collect_stable_anchor_geometries(self, local_map_manager) -> Tuple[List[Dict], Dict[str, Dict]]:
        if local_map_manager is None or not hasattr(local_map_manager, "local_map"):
            return [], {}

        class_lookup = []
        if hasattr(local_map_manager, "visualizer") and hasattr(local_map_manager.visualizer, "obj_classes"):
            try:
                class_lookup = local_map_manager.visualizer.obj_classes.get_classes_arr()
            except Exception:
                pass

        anchors: List[Dict] = []
        anchor_map: Dict[str, Dict] = {}
        for obj in getattr(local_map_manager, "local_map", []):
            if not getattr(obj, "is_stable", False):
                continue
            class_id = getattr(obj, "m_class_id", None)
            if class_id is None:
                class_id = getattr(obj, "class_id", None)
            class_name = None
            if class_id is not None and 0 <= int(class_id) < len(class_lookup):
                class_name = class_lookup[int(class_id)]
            elif class_id is not None and 0 <= int(class_id) < len(self.class_lookup):
                class_name = self.class_lookup[int(class_id)]
            if not class_name or not self._is_anchor_class_name(class_name):
                continue

            center3d = self._extract_object_center(obj)
            if center3d is None:
                continue
            center2d = self._project_to_ground(center3d)
            corners_xy = self._compute_topdown_corners(obj, center2d)
            uid = str(getattr(obj, "uid", f"anchor_{len(anchors):03d}"))
            anchor_entry = {
                "uid": uid,
                "class_name": class_name,
                "center": center2d,
                "corners_xy": corners_xy,
                "frame_indices": self._extract_object_frame_indices(obj),
                "half_extent": self._extract_object_half_extent(obj),
            }
            anchors.append(anchor_entry)
            anchor_map[uid] = anchor_entry
        return anchors, anchor_map

    def _compute_topdown_corners(self, obj, center: np.ndarray) -> np.ndarray:
        half = self._extract_object_half_extent(obj)

        cx, cy = float(center[0]), float(center[1])
        hx, hy = float(half[0]), float(half[1])
        corners = np.array(
            [
                [cx - hx, cy - hy],
                [cx + hx, cy - hy],
                [cx + hx, cy + hy],
                [cx - hx, cy + hy],
            ],
            dtype=np.float32,
        )
        return corners

    def _extract_object_half_extent(self, obj) -> np.ndarray:
        bbox = getattr(obj, "bbox", None)
        if bbox is not None:
            try:
                extent = np.asarray(bbox.get_extent(), dtype=np.float32)
                if extent.size >= 3:
                    planar = np.array([extent[0], extent[2]], dtype=np.float32)
                elif extent.size >= 2:
                    planar = extent[:2]
                else:
                    planar = np.array([0.3, 0.3], dtype=np.float32)
                return np.maximum(planar / 2.0, 0.05)
            except Exception:
                pass
        return np.array([0.15, 0.15], dtype=np.float32)

    def _extract_object_frame_indices(self, obj) -> List[int]:
        frames: Set[int] = set()
        for obs in getattr(obj, "observations", []) or []:
            frame_idx = getattr(obs, "idx", None)
            if frame_idx is None:
                frame_idx = getattr(obs, "frame_idx", None)
            if frame_idx is None:
                frame_idx = getattr(obs, "raw_image_idx", None)
            if frame_idx is None:
                continue
            try:
                frames.add(int(frame_idx))
            except (TypeError, ValueError):
                continue
        return sorted(frames)

    def _extract_seen_anchor_classes(self, stats: FrameStats) -> Set[str]:
        classes: Set[str] = set()
        for obs in getattr(stats, "valid_observations", []) or []:
            class_name = self._get_class_name(obs)
            if class_name and self._is_anchor_class_name(class_name):
                classes.add(class_name)
        return classes

    def _extract_seen_anchor_uids(self, stats: FrameStats) -> Dict[str, Dict]:
        return getattr(stats, "anchor_uid_map", {}) or {}

    def _format_frame_indices(self, indices: List[int], limit: int = 10) -> str:
        if not indices:
            return "-"
        sorted_idx = sorted(indices)
        if len(sorted_idx) <= limit:
            return ", ".join(f"{idx:06d}" for idx in sorted_idx)
        head = ", ".join(f"{idx:06d}" for idx in sorted_idx[:limit])
        return f"{head}, ..."

    def _rotate_2d(self, vec: np.ndarray, angle_rad: float) -> np.ndarray:
        if vec is None or len(vec) < 2:
            return np.array([0.0, 0.0], dtype=np.float32)
        c = float(np.cos(angle_rad))
        s = float(np.sin(angle_rad))
        x, y = float(vec[0]), float(vec[1])
        return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)

    def _project_to_ground(self, vec: Optional[np.ndarray]) -> np.ndarray:
        if vec is None:
            return np.zeros(2, dtype=np.float32)
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size >= 3:
            result = np.array([float(arr[0]), float(arr[2])], dtype=np.float32)
        elif arr.size == 2:
            result = arr.copy()
        else:
            result = np.zeros(2, dtype=np.float32)
        
        #  NaN  Inf,
        if not np.all(np.isfinite(result)):
            self.logger.warning(
                f"[MemoryCluster] _project_to_ground ,.: {vec}"
            )
            return np.zeros(2, dtype=np.float32)
        
        return result

    def _export_anchor_key_images(
        self,
        anchor_geometries: List[Dict],
        summary_data: Dict[str, any],
        local_map_manager=None,
        dataset=None,
        detector=None,
    ) -> None:
        if not anchor_geometries:
            self.logger.warning("[MemoryCluster] No anchor geometries to export (list is empty).")
            return

        if dataset is None:
            self.logger.warning("[MemoryCluster] Dataset is None, cannot export anchor key images.")
            return

        stable_objects = (summary_data or {}).get("stable_anchor_objects") or []
        anchor_dir = os.path.join(self.keyframe_dir, "anchor_key")
        if os.path.exists(anchor_dir):
            shutil.rmtree(anchor_dir)
        os.makedirs(anchor_dir, exist_ok=True)

        stable_lookup = {item["uid"]: item for item in stable_objects if item.get("uid")}
        rep_frames = (summary_data or {}).get("anchor_rep_frames") or {}

        # ID(UIDID 1, 2, 3...)
        # :,manifest
        anchor_id_map = {}
        for idx, anchor_geo in enumerate(anchor_geometries, 1):
            anchor_uid = anchor_geo.get("uid")
            if anchor_uid:
                anchor_id_map[anchor_uid] = idx

        exported = {}

        for anchor in anchor_geometries:
            uid = anchor.get("uid")
            if not uid:
                continue

            cls_name = anchor.get("class_name", "unknown")
            safe_uid = "".join(ch if ch.isalnum() else "_" for ch in uid) or "unknown"
            safe_class = "".join(ch if ch.isalnum() else "_" for ch in cls_name) or "unknown"
            instance_dir = os.path.join(anchor_dir, f"{safe_class}_{safe_uid[:8]}")
            os.makedirs(instance_dir, exist_ok=True)

            observed_frames = stable_lookup.get(uid, {}).get("frame_indices", [])
            representative_frame_list = rep_frames.get(uid, []) or []
            self.logger.info(
                "[MemoryCluster] Anchor UID=%s Class=%s Observed=%d RepFrames=%s",
                uid[:8],
                cls_name,
                len(observed_frames),
                self._format_frame_indices(representative_frame_list, limit=20),
            )

            entry = {
                "uid": uid,
                "class_name": cls_name,
                "observed_frames": observed_frames,
                "representative_frames": representative_frame_list,
                "saved_images": [],
            }

            saved_count = 0
            copied_frames: Set[int] = set()

            def _copy_frames(frame_list: List[int]) -> None:
                nonlocal saved_count
                for frame_idx in frame_list:
                    try:
                        idx_int = int(frame_idx)
                    except (TypeError, ValueError):
                        continue
                    if idx_int in copied_frames:
                        continue
                    dst_filename = f"frame_{idx_int:06d}_representative.png"
                    dst_path = os.path.join(instance_dir, dst_filename)
                    # Load from dataset and mark all stable anchor bboxes with IDs
                    if self._load_and_mark_anchor_center(
                        dataset, dst_path, uid, idx_int, local_map_manager, detector,
                        anchor_geometries, anchor_id_map, draw_all_anchors=True
                    ):
                        entry["saved_images"].append(
                            {
                                "frame_idx": idx_int,
                                "filename": dst_filename,
                                "path": dst_path,
                                "type": "representative",
                                "is_representative": True,
                            }
                        )
                        copied_frames.add(idx_int)
                        saved_count += 1
                    else:
                        self.logger.warning(
                            "[MemoryCluster]  %s (%s)  %06d ,.",
                            uid[:8],
                            cls_name,
                            idx_int,
                        )

            _copy_frames(representative_frame_list)

            if saved_count == 0:
                self.logger.warning(
                    "[MemoryCluster]  %s (%s) ,.",
                    uid[:8],
                    cls_name,
                )

            entry["saved_count"] = saved_count
            exported[uid] = entry

            self.logger.info(
                "[MemoryCluster]  %s (%s)  %d  %s",
                uid[:8],
                cls_name,
                saved_count,
                instance_dir,
            )

        manifest_path = os.path.join(anchor_dir, "manifest.json")
        # manifest,anchor_id_mapID
        # ID
        manifest_data = {
            "anchor_id_map": anchor_id_map,  # UIDsimple_id(ID)
            "anchors": exported
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)
        self.logger.info(
            "[MemoryCluster] Saved anchor_id_map with %d anchors to manifest.json",
            len(anchor_id_map)
        )

        self._write_anchor_frame_report(anchor_dir, exported)

        self.logger.info(
            "[MemoryCluster] Anchor keyframes , %d ,:%s",
            len(exported),
            anchor_dir,
        )

    def _find_keyframe_image_by_frame_idx(
        self,
        frame_idx: int,
        preferred_keyword: Optional[str] = None,
    ) -> Optional[str]:
        if frame_idx is None:
            return None
        pattern = os.path.join(
            self.keyframe_dir,
            f"cluster_*_frame_{int(frame_idx):06d}_*.png",
        )
        matches = sorted(glob.glob(pattern))
        if not matches:
            return None
        if preferred_keyword:
            for path in matches:
                if preferred_keyword.lower() in os.path.basename(path).lower():
                    return path
        return matches[0]

    def _safe_copy(self, src: str, dst: str) -> None:
        if not src or not os.path.exists(src):
            return
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            self.logger.warning("[MemoryCluster]  %s -> %s: %s", src, dst, exc)

    def _load_frame_detections(self, frame_idx: int, detector=None, dataset=None) -> Optional[Dict]:
        """
        Load detection results for a specific frame from disk.

        Args:
            frame_idx: Frame index to load detections for
            detector: Object detector instance with detection_path configured
            dataset: Dataset instance to get frame name

        Returns:
            Detection results dict or None if not found
        """
        if not detector or not hasattr(detector, 'detection_path'):
            self.logger.warning(f"[MemoryCluster] No detector or detection_path available")
            return None

        self.logger.warning(f"[MemoryCluster] Detection path: {detector.detection_path}")
        self.logger.warning(f"[MemoryCluster] Looking for detections for frame {frame_idx}")

        # Try to find the frame by index
        frame_name = None
        if dataset and hasattr(dataset, 'get_color'):
            try:
                color_data = dataset.get_color(frame_idx)
                if color_data and len(color_data) >= 2:
                    frame_name = color_data[1]  # Usually the filename
                    # Extract just the name without extension
                    if isinstance(frame_name, str):
                        frame_name = os.path.splitext(os.path.basename(frame_name))[0]
                    self.logger.warning(f"[MemoryCluster] Got frame name from dataset: {frame_name}")
            except Exception as e:
                self.logger.warning(f"[MemoryCluster] Failed to get frame name for {frame_idx}: {e}")

        if not frame_name:
            # Fallback: try common naming patterns - try both with and without 'frame' prefix
            candidates = []
            # Try frameXXXXXX format first
            for i in range(6, 0, -1):
                candidates.append(f"frame{frame_idx:0{i}d}")
            # Then try XXXXXX format
            for i in range(6, 0, -1):
                candidates.append(f"{frame_idx:0{i}d}")

            for candidate in candidates:
                det_path = detector.detection_path / candidate
                if det_path.exists():
                    frame_name = candidate
                    self.logger.warning(f"[MemoryCluster] Found frame name using fallback: {frame_name}")
                    break

        if not frame_name:
            self.logger.warning(f"[MemoryCluster] Could not determine frame name for frame {frame_idx}")
            return None

        det_path = detector.detection_path / frame_name
        self.logger.warning(f"[MemoryCluster] Looking for detection folder: {det_path}")
        if not det_path.exists():
            self.logger.warning(f"[MemoryCluster] Detection folder does not exist: {det_path}")
            # List all available detection folders for debugging
            if detector.detection_path.exists():
                available_folders = [f.name for f in detector.detection_path.iterdir() if f.is_dir()]
                self.logger.warning(f"[MemoryCluster] Available detection folders: {available_folders[:10]}...")  # Show first 10
            return None

        self.logger.warning(f"[MemoryCluster] Found detection folder: {det_path}")

        # Load detection results similar to object_detector.load_detection_results
        loaded_detections = {}
        try:
            import gzip
            import pickle
            for file_path in det_path.iterdir():
                if file_path.suffix == ".gz" and file_path.suffixes[-2] == ".pkl":
                    key = file_path.name.replace(".pkl.gz", "")
                    with gzip.open(file_path, "rb") as f:
                        loaded_detections[key] = pickle.load(f)
                elif file_path.suffix == ".npz":
                    loaded_detections[file_path.stem] = np.load(file_path)["arr_0"]
        except Exception as e:
            self.logger.warning(f"[MemoryCluster] Failed to load detections for frame {frame_idx}: {e}")
            return None

        return loaded_detections if loaded_detections else None

    def _load_and_mark_anchor_center(
        self,
        dataset,
        dst_path: str,
        anchor_uid: str,
        frame_idx: int,
        local_map_manager=None,
        detector=None,
        anchor_geometries: Optional[List[Dict]] = None,
        anchor_id_map: Optional[Dict[str, int]] = None,
        draw_all_anchors: bool = False,
    ) -> bool:
        """
        Load original image from dataset.
        If draw_all_anchors=True, mark all stable anchor bboxes with simple IDs.
        Otherwise, only mark the anchor center with a dot.
        Uses high-quality PNG compression to minimize quality loss.
        Returns True if successful, False otherwise.
        """
        if dataset is None:
            return False
        
        try:
            # Load original color image from dataset
            color_data = dataset.get_color(frame_idx)
            if not color_data:
                return False
            color_image, _ = color_data

            # Convert tensor to numpy
            if hasattr(color_image, "cpu"):
                color_image = color_image.cpu().numpy()
            color_np = np.asarray(color_image)

            # Ensure HWC format
            if color_np.ndim == 3 and color_np.shape[0] in {1, 3}:
                color_np = np.transpose(color_np, (1, 2, 0))

            # Ensure uint8 RGB
            if color_np.dtype != np.uint8:
                if color_np.max() <= 1.0:
                    color_np = (color_np * 255.0).astype(np.uint8)
                else:
                    color_np = np.clip(color_np, 0, 255).astype(np.uint8)

            if color_np.ndim != 3 or color_np.shape[2] != 3:
                self.logger.warning(
                    "[MemoryCluster]  %d : %s", frame_idx, color_np.shape
                )
                return False

            # Convert RGB to BGR for OpenCV
            img = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]

            # Save clean image (original) with suffix
            if dst_path:
                clean_path = dst_path.replace(".png", "_clean.png")
                try:
                    cv2.imwrite(clean_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    # self.logger.warning(f"[MemoryCluster] Saved clean image to {clean_path}")
                except Exception as e:
                    self.logger.warning(f"[MemoryCluster] Failed to save clean image: {e}")

            # Load detection results for this frame
            detections = None
            if detector:
                detections = self._load_frame_detections(frame_idx, detector, dataset)
                if detections:
                    self.logger.warning(f"[MemoryCluster] Loaded {len(detections.get('xyxy', []))} detections for frame {frame_idx}")
                    self.logger.warning(f"[MemoryCluster] Detection keys: {list(detections.keys())}")
                else:
                    self.logger.warning(f"[MemoryCluster] No detections found for frame {frame_idx}")

            # Draw detection bounding boxes if available
            if detections and 'xyxy' in detections and 'class_id' in detections:
                try:
                    xyxy_boxes = detections['xyxy']
                    class_ids = detections['class_id']
                    confidences = detections.get('confidence', [1.0] * len(xyxy_boxes))

                    self.logger.warning(f"[MemoryCluster] Drawing {len(xyxy_boxes)} detection boxes for frame {frame_idx}")

                    # Get class names if available
                    class_names = []
                    if detector and hasattr(detector, 'obj_classes'):
                        classes_arr = detector.obj_classes.get_classes_arr()
                        class_names = [classes_arr[cid] if cid < len(classes_arr) else f"class_{cid}" for cid in class_ids]
                        self.logger.warning(f"[MemoryCluster] Using class names from detector")
                    else:
                        class_names = [f"class_{cid}" for cid in class_ids]
                        self.logger.warning(f"[MemoryCluster] Using fallback class names")

                    # Draw each detection box
                    boxes_drawn = 0
                    for i, (box, class_name, conf) in enumerate(zip(xyxy_boxes, class_names, confidences)):
                        x1, y1, x2, y2 = box.astype(int)

                        # Skip invalid boxes
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # Choose color based on detection type (you can customize this)
                        if 'fastsam' in class_name.lower() or conf < 0.5:
                            color = (255, 0, 0)  # Red for FastSAM/low confidence
                        else:
                            color = (0, 255, 0)  # Green for YOLO/high confidence

                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                        # Draw label
                        label = f"{class_name} ({conf:.2f})"
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(img, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
                        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        boxes_drawn += 1

                    self.logger.warning(f"[MemoryCluster] Successfully drew {boxes_drawn} detection boxes")

                except Exception as e:
                    self.logger.warning(f"[MemoryCluster] Failed to draw detection boxes for frame {frame_idx}: {e}")
                    import traceback
                    self.logger.warning(f"[MemoryCluster] Traceback: {traceback.format_exc()}")

            # Draw all stable anchor bboxes with simple IDs (only when draw_all_anchors=True)
            if draw_all_anchors and anchor_geometries and anchor_id_map and local_map_manager:
                # First pass: collect all bboxes and labels
                bbox_list = []  # List of (x1, y1, x2, y2)
                label_list = []  # List of (label, bg_x1, bg_y1, bg_x2, bg_y2, text_x, text_y)
                
                for anchor_geo in anchor_geometries:
                    anchor_uid_geo = anchor_geo.get("uid")
                    if not anchor_uid_geo:
                        continue
                    
                    simple_id = anchor_id_map.get(anchor_uid_geo, -1)
                    if simple_id < 0:
                        continue
                    
                    # Find the anchor object in local_map
                    anchor_obj = None
                    for obj in local_map_manager.local_map:
                        if str(getattr(obj, 'uid', None)) == anchor_uid_geo:
                            anchor_obj = obj
                            break
                    
                    if not anchor_obj:
                        continue
                    
                    # Find observation in this frame
                    observations = getattr(anchor_obj, 'observations', []) or []
                    bbox = None
                    for obs in observations:
                        obs_frame_idx = getattr(obs, 'idx', getattr(obs, 'raw_image_idx', None))
                        if obs_frame_idx is not None:
                            try:
                                if int(obs_frame_idx) == frame_idx:
                                    xyxy = getattr(obs, 'xyxy', None)
                                    if xyxy is not None:
                                        bbox = xyxy
                                        break
                            except (TypeError, ValueError):
                                continue
                    
                    if bbox is not None:
                        try:
                            xyxy_arr = np.asarray(bbox, dtype=np.float32).reshape(-1, 4)
                            if len(xyxy_arr) > 0:
                                x1, y1, x2, y2 = map(int, xyxy_arr[0])
                                # Ensure coordinates are within image bounds
                                x1 = max(0, min(x1, w - 1))
                                y1 = max(0, min(y1, h - 1))
                                x2 = max(0, min(x2, w - 1))
                                y2 = max(0, min(y2, h - 1))
                                
                                if x2 > x1 and y2 > y1:
                                    # Store bbox for drawing
                                    bbox_list.append((x1, y1, x2, y2))
                                    
                                    # Calculate label position
                                    label = str(simple_id)
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 0.7
                                    thickness = 2
                                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                                    
                                    # Calculate padding
                                    padding = 4
                                    label_width = text_width + padding * 2
                                    label_height = text_height + padding * 2
                                    
                                    # Try multiple positions to ensure label is always visible
                                    label_pos = None
                                    
                                    # Position 1: Above box, top-left
                                    bg_x1 = x1
                                    bg_y1 = y1 - label_height
                                    bg_x2 = x1 + label_width
                                    bg_y2 = y1
                                    if bg_y1 >= 0 and bg_x2 <= w:
                                        label_pos = (bg_x1, bg_y1, bg_x2, bg_y2, bg_x1 + padding, bg_y2 - padding)
                                    
                                    # Position 2: Inside box, top-left corner
                                    if label_pos is None:
                                        bg_x1 = x1
                                        bg_y1 = y1
                                        bg_x2 = x1 + label_width
                                        bg_y2 = y1 + label_height
                                        if bg_x2 <= x2 and bg_y2 <= y2:
                                            label_pos = (bg_x1, bg_y1, bg_x2, bg_y2, bg_x1 + padding, bg_y2 - padding)
                                    
                                    # Position 3: Inside box, bottom-left corner
                                    if label_pos is None:
                                        bg_x1 = x1
                                        bg_y1 = y2 - label_height
                                        bg_x2 = x1 + label_width
                                        bg_y2 = y2
                                        if bg_y1 >= y1 and bg_x2 <= x2:
                                            label_pos = (bg_x1, bg_y1, bg_x2, bg_y2, bg_x1 + padding, bg_y2 - padding)
                                    
                                    # Position 4: Above box, top-right
                                    if label_pos is None:
                                        bg_x1 = x2 - label_width
                                        bg_y1 = y1 - label_height
                                        bg_x2 = x2
                                        bg_y2 = y1
                                        if bg_y1 >= 0 and bg_x1 >= 0:
                                            label_pos = (bg_x1, bg_y1, bg_x2, bg_y2, bg_x1 + padding, bg_y2 - padding)
                                    
                                    # Position 5: Inside box, top-right corner (last resort)
                                    if label_pos is None:
                                        bg_x1 = x2 - label_width
                                        bg_y1 = y1
                                        bg_x2 = x2
                                        bg_y2 = y1 + label_height
                                        if bg_x1 >= x1 and bg_y2 <= y2:
                                            label_pos = (bg_x1, bg_y1, bg_x2, bg_y2, bg_x1 + padding, bg_y2 - padding)
                                    
                                    if label_pos:
                                        label_list.append((label, label_pos, font, font_scale, thickness))
                                    else:
                                        self.logger.warning(f"[MemoryCluster] Could not find valid position for label on anchor {anchor_uid_geo[:8]} at frame {frame_idx}")
                        except Exception as e:
                            self.logger.warning(f"[MemoryCluster] Failed to process bbox for anchor {anchor_uid_geo[:8]}: {e}")
                
                # Second pass: draw all bboxes first
                for x1, y1, x2, y2 in bbox_list:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Third pass: draw all labels on top (so they won't be covered by other boxes)
                for label, (bg_x1, bg_y1, bg_x2, bg_y2, text_x, text_y), font, font_scale, thickness in label_list:
                    # Draw label background with a border to make it more visible
                    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 255), -1)  # Fill
                    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 1)  # White border
                    cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                
                anchors_drawn = len(bbox_list)
                
                if anchors_drawn > 0:
                    self.logger.info(f"[MemoryCluster] Drew {anchors_drawn} anchor bboxes on frame {frame_idx}")
            else:
                # When not drawing all anchors (e.g., in keyframe2), still draw anchor center as before
                center = self._get_anchor_center_in_frame(anchor_uid, frame_idx, local_map_manager, (h, w))
                if center is not None:
                    cx, cy = int(center[0]), int(center[1])
                    # Ensure center is within image bounds
                    cx = max(0, min(cx, w - 1))
                    cy = max(0, min(cy, h - 1))
                    # Draw a red dot at center (radius 4)
                    dot_radius = 4
                    cv2.circle(img, (cx, cy), dot_radius, (0, 0, 255), -1)  # Red dot, filled
            
            # Save with high quality (PNG is lossless)
            # Use cv2.IMWRITE_PNG_COMPRESSION with level 1 (fast, minimal compression)
            cv2.imwrite(dst_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            return True
            
        except Exception as exc:
            self.logger.warning(
                "[MemoryCluster]   %d -> %s: %s",
                frame_idx,
                dst_path,
                exc,
            )
            return False

    def _get_anchor_center_in_frame(
        self,
        anchor_uid: str,
        frame_idx: int,
        local_map_manager=None,
        image_shape: Tuple[int, int] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Get anchor center pixel coordinates in a specific frame.
        Returns (cx, cy) or None if not found.
        """
        if not local_map_manager or not hasattr(local_map_manager, "local_map"):
            return None
        
        # Find the anchor object
        anchor_obj = None
        for obj in local_map_manager.local_map:
            if str(getattr(obj, "uid", None)) == anchor_uid:
                anchor_obj = obj
                break
        
        if not anchor_obj:
            return None
        
        # Find observation in this frame
        observations = getattr(anchor_obj, "observations", []) or []
        for obs in observations:
            obs_frame_idx = getattr(obs, "idx", getattr(obs, "raw_image_idx", None))
            if obs_frame_idx is None:
                continue
            try:
                if int(obs_frame_idx) == frame_idx:
                    # Get bbox center from xyxy
                    xyxy = getattr(obs, "xyxy", None)
                    if xyxy is not None:
                        xyxy_arr = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
                        if len(xyxy_arr) > 0:
                            x1, y1, x2, y2 = xyxy_arr[0]
                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0
                            return (cx, cy)
            except (TypeError, ValueError):
                continue
        
        return None

    def _write_anchor_frame_report(self, anchor_dir: str, exported: Dict[str, Dict]) -> None:
        if not exported:
            return

        frames_manifest = {}
        for uid, data in exported.items():
            saved_images = data.get("saved_images", [])
            frames_manifest[uid] = {
                "class_name": data.get("class_name", "unknown"),
                "observed_frames": data.get("observed_frames", []),
                "representative_frames": data.get("representative_frames", []),
                "saved_count": data.get("saved_count", 0),
                "saved_images": [
                    {
                        "frame_idx": img.get("frame_idx"),
                        "filename": img.get("filename"),
                        "cluster_id": img.get("cluster_id"),
                        "cluster_type": img.get("cluster_type"),
                        "is_representative": img.get("is_representative", False),
                    }
                    for img in saved_images
                ],
            }

        frames_json = os.path.join(anchor_dir, "anchor_frames.json")
        with open(frames_json, "w", encoding="utf-8") as f:
            json.dump(frames_manifest, f, ensure_ascii=False, indent=2)

        frames_txt = os.path.join(anchor_dir, "anchor_frames.txt")
        lines = []

        def sort_key(item):
            uid, info = item
            return (info.get("class_name", ""), uid)

        for uid, data in sorted(frames_manifest.items(), key=sort_key):
            class_name = data.get("class_name", "unknown")
            short_uid = uid[:8] + "..." if len(uid) > 8 else uid
            observed = " ".join(str(idx) for idx in data.get("observed_frames") or []) or "-"
            representative = " ".join(str(idx) for idx in data.get("representative_frames") or []) or "-"
            saved_count = data.get("saved_count", 0)
            lines.append(
                f"{class_name} (UID: {short_uid})\n"
                f"  Observed Frames: {observed}\n"
                f"  Representative Frames: {representative}\n"
                f"  Saved Images: {saved_count} frames\n"
            )
        content = "\n".join(lines)
        with open(frames_txt, "w", encoding="utf-8") as f:
            f.write(content)

    def _tag_local_objects_representative_frames(
        self,
        local_map_manager,
        representative_frames_map: Dict[str, List[int]],
    ) -> None:
        local_map = getattr(local_map_manager, "local_map", None)
        if not local_map:
            return

        rep_sets = {uid: set(frames or []) for uid, frames in representative_frames_map.items()}

        for obj in local_map:
            uid = getattr(obj, "uid", None)
            if uid is None:
                continue
            uid_str = str(uid)

            # 
            rep_frames = sorted(rep_sets.get(uid_str, []))

            # 
            setattr(obj, "representative_frames", rep_frames)

            rep_lookup = set(rep_frames)

            # 
            for obs in getattr(obj, "observations", []) or []:
                frame_idx = getattr(obs, "idx", getattr(obs, "raw_image_idx", None))
                if frame_idx is None:
                    continue
                try:
                    idx_int = int(frame_idx)
                except (TypeError, ValueError):
                    continue

                obs.is_representative = idx_int in rep_lookup

