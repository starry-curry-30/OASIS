import json
import logging
import os

# Optionally import pdb for debugging purposes
import pdb
import queue
import threading
import time
from pathlib import Path
from collections import defaultdict, Counter
import math

import numpy as np
import psutil
import shutil
import sys
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R
from typing import Optional, List, Tuple, Dict, Any

from utils.global_map_manager import GlobalMapManager
from utils.local_map_manager import LocalMapManager
from utils.navigation_helper import (
    remaining_path,
    remove_sharp_turns_3d,
    NavigationGraph, # Added import
)
from utils.object_detector import Detector
from utils.novelty_detector import NoveltyDetector
from utils.local_observation_cache import LocalObservationCache
from utils.memory_cluster_manager import MemoryClusterManager
from utils.memory_cluster_debug import MemoryClusterDebugRecorder
from utils.memory_cluster_live_viz import MemoryClusterLiveVisualizer
from utils.time_utils import (
    get_map_memory_usage,
    print_timing_results,
    save_timing_results,
    timing_context,
)
from utils.types import DataInput, GoalMode, Observation
from utils.visualizer import ReRunVisualizer
import cv2
# Set up the module-level logger
logger = logging.getLogger(__name__)

_OASIS_ROOT = str(Path(__file__).resolve().parent.parent)


def _parse_nav_runner_result_from_stdout(stdout: str) -> Optional[Dict[str, Any]]:
    """
    Parse Result JSON from NavRunner child stdout.
    The script prints extra lines after JSON (e.g. Task Success, Navigate to frame);
    json.loads on the whole tail would fail and drop timing fields.
    Use JSONDecoder.raw_decode to parse only the first JSON object and ignore the rest.
    """
    if not stdout:
        return None
    marker = "[NavRunner] Result JSON:"
    # Multiple markers per run; use the last occurrence
    pos = stdout.rfind(marker)
    if pos < 0:
        return None
    tail = stdout[pos + len(marker) :].lstrip()
    if not tail.startswith("{"):
        brace = tail.find("{")
        if brace < 0:
            return None
        tail = tail[brace:]
    try:
        obj, _ = json.JSONDecoder().raw_decode(tail)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class _NewInstanceWarningFilter(logging.Filter):
    """
    Suppress WARNING logs tagged [Core][NewInstance] (noisy during new-instance tracking).
    ERROR logs still pass through.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True

        # Drop WARNING lines containing [Core][NewInstance]
        if record.levelno == logging.WARNING and "[Core][NewInstance]" in msg:
            return False
        return True


# Filter applies to this module's logger only
logger.addFilter(_NewInstanceWarningFilter())

BEST_FRAME_CANDIDATE_K = 6
BEST_FRAME_FINAL_K = 3
FAR_FRAME_SAVE_COUNT = 5
BEST_FRAME_METHOD = "unified"  # unified score; no separate macro/micro branch
BEST_FRAME_CONF_THRESHOLD = 0.6


class Oasis:
    def __init__(self, cfg: DictConfig):
        """
        Initialize Oasis with configuration and essential components.
        """
        self.cfg = cfg
        self.dataset = None
        
        # Flag to track if we are in relocalization step (navigating to a new location)
        self.is_relocalization_step = False
        
        # Relocalization loop state
        self.is_relocalization_loop_active = False
        self.relocalization_attempt_count = 0
        self.max_relocalization_attempts = self.cfg.navigation.get("max_relocalization_attempts", 3)
        
        # print config into console
        self.print_cfg()

        # Initialization
        self.visualizer = ReRunVisualizer(cfg)
        self.detector = Detector(cfg)
        self.local_map_manager = LocalMapManager(cfg)
        # Set OASIS instance reference for unified scoring
        self.local_map_manager._oasis_instance = self
        self.global_map_manager = GlobalMapManager(cfg)
        self.anchor_class_whitelist = self._load_anchor_class_whitelist()

        # Preload rep-frame poses into all_frame_poses when preload_global_map (handled after cluster init)
        if self.cfg.get("preload_global_map", False) and hasattr(self, "memory_cluster_manager") and self.memory_cluster_manager:
             pass

        # Additional initialization for visualization
        self.visualizer.set_use_rerun(cfg.use_rerun)
        self.visualizer.init("refactor_mapping")
        self.visualizer.spawn()

        # Keyframe Selection
        self.keyframe_counter = 0
        self.last_keyframe_time = None
        self.last_keyframe_pose = None
        self.time_threshold = cfg.time_threshold
        self.pose_threshold = cfg.pose_threshold
        self.rotation_threshold = cfg.rotation_threshold

        # pose memory
        self.curr_pose = None
        self.prev_pose = None
        self.wait_count = 0

        # check if need to preload the global map
        if self.cfg.preload_global_map:
            logger.warning("[Core][Init] Preloading global map...")
            self.global_map_manager.load_map()

        if self.cfg.preload_layout:
            logger.warning("[Core][Init] Preloading layout...")
            self.detector.load_layout()
            
            # load wall.pcd directly from disk
            self.global_map_manager.load_wall()
            
            # only generate wall.pcd if it doesn't exist
            if self.global_map_manager.layout_map.wall_pcd is None:
                logger.warning("[Core][Init] wall.pcd not found, generating from layout.pcd...")
                layout_pcd = self.detector.get_layout_pointcloud()
                self.global_map_manager.set_layout_info(layout_pcd)

        # flags for monitoring (must be defined before starting monitor thread)
        self.calculate_path = False  # Flag for calculating global path
        self.reset_cal_path_flag = False  # Flag for resetting global path calculation
        self.trigger_find_next = (
            False  # Flag for triggering the next global path planning
        )
        self.reset_trigger_find_next = (
            False  # Flag for resetting the next global path planning
        )

        # Start the file monitoring thread
        self.stop_thread = False  # Signal to stop the thread
        self.monitor_thread = threading.Thread(
            target=self.monitor_config_file, args=(cfg.config_file_path,)
        )
        self.monitor_thread.start()

        # Mode for Getting the Goal
        self.get_goal_mode = GoalMode.RANDOM
        # The request seq for getting the goal
        self.inquiry = ""
        self.inquiry_feat = None

        # Start local planning
        self.begin_local_planning = False

        # Final path for agent to follow
        self.action_path = None
        self.curr_global_path = None
        self.curr_local_path = None
        self.start_action_path = False

        # debug param: path counter
        self.path_counter = 0
        
        # Navigation mode: post-rotation VLM check
        self.nav_best_frame_path = None
        self.nav_inquiry_sentence = None
        self.need_post_rotation_vlm_check = False
        self.post_rotation_vlm_check_done = False
        self.rotation_completed = False  # set by ros_publisher

        # Nav timing JSON under project root time/
        self._nav_timing_json_dir = os.path.join(_OASIS_ROOT, "time")
        self._nav_timing_session_id: Optional[str] = None
        self._nav_timing_stages: List[Dict[str, Any]] = []

        # Parallel for mapping thread
        if self.cfg.use_parallel:
            self.detection_results_queue = queue.Queue(
                # maxsize=10
                maxsize=1000
            )  # Limit queue size to avoid memory leaks
            self._mapping_thread_busy = False
            # Initialize thread
            self.mapping_thread = threading.Thread(
                target=self.run_mapping_thread, daemon=True
            )
            self.mapping_thread.start()
        
        # Novelty detector + memory clusters
        novelty_cfg = getattr(cfg, 'novelty_detection', {})
        memory_cluster_cfg = getattr(cfg, 'memory_cluster', {})
        
        require_novelty_module = novelty_cfg.get('enabled', False) or memory_cluster_cfg.get('enabled', False)

        self.novelty_detector = None
        if require_novelty_module:
            self.novelty_detector = NoveltyDetector(
                clip_threshold=novelty_cfg.get('clip_threshold', 0.8),
                pose_translation_threshold=novelty_cfg.get('pose_translation_threshold', 0.5),
                pose_rotation_threshold=novelty_cfg.get('pose_rotation_threshold', 15.0),
            )
            logger.info("[Core] novelty detector initialized")

        self.memory_cluster_manager = None
        if memory_cluster_cfg.get('enabled', False):
            # Plain dict to avoid ConfigKeyError
            memory_cluster_cfg = dict(memory_cluster_cfg)

            # Inject paths from root cfg to memory_cluster_cfg
            if "representative_frames_path" in cfg:
                memory_cluster_cfg["representative_frames_path"] = cfg.representative_frames_path
                logger.info(f"[Core] Injected representative_frames_path: {cfg.representative_frames_path}")

            if "representative_frames_path" not in memory_cluster_cfg and not self.cfg.get("preload_global_map", False):
                 default_rep_path = os.path.join(self.cfg.output_path, "representative_frames")
                 memory_cluster_cfg["representative_frames_path"] = default_rep_path
                 logger.info(f"[Core] representative_frames_path unset; using default: {default_rep_path}")

            class_lookup = []
            if hasattr(self.detector, 'obj_classes'):
                try:
                    class_lookup = self.detector.obj_classes.get_classes_arr()
                except Exception:
                    class_lookup = []
            # Create unified scoring function
            unified_score_fn = lambda frame_idx: self.compute_frame_score_unified(frame_idx)

            self.memory_cluster_manager = MemoryClusterManager(
                cfg=memory_cluster_cfg,
                anchor_whitelist=self.anchor_class_whitelist,
                class_lookup=class_lookup,
                clip_feature_fn=self._compute_clip_feature_for_frame,
                unified_score_fn=unified_score_fn,
                complex_score_fn=lambda frame_idx: self.compute_frame_score_unified(frame_idx),
                reset_dir=not self.cfg.get("preload_global_map", False),  # keep dir in nav / preload
                clustering_mode="hybrid",
            )

            # OASIS ablation: 9 managers (3 clustering x 3 selection), stats only
            self._frame_score_details = {}
            self._oasis_ablation_managers = []
            if self.cfg.get("run_oasis_ablation", False):
                for c_mode in ("spatial", "visual", "hybrid"):
                    for s_mode in (0, 1, 2):
                        def _make_ablation_score_fn(selection_mode: int):
                            def _score_fn(frame_idx: int) -> float:
                                d = self._frame_score_details.get(frame_idx, {})
                                g = float(d.get("GlobalTerm_norm", 0.0))
                                l = float(d.get("LocalTerm_norm", 0.0))
                                if selection_mode == 0:
                                    return g
                                if selection_mode == 1:
                                    return l
                                return 0.6 * g + 0.4 * l
                            return _score_fn
                        _score_fn = _make_ablation_score_fn(s_mode)
                        m = MemoryClusterManager(
                            cfg=memory_cluster_cfg,
                            anchor_whitelist=self.anchor_class_whitelist,
                            class_lookup=class_lookup,
                            clip_feature_fn=self._compute_clip_feature_for_frame,
                            unified_score_fn=_score_fn,
                            complex_score_fn=_score_fn,
                            reset_dir=False,
                            clustering_mode=c_mode,
                        )
                        self._oasis_ablation_managers.append(m)
                logger.info("[Core] OASIS ablation: 9 managers created (3 clustering × 3 selection).")
            
            # Load existing clusters if rep-frame path exists
            if self.cfg.get("preload_global_map", False) and "representative_frames_path" in memory_cluster_cfg:
                load_dir = memory_cluster_cfg["representative_frames_path"]
                
                # Lifelong nav mode
                lifelong_nav = False
                if hasattr(self.cfg, "navigation") and hasattr(self.cfg.navigation, "lifelong_navigation"):
                    lifelong_nav = self.cfg.navigation.lifelong_navigation
                
                if lifelong_nav:
                    # e.g. .../representative_frames -> .../representative_frames_nav
                    base_dir_name = os.path.basename(load_dir.rstrip('/'))
                    parent_dir = os.path.dirname(load_dir.rstrip('/'))
                    output_keyframe_dir = os.path.join(parent_dir, f"{base_dir_name}_nav")
                    logger.info(f"[Core] [LifelongNav] output dir: {output_keyframe_dir}")
                else:
                    output_keyframe_dir = os.path.join(self.cfg.output_path, "representative_frames")
                
                output_keyframe_dir = os.path.abspath(output_keyframe_dir)

                if os.path.exists(load_dir):
                    logger.info(f"[Core] loading clusters from {load_dir}...")
                    
                    use_existing_output = False
                    if lifelong_nav and os.path.exists(output_keyframe_dir):
                        # Non-empty if manifest exists
                        if os.path.exists(os.path.join(output_keyframe_dir, "anchor_key", "manifest.json")):
                            logger.info(f"[Core] [LifelongNav] reusing existing nav data: {output_keyframe_dir}")
                            use_existing_output = True
                    
                    if not use_existing_output:
                        if os.path.abspath(load_dir) != output_keyframe_dir:
                            logger.info(f"[Core] copying cluster files to: {output_keyframe_dir}")
                            if os.path.exists(output_keyframe_dir):
                                shutil.rmtree(output_keyframe_dir)
                            shutil.copytree(load_dir, output_keyframe_dir)
                    
                    self.memory_cluster_manager.keyframe_dir = output_keyframe_dir
                    self.memory_cluster_manager._keyframe_dir_ready = True

                    self.memory_cluster_manager.load_clusters(output_keyframe_dir)
                    
                    logger.info(f"[Core] [LifelongNav] keyframe_dir={self.memory_cluster_manager.keyframe_dir}")
                    logger.info(f"[Core] [LifelongNav] VLM manifest: {os.path.join(self.memory_cluster_manager.keyframe_dir, 'anchor_key', 'manifest.json')}")

                    if self.memory_cluster_manager.clusters:
                        restored_count = 0
                        for cluster in self.memory_cluster_manager.clusters:
                            rep = cluster.get("representative")
                            if rep and hasattr(rep, "pose") and rep.pose is not None:
                                self.memory_cluster_manager.all_frame_poses.append({
                                    "frame_idx": rep.frame_idx,
                                    "pose": rep.pose
                                })
                                restored_count += 1
                            for frame in cluster.get("frames", []):
                                if frame and hasattr(frame, "pose") and frame.pose is not None:
                                     if not any(p["frame_idx"] == frame.frame_idx for p in self.memory_cluster_manager.all_frame_poses):
                                         self.memory_cluster_manager.all_frame_poses.append({
                                             "frame_idx": frame.frame_idx,
                                             "pose": frame.pose
                                         })
                                         restored_count += 1
                        logger.info(f"[Core] restored {restored_count} frame poses into all_frame_poses")
                    
                    logger.info(f"[Core] keyframe output dir: {self.memory_cluster_manager.keyframe_dir}")

            logger.info(
                "[Core] memory cluster / rep frames enabled, dir=%s",
                self.memory_cluster_manager.keyframe_dir,
            )

        self.memory_cluster_debugger = None
        debug_cfg = getattr(cfg, "memory_cluster_debug", {})
        if (
            memory_cluster_cfg.get("enabled", False)
            and debug_cfg.get("enabled", False)
        ):
            class_lookup = []
            if hasattr(self.detector, "obj_classes"):
                try:
                    class_lookup = self.detector.obj_classes.get_classes_arr()
                except Exception:
                    class_lookup = []
            self.memory_cluster_debugger = MemoryClusterDebugRecorder(
                cfg=debug_cfg,
                class_lookup=class_lookup,
                anchor_whitelist=self.anchor_class_whitelist,
            )

            if self.memory_cluster_manager is not None:
                original_clip_fn = self.memory_cluster_manager.clip_feature_fn

                def _debug_clip_feature_fn(image):
                    feature = None
                    if original_clip_fn is not None:
                        feature = original_clip_fn(image)
                    if self.memory_cluster_debugger is not None:
                        self.memory_cluster_debugger.attach_clip_feature(feature)
                    return feature

                self.memory_cluster_manager.clip_feature_fn = _debug_clip_feature_fn

        self.memory_cluster_live_viz = None
        live_viz_cfg = dict(getattr(cfg, "memory_cluster_live_viz", None) or {})
        if "max_observation_depth_m" not in live_viz_cfg:
            _md = memory_cluster_cfg.get("max_observation_depth_m")
            if _md is not None:
                live_viz_cfg["max_observation_depth_m"] = float(_md)
        if memory_cluster_cfg.get("enabled", False) and live_viz_cfg.get("enabled", False):
            if live_viz_cfg.get("save_video") and not live_viz_cfg.get("video_path"):
                _out = os.path.abspath(os.path.expanduser(str(cfg.output_path)))
                os.makedirs(_out, exist_ok=True)
                # AVI+MJPG is most reliable with OpenCV; MP4/mp4v often produces unreadable files.
                live_viz_cfg["video_path"] = os.path.join(_out, "memory_cluster_live.avi")
            self.memory_cluster_live_viz = MemoryClusterLiveVisualizer(live_viz_cfg)

        # Local observation cache
        self.observation_cache = None
        obs_cache_cfg = getattr(cfg, "observation_cache", None)
        if obs_cache_cfg and obs_cache_cfg.get("enabled", False):
            self.observation_cache = LocalObservationCache(obs_cache_cfg)
            logger.info(
                "[Core] LocalObservation cache mode=%s dir=%s",
                obs_cache_cfg.get("mode", "save"),
                self.observation_cache.output_dir,
            )
        
        # Back-reference for unified frame scoring
        if hasattr(self, "local_map_manager") and self.local_map_manager is not None:
            self.local_map_manager._oasis_instance = self
        
        if self.cfg.get("preload_global_map", False):
            self.new_instance_tracker = {}
            self.new_instance_frames = {}
            dataset_name = self.cfg.get("dataset_name", "unknown")
            scene_id = self.cfg.get("scene_id", "unknown")
            scene_dir = f"{dataset_name}_{scene_id}"
            self.new_instance_save_dir = os.path.join(self.cfg.output_path, scene_dir, "new_instance")
            os.makedirs(self.new_instance_save_dir, exist_ok=True)
            logger.info(f"[Core][NewInstance] tracking on, save_dir={self.new_instance_save_dir}")
        else:
            logger.info(f"[Core][NewInstance] not in nav preload mode, preload_global_map={self.cfg.get('preload_global_map', False)}")

    def _load_anchor_class_whitelist(self):
        anchor_path = getattr(self.cfg.yolo, "given_anchor_classes", None) if hasattr(self.cfg, "yolo") else None
        if not anchor_path:
            anchor_path = getattr(self.cfg, "given_anchor_classes", None)
        if not anchor_path:
            logger.info("[Core][AnchorWhitelist] No anchor class path configured. Ignore anchor whitelist.")
            return None
        
        # Resolve path (Hydra cwd)
        original_path = str(anchor_path)
        anchor_path = original_path
        try:
            from hydra.utils import to_absolute_path

            anchor_path = to_absolute_path(original_path)
        except Exception:
            anchor_path = os.path.abspath(os.path.expanduser(original_path))

        logger.info(f"[Core][AnchorWhitelist] config path: {original_path}")
        logger.info(f"[Core][AnchorWhitelist] absolute path: {anchor_path}")
        
        if not os.path.isfile(anchor_path):
            logger.warning(
                "[Core][AnchorWhitelist] File not found: %s. Ignore anchor whitelist.",
                anchor_path,
            )
            return None
        anchor_set = set()
        try:
            with open(anchor_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = line.strip()
                    if not entry or entry.startswith("#"):
                        continue
                    anchor_set.add(entry.lower())
        except Exception as exc:
            logger.error(
                "[Core][AnchorWhitelist] Failed to load %s: %s. Ignore whitelist.",
                anchor_path,
                exc,
            )
            return None

        if not anchor_set:
            logger.warning(
                "[Core][AnchorWhitelist] No valid classes found in %s. Ignore whitelist.",
                anchor_path,
            )
            return None

        logger.info(
            "[Core][AnchorWhitelist] Loaded %d anchor classes from %s.",
            len(anchor_set),
            anchor_path,
        )
        return anchor_set

    def _is_anchor_class_allowed(self, class_name: Optional[str]) -> bool:
        if not class_name or class_name == "unknown":
            return False
        if not self.anchor_class_whitelist:
            return True
        return class_name.lower() in self.anchor_class_whitelist

    def print_cfg(self):

        log_file_path = ""

        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_path = handler.baseFilename

        # Organize configuration items to be printed into a list
        cfg_items = [
            ("Log Path", log_file_path),
            ("Output Dir", self.cfg.output_path),
            ("Map Save Dir", self.cfg.map_save_path),
            ("Class List Path", self.cfg.yolo.given_classes_path),
            ("Use FastSAM for OV?", self.cfg.use_fastsam),
            ("Running Concrete Map Only?", self.cfg.run_local_mapping_only),
            ("Save Concrete Map?", self.cfg.save_local_map),
            ("Save Global Map?", self.cfg.save_global_map),
            ("Use Preload Global Map?", self.cfg.preload_global_map),
            ("Use Rerun for Visualization?", self.cfg.use_rerun),
            ("Representative Frame Path", self.cfg.representative_frames_path),
        ]

        # if has cfg.ros_dataset_config, add it to the list
        if "ros_stream_config_path" in self.cfg:
            cfg_items.append(
                ("ROS Stream Config Path", self.cfg.ros_stream_config_path)
            )

        # Define separator line length
        line_length = 60
        print("=" * line_length)
        for key, value in cfg_items:
            print(f"{key:<30} : {value}")
        print("=" * line_length)

    def get_keyframe_idx(self):
        return self.keyframe_counter

    def check_keyframe(self, time_stamp, curr_pose):
        """
        Check if the current frame should be selected as a keyframe based on
        time interval, pose difference (translation), and rotation difference.
        """
        is_keyframe = False
        # Translation check
        if self.last_keyframe_pose is not None:
            translation_diff = np.linalg.norm(
                curr_pose[:3, 3] - self.last_keyframe_pose[:3, 3]
            )  # Translation difference
            if translation_diff >= self.pose_threshold:
                self.last_keyframe_time = time_stamp
                self.last_keyframe_pose = curr_pose
                logger.info(
                    "[Core][CheckKeyframe] New keyframe detected by translation"
                )
                is_keyframe = True

            # Rotation check
            curr_rotation = R.from_matrix(curr_pose[:3, :3])
            last_rotation = R.from_matrix(self.last_keyframe_pose[:3, :3])
            rotation_diff = curr_rotation.inv() * last_rotation
            angle_diff = rotation_diff.magnitude() * (180 / np.pi)

            if angle_diff >= self.rotation_threshold:
                self.last_keyframe_time = time_stamp
                self.last_keyframe_pose = curr_pose
                logger.info("[Core][CheckKeyframe] New keyframe detected by rotation")
                is_keyframe = True

        # Time check
        if (
            self.last_keyframe_time is None
            or abs(time_stamp - self.last_keyframe_time) >= self.time_threshold
        ):
            self.last_keyframe_time = time_stamp
            self.last_keyframe_pose = curr_pose
            logger.info("[Core][CheckKeyframe] New keyframe detected by time")
            is_keyframe = True

        if is_keyframe:
            self.keyframe_counter += 1
            logger.info(
                f"[Core][CheckKeyframe] Current frame is keyframe: {self.keyframe_counter}"
            )
            return True
        else:
            # logger.info("Not a new keyframe, abandon")
            return False

    def _compute_clip_feature_for_frame(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self.novelty_detector is None:
            return None
        if not hasattr(self.detector, "clip_model") or self.detector.clip_model is None:
            return None
        if not hasattr(self.detector, "clip_preprocess") or self.detector.clip_preprocess is None:
            return None
        try:
            return self.novelty_detector.compute_full_image_clip_feature(
                image,
                self.detector.clip_model,
                self.detector.clip_preprocess,
                getattr(self.cfg, "device", "cuda"),
            )
        except Exception as exc:
            logger.warning("[Core][MemoryCluster] CLIP feature failed: %s", exc)
            return None

    def _build_memory_cluster_viz_context(
        self,
        result: Optional[Dict[str, Any]],
        pre_rep_by_cluster: Dict[int, int],
    ) -> Dict[str, Any]:
        """Structured context for memory_cluster_live_viz right-hand status panel."""
        if result is None:
            return {"mode": "skipped", "frame_idx": int(self.curr_frame_id)}

        def _u_score(fidx: int) -> Optional[float]:
            try:
                return float(self.compute_frame_score_unified(int(fidx)))
            except Exception:
                return None

        frame_idx = int(result["frame_idx"])
        cid = int(result["cluster_id"])
        curr_s = _u_score(frame_idx)
        if curr_s is None:
            curr_s = float(result.get("score", 0.0))

        is_new = bool(result.get("is_new_cluster"))
        reason = str(result.get("novelty_reason") or "").strip().lower()

        if is_new:
            mk = None
            if reason == "clip":
                mk = "visual"
            elif reason in ("translation", "rotation"):
                mk = "spatial"
            elif reason == "first_frame":
                mk = "first_frame"
            else:
                tf = result.get("trigger_flags") or {}
                if tf.get("translation") or tf.get("rotation"):
                    mk = "spatial"
                elif tf.get("clip"):
                    mk = "visual"
            return {
                "mode": "new_cluster",
                "cluster_id": cid,
                "novelty_reason": reason,
                "mismatch_kind": mk,
                "frame_idx": frame_idx,
                "current_score": curr_s,
                "clip_similarity": result.get("clip_similarity"),
                "translation_diff": result.get("translation_diff"),
                "rotation_diff": result.get("rotation_diff"),
            }

        old_rep = pre_rep_by_cluster.get(cid)
        cluster = None
        for c in self.memory_cluster_manager.clusters or []:
            if int(c.get("id", -1)) == cid:
                cluster = c
                break
        new_rep = None
        if cluster is not None and cluster.get("representative") is not None:
            new_rep = int(cluster["representative"].frame_idx)
        old_sc = _u_score(old_rep) if old_rep is not None else None
        new_sc = _u_score(new_rep) if new_rep is not None else None
        rep_updated = (
            old_rep is not None
            and new_rep is not None
            and old_rep != new_rep
        )
        return {
            "mode": "matched",
            "cluster_id": cid,
            "frame_idx": frame_idx,
            "current_score": curr_s,
            "old_rep_frame": old_rep,
            "new_rep_frame": new_rep,
            "old_rep_score": old_sc,
            "new_rep_score": new_sc,
            "representative_updated": rep_updated,
            "compared_rep_frame_idx": result.get("representative_frame_idx"),
            "clip_similarity": result.get("clip_similarity"),
            "translation_diff": result.get("translation_diff"),
            "rotation_diff": result.get("rotation_diff"),
        }

    def _update_memory_clusters(self, data_input: DataInput, observations: List[Observation], has_new_instance: bool = False) -> Optional[dict]:
        if not self.memory_cluster_manager:
            return None
        try:
            debug_recorder = getattr(self, "memory_cluster_debugger", None)
            if debug_recorder and debug_recorder.enabled:
                try:
                    debug_recorder.capture_frame(
                        frame_idx=self.curr_frame_id,
                        data_input=data_input,
                        observations=observations,
                    )
                except Exception as exc:
                    logger.warning("[Core][MemoryClusterDebug] capture frame failed: %s", exc)
            if (
                not getattr(self.memory_cluster_manager, "class_lookup", None)
                and hasattr(self.detector, "obj_classes")
            ):
                try:
                    self.memory_cluster_manager.update_class_lookup(
                        self.detector.obj_classes.get_classes_arr()
                    )
                except Exception:
                    pass
            pre_rep_by_cluster: Dict[int, int] = {}
            for c in getattr(self.memory_cluster_manager, "clusters", None) or []:
                rep = c.get("representative")
                if rep is None:
                    continue
                try:
                    pre_rep_by_cluster[int(c.get("id", -1))] = int(rep.frame_idx)
                except (TypeError, ValueError):
                    pass
            precomputed_clip = self._compute_clip_feature_for_frame(data_input.color)
            result = self.memory_cluster_manager.process_frame(
                frame_idx=self.curr_frame_id,
                color_image=data_input.color,
                pose=data_input.pose,
                observations=observations,
                precomputed_clip_feature=precomputed_clip,
                has_new_instance=has_new_instance,
            )
            live_viz = getattr(self, "memory_cluster_live_viz", None)
            if (
                live_viz is not None
                and getattr(live_viz, "enabled", False)
                and result is not None
            ):
                try:
                    viz_ctx = self._build_memory_cluster_viz_context(
                        result, pre_rep_by_cluster
                    )
                    live_viz.update(
                        self.memory_cluster_manager,
                        result,
                        data_input.color,
                        self.curr_frame_id,
                        observations=observations,
                        viz_context=viz_ctx,
                    )
                except Exception as viz_exc:
                    logger.debug("[Core][MemoryClusterLiveViz] update: %s", viz_exc)
            if self.cfg.get("run_oasis_ablation", False):
                try:
                    details = self.compute_frame_score_unified(
                        self.curr_frame_id, return_details=True
                    )
                    if isinstance(details, dict):
                        self._frame_score_details[self.curr_frame_id] = details
                except Exception as e:
                    logger.debug("[Core][OASIS] store score details for frame %s: %s", self.curr_frame_id, e)
                for m in getattr(self, "_oasis_ablation_managers", []):
                    if not getattr(m, "class_lookup", None) and hasattr(self.detector, "obj_classes"):
                        try:
                            m.update_class_lookup(self.detector.obj_classes.get_classes_arr())
                        except Exception:
                            pass
                    m.process_frame(
                        frame_idx=self.curr_frame_id,
                        color_image=data_input.color,
                        pose=data_input.pose,
                        observations=observations,
                        precomputed_clip_feature=precomputed_clip,
                        has_new_instance=has_new_instance,
                    )
            if debug_recorder and debug_recorder.enabled:
                debug_recorder.finalize_frame()
            return result
        except Exception as exc:
            debug_recorder = getattr(self, "memory_cluster_debugger", None)
            if debug_recorder and debug_recorder.enabled:
                debug_recorder.abort_frame()
            logger.warning(
                "[Core][MemoryCluster] frame %d failed: %s",
                self.curr_frame_id,
                exc,
            )
            return None

    def _get_class_name_from_obs(self, obs) -> Optional[str]:
        """Class name from observation; SAM without class -> 'unknown'"""
        class_name = getattr(obs, "class_name", None)
        if class_name:
            return class_name
        class_id = getattr(obs, "class_id", None)
        if class_id is None:
            # No class_id: check SAM (mask but no class)
            if hasattr(obs, "mask") and obs.mask is not None:
                # Likely SAM without class label
                return "unknown"
            return None
        try:
            class_idx = int(class_id)
        except (TypeError, ValueError):
            return None
        
        # Check FastSAM unknown_class_id
        if hasattr(self.detector, "unknown_class_id") and class_idx == self.detector.unknown_class_id:
            return "unknown"
        
        if hasattr(self.detector, "obj_classes"):
            try:
                classes_arr = self.detector.obj_classes.get_classes_arr()
                if 0 <= class_idx < len(classes_arr):
                    return classes_arr[class_idx]
            except:
                pass
        
        # Valid class_id but no name: likely SAM
        if class_idx >= 0:
            return "unknown"
        
        return None

    def _get_obs_bbox_xyxy(self, obs, img_width: int, img_height: int) -> Optional[Tuple[int, int, int, int]]:
        """2D bbox (x1,y1,x2,y2); prefer fused xyxy else from mask"""
        import numpy as np
        
        # Prefer fused xyxy
        if hasattr(obs, "xyxy") and obs.xyxy is not None:
            xyxy = np.asarray(obs.xyxy)
            # Empty array check
            if xyxy.size > 0:
                # If 2D, take first row
                if xyxy.ndim == 2 and xyxy.shape[0] > 0:
                    xyxy = xyxy[0]
                # Length check
                if len(xyxy) >= 4:
                    x1 = max(0, int(xyxy[0]))
                    y1 = max(0, int(xyxy[1]))
                    x2 = min(img_width, int(xyxy[2]))
                    y2 = min(img_height, int(xyxy[3]))
                    if x2 > x1 and y2 > y1:
                        return (x1, y1, x2, y2)
        
        # If no xyxy, bbox from fused mask
        if hasattr(obs, "mask") and obs.mask is not None:
            mask = np.asarray(obs.mask)
            # Handle varying mask dims
            if mask.ndim == 3:
                if mask.shape[0] == 1:
                    mask = mask[0]
                elif mask.shape[-1] == 1:
                    mask = mask[..., 0]
            if mask.ndim == 2:
                mask_bool = mask.astype(bool) if mask.dtype != bool else mask
                ys, xs = np.where(mask_bool)
                if len(xs) > 0 and len(ys) > 0:
                    x1 = int(np.clip(xs.min(), 0, img_width))
                    x2 = int(np.clip(xs.max() + 1, 0, img_width))
                    y1 = int(np.clip(ys.min(), 0, img_height))
                    y2 = int(np.clip(ys.max() + 1, 0, img_height))
                    if x2 > x1 and y2 > y1:
                        return (x1, y1, x2, y2)
        
        return None

    def _get_obs_bbox_xyxy_from_2d(self, obs, img_width: int, img_height: int) -> Optional[Tuple[int, int, int, int]]:
        """Project bbox_2d (xy) to image bbox"""
        if not hasattr(obs, "bbox_2d") or obs.bbox_2d is None:
            return None
        
        # Try xyxy first
        if hasattr(obs, "xyxy") and obs.xyxy is not None and len(obs.xyxy) >= 4:
            return self._get_obs_bbox_xyxy(obs, img_width, img_height)
        
        # Else project bbox_2d (needs intrinsics; stub)
        # Return None for now
        return None

    def _compute_bbox_intersection_ratio(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Intersection ratio of two bbox_2d (same as compute_match_by_intersection_ratio)"""
        import torch
        
        # bbox: [min_x, min_y, max_x, max_y]
        bbox1_t = torch.tensor([bbox1], dtype=torch.float32)
        bbox2_t = torch.tensor([bbox2], dtype=torch.float32)
        
        # Intersection
        inter_min_x = torch.max(bbox1_t[:, 0], bbox2_t[:, 0])
        inter_min_y = torch.max(bbox1_t[:, 1], bbox2_t[:, 1])
        inter_max_x = torch.min(bbox1_t[:, 2], bbox2_t[:, 2])
        inter_max_y = torch.min(bbox1_t[:, 3], bbox2_t[:, 3])
        
        inter_width = (inter_max_x - inter_min_x).clamp(min=0)
        inter_height = (inter_max_y - inter_min_y).clamp(min=0)
        inter_area = inter_width * inter_height
        
        # Areas
        bbox1_area = (bbox1_t[:, 2] - bbox1_t[:, 0]) * (bbox1_t[:, 3] - bbox1_t[:, 1])
        bbox2_area = (bbox2_t[:, 2] - bbox2_t[:, 0]) * (bbox2_t[:, 3] - bbox2_t[:, 1])
        
        # Intersection()
        ratio1 = inter_area / bbox1_area
        ratio2 = inter_area / bbox2_area
        ratio = torch.max(ratio1, ratio2)
        
        return ratio.item()

    def _draw_bbox_on_image(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                            class_name: str, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw bbox on image"""
        import cv2
        x1, y1, x2, y2 = bbox
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        # Draw label
        label = class_name
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return image

    def _process_new_instances_nav_realtime(self, curr_obs_list: List, data_input: DataInput, frame_id: int):
        """Nav mode: process new instances on main thread after detection"""
        if not hasattr(self, "new_instance_tracker"):
            return
        
        logger.warning(f"[Core][NewInstance] realtime frame {frame_id},local obs count: {len(curr_obs_list)}")
        
        import numpy as np
        
        # scan frame for new instances
        # heuristic: local obs whose pcd is not in global map
        # fast 3D center match
        new_instances_this_frame = []
        img_height, img_width = data_input.color.shape[:2]
        
        for local_obs in curr_obs_list:
            # check global map by 3D center
            is_new = True
            if self.global_map_manager.has_global_map() and len(self.global_map_manager.global_map) > 0:
                if hasattr(local_obs, "pcd") and len(local_obs.pcd.points) > 0:
                    # local obs center
                    local_center = np.asarray(local_obs.pcd.points).mean(axis=0)
                    
                    # compare to global object centers
                    for global_obj in self.global_map_manager.global_map:
                        if hasattr(global_obj, "pcd_2d") and len(global_obj.pcd_2d.points) > 0:
                            global_center_2d = np.asarray(global_obj.pcd_2d.points).mean(axis=0)
                            # xy only
                            dist_2d = np.sqrt(np.sum((local_center[:2] - global_center_2d[:2])**2))
                            if dist_2d < 0.5:  # tunable threshold
                                is_new = False
                                break
            
            if is_new:
                class_name = self._get_class_name_from_obs(local_obs)
                class_id = getattr(local_obs, "class_id", None)
                
                # check unknown_class_id (SAM/FastSAM)
                is_unknown_class = False
                if hasattr(self.detector, "unknown_class_id") and class_id is not None:
                    try:
                        class_idx = int(class_id)
                        if class_idx == self.detector.unknown_class_id:
                            is_unknown_class = True
                            class_name = "unknown"
                    except (TypeError, ValueError):
                        pass
                
                # if no class_name but mask -> SAM unknown
                if class_name is None:
                    if hasattr(local_obs, "mask") and local_obs.mask is not None:
                        class_name = "unknown"
                        is_unknown_class = True
                    else:
                        continue  # skip: no class and no mask
                
                # empty class_name -> unknown
                if not class_name:
                    class_name = "unknown"
                    is_unknown_class = True
                
                # debug: log unknown detections
                if is_unknown_class or class_name == "unknown":
                    logger.warning(f"[Core][NewInstance] unknown-class instance: class_id={class_id}, class_name={class_name}, has_mask={hasattr(local_obs, 'mask') and local_obs.mask is not None}")
                
                # bbox_2d from xy projection
                bbox_2d = None
                if hasattr(local_obs, "pcd") and len(local_obs.pcd.points) > 0:
                    import open3d as o3d
                    points_2d = np.asarray(local_obs.pcd.points)[:, :2]  # xy only
                    if len(points_2d) > 0:
                        min_xy = points_2d.min(axis=0)
                        max_xy = points_2d.max(axis=0)
                        bbox_2d = [min_xy[0], min_xy[1], max_xy[0], max_xy[1]]
                
                # image-space bbox
                bbox_img = self._get_obs_bbox_xyxy(local_obs, img_width, img_height)
                
                # debug if no bbox_img
                if bbox_img is None:
                    has_xyxy = hasattr(local_obs, "xyxy")
                    xyxy_val = getattr(local_obs, "xyxy", None)
                    if xyxy_val is not None:
                        import numpy as np
                        xyxy_arr = np.asarray(xyxy_val)
                        xyxy_size = xyxy_arr.size
                        xyxy_shape = xyxy_arr.shape
                    else:
                        xyxy_size = 0
                        xyxy_shape = None
                    logger.warning(f"[Core][NewInstance] frame {frame_id} class {class_name} no bbox_img: has_xyxy={has_xyxy}, xyxy_size={xyxy_size}, xyxy_shape={xyxy_shape}")
                
                if bbox_2d or bbox_img:  # need at least one bbox
                    new_instances_this_frame.append({
                        "class_name": class_name,
                        "bbox_2d": bbox_2d,
                        "bbox_img": bbox_img,
                        "local_obs": local_obs
                    })
        
        logger.warning(f"[Core][NewInstance] frame {frame_id} detected {len(new_instances_this_frame)} new instances")
        
        # mark instances active this frame
        active_keys = set()
        
        # tracker: bbox_2d IoU match
        for new_inst in new_instances_this_frame:
            class_name = new_inst["class_name"]
            bbox_2d = new_inst["bbox_2d"]
            bbox_img = new_inst["bbox_img"]
            
            # no bbox_2d -> cannot match
            if bbox_2d is None:
                # new track (no match)
                instance_key = f"inst_{frame_id}_{len(self.new_instance_tracker)}"
                self.new_instance_tracker[instance_key] = {
                    "class_name": class_name,
                    "first_class_name": class_name,
                    "consecutive_count": 1,
                    "first_frame": frame_id,
                    "last_frame": frame_id,
                    "last_bbox_2d": None
                }
                self.new_instance_frames[instance_key] = [{
                    "frame_id": frame_id,
                    "data_input": data_input,
                    "bbox_img": bbox_img,
                    "class_name": class_name,
                    "curr_obs_list": curr_obs_list  # store obs list for viz
                }]
                active_keys.add(instance_key)
                logger.warning(f"[Core][NewInstance] new track {instance_key} (class: {class_name}, no bbox_2d)")
                continue
            
            # match existing by bbox_2d overlap
            matched_key = None
            max_ratio = 0.0
            threshold = 0.3  # overlap threshold
            
            for key, tracker_info in self.new_instance_tracker.items():
                last_bbox_2d = tracker_info.get("last_bbox_2d")
                if last_bbox_2d:
                    # Intersection
                    ratio = self._compute_bbox_intersection_ratio(last_bbox_2d, bbox_2d)
                    if ratio > threshold and ratio > max_ratio:
                        max_ratio = ratio
                        matched_key = key
            
            if matched_key:
                # update streak
                tracker_info = self.new_instance_tracker[matched_key]
                tracker_info["consecutive_count"] += 1
                tracker_info["last_bbox_2d"] = bbox_2d
                tracker_info["last_frame"] = frame_id
                active_keys.add(matched_key)
                # append frame
                self.new_instance_frames[matched_key].append({
                    "frame_id": frame_id,
                    "data_input": data_input,
                    "bbox_img": bbox_img,
                    "class_name": class_name,
                    "curr_obs_list": curr_obs_list  # store obs list for viz
                })
                logger.warning(f"[Core][NewInstance] inst {matched_key} streak {tracker_info['consecutive_count']} frames (class {class_name}, overlap {max_ratio:.3f})")
            else:
                # new track
                instance_key = f"inst_{frame_id}_{len(self.new_instance_tracker)}"
                self.new_instance_tracker[instance_key] = {
                    "class_name": class_name,
                    "first_class_name": class_name,
                    "consecutive_count": 1,
                    "first_frame": frame_id,
                    "last_frame": frame_id,
                    "last_bbox_2d": bbox_2d
                }
                self.new_instance_frames[instance_key] = [{
                    "frame_id": frame_id,
                    "data_input": data_input,
                    "bbox_img": bbox_img,
                    "class_name": class_name,
                    "curr_obs_list": curr_obs_list  # store obs list for viz
                }]
                active_keys.add(instance_key)
                logger.warning(f"[Core][NewInstance] new track {instance_key} (class: {class_name})")
        
        # flush ended tracks
        keys_to_remove = []
        for key, tracker_info in list(self.new_instance_tracker.items()):
            if key not in active_keys:
                # missing this frame -> ended
                if tracker_info.get("should_save", False) or tracker_info["consecutive_count"] >= 1:
                    # save if streak>=1
                    logger.warning(f"[Core][NewInstance] inst {key} ended, streak {tracker_info['consecutive_count']} frames, saving (thresh=1)")
                    self._save_new_instance_frames(key, tracker_info)
                else:
                    logger.warning(f"[Core][NewInstance] inst {key} ended, streak {tracker_info['consecutive_count']} frames, below save thresh")
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in self.new_instance_tracker:
                del self.new_instance_tracker[key]
            if key in self.new_instance_frames:
                del self.new_instance_frames[key]

    def _process_new_instances_nav(self, global_obs_list: List, curr_obs_list: List, data_input: DataInput, frame_id: int):
        """Nav new-instance path: use local obs with matched_obj_idx==-1 (global obs may be empty before stable)"""
        logger.warning(f"[Core][NewInstance] ========== _process_new_instances_nav frame {frame_id} ==========")
        
        if not hasattr(self, "new_instance_tracker"):
            logger.warning(f"[Core][NewInstance] new_instance_tracker missing, skip")
            return
        
        logger.warning(f"[Core][NewInstance] processing frame {frame_id},global obs: {len(global_obs_list)}, local obs count: {len(curr_obs_list)}")
        
        # debug: matched_obj_idx hist
        local_matched_stats = {}
        for local_obs in curr_obs_list:
            matched_idx = getattr(local_obs, "matched_obj_idx", -1)
            local_matched_stats[matched_idx] = local_matched_stats.get(matched_idx, 0) + 1
        logger.warning(f"[Core][NewInstance] local matched_obj_idx stats: {local_matched_stats}")
        
        import numpy as np
        
        # Use matched_obj_idx==-1 locals (global list may be empty)
        new_instances_this_frame = []
        img_height, img_width = data_input.color.shape[:2]
        
        local_new_count = 0
        local_no_bbox2d_count = 0
        local_no_class_count = 0
        
        # Use matched_obj_idx==-1 locals (global list may be empty)
        for local_obs in curr_obs_list:
            matched_idx = getattr(local_obs, "matched_obj_idx", -1)
            if matched_idx == -1:  # only if matched_obj_idx==-1
                local_new_count += 1
                
                # pcdbbox_2d from xy projection
                bbox_2d = None
                if hasattr(local_obs, "pcd") and len(local_obs.pcd.points) > 0:
                    import open3d as o3d
                    points_2d = np.asarray(local_obs.pcd.points)[:, :2]  # xy only
                    if len(points_2d) > 0:
                        min_xy = points_2d.min(axis=0)
                        max_xy = points_2d.max(axis=0)
                        bbox_2d = [min_xy[0], min_xy[1], max_xy[0], max_xy[1]]
                
                if bbox_2d is None:
                    local_no_bbox2d_count += 1
                    logger.warning(f"[Core][NewInstance] matched_obj_idx==-1 but no pcd; skip bbox_2d")
                    continue
                
                class_name = self._get_class_name_from_obs(local_obs)
                class_id = getattr(local_obs, "class_id", None)
                
                # check unknown_class_id (SAM/FastSAM)
                is_unknown_class = False
                if hasattr(self.detector, "unknown_class_id") and class_id is not None:
                    try:
                        class_idx = int(class_id)
                        if class_idx == self.detector.unknown_class_id:
                            is_unknown_class = True
                            class_name = "unknown"
                    except (TypeError, ValueError):
                        pass
                
                # if no class_name but mask -> SAM unknown
                if class_name is None:
                    if hasattr(local_obs, "mask") and local_obs.mask is not None:
                        class_name = "unknown"
                        is_unknown_class = True
                    else:
                        local_no_class_count += 1
                        logger.warning(f"[Core][NewInstance] matched_obj_idx==-1 no class/mask; skip (class_id={class_id})")
                        continue  # skip: no class and no mask
                
                # empty class_name -> unknown
                if not class_name:
                    class_name = "unknown"
                    is_unknown_class = True
                
                # debug: log unknown detections
                if is_unknown_class or class_name == "unknown":
                    logger.warning(f"[Core][NewInstance] unknown-class instance: class_id={class_id}, class_name={class_name}, matched_obj_idx={matched_idx}")
                
                # image-space bbox
                bbox_img = self._get_obs_bbox_xyxy(local_obs, img_width, img_height)
                
                if bbox_2d:
                    new_instances_this_frame.append({
                        "class_name": class_name,
                        "bbox_2d": bbox_2d,
                        "bbox_img": bbox_img,
                        "obs": local_obs
                    })
                    logger.warning(f"[Core][NewInstance] new inst: class={class_name}, bbox_2d={bbox_2d}, bbox_img={'yes' if bbox_img else 'no'}")
        
        logger.warning(f"[Core][NewInstance] frame {frame_id} stats: matched_obj_idx==-1 count={local_new_count}, "
                      f"no_bbox2d={local_no_bbox2d_count}, no_class={local_no_class_count}, "
                      f"final new inst count={len(new_instances_this_frame)}")
        
        # if none, print diagnostics
        if len(new_instances_this_frame) == 0 and local_new_count > 0:
            logger.error(f"[Core][NewInstance] WARN:  {local_new_count}  locals with matched_obj_idx==-1 but zero new inst")
            logger.error(f"[Core][NewInstance] likely: no bbox_2d ({local_no_bbox2d_count}) or no class ({local_no_class_count})")
        elif len(new_instances_this_frame) == 0:
            logger.warning(f"[Core][NewInstance] frame {frame_id}: no new inst (matched_obj_idx==-1 count=0)")
        
        # mark instances active this frame
        active_keys = set()
        
        # tracker: bbox_2d IoU match
        for new_inst in new_instances_this_frame:
            class_name = new_inst["class_name"]
            bbox_2d = new_inst["bbox_2d"]
            bbox_img = new_inst["bbox_img"]
            
            # match existing by bbox_2d overlap
            matched_key = None
            max_ratio = 0.0
            threshold = 0.3  # looser than main 0.8
            
            for key, tracker_info in self.new_instance_tracker.items():
                last_bbox_2d = tracker_info.get("last_bbox_2d")
                if last_bbox_2d:
                    # Intersection( compute_match_by_intersection_ratio )
                    ratio = self._compute_bbox_intersection_ratio(last_bbox_2d, bbox_2d)
                    if ratio > threshold and ratio > max_ratio:
                        max_ratio = ratio
                        matched_key = key
            
            if matched_key:
                # update streak
                tracker_info = self.new_instance_tracker[matched_key]
                tracker_info["consecutive_count"] += 1
                tracker_info["last_bbox_2d"] = bbox_2d
                tracker_info["last_frame"] = frame_id
                active_keys.add(matched_key)
                # append frame
                self.new_instance_frames[matched_key].append({
                    "frame_id": frame_id,
                    "data_input": data_input,
                    "bbox_img": bbox_img,
                    "class_name": class_name,
                    "curr_obs_list": curr_obs_list,  # store obs list for viz
                    "confidence": getattr(new_inst["obs"], "confidence", 0.0) if "obs" in new_inst else 0.0
                })
                logger.info(f"[Core][NewInstance] inst {matched_key} streak {tracker_info['consecutive_count']} frames (class {class_name}, overlap {max_ratio:.3f})")
            else:
                # new track
                instance_key = f"inst_{frame_id}_{len(self.new_instance_tracker)}"
                self.new_instance_tracker[instance_key] = {
                    "class_name": class_name,
                    "first_class_name": class_name,  # first-seen class
                    "consecutive_count": 1,
                    "first_frame": frame_id,
                    "last_frame": frame_id,
                    "last_bbox_2d": bbox_2d
                }
                self.new_instance_frames[instance_key] = [{
                    "frame_id": frame_id,
                    "data_input": data_input,
                    "bbox_img": bbox_img,
                    "class_name": class_name,
                    "curr_obs_list": curr_obs_list,  # store obs list for viz
                    "confidence": getattr(new_inst["obs"], "confidence", 0.0) if "obs" in new_inst else 0.0
                }]
                active_keys.add(instance_key)
                logger.info(f"[Core][NewInstance] new track {instance_key} (class: {class_name})")
        
        # flush ended tracks
        keys_to_remove = []
        for key, tracker_info in list(self.new_instance_tracker.items()):
            if key not in active_keys:
                # missing this frame -> ended
                if tracker_info.get("should_save", False) or tracker_info["consecutive_count"] >= 1:
                    # save if streak>=1
                    logger.info(f"[Core][NewInstance] inst {key} ended, streak {tracker_info['consecutive_count']} frames, saving (thresh=1)")
                    self._save_new_instance_frames(key, tracker_info)
                else:
                    logger.info(f"[Core][NewInstance] inst {key} ended, streak {tracker_info['consecutive_count']} frames, below save thresh")
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in self.new_instance_tracker:
                del self.new_instance_tracker[key]
            if key in self.new_instance_frames:
                del self.new_instance_frames[key]

    def _draw_all_bboxes_on_image(self, image: np.ndarray, curr_obs_list: List, 
                                  highlight_bbox: Optional[Tuple[int, int, int, int]] = None,
                                  highlight_class: Optional[str] = None) -> np.ndarray:
        """draw all boxes (RGB in)"""
        import cv2
        import numpy as np
        
        # RGB->BGR for cv2
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[2] == 3 else image
        img_height, img_width = img_bgr.shape[:2]
        
        # draw each obs bbox
        for obs in curr_obs_list:
            bbox = self._get_obs_bbox_xyxy(obs, img_width, img_height)
            if bbox is None:
                continue
            
            class_name = self._get_class_name_from_obs(obs)
            if not class_name:
                class_name = "unknown"
            
            # highlight color if selected
            is_highlight = (highlight_bbox is not None and 
                          bbox[0] == highlight_bbox[0] and bbox[1] == highlight_bbox[1] and
                          bbox[2] == highlight_bbox[2] and bbox[3] == highlight_bbox[3])
            
            if is_highlight:
                color = (0, 255, 0)  # green BGR highlight
            else:
                color = (255, 0, 0)  # blue BGR others
            
            img_bgr = self._draw_bbox_on_image(img_bgr, bbox, class_name, color)
        
        # BGR->RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr.shape[2] == 3 else img_bgr
        return img_rgb

    def _save_new_instance_frames(self, instance_key: str, tracker_info: Dict):
        """Save frames per instance folder (keyframes only)"""
        if instance_key not in self.new_instance_frames:
            logger.warning(f"[Core][NewInstance] inst {instance_key} has no frame data")
            return
        
        class_name = tracker_info.get("first_class_name", tracker_info.get("class_name", "unknown"))
        
        # ,
        if not hasattr(self, "_saved_instance_count"):
            self._saved_instance_count = {}
        
        if class_name not in self._saved_instance_count:
            self._saved_instance_count[class_name] = 0
        
        self._saved_instance_count[class_name] += 1
        instance_suffix = self._saved_instance_count[class_name]
        
        # (:class_name_suffix)
        instance_dir_name = f"{class_name}_{instance_suffix}"
        instance_dir = os.path.join(self.new_instance_save_dir, instance_dir_name)
        os.makedirs(instance_dir, exist_ok=True)
        
        import cv2
        import numpy as np
        
        frames_data = self.new_instance_frames[instance_key]
        
        best_frame_data = None
        max_confidence = -1.0
        
        for frame_data in frames_data:
            conf = frame_data.get("confidence", 0.0)
            if conf > max_confidence:
                max_confidence = conf
                best_frame_data = frame_data
        
        if best_frame_data:
            frames_to_save = [best_frame_data]
            logger.warning(f"[Core][NewInstance] saving best frame for {instance_key} (conf={max_confidence:.4f}) -> {instance_dir}")
        else:
            frames_to_save = frames_data # Fallback: save all if no best found (should not happen)
            logger.warning(f"[Core][NewInstance] saving {len(frames_data)} frames for {instance_key} -> {instance_dir} (no best frame)")
        
        saved_count = 0
        for frame_data in frames_to_save:
            frame_id = frame_data["frame_id"]
            data_input = frame_data["data_input"]
            bbox_img = frame_data.get("bbox_img")
            frame_class_name = frame_data.get("class_name", class_name)
            
            # (RGB)
            img = data_input.color.copy()
            if len(img.shape) == 3 and img.shape[2] == 3:
                # uint8
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                
                curr_obs_list = frame_data.get("curr_obs_list", [])
                
                # bbox,
                # BGR, _draw_bbox_on_image BGR
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if (len(img.shape) == 3 and img.shape[2] == 3) else img
                
                # clean
                img_bgr_clean = img_bgr.copy()

                # 1.  ()
                clean_save_path = os.path.join(instance_dir, f"frame_{frame_id:06d}_clean.jpg")
                cv2.imwrite(clean_save_path, img_bgr_clean)

                if bbox_img is not None:
                    # 2.  Crop 
                    try:
                        x1, y1, x2, y2 = map(int, bbox_img)
                        h, w = img_bgr_clean.shape[:2]
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))
                        
                        if x2 > x1 and y2 > y1:
                            crop_bgr = img_bgr_clean[y1:y2, x1:x2]
                            crop_save_path = os.path.join(instance_dir, f"frame_{frame_id:06d}_crop.jpg")
                            cv2.imwrite(crop_save_path, crop_bgr)

                            # 3.  CLIP 
                            #  memory_cluster_manager  clip_feature_fn
                            if self.memory_cluster_manager is not None and hasattr(self.memory_cluster_manager, "clip_feature_fn"):
                                # CLIP  RGB 
                                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                                try:
                                    clip_ft = self.memory_cluster_manager.clip_feature_fn(crop_rgb)
                                    if clip_ft is not None:
                                        ft_save_path = os.path.join(instance_dir, f"frame_{frame_id:06d}_clip.npy")
                                        np.save(ft_save_path, clip_ft)
                                except Exception as e:
                                    logger.warning(f"[Core][NewInstance] CLIP feature extraction failed: {e}")
                    except Exception as e:
                         logger.warning(f"[Core][NewInstance] Crop/CLIP processing failed: {e}")

                    # bbox()
                    img_bgr = self._draw_bbox_on_image(img_bgr, bbox_img, frame_class_name, (0, 255, 0))
                
                # 4.  ()
                if hasattr(data_input, "pose") and data_input.pose is not None:
                    pose_save_path = os.path.join(instance_dir, f"frame_{frame_id:06d}_pose.npy")
                    np.save(pose_save_path, data_input.pose)

                #  ()
                save_path = os.path.join(instance_dir, f"frame_{frame_id:06d}.jpg")
                cv2.imwrite(save_path, img_bgr)
                saved_count += 1
            else:
                logger.warning(f"[Core][NewInstance] frame {frame_id} bad image format, skip")
        
        logger.warning(
            f"[Core][NewInstance] saved new instance '{class_name}' ({instance_suffix}) best frame ({saved_count}/{len(frames_data)}) -> {instance_dir}"
        )

    def get_total_memory_by_keyword(self, keyword="applications"):
        total_rss = 0
        for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
            try:
                cmdline = proc.info.get("cmdline")
                if isinstance(cmdline, list) and keyword in " ".join(cmdline):
                    total_rss += proc.info["memory_info"].rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                continue
        return total_rss / 1024 / 1024  # MB

    def sequential_process(self, data_input: DataInput):
        """
        Process input data sequentially.
        Note: Sequential processing is not recommended for large datasets.
        It is primarily used for debugging and testing purposes.
        Also, Sequential processing mode does not support navigation.
        """

        # Get current frame id
        self.curr_frame_id = data_input.idx

        # Get current pose
        self.curr_pose = data_input.pose

        # Set time sequence for visualizer
        self.visualizer.set_time_sequence("frame", self.curr_frame_id)

        # Set current camera information & image
        self.visualizer.set_camera_info(data_input.intrinsics, data_input.pose)
        self.visualizer.set_image(data_input.color)

        self.detector.set_data_input(data_input)

        using_cached_observations = False
        curr_obs_list = None
        if self.observation_cache is not None:
            cached_obs = self.observation_cache.load(self.curr_frame_id)
            if cached_obs is not None:
                curr_obs_list = cached_obs
                using_cached_observations = True
                self.detector.curr_observations = curr_obs_list
                logger.info(
                    "[Core][ObsCache] frame %d using cached LocalObservation (count=%d)",
                    self.curr_frame_id,
                    len(curr_obs_list),
                )

        if not using_cached_observations:
            # Detection process with timing
            with timing_context("Detection Process", self):
                if self.cfg.run_detection:
                    self.detector.process_detections()
                    with timing_context("Save Detection", self):
                        if self.cfg.save_detection:
                            self.detector.save_detection_results()
                else:
                    self.detector.load_detection_results()

                # Detection results -> objects observation
                with timing_context("Vis Detection", self):
                    self.detector.calculate_observations()
                    if self.cfg.use_rerun:
                        self.detector.visualize_detection()

            curr_obs_list = self.detector.get_curr_observations()
            if self.observation_cache is not None and curr_obs_list is not None:
                self.observation_cache.save(self.curr_frame_id, curr_obs_list)

        #  / 
        if curr_obs_list is None:
            curr_obs_list = self.detector.get_curr_observations()
        novelty_info = None
        #  (  )
        lifelong_nav = False
        if hasattr(self.cfg, "navigation") and hasattr(self.cfg.navigation, "lifelong_navigation"):
            lifelong_nav = self.cfg.navigation.lifelong_navigation

        if self.memory_cluster_manager is not None and (not self.cfg.get("preload_global_map", False) or lifelong_nav):
            with timing_context("Memory Cluster", self):
                self._update_memory_clusters(data_input, curr_obs_list, has_new_instance=False)
        if self.novelty_detector is not None and hasattr(self.detector, 'clip_model') and hasattr(self.detector, 'clip_preprocess'):
            with timing_context("Novelty Detection", self):
                is_novel_frame, novelty_info = self.novelty_detector.check_novelty(
                    data_input.color,
                    data_input.pose,
                    self.detector.clip_model,
                    self.detector.clip_preprocess,
                    self.cfg.device
                )
                if not is_novel_frame:
                    logger.info(
                        f"[Core] frame {self.curr_frame_id} repeat, "
                        f"clip_sim={novelty_info.get('clip_similarity', 0):.3f}, "
                        f"repeat_buf={novelty_info.get('repeat_frames_count', 0)}"
                    )

        # Local Mapping with timing
        with timing_context("Local Mapping", self):
            self.detector.update_state()
            self.detector.update_data()
            self.local_map_manager.set_curr_idx(self.curr_frame_id)
            self.local_map_manager.set_curr_data(data_input)
            # local_map_manager()
            if novelty_info is not None:
                # ,
                pass
            self.local_map_manager.process_observations(curr_obs_list)

        #  ( Local Mapping )
        # ,,matched_obj_idx
        # ,
        lifelong_nav = False
        if hasattr(self.cfg, "navigation") and hasattr(self.cfg.navigation, "lifelong_navigation"):
            lifelong_nav = self.cfg.navigation.lifelong_navigation

        if self.memory_cluster_manager is not None and (not self.cfg.get("preload_global_map", False) or lifelong_nav):
            with timing_context("Memory Cluster", self):
                self._update_memory_clusters(data_input, curr_obs_list, has_new_instance=False)

        # Get global observations
        with timing_context("Global Mapping", self):
            global_obs_list = self.local_map_manager.get_global_observations()
            self.local_map_manager.clear_global_observations()
            self.global_map_manager.process_observations(global_obs_list)

    def parallel_process(self, data_input: DataInput):
        """
        Process input data in parallel.
        """
        # Get current frame id
        self.curr_frame_id = data_input.idx

        # Get current pose
        self.curr_pose = data_input.pose

        self.detector.set_data_input(data_input)

        using_cached_observations = False
        curr_obs_list = None
        if self.observation_cache is not None:
            cached_obs = self.observation_cache.load(self.curr_frame_id)
            if cached_obs is not None:
                curr_obs_list = cached_obs
                using_cached_observations = True
                self.detector.curr_observations = curr_obs_list
                logger.info(
                    "[Core][ObsCache] (parallel) frame %d cached LocalObservation (count=%d)",
                    self.curr_frame_id,
                    len(curr_obs_list),
                )

        # Detection process
        start_time = time.time()
        with timing_context("Observation Generation", self):
            novelty_info = None
            if not using_cached_observations:
                with timing_context("Process Detection", self):
                    if self.cfg.run_detection:
                        # Run detection
                        self.detector.process_detections()
                        with timing_context("Save Detection", self):
                            if self.cfg.save_detection:
                                self.detector.save_detection_results()
                    else:
                        self.detector.load_detection_results()

                with timing_context("Observation Formatting", self):
                    self.detector.calculate_observations()

                curr_obs_list = self.detector.get_curr_observations()
                if self.observation_cache is not None and curr_obs_list is not None:
                    self.observation_cache.save(self.curr_frame_id, curr_obs_list)

            if curr_obs_list is None:
                curr_obs_list = []

            #  ( Local Mapping )
            # ,,matched_obj_idx
            # :memory_cluster_manager,(preload_global_map=True)
            # if self.memory_cluster_manager is not None:
            #     with timing_context("Memory Cluster", self):
            #         self._update_memory_clusters(data_input, curr_obs_list)

            # memory cluster update moved after local mapping
            if self.novelty_detector is not None and hasattr(self.detector, 'clip_model') and hasattr(self.detector, 'clip_preprocess'):
                with timing_context("Novelty Detection", self):
                    is_novel_frame, novelty_info = self.novelty_detector.check_novelty(
                        data_input.color,
                        data_input.pose,
                        self.detector.clip_model,
                        self.detector.clip_preprocess,
                        self.cfg.device
                    )
                    if not is_novel_frame:
                        logger.info(
                            f"[Core] frame {self.curr_frame_id} repeat, "
                            f"clip_sim={novelty_info.get('clip_similarity', 0):.3f}, "
                            f"repeat_buf={novelty_info.get('repeat_frames_count', 0)}"
                        )

            self.detector.update_state()
            
            # :mapping,
            # (matched_obj_idx == -1)

            try:
                # DataInput
                self.detection_results_queue.put(
                    (curr_obs_list, self.curr_frame_id, novelty_info, data_input), timeout=1
                )
            except queue.Full:
                logger.warning(
                    f"[Core] Mapping queue is full, skipping frame {self.curr_frame_id}."
                )

        end_time = time.time()

        # Set time sequence for visualizer
        self.visualizer.set_time_sequence("frame", self.curr_frame_id)

        # Set current camera information
        self.visualizer.set_camera_info(data_input.intrinsics, data_input.pose)
        self.visualizer.set_image(data_input.color)
        
        # Check if rotation is completed and perform VLM check
        # ,()
        if self.curr_frame_id % 10 == 0 or self.rotation_completed:
            logger.warning(
                f"[Core] [PostRotation] 🔍 CHECK: "
                f"nav_best_frame_path={'SET' if self.nav_best_frame_path is not None else 'None'}, "
                f"post_rotation_vlm_check_done={self.post_rotation_vlm_check_done}, "
                f"rotation_completed={self.rotation_completed}, Frame={self.curr_frame_id}"
            )
        
        # ,,append frameVLM()
        if self.rotation_completed and not self.post_rotation_vlm_check_done:
            if self.nav_best_frame_path is not None:
                logger.warning(f"[Core] ✅✅✅ ROTATION COMPLETED! nav_best_frame_path is SET! Triggering VLM check NOW! (Frame: {self.curr_frame_id})")
                logger.warning(f"[Core] [PostRotation] nav_best_frame_path={self.nav_best_frame_path}")
                logger.warning(f"[Core] [PostRotation] nav_inquiry_sentence={self.nav_inquiry_sentence}")
                self._check_target_after_rotation(data_input)
            else:
                # ,(fallback)
                logger.warning(f"[Core] ✅✅✅ ROTATION COMPLETED! But nav_best_frame_path is None!")
                logger.warning(f"[Core] [PostRotation] NavRunner may lack best_frame_path; using current frame")
                logger.warning(f"[Core] [PostRotation] append frame + VLM (current frame ref)")
                # (fallback)
                if self.nav_inquiry_sentence:
                    # append frame""()
                    #  NavRunner :parent(map_save_path)/post_rotation_frames
                    scene_dir = os.path.dirname(self.cfg.map_save_path)
                    current_frame_dir = os.path.join(scene_dir, "post_rotation_frames")
                    current_frame_dir = os.path.abspath(current_frame_dir)
                    os.makedirs(current_frame_dir, exist_ok=True)
                    fallback_best_frame_path = os.path.join(current_frame_dir, f"fallback_best_frame_{self.curr_frame_id:06d}.jpg")
                    
                    # append frame
                    import cv2
                    img = data_input.color.copy()
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        if img.dtype != np.uint8:
                            img = np.clip(img, 0, 255).astype(np.uint8)
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(fallback_best_frame_path, img_bgr)
                        logger.warning(f"[Core] [PostRotation] Saved fallback best frame to: {fallback_best_frame_path}")
                        
                        # nav_best_frame_pathVLM
                        self.nav_best_frame_path = fallback_best_frame_path
                        self._check_target_after_rotation(data_input)
                    else:
                        logger.error(f"[Core] [PostRotation] Invalid image format, cannot save fallback frame")
                else:
                    logger.warning(f"[Core] [PostRotation] nav_inquiry_sentence is None, cannot perform VLM check")

        if self.cfg.use_rerun:
            elapsed_time = end_time - start_time
            self.detector.visualize_time(elapsed_time)

            # TODO: psutil seems not that correct
            # mem_usage = self.get_total_memory_by_keyword()
            # self.detector.visualize_memory(mem_usage)

        if self.calculate_path and self.global_map_manager.has_global_map():
            logger.info(f"[Core] Global Navigation enabled! calculate_path={self.calculate_path}, inquiry='{self.inquiry}'")

            self.global_map_manager.has_action_path = False

            # calculate the inquiry sentence to clip feat
            if not self.inquiry:
                logger.warning("[Core] calculate_path=True but inquiry is empty! Please set inquiry_sentence in actions.yaml")
            self.inquiry_feat = self.convert_inquiry_to_feat(self.inquiry)

            # set the global inquiry sentence to global map manager
            self.global_map_manager.inquiry = self.inquiry_feat

            # Get Current layout information from detector
            layout_pcd = self.detector.get_layout_pointcloud()
            self.global_map_manager.set_layout_info(layout_pcd)

            # ---  anchor_task_runner_nav  ---
            #  inquiry 
            if self.inquiry:
                logger.info(f"[Core] nav-mode VLM task: {self.inquiry}")
                self._ensure_vlm_prompt_transcript_session()
                self._nav_timing_reset_session()

                # [Fix] ,
                self._clear_new_instance_dir()
                
                import subprocess
                import json
                
                viz_mode = getattr(self.cfg, "visualize_mode", "auto")
                
                # Top-K
                top_k = 3
                if hasattr(self.cfg, "navigation") and hasattr(self.cfg.navigation, "top_k_anchors"):
                    top_k = self.cfg.navigation.top_k_anchors

                # : memory_cluster  keyframe_dir, representative_frames_path / output 
                kf_root = self._navigation_keyframe_dir()
                manifest_path = os.path.join(kf_root, "anchor_key", "manifest.json")
                anchor_file_path = os.path.join(kf_root, "anchor_key", "anchor_attached_objects.txt")
                logger.warning(f"[Core] [LifelongNav] VLM Search using manifest: {manifest_path}")

                runner_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "anchor_task_runner_nav.py")
                llm_model = getattr(self.cfg, "llm_model", "gpt-5.2")
                vlm_model = getattr(self.cfg, "vlm_model", "gemini-3-pro-preview")
                gemini_key = getattr(self.cfg, "gemini_api_key", None) or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
                cmd = [
                    sys.executable,
                    runner_script,
                    self.inquiry,
                    "--manifest", manifest_path,
                    "--anchor-file", anchor_file_path,
                    "--model", llm_model,
                    "--vlm-model", vlm_model,
                    "--visualize",
                    "--viz-mode", viz_mode,
                    "--top-k", str(top_k),
                ]
                if gemini_key:
                    cmd.extend(["--gemini-api-key", gemini_key])
                if getattr(self, "_vlm_prompt_transcript_path", None):
                    cmd.extend(["--prompt-transcript", self._vlm_prompt_transcript_path])
                
                subprocess_wall_ms = 0.0
                try:
                    logger.warning(f"[Core] Executing: {' '.join(cmd)}")
                    
                    env = os.environ.copy()
                    env["QT_QPA_PLATFORM"] = "offscreen"
                    if getattr(self, "_vlm_prompt_transcript_path", None):
                        from utils.vlm_prompt_transcript import ENV_KEY
                        env[ENV_KEY] = self._vlm_prompt_transcript_path
                    
                    _t_sub = time.perf_counter()
                    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
                    subprocess_wall_ms = (time.perf_counter() - _t_sub) * 1000.0
                    
                    if result.stdout:
                        logger.warning("[Core] ========== NavRunner Output Start ==========")
                        # VLM
                        lines = result.stdout.split('\n')
                        vlm_reasoning_found = False
                        vlm_selection_found = False
                        for line in lines:
                            if line.strip():
                                logger.warning(f"[Core] [NavRunner] {line}")
                                # VLM reasoning
                                if "[NavRunner] VLM Reasoning:" in line:
                                    vlm_reasoning_found = True
                                    reasoning = line.split("[NavRunner] VLM Reasoning:")[-1].strip()
                                    logger.warning(f"[Core] [VLM] Reasoning: {reasoning}")
                                # VLM selection response
                                if "[NavRunner] Response:" in line and "VLM" not in line:
                                    vlm_selection_found = True
                                    response = line.split("[NavRunner] Response:")[-1].strip()
                                    logger.warning(f"[Core] [VLM] Selection Response: {response}")
                                # selected anchors
                                if "[NavRunner] VLM selected anchors to view:" in line:
                                    anchors = line.split("[NavRunner] VLM selected anchors to view:")[-1].strip()
                                    logger.warning(f"[Core] [VLM] Selected Anchors: {anchors}")
                        logger.warning("[Core] ========== NavRunner Output End ==========")
                    
                    if result.stderr and result.returncode != 0:
                        logger.warning("[Core] ========== NavRunner Error ==========")
                        for line in result.stderr.split('\n'):
                            if line.strip():
                                logger.warning(f"[Core] [NavRunner ERR] {line}")
                        logger.warning("[Core] ======================================")

                    if result.returncode == 0:
                        logger.warning("[Core] ✅✅✅ NavRunner completed successfully (returncode=0)!")
                        
                        # JSON
                        target_frame_path = None
                        target_anchor_uid = None
                        frame_idx = None
                        nav_runner_timing_from_json = None
                        navigation_path_planning_ms = 0.0
                        
                        #  stdout  Result JSON( _parse_nav_runner_result_from_stdout )
                        lines = result.stdout.split("\n")
                        result_json = _parse_nav_runner_result_from_stdout(result.stdout)
                        if result_json is not None:
                                nav_runner_timing_from_json = result_json.get("nav_retrieval_timing_ms")
                                
                                # VLM
                                logger.warning(f"[Core] [VLM] Full Result JSON: {json.dumps(result_json, indent=2)}")
                                
                                if result_json.get("success"):
                                    target_frame_path = result_json.get("best_frame_path")
                                    target_anchor_uid = result_json.get("target_anchor_uid")
                                    frame_idx = result_json.get("frame_idx")
                                    method = result_json.get("method", "Unknown")
                                    
                                    logger.warning(f"[Core] [VLM] NavRunner Result: success=True, method={method}, frame_idx={frame_idx}, anchor_uid={target_anchor_uid}")
                                    logger.warning(f"[Core] [VLM] Target Frame Path (raw): {target_frame_path}")
                                    logger.warning(f"[Core] [VLM] Target Anchor UID: {target_anchor_uid}")
                                    
                                    # [CLIP Fallback Fix]
                                    #  CLIP ,NavRunner ()
                                    #  MemoryCluster  UID 
                                    if method == "CLIP" and target_anchor_uid and self.memory_cluster_manager:
                                        logger.warning(f"[Core] [VLM] Method is CLIP. Refining best view pose for object {target_anchor_uid}...")
                                        
                                        # DEBUG: 
                                        total_obs_count = 0
                                        for c in self.memory_cluster_manager.clusters:
                                            if c.get("representative") and hasattr(c["representative"], "valid_observations"):
                                                total_obs_count += len(c["representative"].valid_observations)
                                        logger.warning(f"[Core] [VLM] DEBUG: Total valid_observations in memory: {total_obs_count}")
                                        
                                        refined_pose = None
                                        refined_frame_idx = None
                                        refined_score = -1.0
                                        
                                        # 1.  clusters 
                                        for cluster in self.memory_cluster_manager.clusters:
                                            rep = cluster.get("representative")
                                            if rep:
                                                #  UID
                                                obs_list = getattr(rep, "valid_observations", [])
                                                contains_target = False
                                                for obs in obs_list:
                                                    if str(getattr(obs, "matched_obj_uid", "")) == target_anchor_uid:
                                                        contains_target = True
                                                        break
                                                
                                                if contains_target:
                                                    score = getattr(rep, "score", 0.0)
                                                    if score > refined_score:
                                                        refined_score = score
                                                        refined_pose = rep.pose
                                                        refined_frame_idx = rep.frame_idx
                                                        
                                        # 2. ,()
                                        if refined_pose is None:
                                            for cluster in self.memory_cluster_manager.clusters:
                                                frames = cluster.get("frames", [])
                                                for frame in frames:
                                                    obs_list = getattr(frame, "valid_observations", [])
                                                    for obs in obs_list:
                                                        if str(getattr(obs, "matched_obj_uid", "")) == target_anchor_uid:
                                                            if frame.pose is not None:
                                                                refined_pose = frame.pose
                                                                refined_frame_idx = frame.frame_idx
                                                                logger.warning(f"[Core] [VLM] Found target in non-representative frame {refined_frame_idx}")
                                                                break
                                                    if refined_pose is not None: break
                                                if refined_pose is not None: break
                                        
                                        if refined_pose is not None:
                                            logger.warning(f"[Core] [VLM] ✅ Refined target pose using MemoryCluster! Frame: {refined_frame_idx}")
                                            #  frame_idx, frame_idx  pose
                                            frame_idx = refined_frame_idx
                                            
                                            #  frame_idx 
                                            #  keyframe_dir 
                                            found_new_path = None
                                            if hasattr(self.memory_cluster_manager, 'keyframe_dir'):
                                                kf_dir = self.memory_cluster_manager.keyframe_dir
                                                if os.path.exists(kf_dir):
                                                    for fname in os.listdir(kf_dir):
                                                        if f"frame_{frame_idx:06d}" in fname and fname.endswith(".png"):
                                                            found_new_path = os.path.join(kf_dir, fname)
                                                            break
                                            
                                            if found_new_path:
                                                target_frame_path = found_new_path
                                                logger.warning(f"[Core] [VLM] ✅ Found image for refined frame: {target_frame_path}")
                                            else:
                                                # (),, dataset 
                                                target_frame_path = None 
                                                logger.warning(f"[Core] [VLM] ⚠️ Could not find image file for refined frame {frame_idx}")
                                        else:
                                            logger.warning(f"[Core] [VLM] ⚠️ Could not find target {target_anchor_uid} in any memory cluster frame. Using NavRunner result.")

                                    #  best_frame_path:
                                    # NavRunner  best_frame_path ( image_paths )
                                    # ,
                                    if target_frame_path:
                                        original_path = target_frame_path
                                        
                                        # ,
                                        if not os.path.exists(target_frame_path):
                                            logger.warning(f"[Core] [VLM] ⚠️ best_frame_path does not exist: {target_frame_path}")
                                            logger.warning(f"[Core] [VLM] Attempting to find it in multiple locations...")
                                            
                                            candidate_paths = []
                                            
                                            # 1.  representative_frames 
                                            if self.memory_cluster_manager and hasattr(self.memory_cluster_manager, 'keyframe_dir'):
                                                keyframe_dir = self.memory_cluster_manager.keyframe_dir
                                                candidate_paths.extend([
                                                    os.path.join(keyframe_dir, target_frame_path),
                                                    os.path.join(keyframe_dir, os.path.basename(target_frame_path)),
                                                ])
                                            
                                            # 2.  nav_visualizations 
                                            nav_viz_dir = os.path.join(self.cfg.output_path, "nav_visualizations")
                                            candidate_paths.extend([
                                                os.path.join(nav_viz_dir, target_frame_path),
                                                os.path.join(nav_viz_dir, os.path.basename(target_frame_path)),
                                            ])
                                            
                                            # 3.  representative_frames 
                                            output_rep_frames_dir = os.path.join(self.cfg.output_path, "representative_frames")
                                            candidate_paths.extend([
                                                os.path.join(output_rep_frames_dir, target_frame_path),
                                                os.path.join(output_rep_frames_dir, os.path.basename(target_frame_path)),
                                            ])
                                            
                                            found = False
                                            for candidate in candidate_paths:
                                                if os.path.exists(candidate):
                                                    target_frame_path = candidate
                                                    found = True
                                                    logger.warning(f"[Core] [VLM] ✅ Found best_frame_path at: {target_frame_path}")
                                                    break
                                            
                                            if not found:
                                                logger.warning(f"[Core] [VLM] ⚠️ Could not find best_frame_path in any location!")
                                                logger.warning(f"[Core] [VLM] Tried locations: {candidate_paths}")
                                                logger.warning(f"[Core] [VLM] Will use fallback mechanism when rotation completes")
                                                target_frame_path = None
                                        
                                        if target_frame_path and os.path.exists(target_frame_path):
                                            if not os.path.isabs(target_frame_path):
                                                target_frame_path = os.path.abspath(target_frame_path)
                                            logger.warning(f"[Core] [VLM] ✅ Final best_frame_path (absolute): {target_frame_path}")
                                    
                                    # , nav_visualizations 
                                    if target_frame_path and os.path.exists(target_frame_path):
                                        #  nav_visualizations ()
                                        #  NavRunner :parent(map_save_path)/nav_visualizations
                                        scene_dir = os.path.dirname(self.cfg.map_save_path)
                                        nav_viz_dir = os.path.join(scene_dir, "nav_visualizations")
                                        nav_viz_dir = os.path.abspath(nav_viz_dir)
                                        os.makedirs(nav_viz_dir, exist_ok=True)
                                        
                                        #  nav_visualizations 
                                        if not target_frame_path.startswith(nav_viz_dir):
                                            #  nav_visualizations 
                                            best_frame_basename = os.path.basename(target_frame_path)
                                            nav_viz_path = os.path.join(nav_viz_dir, best_frame_basename)
                                            
                                            # 1.  nav_viz_path ()
                                            if not os.path.exists(nav_viz_path):
                                                try:
                                                    shutil.copy2(target_frame_path, nav_viz_path)
                                                    logger.warning(f"[Core] [VLM] ✅ Copied best_frame (with boxes) to nav_visualizations: {nav_viz_path}")
                                                except Exception as e:
                                                    logger.warning(f"[Core] [VLM] ⚠️ Failed to copy best_frame to nav_visualizations: {e}")
                                            
                                            # 2. / CLEAN  nav_viz_clean_path ( PostRotationVLM)
                                            nav_viz_clean_path = nav_viz_path.replace(".png", "_clean.png")
                                            if not os.path.exists(nav_viz_clean_path):
                                                success_save_clean = False
                                                
                                                # A.  dataset  (CLEAN image)
                                                if frame_idx is not None and hasattr(self, 'dataset') and self.dataset is not None:
                                                    try:
                                                        # Dataset index check
                                                        if 0 <= frame_idx < len(self.dataset):
                                                            raw_data_input = self.dataset[frame_idx]
                                                            raw_image = raw_data_input.color
                                                            
                                                            import cv2
                                                            # RGB -> BGR for saving
                                                            if raw_image.shape[2] == 3:
                                                                raw_image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
                                                            else:
                                                                raw_image_bgr = raw_image
                                                                
                                                            cv2.imwrite(nav_viz_clean_path, raw_image_bgr)
                                                            logger.warning(f"[Core] [VLM] ✅ Saved CLEAN frame {frame_idx} (from dataset) to: {nav_viz_clean_path}")
                                                            success_save_clean = True
                                                    except Exception as e:
                                                        logger.warning(f"[Core] [VLM] ⚠️ Failed to load clean frame from dataset: {e}")
                                                
                                                # B.  dataset , _clean.png 
                                                if not success_save_clean:
                                                    clean_src_path = target_frame_path.replace(".png", "_clean.png")
                                                    if os.path.exists(clean_src_path):
                                                        try:
                                                            shutil.copy2(clean_src_path, nav_viz_clean_path)
                                                            logger.warning(f"[Core] [VLM] ✅ Copied CLEAN frame (from disk: {clean_src_path}) to: {nav_viz_clean_path}")
                                                            success_save_clean = True
                                                        except Exception as e:
                                                            logger.warning(f"[Core] [VLM] ⚠️ Failed to copy clean frame from disk: {e}")
                                                
                                                if not success_save_clean:
                                                    logger.warning(f"[Core] [VLM] ⚠️ Could not find or save CLEAN frame. PostRotationVLM will fallback to boxed frame.")
                                            else:
                                                logger.warning(f"[Core] [VLM] ✅ CLEAN frame already exists: {nav_viz_clean_path}")
                                            
                                            #  nav_visualizations  ()
                                            target_frame_path = nav_viz_path
                                    
                                    # query,VLM
                                    self.nav_best_frame_path = target_frame_path
                                    # VLM,query
                                    self.nav_inquiry_sentence = result_json.get("target_visual_description", self.inquiry)
                                    if not self.nav_inquiry_sentence:
                                        self.nav_inquiry_sentence = self.inquiry
                                    
                                    # UID
                                    self.nav_target_uid = result_json.get("final_uid", None)

                                    self.need_post_rotation_vlm_check = True
                                    self.post_rotation_vlm_check_done = False
                                    self.rotation_completed = False
                                    logger.warning(f"[Core] [VLM] ✅✅✅✅✅ SET nav_best_frame_path={self.nav_best_frame_path}!")
                                    logger.warning(f"[Core] [VLM] ✅✅✅✅✅ SET nav_inquiry_sentence={self.nav_inquiry_sentence}!")
                                    logger.warning(f"[Core] [VLM] Stored navigation info for post-rotation check:")
                                    logger.warning(f"[Core] [VLM]   - nav_best_frame_path={self.nav_best_frame_path}")
                                    logger.warning(f"[Core] [VLM]   - nav_inquiry_sentence={self.nav_inquiry_sentence}")
                                    logger.warning(f"[Core] [VLM]   - need_post_rotation_vlm_check={self.need_post_rotation_vlm_check}")
                                    logger.warning(f"[Core] [VLM]   - post_rotation_vlm_check_done={self.post_rotation_vlm_check_done}")
                                    logger.warning(f"[Core] [VLM]   - rotation_completed={self.rotation_completed}")
                                    
                                    if self.nav_best_frame_path and os.path.exists(self.nav_best_frame_path):
                                        logger.warning(f"[Core] [VLM] ✅ Verified: nav_best_frame_path exists: {self.nav_best_frame_path}")
                                    elif self.nav_best_frame_path:
                                        logger.warning(f"[Core] [VLM] ⚠️ nav_best_frame_path does not exist: {self.nav_best_frame_path}")
                                else:
                                    logger.warning(f"[Core] [VLM] NavRunner returned success=False, result: {result_json}")
                                    
                                    # --- STRATEGY 2: Semantic Reasoning Fallback ---
                                    # VLM , LLM 
                                    logger.warning("[Core] [VLM] Initiating Strategy 2: Semantic Reasoning Fallback...")
                                    
                                    try:
                                        # 1.  ID
                                        visited_ids = []
                                        if 'anchors' in locals() and anchors:
                                            # anchors  "['id1', 'id2']"
                                            import ast
                                            try:
                                                visited_ids = ast.literal_eval(anchors)
                                            except:
                                                visited_ids = [anchors] # Fallback
                                        
                                        logger.warning(f"[Core] [Fallback] Visited anchors: {visited_ids}")
                                        
                                        # 2. 
                                        fallback_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "semantic_fallback.py")
                                        
                                        model = getattr(self.cfg, "llm_model", "gpt-5.2")
                                        base_url = getattr(self.cfg, "vlm_base_url", None) or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                                        api_key = (getattr(self.cfg, "vlm_api_key", None) or os.environ.get("OPENAI_API_KEY", ""))
                                        anchor_file = os.path.join(
                                            self._navigation_keyframe_dir(), "anchor_key", "anchor_attached_objects.txt"
                                        )
                                        
                                        fb_cmd = [
                                            sys.executable,
                                            fallback_script,
                                            "--task", self.inquiry,
                                            "--anchor-file", anchor_file,
                                            "--model", model,
                                            "--base-url", base_url,
                                            "--api-key", api_key,
                                            "--visited"
                                        ] + visited_ids
                                        if getattr(self, "_vlm_prompt_transcript_path", None):
                                            fb_cmd.extend(["--prompt-transcript", self._vlm_prompt_transcript_path])
                                        
                                        fb_res = subprocess.run(fb_cmd, capture_output=True, text=True)
                                        
                                        if fb_res.returncode == 0 and fb_res.stdout:
                                            fb_json = json.loads(fb_res.stdout)
                                            target_anchor_id = fb_json.get("target_anchor_id")
                                            reasoning = fb_json.get("reasoning", "")
                                            
                                            logger.warning(f"[Core] [Fallback] LLM Suggestion: {target_anchor_id}")
                                            logger.warning(f"[Core] [Fallback] Reasoning: {reasoning}")
                                            
                                            if target_anchor_id:
                                                # 3. 
                                                target_pose = None
                                                target_frame_idx = -1
                                                
                                                if self.memory_cluster_manager and self.memory_cluster_manager.clusters:
                                                    for cluster in self.memory_cluster_manager.clusters:
                                                        cluster_uid = str(cluster.get("uid", ""))
                                                        #  ID
                                                        if target_anchor_id in cluster_uid or cluster_uid in target_anchor_id:
                                                            rep = cluster.get("representative")
                                                            if rep and rep.pose is not None:
                                                                target_pose = rep.pose
                                                                target_frame_idx = rep.frame_idx
                                                                logger.warning(f"[Core] [Fallback] Found cluster {cluster_uid}. Best frame: {target_frame_idx}")
                                                                break
                                                
                                                if target_pose is not None:
                                                    # 4.  ()
                                                    target_frame_path = f"fallback_semantic_{target_anchor_id}.png"
                                                    target_anchor_uid = target_anchor_id
                                                    frame_idx = target_frame_idx
                                                    
                                                    self.nav_target_uid = target_anchor_uid
                                                    self.nav_inquiry_sentence = self.inquiry
                                                    self.need_post_rotation_vlm_check = True
                                                    self.post_rotation_vlm_check_done = False
                                                    self.rotation_completed = False
                                                    
                                                    goal_pos = target_pose[:3, 3]
                                                    logger.warning(f"[Core] [Fallback] Planning path to: {goal_pos}")
                                                    
                                                    _t_nav_fb = time.perf_counter()
                                                    nav_graph = self.global_map_manager.nav_graph
                                                    if nav_graph is None:
                                                        #  ( nav_graph )
                                                        from utils.navigation_helper import NavigationGraph
                                                        import open3d as o3d
                                                        total_pcd = o3d.geometry.PointCloud()
                                                        for o in self.global_map_manager.global_map:
                                                            total_pcd += o.pcd_2d
                                                        if hasattr(self.global_map_manager, 'layout_map') and self.global_map_manager.layout_map.wall_pcd is not None:
                                                            total_pcd += self.global_map_manager.layout_map.wall_pcd
                                                        
                                                        occ_map, x_min, y_min, grid_size = NavigationGraph.get_occupancy_map(
                                                            total_pcd, self.cfg.navigation.dilation_radius
                                                        )
                                                        nav_graph = NavigationGraph(occ_map, x_min, y_min, grid_size)
                                                        self.global_map_manager.nav_graph = nav_graph
                                                    
                                                    start_pos = self.curr_pose[:3, 3]
                                                    path_points = nav_graph.plan_path(start_pos, goal_pos)
                                                    navigation_path_planning_ms += (time.perf_counter() - _t_nav_fb) * 1000.0
                                                    
                                                    if path_points is not None:
                                                        if len(path_points) > 2:
                                                            from utils.navigation_helper import remove_sharp_turns_3d
                                                            path_points = remove_sharp_turns_3d(path_points, min_dist=0.1)
                                                        
                                                        self.global_map_manager.nav_graph.save_path(path_points, self.cfg.map_save_path)
                                                        
                                                        # Set rotation
                                                        target_rot = target_pose[:3, :3]
                                                        current_rot = self.curr_pose[:3, :3]
                                                        relative_rot = target_rot @ np.linalg.inv(current_rot)
                                                        from scipy.spatial.transform import Rotation as R
                                                        rot_vec = R.from_matrix(relative_rot).as_rotvec()
                                                        
                                                        self.target_rotation = rot_vec
                                                        self.target_rotation_matrix = target_rot 
                                                        
                                                        logger.warning(f"[Core] [Fallback] Path planned and rotation set.")
                                                    else:
                                                        logger.error("[Core] [Fallback] Failed to plan path.")
                                                else:
                                                    logger.error(f"[Core] [Fallback] Could not find pose for anchor {target_anchor_id}")
                                            else:
                                                logger.warning("[Core] [Fallback] LLM did not select any anchor.")
                                        else:
                                            logger.error(f"[Core] [Fallback] Script failed: {fb_res.stderr}")

                                    except Exception as e:
                                        logger.error(f"[Core] [Fallback] Error: {e}")
                                        import traceback
                                        traceback.print_exc()

                        if result_json is None:
                            logger.warning(
                                "[Core] NavRunner stdout JSON parse failed; text fallback"
                            )
                            target_visual_description = None
                            target_final_uid = None
                            for line in lines:
                                if "Navigate to frame:" in line:
                                    target_frame_path = line.split("Navigate to frame:")[-1].strip()
                                if "Selected target anchor UID:" in line:
                                    parts = line.split("Selected target anchor UID:")
                                    if len(parts) > 1:
                                        target_anchor_uid = parts[1].split("(")[0].strip()
                                if "Frame index:" in line:
                                    try:
                                        frame_idx = int(line.split("Frame index:")[-1].strip())
                                    except:
                                        pass
                                if "\"target_visual_description\":" in line:
                                    #  JSON 
                                    parts = line.split("\"target_visual_description\":")
                                    if len(parts) > 1:
                                        desc = parts[1].strip().strip(",").strip("\"")
                                        if desc:
                                            target_visual_description = desc
                                if "final_uid=" in line:
                                    parts = line.split("final_uid=")
                                    if len(parts) > 1:
                                        #  "final_uid=123-abc,"  "final_uid=123-abc"
                                        uid_cand = parts[1].split(",")[0].strip()
                                        if uid_cand:
                                            target_final_uid = uid_cand
                                if "\"target_visual_description\":" in line:
                                    #  JSON 
                                    parts = line.split("\"target_visual_description\":")
                                    if len(parts) > 1:
                                        desc = parts[1].strip().strip(",").strip("\"")
                                        if desc:
                                            target_visual_description = desc
                        
                        if target_anchor_uid:
                            # [CLIP Fallback Fix for Text Parsing]
                            #  AnchorUID  Frame ( CLIP Fallback), MemoryCluster 
                            if (not target_frame_path or target_frame_path == "None") and self.memory_cluster_manager:
                                logger.warning(f"[Core] [VLM] (Text) Missing frame for object {target_anchor_uid}. Refining best view pose...")
                                
                                #  ()
                                refined_pose = None
                                refined_frame_idx = None
                                refined_score = -1.0
                                
                                # 1.  clusters 
                                for cluster in self.memory_cluster_manager.clusters:
                                    rep = cluster.get("representative")
                                    if rep:
                                        obs_list = getattr(rep, "valid_observations", [])
                                        contains_target = False
                                        for obs in obs_list:
                                            if str(getattr(obs, "matched_obj_uid", "")) == target_anchor_uid:
                                                contains_target = True
                                                break
                                        if contains_target:
                                            score = getattr(rep, "score", 0.0)
                                            if score > refined_score:
                                                refined_score = score
                                                refined_pose = rep.pose
                                                refined_frame_idx = rep.frame_idx
                                                
                                # 2. ,
                                if refined_pose is None:
                                    for cluster in self.memory_cluster_manager.clusters:
                                        frames = cluster.get("frames", [])
                                        for frame in frames:
                                            obs_list = getattr(frame, "valid_observations", [])
                                            for obs in obs_list:
                                                if str(getattr(obs, "matched_obj_uid", "")) == target_anchor_uid:
                                                    if frame.pose is not None:
                                                        refined_pose = frame.pose
                                                        refined_frame_idx = frame.frame_idx
                                                        logger.warning(f"[Core] [VLM] (Text) Found target in non-representative frame {refined_frame_idx}")
                                                        break
                                            if refined_pose is not None: break
                                        if refined_pose is not None: break
                                
                                if refined_pose is not None:
                                    logger.warning(f"[Core] [VLM] (Text) ✅ Refined target pose! Frame: {refined_frame_idx}")
                                    frame_idx = refined_frame_idx
                                    
                                    #  frame_idx 
                                    found_new_path = None
                                    if hasattr(self.memory_cluster_manager, 'keyframe_dir'):
                                        kf_dir = self.memory_cluster_manager.keyframe_dir
                                        if os.path.exists(kf_dir):
                                            for fname in os.listdir(kf_dir):
                                                if f"frame_{frame_idx:06d}" in fname and fname.endswith(".png"):
                                                    found_new_path = os.path.join(kf_dir, fname)
                                                    break
                                    
                                    if found_new_path:
                                        target_frame_path = found_new_path
                                        logger.warning(f"[Core] [VLM] (Text) ✅ Found image: {target_frame_path}")
                            
                            logger.warning(f"[Core] [VLM] NavRunner Selected Target: Frame={target_frame_path}, AnchorUID={target_anchor_uid}, FrameIdx={frame_idx}")

                            # 0. , & , nav_visualizations
                            try:
                                if os.path.exists(target_frame_path):
                                    # ,
                                    if not os.path.isabs(target_frame_path):
                                        target_frame_path = os.path.abspath(target_frame_path)

                                    #  nav_visualizations 
                                    scene_dir = os.path.dirname(self.cfg.map_save_path)
                                    nav_viz_dir = os.path.join(scene_dir, "nav_visualizations")
                                    nav_viz_dir = os.path.abspath(nav_viz_dir)
                                    os.makedirs(nav_viz_dir, exist_ok=True)

                                    #  nav_visualizations ,
                                    if not target_frame_path.startswith(nav_viz_dir):
                                        best_frame_basename = os.path.basename(target_frame_path)
                                        nav_viz_path = os.path.join(nav_viz_dir, best_frame_basename)

                                        # 1.  nav_viz_path ()
                                        if not os.path.exists(nav_viz_path):
                                            try:
                                                shutil.copy2(target_frame_path, nav_viz_path)
                                                logger.warning(f"[Core] [VLM] ✅ (Text fallback) Copied best_frame (with boxes) to nav_visualizations: {nav_viz_path}")
                                            except Exception as e:
                                                logger.warning(f"[Core] [VLM] ⚠️ (Text fallback) Failed to copy best_frame to nav_visualizations: {e}")

                                        # 2. / CLEAN  nav_viz_clean_path ( PostRotationVLM)
                                        nav_viz_clean_path = nav_viz_path.replace(".png", "_clean.png")
                                        if not os.path.exists(nav_viz_clean_path):
                                            success_save_clean = False
                                            
                                            # A.  dataset  (CLEAN image)
                                            if frame_idx is not None and hasattr(self, 'dataset') and self.dataset is not None:
                                                try:
                                                    if 0 <= frame_idx < len(self.dataset):
                                                        raw_data_input = self.dataset[frame_idx]
                                                        raw_image = raw_data_input.color
                                                        
                                                        import cv2
                                                        if raw_image.shape[2] == 3:
                                                            raw_image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
                                                        else:
                                                            raw_image_bgr = raw_image
                                                            
                                                        cv2.imwrite(nav_viz_clean_path, raw_image_bgr)
                                                        logger.warning(f"[Core] [VLM] ✅ (Text fallback) Saved CLEAN frame {frame_idx} (from dataset) to: {nav_viz_clean_path}")
                                                        success_save_clean = True
                                                except Exception as e:
                                                    logger.warning(f"[Core] [VLM] ⚠️ (Text fallback) Failed to load clean frame from dataset: {e}")
                                            
                                            # B.  dataset , _clean.png 
                                            if not success_save_clean:
                                                clean_src_path = target_frame_path.replace(".png", "_clean.png")
                                                if os.path.exists(clean_src_path):
                                                    try:
                                                        shutil.copy2(clean_src_path, nav_viz_clean_path)
                                                        logger.warning(f"[Core] [VLM] ✅ (Text fallback) Copied CLEAN frame (from disk: {clean_src_path}) to: {nav_viz_clean_path}")
                                                        success_save_clean = True
                                                    except Exception as e:
                                                        logger.warning(f"[Core] [VLM] ⚠️ (Text fallback) Failed to copy clean frame from disk: {e}")
                                            
                                            if not success_save_clean:
                                                logger.warning(f"[Core] [VLM] ⚠️ (Text fallback) Could not find or save CLEAN frame. PostRotationVLM will fallback to boxed frame.")
                                        else:
                                            logger.warning(f"[Core] [VLM] ✅ (Text fallback) CLEAN frame already exists: {nav_viz_clean_path}")

                                        #  nav_visualizations 
                                        target_frame_path = nav_viz_path

                                    # (, VLM  fallback)
                                    self.nav_best_frame_path = target_frame_path
                                    # VLM,query
                                    self.nav_inquiry_sentence = target_visual_description if target_visual_description else self.inquiry
                                    
                                    # UID
                                    #  final_uid, anchor_uid  (CLIP)
                                    if target_final_uid:
                                        self.nav_target_uid = target_final_uid
                                    elif target_anchor_uid:
                                        self.nav_target_uid = target_anchor_uid

                                    self.need_post_rotation_vlm_check = True
                                    self.post_rotation_vlm_check_done = False
                                    self.rotation_completed = False

                                    logger.warning(f"[Core] [VLM] ✅✅✅ (Text fallback) SET nav_best_frame_path={self.nav_best_frame_path}!")
                                    logger.warning(f"[Core] [VLM] ✅✅✅ (Text fallback) SET nav_inquiry_sentence={self.nav_inquiry_sentence}!")
                                    if self.nav_best_frame_path and os.path.exists(self.nav_best_frame_path):
                                        logger.warning(f"[Core] [VLM] ✅ (Text fallback) Verified nav_best_frame_path exists: {self.nav_best_frame_path}")
                                    else:
                                        logger.warning(f"[Core] [VLM] ⚠️ (Text fallback) nav_best_frame_path does NOT exist: {self.nav_best_frame_path}")
                                else:
                                    logger.warning(f"[Core] [VLM] ⚠️ (Text fallback) target_frame_path from NavRunner does not exist on disk: {target_frame_path}")
                            except Exception as e:
                                logger.warning(f"[Core] [VLM] ⚠️ (Text fallback) Error while setting nav_best_frame_path/nav_inquiry_sentence: {e}")
                            
                            # 1. 
                            # frame_idxJSON,
                            if frame_idx is None:
                                try:
                                    import re
                                    match = re.search(r'frame_(\d+)', target_frame_path) or re.search(r'_(\d{6})_', target_frame_path)
                                    if match:
                                        frame_idx = int(match.group(1))
                                except Exception:
                                    pass
                                
                            target_pose = None
                            if frame_idx is not None and self.memory_cluster_manager:
                                 #  memory_cluster_manager  all_frame_poses 
                                 for pose_data in self.memory_cluster_manager.all_frame_poses:
                                     if pose_data["frame_idx"] == frame_idx:
                                         target_pose = pose_data["pose"]
                                         break
                            
                            if target_pose is None and frame_idx is not None and hasattr(self, "dataset") and self.dataset is not None:
                                 #  dataset  ( dataset )
                                 try:
                                     target_pose = self.dataset.poses[frame_idx]
                                 except:
                                     pass
                            
                            #  memory_cluster_manager 
                            if target_pose is None and frame_idx is not None and self.memory_cluster_manager:
                                 for pose_data in self.memory_cluster_manager.all_frame_poses:
                                     if pose_data["frame_idx"] == frame_idx:
                                         target_pose = pose_data["pose"]
                                         break

                            if target_pose is not None:
                                logger.warning(f"[Core] [VLM] Found target pose for frame {frame_idx}: {target_pose[:3, 3]}")
                                
                                # 2. 
                                #  calculate_global_path 
                                
                                #  (x, y, z)
                                goal_pos = target_pose[:3, 3]
                                
                                # : calculate_global_path  goal_mode ,
                                #  calculate_global_path  NavigationGraph
                                
                                nav_graph = self.global_map_manager.nav_graph
                                if nav_graph is None:
                                    #  nav_graph ,
                                    #  calculate_global_path 
                                    import open3d as o3d
                                    total_pcd = o3d.geometry.PointCloud()
                                    for obj in self.global_map_manager.global_map:
                                        total_pcd += obj.pcd_2d
                                    curr_point_coords = self.curr_pose[:3, 3]
                                    curr_point = o3d.geometry.PointCloud()
                                    curr_point.points = o3d.utility.Vector3dVector([curr_point_coords])
                                    total_pcd += curr_point
                                    total_pcd += self.global_map_manager.layout_map.wall_pcd
                                    
                                    #  NavigationGraph 
                                    from utils.navigation_helper import NavigationGraph
                                    nav_graph = NavigationGraph(self.cfg, total_pcd, cell_size=0.03)
                                    self.global_map_manager.nav_graph = nav_graph
                                    nav_graph.get_graph()
                                
                                curr_position = self.curr_pose[:3, 3]
                                start_2d = nav_graph.calculate_pos_2d(curr_position)
                                goal_2d_raw = nav_graph.calculate_pos_2d(goal_pos)
                                
                                if not nav_graph.free_space_check(goal_2d_raw):
                                    goal_2d = nav_graph.snap_to_free_space_directional(goal_2d_raw, start_2d, nav_graph.free_space)
                                else:
                                    goal_2d = goal_2d_raw
                                
                                logger.warning(f"[Core] [VLM] Computing path: start_2d={start_2d}, goal_2d_raw={goal_2d_raw}, goal_2d={goal_2d}")
                                path_2d = nav_graph.find_shortest_path(start_2d, goal_2d)
                                
                                if path_2d:
                                    self.curr_global_path = nav_graph.pos_path
                                    logger.warning(f"[Core] [VLM] NavRunner path successfully generated! Path length: {len(self.curr_global_path)} points")
                                    logger.warning(f"[Core] [VLM] Path start: {self.curr_global_path[0] if self.curr_global_path else 'N/A'}")
                                    logger.warning(f"[Core] [VLM] Path end: {self.curr_global_path[-1] if self.curr_global_path else 'N/A'}")
                                    
                                    # ,
                                    self.target_final_pose = target_pose
                                    
                                else:
                                    logger.error(f"[Core] [VLM] Failed to generate path to NavRunner target! start_2d={start_2d}, goal_2d={goal_2d}")
                                    self.curr_global_path = None
                            else:
                                logger.error(f"[Core] [VLM] Failed to find target pose for frame {frame_idx}!")
                                logger.error(f"[Core] [VLM] memory_cluster_manager exists: {self.memory_cluster_manager is not None}")
                                logger.error(f"[Core] [VLM] all_frame_poses count: {len(self.memory_cluster_manager.all_frame_poses) if self.memory_cluster_manager else 0}")
                                logger.error(f"[Core] [VLM] dataset exists: {hasattr(self, 'dataset') and self.dataset is not None}")
                                logger.warning(f"[Core] Could not find pose for frame {frame_idx}")
                                self.curr_global_path = None
                                
                        else:
                            logger.error(f"[Core] [VLM] ❌❌❌ Failed to get target from NavRunner!")
                            logger.error(f"[Core] [VLM] target_frame_path={target_frame_path}, target_anchor_uid={target_anchor_uid}")
                            logger.error(f"[Core] [VLM] Full stdout for debugging:\n{result.stdout}")
                            logger.warning("[Core] NavRunner did not return valid target info.")
                            logger.warning(f"[Core] [VLM] ⚠️ nav_best_frame_path will remain None, VLM check will use fallback")
                            self.curr_global_path = None

                        self._emit_nav_retrieve_nav_timing(
                            nav_runner_timing_from_json,
                            navigation_path_planning_ms,
                            subprocess_wall_ms,
                        )

                    else:
                        logger.error(f"[Core] ❌❌❌ NavRunner failed with code {result.returncode}")
                        logger.error(f"[Core] [VLM] ⚠️ nav_best_frame_path will remain None, VLM check will use fallback")
                        if result.stderr:
                            logger.error(f"[Core] [NavRunner ERR] {result.stderr}")
                        self.curr_global_path = None
                        
                except Exception as e:
                    logger.error(f"[Core] ❌❌❌ Error executing NavRunner: {e}")
                    logger.error(f"[Core] [VLM] ⚠️ nav_best_frame_path will remain None, VLM check will use fallback")
                    import traceback
                    logger.error(traceback.format_exc())
                    self.curr_global_path = None
            else:
                logger.warning("[Core] Skipping VLM task: Inquiry is empty.")
                self.curr_global_path = None

            # --- / ---
            # calculate the path based on current global map
            # self.curr_global_path = self.global_map_manager.calculate_global_path(
            #     self.curr_pose, goal_mode=self.get_goal_mode
            # )
            
            if self.curr_global_path is None:
                 logger.warning("[Core] No global path generated (VLM or Fallback failed).")

            if self.cfg.save_all_path and self.curr_global_path:
                # Create a unique file name using the counter
                self.path_counter += 1

                save_dir = os.path.join(self.cfg.output_path, "path", "global_path")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # build new file name with counter
                file_name = f"{self.path_counter}.json"
                save_path = os.path.join(save_dir, file_name)

                # Write JSON file
                with open(save_path, "w") as f:
                    json.dump(self.curr_global_path, f, indent=4)

                logger.warning(f"[Core] Global path saved to {save_path}")

                # Increment path counter for next save

            # make reset of the information in the yaml
            self.reset_cal_path_flag = True

            # Clear the local mapping results
            self.curr_local_path = None

            # Conditional open the local path planning
            # TODO: In future, for local avoidance, we need always local planning
            if self.get_goal_mode == GoalMode.RANDOM or (
                self.get_goal_mode == GoalMode.CLICK
                and self.global_map_manager.nav_graph.snapped_goal is None
            ):
                self.begin_local_planning = False
            else:
                self.begin_local_planning = True

        # Local Path Planning
        # Local need global plan success
        if self.begin_local_planning and self.local_map_manager.has_local_map():
            # logger.info("[Core] Local Navigation enabled! Triggering functionality...")
            pass
            # # set the global inquiry sentence to global map manager
            # self.local_map_manager.inquiry = self.inquiry_feat
            # 
            # # ... (rest of local planning logic)
            
            # if self.trigger_find_next:
            #     self.begin_local_planning = False
            #     self.reset_trigger_find_next = True
            #     self.global_map_manager.lost_and_found = True

            # # TEMP: Now explicitly pass the goal to local map to establish the workflow
            # global_path = self.curr_global_path

            # if global_path is not None and len(global_path) > 0:
            #     start = global_path[-1]
            #     x, y, z = start  # Unpack coordinates
            #     start_pose = np.eye(4)  # Initialize a 4x4 identity matrix
    
            #     # Set translation part
            #     start_pose[0, 3] = x  # X direction translation
            #     start_pose[1, 3] = y  # Y direction translation
            #     start_pose[2, 3] = z  # Z direction translation
            # else:
            #     # ...
            #     logger.warning("[Core] Global path is None or empty in parallel_process. Skipping local planning setup.")
            #     self.begin_local_planning = False
            #     start_pose = None
            
            # if self.begin_local_planning and start_pose is not None:
            #     # for CLick mode, pass the click goal to local from global
            #     # ...
            #     # ... ()
            #     
            #     # calculate the path based on current global map
            #     # self.curr_local_path = self.local_map_manager.calculate_local_path(
            #     #     start_pose, goal_mode=self.get_goal_mode
            #     # )
            #     
            #     # ...
            # else:
            #     self.curr_local_path = None
        else:
             # ,action_path()
             #  action_path = global_path + local_path
             # ,
             pass

        if self.cfg.save_all_path:
            # Ensure the path directory exists
            save_dir = os.path.join(self.cfg.output_path, "path", "local_path")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
        if self.cfg.save_all_path and self.curr_local_path is not None: # Added check for curr_local_path
            # build new file name with counter
            file_name = f"{self.path_counter}.json"
            save_path = os.path.join(save_dir, file_name)

            # Write JSON file
            # :json,UnboundLocalError
            import json as json_mod
            with open(save_path, "w") as f:
                json_mod.dump(self.curr_local_path, f, indent=4)

            # logger.warning(f"[Core] Local path saved to {save_path}")
            self.prev_pose = self.curr_pose

        # If both global path and local path exist

        self.get_action_path()

    def run_mapping_thread(self):
        """
        Independent thread: Monitor detection results and process local mapping and global mapping.
        """
        while not self.stop_thread:
            try:
                # Get detection results from queue
                # logger.info(f"Queue length: {len(self.detection_results_queue)}")
                queue_item = self.detection_results_queue.get(timeout=1)
                self._mapping_thread_busy = True
                try:
                    data_input = None
                    if len(queue_item) == 4:
                        curr_obs_list, curr_frame_id, novelty_info, data_input = queue_item
                    elif len(queue_item) == 3:
                        curr_obs_list, curr_frame_id, novelty_info = queue_item
                    else:
                        curr_obs_list, curr_frame_id = queue_item
                        novelty_info = None

                    logger.info(
                        f"[Core][MappingThread] Received data for frame {curr_frame_id}, Queue size {self.detection_results_queue.qsize()}"
                    )
                    if novelty_info is not None and not novelty_info.get('reason') == 'novel':
                        logger.debug(
                            f"[Core][MappingThread] frame {curr_frame_id} novelty: {novelty_info.get('reason', 'unknown')}"
                        )

                    # Set time stamp
                    self.visualizer.set_time_sequence("frame", self.curr_frame_id)

                    # Detection Visualization
                    if self.cfg.use_rerun:
                        self.detector.visualize_detection()
                    self.detector.update_data()

                    # Local Mapping
                    with timing_context("Local Mapping", self):
                        self.local_map_manager.set_curr_idx(curr_frame_id)
                        self.local_map_manager.set_curr_data(self.detector.curr_data)
                        self.local_map_manager.process_observations(curr_obs_list)

                    #  ()
                    if self.memory_cluster_manager is not None and data_input is not None:
                        #  (preload_global_map=False)   (lifelong_navigation=True)
                        lifelong_nav = False
                        if hasattr(self.cfg, "navigation") and hasattr(self.cfg.navigation, "lifelong_navigation"):
                            lifelong_nav = self.cfg.navigation.lifelong_navigation

                        if not self.cfg.get("preload_global_map", False) or lifelong_nav:
                            with timing_context("Memory Cluster", self):
                                self._update_memory_clusters(data_input, curr_obs_list, has_new_instance=False)

                    # Get global observations
                    with timing_context("Global Mapping", self):
                        global_obs_list = self.local_map_manager.get_global_observations()
                        self.local_map_manager.clear_global_observations()

                    # Global Mapping
                    self.global_map_manager.process_observations(global_obs_list)

                    preload_enabled = self.cfg.get("preload_global_map", False)
                    logger.warning(f"[Core][NewInstance] check: preload_global_map={preload_enabled}, data_input={'not None' if data_input is not None else 'None'}, global_obs={len(global_obs_list)}, local_obs={len(curr_obs_list)}")
                    if preload_enabled and data_input is not None:
                        # curr_obs_listxyxy
                        logger.warning(f"[Core][NewInstance] ========== call handler frame {curr_frame_id} ==========")
                        self._process_new_instances_nav(global_obs_list, curr_obs_list, data_input, curr_frame_id)
                    elif preload_enabled and data_input is None:
                        logger.warning(f"[Core][NewInstance] nav on but data_input=None, skip. queue_item len: {len(queue_item) if 'queue_item' in locals() else 'unknown'}")
                    elif not preload_enabled:
                        logger.warning(f"[Core][NewInstance] not nav mode, skip")

                    # Get memory usage statistics of local and global maps
                    # mem_stats = get_map_memory_usage(self.local_map_manager.local_map,
                    #                                 self.global_map_manager.global_map)

                    # logger.info(
                    #     f"[Core][MappingThread] Memory Usage - Local Map: {mem_stats['local_map_mb']} MB, "
                    #     f"Global Map: {mem_stats['global_map_mb']} MB"
                    # )

                finally:
                    self._mapping_thread_busy = False

            except queue.Empty:
                continue

    def wait_until_parallel_mapping_idle(self, timeout: float = 7200.0) -> bool:
        """
        After main thread enqueues all keyframes: wait until queue empty and mapping thread idle.
        Used so wall-clock / keyframe stats end after the main pipeline finishes.
        """
        if not getattr(self.cfg, "use_parallel", False):
            return True
        if not hasattr(self, "detection_results_queue"):
            return True
        deadline = time.time() + timeout
        stable = 0
        while time.time() < deadline:
            q_empty = self.detection_results_queue.qsize() == 0
            busy = getattr(self, "_mapping_thread_busy", False)
            if q_empty and not busy:
                stable += 1
                if stable >= 5:
                    return True
            else:
                stable = 0
            time.sleep(0.02)
        logger.warning("[Core] wait_until_parallel_mapping_idle timeout %.0fs", timeout)
        return False

    def get_action_path(self):
        if self.curr_global_path is None:
            logger.info("[Core][ActionPath] No Global Path! Action Path not available!")
            self.action_path = None
            self.global_map_manager.action_path = self.action_path
            return

        # Global path is definitely available here
        if self.curr_local_path is None:
            # Check if we have a VLM target rotation pending
            if self.target_final_pose is not None:
                # VLM , global path,
                logger.warning("[Core][ActionPath] Using Global Path with Final Rotation for VLM Target!")
                
                path_points = list(self.curr_global_path)
                
                target_pose = self.target_final_pose
                target_pos = target_pose[:3, 3]
                
                #  target_pos ()
                
                #  target_pose , Z .
                rotation_matrix = target_pose[:3, :3]
                forward_vector = rotation_matrix[:, 2] 
                forward_vector = forward_vector / np.linalg.norm(forward_vector)
                
                # ,
                if not hasattr(self, '_last_action_path_logged') or self._last_action_path_logged != self.path_counter:
                    logger.warning(f"[Core][ActionPath] Target Forward Vector: {forward_vector}")
                    self._last_action_path_logged = self.path_counter

                #  ( 1cm),
                last_point = np.array(path_points[-1])
                extension_point = last_point + forward_vector * 0.01  # 1cm
                
                if not hasattr(self, '_last_action_path_logged') or self._last_action_path_logged != self.path_counter:
                    logger.warning(f"[Core][ActionPath] Adding rotation guide point: {extension_point} (from {last_point})")

                path_points.append(extension_point.tolist())
                
                self.action_path = path_points
                
                if self.cfg.save_all_path:
                    # Ensure the path directory exists
                    save_dir = os.path.join(self.cfg.output_path, "path", "action_path")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    file_name = f"{self.path_counter}.json"
                    save_path = os.path.join(save_dir, file_name)
                    with open(save_path, "w") as f:
                        json.dump(self.action_path, f, indent=4)
                    if not hasattr(self, '_last_action_path_saved') or self._last_action_path_saved != self.path_counter:
                        logger.warning(f"[Core] Action path (with rotation) saved to {save_path}")
                        self._last_action_path_saved = self.path_counter

                self.global_map_manager.action_path = self.action_path
                self.global_map_manager.has_action_path = True
                
                #  target_final_pose,
                # target_final_pose  ROSPublisher 
                # self.target_final_pose = None
                return
            else:
                # then just use global path as action path
                logger.info(
                    "[Core][ActionPath] No Local Path! Action Path Now Using Global Path!"
                )
                self.action_path = self.curr_global_path
                self.global_map_manager.action_path = self.action_path
                return

        # Triggered by local path computing
        if self.start_action_path:
            logger.info("[Core][ActionPath] Start Action Path Calculation!")

            # delete the start point
            self.action_path = self.curr_global_path + self.curr_local_path[1:]

            self.action_path = remaining_path(self.action_path, self.curr_pose)

            # remove sharp returns
            if self.cfg.use_remove_sharp_turns:
                self.action_path = remove_sharp_turns_3d(self.action_path)

            if self.cfg.save_all_path:
                # Ensure the path directory exists
                save_dir = os.path.join(self.cfg.output_path, "path", "action_path")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # build new file name with counter
                file_name = f"{self.path_counter}.json"
                save_path = os.path.join(save_dir, file_name)

                # Write JSON file
                with open(save_path, "w") as f:
                    json.dump(self.action_path, f, indent=4)

                logger.warning(f"[Core] Action path saved to {save_path}")

            self.global_map_manager.action_path = self.action_path

            self.global_map_manager.has_action_path = True

            self.start_action_path = False

        if self.curr_global_path is not None and self.target_final_pose is not None:
            # VLM , global path,
            logger.warning("[Core][ActionPath] Using Global Path with Final Rotation for VLM Target!")
            
            path_points = list(self.curr_global_path)

            target_pose = self.target_final_pose
            target_pos = target_pose[:3, 3]

            #  target_pos ()
            # ,,
            # : action_path  [p1, p2, ...]
            # "".
            # .
            # ,,.
            
            #  ( Z  X ,)
            #  X  ( ROS)  Z  ()
            # .,Z .
            #  target_pose , Z .
            
            #  target_pose 
            rotation_matrix = target_pose[:3, :3]
            
            #  (Z)
            forward_vector = rotation_matrix[:, 2] 
            
            #  ()
            forward_vector = forward_vector / np.linalg.norm(forward_vector)
            
            logger.warning(f"[Core][ActionPath] Target Forward Vector: {forward_vector}")

            #  ( 1cm),
            # ,
            # :,
            last_point = np.array(path_points[-1])
            extension_point = last_point + forward_vector * 0.01  # 1cm
            
            logger.warning(f"[Core][ActionPath] Adding rotation guide point: {extension_point} (from {last_point})")

            #  numpy array  list 
            path_points.append(extension_point.tolist())
            
            self.action_path = path_points
            
            # , ()
            # if self.cfg.use_remove_sharp_turns:
            #     self.action_path = remove_sharp_turns_3d(self.action_path)

            if self.cfg.save_all_path:
                save_dir = os.path.join(self.cfg.output_path, "path", "action_path")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                file_name = f"{self.path_counter}.json"
                save_path = os.path.join(save_dir, file_name)
                with open(save_path, "w") as f:
                    json.dump(self.action_path, f, indent=4)
                logger.warning(f"[Core] Action path (with rotation) saved to {save_path}")

            self.global_map_manager.action_path = self.action_path
            self.global_map_manager.has_action_path = True
            self.start_action_path = False
            
            #  target_final_pose,
            # target_final_pose  ROSPublisher 
            # self.target_final_pose = None


    def export_anchor_summary(self, output_path: str) -> None:
        """Export anchor objects and best-frame metadata to JSON."""
        if not hasattr(self, "local_map_manager"):
            logger.warning("[ExportAnchorSummary] LocalMapManager not available, skip export.")
            return

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        class_lookup = self.local_map_manager.visualizer.obj_classes.get_classes_arr()
        best_dir = os.path.join(self.cfg.map_save_path, "best_frames_images")

        anchor_entries = []

        for obj in self.local_map_manager.local_map:
            if not getattr(obj, "is_low_mobility", False):
                continue
            if not getattr(obj, "is_stable", False):
                continue

            bbox = getattr(obj, "bbox", None)
            if bbox is None:
                continue

            def to_list(arr):
                return arr.tolist() if arr is not None else None

            min_bound = to_list(bbox.get_min_bound())
            max_bound = to_list(bbox.get_max_bound())
            center = to_list(bbox.get_center())
            extent = to_list(bbox.get_extent())

            class_id = getattr(obj, "final_class_id", getattr(obj, "class_id", None))
            if class_id is not None and 0 <= class_id < len(class_lookup):
                class_name = class_lookup[class_id]
            else:
                class_name = "unknown"

            if not self._is_anchor_class_allowed(class_name):
                continue

            related_objs = self.local_map_manager.get_related_objects(obj.uid)

            try:
                unified_cfg = getattr(self.cfg, "unified_frame_scoring", {})
                (
                    _best_indices,
                    selection_details,
                    _,
                    _,
                    _,
                ) = obj.get_best_observation_indices(
                    k=BEST_FRAME_FINAL_K,
                    method=BEST_FRAME_METHOD,
                    related_objs=related_objs,
                    class_lookup=class_lookup,
                    include_observation_refs=True,
                    candidate_count=BEST_FRAME_CANDIDATE_K,
                    distance_levels=3,
                    local_map_manager=self.local_map_manager,
                    unified_alpha_a=float(unified_cfg.get("alpha_a", 0.7)),
                    unified_alpha_o=float(unified_cfg.get("alpha_o", 0.3)),
                    unified_A_max=float(unified_cfg.get("A_max", 5.0)),
                    unified_O_max=float(unified_cfg.get("O_max", 10.0)),
                    unified_beta_e=float(unified_cfg.get("beta_e", 0.5)),
                    unified_beta_c=float(unified_cfg.get("beta_c", 0.5)),
                    unified_C_max=float(unified_cfg.get("C_max", 5.0)),
                    unified_gamma_1=float(unified_cfg.get("gamma_1", 0.6)),
                    unified_gamma_2=float(unified_cfg.get("gamma_2", 0.4)),
                )
            except Exception as exc:
                logger.warning(
                    "[ExportAnchorSummary] Failed to gather best frames for %s: %s",
                    obj.uid,
                    exc,
                )
                continue

            best_frames_payload = []
            for info in selection_details:
                frame_idx = int(info.get("idx", -1))
                if frame_idx < 0:
                    continue

                image_name = f"{class_name.replace(' ', '_')}_{obj.uid}_{frame_idx}.png"
                image_path = os.path.join(best_dir, image_name)

                obs_ref = info.get("_observation_ref")
                pose_list = None
                if obs_ref is not None:
                    data_input = getattr(obs_ref, "data_input", None)
                    pose = getattr(data_input, "pose", None) if data_input is not None else None
                    if pose is not None:
                        pose_list = pose.tolist()

                best_frames_payload.append(
                    {
                        "frame_idx": frame_idx,
                        "image_path": image_path,
                        "score": info.get("weighted_score"),
                        "raw_score": info.get("raw_score"),
                        "confidence": info.get("confidence"),
                        "edge_quality": info.get("edge_quality"),
                        "support_ratio": info.get("support_score"),
                        "camera_pose": pose_list,
                    }
                )

            mounted_objects = []
            for related_obj in related_objs:
                related_bbox = getattr(related_obj, "bbox", None)
                if related_bbox is None:
                    continue
                
                related_class_id = getattr(related_obj, "final_class_id", getattr(related_obj, "class_id", None))
                if related_class_id is not None and 0 <= related_class_id < len(class_lookup):
                    related_class_name = class_lookup[related_class_id]
                else:
                    related_class_name = "unknown"
                
                related_min_bound = to_list(related_bbox.get_min_bound())
                related_max_bound = to_list(related_bbox.get_max_bound())
                related_center = to_list(related_bbox.get_center())
                related_extent = to_list(related_bbox.get_extent())
                
                mounted_objects.append({
                    "object_uid": str(related_obj.uid),
                    "class_name": related_class_name,
                    "bbox": {
                        "min_bound": related_min_bound,
                        "max_bound": related_max_bound,
                    },
                    "position": {
                        "center": related_center,
                        "extent": related_extent,
                    },
                })

            anchor_entries.append(
                {
                    "anchor_uid": str(obj.uid),
                    "class_name": class_name,
                    "bbox": {
                        "min_bound": min_bound,
                        "max_bound": max_bound,
                    },
                    "position": {
                        "center": center,
                        "extent": extent,
                    },
                    "orientation": {
                        "major_plane_z": getattr(obj, "major_plane_info", None),
                    },
                    "best_frames": best_frames_payload,
                    "mounted_objects": mounted_objects,
                }
            )

        payload = {
            "map_path": self.cfg.map_save_path,
            "anchor_count": len(anchor_entries),
            "anchors": anchor_entries,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(
            "[ExportAnchorSummary] Exported %d anchors to %s",
            len(anchor_entries),
            output_path,
        )

    def save_high_confidence_frame_images(self, confidence_threshold: float = 0.6):
        """
        Read high_confidence_frames.txt, load full images, draw YOLO boxes.
        Saves as class_UUID_frame.png
        """
        import imageio.v2 as imageio
        import supervision as sv
        from utils.visualizer import visualize_result_rgb
        
        if self.dataset is None:
            logger.warning("[SaveHighConfidenceFrames] Dataset not available, skipping image saving.")
            return
        
        #  local_map_manager 
        if not hasattr(self, 'local_map_manager'):
            logger.warning("[SaveHighConfidenceFrames] LocalMapManager not available, skipping image saving.")
            return
        
        high_conf_frames_txt = os.path.join(self.cfg.map_save_path, "high_confidence_frames.txt")
        if not os.path.exists(high_conf_frames_txt):
            logger.warning(f"[SaveHighConfidenceFrames] File not found: {high_conf_frames_txt}")
            return
        
        save_dir = os.path.join(self.cfg.map_save_path, "high_confidence_images")
        
        # ()
        if os.path.exists(save_dir):
            import shutil
            try:
                shutil.rmtree(save_dir)
                logger.info(f"[SaveHighConfidenceFrames] Cleared existing directory: {save_dir}")
            except Exception as e:
                logger.error(f"[SaveHighConfidenceFrames] Error clearing directory {save_dir}: {e}")
                return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # UUID
        uuid_to_obj = {str(obj.uid): obj for obj in self.local_map_manager.local_map}
        
        saved_count = 0
        
        with open(high_conf_frames_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                frame_indices = []
                idx = len(parts) - 1
                while idx >= 0:
                    try:
                        frame_indices.insert(0, int(parts[idx]))
                        idx -= 1
                    except ValueError:
                        break
                
                if not frame_indices:
                    logger.warning(f"[SaveHighConfidenceFrames] : {line}")
                    continue
                
                obj_uid = parts[0]
                class_name = ' '.join(parts[1:idx+1])
                
                if obj_uid not in uuid_to_obj:
                    logger.warning(f"[SaveHighConfidenceFrames] Object {obj_uid} not found in local_map")
                    continue
                
                obj = uuid_to_obj[obj_uid]
                
                for frame_idx in frame_indices:
                    safe_class_name = class_name.replace(' ', '_')
                    filename = f"{safe_class_name}_{obj_uid}_{frame_idx}.png"
                    
                    try:
                        final_class_id = getattr(obj, "final_class_id", getattr(obj, "class_id", None))

                        target_obs = None
                        for obs in obj.observations:
                            if hasattr(obs, 'idx') and obs.idx == frame_idx:
                                target_obs = obs
                                break

                        if target_obs is None:
                            logger.warning(
                                f"[SaveHighConfidenceFrames] No observation found for object {obj_uid} at frame {frame_idx}"
                            )
                            continue
                        
                        actual_idx = getattr(target_obs, "raw_image_idx", frame_idx)

                        if actual_idx < 0 or actual_idx >= len(self.dataset.color_paths):
                            logger.warning(
                                f"[SaveHighConfidenceFrames] Invalid raw index {actual_idx} for frame {frame_idx}, "
                                f"total frames: {len(self.dataset.color_paths)}"
                            )
                            continue

                        color_path = self.dataset.color_paths[actual_idx]
                        full_image = imageio.imread(str(color_path))
                        # imageio.imread  RGB ,
                        
                        #  Detections 
                        xyxy = target_obs.xyxy
                        if xyxy.ndim == 1:
                            xyxy = xyxy.reshape(1, -1)
                        
                        # ID
                        class_id = target_obs.class_id
                        conf = target_obs.conf
                        
                        #  SAM 
                        mask_for_vis = None
                        mask = getattr(target_obs, "mask", None)
                        if mask is not None:
                            mask_np = np.asarray(mask)
                            if mask_np.size > 0:
                                if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                                    mask_np = mask_np[0]
                                if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
                                    mask_np = mask_np[..., 0]

                                mask_np = (mask_np > 0).astype(np.uint8)

                                img_h, img_w = full_image.shape[:2]
                                mask_h, mask_w = mask_np.shape[:2]
                                if (mask_h, mask_w) != (img_h, img_w):
                                    mask_np = cv2.resize(
                                        mask_np,
                                        (img_w, img_h),
                                        interpolation=cv2.INTER_NEAREST,
                                    )

                                mask_for_vis = (mask_np > 0)[np.newaxis, ...]

                        #  Detections
                        single_detection = sv.Detections(
                            xyxy=xyxy,
                            confidence=np.array([conf]),
                            class_id=np.array([class_id]),
                            mask=mask_for_vis,
                        )
                        
                        annotated_image, _ = visualize_result_rgb(
                            full_image,
                            single_detection,
                            self.local_map_manager.visualizer.obj_classes.get_classes_arr(),
                            draw_bboxes=True,
                        )
                        
                        save_path = os.path.join(save_dir, filename)
                        imageio.imwrite(save_path, annotated_image)
                        saved_count += 1
                        logger.debug(f"[SaveHighConfidenceFrames] Saved {filename} from frame {actual_idx}")
                        
                    except Exception as e:
                        logger.error(f"[SaveHighConfidenceFrames] Error saving {filename}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        continue
        
        logger.info(f"[SaveHighConfidenceFrames] Saved {saved_count} images to {save_dir}")


    def end_process(self):
        """
        The process of ending the sequnce.
        """
        end_frame_id = self.curr_frame_id

        self.stop_threading()

        # end duration
        end_range = self.cfg.active_window_size + self.cfg.max_pending_count + 1

        for i in range(end_range):
            # Set timestamp for visualizer
            logger.info("[Core][EndProcess] End Counter: %d", end_frame_id + i + 1)
            self.visualizer.set_time_sequence("frame", end_frame_id + i + 1)

            # local end_process
            # set fake timestamp
            self.local_map_manager.set_curr_idx(end_frame_id + i + 1)
            self.local_map_manager.end_process()
            local_map_obj_num = len(self.local_map_manager.local_map)
            logger.info("[Core][EndProcess] Local Objects num: %d", local_map_obj_num)

            # global process
            global_obs_list = self.local_map_manager.get_global_observations()
            self.local_map_manager.clear_global_observations()
            self.global_map_manager.process_observations(global_obs_list)
            global_map_obj_num = len(self.global_map_manager.global_map)
            logger.info("[Core][EndProcess] Global Objects num: %d", global_map_obj_num)

            if local_map_obj_num == 0:
                logger.warning(
                    "[EndProcess] End Processing End. to: %d", end_frame_id + i + 1
                )
                break

        with timing_context("Merging", self):
            if self.cfg.merge_local_map:
                self.local_map_manager.merge_local_map()
                self.visualizer.set_time_sequence("frame", end_range + 1)
                logger.info("[Core][EndProcess] Local Map Merged")

        # save the local mapping results
        if self.cfg.save_local_map:
            self.local_map_manager.save_map()

            summary_path = os.path.join(_OASIS_ROOT, "anchor_summary.json")
            self.export_anchor_summary(summary_path)
            # /,
            # self.select_macro_frames()

        # save the global mapping results
        if self.cfg.save_global_map:
            self.global_map_manager.save_map()

        if self.cfg.save_layout:
            self.detector.save_layout()

        # OASIS process timing
        if hasattr(self, "timing_results"):
            print_timing_results("OASIS", self.timing_results)
            system_time_path = self.cfg.map_save_path + "/../system_time.csv"
            save_timing_results(self.timing_results, system_time_path)

        # Detector process timing
        if hasattr(self.detector, "timing_results"):
            print_timing_results("Detector", self.detector.timing_results)
            detector_time_path = self.cfg.map_save_path + "/../detector_time.csv"
            save_timing_results(self.detector.timing_results, detector_time_path)

        # Memory cluster anchor instances summary
        if self.memory_cluster_manager is not None:
            if getattr(self, "memory_cluster_debugger", None):
                try:
                    self.memory_cluster_debugger.save_local_map_snapshot(
                        self.local_map_manager
                    )
                except Exception as exc:
                    logger.warning(
                        "[Core][MemoryClusterDebug] local map snapshot save failed: %s", exc
                    )
            dataset = getattr(self, "dataset", None)
            if dataset is None:
                logger.warning("[Core] Dataset is None in end_process, cannot export anchor key images correctly.")
            
            self.memory_cluster_manager.log_anchor_instances_summary(
                local_map_manager=self.local_map_manager,
                dataset=dataset,
                detector=self.detector,
            )
        
        # Save anchor information with attached objects
        # ,(memory_cluster_manager),
        #  anchor_attached_objects.txt
        if self.cfg.save_local_map:
            logger.info("[Core] Saving anchor attached objects...")
            try:
                self.save_anchor_attached_objects()
                logger.info("[Core] Anchor attached objects saved.")
            except Exception as e:
                logger.warning(f"[Core] Failed to save anchor attached objects: {e}")
                
            #  memory cluster  (manifest.json, images)
            if self.memory_cluster_manager is not None:
                logger.info("[Core] Saving memory cluster info...")
                try:
                    cluster_save_path = os.path.join(self.cfg.output_path, "map_results")
                    #  save_clusters ( manifest.json)
                    self.memory_cluster_manager.save_clusters(cluster_save_path)
                    logger.info("[Core] Memory cluster info saved.")
                except Exception as e:
                    logger.warning(f"[Core] Failed to save memory cluster info: {e}")

    def _infer_frame_image_area(self, frame_idx: int, valid_observations: List) -> Optional[float]:
        """Infer A_img (W*H) from mask size or cached full image in clusters."""
        for obs in valid_observations or []:
            mask = getattr(obs, "mask", None)
            if mask is None:
                continue
            mask_np = np.asarray(mask)
            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                mask_np = mask_np[0]
            if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
                mask_np = mask_np[..., 0]
            if mask_np.ndim == 2:
                return float(mask_np.shape[0] * mask_np.shape[1])
        if self.memory_cluster_manager is not None:
            for cluster in self.memory_cluster_manager.clusters:
                candidates: List[Any] = []
                rep = cluster.get("representative")
                if rep is not None:
                    candidates.append(rep)
                candidates.extend(cluster.get("frames", []))
                for fs in candidates:
                    if fs is None or getattr(fs, "frame_idx", -1) != frame_idx:
                        continue
                    img = getattr(fs, "image", None)
                    if img is None:
                        continue
                    im = np.asarray(img)
                    if im.ndim >= 2:
                        return float(im.shape[0] * im.shape[1])
        return None

    def compute_frame_score_unified(
        self,
        frame_idx: int,
        alpha_a: Optional[float] = None,
        alpha_o: Optional[float] = None,
        A_max: Optional[float] = None,
        O_max: Optional[float] = None,
        beta_e: Optional[float] = None,
        beta_c: Optional[float] = None,
        C_max: Optional[float] = None,
        gamma_1: Optional[float] = None,
        gamma_2: Optional[float] = None,
        A_img: Optional[float] = None,
        lambda_scale: Optional[float] = None,
        return_details: bool = False,
    ):
        """
        Unified frame score in [0, 1].

        Args:
            frame_idx: frame index
            Other args: formula weights (optional overrides)

        Returns:
            Normalized score or details dict if return_details
        """
        if not hasattr(self, "local_map_manager"):
            return 0.0 if not return_details else {}

        unified_cfg = getattr(self.cfg, "unified_frame_scoring", None)

        def get_cfg_value(key: str, default: float) -> float:
            if unified_cfg is None:
                return default
            try:
                # OmegaConf DictConfig
                if hasattr(unified_cfg, key):
                    val = getattr(unified_cfg, key)
                    return float(val)
                elif isinstance(unified_cfg, dict) and key in unified_cfg:
                    val = unified_cfg[key]
                    return float(val)
                else:
                    return default
            except (AttributeError, KeyError, TypeError, ValueError) as e:
                print(f"[DEBUG] Failed to get config value for {key}: {e}, using default {default}")
                return default

        if alpha_a is None:
            alpha_a = get_cfg_value("alpha_a", 0.7)
        if alpha_o is None:
            alpha_o = get_cfg_value("alpha_o", 0.3)
        if A_max is None:
            A_max = get_cfg_value("A_max", 5.0)
        if O_max is None:
            O_max = get_cfg_value("O_max", 10.0)
        if beta_e is None:
            beta_e = get_cfg_value("beta_e", 0.5)
        if beta_c is None:
            beta_c = get_cfg_value("beta_c", 0.5)
        if C_max is None:
            C_max = get_cfg_value("C_max", 5.0)
        if gamma_1 is None:
            gamma_1 = get_cfg_value("gamma_1", 0.6)
        if gamma_2 is None:
            gamma_2 = get_cfg_value("gamma_2", 0.4)
        if lambda_scale is None:
            lambda_scale = get_cfg_value("lambda_scale", 0.7)

        class_lookup = self.local_map_manager.visualizer.obj_classes.get_classes_arr()

        #  memory_cluster_manager (YOLO)
        valid_observations = []
        if self.memory_cluster_manager is not None:
            #  FrameStats
            for cluster in self.memory_cluster_manager.clusters:
                rep = cluster.get("representative")
                if rep is not None and rep.frame_idx == frame_idx:
                    valid_observations = getattr(rep, "valid_observations", [])
                    break
                frames = cluster.get("frames", [])
                for frame_stats in frames:
                    if frame_stats.frame_idx == frame_idx:
                        valid_observations = getattr(frame_stats, "valid_observations", [])
                        break
                if valid_observations:
                    break

        # ()
        anchor_areas = []
        anchor_completeness = []
        anchor_interaction_counts = []
        other_object_count = 0

        # ,
        for obs in valid_observations:
            class_name = None
            class_id = getattr(obs, "class_id", None)
            if class_id is not None and 0 <= class_id < len(class_lookup):
                class_name = class_lookup[class_id]
            else:
                class_name = getattr(obs, "class_name", "unknown")

            # (,)
            is_anchor = self._is_anchor_class_allowed(class_name)

            if is_anchor:
                # (maskbbox)
                area = 0.0
                mask = getattr(obs, "mask", None)
                if mask is not None:
                    mask_np = np.asarray(mask)
                    if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                        mask_np = mask_np[0]
                    if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
                        mask_np = mask_np[..., 0]
                    area = float((mask_np > 0).sum())
                else:
                    xyxy = getattr(obs, "xyxy", None)
                    if xyxy is not None:
                        arr = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)[0]
                        w = max(0.0, float(arr[2] - arr[0]))
                        h = max(0.0, float(arr[3] - arr[1]))
                        area = w * h

                anchor_areas.append(area)

                # (edge_quality)
                edge_quality = 1.0
                if mask is not None:
                    mask_np = np.asarray(mask)
                    if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                        mask_np = mask_np[0]
                    if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
                        mask_np = mask_np[..., 0]
                    mask_bool = mask_np.astype(bool)
                    total_pixels = int(mask_bool.sum())
                    if total_pixels > 0:
                        up = np.zeros_like(mask_bool, dtype=bool)
                        up[1:] = mask_bool[:-1]
                        down = np.zeros_like(mask_bool, dtype=bool)
                        down[:-1] = mask_bool[1:]
                        left = np.zeros_like(mask_bool, dtype=bool)
                        left[:, 1:] = mask_bool[:, :-1]
                        right = np.zeros_like(mask_bool, dtype=bool)
                        right[:, :-1] = mask_bool[:, 1:]
                        interior = mask_bool & up & down & left & right
                        edge_mask = mask_bool & (~interior)
                        edge_pixel_count = int(edge_mask.sum())
                        if edge_pixel_count > 0:
                            edge_contact_count = int(
                                edge_mask[0, :].sum()
                                + edge_mask[-1, :].sum()
                                + edge_mask[:, 0].sum()
                                + edge_mask[:, -1].sum()
                            )
                            edge_touch_ratio = edge_contact_count / float(edge_pixel_count)
                            edge_quality = 1.0 - edge_touch_ratio

                anchor_completeness.append(max(0.0, min(1.0, edge_quality)))

                # :YOLOYOLO
                interaction_count = 0
                # YOLO
                anchor_xyxy = getattr(obs, "xyxy", None)
                if anchor_xyxy is not None:
                    anchor_xyxy_arr = np.asarray(anchor_xyxy, dtype=np.float32).reshape(-1, 4)[0]
                    ax1, ay1, ax2, ay2 = anchor_xyxy_arr
                    
                    # ,
                    for other_obs in valid_observations:
                        if other_obs is obs:
                            continue
                        
                        # ()
                        other_class_id = getattr(other_obs, "class_id", None)
                        other_class_name = None
                        if other_class_id is not None and 0 <= other_class_id < len(class_lookup):
                            other_class_name = class_lookup[other_class_id]
                        else:
                            other_class_name = getattr(other_obs, "class_name", "unknown")
                        
                        if not self._is_anchor_class_allowed(other_class_name):
                            # YOLO
                            other_xyxy = getattr(other_obs, "xyxy", None)
                            if other_xyxy is not None:
                                other_xyxy_arr = np.asarray(other_xyxy, dtype=np.float32).reshape(-1, 4)[0]
                                bx1, by1, bx2, by2 = other_xyxy_arr
                                
                                # :
                                # 1. ax2 <= bx1 (,)
                                # 2. bx2 <= ax1 (,)
                                # 3. ay2 <= by1 (,)
                                # 4. by2 <= ay1 (,)
                                if not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1):
                                    interaction_count += 1
                
                anchor_interaction_counts.append(interaction_count)
            else:
                # ,
                other_object_count += 1

        if A_img is None:
            A_img = self._infer_frame_image_area(frame_idx, valid_observations)

        from utils.object import LocalObject
        result = LocalObject.compute_unified_frame_score(
            anchor_areas=anchor_areas,
            anchor_completeness=anchor_completeness,
            anchor_interaction_counts=anchor_interaction_counts,
            other_object_count=other_object_count,
            alpha_a=alpha_a,
            alpha_o=alpha_o,
            A_max=A_max,
            O_max=O_max,
            beta_e=beta_e,
            beta_c=beta_c,
            C_max=C_max,
            gamma_1=gamma_1,
            gamma_2=gamma_2,
            A_img=A_img,
            lambda_scale=lambda_scale,
            return_details=return_details,
        )

        if return_details:
            # ,()
            anchor_details = []
            other_object_details = []
            
            for obs in valid_observations:
                class_name = None
                class_id = getattr(obs, "class_id", None)
                if class_id is not None and 0 <= class_id < len(class_lookup):
                    class_name = class_lookup[class_id]
                else:
                    class_name = getattr(obs, "class_name", "unknown")
                
                # ()
                is_anchor = self._is_anchor_class_allowed(class_name)
                
                if is_anchor:
                    anchor_idx = len(anchor_details)
                    if anchor_idx < len(anchor_areas):
                        anchor_details.append({
                            "uid": str(getattr(obs, "matched_obj_uid", "unknown")),
                            "class_name": class_name,
                            "area": anchor_areas[anchor_idx],
                            "completeness": anchor_completeness[anchor_idx] if anchor_idx < len(anchor_completeness) else 0.0,
                            "interaction_count": anchor_interaction_counts[anchor_idx] if anchor_idx < len(anchor_interaction_counts) else 0,
                        })
                else:
                    # ,
                    other_object_details.append({
                        "uid": str(getattr(obs, "matched_obj_uid", "unknown")),
                        "class_name": class_name,
                    })
            
            if len(anchor_details) != len(anchor_areas):
                print(f"[WARNING] compute_frame_score_unified: anchor count mismatch anchor_details={len(anchor_details)}, anchor_areas={len(anchor_areas)}")
            if len(other_object_details) != other_object_count:
                print(f"[WARNING] compute_frame_score_unified: other-object count mismatch other_object_details={len(other_object_details)}, other_object_count={other_object_count}")
            
            result["anchor_details"] = anchor_details
            result["other_object_count"] = other_object_count
            result["other_object_details"] = other_object_details
            return result
        else:
            return result

    def monitor_config_file(self, config_file_path: str):
        """
        Monitor the configuration file at a given path and act based on its content.

        Parameters:
        - config_file_path: Path to the configuration file.
        """
        config_file = Path(config_file_path)
        if not config_file.is_absolute():
            # ,
            project_root = Path(__file__).parent.parent
            config_file = project_root / config_file_path
        logger.info(f"[Core][Monitor] Monitoring file: {config_file} (exists: {config_file.exists()})")

        while not self.stop_thread:
            try:
                if config_file.exists():
                    # Load the configuration file
                    config_data = OmegaConf.load(config_file)

                    # Access the field
                    if "calculate_path" in config_data:
                        calculate_path = config_data.calculate_path
                        
                        # Check if the field needs resetting (only reset if path was already calculated)
                        if self.reset_cal_path_flag and calculate_path:
                            # Reset the flag to avoid triggering multiple times
                            self.reset_cal_path_flag = False

                            # Reset the global path calculation flag in YAML from true to false
                            config_data.calculate_path = False

                            # Save the modified configuration back to the file
                            with open(config_file, "w") as file:
                                OmegaConf.save(config_data, file)
                            
                            logger.info("[Core][Monitor] Path calculation completed, resetting calculate_path to False")
                            self.calculate_path = False
                        else:
                            # Check the field value and trigger functionality
                            if calculate_path:
                                logger.info("[Core][Monitor] Navigation enabled! calculate_path=True")
                                # calculate_pathfalsetrue,new_instance
                                # calculate_path()
                                was_calculating = self.calculate_path
                                if not was_calculating and hasattr(self, "new_instance_save_dir") and os.path.exists(self.new_instance_save_dir):
                                    logger.warning(f"[Core][NewInstance] calculate_path false->true, clearing new_instance: {self.new_instance_save_dir}")
                                    try:
                                        # ()
                                        import stat
                                        def remove_readonly(func, path, exc_info):
                                            """Remove read-only files under dir."""
                                            os.chmod(path, stat.S_IWRITE)
                                            func(path)
                                        
                                        shutil.rmtree(self.new_instance_save_dir, onerror=remove_readonly)
                                        
                                        # ,sudo()
                                        if os.path.exists(self.new_instance_save_dir):
                                            logger.warning(f"[Core][NewInstance] rm failed, trying sudo")
                                            import subprocess
                                            result = subprocess.run(
                                                ["sudo", "rm", "-rf", self.new_instance_save_dir],
                                                capture_output=True,
                                                text=True,
                                                timeout=10
                                            )
                                            if result.returncode != 0:
                                                logger.error(f"[Core][NewInstance] sudo rm failed: {result.stderr}")
                                        
                                        os.makedirs(self.new_instance_save_dir, exist_ok=True)
                                        os.chmod(self.new_instance_save_dir, 0o777)
                                        
                                        if hasattr(self, "new_instance_tracker"):
                                            self.new_instance_tracker = {}
                                        if hasattr(self, "new_instance_frames"):
                                            self.new_instance_frames = {}
                                        if hasattr(self, "_saved_instance_count"):
                                            self._saved_instance_count = {}
                                        logger.warning(f"[Core][NewInstance] new_instance dir recreated")
                                    except Exception as e:
                                        logger.error(f"[Core][NewInstance] failed to clear new_instance: {e}")
                                        import traceback
                                        logger.error(f"[Core][NewInstance] Traceback: {traceback.format_exc()}")
                                self.calculate_path = True
                            else:
                                logger.debug("[Core][Monitor] Navigation disabled. calculate_path=False")
                                self.calculate_path = False

                    if "trigger_find_next" in config_data:
                        # Check if the field needs resetting
                        if self.reset_trigger_find_next:
                            # Reset the flag to avoid triggering multiple times
                            self.reset_trigger_find_next = False

                            # Reset the global path calculation flag in YAML from true to false
                            config_data.trigger_find_next = False
                            config_data.calculate_path = True

                            # Save the modified configuration back to the file
                            with open(config_file, "w") as file:
                                OmegaConf.save(config_data, file)

                        trigger_find_next = config_data.trigger_find_next

                        # Check the field value and trigger functionality
                        if trigger_find_next:
                            # logger.info("Navigation enabled! Triggering functionality...")
                            self.trigger_find_next = True

                        else:
                            # logger.info("Navigation disabled.")
                            self.trigger_find_next = False

                    # Access and handle 'get_goal_mode'
                    if "get_goal_mode" in config_data:
                        mode_value = config_data.get_goal_mode
                        try:
                            self.get_goal_mode = GoalMode(mode_value)
                            # logger.info(f"Goal Mode set to: {self.get_goal_mode.name}")
                        except ValueError:
                            logger.warning(
                                f"[Core][Monitor] Invalid mode '{mode_value}' in configuration. Defaulting to RANDOM."
                            )
                            self.get_goal_mode = GoalMode.RANDOM
                            config_data.get_goal_mode = GoalMode.RANDOM.value
                            with open(config_file, "w") as file:
                                OmegaConf.save(config_data, file)
                    # Access and handle 'inquiry_sentence'
                    # Get the sentence for inqury session
                    if "inquiry_sentence" in config_data:
                        new_inquiry = config_data.inquiry_sentence
                        if new_inquiry != self.inquiry:
                            logger.info(f"[Core][Monitor] Inquiry sentence updated: '{self.inquiry}' -> '{new_inquiry}'")
                            self.inquiry = new_inquiry
                        else:
                            logger.debug(f"[Core][Monitor] Inquiry sentence unchanged: '{self.inquiry}'")

                    # You can check and act on other fields similarly
                else:
                    logger.error(
                        f"[Core][Monitor] Config file not found: {config_file}"
                    )
            except Exception as e:
                logger.error(f"[Core][Monitor] Error monitoring config file: {e}")

            # Wait for the specified interval before checking again
            time.sleep(self.cfg.monitor_interval)

    def set_calculate_path(self, config_file_path: str):
        config_file = Path(config_file_path)
        config_data = OmegaConf.load(config_file)
        if "calculate_path" in config_data:
            # Reset the global path calculation flag in YAML from false to true
            config_data.calculate_path = True
            # config_data.trigger_find_next = False
            # if self.trigger_find_next:
            #     # Reset the flag to avoid triggering multiple times
            #     self.reset_trigger_find_next = False

            with open(config_file, "w") as file:
                OmegaConf.save(config_data, file)

    def _calculate_nearby_objects(self, anchor_obj, attached_objects, class_lookup):
        """
        Calculate nearby objects: stable objects that appear in the same representative frames
        as the anchor, excluding attached objects.
        Returns a dictionary: {class_name: {"count": total_count, "frames": set(frame_indices), "uids": []}}
        """
        anchor_uid = str(getattr(anchor_obj, "uid", "unknown"))

        # Get anchor's representative frames
        anchor_rep_frames = set(getattr(anchor_obj, "representative_frames", []))

        if not anchor_rep_frames:
            return {}

        # Get UIDs of attached objects to exclude
        attached_uids = {str(getattr(obj, "uid", "unknown")) for obj in attached_objects}

        # Find objects that appear in the same representative frames
        nearby_class_info = {} # {class_name: {"count": 0, "frames": set(), "uids": []}}
        objects_checked = 0
        rep_frame_overlap_found = 0

        for obj in self.local_map_manager.local_map:
            obj_uid = str(getattr(obj, "uid", "unknown"))

            # Skip self
            if obj_uid == anchor_uid:
                continue

            # Skip attached objects
            if obj_uid in attached_uids:
                continue

            # Only consider stable objects for nearby objects
            if not getattr(obj, "is_stable", False):
                continue
            
            # Only include objects that will be saved (save_path is not None)
            if getattr(obj, "save_path", None) is None:
                continue

            objects_checked += 1

            # Check if this object appears in any of the anchor's representative frames
            obj_rep_frames = set(getattr(obj, "representative_frames", []))

            # Check if there's overlap in representative frames
            # intersection = anchor_rep_frames & obj_rep_frames
            # NOTE: Logic change requested - we want to list WHICH anchor rep frames contain the object.
            # So we check intersection with anchor_rep_frames.
            
            # ,:,
            # The object must be present in the anchor's representative frame to be "nearby" in that view.
            #  obj.representative_frames ,.
            # ,.
            # :.
            
            common_frames = anchor_rep_frames & obj_rep_frames
            
            if common_frames:  # Intersection
                rep_frame_overlap_found += 1

                # Get object class
                class_id = getattr(obj, "m_class_id", None) or getattr(obj, "class_id", None)

                try:
                    if class_id is not None and 0 <= class_id < len(class_lookup):
                        obj_class = class_lookup[class_id]
                    else:
                        obj_class = "unknown"
                except Exception:
                    obj_class = "unknown"

                # Skip unknown objects
                # if obj_class == "unknown":
                #     continue

                if obj_class not in nearby_class_info:
                    nearby_class_info[obj_class] = {"count": 0, "frames": set(), "uids": []}
                
                nearby_class_info[obj_class]["count"] += 1
                nearby_class_info[obj_class]["frames"].update(common_frames)
                nearby_class_info[obj_class]["uids"].append(obj_uid)

        logger.info(f"[Nearby] Anchor {anchor_uid}: checked {objects_checked} stable objects, {rep_frame_overlap_found} with rep frame overlap, final nearby classes: {sorted(nearby_class_info.keys())}")

        return nearby_class_info

    def _get_short_anchor_id(self, uid: str) -> str:
        """Generate short anchor ID consistent with folder naming."""
        if not uid:
            return "unknown"
        # First split by "-" to get the first segment, then make it safe
        first_segment = uid.split("-")[0] if "-" in uid else uid
        safe_uid = "".join(ch if ch.isalnum() else "_" for ch in first_segment) or "unknown"
        return safe_uid[:8]

    def save_anchor_attached_objects(self):
        """
        Save anchor information with attached objects in the specified format.
        Format:
        Anchors in the scene:
        
        Anchor id: <anchor_id_1>
        Category: <anchor_class_1>
        Attached objects:
        <object_class_a> x <count_a>
        <object_class_b> x <count_b>
        ...
        
        Anchor id: <anchor_id_2>
        Category: <anchor_class_2>
        Attached objects:
        ...
        """
        if not hasattr(self.local_map_manager, "local_map"):
            logger.warning("[Core] Local map manager not available, skipping anchor info save.")
            return
        
        class_lookup = self.local_map_manager.visualizer.obj_classes.get_classes_arr()
        
        # Get all stable anchor objects (is_low_mobility=True and is_stable=True)
        anchor_objects = []
        total_objects = len(self.local_map_manager.local_map)
        low_mobility_count = 0
        stable_count = 0
        anchor_count = 0

        for obj in self.local_map_manager.local_map:
            is_low_mobility = getattr(obj, "is_low_mobility", False)
            is_stable = getattr(obj, "is_stable", False)

            if is_low_mobility:
                low_mobility_count += 1
            if is_stable:
                stable_count += 1
            # Check anchor class whitelist directly
            is_anchor_class = False
            if self.anchor_class_whitelist:
                obj_class_id = getattr(obj, "m_class_id", None)
                if obj_class_id is None:
                    obj_class_id = getattr(obj, "class_id", None)
                
                if obj_class_id is not None:
                    try:
                        obj_class_id = int(obj_class_id)
                        if 0 <= obj_class_id < len(class_lookup):
                            cls_name = class_lookup[obj_class_id]
                            if cls_name.lower() in self.anchor_class_whitelist:
                                is_anchor_class = True
                    except Exception:
                        pass

            if is_stable and (is_low_mobility or is_anchor_class):
                anchor_count += 1
                anchor_objects.append(obj)

        logger.info(f"[Core] Object counts: total={total_objects}, low_mobility={low_mobility_count}, stable={stable_count}, anchors={anchor_count}")

        if not anchor_objects:
            logger.info("[Core] No stable anchor objects found.")
            # ,
            anchor_info_list = []
        
        # Track which objects are attached to anchors
        attached_object_uids = set()
        anchor_info_list = []
        
        # Process each anchor
        for anchor_obj in anchor_objects:
            anchor_uid = str(getattr(anchor_obj, "uid", "unknown"))
            class_id = getattr(anchor_obj, "m_class_id", None) or getattr(anchor_obj, "class_id", None)
            
            try:
                if class_id is not None and 0 <= class_id < len(class_lookup):
                    anchor_class = class_lookup[class_id]
                else:
                    anchor_class = "unknown"
            except Exception:
                anchor_class = "unknown"
            
            # Skip unknown anchors
            if anchor_class == "unknown":
                continue
            
            # Get attached objects through graph relations AND direct spatial check
            # First try graph relations
            related_objects = self.local_map_manager.get_related_objects(anchor_uid)
            related_uids_from_graph = {str(getattr(obj, "uid", "unknown")) for obj in related_objects}
            
            # Also check all other objects directly using spatial relation check
            # This ensures we don't miss any relations that weren't captured in the graph
            for other_obj in self.local_map_manager.local_map:
                other_uid = str(getattr(other_obj, "uid", "unknown"))
                
                # Skip self
                if other_uid == anchor_uid:
                    continue
                
                # Skip if already found in graph
                if other_uid in related_uids_from_graph:
                    continue
                
                # Skip anchors
                if getattr(other_obj, "is_low_mobility", False):
                    continue
                
                # Check spatial relation directly
                try:
                    if self.local_map_manager.on_relation_check(anchor_obj, other_obj):
                        related_objects.append(other_obj)
                        related_uids_from_graph.add(other_uid)
                except Exception:
                    # If relation check fails, skip
                    continue
            
            # Count objects by class and collect UIDs
            # Only include stable objects that will be saved (save_path is not None)
            class_info = {} # {class_name: {"count": 0, "uids": []}}
            for related_obj in related_objects:
                related_uid = str(getattr(related_obj, "uid", "unknown"))
                
                # Only include stable objects
                if not getattr(related_obj, "is_stable", False):
                    continue
                
                # Only include objects that will be saved (save_path is not None)
                if getattr(related_obj, "save_path", None) is None:
                    continue
                
                related_class_id = getattr(related_obj, "m_class_id", None) or getattr(related_obj, "class_id", None)
                
                try:
                    if related_class_id is not None and 0 <= related_class_id < len(class_lookup):
                        related_class = class_lookup[related_class_id]
                    else:
                        related_class = "unknown"
                except Exception:
                    related_class = "unknown"
                
                # Skip unknown objects
                # if related_class == "unknown":
                #     continue
                
                if related_class not in class_info:
                    class_info[related_class] = {"count": 0, "uids": []}
                
                class_info[related_class]["count"] += 1
                class_info[related_class]["uids"].append(related_uid)
                attached_object_uids.add(related_uid)
            
            # Calculate nearby objects: stable objects that appear in the same representative frames
            # as this anchor, excluding attached objects
            nearby_objects = self._calculate_nearby_objects(anchor_obj, related_objects, class_lookup)
            
            # Filter out empty or None class names from nearby_objects
            # This fixes the issue where " x 5" appears in the output
            if nearby_objects:
                nearby_objects = {
                    k: v for k, v in nearby_objects.items() 
                    if k is not None and str(k).strip() != "" and k != "None"
                }

            short_anchor_id = self._get_short_anchor_id(anchor_uid)

            anchor_info_list.append({
                "uid": short_anchor_id,
                "category": anchor_class,
                "attached_objects": class_info,
                "nearby_objects": nearby_objects,
            })
        
        # Find objects not attached to any anchor (others)
        others_class_counts = {}
        for obj in self.local_map_manager.local_map:
            obj_uid = str(getattr(obj, "uid", "unknown"))

            # Skip anchors themselves
            if getattr(obj, "is_low_mobility", False):
                continue

            # Skip if already attached to an anchor
            if obj_uid in attached_object_uids:
                continue

            # Only include stable objects
            if not getattr(obj, "is_stable", False):
                continue
            
            # Only include objects that will be saved (save_path is not None)
            if getattr(obj, "save_path", None) is None:
                continue

            class_id = getattr(obj, "m_class_id", None) or getattr(obj, "class_id", None)

            try:
                if class_id is not None and 0 <= class_id < len(class_lookup):
                    obj_class = class_lookup[class_id]
                else:
                    obj_class = "unknown"
            except Exception:
                obj_class = "unknown"

            # Skip unknown objects
            # if obj_class == "unknown":
            #     continue

            others_class_counts[obj_class] = others_class_counts.get(obj_class, 0) + 1
        
        # Save to file in representative_frames/anchor_key directory
        #  memory_cluster_manager , representative_frames_path
        base_dir = None
        if getattr(self, "memory_cluster_manager", None) is not None:
             base_dir = self.memory_cluster_manager.keyframe_dir
        
        if base_dir is None:
             base_dir = getattr(
                 self.cfg,
                 "representative_frames_path",
                 os.path.join(_OASIS_ROOT, "output", "keyframe2", "default"),
             )
        
        save_dir = os.path.join(base_dir, "anchor_key")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "anchor_attached_objects.txt")
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("Anchors in the scene:\n\n")
            
            for anchor_info in anchor_info_list:
                f.write(f"Anchor id: {anchor_info['uid']}\n")
                f.write(f"Category: {anchor_info['category']}\n")
                f.write("Attached objects:\n")

                if anchor_info['attached_objects']:
                    # Sort by class name for consistency
                    sorted_classes = sorted(anchor_info['attached_objects'].items())
                    for cls_name, info in sorted_classes:
                        uids_str = ", ".join(info["uids"])
                        f.write(f"{cls_name} x {info['count']} (UID: {uids_str})\n")
                else:
                    f.write("(none)\n")

                f.write("Nearby objects:\n")
                if anchor_info['nearby_objects']:
                    # Sort by class name for consistency
                    sorted_nearby = sorted(anchor_info['nearby_objects'].items())
                    for cls_name, info in sorted_nearby:
                        # Check if uids are available in info
                        uids_str = ""
                        if "uids" in info:
                            uids_str = f", UID: {', '.join(info['uids'])}"
                        
                        if isinstance(info, dict) and "count" in info and "frames" in info:
                            count = info["count"]
                            frames = sorted(list(info["frames"]))
                            frames_str = ", ".join(str(f) for f in frames)
                            f.write(f"{cls_name} x {count} (Frames: {frames_str}{uids_str})\n")
                        else:
                            # Fallback for old format or simple count
                            f.write(f"{cls_name} x {info}\n")
                else:
                    f.write("(none)\n")
                
                f.write("\n")
            
            # Add others section if there are any
            if others_class_counts:
                f.write("Others:\n")
                sorted_others = sorted(others_class_counts.items())
                for obj_class, count in sorted_others:
                    f.write(f"{obj_class} x {count}\n")
        
        logger.info(
            "[Core] Saved anchor attached objects information to %s (%d anchors, %d others)",
            save_path,
            len(anchor_info_list),
            len(others_class_counts),
        )

    def stop_threading(self):
        self.stop_thread = True
        self.monitor_thread.join()

        if self.cfg.use_parallel:
            self.mapping_thread.join()

        logger.info("[Core] Stopped monitoring config file and mapping thread.")

    def _check_target_after_rotation(self, data_input: DataInput):
        """
        After rotation: append current frame and run VLM (post_rotation_vlm_check.py).
        Compares before/after images for target visibility.
        """
        logger.warning("[Core] [PostRotation] ========== _check_target_after_rotation CALLED! ==========")
        logger.warning(f"[Core] [PostRotation] nav_best_frame_path={self.nav_best_frame_path}")
        logger.warning(f"[Core] [PostRotation] nav_inquiry_sentence={self.nav_inquiry_sentence}")
        
        if not self.nav_best_frame_path or not self.nav_inquiry_sentence:
            logger.warning("[Core] [PostRotation] ❌ Missing navigation info, skipping VLM check")
            logger.warning(f"[Core] [PostRotation] nav_best_frame_path is None: {self.nav_best_frame_path is None}")
            logger.warning(f"[Core] [PostRotation] nav_inquiry_sentence is None: {self.nav_inquiry_sentence is None}")
            return

        self._ensure_vlm_prompt_transcript_session(self.nav_inquiry_sentence or self.inquiry or "")
        
        try:
            import subprocess
            
            _t_post_prep0 = time.perf_counter()
            # 1. append frame
            #  NavRunner :parent(map_save_path)/post_rotation_frames
            scene_dir = os.path.dirname(self.cfg.map_save_path)
            current_frame_dir = os.path.join(scene_dir, "post_rotation_frames")
            current_frame_dir = os.path.abspath(current_frame_dir)
            os.makedirs(current_frame_dir, exist_ok=True)
            current_frame_path = os.path.join(current_frame_dir, f"current_frame_{self.curr_frame_id:06d}.jpg")
            
            # RGB,BGR
            img = data_input.color.copy()
            if len(img.shape) == 3 and img.shape[2] == 3:
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(current_frame_path, img_bgr)
                logger.warning(f"[Core] [PostRotation] Saved current frame to: {current_frame_path}")
            else:
                logger.error(f"[Core] [PostRotation] Invalid image format, skipping VLM check")
                return
            
            # 2. ()
            best_frame_path = self.nav_best_frame_path
            if not os.path.isabs(best_frame_path):
                # ,
                possible_paths = [
                    best_frame_path,
                    os.path.join(self.cfg.output_path, best_frame_path),
                    os.path.join(os.path.dirname(self.cfg.output_path), best_frame_path),
                ]
                # best_frame_path,representative_frames
                if os.path.basename(best_frame_path) == best_frame_path:
                    rep_frames_dir = os.path.join(self.cfg.output_path, "representative_frames")
                    possible_paths.append(os.path.join(rep_frames_dir, best_frame_path))
                
                found = False
                for path in possible_paths:
                    if os.path.exists(path):
                        best_frame_path = path
                        found = True
                        break
                
                if not found:
                    logger.error(f"[Core] [PostRotation] Best frame path does not exist: {self.nav_best_frame_path}")
                    logger.error(f"[Core] [PostRotation] Tried paths: {possible_paths}")
                    return
            elif not os.path.exists(best_frame_path):
                logger.error(f"[Core] [PostRotation] Best frame path does not exist: {best_frame_path}")
                return
            
            # 2.1  CLEAN  ()
            # best_frame_path  ( nav_visualizations )
            #  _clean.png
            clean_frame_path = best_frame_path.replace(".png", "_clean.png")
            if os.path.exists(clean_frame_path):
                logger.warning(f"[Core] [PostRotation] ✅ Using CLEAN frame for VLM check: {clean_frame_path}")
                best_frame_path = clean_frame_path
            else:
                logger.warning(f"[Core] [PostRotation] ⚠️ CLEAN frame not found ({clean_frame_path}), using original frame: {best_frame_path}")
            
            # 3. VLM
            runner_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "post_rotation_vlm_check.py")
            
            model = getattr(self.cfg, "vlm_model", "gemini-3-pro-preview")
            gemini_key = getattr(self.cfg, "gemini_api_key", None) or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
            
            check_mode = "relocalization" if getattr(self, "is_relocalization_step", False) else "initial"
            logger.warning(f"[Core] [PostRotation] Check Mode: {check_mode}")
            
            viz_mode = getattr(self.cfg, "visualize_mode", "auto")

            post_prep_ms = (time.perf_counter() - _t_post_prep0) * 1000.0
            self._nav_timing_record(
                "post_rotation_vlm_prep_ms",
                post_prep_ms,
                {"check_mode": check_mode, "note": "append frame, resolve ref path, build cmd"},
            )
            
            cmd = [
                sys.executable,
                runner_script,
                "--task", self.nav_inquiry_sentence,
                "--best-frame", best_frame_path,
                "--current-frame", current_frame_path,
                "--check-mode", check_mode,
                "--model", model,
                "--visualize",
                "--viz-mode", viz_mode,
            ]
            if gemini_key:
                cmd.extend(["--gemini-api-key", gemini_key])
            if getattr(self, "_vlm_prompt_transcript_path", None):
                cmd.extend(["--prompt-transcript", self._vlm_prompt_transcript_path])
            
            logger.warning(f"[Core] [PostRotation] Calling VLM check script: {' '.join(cmd)}")
            logger.warning(f"[Core] [PostRotation] Task: {self.nav_inquiry_sentence}")
            logger.warning(f"[Core] [PostRotation] Best frame: {best_frame_path}")
            logger.warning(f"[Core] [PostRotation] Current frame: {current_frame_path}")
            
            # Popen
            _t_post_vlm0 = time.perf_counter()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            json_lines = []
            json_started = False
            
            logger.warning("[Core] [PostRotation] ========== VLM Check Script Output Start ==========")
            
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    logger.warning(f"[Core] [PostRotationScript] {line}")
                    
                    # JSON
                    if "[PostRotationVLM] Result JSON:" in line:
                        json_started = True
                        continue
                    if json_started:
                        json_lines.append(line)
            
            process.stdout.close()
            return_code = process.wait(timeout=60)
            post_vlm_sub_ms = (time.perf_counter() - _t_post_vlm0) * 1000.0
            
            logger.warning("[Core] [PostRotation] ========== VLM Check Script Output End ==========")
            
            target_found = False
            target_disappeared = False
            if return_code == 0:
                if json_lines:
                    try:
                        result_json = json.loads('\n'.join(json_lines))
                        logger.warning(f"[Core] [PostRotation] VLM Result JSON Parsed Successfully")
                        
                        target_found = result_json.get("target_found", False)
                        reasoning = result_json.get("reasoning", "")
                        target_disappeared = result_json.get("target_disappeared", False)
                        disappearance_analysis = result_json.get("disappearance_analysis", "")
                        confidence = result_json.get("confidence", "unknown")
                        
                        logger.warning(f"[Core] [PostRotation] ========== VLM Check Results Summary ==========")
                        logger.warning(f"[Core] [PostRotation] Target Found: {target_found}")
                        logger.warning(f"[Core] [PostRotation] Reasoning: {reasoning}")
                        if not target_found:
                            logger.warning(f"[Core] [PostRotation] Target Disappeared: {target_disappeared}")
                            logger.warning(f"[Core] [PostRotation] Disappearance Analysis: {disappearance_analysis}")
                        logger.warning(f"[Core] [PostRotation] Confidence: {confidence}")
                        logger.warning(f"[Core] [PostRotation] ==============================================")
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"[Core] [PostRotation] Failed to parse JSON from script output: {e}")
                else:
                    logger.error(f"[Core] [PostRotation] No JSON found in script output")
            else:
                logger.error(f"[Core] [PostRotation] VLM check script failed with code {return_code}")
            
            self._nav_timing_record(
                "post_rotation_vlm_subprocess_ms",
                post_vlm_sub_ms,
                {
                    "check_mode": check_mode,
                    "return_code": return_code,
                    "target_found": target_found,
                    "target_disappeared": target_disappeared if return_code == 0 and json_lines else None,
                },
            )
            
            self.post_rotation_vlm_check_done = True
            
            if 'target_found' in locals() and target_found:
                logger.warning(f"[Core] [PostRotation] ✅ Target SUCCESSFULLY FOUND! Loop terminated.")
                self.is_relocalization_loop_active = False
                self.relocalization_attempt_count = 0
                return

            should_loop = False
            
            # 1. :
            if 'target_disappeared' in locals() and target_disappeared:
                should_loop = True
            
            # 2. : ( VLM  Reloc  target_disappeared)
            if (self.is_relocalization_step or self.is_relocalization_loop_active) and ('target_found' in locals() and not target_found):
                should_loop = True
                
            if should_loop:
                if not self.is_relocalization_loop_active:
                    self.is_relocalization_loop_active = True
                    self.relocalization_attempt_count = 0
                
                self.relocalization_attempt_count += 1
                
                logger.warning(f"[Core] [PostRotation] ⚠️ Target not found. Entering Relocalization Loop Attempt {self.relocalization_attempt_count}/{self.max_relocalization_attempts}")
                
                if self.relocalization_attempt_count > self.max_relocalization_attempts:
                    logger.error(f"[Core] [PostRotation] ❌ Max relocalization attempts ({self.max_relocalization_attempts}) reached. Mission FAILED.")
                    self.is_relocalization_loop_active = False
                    return
                
                self._handle_target_disappearance(self.nav_target_uid, self.nav_inquiry_sentence, self.nav_best_frame_path)
                
        except subprocess.TimeoutExpired:
            logger.error(f"[Core] [PostRotation] VLM check script timed out after 60 seconds")
            if 'process' in locals():
                process.kill()
            self.post_rotation_vlm_check_done = True
        except Exception as e:
            logger.error(f"[Core] [PostRotation] Error in post-rotation VLM check: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # ,
            self.post_rotation_vlm_check_done = True

    def convert_inquiry_to_feat(self, inquiry_sentence: str):
        text_query_tokenized = self.detector.clip_tokenizer(inquiry_sentence).to("cuda")
        text_query_ft = self.detector.clip_model.encode_text(text_query_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()

        return text_query_ft

    def _get_target_clip_feature(self, uid, text_query):
        """
        Target CLIP feature: (1) global map by UID (2) else text encode if available.
        """
        # 1. Try to get from global map
        if uid and hasattr(self, "global_map_manager") and self.global_map_manager:
            for obj in self.global_map_manager.global_map:
                if str(obj.uid) == str(uid):
                    if hasattr(obj, "clip_ft") and obj.clip_ft is not None:
                         #  1D array
                         return obj.clip_ft.flatten()
        
        # 2. If failed, try text encoding using detector's clip model
        #  clip_model  MobileCLIP  encode_text 
        if hasattr(self, "detector") and hasattr(self.detector, "clip_model"):
             try:
                 #  detector.clip_model  encode_text , list of text
                 # MobileCLIP wrapper  encode_text(tokenizer(text))
                 # , obj.clip_ft.
                 # , tokenizer.
                 #  Object Map .
                 pass
             except Exception as e:
                 logger.warning(f"[Core] Failed to encode text feature: {e}")

        # 3. Try to load from disk directly
        candidate_dirs = []
        if hasattr(self.cfg, "preload_path") and self.cfg.preload_path:
            candidate_dirs.append(self.cfg.preload_path)
        if hasattr(self.cfg, "map_save_path") and self.cfg.map_save_path:
            candidate_dirs.append(self.cfg.map_save_path)
            
        for d in candidate_dirs:
            if not d or not os.path.exists(d):
                continue
            pkl_path = os.path.join(d, f"{uid}.pkl")
            if os.path.exists(pkl_path):
                try:
                    import pickle
                    with open(pkl_path, "rb") as f:
                        obj = pickle.load(f)
                    if hasattr(obj, "clip_ft") and obj.clip_ft is not None:
                        logger.warning(f"[Core] Loaded CLIP feature for {uid} from disk: {pkl_path}")
                        return obj.clip_ft.flatten()
                except Exception as e:
                    logger.warning(f"[Core] Failed to load object {uid} from disk: {e}")

        logger.warning(f"[Core] Target object {uid} not found or has no CLIP feature.")
        return None

    def _handle_target_disappearance(self, target_uid, text_query, ref_image_path):
        """Search new instances when target disappears."""
        self._ensure_vlm_prompt_transcript_session(text_query or self.inquiry or "")
        logger.warning(f"[Core] [Re-Search] Target {target_uid} disappeared. Searching in new instances...")
        _t_reloc_prep0 = time.perf_counter()
        
        # --- OLD LOGIC: Use 3D object feature ---
        # target_ft = self._get_target_clip_feature(target_uid, text_query)
        # if target_ft is None:
        #      logger.error("[Core] [Re-Search] Cannot get target feature vector. Aborting search.")
        #      # fallback(VLM candidates ,Top 5)
        #      return
        
        # --- NEW LOGIC: Use Text Query Feature (VLM Description) ---
        logger.warning(f"[Core] [Re-Search] Using text query for matching: '{text_query}'")
        target_ft = None
        try:
            import torch
            if hasattr(self.detector, "clip_model") and hasattr(self.detector, "clip_tokenizer"):
                device = self.cfg.device
                text_tokens = self.detector.clip_tokenizer([text_query]).to(device)
                with torch.no_grad():
                    text_features = self.detector.clip_model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    target_ft = text_features.cpu().numpy().flatten()
            else:
                logger.error("[Core] [Re-Search] CLIP model not available in detector.")
                return
        except Exception as e:
            logger.error(f"[Core] [Re-Search] Failed to encode text query: {e}")
            return
             
        #  feature
        candidates = []
        if not hasattr(self, "new_instance_save_dir") or not self.new_instance_save_dir or not os.path.exists(self.new_instance_save_dir):
            logger.warning("[Core] [Re-Search] No new_instance directory found.")
            return

        import numpy as np
        
        for instance_dir_name in os.listdir(self.new_instance_save_dir):
            instance_dir = os.path.join(self.new_instance_save_dir, instance_dir_name)
            if not os.path.isdir(instance_dir):
                continue
            
            #  _clip.npy
            best_clip_path = None
            best_img_path = None # _clean.jpg
            best_pose_path = None
            
            for fname in os.listdir(instance_dir):
                if fname.endswith("_clip.npy"):
                    best_clip_path = os.path.join(instance_dir, fname)
                    prefix = fname.replace("_clip.npy", "")
                    best_img_path = os.path.join(instance_dir, prefix + "_clean.jpg")
                    best_pose_path = os.path.join(instance_dir, prefix + "_pose.npy")
                    
                    # ( new instances,)
                    # ,..
                    break 
            
            if best_clip_path and os.path.exists(best_clip_path) and best_img_path and os.path.exists(best_img_path):
                try:
                    ft = np.load(best_clip_path).flatten()
                    
                    # Cosine Sim: (A . B) / (|A| * |B|)
                    norm_target = np.linalg.norm(target_ft)
                    norm_ft = np.linalg.norm(ft)
                    
                    if norm_target > 0 and norm_ft > 0:
                        sim = np.dot(target_ft, ft) / (norm_target * norm_ft)
                    else:
                        sim = 0.0
                    
                    candidates.append({
                        "sim": sim,
                        "img_path": best_img_path,
                        "pose_path": best_pose_path,
                        "instance_id": instance_dir_name
                    })
                except Exception as e:
                    logger.warning(f"[Core] Error loading clip feature for {instance_dir_name}: {e}")

        if not candidates:
            logger.warning("[Core] [Re-Search] No valid candidates found in new instances.")
            return

        # Sort and take Top 5
        candidates.sort(key=lambda x: x["sim"], reverse=True)
        top_candidates = candidates[:5]
        
        sim_scores = [f"{c['sim']:.4f}" for c in top_candidates]
        logger.warning(f"[Core] [Re-Search] Found {len(candidates)} new instances. Top 5 similarity: {sim_scores}")
        
        #  VLM
        candidate_paths = [c["img_path"] for c in top_candidates]
        
        runner_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "find_target_in_new_instances.py")
        
        model = getattr(self.cfg, "vlm_model", "gemini-3-pro-preview")
        gemini_key = getattr(self.cfg, "gemini_api_key", None) or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        viz_mode = getattr(self.cfg, "visualize_mode", "auto")
        
        cmd = [
            sys.executable,
            runner_script,
            "--task", text_query,
            "--ref-image", ref_image_path,
            "--model", model,
        ]
        if gemini_key:
            cmd.extend(["--gemini-api-key", gemini_key])
        cmd.extend(
            [
                "--visualize",
                "--viz-mode", viz_mode,
                "--candidates",
            ]
            + candidate_paths
        )
        if getattr(self, "_vlm_prompt_transcript_path", None):
            cmd.extend(["--prompt-transcript", self._vlm_prompt_transcript_path])
        
        reloc_prep_ms = (time.perf_counter() - _t_reloc_prep0) * 1000.0
        self._nav_timing_record(
            "relocalization_clip_and_candidates_prep_ms",
            reloc_prep_ms,
            {
                "num_candidates_scanned": len(candidates),
                "top_k_sent_to_vlm": len(top_candidates),
            },
        )
        
        logger.warning(f"[Core] [Re-Search] Calling VLM Finder script...")
        
        finder_vlm_ms = 0.0
        try:
            import subprocess
            _t_finder_vlm0 = time.perf_counter()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            json_lines = []
            json_started = False
            
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    logger.warning(f"[Core] [FinderScript] {line}")
                    if "[FindTarget] Result JSON:" in line:
                        json_started = True
                        continue
                    if json_started:
                        json_lines.append(line)
            
            process.stdout.close()
            process.wait(timeout=60)
            finder_vlm_ms = (time.perf_counter() - _t_finder_vlm0) * 1000.0
            
            found = False
            idx = -1
            if json_lines:
                try:
                    finder_json = json.loads('\n'.join(json_lines))
                    found = finder_json.get("found", False)
                    idx = finder_json.get("selected_index", -1)
                    reasoning = finder_json.get("reasoning", "")
                    
                    logger.warning(f"[Core] [Re-Search] VLM Result: Found={found}, Index={idx}")
                    logger.warning(f"[Core] [Re-Search] Reasoning: {reasoning}")
                    
                    if found and 1 <= idx <= len(top_candidates):
                        selected_candidate = top_candidates[idx-1]
                        logger.warning(f"[Core] [Re-Search] ✅ Target RE-LOCATED in new instance: {selected_candidate['instance_id']}")
                        
                        #  new_location 
                        try:
                            if self.new_instance_save_dir:
                                scene_dir = os.path.dirname(self.new_instance_save_dir)
                                new_loc_dir = os.path.join(scene_dir, "new_location")
                                os.makedirs(new_loc_dir, exist_ok=True)
                                
                                src_path = selected_candidate['img_path']
                                dst_path = os.path.join(new_loc_dir, os.path.basename(src_path))
                                
                                import shutil
                                shutil.copy2(src_path, dst_path)
                                logger.warning(f"[Core] [Re-Search] Saved new location image to: {dst_path}")
                        except Exception as e:
                            logger.error(f"[Core] [Re-Search] Failed to save new location image: {e}")

                        # Load Pose
                        if selected_candidate["pose_path"] and os.path.exists(selected_candidate["pose_path"]):
                            pose = np.load(selected_candidate["pose_path"])
                            logger.warning(f"[Core] [Re-Search] New Pose:\n{pose}")
                            
                            # Trigger navigation to the new pose
                            logger.warning(f"[Core] [Re-Search] Triggering navigation to new location...")
                            
                            # Set relocalization flag
                            self.is_relocalization_step = True
                            
                            self._plan_and_set_path(pose)
                            
                            # Reset flags to allow checking upon arrival
                            self.rotation_completed = False
                            self.post_rotation_vlm_check_done = False
                            self.need_post_rotation_vlm_check = True
                        else:
                            logger.warning(f"[Core] [Re-Search] Pose file missing for selected candidate.")
                    else:
                        logger.warning(f"[Core] [Re-Search] VLM did not find the target in the candidates.")
                        
                        # --- STRATEGY 2: Semantic Reasoning Fallback (After Re-Search Failure) ---
                        # , LLM 
                        logger.warning("[Core] [Re-Search] Initiating Semantic Reasoning Fallback (Strategy 2)...")
                        
                        try:
                            # 1.  ID ( nav_target_uid )
                            visited_ids = []
                            if hasattr(self, 'nav_target_uid') and self.nav_target_uid:
                                visited_ids.append(self.nav_target_uid)
                            
                            logger.warning(f"[Core] [Re-Search Fallback] Visited anchors: {visited_ids}")
                            
                            # 2. 
                            fallback_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "semantic_fallback.py")
                            
                            model = getattr(self.cfg, "llm_model", "gpt-5.2")
                            base_url = getattr(self.cfg, "vlm_base_url", None) or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                            api_key = (getattr(self.cfg, "vlm_api_key", None) or os.environ.get("OPENAI_API_KEY", ""))
                            anchor_file = os.path.join(
                                self._navigation_keyframe_dir(), "anchor_key", "anchor_attached_objects.txt"
                            )
                            
                            fb_cmd = [
                                sys.executable,
                                fallback_script,
                                "--task", self.inquiry,
                                "--anchor-file", anchor_file,
                                "--model", model,
                                "--base-url", base_url,
                                "--api-key", api_key,
                                "--visualize",  # Enable visualization
                                "--visited"
                            ] + visited_ids
                            if getattr(self, "_vlm_prompt_transcript_path", None):
                                fb_cmd.extend(["--prompt-transcript", self._vlm_prompt_transcript_path])
                            
                            import subprocess
                            _t_fb = time.perf_counter()
                            fb_res = subprocess.run(fb_cmd, capture_output=True, text=True)
                            fb_llm_ms = (time.perf_counter() - _t_fb) * 1000.0
                            self._nav_timing_record(
                                "relocalization_semantic_fallback_llm_ms",
                                fb_llm_ms,
                                {"returncode": fb_res.returncode},
                            )
                            
                            if fb_res.returncode == 0 and fb_res.stdout:
                                # import json  <-- REMOVED: Use global json import
                                try:
                                    fb_json = json.loads(fb_res.stdout)
                                except json.JSONDecodeError:
                                    # , JSON 
                                    logger.warning(f"[Core] [Re-Search Fallback] JSON decode error, trying to extract JSON from: {fb_res.stdout}")
                                    try:
                                        #  {  }
                                        start_idx = fb_res.stdout.find('{')
                                        end_idx = fb_res.stdout.rfind('}')
                                        if start_idx != -1 and end_idx != -1:
                                            json_str = fb_res.stdout[start_idx:end_idx+1]
                                            fb_json = json.loads(json_str)
                                        else:
                                            raise ValueError("No JSON object found in output")
                                    except Exception as e:
                                        logger.error(f"[Core] [Re-Search Fallback] Failed to extract JSON: {e}")
                                        fb_json = {}

                                target_anchor_id = fb_json.get("target_anchor_id")
                                reasoning = fb_json.get("reasoning", "")
                                
                                logger.warning(f"[Core] [Re-Search Fallback] LLM Suggestion: {target_anchor_id}")
                                logger.warning(f"[Core] [Re-Search Fallback] Reasoning: {reasoning}")
                                
                                if target_anchor_id:
                                    # 3. [FIX]  manifest.json  Anchor 
                                    #  Initial Search ,
                                    target_pose = None
                                    target_frame_idx = -1
                                    
                                    try:
                                        manifest_path = os.path.join(
                                            self._navigation_keyframe_dir(), "anchor_key", "manifest.json"
                                        )
                                        if os.path.exists(manifest_path):
                                            with open(manifest_path, 'r') as f:
                                                manifest = json.load(f)
                                            
                                            # manifest  {"anchors": {uid: data}}  {"anchors": [data]}
                                            anchors_data = manifest.get("anchors", manifest)
                                            
                                            anchors_list = []
                                            if isinstance(anchors_data, dict):
                                                # , values 
                                                anchors_list = list(anchors_data.values())
                                            elif isinstance(anchors_data, list):
                                                anchors_list = anchors_data
                                            
                                            for anchor in anchors_list:
                                                # , manifest 
                                                if not isinstance(anchor, dict):
                                                    continue
                                                    
                                                #  ID  (uid  id)
                                                anchor_id = str(anchor.get("uid", anchor.get("id", "")))
                                                
                                                # [FIX] , LLM  ID ( 6272a413)  manifest  UUID
                                                target_id_str = str(target_anchor_id)
                                                if target_id_str and (target_id_str in anchor_id or anchor_id in target_id_str):
                                                    #  Anchor
                                                    #  saved_images ()
                                                    saved_images = anchor.get("saved_images", [])
                                                    if saved_images:
                                                        # ()
                                                        best_img = saved_images[0]
                                                        
                                                        if "pose" in best_img:
                                                            target_pose = np.array(best_img["pose"])
                                                            logger.warning(f"[Core] [Re-Search Fallback] ✅ Found pose in manifest for {target_anchor_id}")
                                                        
                                                        if "frame_idx" in best_img:
                                                            target_frame_idx = int(best_img["frame_idx"])
                                                        elif "path" in best_img:
                                                            import re
                                                            match = re.search(r'frame_(\d+)', best_img["path"])
                                                            if match:
                                                                target_frame_idx = int(match.group(1))
                                                        
                                                        # [FIX] 
                                                        if target_pose is None and target_frame_idx != -1:
                                                            #  A:  all_frame_poses ()
                                                            if self.memory_cluster_manager.all_frame_poses:
                                                                for pose_data in self.memory_cluster_manager.all_frame_poses:
                                                                    if pose_data["frame_idx"] == target_frame_idx:
                                                                        target_pose = pose_data["pose"]
                                                                        logger.warning(f"[Core] [Re-Search Fallback] ✅ Found pose in all_frame_poses for frame {target_frame_idx}")
                                                                        break
                                                            
                                                            #  B:  Clusters (, Clusters )
                                                            if target_pose is None and self.memory_cluster_manager.clusters:
                                                                for cluster in self.memory_cluster_manager.clusters:
                                                                    rep = cluster.get("representative")
                                                                    if rep and rep.frame_idx == target_frame_idx and rep.pose is not None:
                                                                        target_pose = rep.pose
                                                                        logger.warning(f"[Core] [Re-Search Fallback] ✅ Found pose in Loaded Clusters for frame {target_frame_idx}")
                                                                        break
                                                            
                                                            #  C:  Dataset ()
                                                            if target_pose is None and hasattr(self, 'dataset') and self.dataset is not None:
                                                                try:
                                                                    if 0 <= target_frame_idx < len(self.dataset):
                                                                        target_pose = self.dataset.poses[target_frame_idx]
                                                                        logger.warning(f"[Core] [Re-Search Fallback] ✅ Found pose in Dataset for frame {target_frame_idx}")
                                                                except Exception as e:
                                                                    logger.warning(f"[Core] [Re-Search Fallback] Dataset pose lookup failed: {e}")

                                                    break
                                    except Exception as e:
                                        logger.error(f"[Core] [Re-Search Fallback] Error reading manifest: {e}")
                                    
                                    if target_pose is not None:
                                        # 4. 
                                        self.nav_target_uid = target_anchor_id
                                        self.nav_inquiry_sentence = self.inquiry
                                        self.need_post_rotation_vlm_check = True
                                        self.post_rotation_vlm_check_done = False
                                        self.rotation_completed = False
                                        
                                        # ,VLM ""
                                        if 'ref_image_path' in locals() and ref_image_path:
                                            self.nav_best_frame_path = ref_image_path
                                            logger.warning(f"[Core] [Re-Search Fallback] Set reference image to original: {self.nav_best_frame_path}")
                                        else:
                                            # ref_image_path,None (fallback,)
                                            #  self.nav_best_frame_path  ()
                                            if not self.nav_best_frame_path:
                                                logger.warning("[Core] [Re-Search Fallback] No reference image available!")
                                        
                                        self._plan_and_set_path(target_pose)
                                    else:
                                        logger.error(f"[Core] [Re-Search Fallback] Could not find pose for anchor {target_anchor_id}")
                                else:
                                    logger.warning("[Core] [Re-Search Fallback] LLM did not select any anchor.")
                            else:
                                logger.error(f"[Core] [Re-Search Fallback] Script failed: {fb_res.stderr}")
                        except Exception as e:
                            logger.error(f"[Core] [Re-Search Fallback] Error: {e}")
                        
                except Exception as e:
                    logger.error(f"[Core] [Re-Search] Failed to parse result: {e}")
            else:
                logger.error("[Core] [Re-Search] No JSON result from script.")
            
            self._nav_timing_record(
                "relocalization_find_target_vlm_subprocess_ms",
                finder_vlm_ms,
                {
                    "found": found,
                    "selected_index": idx,
                    "had_json_lines": bool(json_lines),
                },
            )
                
        except Exception as e:
            logger.error(f"[Core] [Re-Search] Error executing script: {e}")

    def _ensure_vlm_prompt_transcript_session(self, inquiry: Optional[str] = None) -> None:
        """Create or reuse a single transcript file under task_prompt for this session."""
        from datetime import datetime

        from utils.vlm_prompt_transcript import (
            DEFAULT_TASK_PROMPT_DIR,
            ENV_KEY,
            init_session_header,
        )

        if getattr(self, "_vlm_prompt_transcript_path", None) and os.path.isfile(
            self._vlm_prompt_transcript_path
        ):
            os.environ[ENV_KEY] = self._vlm_prompt_transcript_path
            return

        q = (inquiry if inquiry is not None else (self.inquiry or "")).strip()
        DEFAULT_TASK_PROMPT_DIR.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in q[:50]).strip().replace(" ", "_")
        if not safe:
            safe = "task"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = str(DEFAULT_TASK_PROMPT_DIR / f"session_{ts}_{safe}.txt")
        init_session_header(path, inquiry=q or "(empty)")
        self._vlm_prompt_transcript_path = path
        os.environ[ENV_KEY] = path

    def _navigation_keyframe_dir(self) -> str:
        """
        Root for representative_frames (anchor_key/). Falls back to cfg or output_path.
        """
        if getattr(self, "memory_cluster_manager", None) is not None:
            kd = getattr(self.memory_cluster_manager, "keyframe_dir", None)
            if kd:
                return os.path.abspath(os.path.expanduser(str(kd)))
        rp = getattr(self.cfg, "representative_frames_path", None)
        if rp:
            return os.path.abspath(os.path.expanduser(str(rp)))
        return os.path.abspath(os.path.join(self.cfg.output_path, "representative_frames"))

    def _clear_new_instance_dir(self):
        """Clears the new_instance directory to ensure fresh observations."""
        if hasattr(self, "new_instance_save_dir") and self.new_instance_save_dir:
            if os.path.exists(self.new_instance_save_dir):
                logger.warning(f"[Core] Clearing new_instance directory: {self.new_instance_save_dir}")
                import shutil
                shutil.rmtree(self.new_instance_save_dir)
                os.makedirs(self.new_instance_save_dir, exist_ok=True)

    def _nav_timing_reset_session(self) -> None:
        """Reset nav timing session for a new task."""
        self._nav_timing_session_id = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        self._nav_timing_stages = []
        logger.warning(
            f"[Core][NavTiming] new session_id={self._nav_timing_session_id} json_dir={self._nav_timing_json_dir}"
        )

    def _nav_timing_flush_json(self) -> None:
        """Write nav_timing_session.json and append nav_timing_events.jsonl."""
        try:
            os.makedirs(self._nav_timing_json_dir, exist_ok=True)
            doc: Dict[str, Any] = {
                "session_id": self._nav_timing_session_id,
                "inquiry": getattr(self, "inquiry", "") or "",
                "nav_inquiry_sentence": getattr(self, "nav_inquiry_sentence", None),
                "updated_at": time.time(),
                "stages": list(self._nav_timing_stages),
            }
            session_path = os.path.join(self._nav_timing_json_dir, "nav_timing_session.json")
            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            if self._nav_timing_stages:
                jlp = os.path.join(self._nav_timing_json_dir, "nav_timing_events.jsonl")
                with open(jlp, "a", encoding="utf-8") as f:
                    f.write(json.dumps(self._nav_timing_stages[-1], ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[Core][NavTiming] write failed {self._nav_timing_json_dir}: {e}")

    def _nav_timing_record(
        self,
        stage: str,
        duration_ms: float,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log one timing stage and flush JSON."""
        if self._nav_timing_session_id is None:
            self._nav_timing_reset_session()
        entry: Dict[str, Any] = {
            "ts": time.time(),
            "stage": stage,
            "duration_ms": round(float(duration_ms), 4),
        }
        if detail:
            entry["detail"] = detail
        self._nav_timing_stages.append(entry)
        extra = f" | detail={detail}" if detail else ""
        logger.warning(f"[Core][NavTiming] stage={stage} | ms={duration_ms:.2f}{extra}")
        self._nav_timing_flush_json()

    def _emit_nav_retrieve_nav_timing(
        self,
        runner_timing: Optional[Dict[str, Any]],
        navigation_path_planning_ms: float,
        subprocess_wall_ms: float,
    ) -> None:
        """NavRunner wall time + breakdown + global path planning."""
        r = runner_timing or {}
        local_sg = float(r.get("local_scene_graph_ms", 0.0) or 0.0)
        llm_ms = float(r.get("llm_anchor_proposal_ms", 0.0) or 0.0)
        vlm_ms = float(r.get("vlm_verification_ms", 0.0) or 0.0)
        nav_ms = float(navigation_path_planning_ms or 0.0)
        sw = float(subprocess_wall_ms or 0.0)

        self._nav_timing_record(
            "1_nav_runner_subprocess_wall_ms",
            sw,
            {"note": "includes subprocess CLIP/map load; often > sum of breakdown"},
        )
        self._nav_timing_record(
            "2_runner_local_scene_graph_ms",
            local_sg,
            {
                "compact_anchor_io_ms": r.get("compact_anchor_io_ms"),
                "prompt_build_ms": r.get("prompt_build_ms"),
                "local_clip_refinement_ms": r.get("local_clip_refinement_ms"),
            },
        )
        self._nav_timing_record("3_runner_llm_anchor_proposal_ms", llm_ms, None)
        self._nav_timing_record("4_runner_vlm_verification_ms", vlm_ms, None)
        self._nav_timing_record("5_initial_global_path_planning_ms", nav_ms, None)
        sum_core = local_sg + llm_ms + vlm_ms + nav_ms
        logger.warning(
            "[Core][NavTiming] summary (excl. extra subprocess load): "
            f"local_sg+llm+vlm+path={sum_core:.2f}ms | subprocess_wall={sw:.2f}ms"
        )

    def _plan_and_set_path(self, target_pose):
        """
        Plans a path to the target pose and triggers navigation (by saving the path).
        """
        # [Fix] 1. ,
        self._clear_new_instance_dir()
        
        # [Fix] 2.  calculate_path 
        self.calculate_path = False
        
        _t_replan = time.perf_counter()
        _replan_detail: Dict[str, Any] = {
            "is_relocalization_step": getattr(self, "is_relocalization_step", False),
        }
        try:
            from utils.navigation_helper import NavigationGraph
            import open3d as o3d
            
            goal_pos = target_pose[:3, 3]
            logger.warning(f"[Core] [Re-Nav] Planning path to target: {goal_pos}")
            
            # 1. Initialize NavGraph if needed
            nav_graph = self.global_map_manager.nav_graph
            if nav_graph is None:
                total_pcd = o3d.geometry.PointCloud()
                for obj in self.global_map_manager.global_map:
                    total_pcd += obj.pcd_2d
                
                if self.curr_pose is not None:
                    curr_point_coords = self.curr_pose[:3, 3]
                    curr_point = o3d.geometry.PointCloud()
                    curr_point.points = o3d.utility.Vector3dVector([curr_point_coords])
                    total_pcd += curr_point
                
                if hasattr(self.global_map_manager, 'layout_map') and self.global_map_manager.layout_map.wall_pcd is not None:
                    total_pcd += self.global_map_manager.layout_map.wall_pcd
                
                from utils.navigation_helper import NavigationGraph
                nav_graph = NavigationGraph(self.cfg, total_pcd, cell_size=0.03)
                self.global_map_manager.nav_graph = nav_graph
                nav_graph.get_graph()
            
            if self.curr_pose is None:
                logger.error("[Core] [Re-Nav] Current pose is None, cannot plan path.")
                _replan_detail["early_exit"] = "curr_pose_none"
                return

            # 2. Calculate Path
            curr_position = self.curr_pose[:3, 3]
            distance = np.linalg.norm(curr_position - goal_pos)
            
            logger.warning(f"[Core] [Re-Nav] 📍 Position Check:")
            logger.warning(f"  - Current: {curr_position}")
            logger.warning(f"  - Target : {goal_pos}")
            logger.warning(f"  - Distance: {distance:.3f} meters")
            
            if distance < 0.1:
                logger.warning("[Core] [Re-Nav] ⚠️ Target is extremely close (<10cm). Robot may only rotate.")
            
            start_2d = nav_graph.calculate_pos_2d(curr_position)
            goal_2d_raw = nav_graph.calculate_pos_2d(goal_pos)
            
            logger.warning(f"[Core] [Re-Nav] Coordinates: Start(3D)={curr_position}, Goal(3D)={goal_pos}")
            logger.warning(f"[Core] [Re-Nav] Grid Coords: Start(2D)={start_2d}, Goal(2D)={goal_2d_raw}")
            
            if not nav_graph.free_space_check(goal_2d_raw):
                goal_2d = nav_graph.snap_to_free_space_directional(goal_2d_raw, start_2d, nav_graph.free_space)
            else:
                goal_2d = goal_2d_raw
                
            path_2d = nav_graph.find_shortest_path(start_2d, goal_2d)
            
            if path_2d:
                self.curr_global_path = nav_graph.pos_path
                self.target_final_pose = target_pose
                
                target_rot = target_pose[:3, :3]
                if self.curr_pose is not None:
                    current_rot = self.curr_pose[:3, :3]
                    relative_rot = target_rot @ np.linalg.inv(current_rot)
                    from scipy.spatial.transform import Rotation as R
                    rot_vec = R.from_matrix(relative_rot).as_rotvec()
                    
                    self.target_rotation = rot_vec
                    self.target_rotation_matrix = target_rot
                    logger.warning(f"[Core] [Re-Nav] Rotation set: {rot_vec}")
                
                logger.warning(f"[Core] [Re-Nav] Path generated! Length: {len(self.curr_global_path)}")
                
                # 3. Save path to trigger runner
                if self.cfg.save_all_path:
                    # Reuse path counter logic if possible, or just increment
                    if not hasattr(self, 'path_counter'):
                        self.path_counter = 0
                    self.path_counter += 1
                    
                    save_dir = os.path.join(self.cfg.output_path, "path", "global_path")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    file_name = f"{self.path_counter}.json"
                    save_path = os.path.join(save_dir, file_name)
                    
                    with open(save_path, "w") as f:
                        json.dump(self.curr_global_path, f, indent=4)
                    
                    logger.warning(f"[Core] [Re-Nav] Global path saved to {save_path} (Triggers Robot)")
            else:
                logger.error("[Core] [Re-Nav] Failed to find path.")
                self.curr_global_path = None

            _replan_detail["path_found"] = self.curr_global_path is not None
                
        except Exception as e:
            logger.error(f"[Core] [Re-Nav] Error during path planning: {e}")
            import traceback
            logger.error(traceback.format_exc())
            _replan_detail["exception"] = repr(e)
        finally:
            replan_ms = (time.perf_counter() - _t_replan) * 1000.0
            self._nav_timing_record("replan_navigation_path_ms", replan_ms, _replan_detail)
