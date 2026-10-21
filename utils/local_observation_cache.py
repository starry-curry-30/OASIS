import gzip
import logging
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import open3d as o3d

from utils.types import DataInput, LocalObservation


class LocalObservationCache:
    """
    Per-frame LocalObservation cache:
    - mode: save / load / both
    - storage: gzip + pickle (internal), extension .obs.gz
    """

    def __init__(self, cfg: Dict):
        self.logger = logging.getLogger(__name__)
        self.enabled = bool(cfg.get("enabled", False))
        self.mode = cfg.get("mode", "save").lower()
        self.save_enabled = self.mode in ("save", "both")
        self.load_enabled = self.mode in ("load", "both")

        base_dir = cfg.get("output_dir", "cached_local_observations")
        self.output_dir = os.path.abspath(os.path.expanduser(str(base_dir)))

        if self.save_enabled:
            os.makedirs(self.output_dir, exist_ok=True)
            self.logger.info(
                "[ObsCache] save mode on, dir=%s", self.output_dir
            )
        if self.load_enabled:
            self.logger.info(
                "[ObsCache] load mode on, dir=%s", self.output_dir
            )

    def _frame_path(self, frame_idx: int) -> str:
        return os.path.join(self.output_dir, f"{frame_idx:06d}.obs.gz")

    def load(self, frame_idx: int) -> Optional[List[LocalObservation]]:
        if not (self.enabled and self.load_enabled):
            return None
        path = self._frame_path(frame_idx)
        if not os.path.exists(path):
            return None
        try:
            with gzip.open(path, "rb") as f:
                records = pickle.load(f)
        except Exception as exc:
            self.logger.warning(
                "[ObsCache] load frame %d failed: %s", frame_idx, exc
            )
            return None

        observations = [self._deserialize(rec) for rec in records]
        self.logger.debug(
            "[ObsCache] loaded frame %d, obs count=%d", frame_idx, len(observations)
        )
        return observations

    def save(self, frame_idx: int, observations: List[LocalObservation]) -> None:
        if not (self.enabled and self.save_enabled):
            return
        path = self._frame_path(frame_idx)
        records = [self._serialize(obs) for obs in observations]
        try:
            with gzip.open(path, "wb") as f:
                pickle.dump(records, f)
            self.logger.debug(
                "[ObsCache] saved frame %d, obs count=%d -> %s",
                frame_idx,
                len(observations),
                path,
            )
        except Exception as exc:
            self.logger.warning(
                "[ObsCache] save frame %d failed: %s", frame_idx, exc
            )

    def _serialize(self, obs: LocalObservation) -> Dict:
        mask = getattr(obs, "mask", None)
        mask_shape = None
        mask_data = None
        if mask is not None and mask.size > 0:
            mask_arr = np.asarray(mask, dtype=np.uint8)
            mask_shape = mask_arr.shape
            mask_data = np.packbits(mask_arr.reshape(-1))

        points = None
        colors = None
        if hasattr(obs, "pcd") and obs.pcd is not None:
            try:
                points = np.asarray(obs.pcd.points, dtype=np.float32)
                colors = np.asarray(obs.pcd.colors, dtype=np.float32)
                if points.size == 0:
                    points = None
                    colors = None
            except Exception:
                points = None
                colors = None

        bbox_min = None
        bbox_max = None
        if hasattr(obs, "bbox") and obs.bbox is not None:
            try:
                bbox_min = obs.bbox.get_min_bound().tolist()
                bbox_max = obs.bbox.get_max_bound().tolist()
            except Exception:
                bbox_min = None
                bbox_max = None

        data_input = getattr(obs, "data_input", None)

        return {
            "idx": obs.idx,
            "raw_image_idx": getattr(obs, "raw_image_idx", obs.idx),
            "class_id": int(getattr(obs, "class_id", -1)),
            "conf": float(getattr(obs, "conf", 0.0)),
            "distance": float(getattr(obs, "distance", 0.0)),
            "is_low_mobility": bool(getattr(obs, "is_low_mobility", False)),
            "xyxy": np.asarray(getattr(obs, "xyxy", []), dtype=np.float32),
            "clip_ft": np.asarray(getattr(obs, "clip_ft", []), dtype=np.float32),
            "mask_shape": mask_shape,
            "mask_data": mask_data,
            "pcd_points": points,
            "pcd_colors": colors,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "data_input_pose": np.asarray(getattr(data_input, "pose", None), dtype=np.float32)
            if data_input is not None
            else None,
            "data_input_intrinsics": np.asarray(getattr(data_input, "intrinsics", None), dtype=np.float32)
            if data_input is not None
            else None,
            "data_input_idx": getattr(data_input, "idx", None),
            "data_input_raw_idx": getattr(data_input, "raw_idx", None),
        }

    def _deserialize(self, record: Dict) -> LocalObservation:
        obs = LocalObservation()
        obs.idx = record.get("idx", 0)
        obs.raw_image_idx = record.get("raw_image_idx", obs.idx)
        obs.class_id = record.get("class_id", -1)
        obs.conf = record.get("conf", 0.0)
        obs.distance = record.get("distance", 0.0)
        obs.is_low_mobility = record.get("is_low_mobility", False)
        obs.xyxy = np.asarray(record.get("xyxy", []), dtype=np.float32)
        clip_ft = record.get("clip_ft")
        if clip_ft is not None:
            obs.clip_ft = np.asarray(clip_ft, dtype=np.float32)

        mask_shape = record.get("mask_shape")
        mask_data = record.get("mask_data")
        if mask_shape is not None and mask_data is not None:
            unpack = np.unpackbits(mask_data)[: int(np.prod(mask_shape))]
            obs.mask = unpack.reshape(mask_shape).astype(bool)
        else:
            obs.mask = np.empty((0, 0), dtype=bool)

        points = record.get("pcd_points")
        colors = record.get("pcd_colors")
        if points is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float32))
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float32))
            obs.pcd = pcd
        else:
            obs.pcd = o3d.geometry.PointCloud()

        bbox_min = record.get("bbox_min")
        bbox_max = record.get("bbox_max")
        if bbox_min is not None and bbox_max is not None:
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                np.asarray(bbox_min, dtype=np.float32),
                np.asarray(bbox_max, dtype=np.float32),
            )
            obs.bbox = bbox
        else:
            obs.bbox = o3d.geometry.AxisAlignedBoundingBox()

        pose = record.get("data_input_pose")
        intrinsics = record.get("data_input_intrinsics")
        if pose is not None:
            di = DataInput()
            di.idx = record.get("data_input_idx", obs.idx)
            di.raw_idx = record.get("data_input_raw_idx", obs.raw_image_idx)
            di.pose = np.asarray(pose, dtype=np.float32)
            if intrinsics is not None:
                di.intrinsics = np.asarray(intrinsics, dtype=np.float32)
            obs.data_input = di

        return obs

