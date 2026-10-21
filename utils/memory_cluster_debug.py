import gzip
import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from types import SimpleNamespace

logger = logging.getLogger(__name__)


@dataclass
class SerializedObservation:
    class_id: int
    class_name: Optional[str]
    conf: float
    xyxy: np.ndarray
    mask_bits: Optional[np.ndarray]
    mask_shape: Optional[tuple]
    matched_obj_uid: Optional[str] = None


def _pack_mask(mask: Optional[np.ndarray]) -> (Optional[np.ndarray], Optional[tuple]):
    if mask is None:
        return None, None

    mask_arr = np.asarray(mask)
    if mask_arr.size == 0:
        return None, None
    mask_bool = mask_arr.astype(bool)
    packed = np.packbits(mask_bool.reshape(-1))
    return packed, mask_bool.shape


def _unpack_mask(mask_bits: Optional[np.ndarray], mask_shape: Optional[tuple]) -> Optional[np.ndarray]:
    if mask_bits is None or mask_shape is None:
        return None
    total = int(np.prod(mask_shape))
    unpacked = np.unpackbits(mask_bits)[:total]
    return unpacked.reshape(mask_shape).astype(bool)


def serialize_observation(obs) -> SerializedObservation:
    mask_bits, mask_shape = _pack_mask(getattr(obs, "mask", None))
    xyxy = np.asarray(getattr(obs, "xyxy", np.zeros(4)), dtype=np.float32)
    matched_uid = getattr(obs, "matched_obj_uid", None)
    if matched_uid is not None:
        matched_uid = str(matched_uid)
    return SerializedObservation(
        class_id=int(getattr(obs, "class_id", -1)),
        class_name=getattr(obs, "class_name", None),
        conf=float(getattr(obs, "conf", 0.0)),
        xyxy=xyxy,
        mask_bits=mask_bits,
        mask_shape=mask_shape,
        matched_obj_uid=matched_uid,
    )


def deserialize_observation(serialized: SerializedObservation):
    mask = _unpack_mask(serialized.mask_bits, serialized.mask_shape)
    return SimpleNamespace(
        class_id=serialized.class_id,
        class_name=serialized.class_name,
        conf=serialized.conf,
        xyxy=np.asarray(serialized.xyxy, dtype=np.float32),
        mask=mask,
        matched_obj_uid=serialized.matched_obj_uid,
    )


def _ensure_dir(path: str, reset: bool) -> None:
    path_obj = Path(path)
    if path_obj.exists() and reset:
        for child in path_obj.glob("*"):
            if child.is_file():
                child.unlink()
            else:
                import shutil

                shutil.rmtree(child)
    path_obj.mkdir(parents=True, exist_ok=True)


class MemoryClusterDebugRecorder:
    """
    Helper to capture per-frame inputs for offline MemoryCluster debugging.
    """

    def __init__(
        self,
        cfg,
        class_lookup: Optional[Iterable[str]],
        anchor_whitelist: Optional[Iterable[str]],
    ) -> None:
        self.enabled = bool(cfg.get("enabled", False))
        if not self.enabled:
            return

        self.output_dir = os.path.abspath(
            os.path.expanduser(cfg.get("output_dir", "./outputs/memory_cluster_debug"))
        )
        self.reset_output = bool(cfg.get("reset_output", True))
        self.max_frames = cfg.get("max_frames")

        self._pending_frame = None
        self._pending_path = None
        self._saved = 0

        _ensure_dir(self.output_dir, self.reset_output)
        self._metadata_path = os.path.join(self.output_dir, "metadata.json")
        self._local_map_snapshot_path = os.path.join(
            self.output_dir, "local_map_snapshot.pkl.gz"
        )
        self._write_metadata(cfg, class_lookup, anchor_whitelist)

    def _write_metadata(self, cfg, class_lookup, anchor_whitelist) -> None:
        try:
            from omegaconf import OmegaConf

            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            try:
                cfg_dict = dict(cfg)
            except Exception:
                cfg_dict = {}

        meta = {
            "class_lookup": list(class_lookup or []),
            "anchor_whitelist": sorted(set(anchor_whitelist or [])),
            "max_frames": self.max_frames,
            "recorder_cfg": {
                k: v for k, v in (cfg_dict or {}).items() if k != "enabled"
            },
        }
        with open(self._metadata_path, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)

    def _should_skip(self) -> bool:
        if self.max_frames is None:
            return False
        return self._saved >= int(self.max_frames)

    def capture_frame(self, frame_idx: int, data_input, observations: List) -> None:
        if not self.enabled or self._should_skip():
            return

        serialized_obs = [serialize_observation(obs) for obs in observations or []]
        payload = {
            "frame_idx": int(frame_idx),
            "raw_idx": int(getattr(data_input, "raw_idx", frame_idx)),
            "time_stamp": float(getattr(data_input, "time_stamp", 0.0)),
            "color_name": getattr(data_input, "color_name", ""),
            "pose": np.asarray(data_input.pose).astype(np.float32),
            "intrinsics": np.asarray(data_input.intrinsics).astype(np.float32),
            "color": np.asarray(data_input.color).astype(np.uint8).copy(),
            "observations": serialized_obs,
            "clip_feature": None,
        }
        self._pending_frame = payload
        self._pending_path = os.path.join(self.output_dir, f"frame_{frame_idx:06d}.pkl.gz")

    def attach_clip_feature(self, clip_feature: Optional[np.ndarray]) -> None:
        if not self.enabled or self._pending_frame is None:
            return
        if clip_feature is None:
            self._pending_frame["clip_feature"] = None
        else:
            self._pending_frame["clip_feature"] = np.asarray(clip_feature, dtype=np.float32)

    def finalize_frame(self) -> None:
        if not self.enabled or self._pending_frame is None or self._should_skip():
            self._pending_frame = None
            self._pending_path = None
            return

        try:
            with gzip.open(self._pending_path, "wb") as fp:
                pickle.dump(self._pending_frame, fp, protocol=pickle.HIGHEST_PROTOCOL)
            self._saved += 1
        except OSError as exc:
            logger.warning("[MemoryClusterDebug] Failed to write frame %s: %s", self._pending_path, exc)
        finally:
            self._pending_frame = None
            self._pending_path = None

    def abort_frame(self) -> None:
        self._pending_frame = None
        self._pending_path = None

    def save_local_map_snapshot(self, local_map_manager) -> None:
        if not self.enabled or local_map_manager is None:
            return
        entries = []
        local_map = getattr(local_map_manager, "local_map", [])
        for obj in local_map or []:
            entry = self._extract_local_map_entry(obj)
            if entry is not None:
                entries.append(entry)
        payload = {"objects": entries}
        try:
            with gzip.open(self._local_map_snapshot_path, "wb") as fp:
                pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError as exc:
            logger.warning("[MemoryClusterDebug] Snapshot save failed: %s", exc)

    def _extract_local_map_entry(self, obj) -> Optional[dict]:
        if not getattr(obj, "is_stable", False):
            return None
        class_id = getattr(obj, "m_class_id", getattr(obj, "class_id", None))
        center = self._get_obj_center(obj)
        extent = self._get_obj_extent(obj)
        if center is None:
            return None
        return {
            "uid": str(getattr(obj, "uid", "")),
            "is_stable": bool(getattr(obj, "is_stable", False)),
            "class_id": int(class_id) if class_id is not None else None,
            "m_class_id": int(getattr(obj, "m_class_id")) if getattr(obj, "m_class_id", None) is not None else None,
            "observed_num": int(getattr(obj, "observed_num", 0) or 0),
            "center": center,
            "extent": extent,
        }

    def _get_obj_center(self, obj) -> Optional[List[float]]:
        bbox = getattr(obj, "bbox", None)
        if bbox is not None:
            try:
                return np.asarray(bbox.get_center(), dtype=np.float32).tolist()
            except Exception:
                pass
        if hasattr(obj, "center"):
            try:
                return list(map(float, obj.center[:3]))
            except Exception:
                return None
        pcd = getattr(obj, "pcd", None)
        if pcd is not None:
            try:
                pts = np.asarray(pcd.points)
                if pts.size > 0:
                    return pts.mean(axis=0).astype(np.float32).tolist()
            except Exception:
                return None
        return None

    def _get_obj_extent(self, obj) -> Optional[List[float]]:
        bbox = getattr(obj, "bbox", None)
        if bbox is not None:
            try:
                return np.asarray(bbox.get_extent(), dtype=np.float32).tolist()
            except Exception:
                return None
        return None


def load_metadata(replay_dir: str) -> dict:
    meta_path = os.path.join(replay_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found in {replay_dir}")
    with open(meta_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_frame(frame_path: str) -> dict:
    with gzip.open(frame_path, "rb") as fp:
        return pickle.load(fp)


def build_observations(serialized_list: List[SerializedObservation]):
    return [deserialize_observation(item) for item in serialized_list]


def load_local_map_snapshot(replay_dir: str) -> Optional[dict]:
    path = os.path.join(replay_dir, "local_map_snapshot.pkl.gz")
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rb") as fp:
        return pickle.load(fp)


def build_stub_local_map_manager(snapshot: Optional[dict], class_lookup: Optional[List[str]]):
    if not snapshot:
        return None

    class _StubObjClasses:
        def __init__(self, classes):
            self._classes = classes or []

        def get_classes_arr(self):
            return self._classes

    class _StubBBox:
        def __init__(self, center, extent):
            self._center = np.asarray(center, dtype=np.float32)
            self._extent = np.asarray(extent, dtype=np.float32)

        def get_center(self):
            return self._center

        def get_extent(self):
            return self._extent

    objects = []
    for item in snapshot.get("objects", []):
        center = item.get("center")
        if center is None:
            continue
        extent = item.get("extent")
        bbox = None
        if extent is not None:
            bbox = _StubBBox(center, extent)
        obj = SimpleNamespace(
            uid=item.get("uid"),
            is_stable=item.get("is_stable", False),
            class_id=item.get("class_id"),
            m_class_id=item.get("m_class_id"),
            observed_num=item.get("observed_num", 0),
            bbox=bbox,
            pcd=None,
        )
        objects.append(obj)

    visualizer = SimpleNamespace(obj_classes=_StubObjClasses(class_lookup))
    return SimpleNamespace(local_map=objects, visualizer=visualizer)

