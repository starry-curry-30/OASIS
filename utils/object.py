# Standard library imports
import copy
import logging
import os
import pdb
import pickle
import uuid
from collections import Counter, deque
from enum import Enum
from typing import List, Optional

import math

# Third-party imports
import numpy as np
import open3d as o3d
from omegaconf import DictConfig

# Local module imports
from utils.types import Observation

# Set up the module-level logger
logger = logging.getLogger(__name__)


class LocalObjStatus(Enum):
    UPDATING = "updating"
    PENDING = "pending for updating"
    ELIMINATION = "elimination"
    LM_ELIMINATION = "elimination for low mobility"
    HM_ELIMINATION = "elimination for high mpbility"
    WAITING = "waiting for stable obj process"


class BaseObject:

    # Global variable for config
    _cfg = None

    def __init__(self):
        # id
        self.uid = uuid.uuid4()

        # obs info
        self.observed_num = 0
        self.observations: List[str] = []

        # Spatial primitives
        self.pcd: Optional[o3d.geometry.PointCloud] = o3d.geometry.PointCloud()
        self.bbox: Optional[o3d.geometry.AxisAlignedBoundingBox] = (
            o3d.geometry.AxisAlignedBoundingBox()
        )

        # high level feats
        self.clip_ft: Optional[np.ndarray] = np.empty(0, dtype=np.float32)

        # class id
        self.class_id: Optional[int] = None

        # Initialize save_path
        self.save_path = self._initialize_save_path()

        # is navigation goal flag
        self.nav_goal = False

    def __getstate__(self):
        # Prepare the state dictionary for serialization
        state = {
            "uid": self.uid,
            "pcd_points": np.asarray(self.pcd.points).tolist(),  # Convert to list
            "pcd_colors": np.asarray(self.pcd.colors).tolist(),  # Convert to list
            "clip_ft": self.clip_ft.tolist(),
            "class_id": self.class_id,
            "nav_goal": self.nav_goal,
        }
        return state

    def __setstate__(self, state):
        self.uid = state.get("uid")

        # Restore PointCloud from points & colors
        points = np.array(state.get("pcd_points"))
        colors = np.array(state.get("pcd_colors"))

        if (len(points) != 0) or (len(colors) != 0):
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            # Recompute bbox from pcd
            self.bbox = self.pcd.get_axis_aligned_bounding_box()
        else:
            self.pcd = o3d.geometry.PointCloud()
            self.bbox = o3d.geometry.AxisAlignedBoundingBox()

        self.clip_ft = np.array(state.get("clip_ft"))
        self.class_id = state.get("class_id")
        self.nav_goal = state.get("nav_goal")

        self.observed_num = 0
        self.observations: List[str] = []

        self.save_path = self._initialize_save_path()

    @classmethod
    def initialize_config(cls, config: DictConfig):
        cls._cfg = config

        classes_path = config.yolo.classes_path
        if config.yolo.use_given_classes:
            classes_path = config.yolo.given_classes_path
            logger.info(f"[BaseObject] Using given classes, path:{classes_path}")

        with open(classes_path, "r") as file:
            lines = file.readlines()
            cls.obj_classes = [line.strip() for line in lines]
            num_classes = len(lines)
        # set num_classes for bayesian class filter
        cls._cfg.yolo.num_classes = num_classes

        # Load anchor classes
        cls.anchor_classes = set()
        anchor_path = getattr(config.yolo, 'given_anchor_classes', None)
        if anchor_path and os.path.exists(anchor_path):
            with open(anchor_path, "r") as f:
                cls.anchor_classes = {line.strip().lower() for line in f if line.strip()}
            logger.info(f"[BaseObject] Loaded {len(cls.anchor_classes)} anchor classes from {anchor_path}")

    def _initialize_save_path(self):
        if self._cfg:
            # save dir construction
            save_dir = self._cfg.map_save_path
            # If not exist, then create
            os.makedirs(save_dir, exist_ok=True)
            return os.path.join(save_dir, f"{self.uid}.pkl")

        return None

    def copy(self):
        return copy.deepcopy(self)

    def save_to_disk(self):
        """Save the object to disk using pickle."""
        with open(self.save_path, "wb") as f:
            pickle.dump(self, f)

        if self._cfg.save_cropped:
            # save the cropped image in the observation
            save_dir = self._cfg.map_save_path
            save_dir = os.path.join(save_dir, f"{self.class_id}_{self.uid}")
            cropped_save_dir = os.path.join(save_dir, "cropped")
            masked_save_dir = os.path.join(save_dir, "masked")
            os.makedirs(cropped_save_dir, exist_ok=True)
            os.makedirs(masked_save_dir, exist_ok=True)
            for obs in self.observations:
                obs_idx = obs.idx
                cropped_image = obs.cropped_image
                masked_image = obs.masked_image
                cropped_image_dir = os.path.join(cropped_save_dir, f"{obs_idx}.png")
                masked_image_dir = os.path.join(masked_save_dir, f"{obs_idx}.png")
                # both cropped and masked images are np.ndarray, so save as png
                import imageio.v2 as imageio

                imageio.imwrite(cropped_image_dir, cropped_image)
                imageio.imwrite(masked_image_dir, masked_image)

    @staticmethod
    def load_from_disk(filename: str):
        """Load the object from disk using pickle."""
        with open(filename, "rb") as f:
            obj = pickle.load(f)
            return obj

    def voxel_downsample_2d(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float,
    ) -> o3d.geometry.PointCloud:
        # Input 3d point cloud, voxel size, return downsampled 2d point cloud
        # This function is to avoid the warning caused by o3d.geometry.voxel_down_sample
        # TODO: Color is not right

        # Get point cloud's points
        points_arr = np.asarray(pcd.points)
        colors_arr = np.asarray(pcd.colors)

        # Only retain X and Y coordinates
        points_2d = points_arr[:, :2]

        # 2D voxel downsample based on voxel size
        grid_indices = np.floor(points_2d / voxel_size).astype(np.int32)
        unique_indices, inverse_indices = np.unique(
            grid_indices, axis=0, return_inverse=True
        )

        # Calculate average points for each voxel
        downsampled_points_2d = np.zeros_like(unique_indices, dtype=np.float64)
        downsampled_colors = np.zeros((len(unique_indices), 3), dtype=np.float64)

        # calculate the mean of points in each voxel
        for i in range(len(unique_indices)):
            mask = inverse_indices == i
            downsampled_points_2d[i] = points_2d[mask].mean(axis=0)
            downsampled_colors[i] = colors_arr[mask].mean(axis=0)

        # restore the Z with given Z
        downsampled_points = np.zeros((len(downsampled_points_2d), 3))
        downsampled_points[:, :2] = downsampled_points_2d
        downsampled_points[:, 2] = self._cfg.floor_height

        # Generate the downsampled point cloud
        downsampled_pcd = o3d.geometry.PointCloud()
        downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
        downsampled_pcd.colors = o3d.utility.Vector3dVector(downsampled_colors)

        return downsampled_pcd


class LocalObject(BaseObject):

    # Global variable for overall idx
    _curr_idx = 0

    def __init__(self):
        super().__init__()

        # lm sign
        self.is_low_mobility: Optional[bool] = False
        # major plane info, the z value of the major plane
        self.major_plane_info = None

        # Split Check info dict
        self.split_info: Optional[dict] = {}
        self.max_common: int = 0
        self.should_split: Optional[bool] = False
        # debug for split feat
        self.split_class_id_one: Optional[int] = 0
        self.split_class_id_two: Optional[int] = 0

        # Spatial Stablity Check Info List
        self.spatial_stable_info: Optional[list] = []

        # status
        self.status = LocalObjStatus.UPDATING
        self.is_stable = False
        self.pending_count = 0
        self.waiting_count = 0

        # # bayesian stable
        self.num_classes = self._cfg.yolo.num_classes
        # # init the prob
        self.class_probs = np.ones(self.num_classes) / self.num_classes
        self.class_probs_history: List[str] = []
        self.max_prob = 0.0
        self.entropy = 0.0
        self.change_rate = 0.0

        # for local map merging
        self.is_merged = False

        self.best_frames_realtime: List[int] = []  # 
        self.best_frames_max_k: int = getattr(self._cfg, 'realtime_best_frames', {}).get('max_frames_per_object', 5)  # 
        self.last_best_frame_update: int = -1  # 
        
        ################
        # Debug Variable
        ################
        self.downsample_num: int = 0

    @classmethod
    def set_curr_idx(cls, idx: int):
        cls._curr_idx = idx

    def add_observation(self, observation: Observation) -> None:
        self.observations.append(observation)
        self.observed_num += 1

        self.update_best_frames_realtime()

    def get_latest_observation(self) -> Observation:
        return self.observations[-1] if self.observations else None

    def clear_info(
        self,
    ) -> None:
        self.observed_num = 0
        self.observations = []
        self.pcd = o3d.geometry.PointCloud()
        # self.bbox = o3d.geometry.OrientedBoundingBox()
        self.class_id = None
        self.split_info = None
        self.max_common = 0
        self.should_split = False
        self.split_class_id_one = None
        self.split_class_id_two = None
        self.spatial_stable_info = None

    # Baye: Update posterior probability
    def update_class_probs(
        self,
        alpha=0.6,
    ) -> None:

        latest_obs = self.get_latest_observation()

        distance = latest_obs.distance

        # Current observation's confidence and class
        class_id = latest_obs.class_id
        conf = latest_obs.conf

        # Create class probability distribution for observation
        observation_probs = np.zeros(self.num_classes)

        # Distribute remaining confidence evenly among other classes
        remaining_conf = 1 - conf

        # Find top 10 classes with highest scores excluding current class
        all_indices = np.arange(self.num_classes)
        other_indices = all_indices[
            all_indices != class_id
        ]  # Indices excluding current class
        top_10_indices = np.argsort(self.class_probs[other_indices])[
            -10:
        ]  # Get top 10 classes with highest scores
        top_10_real_indices = other_indices[top_10_indices]  # Get real class indices

        # Distribute remaining confidence evenly among these 3 classes
        for idx in top_10_real_indices:
            observation_probs[idx] = remaining_conf / 10

        # Confidence of the current observation class
        observation_probs[class_id] = conf
        observation_probs /= np.sum(observation_probs)

        # Sliding window smoothing
        window_size = 5

        self.class_probs_history.append(observation_probs)
        if len(self.class_probs_history) > window_size:
            self.class_probs_history.pop(0)

        smoothed_probs = np.mean(self.class_probs_history, axis=0)

        # calculate alpha
        max_distance = 10.0
        k = 1.0
        alpha = np.exp(-k * distance / max_distance)

        # Bayesian update
        self.class_probs = (1 - alpha) * self.class_probs + alpha * smoothed_probs

        # Normalize
        self.class_probs /= np.sum(self.class_probs)

    def update_info(self) -> None:
        # Local Obj
        # Integrate information from observation into external
        # This is the update part in the filter
        # TODO: Check how many times is appropriate for point cloud downsampling

        latest_obs = self.get_latest_observation()

        if self.observed_num == 0:
            logger.error("[LocalObject] No observation in this object")
            return

        if self.observed_num == 1:
            self.pcd = latest_obs.pcd
            self.bbox = latest_obs.bbox
            self.clip_ft = latest_obs.clip_ft
            self.class_id = latest_obs.class_id
            self.is_low_mobility = latest_obs.is_low_mobility
            return

        # Split dict update
        self.update_split_info(latest_obs)
        if self.should_split:
            return

        # Get outside infomations
        # Merge pcd
        # Simply add
        self.pcd += latest_obs.pcd

        # Get new bbox
        # self.bbox = self.pcd.get_oriented_bounding_box(robust=True)
        self.bbox = self.pcd.get_axis_aligned_bounding_box()

        # Merge clip_ft
        self.clip_ft = (
            (self.clip_ft * (self.observed_num - 1) + latest_obs.clip_ft)
            * 1.0
            / (self.observed_num)
        )
        # Normalize the new clip_ft
        norm = np.linalg.norm(self.clip_ft)
        self.clip_ft = self.clip_ft / norm

        # Update spatial stable info list
        self.update_spatial_stable_info(latest_obs)

        # Get major class id
        # get class id list
        class_ids = [obs.class_id for obs in self.observations]
        obj_class_id_counter = Counter(class_ids)
        most_common_class_id = obj_class_id_counter.most_common(1)[0][0]
        self.class_id = most_common_class_id

        # Bayesian update
        self.update_class_probs()

        # Get low mobility info
        # get low mobility info list
        low_mobility_infos = [obs.is_low_mobility for obs in self.observations]
        num_true = sum(low_mobility_infos)
        num_false = len(low_mobility_infos) - num_true
        most_common_lm = True if num_true > num_false else False

        self.is_low_mobility = most_common_lm

        # Periodically downsample the pcd
        if self.observed_num % self._cfg.downsample_interval == 0:
            self.downsample_num += 1
            self.pcd = self.pcd.voxel_down_sample(
                voxel_size=self._cfg.downsample_voxel_size
            )

            # only after downsample, we calculate the major plane info
            # If the object is low mobility, do major plane info calculation
            # TODO: do not need to calculate major plane info from scratch, incrementally calculate z_value
            # The major plane info can be calculated in the Observation Generation
            # Merging using the point number as the weight process
            if self.is_low_mobility:
                self.major_plane_info = self.find_major_plane_info()
            else:
                self.major_plane_info = None

    def update_split_info(self, latest_obs: Observation) -> None:
        # https://www.yuque.com/u21262689/fxzc7g/fmfk1gkv6fbemlus?singleDoc#
        # Boundary condition check
        if latest_obs is None:
            raise ValueError("latest_obs cannot be None")
        if not hasattr(latest_obs, "class_id") or not hasattr(latest_obs, "idx"):
            raise ValueError("latest_obs must have class_id and idx attributes")

        # Split dict update
        if latest_obs.class_id not in self.split_info:
            self.split_info[latest_obs.class_id] = deque()
        self.split_info[latest_obs.class_id].append(latest_obs.idx)

        # Make sure the idx in the active window
        # Current window size is 10
        for class_id, idx_deque in self.split_info.items():
            while (
                idx_deque
                and idx_deque[0] <= latest_obs.idx - self._cfg.active_window_size
            ):
                idx_deque.popleft()
            # # if empty, delete
            # if len(idx_deque) == 0:
            #     del self.split_info[class_id]
            if len(idx_deque) < self._cfg.active_window_size:
                continue

        # logger.info("current idx: ", latest_obs.idx)
        # for class_id, idx_deque in self.split_info.items():
        #     logger.info(f"Class ID: {class_id}, Observations: {list(idx_deque)}")

        # TODO: WHAT about three deque? currently only two considered
        split_tuple = self.find_max_common_elements(self.split_info)
        # logger.info slipt tupe
        # logger.info("Max common elements:", split_tuple)
        self.max_common = split_tuple[0]
        if split_tuple[0] != 0:
            self.split_class_id_one = split_tuple[1][0]
            self.split_class_id_two = split_tuple[1][1]
        else:
            self.split_class_id_one = 0
            self.split_class_id_two = 0

        if self.max_common > self._cfg.max_common_th:
            self.should_split = True

    def print_split_info(self) -> None:
        # logger.info split info into a string
        for class_id, idx_deque in self.split_info.items():
            logger.info(
                f"[LocalObject] Class ID: {class_id}, Observations: {list(idx_deque)}"
            )

    def print_split_info(
        self,
    ) -> str:
        split_info = ""
        for class_id, idx_deque in self.split_info.items():
            string_split = f"{class_id}" + "-" + f"{list(idx_deque)}"
            split_info = split_info + string_split + "||"
        return split_info

    def find_max_common_elements(
        self,
        data: dict,
    ) -> tuple:
        # Find max common elements in split info

        max_common_count = 0
        max_common_pair = (None, None)

        class_ids = list(data.keys())

        for i in range(len(class_ids)):
            for j in range(i + 1, len(class_ids)):
                class_id1 = class_ids[i]
                class_id2 = class_ids[j]

                deque1 = data[class_id1]
                deque2 = data[class_id2]

                # jump the empty deque
                if not deque1 or not deque2:
                    continue

                set1 = set(deque1)
                set2 = set(deque2)

                common_elements = set1 & set2
                common_count = len(common_elements)

                if common_count > max_common_count:
                    max_common_count = common_count
                    max_common_pair = (class_id1, class_id2)

        max_common_tuple = (max_common_count, max_common_pair)

        return max_common_tuple

    def update_info_from_observations(
        self,
    ) -> None:
        # This function will be used to update the object class info based on all observations
        # ALso can use iteration of the update_info function
        counter = 0
        for obs in self.observations:
            counter += 1
            if counter == 1:
                self.pcd = obs.pcd
                self.clip_ft = obs.clip_ft
                continue
            self.pcd += obs.pcd
            self.clip_ft += obs.clip_ft

        # downsample pcd
        self.pcd = self.pcd.voxel_down_sample(
            voxel_size=self._cfg.downsample_voxel_size
        )
        # Group to majority
        from utils.pcd_utils import init_pcd_denoise_dbscan

        self.pcd = init_pcd_denoise_dbscan(
            self.pcd, self._cfg.dbscan_eps, self._cfg.dbscan_min_points
        )

        # self.bbox = self.pcd.get_oriented_bounding_box(robust=True)
        self.bbox = self.pcd.get_axis_aligned_bounding_box()

        # norm feat
        self.clip_ft = (self.clip_ft) * 1.0 / (self.observed_num)
        # Normalize the new clip_ft
        norm = np.linalg.norm(self.clip_ft)
        self.clip_ft = self.clip_ft / norm

        # Get major class id
        # get class id list
        class_ids = [obs.class_id for obs in self.observations]
        obj_class_id_counter = Counter(class_ids)
        most_common_class_id = obj_class_id_counter.most_common(1)[0][0]
        self.class_id = most_common_class_id

        # Get low mobility info
        # get low mobility info list
        low_mobility_infos = [obs.is_low_mobility for obs in self.observations]
        num_true = sum(low_mobility_infos)
        num_false = len(low_mobility_infos) - num_true
        most_common_lm = True if num_true > num_false else False

        self.is_low_mobility = most_common_lm

        # get major plane info
        if self.is_low_mobility:
            self.major_plane_info = self.find_major_plane_info()

    def update_spatial_stable_info(self, latest_obs: Observation) -> None:
        # Including two parts of the spatial check
        # 1. latest obs bbox( or other primitives ) compare with the overall bbox or pcd
        # 2. latest obs bbox with prev obs bbox
        pass

    def update_status(
        self,
    ) -> None:
        # Object life cycle
        # Updating the status of the object by stability and infos

        # last observation time
        last_obs = self.get_latest_observation()

        # 1. if the object is in inside the sliding window, status will be UPDATING
        # No matter what previous status is, the object will always be UPDATING if in the window
        if last_obs.idx <= self._curr_idx and last_obs.idx >= max(
            self._curr_idx - self._cfg.active_window_size, 0
        ):
            self.status = LocalObjStatus.UPDATING
            self.pending_count = 0
            self.waiting_count = 0
            return

        # 2. if the object is out of the sliding window, check the stability first
        # and now the object is out of the window
        # Once the object is set as stable, it will always be stable
        self.stability_check()

        # if not stable, pending for next update, set status to PENDING
        if self.is_stable == False:
            self.status = LocalObjStatus.PENDING
            self.pending_count += 1

            # if pending count is large, status set as ELIMINATION
            if self.pending_count > self._cfg.max_pending_count:
                self.status = LocalObjStatus.ELIMINATION
                return

            return

        # 3. if the object is stable, waiting first
        # and now the obj is set as stable
        self.status = LocalObjStatus.WAITING
        self.waiting_count += 1

        if self.waiting_count < self._cfg.max_pending_count:
            # still in the waiting status, just return
            return

        # if waiting enough, then judge next step by lm status
        if self.is_low_mobility:
            self.status = LocalObjStatus.LM_ELIMINATION
            return
        else:
            self.status = LocalObjStatus.HM_ELIMINATION
            return

    def stability_check(
        self,
    ) -> None:
        # Check if the object is stable
        # the only function change the is_stable flag
        # Enhance the ways of stability check, here we only use label check
        # The initial check here is very simple, then goes bayesian stability check

        # 1. obs num should be large
        if self.observed_num < self._cfg.stable_num:
            self.is_stable = False
            return

        # 2. if the largest label over 1/2 of the observed num, just set as stable
        class_ids = [obs.class_id for obs in self.observations]
        obj_class_id_counter = Counter(class_ids)
        most_common_class_id, most_common_count = obj_class_id_counter.most_common(1)[0]

        if most_common_count > self.observed_num / 3:
            self.is_stable = True
            return

        # 3. if the object is stable by the filter, then set as stable
        if self.is_class_converged():
            self.is_stable = True
            return

        self.is_stable = False

    def is_class_converged(
        self,
        entropy_threshold=0.2,
        prob_threshold=0.50,
        change_rate_threshold=0.2,
        window_size=3,
    ) -> bool:

        # 1. Major class probability check
        max_prob = np.max(self.class_probs)
        self.max_prob = max_prob
        if max_prob > prob_threshold:
            return True

        # 2. Entropy check
        entropy = -np.sum(
            self.class_probs * np.log(self.class_probs + 1e-10)
        )  # prevent log(0)
        self.entropy = entropy
        if entropy < entropy_threshold:
            return True

        # 3. Change rate check
        if len(self.class_probs_history) >= window_size:
            recent_probs = np.array(self.class_probs_history[-window_size:])
            change_rate = np.mean(np.abs(recent_probs[1:] - recent_probs[:-1]), axis=0)
            self.change_rate = np.max(change_rate)
            if np.max(change_rate) < change_rate_threshold:
                return True

        return False

    def find_major_plane_info(
        self,
        bin_size=0.02,
    ) -> Optional[float]:
        # This function will find the major plane of the object
        # return the major plane z value
        pts = np.asarray(self.pcd.points)
        if pts.size == 0 or pts.shape[0] == 0:
            return None
        z_axis = pts[:, 2].astype(np.float64, copy=False)
        z_min, z_max = float(z_axis.min()), float(z_axis.max())
        if not np.isfinite(z_min) or not np.isfinite(z_max):
            return None
        #  arange ,histogram  hist
        if z_max - z_min < 1e-9:
            return z_min

        bin_edges = np.arange(z_min, z_max + bin_size, bin_size)
        if len(bin_edges) < 2:
            return float(np.median(z_axis))

        hist, bin_edges = np.histogram(z_axis, bins=bin_edges)
        if hist.size == 0 or int(np.sum(hist)) == 0:
            return float(np.median(z_axis))

        peak_index = int(np.argmax(hist))
        major_plane_z = (bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2.0
        return float(major_plane_z)

    def update_best_frames_realtime(self) -> None:
        """
        ,
        
        """
        # ,
        if self.observed_num < 3:
            return
            
        # (5,)
        update_interval = 5
        if (self.observed_num - self.last_best_frame_update) < update_interval and self.last_best_frame_update != -1:
            return
            
        try:
            # ,
            best_idxs, _, _, _, _ = self.get_best_observation_indices(
                k=min(self.best_frames_max_k, self.observed_num),
                method="weighted",
                verbose=False
            )
            
            if isinstance(best_idxs, np.ndarray):
                best_idxs = best_idxs.tolist()
                
            # 
            self.best_frames_realtime = [int(idx) for idx in best_idxs if idx is not None]
            self.last_best_frame_update = self.observed_num
            
        except Exception as e:
            # ,
            if hasattr(self, 'observations') and self.observations:
                confs_with_idx = [
                    (float(getattr(obs, "conf", 0.0)), int(getattr(obs, "idx", i)))
                    for i, obs in enumerate(self.observations)
                    if getattr(obs, "idx", None) is not None
                ]
                confs_with_idx.sort(key=lambda x: -x[0])  # 
                self.best_frames_realtime = [
                    idx for _, idx in confs_with_idx[:self.best_frames_max_k]
                ]
                self.last_best_frame_update = self.observed_num

    def get_best_observation_indices(
        self,
        k: int = 1,
        method: str = "unified",  # "conf" | "edge" | "support" | "weighted" | "unified"
        alpha: float = 0.15,  # :(weighted)
        beta: float = 0.3,   # :(weighted)
        gamma: float = 0.3,  # :(weighted)
        delta: float = 0.25, # :(weighted)
        verbose: bool = False,
        confidence_threshold: float = 0.6,  # 
        related_objs: Optional[List["LocalObject"]] = None,
        class_lookup: Optional[List[str]] = None,
        include_observation_refs: bool = False,
        candidate_count: Optional[int] = None,
        diversity_angle_deg: float = 30.0,
        distance_levels: int = 3,
        optimal_distance_min: float = 1.0,  # 
        optimal_distance_max: float = 3.5,  # 
        # 
        local_map_manager=None,  # unified,local_map
        unified_alpha_a: float = 0.7,
        unified_alpha_o: float = 0.3,
        unified_A_max: float = 5.0,
        unified_O_max: float = 10.0,
        unified_beta_e: float = 0.5,
        unified_beta_c: float = 0.5,
        unified_C_max: float = 5.0,
        unified_gamma_1: float = 0.6,
        unified_gamma_2: float = 0.4,
    ):
        """
         k "",.

        :
        1. ()
        2. SAM (, 1-)
        3. ()
        4. ()

        - method="conf":  obs.conf 
        - method="edge": 
        - method="support": 
        - method="weighted": α*norm(conf) + β*norm(edge_quality) + γ*norm(support_ratio) + δ*norm(distance_score)

        : (indices, info_list, high_conf_indices)
            indices: List[int] 
            info_list: List[dict] 
            high_conf_indices: List[int] ()
        """
        if self.observed_num == 0 or not hasattr(self, "observations"):
            return [], [], [], [], []

        def get_class_name(cls_id: Optional[int]) -> str:
            if cls_id is None:
                return "unknown"
            if class_lookup and 0 <= cls_id < len(class_lookup):
                return class_lookup[cls_id]
            return str(cls_id)

        final_class_id = getattr(self, "final_class_id", getattr(self, "class_id", None))
        anchor_class_name = get_class_name(final_class_id)

        related_objs = related_objs or []
        support_lookup = []
        for related_obj in related_objs:
            final_rel_class = getattr(
                related_obj,
                "final_class_id",
                getattr(related_obj, "class_id", None),
            )
            related_class_name = get_class_name(final_rel_class)
            observed_frames = set()
            for related_obs in getattr(related_obj, "observations", []):
                obs_idx = getattr(related_obs, "idx", getattr(related_obs, "idxx", -1))
                if obs_idx < 0:
                    continue
                rel_class_id = getattr(related_obs, "class_id", None)
                if final_rel_class is None or rel_class_id == final_rel_class:
                    observed_frames.add(int(obs_idx))
            support_lookup.append(
                {"name": related_class_name, "frames": observed_frames}
            )

        expected_support_names = [entry["name"] for entry in support_lookup]

        candidate_infos = []

        filtered_observations = []
        for obs in self.observations:
            obs_class_id = getattr(obs, "class_id", None)

            # 
            if (
                final_class_id is not None
                and obs_class_id is not None
                and obs_class_id != final_class_id
            ):
                continue

            filtered_observations.append(obs)

        if not filtered_observations:
            return [], [], [], [], []

        for obs in filtered_observations:
            frame_idx = int(getattr(obs, "idx", getattr(obs, "idxx", -1)))
            if frame_idx < 0:
                continue

            conf = float(getattr(obs, "assured_conf", getattr(obs, "conf", 0.0)))

            # 
            mask = getattr(obs, "mask", None)
            edge_touch_ratio = 0.0
            total_pixels = 0
            edge_pixel_count = 0
            edge_contact_count = 0

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
                        edge_touch_ratio = edge_contact_count / float(edge_pixel_count) if edge_pixel_count > 0 else 0.0

            edge_touch_ratio = min(max(edge_touch_ratio, 0.0), 1.0)
            edge_quality = 1.0 - edge_touch_ratio

            # 
            recognized = 0
            recognized_names: List[str] = []
            if support_lookup:
                for entry in support_lookup:
                    if frame_idx in entry["frames"]:
                        recognized += 1
                        recognized_names.append(entry["name"])

            total_related = len(support_lookup)
            support_score = recognized / total_related if total_related > 0 else 1.0

            # 
            view_dir = None
            distance_val = float(getattr(obs, "distance", float("nan")))
            original_distance_val = distance_val
            cam_origin = None
            obj_center = None
            norm_val = float("nan")
            distance_source = "observation" if math.isfinite(distance_val) else "unknown"
            try:
                data_input = getattr(obs, "data_input", None)
                pose = getattr(data_input, "pose", None)
                if pose is not None:
                    cam_origin = pose[:3, 3]
                    pcd = getattr(obs, "pcd", None)
                    if pcd is not None and len(pcd.points) > 0:
                        obj_center = np.asarray(pcd.points).mean(axis=0)
                    else:
                        bbox = getattr(obs, "bbox", None)
                        if bbox is not None:
                            obj_center = bbox.get_center()
                    if obj_center is not None:
                        vec = obj_center - cam_origin
                        norm = np.linalg.norm(vec)
                        norm_val = float(norm)
                        if norm > 1e-9:
                            # 
                            view_dir = (vec / norm).tolist()
                        if not math.isfinite(distance_val):
                            distance_val = float(norm)
                            distance_source = "pose_recompute"
                        else:
                            distance_source = "observation_pose"
            except Exception:
                pass

            distance_meta = {
                "source": distance_source,
                "cam_origin": cam_origin.tolist() if cam_origin is not None else None,
                "obj_center": obj_center.tolist() if obj_center is not None else None,
                "vector_norm": norm_val,
                "distance_observation": original_distance_val,
                "distance_final": distance_val,
            }

            info_entry = {
                    "idx": frame_idx,
                    "confidence": conf,
                    "edge_quality": edge_quality,
                    "edge_ratio": edge_touch_ratio,
                    "edge_contact_count": edge_contact_count,
                    "edge_perimeter_count": edge_pixel_count,
                    "edge_total_pixels": total_pixels,
                    "support_score": support_score,
                    "support_recognized": recognized,
                    "support_total": total_related,
                    "support_expected_names": expected_support_names,
                    "support_recognized_names": recognized_names,
                    "anchor_class_id": final_class_id,
                    "anchor_class_name": anchor_class_name,
                    "passes_conf_threshold": conf >= confidence_threshold,
                    "view_dir": view_dir,
                    "distance": distance_val,
                    "distance_meta": distance_meta,
                }

            if include_observation_refs:
                info_entry["_observation_ref"] = obs

            candidate_infos.append(info_entry)

        if not candidate_infos:
            return [], [], [], [], []

        if candidate_count is None:
            candidate_count = max(k, 6)

        confs = np.array([info["confidence"] for info in candidate_infos], dtype=float)
        edge_touch_scores = np.array(
            [info["edge_quality"] for info in candidate_infos], dtype=float
        )
        support_scores = np.array(
            [info["support_score"] for info in candidate_infos], dtype=float
        )
        idxs = np.array([info["idx"] for info in candidate_infos], dtype=int)
        distance_array = np.array([info.get("distance", float("nan")) for info in candidate_infos], dtype=float)

        # :
        def compute_distance_score(distances, min_dist, max_dist):
            """,1,"""
            scores = np.zeros_like(distances)
            valid_mask = np.isfinite(distances)
            
            if not np.any(valid_mask):
                return scores  # ,0
            
            valid_distances = distances[valid_mask]
            
            # 1
            in_range_mask = (valid_distances >= min_dist) & (valid_distances <= max_dist)
            scores[valid_mask] = np.where(in_range_mask, 1.0, 0.0)
            
            # ,
            out_range_mask = ~in_range_mask
            if np.any(out_range_mask):
                out_distances = valid_distances[out_range_mask]
                
                # 
                too_close_mask = out_distances < min_dist
                too_far_mask = out_distances > max_dist
                
                # :
                if np.any(too_close_mask):
                    close_distances = out_distances[too_close_mask]
                    close_scores = np.clip(close_distances / min_dist, 0.1, 1.0)
                    scores[valid_mask][out_range_mask][too_close_mask] = close_scores
                
                # :
                if np.any(too_far_mask):
                    far_distances = out_distances[too_far_mask]
                    # ,50.1
                    far_scores = np.exp(-(far_distances - max_dist) / 2.0)
                    far_scores = np.clip(far_scores, 0.1, 1.0)
                    scores[valid_mask][out_range_mask][too_far_mask] = far_scores
            
            return scores

        distance_scores = compute_distance_score(distance_array, optimal_distance_min, optimal_distance_max)

        # :
        raw_scores = alpha * confs + beta * edge_touch_scores + gamma * support_scores + delta * distance_scores

        # 
        def safe_norm(x: np.ndarray) -> np.ndarray:
            x = x.astype(float)
            if x.size == 0:
                return x
            rng = x.max() - x.min()
            if rng <= 1e-9:
                return np.zeros_like(x) if x.size > 0 else x
            return (x - x.min()) / max(rng, 1e-9)

        n_conf = safe_norm(confs)
        n_edge = safe_norm(edge_touch_scores)
        n_support = safe_norm(support_scores)
        n_distance = safe_norm(distance_scores)
        weighted_scores = alpha * n_conf + beta * n_edge + gamma * n_support + delta * n_distance

        # 
        if method == "unified":
            if local_map_manager is None:
                logger.warning("[get_best_observation_indices] unified method requires local_map_manager, falling back to weighted")
                method = "weighted"
            else:
                # 
                unified_scores = []
                for info in candidate_infos:
                    frame_idx = info["idx"]
                    # dualmapcompute_frame_score_unified
                    # object.py,local_map_managerdualmap
                    # 
                    try:
                        # local_map_managerdualmap
                        dualmap_instance = getattr(local_map_manager, "_dualmap_instance", None)
                        if dualmap_instance is not None:
                            # Nonecompute_frame_score_unified
                            score = dualmap_instance.compute_frame_score_unified(
                                frame_idx=frame_idx,
                                alpha_a=unified_alpha_a,
                                alpha_o=unified_alpha_o,
                                A_max=unified_A_max,
                                O_max=unified_O_max,
                                beta_e=unified_beta_e,
                                beta_c=unified_beta_c,
                                C_max=unified_C_max,
                                gamma_1=unified_gamma_1,
                                gamma_2=unified_gamma_2,
                            )
                        else:
                            # dualmap,()
                            score = 0.0
                            # ,dualmap
                            logger.warning(f"[get_best_observation_indices] No dualmap instance found for unified scoring at frame {frame_idx}")
                        unified_scores.append(score)
                    except Exception as e:
                        logger.warning(f"[get_best_observation_indices] Error computing unified score for frame {frame_idx}: {e}")
                        unified_scores.append(0.0)
                
                unified_scores = np.array(unified_scores, dtype=float)
                for i, info in enumerate(candidate_infos):
                    info["unified_score"] = float(unified_scores[i])
                    info["weighted_score"] = float(unified_scores[i])  # weighted_score
                order = np.argsort(-unified_scores)
        else:
            for i, info in enumerate(candidate_infos):
                info["distance_score"] = float(distance_scores[i])
                info["raw_score"] = float(raw_scores[i])
                info["weighted_score"] = float(weighted_scores[i])

            if method == "conf":
                order = np.argsort(-confs)
            elif method == "edge":
                order = np.argsort(-edge_touch_scores)
            elif method == "support":
                order = np.argsort(-support_scores)
            elif method == "distance":
                distance_sort = np.nan_to_num(distance_array, nan=-np.inf)
                order = np.argsort(-distance_sort)
            else:
                order = np.argsort(-weighted_scores)

        candidate_count = max(0, int(candidate_count))
        candidate_order = order[:candidate_count]
        candidate_pool = [candidate_infos[i] for i in candidate_order]

        selected_infos: List[dict] = []
        
        # :k
        if len(candidate_pool) == 0:
            return [], [], [], [], []
        
        # k
        selected_infos = candidate_pool[:k]
        
        # # ()
        # # :10,10xy3
        # top_10_count = 10
        # top_10_pool = candidate_pool[:top_10_count]
        # 
        # if len(top_10_pool) == 0:
        #     return [], [], [], [], []
        # 
        # # ()
        # best_info = top_10_pool[0]
        # 
        # # xy
        # def compute_xy_angle(info1, info2):
        #     """xy()"""
        #     view_dir1 = info1.get("view_dir")
        #     view_dir2 = info2.get("view_dir")
        #     
        #     if view_dir1 is None or view_dir2 is None:
        #         return 0.0
        #     
        #     view_dir1 = np.array(view_dir1)
        #     view_dir2 = np.array(view_dir2)
        #     
        #     # xy(z=0)
        #     dir1_xy = view_dir1[:2]
        #     dir2_xy = view_dir2[:2]
        #     
        #     # 
        #     norm1 = np.linalg.norm(dir1_xy)
        #     norm2 = np.linalg.norm(dir2_xy)
        #     
        #     if norm1 < 1e-9 or norm2 < 1e-9:
        #         return 0.0
        #     
        #     dir1_xy = dir1_xy / norm1
        #     dir2_xy = dir2_xy / norm2
        #     
        #     # ()
        #     dot = np.clip(np.dot(dir1_xy, dir2_xy), -1.0, 1.0)
        #     angle = np.degrees(np.arccos(dot))
        #     
        #     return float(angle)
        # 
        # # 
        # selected_indices_set = set()
        # selected_infos = []
        # 
        # # 
        # selected_infos.append(best_info.copy())
        # selected_indices_set.add(best_info["idx"])
        # 
        # # ,
        # remaining_pool = [info for info in top_10_pool if info["idx"] not in selected_indices_set]
        # 
        # while len(selected_infos) < k and len(remaining_pool) > 0:
        #     best_candidate = None
        #     best_min_angle = -1.0
        #     
        #     # ,
        #     for candidate_info in remaining_pool:
        #         min_angle = float('inf')
        #         
        #         # ,
        #         for selected_info in selected_infos:
        #             angle = compute_xy_angle(selected_info, candidate_info)
        #             min_angle = min(min_angle, angle)
        #         
        #         # ()
        #         if min_angle > best_min_angle:
        #             best_min_angle = min_angle
        #             best_candidate = candidate_info
        #     
        #     # 
        #     if best_candidate is not None:
        #         selected_infos.append(best_candidate.copy())
        #         selected_indices_set.add(best_candidate["idx"])
        #         remaining_pool.remove(best_candidate)
        #     else:
        #         break
        # 
        # # k,()
        # if len(selected_infos) < k:
        #     for info in candidate_pool:
        #         if len(selected_infos) >= k:
        #             break
        #         if info["idx"] not in selected_indices_set:
        #             selected_infos.append(info.copy())
        #             selected_indices_set.add(info["idx"])

        angles_between = []
        for i in range(len(selected_infos)):
            dir_i = selected_infos[i].get("view_dir")
            for j in range(i + 1, len(selected_infos)):
                dir_j = selected_infos[j].get("view_dir")
                if dir_i is None or dir_j is None:
                    continue
                dot = np.clip(np.dot(dir_i, dir_j), -1.0, 1.0)
                angle = float(np.degrees(np.arccos(dot)))
                angles_between.append(
                    (int(selected_infos[i]["idx"]), int(selected_infos[j]["idx"]), angle)
                )

        selected_indices = [int(info["idx"]) for info in selected_infos]

        if verbose:
            print(f"\n: {selected_indices}")
            print("=" * 50)

        high_confidence_indices = sorted(
            {int(info["idx"]) for info in candidate_infos if info.get("passes_conf_threshold", False)}
        )

        distance_summary: List[dict] = []
        valid_distances = [
            (info, info.get("distance", float("nan")))
            for info in candidate_infos
            if math.isfinite(info.get("distance", float("nan")))
        ]
        if valid_distances:
            distance_levels = max(1, int(distance_levels))
            dist_values = np.array([dist for _, dist in valid_distances], dtype=float)
            min_dist = float(np.min(dist_values))
            max_dist = float(np.max(dist_values))
            if distance_levels <= 1 or np.isclose(min_dist, max_dist):
                bin_ranges = [(min_dist, max_dist)]
                bin_frames = [[]]
                for info, _ in valid_distances:
                    bin_frames[0].append(int(info["idx"]))
            else:
                edges = np.linspace(min_dist, max_dist, distance_levels + 1)
                edges = np.unique(edges)
                if len(edges) < 2:
                    edges = np.array([min_dist, max_dist])
                level_count = len(edges) - 1
                bin_frames = [[] for _ in range(level_count)]
                expanded_edges = edges.copy()
                expanded_edges[-1] = np.nextafter(expanded_edges[-1], expanded_edges[-1] + 1.0)
                for info, dist in valid_distances:
                    bin_idx = np.searchsorted(expanded_edges, dist, side="right") - 1
                    bin_idx = max(0, min(level_count - 1, bin_idx))
                    bin_frames[bin_idx].append(int(info["idx"]))
                bin_ranges = [(float(edges[i]), float(edges[i + 1])) for i in range(level_count)]

            distance_summary = [
                {
                    "bin_index": i,
                    "range": bin_ranges[i],
                    "count": len(bin_frames[i]),
                    "frames": bin_frames[i],
                }
                for i in range(len(bin_frames))
            ]

        if not include_observation_refs:
            selected_infos = [
                {key: value for key, value in info.items() if key != "_observation_ref"}
                for info in selected_infos
            ]

        return (
            selected_indices,
            selected_infos,
            high_confidence_indices,
            angles_between,
            distance_summary,
        )

    @staticmethod
    def compute_unified_frame_score(
        anchor_areas: List[float],
        anchor_completeness: List[float],
        anchor_interaction_counts: List[int],
        other_object_count: int,
        # ()
        alpha_a: float = 0.7,  # 
        alpha_o: float = 0.3,  # 
        A_max: float = 5.0,    # 
        O_max: float = 10.0,   # 
        beta_e: float = 0.5,   # 
        beta_c: float = 0.5,   # 
        C_max: float = 5.0,    # 
        gamma_1: float = 0.6,  # LocalMax()
        gamma_2: float = 0.4,  # LocalMean
        A_img: Optional[float] = None,  # , r = sum area_j / A_img
        lambda_scale: float = 0.7,  # λ:( 0.7)
        return_details: bool = False,
    ):
        """
        Scale-aware : S(f) = w_global * S_global + w_local * S_local.
        S_global: global term; S_local: local term.
        w_global: M<=1 use 1-r(f); M>1 use (1-λ)(1-r) + λ * H(f)/log(M); w_local = 1 - w_global.
         H is entropy in log space; p_j = area_j / sum_m area_m.
        """
        M = len(anchor_areas)  # 

        def _global_local_norm_terms():
            M_capped = min(M, A_max)
            N_obj_capped = min(other_object_count, O_max)
            GlobalTerm_raw = alpha_a * M_capped + alpha_o * N_obj_capped
            GlobalTerm_max = alpha_a * A_max + alpha_o * O_max
            GlobalTerm_norm = GlobalTerm_raw / GlobalTerm_max if GlobalTerm_max > 0 else 0.0

            LocalValue_list = []
            for i in range(M):
                g_C_i = min(anchor_interaction_counts[i], C_max)
                E_i = anchor_completeness[i] if i < len(anchor_completeness) else 0.0
                LocalValue_list.append(beta_e * E_i + beta_c * g_C_i)

            if not LocalValue_list:
                return (
                    M_capped,
                    N_obj_capped,
                    GlobalTerm_raw,
                    GlobalTerm_max,
                    GlobalTerm_norm,
                    [],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            LocalMax = max(LocalValue_list)
            LocalMean = sum(LocalValue_list) / len(LocalValue_list)
            LocalTerm_raw = gamma_1 * LocalMax + gamma_2 * LocalMean
            LocalValue_max = beta_e * 1.0 + beta_c * C_max
            LocalTerm_max = (gamma_1 + gamma_2) * LocalValue_max
            LocalTerm_norm = LocalTerm_raw / LocalTerm_max if LocalTerm_max > 0 else 0.0
            return (
                M_capped,
                N_obj_capped,
                GlobalTerm_raw,
                GlobalTerm_max,
                GlobalTerm_norm,
                LocalValue_list,
                LocalMax,
                LocalMean,
                LocalTerm_raw,
                LocalTerm_max,
                LocalTerm_norm,
            )

        if M == 0:
            M_capped = 0
            N_obj_capped = min(other_object_count, O_max)
            GlobalTerm_raw = alpha_o * N_obj_capped
            GlobalTerm_max = alpha_a * A_max + alpha_o * O_max
            GlobalTerm_norm = GlobalTerm_raw / GlobalTerm_max if GlobalTerm_max > 0 else 0.0
            LocalTerm_norm = 0.0
            S_global = GlobalTerm_norm
            S_local = 0.0
            w_global, w_local, r_cov = 1.0, 0.0, 0.0
            H_anchor = 0.0
            H_balance_norm = 0.0
            Score_frame = max(0.0, min(1.0, w_global * S_global + w_local * S_local))
            if return_details:
                return {
                    "score": Score_frame,
                    "S_global": S_global,
                    "S_local": S_local,
                    "M": 0,
                    "A_sum": 0.0,
                    "A_img": A_img,
                    "r_coverage": r_cov,
                    "p_list": [],
                    "H_anchor": H_anchor,
                    "H_balance_norm": H_balance_norm,
                    "H_norm": 0.0,
                    "other_object_count": other_object_count,
                    "M_capped": M_capped,
                    "N_obj_capped": N_obj_capped,
                    "GlobalTerm_raw": GlobalTerm_raw,
                    "GlobalTerm_max": GlobalTerm_max,
                    "GlobalTerm_norm": GlobalTerm_norm,
                    "LocalValue_list": [],
                    "LocalMax": 0.0,
                    "LocalMean": 0.0,
                    "LocalTerm_raw": 0.0,
                    "LocalTerm_max": (gamma_1 + gamma_2) * (beta_e * 1.0 + beta_c * C_max),
                    "LocalTerm_norm": LocalTerm_norm,
                    "w_global": w_global,
                    "w_local": w_local,
                    "lambda_scale": lambda_scale,
                    "constants": {
                        "alpha_a": alpha_a,
                        "alpha_o": alpha_o,
                        "A_max": A_max,
                        "O_max": O_max,
                        "beta_e": beta_e,
                        "beta_c": beta_c,
                        "C_max": C_max,
                        "gamma_1": gamma_1,
                        "gamma_2": gamma_2,
                    },
                }
            return Score_frame

        A_sum = sum(anchor_areas)
        if A_sum <= 0:
            return 0.0 if not return_details else {"score": 0.0, "M": M, "A_sum": 0.0}

        p_list = [area_i / A_sum for area_i in anchor_areas]
        H_anchor = 0.0
        for p_i in p_list:
            if p_i > 0:
                H_anchor -= p_i * math.log(p_i)

        if A_img is not None and A_img > 0:
            r_cov = min(1.0, max(0.0, A_sum / A_img))
        else:
            r_cov = 0.0

        if M <= 1:
            w_global = 1.0 - r_cov
        else:
            log_m = math.log(M)
            h_bal = H_anchor / log_m if log_m > 0 else 0.0
            h_bal = max(0.0, min(1.0, h_bal))
            w_global = (1.0 - lambda_scale) * (1.0 - r_cov) + lambda_scale * h_bal
        w_global = max(0.0, min(1.0, w_global))
        w_local = 1.0 - w_global

        log_m = math.log(M) if M > 1 else 0.0
        H_balance_norm = H_anchor / log_m if log_m > 0 else 0.0

        (
            M_capped,
            N_obj_capped,
            GlobalTerm_raw,
            GlobalTerm_max,
            GlobalTerm_norm,
            LocalValue_list,
            LocalMax,
            LocalMean,
            LocalTerm_raw,
            LocalTerm_max,
            LocalTerm_norm,
        ) = _global_local_norm_terms()

        S_global = GlobalTerm_norm
        S_local = LocalTerm_norm
        Score_frame = w_global * S_global + w_local * S_local
        Score_frame = max(0.0, min(1.0, Score_frame))

        if return_details:
            return {
                "score": Score_frame,
                "S_global": S_global,
                "S_local": S_local,
                "M": M,
                "A_sum": A_sum,
                "A_img": A_img,
                "r_coverage": r_cov,
                "p_list": p_list,
                "H_anchor": H_anchor,
                "H_balance_norm": H_balance_norm,
                "H_norm": H_balance_norm,
                "M_capped": M_capped,
                "N_obj_capped": N_obj_capped,
                "other_object_count": other_object_count,
                "GlobalTerm_raw": GlobalTerm_raw,
                "GlobalTerm_max": GlobalTerm_max,
                "GlobalTerm_norm": GlobalTerm_norm,
                "LocalValue_list": LocalValue_list,
                "LocalMax": LocalMax,
                "LocalMean": LocalMean,
                "LocalTerm_raw": LocalTerm_raw,
                "LocalTerm_max": LocalTerm_max,
                "LocalTerm_norm": LocalTerm_norm,
                "w_global": w_global,
                "w_local": w_local,
                "lambda_scale": lambda_scale,
                "constants": {
                    "alpha_a": alpha_a,
                    "alpha_o": alpha_o,
                    "A_max": A_max,
                    "O_max": O_max,
                    "beta_e": beta_e,
                    "beta_c": beta_c,
                    "C_max": C_max,
                    "gamma_1": gamma_1,
                    "gamma_2": gamma_2,
                },
            }
        return Score_frame

    def _calculate_view_angle_score(self, observation, verbose=False):
        """
        .
        ,30°~60°,.
        0~1,1.
        """
        try:
            # (4x4)
            camera_pose = getattr(observation, 'pose', None)
            if camera_pose is None:
                # pose,DataInput
                data_input = getattr(observation, 'data_input', None)
                if data_input is not None:
                    camera_pose = getattr(data_input, 'pose', None)
            
            if camera_pose is None:
                return 0.5  # 
            
            # (Z)
            camera_forward = -camera_pose[:3, 2]  # Z
            
            # (PCA)
            object_normal = self._estimate_object_normal(verbose)
            
            # (0°~90°)
            dot_product = np.dot(camera_forward, object_normal)
            angle_rad = np.arccos(np.clip(np.abs(dot_product), 0.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            
            # :30°~60°,1
            if angle_deg < 30:
                # 0°~30°:1
                score = angle_deg / 30.0
            elif angle_deg <= 60:
                # 30°~60°:,1
                score = 1.0
            else:
                # 60°~90°:0.5
                score = max(0.5, 1.0 - (angle_deg - 60) / 60.0)
            
            return score
            
        except Exception as e:
            if verbose:
                print(f": {e}")
            return 0.5  # 

    def _estimate_object_normal(self, verbose=False):
        """
        .
        , ; , PCA.
        """
        try:
            # ,
            if getattr(self, 'is_low_mobility', False) and getattr(self, 'major_plane_info', None) is not None:
                # ,
                return np.array([0.0, 0.0, 1.0])
            
            # PCA
            if self.pcd is None or len(self.pcd.points) == 0:
                return np.array([0.0, 0.0, 1.0])  # 
            
            points = np.asarray(self.pcd.points)
            
            # 
            centroid = np.mean(points, axis=0)
            centered_points = points - centroid
            
            # PCA
            covariance_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
            
            # 
            normal_vector = eigenvectors[:, 0]
            
            # (Z)
            if normal_vector[2] < 0:
                normal_vector = -normal_vector
                
            return normal_vector / np.linalg.norm(normal_vector)
            
        except Exception as e:
            if verbose:
                print(f": {e}")
            return np.array([0.0, 0.0, 1.0])  # 

class GlobalObject(BaseObject):
    def __init__(self, observation=None):
        super().__init__()

        # Spatial primitives
        self.pcd_2d: Optional[o3d.geometry.PointCloud] = o3d.geometry.PointCloud()
        self.bbox_2d: Optional[o3d.geometry.AxisAlignedBoundingBox] = (
            o3d.geometry.AxisAlignedBoundingBox()
        )

        # Related objs
        # Current we "only save clip feats" <-- PAY Attention!
        self.related_objs: List[np.ndarray] = []

        # for visualization in rerun
        self.related_bbox = []
        self.related_color = []

        # If provide the LocalObject, then initialize the GlobalObject from it
        if observation is not None:
            self.init_from_global_obs(observation)

    def __getstate__(self):
        # Base Object getstate
        state = super().__getstate__()

        # serialize List[np.ndarray]
        state["related_objs"] = [arr.tolist() for arr in self.related_objs]

        # serialize pcd_2d
        state["pcd_2d_points"] = np.asarray(self.pcd_2d.points).tolist()
        state["pcd_2d_colors"] = np.asarray(self.pcd_2d.colors).tolist()

        # Serialize related_bbox (convert AxisAlignedBoundingBox to dict)
        state["related_bbox"] = [
            {
                "min_bound": bbox.get_min_bound().tolist(),
                "max_bound": bbox.get_max_bound().tolist(),
            }
            for bbox in self.related_bbox
        ]

        # Serialize related_color (class IDs list)
        state["related_color"] = self.related_color  # assuming it's a list of class IDs

        return state

    def __setstate__(self, state):
        # Base Object setstate
        super().__setstate__(state)

        # Restore related_objs as np.ndarray
        self.related_objs = [np.array(arr) for arr in state.get("related_objs", [])]

        # Restore pcd_2d (points and colors)
        points = np.array(state.get("pcd_2d_points"))
        colors = np.array(state.get("pcd_2d_colors"))

        self.pcd_2d = o3d.geometry.PointCloud()
        self.pcd_2d.points = o3d.utility.Vector3dVector(points)
        self.pcd_2d.colors = o3d.utility.Vector3dVector(colors)

        self.bbox_2d = self.pcd_2d.get_axis_aligned_bounding_box()

        self.related_bbox = [
            o3d.geometry.AxisAlignedBoundingBox(
                min_bound=np.array(bbox_dict["min_bound"]),
                max_bound=np.array(bbox_dict["max_bound"]),
            )
            for bbox_dict in state.get("related_bbox", [])
        ]

        # Restore related_color (assuming it's stored as a list of class IDs)
        self.related_color = state.get("related_color", [])

        # Set obs num to 1 to avoid global updating bug
        self.observed_num = 1

    def copy(self):
        return copy.deepcopy(self)

    def init_from_global_obs(self, observation):

        self.uid = observation.uid

        self.pcd = observation.pcd
        self.bbox = observation.bbox

        self.pcd_2d = observation.pcd_2d
        self.bbox_2d = observation.bbox_2d

        self.clip_ft = observation.clip_ft

        # Class ID
        self.class_id = observation.class_id

        # Related objs
        self.related_objs = observation.related_objs

    def add_observation(self, observation: Observation) -> None:
        self.observations.append(observation)
        self.observed_num += 1

    def get_latest_observation(self) -> Observation:
        return self.observations[-1] if self.observations else None

    def update_info(self) -> None:
        # Global Obj

        latest_obs = self.get_latest_observation()

        if self.observed_num == 0:
            logger.error("[GlobalObject] No observation in this object")
            return

        if self.observed_num == 1:
            self.uid = latest_obs.uid

            self.pcd = latest_obs.pcd
            self.bbox = latest_obs.bbox
            self.pcd_2d = latest_obs.pcd_2d
            self.bbox_2d = latest_obs.bbox_2d

            self.clip_ft = latest_obs.clip_ft
            self.class_id = latest_obs.class_id

            self.related_objs = latest_obs.related_objs

            # for visualization in rerun
            self.related_bbox = latest_obs.related_bbox
            self.related_color = latest_obs.related_color
            return

        # Update the information for outside

        # Merge pcd
        # Simply add
        self.pcd += latest_obs.pcd
        self.pcd = self.pcd.voxel_down_sample(
            voxel_size=self._cfg.downsample_voxel_size
        )

        # Get new bbox
        self.bbox = self.pcd.get_axis_aligned_bounding_box()

        # Merge pcd_2d
        # get the points num of the latest obs and current pcd_2d
        original_num = len(self.pcd_2d.points)
        incoming_num = len(latest_obs.pcd_2d.points)

        self.pcd_2d += latest_obs.pcd_2d
        self.pcd_2d = self.voxel_downsample_2d(
            pcd=self.pcd_2d, voxel_size=self._cfg.downsample_voxel_size
        )
        self.bbox_2d = self.pcd_2d.get_axis_aligned_bounding_box()

        # TODO: Should we merge clip feat? Any judgement?
        # Answer: Weighted merge

        # TODO: for class id, we choose the bigger one? Currently we choose the bigger one as the output
        # for class id, we choose the bigger one as the output
        # bigger or smaller depends on the pointcloud size of the pcd_2d
        if original_num < incoming_num:
            self.class_id = latest_obs.class_id
            self.clip_ft = latest_obs.clip_ft

        # TODO: Any other matching strategy on related objs?
        # Maintain the related objs, simply add the objs from the latest observation (Current Strategy)
        self.related_objs += latest_obs.related_objs

        # for visualization in rerun
        self.related_bbox += latest_obs.related_bbox
        self.related_color += latest_obs.related_color

        pass
