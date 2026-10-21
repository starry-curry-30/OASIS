import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Tuple
from utils.types import Observation

logger = logging.getLogger(__name__)


class NoveltyDetector:
    """
    Novelty / keyframe gate from CLIP cosine similarity and pose delta.
    (Simplified: no separate novelty score; CLIP + pose thresholds only.)
    """

    def __init__(
        self,
        clip_threshold: float = 0.8,  # below => large appearance change
        pose_translation_threshold: float = 0.5,
        pose_rotation_threshold: float = 15.0,  # degrees
    ):
        """
        Args:
            clip_threshold: cosine similarity; below this => novel
            pose_translation_threshold: meters
            pose_rotation_threshold: degrees
        """
        self.clip_threshold = clip_threshold
        self.pose_translation_threshold = pose_translation_threshold
        self.pose_rotation_threshold = pose_rotation_threshold

        # Repeat frames before fusion
        self.repeat_frames: List[Tuple[np.ndarray, np.ndarray]] = []  # [(clip_feat, pose), ...]
        self.last_novel_frame_clip: Optional[np.ndarray] = None
        self.last_novel_frame_pose: Optional[np.ndarray] = None
        
    def compute_full_image_clip_feature(
        self,
        image: np.ndarray,
        clip_model,
        clip_preprocess,
        device: str = "cuda"
    ) -> Optional[np.ndarray]:
        """Encode full-image CLIP features."""
        try:
            from PIL import Image
            import torch

            # To PIL
            if isinstance(image, np.ndarray):
                image = image.astype(np.uint8)
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            preprocessed_image = clip_preprocess(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(preprocessed_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            clip_feat = image_features.detach().cpu().numpy().flatten()

            return clip_feat
        except Exception as e:
            logger.error(f"[NoveltyDetector] full-image CLIP encode failed: {e}")
            return None
    
    def compute_pose_change(
        self, 
        pose1: np.ndarray, 
        pose2: np.ndarray
    ) -> Tuple[float, float]:
        """Translation norm and rotation angle (deg) between two poses."""
        translation_diff = np.linalg.norm(pose2[:3, 3] - pose1[:3, 3])

        try:
            rot1 = pose1[:3, :3]
            rot2 = pose2[:3, :3]

            if np.any(np.isnan(rot1)) or np.any(np.isinf(rot1)) or \
               np.any(np.isnan(rot2)) or np.any(np.isinf(rot2)):
                logger.warning(
                    "[NoveltyDetector] rotation matrix has NaN/Inf; using 0 deg"
                )
                return translation_diff, 0.0

            # SVD orthogonalization if needed
            def orthogonalize_rotation(R_mat):
                """Project to SO(3)."""
                try:
                    U, s, Vt = np.linalg.svd(R_mat)
                    R_ortho = U @ Vt
                    if np.linalg.det(R_ortho) < 0:
                        U[:, -1] *= -1
                        R_ortho = U @ Vt
                    return R_ortho
                except np.linalg.LinAlgError:
                    logger.warning(
                        "[NoveltyDetector] orthogonalize failed; using I"
                    )
                    return np.eye(3)

            def is_valid_rotation(R_mat, tol=1e-3):
                """Near SO(3) check."""
                det = np.linalg.det(R_mat)
                if abs(det - 1.0) > tol:
                    return False
                should_be_identity = R_mat @ R_mat.T
                identity = np.eye(3)
                if np.linalg.norm(should_be_identity - identity) > tol:
                    return False
                return True

            if not is_valid_rotation(rot1):
                rot1 = orthogonalize_rotation(rot1)
            if not is_valid_rotation(rot2):
                rot2 = orthogonalize_rotation(rot2)

            rotation1 = R.from_matrix(rot1)
            rotation2 = R.from_matrix(rot2)
            rotation_diff = (rotation2.inv() * rotation1).magnitude() * (180.0 / np.pi)
            
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(
                f"[NoveltyDetector] rotation diff failed: {e}; using 0.0 deg"
            )
            rotation_diff = 0.0
        
        return translation_diff, rotation_diff
    
    def compute_clip_similarity(
        self, 
        feat1: np.ndarray, 
        feat2: np.ndarray
    ) -> float:
        """Cosine similarity between two CLIP vectors (higher = more similar)."""
        feat1 = feat1.flatten()
        feat2 = feat2.flatten()
        
        if np.linalg.norm(feat1) > 0:
            feat1 = feat1 / np.linalg.norm(feat1)
        if np.linalg.norm(feat2) > 0:
            feat2 = feat2 / np.linalg.norm(feat2)
        
        similarity = np.dot(feat1, feat2)
        return float(similarity)
    
    def fuse_clip_features(self, clip_features: List[np.ndarray]) -> Optional[np.ndarray]:
        """Mean-pool CLIP vectors then L2-normalize."""
        if not clip_features:
            return None
        
        clip_features = np.array(clip_features)
        fused_feat = np.mean(clip_features, axis=0)
        
        norm = np.linalg.norm(fused_feat)
        if norm > 0:
            fused_feat = fused_feat / norm
        
        return fused_feat
    
    def check_novelty(
        self,
        image: np.ndarray,
        pose: np.ndarray,
        clip_model=None,
        clip_preprocess=None,
        device: str = "cuda"
    ) -> Tuple[bool, dict]:
        """
        Returns (is_novel, info). Novel if CLIP sim < threshold or pose delta large.
        """
        # Full-image CLIP
        if clip_model is not None and clip_preprocess is not None:
            curr_clip_feat = self.compute_full_image_clip_feature(
                image, clip_model, clip_preprocess, device
            )
        else:
            logger.error("[NoveltyDetector] clip_model/clip_preprocess missing")
            return True, {
                "reason": "no_clip_model",
                "clip_similarity": None,
                "pose_translation_diff": None,
                "pose_rotation_diff": None,
            }
        
        if curr_clip_feat is None:
            logger.warning("[NoveltyDetector] CLIP encode failed; treating as novel")
            self.last_novel_frame_clip = None
            self.last_novel_frame_pose = pose
            self.repeat_frames = []
            return True, {
                "reason": "clip_computation_failed",
                "clip_similarity": None,
                "pose_translation_diff": None,
                "pose_rotation_diff": None,
            }
        
        # First frame is always novel
        if self.last_novel_frame_clip is None:
            self.last_novel_frame_clip = curr_clip_feat
            self.last_novel_frame_pose = pose
            self.repeat_frames = []
            return True, {
                "reason": "first_frame",
                "clip_similarity": 1.0,
                "pose_translation_diff": 0.0,
                "pose_rotation_diff": 0.0,
            }
        
        # CLIP sim vs last novel
        clip_similarity = self.compute_clip_similarity(
            curr_clip_feat, 
            self.last_novel_frame_clip
        )
        
        # Pose delta vs last novel
        pose_translation_diff, pose_rotation_diff = self.compute_pose_change(
            self.last_novel_frame_pose,
            pose
        )
        
        # Novel if low CLIP sim or large pose move (uses 0.8 here, not self.clip_threshold -- legacy)
        is_novel = (
            clip_similarity < 0.8 or
            pose_translation_diff > self.pose_translation_threshold or
            pose_rotation_diff > self.pose_rotation_threshold
        )
        
        info = {
            "reason": "novel" if is_novel else "repeat",
            "clip_similarity": clip_similarity,
            "pose_translation_diff": pose_translation_diff,
            "pose_rotation_diff": pose_rotation_diff,
            "repeat_frames_count": len(self.repeat_frames),
        }
        
        if is_novel:
            self.last_novel_frame_clip = curr_clip_feat
            self.last_novel_frame_pose = pose
            self.repeat_frames = []
            logger.info(
                f"[NoveltyDetector] novel frame: "
                f"clip_sim={clip_similarity:.3f}, "
                f"trans={pose_translation_diff:.3f}m, "
                f"rot={pose_rotation_diff:.2f}deg"
            )
        else:
            self.repeat_frames.append((curr_clip_feat.copy(), pose.copy()))
            logger.debug(
                f"[NoveltyDetector] repeat frame: "
                f"clip_sim={clip_similarity:.3f}, "
                f"trans={pose_translation_diff:.3f}m, "
                f"rot={pose_rotation_diff:.2f}deg, "
                f"repeat_buf={len(self.repeat_frames)}"
            )

            # Fuse repeat-buffer CLIP and re-check
            if len(self.repeat_frames) >= 1:
                repeat_clip_features = [feat for feat, _ in self.repeat_frames]
                fused_clip_feat = self.fuse_clip_features(repeat_clip_features)
                
                if fused_clip_feat is not None:
                    fused_similarity = self.compute_clip_similarity(
                        curr_clip_feat,
                        fused_clip_feat
                    )
                    
                    first_repeat_pose = self.repeat_frames[0][1]
                    fused_translation_diff, fused_rotation_diff = self.compute_pose_change(
                        first_repeat_pose,
                        pose
                    )
                    
                    is_novel_fused = (
                        fused_similarity < 0.8 or
                        fused_translation_diff > self.pose_translation_threshold or
                        fused_rotation_diff > self.pose_rotation_threshold
                    )
                    
                    info["fused_clip_similarity"] = fused_similarity
                    info["fused_pose_translation_diff"] = fused_translation_diff
                    info["fused_pose_rotation_diff"] = fused_rotation_diff
                    
                    if is_novel_fused:
                        is_novel = True
                        self.last_novel_frame_clip = curr_clip_feat
                        self.last_novel_frame_pose = pose
                        self.repeat_frames = []
                        info["reason"] = "novel_after_fusion"
                        logger.info(
                            f"[NoveltyDetector] novel after fusion: fused_clip_sim={fused_similarity:.3f}"
                        )
                    else:
                        self.repeat_frames = []
                        info["reason"] = "repeat_after_fusion"
                        logger.debug(
                            f"[NoveltyDetector] still repeat after fusion: fused_clip_sim={fused_similarity:.3f}"
                        )
        
        return is_novel, info
    
    def reset(self):
        """Reset internal state."""
        self.repeat_frames = []
        self.last_novel_frame_clip = None
        self.last_novel_frame_pose = None
        logger.info("[NoveltyDetector] reset")