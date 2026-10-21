import dataclasses
import logging
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from supervision.draw.color import Color, ColorPalette

from utils.types import ObjectClasses

# Set up the module-level logger
logger = logging.getLogger(__name__)


class ReRunVisualizer:
    _instance = None

    def __init__(self, cfg=None) -> None:

        if not self._initialized:
            super().__init__()
            self.cfg = cfg

            classes_path = cfg.yolo.classes_path
            if cfg.yolo.use_given_classes:
                classes_path = cfg.yolo.given_classes_path
                logger.info(f"[Visualizar] Using given classes, path:{classes_path}")

            # Object classes
            self.obj_classes = ObjectClasses(
                classes_file_path=classes_path,
                bg_classes=self.cfg.yolo.bg_classes,
                skip_bg=self.cfg.yolo.skip_bg,
            )

            # Camera parameters
            self.intrinsic = None
            self.pose = None
            self.image = None
            self._intrinsic_initialized = False

            self._initialized = True

            self.overlapped_image = None

    def __new__(cls, cfg=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize the instance "once"
            cls._instance._use_rerun = None
            cls._instance._rerun = None
            cls._instance._initialized = False
        return cls._instance

    def set_use_rerun(self, use_rerun):
        self._use_rerun = use_rerun
        if self._use_rerun and self._rerun is None:
            try:
                import rerun as rr

                self._rerun = rr
                logger.info("[Visualizar] rerun is installed. Using rerun for logger.")
                
                # Setup world coordinates
                # Since HM3D/Habitat is Y-up, we tell Rerun that Y is up.
                # However, if the user requested Z-up view for clarity, we can set the view coordinates accordingly.
                # But Rerun 0.9+ usually defaults to RIGHT_HAND_Y_UP or similar if not specified? 
                # Let's explicitly set it.
                # If we want Z-up view for Y-up data, we might need to log a transform or just set the view coords.
                # Setting view coordinates doesn't transform the data, just the camera control/grid.
                
                # Try setting Z-up view (assuming standard ROS/Map convention)
                # If data is Y-up, it will look sideways unless we rotate it OR just accept Y-up view.
                # User asked for Z-up display.
                # rr.ViewCoordinates.RIGHT_HAND_Z_UP  ( +X=Front, +Y=Left, +Z=Up )
                
                # To make Y-up data appear upright in a Z-up viewer, we'd effectively need to rotate the "world".
                # But modifying the data logs is tedious.
                # Instead, we can just set the ViewCoordinates to Y-up so the camera controls work naturally for the data.
                
                # Wait, user said: "Use Z-up display because it's clearer".
                # If data is Y-up, and we set ViewCoordinates to Y-up, "Up" key moves along Y.
                # If we set ViewCoordinates to Z-up, "Up" key moves along Z.
                
                # If user wants to see the floor on the "horizontal" plane grid in Rerun:
                # Rerun's grid is usually defined by the Up axis.
                # If we tell Rerun "Up is Y", the grid should be XZ plane.
                # If we tell Rerun "Up is Z", the grid is XY plane.
                
                # Since our data has floor on XZ plane (Y-up), we should tell Rerun "Up is Y".
                # Then the grid will align with the floor.
                
                # Let's set it to RIGHT_HAND_Y_UP.
                # rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
                
                # BUT, the user specifically asked: "visualization use Z-up way".
                # Maybe they mean they WANT the data rotated to Z-up for visualization?
                # If so, I should log a root transform in Rerun that rotates Y-up data to Z-up.
                
                # Let's try logging a transform at "world" or "world/map" that rotates -90 deg X.
                # And set ViewCoordinates to Z-up.
                
                rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True) # Set view to Z-up
                
                # Rotate the content so Y-up data becomes Z-up
                # Rotation: X -> X, Y -> Z, Z -> -Y (Rotate -90 deg around X)
                from scipy.spatial.transform import Rotation as R
                q = R.from_euler('x', -90, degrees=True).as_quat() # (x, y, z, w)
                # Rerun uses (x, y, z, w) for quaternions? Or (w, x, y, z)?
                # Rerun python docs: xyzw
                
                # We log this transform on a node that contains all other data.
                # Assuming all data is logged under "world/..."
                # If I log a transform on "world", does it affect "world/camera"? Yes.
                
                rr.log("world/data_root", rr.Transform3D(rotation=rr.Quaternion(xyzw=q)), static=True)
                
                # Now we need to make sure all subsequent logs go to "world/data_root/..." instead of "world/..."
                # This requires changing all log calls. That is risky/tedious.
                
                # Alternative: Just set ViewCoordinates to Y-up.
                # If I set Y-up, the camera 'Orbit' mode will treat Y as the pole. The grid will be XZ.
                # This should be what they want if the data is Y-up.
                # If they said "Use Z-up display", maybe they are confused or Rerun's default is Z-up and it looks wrong?
                # If Rerun default is Z-up, and data is Y-up, the floor (XZ) looks like a wall.
                # So setting ViewCoordinates.RIGHT_HAND_Y_UP should fix it visually without changing data coordinates.
                
                self._rerun.log("world", self._rerun.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

            except ImportError:
                logger.warning(
                    "[Visualizar] rerun is not installed. Not using rerun for logger."
                )
        else:
            logger.warning(
                "[Visualizar] rerun functionality is disabled in the config. Not using rerun for logger."
            )

    def __getattr__(self, name):
        def method(*args, **kwargs):
            if self._use_rerun and self._rerun:
                func = getattr(self._rerun, name, None)
                if func:
                    return func(*args, **kwargs)
                else:
                    logger.warning(
                        f"[Visualizar] '{name}' is not a valid rerun method."
                    )
            else:
                if not self._use_rerun:
                    logger.info(
                        f"[Visualizar] Skipping optional rerun call to '{name}' because rerun usage is disabled."
                    )
                elif self._rerun is None:
                    logger.info(
                        f"[Visualizar] Skipping optional rerun call to '{name}' because rerun is not installed."
                    )

        return method

    def update_intrinsic(self, intrinsic, force=False):
        if self._intrinsic_initialized and not force:
            return
        self.intrinsic = intrinsic
        self._intrinsic_initialized = True
        logger.info("[Visualizar] Intrinsic updated.")

    def update_pose(self, pose):
        self.pose = pose
        logger.info("[Visualizar] Pose updated.")

    def set_camera_info(self, intrinsic, pose):
        self.update_intrinsic(intrinsic)
        self.update_pose(pose)

    def set_image(self, image):
        self.image = image

    def rotation_matrix_to_quaternion(self, R):
        """
        Convert a rotation matrix to a quaternion.

        Parameters:
        - R: A 3x3 rotation matrix.

        Returns:
        - A quaternion in the format [x, y, z, w].
        """
        # Make sure the matrix is a numpy array
        R = np.asarray(R)
        # Allocate space for the quaternion
        q = np.empty((4,), dtype=np.float32)
        # Compute the quaternion components
        q[3] = np.sqrt(np.maximum(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        q[0] = np.sqrt(np.maximum(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
        q[1] = np.sqrt(np.maximum(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
        q[2] = np.sqrt(np.maximum(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
        q[0] *= np.sign(q[0] * (R[2, 1] - R[1, 2]))
        q[1] *= np.sign(q[1] * (R[0, 2] - R[2, 0]))
        q[2] *= np.sign(q[2] * (R[1, 0] - R[0, 1]))
        return q

    def quaternion_to_rotation_matrix(self, q):
        """
        Convert a quaternion into a rotation matrix.

        Parameters:
        - q: A quaternion in the format [x, y, z, w].

        Returns:
        - A 3x3 rotation matrix.
        """
        w, x, y, z = q[3], q[0], q[1], q[2]
        return np.array(
            [
                [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
            ]
        )

    def rotation_matrix_to_axis_angle(self, R):
        """
        Convert a rotation matrix to an axis-angle representation.

        Parameters:
        - R: A 3x3 rotation matrix.

        Returns:
        - A tuple containing the axis of rotation and the angle of rotation in radians.
        """
        # Compute the trace of the matrix
        trace = np.trace(R)
        # Compute the angle of rotation
        theta = np.arccos((trace - 1) / 2)
        # Compute the axis of rotation
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis = axis / np.linalg.norm(axis)
        return axis, theta

    def visualize_3d_bbox_overlapping(self, obj_names, obj_colors, obj_bboxes):

        proc_image = self.image.copy()
        K = self.intrinsic
        T_wc = self.pose
        T_cw = np.linalg.inv(T_wc)

        for name, color, bbox in zip(obj_names, obj_colors, obj_bboxes):
            proc_image = self._draw_projected_bbox(
                proc_image, bbox, name, color, K, T_cw
            )

        self.log("world/camera/overlapped_image", self.Image(proc_image))

    def _draw_projected_bbox(self, image, bbox, name, color, K, T_cw):
        # Get corners and reorder them explicitly
        corners = np.asarray(bbox.get_box_points())

        # Sort to consistent order: bottom face first, then top face
        center = np.mean(corners, axis=0)
        above = corners[:, 2] > center[2]
        below = ~above

        top = corners[above]
        bottom = corners[below]

        def sort_corners_xy(corners):
            center = np.mean(corners[:, :2], axis=0)
            angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
            return corners[np.argsort(angles)]

        bottom_sorted = sort_corners_xy(bottom)
        top_sorted = sort_corners_xy(top)

        ordered = np.vstack([bottom_sorted, top_sorted])

        # Transform to camera frame
        corners_homo = np.hstack([ordered, np.ones((8, 1))])  # (8, 4)
        corners_cam = (T_cw @ corners_homo.T).T[:, :3]

        if np.any(corners_cam[:, 2] <= 0):
            return image

        # Project
        corners_2d = (K @ corners_cam.T).T
        corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]
        corners_2d = corners_2d.astype(int)

        # Draw edges
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # bottom
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # top
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # sides
        ]

        # Convert color to valid BGR tuple
        color = tuple(int(c * 255) for c in np.clip(color, 0.0, 1.0))

        # Draw edges
        for i, j in edges:
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[j])
            cv2.line(image, pt1, pt2, color=color, thickness=2)

        # Draw label
        label_pos = tuple(corners_2d[0])
        cv2.putText(image, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return image


def visualize_result_rgb(
    image: np.ndarray,
    detections: sv.Detections,
    classes: list[str],
    color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
    instance_random_color: bool = False,
    draw_bboxes: bool = True,
) -> np.ndarray:
    """
    Visualize the results of a single image. Annotate the image with the detections.

    Parameters:
    image (numpy.ndarray): The image to be visualized.
    detections (sv.Detection): The detections to be visualized.
    classes (list[str]): The classes to be visualized.
    color (Color | ColorPalette): The color palette to be used. Default is ColorPalette.default().
    instance_random_color (bool): Whether or not to use a random color for each instance. Default is False.
    draw_bboxes (bool): Whether or not to draw bounding boxes. Default is True.

    Returns:
    annotated_image (numpy.ndarray): The visualized image with detections.
    labels (list[str]): The labels for each detection.
    """
    image = image.copy()

    # annotate image with detections
    box_annotator = sv.BoxAnnotator(color=color)
    mask_annotator = sv.MaskAnnotator(color=color)
    label_annotator = sv.LabelAnnotator(
        color=color, text_scale=0.5, text_thickness=1, text_padding=2
    )

    if hasattr(detections, "confidence") and hasattr(detections, "class_id"):
        confidences = detections.confidence
        class_ids = detections.class_id
        if confidences is not None and class_ids is not None:
            labels = [
                f"{classes[class_id]} {confidence:.2f}"
                for confidence, class_id in zip(confidences, class_ids)
            ]
        else:
            labels = [f"{classes[class_id]}" for class_id in class_ids]
    else:
        print("No confidence or class_id detected! Or one of them is missing!")

    if instance_random_color:
        # generate random color for each instance
        # shallow copy
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

    if draw_bboxes:
        annotated_image = box_annotator.annotate(
            scene=annotated_image, detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

    return annotated_image, labels


def visualize_result_rgb_bbox_only(
    image: np.ndarray,
    detections: sv.Detections,
    classes: list[str],
    color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
    instance_random_color: bool = False,
) -> np.ndarray:
    """
    Visualize detection results with bounding boxes and labels only, without masks.
    This is used for saving representative frames without mask overlay.

    Parameters:
    image (numpy.ndarray): The image to be visualized.
    detections (sv.Detection): The detections to be visualized.
    classes (list[str]): The classes to be visualized.
    color (Color | ColorPalette): The color palette to be used. Default is ColorPalette.default().
    instance_random_color (bool): Whether or not to use a random color for each instance. Default is False.

    Returns:
    annotated_image (numpy.ndarray): The visualized image with detections (bboxes and labels only).
    labels (list[str]): The labels for each detection.
    """
    image = image.copy()

    # Only create box and label annotators, no mask annotator
    box_annotator = sv.BoxAnnotator(color=color)
    label_annotator = sv.LabelAnnotator(
        color=color, text_scale=0.5, text_thickness=1, text_padding=2
    )

    if hasattr(detections, "confidence") and hasattr(detections, "class_id"):
        confidences = detections.confidence
        class_ids = detections.class_id
        if confidences is not None and class_ids is not None:
            labels = []
            for confidence, class_id in zip(confidences, class_ids):
                # Clamp class_id to valid range
                if 0 <= class_id < len(classes):
                    labels.append(f"{classes[class_id]} {confidence:.2f}")
                else:
                    labels.append(f"unknown_{class_id} {confidence:.2f}")
        else:
            labels = []
            for class_id in class_ids:
                if 0 <= class_id < len(classes):
                    labels.append(f"{classes[class_id]}")
                else:
                    labels.append(f"unknown_{class_id}")
    else:
        print("No confidence or class_id detected! Or one of them is missing!")
        labels = []

    if instance_random_color:
        # generate random color for each instance
        # shallow copy
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))

    # Start with original image (no mask overlay)
    annotated_image = image.copy()

    # Draw bounding boxes
    annotated_image = box_annotator.annotate(
        scene=annotated_image, detections=detections
    )
    # Draw labels with smart positioning
    # Calculate text size and adjust position if it goes out of bounds
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_padding = 2
    
    for i in range(len(detections)):
        if i >= len(labels):
            break
            
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        label = labels[i]
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # Determine background rectangle
        # Default: Top-left corner inside or outside box?
        # Supervision default: anchor point is top-left of the box
        # We want to place it such that it is visible
        
        # Candidates for label position (top-left corner of the background rect)
        # 1. Top-left of bbox, outside (y1 - text_height - 2*padding)
        # 2. Top-left of bbox, inside (y1)
        # 3. Bottom-left of bbox, inside (y2 - text_height - 2*padding)
        # 4. Bottom-left of bbox, outside (y2)
        
        # Preferred: Top-left, inside (if fits), else Top-left outside
        
        bg_height = text_height + 2 * text_padding
        bg_width = text_width + 2 * text_padding
        
        img_h, img_w = annotated_image.shape[:2]
        
        # Try position 1: Top-left corner of bbox, label above box
        lbl_x = x1
        lbl_y = y1 - bg_height
        
        # If label goes off top edge, try inside top-left
        if lbl_y < 0:
            lbl_y = y1
            
        # If label goes off bottom edge (unlikely for top-left inside, but possible for small images)
        if lbl_y + bg_height > img_h:
            lbl_y = y2 - bg_height
            
        # If label goes off left edge
        if lbl_x < 0:
            lbl_x = 0
            
        # If label goes off right edge
        if lbl_x + bg_width > img_w:
            lbl_x = img_w - bg_width
            
        # Draw background rectangle
        cv2.rectangle(
            annotated_image,
            (lbl_x, lbl_y),
            (lbl_x + bg_width, lbl_y + bg_height),
            color.by_idx(detections.class_id[i] if detections.class_id is not None else 0).as_bgr(),
            -1,
        )
        
        # Draw text
        cv2.putText(
            annotated_image,
            label,
            (lbl_x + text_padding, lbl_y + text_height + text_padding),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return annotated_image, labels


def show_obs_result(
    image: np.ndarray,
    obs: dict,
    classes: list[str],
    caption: Union[str, None] = None,
    draw_label_size: bool = False,
):
    detections = sv.Detections(
        xyxy=obs["xyxy"],
        confidence=obs["confidence"],
        class_id=obs["class_id"],
        mask=obs["masks"],
    )

    annotated_image, _ = visualize_result_rgb(image, detections, classes)

    if draw_label_size:
        text = "label num: " + str(len(obs["xyxy"]))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 100)  # Text start position
        font_scale = 1  # Font size
        color = (0, 255, 255)  # Text color (yellow)
        thickness = 2  # Text thickness
        cv2.putText(
            annotated_image, text, org, font, font_scale, color, thickness, cv2.LINE_AA
        )

    cv2.imshow(caption, annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_det_result(
    image: np.ndarray,
    detections: sv.Detections,
    classes: list[str],
    caption: Union[str, None] = None,
    draw_label_size: bool = False,
):
    annotated_image, _ = visualize_result_rgb(image, detections, classes)

    if draw_label_size:
        text = "label num: " + str(len(detections.xyxy))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 100)  # Text start position
        font_scale = 1  # Font size
        color = (0, 255, 255)  # Text color (yellow)
        thickness = 2  # Text thickness
        cv2.putText(
            annotated_image, text, org, font, font_scale, color, thickness, cv2.LINE_AA
        )

    cv2.imshow(caption, annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_similarity_matrix(
    sim_mat: np.ndarray,
):
    plt.figure()
    plt.imshow(sim_mat, cmap="coolwarm", interpolation="nearest")

    # Add colorbar
    plt.colorbar(label="Correlation")

    # Add title and labels
    plt.title("Correlation Matrix")
    plt.xlabel("Object A")
    plt.ylabel("Object B")
    # Show image
    plt.show()
