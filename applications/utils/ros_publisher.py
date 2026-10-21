import struct
import time

import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, String


class ROSPublisher:
    def __init__(self, node, cfg):
        self.cfg = cfg
        self.node = node
        self.bridge = CvBridge()
        
        # Track if robot has reached goal position
        self.goal_reached = False
        self.goal_reach_threshold = 0.3  # meters - distance threshold to consider goal reached
        
        # Track robot movement to detect when it stops (required for rotation setting)
        self.last_position = None
        self.position_history = []  # Store recent positions to detect if robot stopped
        self.movement_threshold = 0.05  # meters - if movement < this, consider robot stopped
        self.position_history_size = 10  # Number of recent positions to track

        # Create all required publishers
        self.global_path_publisher = node.create_publisher(Path, "/global_path", 10)
        self.local_path_publisher = node.create_publisher(Path, "/local_path", 10)
        self.action_path_publisher = node.create_publisher(Path, "/action_path", 10)

        self.image_publisher = node.create_publisher(Image, "/annotated_image", 10)
        self.fs_image_publisher = node.create_publisher(Image, "/fastsam_image", 10)
        self.fs_image_after_publisher = node.create_publisher(
            Image, "/fastsam_image_after", 10
        )

        self.pose_publisher = node.create_publisher(Odometry, "/odom", 10)
        
        # Target pose publisher for setting camera orientation at goal
        self.target_pose_publisher = node.create_publisher(PoseStamped, "/target_pose", 10)
        
        # Rotation command publisher for rotating camera after reaching goal
        # Publish to /cmd_vel so the robot controller can receive rotation commands
        self.cmd_vel_publisher = node.create_publisher(Twist, "/cmd_vel", 10)
        
        # Discrete rotation command publisher
        self.discrete_rot_publisher = node.create_publisher(String, "/rotation_command", 10)
        
        # Track rotation completion
        self.rotation_completed = False
        self.yaw_completed = False
        self.pitch_completed = False
        self.rotation_threshold = 0.1  # radians - angle threshold to consider rotation completed
        self.rotation_angular_velocity = 0.5  # rad/s - angular velocity for rotation command
        
        # Control mode: 'velocity' (use cmd_vel), 'direct' (publish target pose directly), 'discrete', or 'hybrid'
        # 'hybrid': Yaw uses velocity (cmd_vel), Pitch uses discrete commands (rotation_command)
        # FORCE hybrid mode as requested by user
        self.rotation_control_mode = 'hybrid' 
        self.kp_angular = 0.5  # Proportional gain for velocity control
        self.rotation_threshold = 0.1  # radians (approx 2.86 degrees) - Tighter threshold for better alignment
        self.node.get_logger().warning(
            f"[ROSPublisher] Initialized with rotation_control_mode: {self.rotation_control_mode} (FORCED HYBRID)"
        )

        self.local_rgb_publisher = node.create_publisher(
            PointCloud2, "/local_map/rgb", 10
        )
        self.local_sem_publisher = node.create_publisher(
            PointCloud2, "/local_map/semantic", 10
        )

        self.global_rgb_publisher = node.create_publisher(
            PointCloud2, "/global_map/rgb", 10
        )
        self.global_sem_publisher = node.create_publisher(
            PointCloud2, "/global_map/semantic", 10
        )

    def publish_all(self, oasis):
        """
        Publish all messages: paths, images, poses, and point clouds.
        """
        # 1. Publish paths
        # Pass goal pose into path publisher when publishing global path
        target_final_pose = getattr(oasis, 'target_final_pose', None)
        if target_final_pose is not None and oasis.curr_global_path is not None:
            # Reduce log frequency - only log occasionally
            if not hasattr(self, '_last_global_path_log_time'):
                self._last_global_path_log_time = 0
            current_time = time.time()
            if current_time - self._last_global_path_log_time > 2.0:  # Log every 2 seconds
                self.node.get_logger().warning(
                    f"[ROSPublisher] Publishing global_path with target_final_pose. "
                    f"Path length: {len(oasis.curr_global_path)}, "
                    f"Target position: {target_final_pose[:3, 3]}, "
                    f"Control mode: {self.rotation_control_mode}"
                )
                self._last_global_path_log_time = current_time
        elif target_final_pose is None:
            # Log once when target_final_pose is None
            if not hasattr(self, '_warned_no_target_pose_init'):
                self.node.get_logger().warning(
                    "[ROSPublisher] target_final_pose is None - no target pose available"
                )
                self._warned_no_target_pose_init = True
        self._publish_path(oasis.curr_global_path, "global", target_final_pose=target_final_pose)
        self._publish_path(oasis.curr_local_path, "local")
        self._publish_path(oasis.action_path, "action")

        # 2. Check if robot has reached goal and publish rotation command for camera orientation
        # This should always run, regardless of use_rviz setting
        if target_final_pose is not None:
            # Log once when target_final_pose becomes available
            if not hasattr(self, '_target_pose_available_logged'):
                self.node.get_logger().warning(
                    f"[ROSPublisher] target_final_pose is now available! "
                    f"Target position: {target_final_pose[:3, 3]}, Control mode: {self.rotation_control_mode}"
                )
                self._target_pose_available_logged = True
            # Always publish target pose when available (for path planning)
            # But also check if robot has reached goal position
            if oasis.curr_pose is not None:
                current_pos = oasis.curr_pose[:3, 3]
                target_pos = target_final_pose[:3, 3]
                distance_to_goal = np.linalg.norm(current_pos[:2] - target_pos[:2])  # XY distance only
                
                # Check for target pose change (content-wise)
                pose_changed = False
                if not hasattr(self, 'last_target_pose_matrix'):
                    self.last_target_pose_matrix = None
                
                if self.last_target_pose_matrix is None:
                    self.last_target_pose_matrix = target_final_pose.copy()
                elif not np.allclose(target_final_pose, self.last_target_pose_matrix, atol=1e-3):
                    pose_changed = True
                    self.last_target_pose_matrix = target_final_pose.copy()

                # Check if this is a new goal (distance significantly larger than threshold OR pose changed)
                # If so, reset completion flags to allow movement
                if (distance_to_goal > self.goal_reach_threshold * 2) or pose_changed:
                    if self.goal_reached or self.rotation_completed:
                        reason = "distance" if distance_to_goal > self.goal_reach_threshold * 2 else "pose_changed"
                        self.node.get_logger().warning(
                            f"[ROSPublisher] 🔄 New goal detected ({reason}). "
                            f"Resetting goal_reached and rotation status."
                        )
                        self.goal_reached = False
                        self.rotation_completed = False
                        self.yaw_completed = False
                        self.pitch_completed = False
                
                # Debug logging (only log occasionally to avoid spam)
                if not hasattr(self, '_last_debug_log_time'):
                    self._last_debug_log_time = 0
                current_time = time.time()
                if current_time - self._last_debug_log_time > 5.0:  # Log every 5 seconds (reduced frequency)
                    self.node.get_logger().warning(
                        f"[ROSPublisher] Target pose check: distance={distance_to_goal:.3f}m, "
                        f"threshold={self.goal_reach_threshold}m, "
                        f"current_pos={current_pos[:2]}, target_pos={target_pos[:2]}"
                    )
                    self._last_debug_log_time = current_time
                
                # Check if rotation has started or completed - if so, ignore distance checks
                rotation_in_progress = self.goal_reached or (hasattr(self, 'rotation_completed') and self.rotation_completed)
                
                if distance_to_goal < self.goal_reach_threshold or rotation_in_progress:
                    if not self.goal_reached:
                        self.goal_reached = True
                        self.rotation_completed = False
                        self.yaw_completed = False
                        self.pitch_completed = False
                        self.node.get_logger().warning(
                            f"[ROSPublisher] ✅ Robot reached goal! Distance: {distance_to_goal:.3f}m. "
                            f"Publishing rotation command for camera orientation (yaw + pitch)."
                        )
                    
                    # Check if rotation is needed and publish rotation command
                    # For habitat-sim, we can directly set the target pose instead of using velocity control
                    if self.rotation_control_mode == 'direct':
                        # Direct pose setting mode: publish target pose directly
                        # The controller should set agent state to this pose when reached
                        # Reduce log frequency for this message
                        if not hasattr(self, '_last_direct_mode_log_time'):
                            self._last_direct_mode_log_time = 0
                        current_time = time.time()
                        if current_time - self._last_direct_mode_log_time > 2.0:  # Log every 2 seconds
                            self.node.get_logger().warning(
                                f"[ROSPublisher] 🔵 Direct mode: Calling _publish_target_pose_for_direct_setting. "
                                f"target_final_pose exists: {target_final_pose is not None}, "
                                f"curr_pose exists: {oasis.curr_pose is not None}, "
                                f"rotation_in_progress: {rotation_in_progress}"
                            )
                            self._last_direct_mode_log_time = current_time
                        self._publish_target_pose_for_direct_setting(target_final_pose, oasis.curr_pose, oasis)
                    elif self.rotation_control_mode == 'discrete':
                        # Discrete rotation mode: publish look_up/down/left/right commands
                        self._check_and_publish_discrete_rotation(oasis.curr_pose, target_final_pose, oasis)
                    elif self.rotation_control_mode == 'hybrid':
                        # Hybrid rotation mode: Yaw via velocity, Pitch via discrete commands
                        # self.node.get_logger().warning("[ROSPublisher] 🟢 Executing Hybrid Rotation Check...")
                        self._check_and_publish_hybrid_rotation(oasis.curr_pose, target_final_pose, oasis)
                    else:
                        # Velocity control mode: use cmd_vel for gradual rotation
                        self._check_and_publish_rotation_command(oasis.curr_pose, target_final_pose, oasis)
                        # Also publish target pose for reference in velocity mode
                        # self._publish_target_pose(target_final_pose) # STOP sending this to avoid direct setting!
                else:
                    # Not reached yet - only reset goal_reached if rotation hasn't started
                    if self.goal_reached and not rotation_in_progress:
                        self.goal_reached = False
                        self.rotation_completed = False
                        self.yaw_completed = False
                        self.pitch_completed = False
                        self.node.get_logger().warning(
                            f"[ROSPublisher] Robot moved away from goal. Distance: {distance_to_goal:.3f}m"
                        )
                    # Occasionally output current position even when not reached
                    if not hasattr(self, '_last_not_reached_pos_log_time'):
                        self._last_not_reached_pos_log_time = 0
                    current_time = time.time()
                    if current_time - self._last_not_reached_pos_log_time > 5.0:  # Log every 5 seconds
                        self.node.get_logger().warning(
                            f"[ROSPublisher] 📍 Current robot position (not reached goal yet): "
                            f"X={current_pos[0]:.3f}, Y={current_pos[1]:.3f}, Z={current_pos[2]:.3f}, "
                            f"Distance to goal: {distance_to_goal:.3f}m"
                        )
                        self._last_not_reached_pos_log_time = current_time
                    # Still publish target pose even when not reached (for path planning)
                    # In direct mode, controller might want to know the target pose even before reaching
                    if self.rotation_control_mode == 'direct':
                        # In direct mode, don't publish target pose until we're close
                        pass
                    else:
                        # self._publish_target_pose(target_final_pose) # DISABLED
                        pass
            else:
                # If no current pose, still publish target pose if available
                if not hasattr(self, '_warned_no_pose'):
                    self.node.get_logger().warning(
                        "[ROSPublisher] Warning: curr_pose is None, cannot check goal distance"
                    )
                    self._warned_no_pose = True
                # self._publish_target_pose(target_final_pose) # DISABLED
        else:
            # Debug: log when target_final_pose is None
            if not hasattr(self, '_warned_no_target_pose'):
                self.node.get_logger().warning(
                    "[ROSPublisher] Warning: target_final_pose is None, cannot publish rotation command"
                )
                self._warned_no_target_pose = True

        if self.cfg.use_rviz:

            # 3. Publish images
            self._publish_image(oasis.detector.annotated_image, "annotated")
            self._publish_image(oasis.detector.annotated_image_fs, "fastsam")
            self._publish_image(
                oasis.detector.annotated_image_fs_after, "fastsam_after"
            )

            # 4. Publish pose
            self._publish_pose(oasis.curr_pose)

            # 5. Publish local map (RGB point cloud + semantic point cloud)
            if len(oasis.local_map_manager.local_map):
                start_time = time.time()
                self._publish_local_map(
                    oasis.local_map_manager, oasis.visualizer, publish_rgb=False
                )
                print(
                    f"Publishing local map took {time.time() - start_time:.2f} seconds."
                )

            if len(oasis.global_map_manager.global_map):
                self._publish_global_map(
                    oasis.global_map_manager, oasis.visualizer, publish_rgb=False
                )

    def _publish_path(self, path, path_type, target_final_pose=None):
        if path is None:
            return

        path_msg = Path()
        path_msg.header.stamp = self.node.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for i, pos in enumerate(path):
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = pos[0]
            pose_stamped.pose.position.y = pos[1]
            pose_stamped.pose.position.z = pos[2]
            
            # Last waypoint: use goal orientation if provided
            if i == len(path) - 1 and target_final_pose is not None:
                quaternion = self.rotation_matrix_to_quaternion(target_final_pose[:3, :3])
                pose_stamped.pose.orientation.x = float(quaternion[0])
                pose_stamped.pose.orientation.y = float(quaternion[1])
                pose_stamped.pose.orientation.z = float(quaternion[2])
                pose_stamped.pose.orientation.w = float(quaternion[3])
                
                # Log the target orientation for debugging (reduced frequency)
                if path_type == "global":
                    if not hasattr(self, '_last_path_orientation_log_time'):
                        self._last_path_orientation_log_time = 0
                    current_time = time.time()
                    if current_time - self._last_path_orientation_log_time > 2.0:  # Log every 2 seconds
                        self.node.get_logger().info(
                            f"[ROSPublisher] Setting final path point orientation: "
                            f"quaternion({quaternion[3]:.6f}, {quaternion[0]:.6f}, {quaternion[1]:.6f}, {quaternion[2]:.6f})"
                        )
                        self._last_path_orientation_log_time = current_time
            else:
                pose_stamped.pose.orientation.w = 1.0
            path_msg.poses.append(pose_stamped)

        publisher = {
            "global": self.global_path_publisher,
            "local": self.local_path_publisher,
            "action": self.action_path_publisher,
        }.get(path_type, None)

        if publisher:
            publisher.publish(path_msg)
            # self.node.get_logger().info(f"Published {path_type} path.")

    def _publish_image(self, image, image_type):
        if image is None:
            return

        ros_image = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        publisher = {
            "annotated": self.image_publisher,
            "fastsam": self.fs_image_publisher,
            "fastsam_after": self.fs_image_after_publisher,
        }.get(image_type, None)

        if publisher:
            publisher.publish(ros_image)

    def _publish_pose(self, pose_matrix):
        if pose_matrix is None:
            return

        # Extract translation and rotation
        translation = pose_matrix[:3, 3]
        quaternion = self.rotation_matrix_to_quaternion(pose_matrix[:3, :3])

        # Create Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.node.get_clock().now().to_msg()  # Timestamp
        odom_msg.header.frame_id = "map"  # Parent frame (global coordinate system)
        odom_msg.child_frame_id = ""  # Child frame (robot coordinate system)

        # Set position and orientation
        odom_msg.pose.pose.position.x = translation[0]
        odom_msg.pose.pose.position.y = translation[1]
        odom_msg.pose.pose.position.z = translation[2]
        odom_msg.pose.pose.orientation.x = float(quaternion[0])
        odom_msg.pose.pose.orientation.y = float(quaternion[1])
        odom_msg.pose.pose.orientation.z = float(quaternion[2])
        odom_msg.pose.pose.orientation.w = float(quaternion[3])

        # Set velocity information (default to 0)
        odom_msg.twist.twist.linear.x = 0.0
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = 0.0

        # Publish Odometry message
        self.pose_publisher.publish(odom_msg)

    def _publish_target_pose(self, pose_matrix):
        """
        Publish target pose for setting camera orientation at goal.
        
        Parameters:
        - pose_matrix: 4x4 numpy array representing the target transformation matrix.
        """
        if pose_matrix is None:
            self.node.get_logger().error("[ROSPublisher] _publish_target_pose: pose_matrix is None!")
            return

        try:
            # Check if publisher exists
            if self.target_pose_publisher is None:
                self.node.get_logger().error("[ROSPublisher] target_pose_publisher is None!")
                return

            # Extract translation and rotation
            translation = pose_matrix[:3, 3]
            quaternion = self.rotation_matrix_to_quaternion(pose_matrix[:3, :3])
            
            # Log the full pose matrix for debugging coordinate issues
            self.node.get_logger().warning(
                f"[ROSPublisher] 🔵 Extracted pose: translation={translation}, quaternion={quaternion}"
            )
            self.node.get_logger().warning(
                f"[ROSPublisher] 🔵 Full pose matrix:\n{pose_matrix}"
            )
            self.node.get_logger().warning(
                f"[ROSPublisher] 🔵 Position breakdown: X={translation[0]:.3f}, Y={translation[1]:.3f}, Z={translation[2]:.3f}"
            )

            # Create PoseStamped message
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.node.get_clock().now().to_msg()
            pose_msg.header.frame_id = "map"

            # Set position and orientation
            pose_msg.pose.position.x = float(translation[0])
            pose_msg.pose.position.y = float(translation[1])
            pose_msg.pose.position.z = float(translation[2])
            pose_msg.pose.orientation.x = float(quaternion[0])
            pose_msg.pose.orientation.y = float(quaternion[1])
            pose_msg.pose.orientation.z = float(quaternion[2])
            pose_msg.pose.orientation.w = float(quaternion[3])
            
            self.node.get_logger().warning(
                f"[ROSPublisher] 🔵 Created PoseStamped message: "
                f"position=({pose_msg.pose.position.x:.3f}, {pose_msg.pose.position.y:.3f}, {pose_msg.pose.position.z:.3f}), "
                f"orientation=({pose_msg.pose.orientation.x:.3f}, {pose_msg.pose.orientation.y:.3f}, "
                f"{pose_msg.pose.orientation.z:.3f}, {pose_msg.pose.orientation.w:.3f})"
            )

            # Check publisher status
            subscription_count = self.target_pose_publisher.get_subscription_count()
            self.node.get_logger().warning(
                f"[ROSPublisher] 🔵 Publisher status: subscription_count={subscription_count}, "
                f"topic_name={self.target_pose_publisher.topic_name}"
            )
            
            # Publish target pose message
            self.node.get_logger().warning(
                f"[ROSPublisher] 🔵 About to publish to /target_pose topic..."
            )
            try:
                self.target_pose_publisher.publish(pose_msg)
                self.node.get_logger().warning(
                    f"[ROSPublisher] ✅ Successfully called publish() on /target_pose topic! "
                    f"(Note: message may be queued if no subscribers are connected)"
                )
            except Exception as publish_error:
                self.node.get_logger().error(
                    f"[ROSPublisher] ❌ Error during publish(): {str(publish_error)}"
                )
                raise
            
            # Log occasionally to confirm publishing
            if not hasattr(self, '_last_target_pose_publish_log_time'):
                self._last_target_pose_publish_log_time = 0
            current_time = time.time()
            if current_time - self._last_target_pose_publish_log_time > 2.0:  # Log every 2 seconds
                self.node.get_logger().warning(
                    f"[ROSPublisher] Published target_pose: position=[{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}], "
                    f"quaternion=[{quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f}]"
                )
                self._last_target_pose_publish_log_time = current_time
        except Exception as e:
            self.node.get_logger().error(
                f"[ROSPublisher] ❌ Error in _publish_target_pose: {str(e)}"
            )
            import traceback
            self.node.get_logger().error(f"[ROSPublisher] Traceback: {traceback.format_exc()}")

    def _publish_target_pose_for_direct_setting(self, target_pose, current_pose=None, oasis=None):
        """
        Publish target pose for direct setting (e.g., habitat-sim agent.set_state()).
        This is used when rotation_control_mode is 'direct'.
        
        Parameters:
        - target_pose: 4x4 numpy array representing target transformation matrix
        - current_pose: 4x4 numpy array representing current transformation matrix (for checking if reached)
        - oasis: Optional Oasis instance for clearing target_final_pose after completion
        """
        # Reduce log frequency for entry message
        if not hasattr(self, '_last_direct_setting_entry_log_time'):
            self._last_direct_setting_entry_log_time = 0
        current_time = time.time()
        if current_time - self._last_direct_setting_entry_log_time > 2.0:  # Log every 2 seconds
            self.node.get_logger().warning(
                f"[ROSPublisher] 🟢 _publish_target_pose_for_direct_setting ENTERED! "
                f"target_pose is None: {target_pose is None}, "
                f"current_pose is None: {current_pose is None}"
            )
            self._last_direct_setting_entry_log_time = current_time
        if target_pose is None:
            self.node.get_logger().warning(
                "[ROSPublisher] _publish_target_pose_for_direct_setting: target_pose is None!"
            )
            return
        
        # Always publish target pose when in direct mode (controller will handle when to apply it)
        # But check if we're close enough to the target position
        if current_pose is not None:
            current_pos = current_pose[:3, 3]
            target_pos = target_pose[:3, 3]
            distance_to_goal = np.linalg.norm(current_pos[:2] - target_pos[:2])
            
            # Only publish target pose if we're close to the target position
            # Reduce log frequency for distance check
            if not hasattr(self, '_last_distance_check_log_time'):
                self._last_distance_check_log_time = 0
            current_time = time.time()
            if current_time - self._last_distance_check_log_time > 2.0:  # Log every 2 seconds
                self.node.get_logger().warning(
                    f"[ROSPublisher] 🟡 Distance check: distance_to_goal={distance_to_goal:.3f}m, "
                    f"threshold={self.goal_reach_threshold}m, "
                    f"current_pos={current_pos}, target_pos={target_pos}"
                )
                self._last_distance_check_log_time = current_time
            # Only set rotation when robot has stopped moving (no distance check)
            should_set_rotation = False
            rotation_reason = ""
            
            # Track position history to detect if robot stopped moving
            self.position_history.append(current_pos.copy())
            if len(self.position_history) > self.position_history_size:
                self.position_history.pop(0)
            
            # Check if robot has stopped moving (last few positions are very similar)
            # OR if rotation has already started (ignore distance after rotation starts)
            rotation_already_started = self.goal_reached
            
            if len(self.position_history) >= 5:
                recent_positions = np.array(self.position_history[-5:])
                # Calculate movement in last few frames
                position_changes = np.diff(recent_positions, axis=0)
                total_movement = np.sum(np.linalg.norm(position_changes, axis=1))
                avg_movement_per_frame = total_movement / len(position_changes)
                
                if avg_movement_per_frame < self.movement_threshold or rotation_already_started:
                    should_set_rotation = True
                    if rotation_already_started:
                        rotation_reason = f"Rotation already started (ignoring distance check, current distance={distance_to_goal:.3f}m)"
                    else:
                        rotation_reason = f"Robot stopped moving (avg_movement={avg_movement_per_frame:.4f}m < {self.movement_threshold}m)"
            elif rotation_already_started:
                # If rotation already started but don't have enough position history, still set rotation
                should_set_rotation = True
                rotation_reason = f"Rotation already started (position history insufficient, current distance={distance_to_goal:.3f}m)"
            
            if should_set_rotation:
                self.node.get_logger().warning(
                    f"[ROSPublisher] 🟢 Setting rotation: {rotation_reason}"
                )
                # Create a modified target pose: keep current position, use target rotation only
                # This prevents position errors and only changes camera orientation
                modified_target_pose = target_pose.copy()
                # Keep current position (X, Y, Z) - only change rotation
                modified_target_pose[0, 3] = current_pos[0]  # Keep current X
                modified_target_pose[1, 3] = current_pos[1]  # Keep current Y
                modified_target_pose[2, 3] = current_pos[2]  # Keep current Z
                
                self.node.get_logger().warning(
                    f"[ROSPublisher] 🔵 Modified target pose: "
                    f"Position preserved from current robot: X={current_pos[0]:.3f}, Y={current_pos[1]:.3f}, Z={current_pos[2]:.3f}, "
                    f"Rotation from target pose (only orientation will be set)"
                )
                
                # Publish target pose for direct setting
                # The controller should read this and set agent state directly
                # Note: Only rotation will be applied, position stays the same
                self._publish_target_pose(modified_target_pose)
                
                # Output current robot position after publishing target pose
                # This is important for debugging coordinate issues
                self.node.get_logger().warning(
                    f"[ROSPublisher] 📍 Current robot position (preserved): "
                    f"X={current_pos[0]:.3f}, Y={current_pos[1]:.3f}, Z={current_pos[2]:.3f}"
                )
                self.node.get_logger().warning(
                    f"[ROSPublisher] 📍 Only rotation will be set, position remains unchanged"
                )
                
                # Check if rotation is also aligned
                from scipy.spatial.transform import Rotation as R
                current_rot = current_pose[:3, :3]
                target_rot = target_pose[:3, :3]
                r_current = R.from_matrix(current_rot)
                r_target = R.from_matrix(target_rot)
                r_rel = r_target * r_current.inv()
                angle = r_rel.magnitude()
                
                # Calculate Euler angles for detailed logging
                euler_current = r_current.as_euler('xyz', degrees=True)
                euler_target = r_target.as_euler('xyz', degrees=True)
                yaw_diff = euler_target[2] - euler_current[2]
                pitch_diff = euler_target[1] - euler_current[1]
                
                # Normalize angles to [-180, 180]
                while yaw_diff > 180:
                    yaw_diff -= 360
                while yaw_diff < -180:
                    yaw_diff += 360
                while pitch_diff > 180:
                    pitch_diff -= 360
                while pitch_diff < -180:
                    pitch_diff += 360
                
                # CRITICAL: Always check rotation completion, regardless of logging frequency
                # This ensures rotation_completed flag is set immediately when rotation is done
                
                # Relaxed check: Only check Yaw alignment for ground robots
                # Convert yaw_diff (degrees) to radians for comparison
                yaw_diff_rad = np.radians(abs(yaw_diff))
                
                # Use angle (3D) OR yaw only (2D)
                # If the robot cannot pitch/roll, 3D angle will never be small enough
                # So we accept if Yaw is aligned properly
                if angle < self.rotation_threshold or yaw_diff_rad < self.rotation_threshold:
                    if not self.rotation_completed:
                        self.rotation_completed = True
                        self.node.get_logger().warning(
                            f"[ROSPublisher] ✅✅✅ Rotation completed! "
                            f"Distance: {distance_to_goal:.3f}m, Angle: {np.degrees(angle):.2f}°, Yaw Diff: {yaw_diff:.2f}° (Threshold: {np.degrees(self.rotation_threshold):.2f}°)"
                        )
                        # Clear target_final_pose after completion
                        if oasis is not None and hasattr(oasis, 'target_final_pose'):
                            oasis.target_final_pose = None
                            self.node.get_logger().warning(
                                "[ROSPublisher] Cleared target_final_pose after direct pose setting completion"
                            )
                        # Notify Oasis: rotation done; Core decides VLM check
                        if oasis is not None and hasattr(oasis, "rotation_completed"):
                            oasis.rotation_completed = True
                            self.node.get_logger().warning(
                                "[ROSPublisher] rotation_completed=True (Core may run VLM check)"
                            )
                else:
                    if self.rotation_completed:
                        self.rotation_completed = False
                        if oasis is not None and hasattr(oasis, 'rotation_completed'):
                            oasis.rotation_completed = False
                
                # Log rotation status occasionally (for debugging, but don't block rotation completion check)
                if not hasattr(self, '_last_direct_pose_log_time'):
                    self._last_direct_pose_log_time = 0
                current_time = time.time()
                should_log = (current_time - self._last_direct_pose_log_time > 2.0) or (not hasattr(self, '_direct_pose_logged_once'))
                
                # Force log if angle is close but not quite there (for debugging)
                if angle < self.rotation_threshold * 2 and angle >= self.rotation_threshold:
                    should_log = True

                if should_log:
                    if angle >= self.rotation_threshold:
                        self.node.get_logger().warning(
                            f"[ROSPublisher] ⏳ Waiting for rotation alignment... "
                            f"Current Angle Diff: {np.degrees(angle):.2f}° > Threshold: {np.degrees(self.rotation_threshold):.2f}° "
                            f"(Yaw Diff: {yaw_diff:.2f}°, Pitch Diff: {pitch_diff:.2f}°)"
                        )
                    self._last_direct_pose_log_time = current_time
                    if not hasattr(self, '_direct_pose_logged_once'):
                        self._direct_pose_logged_once = True
            else:
                # Robot is still moving, don't set rotation yet
                # Log movement status occasionally
                if len(self.position_history) >= 5:
                    recent_positions = np.array(self.position_history[-5:])
                    position_changes = np.diff(recent_positions, axis=0)
                    total_movement = np.sum(np.linalg.norm(position_changes, axis=1))
                    avg_movement_per_frame = total_movement / len(position_changes)
                    
                    if not hasattr(self, '_last_movement_log_time'):
                        self._last_movement_log_time = 0
                    current_time = time.time()
                    if current_time - self._last_movement_log_time > 2.0:  # Log every 2 seconds
                        self.node.get_logger().info(
                            f"[ROSPublisher] ⏳ Robot still moving: "
                            f"avg_movement={avg_movement_per_frame:.4f}m/frame "
                            f"(threshold={self.movement_threshold}m), "
                            f"distance_to_goal={distance_to_goal:.3f}m"
                        )
                        self._last_movement_log_time = current_time
                if self.rotation_completed:
                    self.rotation_completed = False
        else:
            # If no current pose, still publish target pose (controller will handle it)
            self.node.get_logger().warning(
                "[ROSPublisher] No current_pose available, publishing target_pose anyway"
            )
            self._publish_target_pose(target_pose)

    def _check_and_publish_discrete_rotation(self, current_pose, target_pose, oasis=None):
        """
        Check if rotation is needed and publish discrete rotation command (String).
        Commands: look_up, look_down, turn_left, turn_right
        """
        if current_pose is None or target_pose is None:
            return
        
        # Extract rotation matrices
        current_rot = current_pose[:3, :3]
        target_rot = target_pose[:3, :3]
        
        from scipy.spatial.transform import Rotation as R
        r_current = R.from_matrix(current_rot)
        r_target = R.from_matrix(target_rot)
        
        # --- STRATEGY: Hybrid Approach ---
        # 1. Yaw (Left/Right): Use ABSOLUTE Euler angle difference.
        #    Previous logs showed this works correctly for convergence.
        # 2. Pitch (Up/Down): Use RELATIVE rotation difference.
        #    This solves the "Pitch=0" issue caused by coordinate system offsets (Roll=-90).
        
        # 1. Calculate Absolute Yaw Diff
        euler_current = r_current.as_euler('xyz', degrees=True)
        euler_target = r_target.as_euler('xyz', degrees=True)
        
        yaw_diff_abs = euler_target[2] - euler_current[2]
        # Normalize to [-180, 180]
        while yaw_diff_abs > 180: yaw_diff_abs -= 360
        while yaw_diff_abs < -180: yaw_diff_abs += 360
        
        # 2. Calculate Relative Pitch Diff
        # R_rel = R_current^T * R_target (Rotation needed in Current Frame)
        r_rel = r_current.inv() * r_target
        euler_rel = r_rel.as_euler('xyz', degrees=True)
        pitch_diff_rel = euler_rel[1] # Use Relative Y-axis rotation for Pitch
        
        # Use these for control
        yaw_diff = yaw_diff_abs
        pitch_diff = pitch_diff_rel
        
        # Log for debugging
        self.node.get_logger().warning(
            f"[ROSPublisher] DEBUG Hybrid Control: "
            f"AbsYawDiff={yaw_diff:.2f}, RelPitchDiff={pitch_diff:.2f}, "
            f"(RelYawWas={euler_rel[2]:.2f})"
        )
        
        # Convert to radians for logic checks
        yaw_diff_rad = np.radians(yaw_diff)
        pitch_diff_rad = np.radians(pitch_diff)
        
        # Check thresholds
        yaw_needed = abs(yaw_diff_rad) > self.rotation_threshold
        pitch_needed = abs(pitch_diff_rad) > self.rotation_threshold
        
        command_msg = String()
        command = ""
        
        # Prioritize Yaw rotation
        if yaw_needed:
            if yaw_diff > 0:
                command = "turn_left"
            else:
                command = "turn_right"
            self.yaw_completed = False
        elif pitch_needed:
            if pitch_diff > 0:
                command = "look_up" # Assuming positive pitch is Look Up
            else:
                command = "look_down"
            self.pitch_completed = False
        else:
            # Completed
            if not self.rotation_completed:
                self.rotation_completed = True
                self.node.get_logger().warning(
                    f"[ROSPublisher] ✅ Discrete Rotation completed! RelYaw: {yaw_diff:.2f}, RelPitch: {pitch_diff:.2f}"
                )
                if oasis is not None and hasattr(oasis, 'target_final_pose'):
                    oasis.target_final_pose = None
                if oasis is not None and hasattr(oasis, 'need_post_rotation_vlm_check'):
                    if oasis.need_post_rotation_vlm_check and not oasis.post_rotation_vlm_check_done:
                        oasis.rotation_completed = True
            return

        # Publish command
        command_msg.data = command
        self.discrete_rot_publisher.publish(command_msg)
        
        self.node.get_logger().warning(
            f"[ROSPublisher] 🔄 Discrete Rotation: {command} (RelYaw: {yaw_diff:.2f}, RelPitch: {pitch_diff:.2f})"
        )

    def _check_and_publish_hybrid_rotation(self, current_pose, target_pose, oasis=None):
        """
        Hybrid rotation control:
        - Yaw: Continuous velocity control via cmd_vel (using Absolute Euler Diff)
        - Pitch: Discrete commands via rotation_command (using Relative Rotation Diff)
        """
        if current_pose is None or target_pose is None:
            return

        # Extract rotation matrices
        current_rot = current_pose[:3, :3]
        target_rot = target_pose[:3, :3]
        
        from scipy.spatial.transform import Rotation as R
        r_current = R.from_matrix(current_rot)
        r_target = R.from_matrix(target_rot)

        # DEBUG: Log Absolute Euler Angles to diagnose Pitch issues
        euler_current_abs = r_current.as_euler('xyz', degrees=True)
        euler_target_abs = r_target.as_euler('xyz', degrees=True)
        
        self.node.get_logger().warning(
            f"[ROSPublisher] DEBUG Rotation Check:\n"
            f"  Current Abs Euler (deg): R={euler_current_abs[0]:.2f}, P={euler_current_abs[1]:.2f}, Y={euler_current_abs[2]:.2f}\n"
            f"  Target  Abs Euler (deg): R={euler_target_abs[0]:.2f}, P={euler_target_abs[1]:.2f}, Y={euler_target_abs[2]:.2f}"
        )

        # 1. Calculate Absolute Yaw Diff (for Velocity Control)
        # We verified that absolute yaw diff works well for ground robot heading
        euler_current = r_current.as_euler('xyz', degrees=False) # Radians for velocity calc
        euler_target = r_target.as_euler('xyz', degrees=False)
        
        yaw_diff = euler_target[2] - euler_current[2]
        # Normalize to [-pi, pi]
        while yaw_diff > np.pi: yaw_diff -= 2 * np.pi
        while yaw_diff < -np.pi: yaw_diff += 2 * np.pi
        
        # 2. Calculate Relative Pitch Diff (for Discrete Control)
        # We verified that relative pitch diff works even with coordinate system offsets (Roll=-90)
        r_rel = r_current.inv() * r_target
        euler_rel = r_rel.as_euler('xyz', degrees=True) # Degrees for discrete logic
        
        # DEBUG: Log Relative Euler to find where the Pitch is hiding
        self.node.get_logger().warning(
            f"[ROSPublisher] DEBUG Relative Euler (deg): R={euler_rel[0]:.2f}, P={euler_rel[1]:.2f}, Y={euler_rel[2]:.2f}"
        )

        # INTELLIGENT PITCH DETECTION:
        # In Camera Frame (X-Right), Pitch is rotation around X-axis (index 0).
        # In ROS Frame (Y-Left), Pitch is rotation around Y-axis (index 1).
        # We check which axis has the dominant rotation (excluding Yaw/Z).
        
        if abs(euler_rel[0]) > abs(euler_rel[1]):
            # Dominant rotation is around X-axis (Roll) -> This is likely Camera Frame Pitch
            # Positive X-rotation (Right-hand rule around Right vector) usually means Look Up (Y-Down -> Z-Forward)
            pitch_diff_deg = euler_rel[0]
            self.node.get_logger().warning("[ROSPublisher] 🧠 Detected Pitch on ROLL axis (Camera Frame)")
        else:
            # Dominant rotation is around Y-axis (Pitch) -> This is likely ROS Frame Pitch
            pitch_diff_deg = euler_rel[1]
            self.node.get_logger().warning("[ROSPublisher] 🧠 Detected Pitch on PITCH axis (ROS Frame)")
            
        pitch_diff_rad = np.radians(pitch_diff_deg)

        # Check thresholds
        yaw_needed = abs(yaw_diff) > self.rotation_threshold
        pitch_needed = abs(pitch_diff_rad) > self.rotation_threshold
        
        # --- Control Logic ---
        
        # Priority 1: Yaw Alignment via Velocity
        # We prioritize Yaw because camera frame pitch is often relative to the base heading
        if yaw_needed:
            cmd_vel = Twist()
            # Proportional control for Yaw
            # Increase gain slightly for faster response if needed, currently self.kp_angular (0.5)
            yaw_velocity = np.clip(yaw_diff * self.kp_angular, -self.rotation_angular_velocity, self.rotation_angular_velocity)
            cmd_vel.angular.z = float(yaw_velocity)
            
            # Publish cmd_vel
            self.cmd_vel_publisher.publish(cmd_vel)
            self.yaw_completed = False
            
            # Log occasionally
            self.node.get_logger().warning(
                f"[ROSPublisher] 🔄 Hybrid Yaw (Velocity): val={yaw_velocity:.3f} rad/s (Diff: {np.degrees(yaw_diff):.2f}°)"
            )
            return # Block Pitch control until Yaw is aligned

        else:
            # Stop Yaw rotation if needed
            if not self.yaw_completed:
                stop_vel = Twist()
                self.cmd_vel_publisher.publish(stop_vel)
                self.yaw_completed = True
                self.node.get_logger().warning("[ROSPublisher] ✅ Hybrid Yaw Aligned!")

        # Priority 2: Pitch Alignment via Discrete Commands
        # Only start Pitch control after Yaw is aligned
        if pitch_needed:
            command_msg = String()
            if pitch_diff_deg > 0:
                command = "look_up" # Assuming positive Y-rotation relative to camera is Look Up
            else:
                command = "look_down"
                
            command_msg.data = command
            self.discrete_rot_publisher.publish(command_msg)
            self.pitch_completed = False
            
            self.node.get_logger().warning(
                f"[ROSPublisher] 🔄 Hybrid Pitch (Discrete): {command} (Diff: {pitch_diff_deg:.2f}°)"
            )
            return

        else:
            self.pitch_completed = True

        # Completion Check
        if self.yaw_completed and self.pitch_completed:
            if not self.rotation_completed:
                self.rotation_completed = True
                self.node.get_logger().warning(
                    f"[ROSPublisher] ✅✅ Hybrid Rotation ALL Completed! (YawDiff: {np.degrees(yaw_diff):.2f}°, PitchDiff: {pitch_diff_deg:.2f}°)"
                )
                if oasis is not None and hasattr(oasis, 'target_final_pose'):
                    oasis.target_final_pose = None
                if oasis is not None and hasattr(oasis, 'need_post_rotation_vlm_check'):
                    if oasis.need_post_rotation_vlm_check and not oasis.post_rotation_vlm_check_done:
                        oasis.rotation_completed = True
            return

    # Old implementation below for reference/replacement
    def _check_and_publish_rotation_command_OLD(self, current_pose, target_pose, oasis=None):
        """
        Check if rotation is needed and publish rotation command.
        
        Parameters:
        - current_pose: 4x4 numpy array representing current transformation matrix
        - target_pose: 4x4 numpy array representing target transformation matrix
        - oasis: Optional Oasis instance for clearing target_final_pose after rotation
        """
        if current_pose is None or target_pose is None:
            self.node.get_logger().warning(
                f"[ROSPublisher] Cannot check rotation: current_pose={current_pose is not None}, "
                f"target_pose={target_pose is not None}"
            )
            return
        
        # Extract rotation matrices
        current_rot = current_pose[:3, :3]
        target_rot = target_pose[:3, :3]
        
        # Calculate relative rotation: R_rel = R_target * R_current^T
        relative_rot = target_rot @ current_rot.T
        
        # Convert rotation matrix to axis-angle representation
        # Using scipy's Rotation class for robust conversion
        from scipy.spatial.transform import Rotation as R
        r_current = R.from_matrix(current_rot)
        r_target = R.from_matrix(target_rot)
        r_rel = r_target * r_current.inv()
        
        # Get rotation angle (magnitude)
        angle = r_rel.magnitude()
        
        # Get rotation axis
        axis_angle = r_rel.as_rotvec()
        
            # Check if rotation is needed
        # Calculate Euler angles (XYZ order: roll, pitch, yaw)
        euler_current = r_current.as_euler('xyz', degrees=False)
        euler_target = r_target.as_euler('xyz', degrees=False)
        
        # Calculate yaw difference (rotation around Z axis - horizontal rotation)
        yaw_diff = euler_target[2] - euler_current[2]
        # Normalize yaw difference to [-pi, pi]
        while yaw_diff > np.pi:
            yaw_diff -= 2 * np.pi
        while yaw_diff < -np.pi:
            yaw_diff += 2 * np.pi
        
        # Calculate pitch difference (rotation around Y axis - vertical rotation, look up/down)
        pitch_diff = euler_target[1] - euler_current[1]
        # Normalize pitch difference to [-pi, pi]
        while pitch_diff > np.pi:
            pitch_diff -= 2 * np.pi
        while pitch_diff < -np.pi:
            pitch_diff += 2 * np.pi
        
        # Check if rotation is needed
        yaw_needed = abs(yaw_diff) > self.rotation_threshold
        # FORCE IGNORE PITCH for ground robots to avoid getting stuck
        pitch_needed = False # abs(pitch_diff) > self.rotation_threshold
        
        if yaw_needed or pitch_needed:
            # Create rotation command (Twist message)
            cmd_vel = Twist()
            
            # Set yaw angular velocity (horizontal rotation)
            if yaw_needed:
                yaw_angular_vel = np.clip(yaw_diff * 2.0, -self.rotation_angular_velocity, self.rotation_angular_velocity)
                cmd_vel.angular.z = float(yaw_angular_vel)
                self.yaw_completed = False
            else:
                cmd_vel.angular.z = 0.0
                self.yaw_completed = True
            
            # Set pitch angular velocity (vertical rotation, look up/down)
            # Note: angular.y in Twist typically controls pitch for some robot platforms
            # If your robot controller doesn't support this, you may need a separate camera control topic
            if pitch_needed:
                pitch_angular_vel = np.clip(pitch_diff * 2.0, -self.rotation_angular_velocity, self.rotation_angular_velocity)
                cmd_vel.angular.y = float(pitch_angular_vel)  # Pitch control via angular.y
                self.pitch_completed = False
            else:
                cmd_vel.angular.y = 0.0
                self.pitch_completed = True
            
            # Keep linear velocity zero during rotation
            cmd_vel.linear.x = 0.0
            cmd_vel.linear.y = 0.0
            cmd_vel.linear.z = 0.0
            cmd_vel.angular.x = 0.0  # Roll is typically not controlled
            
            # Publish rotation command to /cmd_vel topic
            self.cmd_vel_publisher.publish(cmd_vel)
            
            if not self.rotation_completed:
                yaw_str = f"yaw: {np.degrees(yaw_diff):.2f}°" if yaw_needed else "yaw: ✓"
                pitch_str = f"pitch: {np.degrees(pitch_diff):.2f}°" if pitch_needed else "pitch: ✓"
                self.node.get_logger().warning(
                    f"[ROSPublisher] 🔄 Rotation needed: {yaw_str}, {pitch_str}. "
                    f"Publishing cmd_vel: angular.z={cmd_vel.angular.z:.3f} rad/s, "
                    f"angular.y={cmd_vel.angular.y:.3f} rad/s"
                )
        else:
            # Both yaw and pitch are within threshold
            if not self.rotation_completed:
                self.rotation_completed = True
                self.yaw_completed = True
                self.pitch_completed = True
                self.node.get_logger().warning(
                    f"[ROSPublisher] ✅ Rotation completed! "
                    f"Final yaw: {np.degrees(yaw_diff):.2f}°, pitch: {np.degrees(pitch_diff):.2f}°"
                )
                # Clear target_final_pose after rotation is completed (if oasis is provided)
                # This allows get_action_path to reuse it if needed, but prevents infinite rotation
                if oasis is not None and hasattr(oasis, 'target_final_pose'):
                    oasis.target_final_pose = None
                    self.node.get_logger().warning(
                        "[ROSPublisher] Cleared target_final_pose after rotation completion"
                    )
                # Trigger post-rotation VLM check if needed
                if oasis is not None and hasattr(oasis, 'need_post_rotation_vlm_check'):
                    if oasis.need_post_rotation_vlm_check and not oasis.post_rotation_vlm_check_done:
                        oasis.rotation_completed = True
                        self.node.get_logger().warning(
                            "[ROSPublisher] Rotation completed, VLM check will be performed in next frame"
                        )
            # Stop rotation command (zero angular velocity)
            cmd_vel = Twist()
            cmd_vel.angular.z = 0.0
            cmd_vel.angular.y = 0.0
            cmd_vel.linear.x = 0.0
            cmd_vel.linear.y = 0.0
            cmd_vel.linear.z = 0.0
            cmd_vel.angular.x = 0.0
            self.cmd_vel_publisher.publish(cmd_vel)

    def _publish_local_map(self, local_map_manager, visualizer, publish_rgb=True):
        all_positions = []
        all_rgb_colors = []
        all_semantic_colors = []

        for local_obj in local_map_manager.local_map:
            obj_name = visualizer.obj_classes.get_classes_arr()[local_obj.class_id]
            positions = np.asarray(local_obj.pcd.points)
            colors = (np.asarray(local_obj.pcd.colors) * 255).astype(np.uint8)
            curr_obj_color = (
                np.array(visualizer.obj_classes.get_class_color(obj_name)) * 255
            )
            curr_obj_color = curr_obj_color.astype(np.uint8)
            semantic_colors = np.tile(curr_obj_color, (positions.shape[0], 1))

            all_positions.append(positions)
            all_rgb_colors.append(colors)
            all_semantic_colors.append(semantic_colors)

        if not all_positions:
            return

        all_positions = np.vstack(all_positions)
        all_rgb_colors = np.vstack(all_rgb_colors)
        all_semantic_colors = np.vstack(all_semantic_colors)

        if publish_rgb:
            self.publish_pointcloud(
                all_positions, all_rgb_colors, self.local_rgb_publisher, "map"
            )

        self.publish_pointcloud(
            all_positions, all_semantic_colors, self.local_sem_publisher, "map"
        )

    def _publish_global_map(self, global_map_manager, visualizer, publish_rgb=True):
        all_positions = []
        all_rgb_colors = []
        all_semantic_colors = []

        for global_obj in global_map_manager.global_map:
            obj_name = visualizer.obj_classes.get_classes_arr()[global_obj.class_id]
            positions = np.asarray(global_obj.pcd_2d.points)
            colors = (np.asarray(global_obj.pcd_2d.colors) * 255).astype(np.uint8)
            curr_obj_color = (
                np.array(visualizer.obj_classes.get_class_color(obj_name)) * 255
            )
            curr_obj_color = curr_obj_color.astype(np.uint8)
            semantic_colors = np.tile(curr_obj_color, (positions.shape[0], 1))

            all_positions.append(positions)
            all_rgb_colors.append(colors)
            all_semantic_colors.append(semantic_colors)

        if not all_positions:
            return

        all_positions = np.vstack(all_positions)
        all_rgb_colors = np.vstack(all_rgb_colors)
        all_semantic_colors = np.vstack(all_semantic_colors)

        if publish_rgb:
            self.publish_pointcloud(
                all_positions, all_rgb_colors, self.global_rgb_publisher, "map"
            )

        self.publish_pointcloud(
            all_positions, all_semantic_colors, self.global_sem_publisher, "map"
        )

    def publish_pointcloud(self, points, colors, publisher, frame_id):
        """
        Efficient point cloud publishing: batch pack RGB data with NumPy
        """
        num_points = points.shape[0]

        # Efficient RGB packing: avoid Python loops
        r = colors[:, 0].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 2].astype(np.uint32)
        rgb_packed = (r << 16) | (g << 8) | b  # RGB → UINT32

        # Efficient data concatenation: build PointCloud data at once
        cloud_data = np.zeros((num_points, 4), dtype=np.float32)
        cloud_data[:, :3] = points
        cloud_data[:, 3] = rgb_packed.view(np.float32)  # direct conversion

        # PointCloud2 field definition
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]

        # Message header
        header = Header()
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = frame_id

        # Build PointCloud2 message
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = header
        pointcloud_msg.height = 1
        pointcloud_msg.width = num_points
        pointcloud_msg.fields = fields
        pointcloud_msg.is_bigendian = False
        pointcloud_msg.point_step = 16
        pointcloud_msg.row_step = 16 * num_points
        pointcloud_msg.is_dense = True
        pointcloud_msg.data = cloud_data.tobytes()

        # Publish point cloud
        publisher.publish(pointcloud_msg)

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
