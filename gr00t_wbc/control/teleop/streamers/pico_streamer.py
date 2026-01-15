import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from gr00t_wbc.control.teleop.device.pico.xr_client import XrClient
from gr00t_wbc.control.teleop.streamers.base_streamer import BaseStreamer, StreamerOutput

R_HEADSET_TO_WORLD = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)


class PicoStreamer(BaseStreamer):
    def __init__(self):
        self.xr_client = XrClient()
        self.run_pico_service()

        self.reset_status()

    def run_pico_service(self):
        # Run the pico service
        self.pico_service_pid = subprocess.Popen(
            ["bash", "/opt/apps/roboticsservice/runService.sh"]
        )
        print(f"Pico service running with pid {self.pico_service_pid.pid}")

    def stop_pico_service(self):
        # find pid and kill it
        if self.pico_service_pid:
            subprocess.Popen(["kill", "-9", str(self.pico_service_pid.pid)])
            print(f"Pico service killed with pid {self.pico_service_pid.pid}")
        else:
            print("Pico service not running")

    def reset_status(self):
        self.current_base_height = 0.74  # Initial base height, 0.74m (standing height)
        self.toggle_policy_action_last = False
        self.toggle_activation_last = False

    def start_streaming(self):
        pass

    def stop_streaming(self):
        self.xr_client.close()

    def get(self) -> StreamerOutput:
        pico_data = self._get_pico_data()

        raw_data = self._generate_unified_raw_data(pico_data)
        return raw_data

    def __del__(self):
        pass

    def _get_pico_data(self):
        pico_data = {}

        # Get the pose of the left and right controllers and the headset
        pico_data["left_pose"] = self.xr_client.get_pose_by_name("left_controller")
        pico_data["right_pose"] = self.xr_client.get_pose_by_name("right_controller")
        pico_data["head_pose"] = self.xr_client.get_pose_by_name("headset")

        # Get key value of the left and right controllers
        pico_data["left_trigger"] = self.xr_client.get_key_value_by_name("left_trigger")
        pico_data["right_trigger"] = self.xr_client.get_key_value_by_name("right_trigger")
        pico_data["left_grip"] = self.xr_client.get_key_value_by_name("left_grip")
        pico_data["right_grip"] = self.xr_client.get_key_value_by_name("right_grip")

        # Get button state of the left and right controllers
        pico_data["A"] = self.xr_client.get_button_state_by_name("A")
        pico_data["B"] = self.xr_client.get_button_state_by_name("B")
        pico_data["X"] = self.xr_client.get_button_state_by_name("X")
        pico_data["Y"] = self.xr_client.get_button_state_by_name("Y")
        pico_data["left_menu_button"] = self.xr_client.get_button_state_by_name("left_menu_button")
        pico_data["right_menu_button"] = self.xr_client.get_button_state_by_name(
            "right_menu_button"
        )
        pico_data["left_axis_click"] = self.xr_client.get_button_state_by_name("left_axis_click")
        pico_data["right_axis_click"] = self.xr_client.get_button_state_by_name("right_axis_click")

        # Get the timestamp of the left and right controllers
        pico_data["timestamp"] = self.xr_client.get_timestamp_ns()

        # Get the hand tracking state of the left and right controllers
        pico_data["left_hand_tracking_state"] = self.xr_client.get_hand_tracking_state("left")
        pico_data["right_hand_tracking_state"] = self.xr_client.get_hand_tracking_state("right")

        # Get the joystick state of the left and right controllers
        pico_data["left_joystick"] = self.xr_client.get_joystick_state("left")
        pico_data["right_joystick"] = self.xr_client.get_joystick_state("right")

        # Get the motion tracker data
        pico_data["motion_tracker_data"] = self.xr_client.get_motion_tracker_data()

        # Get the body tracking data
        pico_data["body_tracking_data"] = self.xr_client.get_body_tracking_data()

        return pico_data

    def _generate_unified_raw_data(self, pico_data):
        # Get controller position and orientation in z up world frame
        left_controller_T = self._process_xr_pose(pico_data["left_pose"], pico_data["head_pose"])
        right_controller_T = self._process_xr_pose(pico_data["right_pose"], pico_data["head_pose"])

        # Get navigation commands
        DEAD_ZONE = 0.1
        MAX_LINEAR_VEL = 0.5  # m/s
        MAX_ANGULAR_VEL = 1.0  # rad/s

        fwd_bwd_input = pico_data["left_joystick"][1]
        strafe_input = -pico_data["left_joystick"][0]
        yaw_input = -pico_data["right_joystick"][0]

        lin_vel_x = self._apply_dead_zone(fwd_bwd_input, DEAD_ZONE) * MAX_LINEAR_VEL
        lin_vel_y = self._apply_dead_zone(strafe_input, DEAD_ZONE) * MAX_LINEAR_VEL
        ang_vel_z = self._apply_dead_zone(yaw_input, DEAD_ZONE) * MAX_ANGULAR_VEL

        # Note: X/Y buttons are now used for left thumb rotation
        
        # Get gripper commands (returns dict with finger/thumb values)
        left_fingers = self._generate_finger_data(pico_data, "left")
        right_fingers = self._generate_finger_data(pico_data, "right")

        # Get activation commands
        toggle_policy_action_tmp = pico_data["left_menu_button"] and (
            pico_data["left_trigger"] > 0.5
        )
        toggle_activation_tmp = pico_data["left_menu_button"] and (pico_data["right_trigger"] > 0.5)

        if self.toggle_policy_action_last != toggle_policy_action_tmp:
            toggle_policy_action = toggle_policy_action_tmp
        else:
            toggle_policy_action = False
        self.toggle_policy_action_last = toggle_policy_action_tmp

        if self.toggle_activation_last != toggle_activation_tmp:
            toggle_activation = toggle_activation_tmp
        else:
            toggle_activation = False
        self.toggle_activation_last = toggle_activation_tmp

        # Note: A/B buttons are now used for right thumb rotation
        # Data collection can be toggled via menu buttons or joystick clicks
        toggle_data_collection = False
        toggle_data_abort = False

        return StreamerOutput(
            ik_data={
                "left_wrist": left_controller_T,
                "right_wrist": right_controller_T,
                "left_fingers": left_fingers,  # Dict with "position" and "gripper_value"
                "right_fingers": right_fingers,  # Dict with "position" and "gripper_value"
            },
            control_data={
                "base_height_command": self.current_base_height,
                "navigate_cmd": [lin_vel_x, lin_vel_y, ang_vel_z],
                "toggle_policy_action": toggle_policy_action,
            },
            teleop_data={
                "toggle_activation": toggle_activation,
            },
            data_collection_data={
                "toggle_data_collection": toggle_data_collection,
                "toggle_data_abort": toggle_data_abort,
            },
            source="pico",
        )

    def _process_xr_pose(self, controller_pose, headset_pose):
        # Convert controller pose to x, y, z, w quaternion
        xr_pose_xyz = np.array(controller_pose)[:3]  # x, y, z
        xr_pose_quat = np.array(controller_pose)[3:]  # x, y, z, w

        # Handle all-zero quaternion case by using identity quaternion
        if np.allclose(xr_pose_quat, 0):
            xr_pose_quat = np.array([0, 0, 0, 1])  # identity quaternion: x, y, z, w

        # Convert from y up to z up
        xr_pose_xyz = R_HEADSET_TO_WORLD @ xr_pose_xyz
        xr_pose_rotation = R.from_quat(xr_pose_quat).as_matrix()
        xr_pose_rotation = R_HEADSET_TO_WORLD @ xr_pose_rotation @ R_HEADSET_TO_WORLD.T

        # Convert headset pose to x, y, z, w quaternion
        headset_pose_xyz = np.array(headset_pose)[:3]
        headset_pose_quat = np.array(headset_pose)[3:]

        if np.allclose(headset_pose_quat, 0):
            headset_pose_quat = np.array([0, 0, 0, 1])  # identity quaternion: x, y, z, w

        # Convert from y up to z up
        headset_pose_xyz = R_HEADSET_TO_WORLD @ headset_pose_xyz
        headset_pose_rotation = R.from_quat(headset_pose_quat).as_matrix()
        headset_pose_rotation = R_HEADSET_TO_WORLD @ headset_pose_rotation @ R_HEADSET_TO_WORLD.T

        # Calculate the delta between the controller and headset positions
        xr_pose_xyz_delta = xr_pose_xyz - headset_pose_xyz

        # Calculate the yaw of the headset
        R_headset_to_world = R.from_matrix(headset_pose_rotation)
        headset_pose_yaw = R_headset_to_world.as_euler("xyz")[2]  # Extract yaw (Z-axis rotation)
        inverse_yaw_rotation = R.from_euler("z", -headset_pose_yaw).as_matrix()

        # Align with headset yaw to controller position delta and rotation
        xr_pose_xyz_delta_compensated = inverse_yaw_rotation @ xr_pose_xyz_delta
        xr_pose_rotation_compensated = inverse_yaw_rotation @ xr_pose_rotation

        xr_pose_T = np.eye(4)
        xr_pose_T[:3, :3] = xr_pose_rotation_compensated
        xr_pose_T[:3, 3] = xr_pose_xyz_delta_compensated
        return xr_pose_T

    def _apply_dead_zone(self, value, dead_zone):
        """Apply dead zone and normalize."""
        if abs(value) < dead_zone:
            return 0.0
        sign = 1 if value > 0 else -1
        # Normalize the output to be between -1 and 1 after dead zone
        return sign * (abs(value) - dead_zone) / (1.0 - dead_zone)

    def _generate_finger_data(self, pico_data, hand):
        """Generate finger position data with analog trigger and grip values for decoupled control.
        
        For Inspire hands:
        - Trigger controls 4 fingers (pinky, ring, middle, index)
        - Grip controls thumb bend
        - Right hand: B (+) / A (-) for thumb rotation
        - Left hand: Y (+) / X (-) for thumb rotation
        
        This decoupling allows more natural grasping where thumb can be controlled
        independently for precision grips.
        """
        fingertips = np.zeros([25, 4, 4])

        trigger_val = pico_data[f"{hand}_trigger"]
        grip_val = pico_data[f"{hand}_grip"]
        
        # Get thumb rotation button states based on hand
        if hand == "right":
            # Right hand: B (+), A (-)
            thumb_rot_plus = pico_data["B"]
            thumb_rot_minus = pico_data["A"]
        else:
            # Left hand: Y (+), X (-)
            thumb_rot_plus = pico_data["Y"]
            thumb_rot_minus = pico_data["X"]

        # DEBUG: Print when grip or thumb buttons are pressed
        if grip_val > 0.1:
            print(f"[PicoDebug-{hand}] GRIP pressed: {grip_val:.2f}")
        if thumb_rot_plus:
            print(f"[PicoDebug-{hand}] THUMB_ROT+ pressed (B/Y)")
        if thumb_rot_minus:
            print(f"[PicoDebug-{hand}] THUMB_ROT- pressed (A/X)")
        if trigger_val > 0.1:
            print(f"[PicoDebug-{hand}] TRIGGER pressed: {trigger_val:.2f}")
        
        # Store all values for decoupled finger/thumb control
        # trigger_val: 0.0 = released (open), 1.0 = pressed (close) - controls 4 fingers
        # grip_val: 0.0 = released (open), 1.0 = pressed (close) - controls thumb bend
        # thumb_rot_plus/minus: button states for thumb rotation control
        finger_data = {
            "position": fingertips,
            "gripper_value": trigger_val,      # 4 fingers (pinky, ring, middle, index)
            "thumb_value": grip_val,           # Thumb bend control
            "thumb_rot_plus": thumb_rot_plus,  # Thumb rotation + button
            "thumb_rot_minus": thumb_rot_minus,  # Thumb rotation - button
        }

        return finger_data


if __name__ == "__main__":
    # from gr00t_wbc.control.utils.debugger import wait_for_debugger
    # wait_for_debugger()

    streamer = PicoStreamer()
    streamer.start_streaming()
    while True:
        raw_data = streamer.get()
        print(
            f"left_wrist: {raw_data.ik_data['left_wrist']}, right_wrist: {raw_data.ik_data['right_wrist']}"
        )
        time.sleep(0.1)
