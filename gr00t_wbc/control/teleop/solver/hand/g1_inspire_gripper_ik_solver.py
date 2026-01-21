"""
Inspire Hand IK Solver - Decoupled Finger/Thumb Control
Controls 4 fingers (pinky, ring, middle, index) via Pico trigger.
Controls thumb bend independently via Pico grip button.
Controls thumb rotation via buttons: Right B(+)/A(-), Left Y(+)/X(-).
This decoupling allows for more natural grasping where the thumb can be
positioned independently for precision grips.
Features smooth linear interpolation for natural hand movements.
Inspire hand DOF order: [pinky, ring, middle, index, thumb_bend, thumb_rotation]
Values are in range [0, 1] where 0=close, 1=open

Configuration can be customized via inspire_hand_config.yaml
"""

import os
import numpy as np
import yaml

from gr00t_wbc.control.teleop.solver.solver import Solver


# Output 7 DOF to match robot model (Dex3 compatibility), actual Inspire hand has 6 DOF
INSPIRE_ROBOT_MODEL_DOF = 7
INSPIRE_ACTUAL_DOF = 6

# Default config path (same directory as this file)
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "inspire_hand_config.yaml")


def load_hand_config(config_path=None):
    """Load hand configuration from YAML file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    # Default values if config file doesn't exist
    default_config = {
        "finger_limits": {
            "thumb_index_max_close": 0.9,
            "other_fingers_max_close": 0.7,
            "min_open_floor": 0.1,
        },
        "interpolation": {
            "finger_speed": 0.7,
            "thumb_speed": 0.5,
        },
        "thumb_rotation": {
            "increment": 0.05,
        },
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                print(f"[InspireHand] Loaded config from {config_path}")
                return config
        except Exception as e:
            print(f"[InspireHand] Warning: Could not load config from {config_path}: {e}")
            print("[InspireHand] Using default configuration")
    else:
        print(f"[InspireHand] Config file not found at {config_path}, using defaults")
    
    return default_config


class G1InspireGripperIKSolver(Solver):
    """
    IK Solver for Inspire hands - Decoupled finger/thumb control.
    
    Controls fingers and thumb independently:
    - Trigger controls 4 fingers (pinky, ring, middle, index)
    - Grip button controls thumb bend
    - Buttons control thumb rotation: Right B(+)/A(-), Left Y(+)/X(-)
    
    Control mapping:
    - Trigger released (0.0) → 4 fingers open (1.0)
    - Trigger pressed (1.0) → 4 fingers closed (0.0)
    - Grip released (0.0) → Thumb open (1.0)
    - Grip pressed (1.0) → Thumb closed (0.0)
    - Thumb rotation +/- buttons → Increment/decrement thumb rotation
    
    Features smooth interpolation to prevent jerky movements.
    Configuration loaded from inspire_hand_config.yaml
    
    The robot model defines 7 DOF per hand (for Dex3 compatibility), but Inspire
    hand only has 6 DOF. This solver outputs 7 values to match the robot model's
    joint indexing, which then gets mapped to 6 DOF in G1InspireHand.
    
    Output format (7 DOF for robot model compatibility):
    [pinky, ring, middle, index, thumb_bend, thumb_rotation, padding]
    
    Values are in range [0, 1] where 0=close, 1=open
    """

    def __init__(self, side, config_path=None) -> None:
        self.side = "L" if side.lower() == "left" else "R"
        # Output 7 DOF to match robot model, actual Inspire hand has 6 DOF
        self.robot_model_dof = INSPIRE_ROBOT_MODEL_DOF
        self.actual_dof = INSPIRE_ACTUAL_DOF

        # Load configuration from YAML
        self.config = load_hand_config(config_path)
        
        # Extract config values
        finger_limits = self.config.get("finger_limits", {})
        interpolation = self.config.get("interpolation", {})
        thumb_rotation = self.config.get("thumb_rotation", {})
        
        # Finger closing limits (convert max_close to min_open)
        # max_close 0.9 means finger can close 90%, so min_open = 1.0 - 0.9 = 0.1
        thumb_index_max_close = finger_limits.get("thumb_index_max_close", 0.9)
        other_fingers_max_close = finger_limits.get("other_fingers_max_close", 0.7)
        self.min_open_floor = finger_limits.get("min_open_floor", 0.1)
        
        self.thumb_index_min_open = 1.0 - thumb_index_max_close  # 0.1 for 90% close
        self.other_fingers_min_open = 1.0 - other_fingers_max_close  # 0.3 for 70% close
        
        # Ensure min_open doesn't go below the floor
        self.thumb_index_min_open = max(self.thumb_index_min_open, self.min_open_floor)
        self.other_fingers_min_open = max(self.other_fingers_min_open, self.min_open_floor)
        
        # Interpolation speeds
        self.finger_interpolation_speed = interpolation.get("finger_speed", 0.7)
        self.thumb_interpolation_speed = interpolation.get("thumb_speed", 0.5)
        
        # Thumb rotation increment
        self.thumb_rotation_increment = thumb_rotation.get("increment", 0.05)
        
        print(f"[InspireHand-{self.side}] Config: thumb/index min_open={self.thumb_index_min_open:.2f}, "
              f"other min_open={self.other_fingers_min_open:.2f}, "
              f"finger_speed={self.finger_interpolation_speed}, thumb_speed={self.thumb_interpolation_speed}")

        # Current interpolated positions (start fully open)
        self._current_finger_position = 1.0  # For 4 fingers
        self._current_thumb_position = 1.0   # For thumb bend
        self._current_thumb_rotation = 1.0   # For thumb rotation (1.0 = neutral/open)

    def register_robot(self, robot):
        pass

    def _lerp(self, current: float, target: float, speed: float) -> float:
        """Linear interpolation from current to target at given speed."""
        return current + (target - current) * speed

    def __call__(self, finger_data):
        """
        Convert trigger/grip/button values to Inspire hand joint positions.
        
        Uses linear interpolation for smooth movement.
        
        Args:
            finger_data: Dict with:
                - "gripper_value": analog trigger value [0, 1] for 4 fingers
                - "thumb_value": analog grip value [0, 1] for thumb bend
                - "thumb_rot_plus": button state for thumb rotation +
                - "thumb_rot_minus": button state for thumb rotation -
                where 0=released, 1=pressed
        
        Returns:
            np.ndarray: 7 DOF array (for robot model compatibility)
                       First 6 values are Inspire hand positions [0, 1]
                       where 0=close, 1=open
                       Last value is padding (0.0)
        """
        pinch_mode = finger_data.get("pinch_mode", False)
        # Initialize output array (7 DOF for robot model, but only first 6 are used)
        q_desired = np.ones(self.robot_model_dof)
        q_desired[-1] = 0.0  # Padding value

        # Control 4 fingers with trigger
        if "gripper_value" in finger_data:
            trigger_val = finger_data["gripper_value"]
            finger_target = 1.0 - trigger_val  # Invert: 0=close, 1=open

            # Apply linear interpolation for smooth movement
            self._current_finger_position = self._lerp(
                self._current_finger_position, 
                finger_target, 
                self.finger_interpolation_speed
            )

            # Clamp to valid range with different limits for different fingers
            # Index finger: uses thumb_index_min_open (e.g., 0.1 for 90% close)
            index_finger_open = np.clip(self._current_finger_position, self.thumb_index_min_open, 0.99)
            # Other fingers: uses other_fingers_min_open (e.g., 0.3 for 70% close)
            other_finger_open = np.clip(self._current_finger_position, self.other_fingers_min_open, 0.99)

            if pinch_mode:
                # Pinch mode: keep pinky/ring/middle open, index follows trigger
                q_desired[0] = 1.0  # pinky open
                q_desired[1] = 1.0  # ring open
                q_desired[2] = 1.0  # middle open
                q_desired[3] = index_finger_open  # index (0.9 max close)
            else:
                # Grip mode: 4 fingers follow trigger with different limits
                q_desired[0] = other_finger_open  # pinky (0.75 max close)
                q_desired[1] = other_finger_open  # ring (0.75 max close)
                q_desired[2] = other_finger_open  # middle (0.75 max close)
                q_desired[3] = index_finger_open  # index (0.9 max close)
        else:
            finger_open = 1.0

        # Control thumb bend independently with grip
        if "thumb_value" in finger_data:
            grip_val = finger_data["thumb_value"]
            thumb_target = 1.0 - grip_val  # Invert: 0=close, 1=open

            # Apply linear interpolation for smooth movement (slightly slower for precision)
            self._current_thumb_position = self._lerp(
                self._current_thumb_position, 
                thumb_target, 
                self.thumb_interpolation_speed
            )

            # Clamp to valid range - thumb uses same limit as index
            thumb_open = np.clip(self._current_thumb_position, self.thumb_index_min_open, 0.99)

            # Set thumb_bend
            q_desired[4] = thumb_open  # thumb_bend (0.9 max close)
        else:
            thumb_open = 1.0
            # Fallback: if no thumb_value, follow index finger (with same closing limit)
            if "gripper_value" in finger_data:
                q_desired[4] = index_finger_open
            else:
                q_desired[4] = 1.0  # Default to open

        # Control thumb rotation with buttons
        # Right hand: B (+), A (-)
        # Left hand: Y (+), X (-)
        thumb_rot_plus = finger_data.get("thumb_rot_plus", False)
        thumb_rot_minus = finger_data.get("thumb_rot_minus", False)

        if thumb_rot_plus:
            self._current_thumb_rotation += self.thumb_rotation_increment
        if thumb_rot_minus:
            self._current_thumb_rotation -= self.thumb_rotation_increment

        # Clamp thumb rotation to valid range
        self._current_thumb_rotation = np.clip(self._current_thumb_rotation, 0.0, 1.0)
        q_desired[5] = self._current_thumb_rotation  # thumb_rotation

        return q_desired