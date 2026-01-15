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
"""

import numpy as np

from gr00t_wbc.control.teleop.solver.solver import Solver


# Output 7 DOF to match robot model (Dex3 compatibility), actual Inspire hand has 6 DOF
INSPIRE_ROBOT_MODEL_DOF = 7
INSPIRE_ACTUAL_DOF = 6


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
    
    The robot model defines 7 DOF per hand (for Dex3 compatibility), but Inspire
    hand only has 6 DOF. This solver outputs 7 values to match the robot model's
    joint indexing, which then gets mapped to 6 DOF in G1InspireHand.
    
    Output format (7 DOF for robot model compatibility):
    [pinky, ring, middle, index, thumb_bend, thumb_rotation, padding]
    
    Values are in range [0, 1] where 0=close, 1=open
    """
    
    # Interpolation speed: how fast to move toward target (0.0-1.0)
    # Lower = smoother/slower, Higher = faster/more responsive
    # 1.0 = instant (no smoothing), 0.1 = very smooth
    FINGER_INTERPOLATION_SPEED = 0.5  # Speed for 4 fingers
    THUMB_INTERPOLATION_SPEED = 0.25  # Slightly slower for thumb (more precise control)
    
    # Thumb rotation increment per update when button is held
    THUMB_ROTATION_INCREMENT = 0.05
    
    def __init__(self, side) -> None:
        self.side = "L" if side.lower() == "left" else "R"
        # Output 7 DOF to match robot model, actual Inspire hand has 6 DOF
        self.robot_model_dof = INSPIRE_ROBOT_MODEL_DOF
        self.actual_dof = INSPIRE_ACTUAL_DOF
        
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
                self.FINGER_INTERPOLATION_SPEED
            )
            
            # Clamp to valid range
            finger_open = np.clip(self._current_finger_position, 0.01, 0.99)

            if pinch_mode:
                # Pinch mode: keep pinky/ring/middle open, index follows trigger
                q_desired[0] = 1.0  # pinky open
                q_desired[1] = 1.0  # ring open
                q_desired[2] = 1.0  # middle open
                q_desired[3] = finger_open  # index
            else:
                # Grip mode: 4 fingers follow trigger
                q_desired[0] = finger_open  # pinky
                q_desired[1] = finger_open  # ring
                q_desired[2] = finger_open  # middle
                q_desired[3] = finger_open  # index
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
                self.THUMB_INTERPOLATION_SPEED
            )
            
            # Clamp to valid range
            thumb_open = np.clip(self._current_thumb_position, 0.01, 0.99)
            
            # Set thumb_bend
            q_desired[4] = thumb_open  # thumb_bend
        else:
            thumb_open = 1.0
            # Fallback: if no thumb_value, follow fingers
            q_desired[4] = finger_open
        
        # Control thumb rotation with buttons
        # Right hand: B (+), A (-)
        # Left hand: Y (+), X (-)
        thumb_rot_plus = finger_data.get("thumb_rot_plus", False)
        thumb_rot_minus = finger_data.get("thumb_rot_minus", False)
        
        if thumb_rot_plus:
            self._current_thumb_rotation += self.THUMB_ROTATION_INCREMENT
        if thumb_rot_minus:
            self._current_thumb_rotation -= self.THUMB_ROTATION_INCREMENT
        
        # Clamp thumb rotation to valid range
        self._current_thumb_rotation = np.clip(self._current_thumb_rotation, 0.0, 1.0)
        q_desired[5] = self._current_thumb_rotation  # thumb_rotation
        
        return q_desired
