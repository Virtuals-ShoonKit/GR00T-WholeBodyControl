import time

import gymnasium as gym
import numpy as np

from gr00t_wbc.control.base.env import Env
from gr00t_wbc.control.envs.g1.utils.command_sender import (
    HandCommandSender,
    InspireHandCommandSender,
    get_hand_command_sender,
)
from gr00t_wbc.control.envs.g1.utils.state_processor import HandStateProcessor


class G1ThreeFingerHand(Env):
    """G1 Dex3 three-finger hand controller."""
    
    def __init__(self, is_left: bool = True):
        super().__init__()
        self.is_left = is_left
        self.hand_state_processor = HandStateProcessor(is_left=self.is_left)
        self.hand_command_sender = HandCommandSender(is_left=self.is_left)
        self.hand_q_offset = np.zeros(7)

    def observe(self) -> dict[str, any]:
        hand_state = self.hand_state_processor._prepare_low_state()  # (1, 28)
        assert hand_state.shape == (1, 28)

        # Apply offset to the hand state
        hand_state[0, :7] = hand_state[0, :7] + self.hand_q_offset

        hand_q = hand_state[0, :7]
        hand_dq = hand_state[0, 7:14]
        hand_ddq = hand_state[0, 21:28]
        hand_tau_est = hand_state[0, 14:21]

        # Return the state for this specific hand (left or right)
        return {
            "hand_q": hand_q,
            "hand_dq": hand_dq,
            "hand_ddq": hand_ddq,
            "hand_tau_est": hand_tau_est,
        }

    def queue_action(self, action: dict[str, any]):
        # Apply offset to the hand target
        action["hand_q"] = action["hand_q"] - self.hand_q_offset

        # action should contain hand_q
        self.hand_command_sender.send_command(action["hand_q"])

    def observation_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {
                "hand_q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "hand_dq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "hand_ddq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "hand_tau_est": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
            }
        )

    def action_space(self) -> gym.Space:
        return gym.spaces.Dict({"hand_q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))})

    def calibrate_hand(self):
        hand_obs = self.observe()
        hand_q = hand_obs["hand_q"]

        hand_q_target = np.zeros_like(hand_q)
        hand_q_target[0] = hand_q[0]

        # joint limit
        hand_q0_upper_limit = np.deg2rad(60)  # lower limit is -60

        # move the figure counterclockwise until the limit
        while True:

            if hand_q_target[0] - hand_q[0] < np.deg2rad(60):
                hand_q_target[0] += np.deg2rad(10)
            else:
                self.hand_q_offset[0] = hand_q0_upper_limit - hand_q[0]
                break

            self.queue_action({"hand_q": hand_q_target})

            hand_obs = self.observe()
            hand_q = hand_obs["hand_q"]

            time.sleep(0.1)

        print("done calibration, q0 offset (deg):", np.rad2deg(self.hand_q_offset[0]))

        # done calibrating, set target to zero
        self.hand_q_target = np.zeros_like(hand_q)
        self.queue_action({"hand_q": self.hand_q_target})


class G1InspireHand(Env):
    """
    G1 Inspire hand controller.
    
    The Inspire hand has 6 motors per hand:
    - Joint order: [pinky, ring, middle, index, thumb_bend, thumb_rotation]
    - Values are in range [0, 1] where 0=close, 1=open
    
    Commands are sent via DDS to "rt/inspire/cmd" topic.
    
    NOTE: The robot model defines 7 DOF per hand (for Dex3 compatibility), so this
    class accepts 7 DOF input but only uses the first 6 for the actual Inspire hand.
    For observations, it returns 7 DOF with the 7th value padded to 0.
    """
    
    # Number of DOFs for actual Inspire hand hardware
    INSPIRE_DOF = 6
    # Number of DOFs in robot model (for Dex3 compatibility)
    ROBOT_MODEL_DOF = 7
    
    def __init__(self, is_left: bool = True):
        super().__init__()
        self.is_left = is_left
        self.hand_command_sender = InspireHandCommandSender(is_left=self.is_left)
        self.inspire_dof = self.INSPIRE_DOF
        self.robot_model_dof = self.ROBOT_MODEL_DOF
        
        # Note: Inspire hand state is received via the InspireStreamer if needed
        # For now, we don't have a state processor since the Inspire hand
        # uses different DDS topics for state (rt/inspire/state)
        self._last_hand_q = np.ones(self.inspire_dof) * 0.5  # Start at half open
    
    def observe(self) -> dict[str, any]:
        """
        Get the current hand state.
        
        Returns 7 DOF to match robot model expectations (Dex3 compatibility).
        The 7th value is always 0 (padding).
        
        Note: For real observation from Inspire hand, you would need to
        subscribe to "rt/inspire/state" topic. For now, returns last commanded position.
        """
        # Pad to 7 DOF to match robot model
        hand_q_padded = np.zeros(self.robot_model_dof)
        hand_q_padded[:self.inspire_dof] = self._last_hand_q
        
        return {
            "hand_q": hand_q_padded,
            "hand_dq": np.zeros(self.robot_model_dof),
            "hand_ddq": np.zeros(self.robot_model_dof),
            "hand_tau_est": np.zeros(self.robot_model_dof),
        }
    
    def queue_action(self, action: dict[str, any]):
        """
        Send command to the Inspire hand.
        
        Args:
            action: Dict with "hand_q" key containing array of motor positions.
                   Accepts either 6 DOF (direct Inspire) or 7 DOF (robot model compatible).
                   If 7 DOF, the 7th value is ignored (padding).
                   Values should be in [0, 1] where 0=close, 1=open.
        """
        hand_q_input = action["hand_q"]
        
        # Handle both 6 DOF and 7 DOF input
        # Robot model sends 7 DOF, but Inspire only needs first 6
        if len(hand_q_input) >= self.inspire_dof:
            hand_q = hand_q_input[:self.inspire_dof]
        else:
            # Fallback: pad with 0.5 (neutral) if too few values
            hand_q = np.ones(self.inspire_dof) * 0.5
            hand_q[:len(hand_q_input)] = hand_q_input
        
        # Ensure values are in valid range for Inspire hand
        hand_q = np.clip(hand_q, 0.01, 0.99)
        
        # Store last commanded position
        self._last_hand_q = hand_q.copy()
        
        # Send command via DDS
        self.hand_command_sender.send_command(hand_q)
    
    def observation_space(self) -> gym.Space:
        # Return 7 DOF space to match robot model
        return gym.spaces.Dict(
            {
                "hand_q": gym.spaces.Box(low=0.0, high=1.0, shape=(self.robot_model_dof,)),
                "hand_dq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model_dof,)),
                "hand_ddq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model_dof,)),
                "hand_tau_est": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model_dof,)),
            }
        )
    
    def action_space(self) -> gym.Space:
        # Accept 7 DOF to match robot model (only first 6 used)
        return gym.spaces.Dict(
            {"hand_q": gym.spaces.Box(low=0.0, high=1.0, shape=(self.robot_model_dof,))}
        )
    
    def calibrate_hand(self):
        """Calibrate the Inspire hand - opens the hand to a neutral position."""
        print(f"Calibrating {'left' if self.is_left else 'right'} Inspire hand...")
        # Open the hand to neutral position (use 7 DOF for compatibility)
        neutral_position = np.ones(self.robot_model_dof) * 0.5
        neutral_position[-1] = 0.0  # Padding
        self.queue_action({"hand_q": neutral_position})
        time.sleep(0.5)
        print(f"{'Left' if self.is_left else 'Right'} Inspire hand calibration complete.")


def get_hand_class(hand_type: str):
    """
    Factory function to get the appropriate hand class.
    
    Args:
        hand_type: "dex3" for three-finger hands, "inspire" for Inspire hands
    
    Returns:
        Hand class (G1ThreeFingerHand or G1InspireHand)
    """
    if hand_type == "inspire":
        return G1InspireHand
    else:  # Default to dex3
        return G1ThreeFingerHand
