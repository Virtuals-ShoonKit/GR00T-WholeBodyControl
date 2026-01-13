import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro

# Use local standalone GR00T client (no gr00t package dependency needed)

from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC, STATE_TOPIC_NAME
from gr00t_wbc.control.sensor.composed_camera import ComposedCameraClientSensor
from gr00t_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber
from gr00t_wbc.control.utils.telemetry import Telemetry


INFERENCE_NODE_NAME = "InferencePolicy"


@dataclass
class InferenceConfig:
    """Configuration for GR00T inference policy loop."""

    # GR00T server configuration (runs OUTSIDE Docker on host)
    groot_server_host: str = "127.0.0.1"
    """Host address for GR00T inference server (use 127.0.0.1 with --network=host)"""

    groot_server_port: int = 5556
    """Port for GR00T inference server"""

    # Camera configuration (runs on G1 robot)
    camera_host: str = "192.168.123.164"
    """Host address for camera server (on G1 robot)"""

    camera_port: int = 5555
    """Port for camera server"""

    # Inference settings
    inference_frequency: int = 10
    """Frequency of inference loop (Hz)"""

    task_prompt: str = "pick up the apple and place it on the plate"
    """Language task prompt for GR00T"""

    # Robot configuration
    enable_waist: bool = True
    """Whether waist is enabled"""

    with_hands: bool = False
    """Whether hands are enabled"""

    # Action execution
    action_horizon: int = 8
    """Number of action steps to execute per inference"""

    # Debug
    verbose: bool = False
    """Print verbose timing info"""

    # Visualization
    enable_visualization: bool = False
    """Enable Rerun visualization of action trajectories and camera images"""

    rerun_port: int = 9876
    """Port for Rerun visualization (run 'rerun --port 9876' on host first)"""

    in_docker: bool = True
    """Whether running inside Docker (forwards viz data to host)"""

    dry_run: bool = False
    """Don't send actions to robot (test mode) - useful for verifying setup"""


def main(config: InferenceConfig):
    print("=" * 60)
    print("GR00T Inference Policy Loop")
    print("=" * 60)

    # Initialize ROS
    ros_manager = ROSManager(node_name=INFERENCE_NODE_NAME)

    # === 1. Connect to GR00T Policy Server ===
    from gr00t_wbc.control.main.inference.groot_client import Gr00tPolicyClient

    print(f"\n[1/3] Connecting to GR00T server at {config.groot_server_host}:{config.groot_server_port}")
    groot_client = Gr00tPolicyClient(
        host=config.groot_server_host,
        port=config.groot_server_port,
        timeout_ms=15000,
    )

    print("    Waiting for GR00T server...")
    retry_count = 0
    while not groot_client.ping():
        retry_count += 1
        if retry_count % 5 == 0:
            print(f"    Still waiting for GR00T server... (attempt {retry_count})")
            print(f"    Make sure GR00T server is running on host at port {config.groot_server_port}")
        time.sleep(1.0)
    print("    ✓ GR00T server connected!")

    modality_config = groot_client.get_modality_config()
    print(f"    Modality keys: {list(modality_config.keys())}")

    # === 2. Connect to Camera Server ===
    print(f"\n[2/3] Connecting to camera server at {config.camera_host}:{config.camera_port}")
    camera_client = ComposedCameraClientSensor(
        server_ip=config.camera_host,
        port=config.camera_port,
    )
    print("    ✓ Camera client initialized!")

    # === 3. Setup ROS Publisher and State Subscriber ===
    print("\n[3/3] Setting up ROS publisher and state subscriber")
    action_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    print(f"    ✓ Publishing to: {CONTROL_GOAL_TOPIC}")
    
    # Subscribe to robot state from control loop
    state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
    print(f"    ✓ Subscribing to: {STATE_TOPIC_NAME}")

    # Telemetry for timing
    telemetry = Telemetry(window_size=100)

    # === Setup Visualization (optional) ===
    viz = None
    if config.enable_visualization:
        from gr00t_wbc.data.viz.rerun_viz import RerunViz
        print("\n[Viz] Initializing Rerun visualization...")
        print(f"      Make sure 'rerun --port {config.rerun_port}' is running on host!")
        
        # Define what to visualize
        image_keys = ["ego_view", "head_view"]
        tensor_keys = [
            "left_arm",
            "right_arm",
            "waist",
            "navigate_cmd",
            "base_height",
        ]
        if config.with_hands:
            tensor_keys.extend(["left_hand", "right_hand"])
        
        viz = RerunViz(
            image_keys=image_keys,
            tensor_keys=tensor_keys,
            app_name="GR00T_Inference",
            window_size=10.0,
            port=config.rerun_port,
            in_docker=config.in_docker,
        )
        print("      ✓ Rerun visualization ready!")

    print(f"\n{'=' * 60}")
    print(f"Starting inference loop at {config.inference_frequency} Hz")
    print(f"Task: '{config.task_prompt}'")
    print(f"Action horizon: {config.action_horizon}")
    print(f"Visualization: {'enabled' if config.enable_visualization else 'disabled'}")
    print(f"Dry run: {'enabled (NOT sending actions)' if config.dry_run else 'disabled'}")
    print(f"{'=' * 60}")
    print("\nPress 'l' in control loop to start receiving actions")
    print("Press Ctrl+C to stop\n")

    # Action buffer for action chunking
    action_buffer = []
    action_buffer_idx = 0
    loop_count = 0

    # Robot state - will be updated from control loop via ROS
    robot_state = get_default_robot_state()
    state_received = False

    try:
        while ros_manager.ok():
            t_start = time.monotonic()

            with telemetry.timer("total_loop"):
                # Check if we need new actions from the policy
                if len(action_buffer) == 0 or action_buffer_idx >= len(action_buffer):
                    # 1. Get camera images
                    with telemetry.timer("get_camera"):
                        camera_data = camera_client.read()
                        if camera_data is None:
                            if config.verbose:
                                print("Warning: No camera data received")
                            time.sleep(0.05)
                            continue

                    # 1.5. Get robot state from control loop
                    with telemetry.timer("get_state"):
                        state_msg = state_subscriber.get_msg()
                        if state_msg is not None:
                            robot_state = update_robot_state_from_msg(robot_state, state_msg)
                            if not state_received:
                                print("    ✓ Receiving robot state from control loop")
                                state_received = True

                    # 2. Format observation for GR00T
                    with telemetry.timer("format_observation"):
                        observation = format_observation_for_groot(
                            camera_data=camera_data,
                            robot_state=robot_state,
                            task_prompt=config.task_prompt,
                            modality_config=modality_config,
                        )

                    # 3. Get action from GR00T
                    with telemetry.timer("get_action"):
                        action, info = groot_client.get_action(observation)

                    # 4. Buffer actions for execution
                    action_buffer = extract_action_chunk(action, config.action_horizon, config.enable_waist, config.with_hands)
                    action_buffer_idx = 0

                    # 5. Visualize (if enabled)
                    if viz is not None:
                        with telemetry.timer("visualize"):
                            visualize_inference(
                                viz=viz,
                                camera_data=camera_data,
                                action=action,
                                config=config,
                                timestamp=t_start,
                            )

                    loop_count += 1
                    if loop_count % 10 == 0:
                        fps = 10.0 / (time.monotonic() - t_start) if loop_count == 10 else camera_client.fps()
                        print(f"  Inference #{loop_count}, buffer size: {len(action_buffer)}, cam FPS: {fps:.1f}")

                # 5. Execute one action from buffer
                with telemetry.timer("publish_action"):
                    if action_buffer_idx < len(action_buffer):
                        wbc_command = action_buffer[action_buffer_idx]

                        # Add timing info (same as teleop)
                        t_now = time.monotonic()
                        wbc_command["timestamp"] = t_now
                        wbc_command["target_time"] = t_now + (1 / config.inference_frequency)

                        if not config.dry_run:
                            action_publisher.publish(wbc_command)
                        elif config.verbose:
                            print(f"  [DRY RUN] Would publish action {action_buffer_idx}")
                        action_buffer_idx += 1

            # Check timing
            elapsed = time.monotonic() - t_start
            if elapsed > (1 / config.inference_frequency) and config.verbose:
                telemetry.log_timing_info(context="Inference Loop Missed", threshold=0.001)

            # Sleep to maintain frequency
            sleep_time = (1.0 / config.inference_frequency) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except ros_manager.exceptions() as e:
        print(f"\nInference loop interrupted: {e}")
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        print("Cleaning up inference loop...")
        camera_client.close()
        if viz is not None:
            viz.close()
        ros_manager.shutdown()
        print("Done.")


def get_default_robot_state() -> dict:
    """Get default robot state for observation formatting.
    
    Note: The actual robot state comes from the control loop.
    This is just for formatting the observation structure.
    In real deployment, you may want to subscribe to robot state from control loop.
    
    Dimensions for G1 (from GR00T-N1.6-G1-PnPAppleToPlate model):
    - left_leg: 6 joints
    - right_leg: 6 joints
    - waist: 3 joints
    - left_arm: 7 joints
    - right_arm: 7 joints
    - left_hand: 7 joints (model expects 7)
    - right_hand: 7 joints (model expects 7)
    """
    return {
        "left_leg": np.zeros(6, dtype=np.float32),
        "right_leg": np.zeros(6, dtype=np.float32),
        "waist": np.zeros(3, dtype=np.float32),
        "left_arm": np.zeros(7, dtype=np.float32),
        "right_arm": np.zeros(7, dtype=np.float32),
        "left_hand": np.zeros(7, dtype=np.float32),  # Model expects 7 dimensions
        "right_hand": np.zeros(7, dtype=np.float32),  # Model expects 7 dimensions
    }


def update_robot_state_from_msg(robot_state: dict, msg: dict) -> dict:
    """Update robot state from control loop message.
    
    The control loop publishes observation on STATE_TOPIC_NAME with 'q' containing
    all joint positions in robot model order. We need to extract the relevant parts.
    
    G1 joint order in 'q' (43 total):
    - left_leg: indices 0-5 (6 joints)
    - right_leg: indices 6-11 (6 joints)  
    - waist: indices 12-14 (3 joints)
    - left_arm: indices 15-21 (7 joints)
    - left_hand: indices 22-28 (7 joints)
    - right_arm: indices 29-35 (7 joints)
    - right_hand: indices 36-42 (7 joints)
    """
    q = msg.get("q")
    if q is None:
        return robot_state
    
    q = np.asarray(q, dtype=np.float32)
    
    # Extract each joint group from the full q vector
    if len(q) >= 43:
        robot_state["left_leg"] = q[0:6].copy()
        robot_state["right_leg"] = q[6:12].copy()
        robot_state["waist"] = q[12:15].copy()
        robot_state["left_arm"] = q[15:22].copy()
        robot_state["left_hand"] = q[22:29].copy()
        robot_state["right_arm"] = q[29:36].copy()
        robot_state["right_hand"] = q[36:43].copy()
    
    return robot_state


def _get_modality_keys(config, default_keys):
    """Helper to extract modality_keys from either ModalityConfig object or dict."""
    if hasattr(config, "modality_keys"):
        # ModalityConfig object
        return config.modality_keys
    elif isinstance(config, dict) and "modality_keys" in config:
        # Plain dict (e.g., after ZMQ serialization)
        return config["modality_keys"]
    else:
        return default_keys


def format_observation_for_groot(
    camera_data: dict,
    robot_state: dict,
    task_prompt: str,
    modality_config: dict,
) -> dict:
    """Format observation for GR00T policy.

    GR00T expects:
    - video: {key: np.uint8 array (B=1, T=1, H, W, C=3)}
    - state: {key: np.float32 array (B=1, T=1, D)}
    - language: {key: [[str]]}
    """
    obs = {"video": {}, "state": {}, "language": {}}

    # === Video ===
    images = camera_data.get("images", {})
    video_config = modality_config.get("video", {})
    video_keys = _get_modality_keys(video_config, ["ego_view"])

    for key in video_keys:
        img = None
        if key in images:
            img = images[key]
        elif "ego_view" in images:
            img = images["ego_view"]
        elif "head" in images:
            img = images["head"]
        elif len(images) > 0:
            # Use first available image
            img = list(images.values())[0]

        if img is not None:
            # GR00T expects shape: (B=1, T=1, H, W, C=3), dtype=uint8
            obs["video"][key] = img[None, None, ...].astype(np.uint8)

    # === State ===
    state_config = modality_config.get("state", {})
    state_keys = _get_modality_keys(state_config, list(robot_state.keys()))

    for key in state_keys:
        if key in robot_state:
            state_val = robot_state[key].astype(np.float32)
            obs["state"][key] = state_val[None, None, :]  # (1, 1, D)

    # === Language ===
    language_config = modality_config.get("language", {})
    language_keys = _get_modality_keys(language_config, ["annotation.human.task_description"])

    for key in language_keys:
        obs["language"][key] = [[task_prompt]]

    return obs


def extract_action_chunk(action: dict, horizon: int, enable_waist: bool, with_hands: bool) -> list:
    """Extract action chunk from GR00T output and convert to WBC commands.

    GR00T outputs actions like:
    {
        "left_arm": np.array (B=1, T=30, D=7),
        "right_arm": np.array (B=1, T=30, D=7),
        "waist": np.array (B=1, T=30, D=3),
        "left_hand": np.array (B=1, T=30, D=7),
        "right_hand": np.array (B=1, T=30, D=7),
        "navigate_command": np.array (B=1, T=30, D=3),
        "base_height_command": np.array (B=1, T=30, D=1),
    }

    WBC control loop expects (same as teleop):
    {
        "target_upper_body_pose": np.array,  # concatenated upper body joints
        "navigate_cmd": np.array([vx, vy, omega]),
        "base_height_command": float,
        "timestamp": float,
        "target_time": float,
    }
    
    WBC always expects full upper_body group which includes hands:
    - arms (14) + waist (3) + hands (14) = 31 joints (when waist in upper body)
    - arms (14) + hands (14) = 28 joints (when waist NOT in upper body)
    
    When with_hands=False, we send zeros for hand joints but still include them
    to match the expected WBC dimension.
    """
    wbc_commands = []

    # Find the action horizon from the output
    action_horizon = horizon
    for key, value in action.items():
        if hasattr(value, 'shape') and len(value.shape) >= 2:
            action_horizon = min(horizon, value.shape[1])
            break

    # Create WBC command for each timestep
    for t in range(action_horizon):
        cmd = {}

        # Concatenate upper body joints in the order expected by WBC
        # Order: left_arm (7), right_arm (7), waist (3), left_hand (7), right_hand (7)
        # NOTE: WBC always expects hands in upper_body, so we must include them
        upper_body_parts = []

        for key in ["left_arm", "right_arm"]:
            if key in action:
                upper_body_parts.append(action[key][0, t])

        if enable_waist and "waist" in action:
            upper_body_parts.append(action["waist"][0, t])

        # WBC always expects hands - use actual values if with_hands, else zeros
        for key in ["left_hand", "right_hand"]:
            if with_hands and key in action:
                # Use actual hand actions from GR00T
                upper_body_parts.append(action[key][0, t])
            elif key in action:
                # Send zeros but maintain dimension (7 joints per hand)
                upper_body_parts.append(np.zeros(action[key].shape[-1], dtype=np.float32))
            else:
                # Fallback: 7 zeros per hand (G1 hand has 7 joints)
                upper_body_parts.append(np.zeros(7, dtype=np.float32))

        if upper_body_parts:
            cmd["target_upper_body_pose"] = np.concatenate(upper_body_parts).astype(np.float64)

        # Navigation command [vx, vy, omega]
        if "navigate_command" in action:
            cmd["navigate_cmd"] = action["navigate_command"][0, t].astype(np.float64)
        else:
            cmd["navigate_cmd"] = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Base height command
        if "base_height_command" in action:
            val = action["base_height_command"][0, t]
            cmd["base_height_command"] = float(val[0] if hasattr(val, '__len__') else val)
        else:
            cmd["base_height_command"] = 0.74  # Default standing height

        wbc_commands.append(cmd)

    return wbc_commands


def visualize_inference(
    viz,
    camera_data: dict,
    action: dict,
    config: InferenceConfig,
    timestamp: float,
) -> None:
    """Visualize camera images and action trajectories using Rerun.
    
    This plots:
    - Camera images (ego_view, head_view)
    - Action trajectories for each timestep in the horizon:
      - left_arm (7 joints)
      - right_arm (7 joints)
      - waist (3 joints)
      - navigate_cmd (vx, vy, omega)
      - base_height
      - hands (if enabled)
    """
    # === Plot Camera Images ===
    images = camera_data.get("images", {})
    images_to_plot = {}
    
    # Map camera keys to visualization keys
    key_mapping = {
        "ego_view": "ego_view",
        "egoview": "ego_view",
        "head": "head_view",
        "head_view": "head_view",
    }
    
    for cam_key, img in images.items():
        viz_key = key_mapping.get(cam_key, cam_key)
        if img is not None:
            images_to_plot[viz_key] = img
    
    if images_to_plot:
        viz.plot_images(images_to_plot, timestamp)
    
    # === Plot Action Trajectories ===
    # Get first timestep of each action for visualization
    # Action shape is (B=1, T, D)
    tensor_data = {}
    
    # Arm joints
    if "left_arm" in action:
        tensor_data["left_arm"] = action["left_arm"][0, 0]  # (7,)
    if "right_arm" in action:
        tensor_data["right_arm"] = action["right_arm"][0, 0]  # (7,)
    
    # Waist
    if "waist" in action and config.enable_waist:
        tensor_data["waist"] = action["waist"][0, 0]  # (3,)
    
    # Navigation command
    if "navigate_command" in action:
        tensor_data["navigate_cmd"] = action["navigate_command"][0, 0]  # (3,)
    
    # Base height
    if "base_height_command" in action:
        val = action["base_height_command"][0, 0]
        tensor_data["base_height"] = np.array([val[0] if hasattr(val, '__len__') else val])
    
    # Hands (if enabled)
    if config.with_hands:
        if "left_hand" in action:
            tensor_data["left_hand"] = action["left_hand"][0, 0]  # (7,)
        if "right_hand" in action:
            tensor_data["right_hand"] = action["right_hand"][0, 0]  # (7,)
    
    if tensor_data:
        viz.plot_tensors(tensor_data, timestamp)


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)

