"""RealSense camera sensor implementation."""

import time
from typing import Any, Dict, Optional

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
    print("pyrealsense2 not installed. RealSense camera will not be available.")

from gr00t_wbc.control.base.sensor import Sensor


class RealSenseSensor(Sensor):
    """Intel RealSense camera sensor."""

    def __init__(
        self,
        mount_position: str = "ego_view",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = False,
    ):
        """Initialize RealSense sensor.

        Args:
            mount_position: Camera mount position identifier
            width: Image width
            height: Image height
            fps: Frames per second
            enable_depth: Whether to enable depth stream
        """
        if rs is None:
            raise ImportError("pyrealsense2 is required for RealSense camera")

        self.mount_position = mount_position
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if enable_depth:
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # Start pipeline
        self.profile = self.pipeline.start(self.config)

        # Get device info
        device = self.profile.get_device()
        self.device_name = device.get_info(rs.camera_info.name)
        print(f"RealSense camera initialized: {self.device_name}")

        # Warm up camera
        for _ in range(30):
            self.pipeline.wait_for_frames()

    def read(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Read frame from RealSense camera.

        Returns:
            Dictionary with timestamps and images, or None if read failed
        """
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()

            if not color_frame:
                return None

            # Convert to numpy array (BGR format)
            color_image = np.asanyarray(color_frame.get_data())

            # Convert BGR to RGB
            color_image_rgb = color_image[:, :, ::-1].copy()

            timestamp = time.time()

            result = {
                "timestamps": {self.mount_position: timestamp},
                "images": {self.mount_position: color_image_rgb},
            }

            if self.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    result["depth"] = {self.mount_position: depth_image}

            return result

        except Exception as e:
            print(f"RealSense read error: {e}")
            return None

    def close(self):
        """Stop the RealSense pipeline."""
        try:
            self.pipeline.stop()
            print("RealSense camera closed")
        except Exception as e:
            print(f"Error closing RealSense: {e}")

    def observation_space(self):
        """Return the observation space."""
        import gymnasium as gym

        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8,
        )

