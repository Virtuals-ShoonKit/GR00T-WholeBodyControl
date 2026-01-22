"""ZED camera sensor implementation."""

import time
from typing import Any, Dict, Optional

import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    sl = None
    print("pyzed not installed. ZED camera will not be available.")

from gr00t_wbc.control.base.sensor import Sensor


class ZEDSensor(Sensor):
    """Stereolabs ZED camera sensor."""

    def __init__(
        self,
        mount_position: str = "ego_view",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        enable_depth: bool = False,
        serial_number: Optional[int] = None,
    ):
        """Initialize ZED sensor.

        Args:
            mount_position: Camera mount position identifier
            width: Image width (default 1280 for ZED 720p)
            height: Image height (default 720 for ZED 720p)
            fps: Frames per second
            enable_depth: Whether to enable depth stream
            serial_number: Optional ZED camera serial number for multi-camera setups
        """
        if sl is None:
            raise ImportError("pyzed is required for ZED camera. Install ZED SDK first.")

        self.mount_position = mount_position
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth

        # Initialize ZED camera
        self.zed = sl.Camera()
        
        # Set initialization parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = self._get_resolution(width, height)
        init_params.camera_fps = fps
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA if enable_depth else sl.DEPTH_MODE.NONE
        init_params.coordinate_units = sl.UNIT.METER
        init_params.sdk_verbose = 0  # Disable verbose output
        
        # Set specific camera by serial number if provided
        if serial_number is not None:
            init_params.set_from_serial_number(serial_number)

        # Open the camera
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {status}")

        # Get camera info
        camera_info = self.zed.get_camera_information()
        self.device_name = f"ZED {camera_info.camera_model} (SN: {camera_info.serial_number})"
        print(f"ZED camera initialized: {self.device_name}")

        # Create image containers
        self.image = sl.Mat()
        self.depth = sl.Mat() if enable_depth else None

        # Set runtime parameters
        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.enable_fill_mode = True

        # Warm up camera
        for _ in range(30):
            self.zed.grab(self.runtime_params)

    def _get_resolution(self, width: int, height: int) -> sl.RESOLUTION:
        """Map width/height to ZED resolution enum."""
        # Common resolutions
        if width >= 2208 and height >= 1242:
            return sl.RESOLUTION.HD2K
        elif width >= 1920 and height >= 1080:
            return sl.RESOLUTION.HD1080
        elif width >= 1280 and height >= 720:
            return sl.RESOLUTION.HD720
        elif width >= 672 and height >= 376:
            return sl.RESOLUTION.VGA
        else:
            return sl.RESOLUTION.HD720  # Default

    def read(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Read frame from ZED camera.

        Returns:
            Dictionary with timestamps and images, or None if read failed
        """
        try:
            # Grab a new frame
            status = self.zed.grab(self.runtime_params)
            if status != sl.ERROR_CODE.SUCCESS:
                print(f"ZED grab error: {status}")
                return None

            # Retrieve left RGB image
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            
            # Get image data as numpy array (BGRA format)
            image_data = self.image.get_data()
            
            # Convert BGRA to RGB
            color_image_rgb = image_data[:, :, :3][:, :, ::-1].copy()

            timestamp = time.time()

            result = {
                "timestamps": {self.mount_position: timestamp},
                "images": {self.mount_position: color_image_rgb},
            }

            if self.enable_depth and self.depth is not None:
                self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
                depth_data = self.depth.get_data()
                result["depth"] = {self.mount_position: depth_data}

            return result

        except Exception as e:
            print(f"ZED read error: {e}")
            return None

    def close(self):
        """Close the ZED camera."""
        try:
            self.zed.close()
            print("ZED camera closed")
        except Exception as e:
            print(f"Error closing ZED: {e}")

    def observation_space(self):
        """Return the observation space."""
        import gymnasium as gym

        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8,
        )

