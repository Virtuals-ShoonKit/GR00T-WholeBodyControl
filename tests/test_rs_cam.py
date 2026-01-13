import sys
try:
    import pyrealsense2 as rs
    print("✓ pyrealsense2 imported successfully")
except ImportError:
    print("✗ pyrealsense2 not installed")
    sys.exit(1)

# List connected devices
ctx = rs.context()
devices = ctx.query_devices()
print(f"\nFound {len(devices)} RealSense device(s):")
for i, dev in enumerate(devices):
    print(f"  [{i}] {dev.get_info(rs.camera_info.name)} - S/N: {dev.get_info(rs.camera_info.serial_number)}")

if len(devices) == 0:
    print("✗ No RealSense devices found!")
    sys.exit(1)

# Try to start a simple pipeline
print("\nTrying to start camera pipeline...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    profile = pipeline.start(config)
    print("✓ Pipeline started successfully!")
    
    # Grab a few frames
    for i in range(5):
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        color_frame = frames.get_color_frame()
        if color_frame:
            print(f"  Frame {i+1}: {color_frame.get_width()}x{color_frame.get_height()}")
    
    pipeline.stop()
    print("✓ Camera test PASSED!")
    
except Exception as e:
    print(f"✗ Pipeline error: {e}")
    sys.exit(1)
