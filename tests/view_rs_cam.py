#!/usr/bin/env python3
"""Camera viewer client for Unitree G1 RealSense stream."""

import base64
import cv2
import msgpack
import numpy as np
import zmq

# === CHANGE THIS TO YOUR ROBOT'S IP ===
ROBOT_IP = "192.168.123.164"  # Replace with your Unitree's IP
PORT = 5555
# ======================================

def decode_image(image_b64: str) -> np.ndarray:
    """Decode base64 JPEG image."""
    color_data = base64.b64decode(image_b64)
    color_array = np.frombuffer(color_data, dtype=np.uint8)
    return cv2.imdecode(color_array, cv2.IMREAD_COLOR)

def main():
    print(f"Connecting to camera server at tcp://{ROBOT_IP}:{PORT}...")
    
    # Setup ZMQ subscriber
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.CONFLATE, True)  # Only keep latest message
    socket.setsockopt(zmq.RCVHWM, 3)
    socket.connect(f"tcp://{ROBOT_IP}:{PORT}")
    
    print("Connected! Press 'q' to quit.")
    
    try:
        while True:
            # Receive and unpack message
            packed = socket.recv()
            data = msgpack.unpackb(packed)
            
            # Decode and display each camera image
            images = data.get("images", {})
            for camera_name, image_b64 in images.items():
                image = decode_image(image_b64)
                
                # Fix color: RealSense sends RGB, OpenCV expects BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Add camera name overlay
                cv2.putText(image, camera_name, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow(f"Unitree Camera - {camera_name}", image)
            
            # Quit on 'q' key (click on window first for focus)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        socket.close()
        context.term()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()