"""
Standalone GR00T Policy Client for WBC inference.

This is a self-contained client that communicates with the GR00T server
without requiring the full gr00t package to be installed.
"""

from dataclasses import dataclass
import io
from typing import Any

import msgpack
import numpy as np
import zmq


class MsgSerializer:
    """Serializer for ZMQ messages using msgpack with numpy support."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        # Decode ModalityConfig as plain dict
        if "__ModalityConfig_class__" in obj:
            return obj["as_json"]
        # Decode numpy arrays
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        # Encode numpy arrays
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class Gr00tPolicyClient:
    """
    Client for communicating with a GR00T Policy Server over ZMQ.
    
    This is a standalone implementation that doesn't require the gr00t package.
    
    Usage:
        client = Gr00tPolicyClient(host="127.0.0.1", port=5556)
        
        # Wait for server
        while not client.ping():
            time.sleep(1)
        
        # Get modality config
        modality_config = client.get_modality_config()
        
        # Get action from observation
        action, info = client.get_action(observation)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        """Check if server is available."""
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """Kill the server."""
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> Any:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure we are running the correct policy server.")
        response = MsgSerializer.from_bytes(message)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def __del__(self):
        """Cleanup resources on destruction"""
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'context'):
            self.context.term()

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Get action from the policy server.
        
        Args:
            observation: Dict with keys 'video', 'state', 'language'
            options: Optional parameters
            
        Returns:
            Tuple of (action_dict, info_dict)
        """
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)  # Convert list (from msgpack) to tuple of (action, info)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy."""
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self) -> dict[str, dict]:
        """
        Get modality configuration from the server.
        
        Returns:
            Dict mapping modality names ('video', 'state', 'action', 'language')
            to their configurations (as plain dicts).
        """
        return self.call_endpoint("get_modality_config", requires_input=False)

