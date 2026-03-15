#!/usr/bin/env python3
"""
NanoOwl XML-RPC Server

Provides an XML-RPC server for controlling a NanoOwl instance
for open-vocabulary object detection using NanoOWL.

Server listens on 0.0.0.0:8000 by default.

Handles SIGTERM gracefully for systemctl stop/restart operations.

Supports enable/disable for low power mode - starts disabled by default.
Auto-disables after 30s of no get_detections() calls.

Frame source is set at startup: "camera" for local USB camera,
"network" for frames pushed by the client via push_frame().
"""

import os
import signal
import subprocess
import sys
import threading
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler

from nano_owl import NanoOwl

# Global references for signal handler
_server = None
_service = None
_shutdown_event = threading.Event()

def signal_handler(signum, frame):
    """Handle SIGTERM and SIGINT signals for graceful shutdown."""
    global _server, _service, _shutdown_event

    sig_name = signal.Signals(signum).name
    print(f"\nReceived {sig_name} signal. Initiating graceful shutdown...", flush=True)

    _shutdown_event.set()

    if _service is not None:
        print("Shutting down NanoOwl service...", flush=True)
        _service._shutdown_internal()

    if _server is not None:
        print("Shutting down XML-RPC server...", flush=True)
        shutdown_thread = threading.Thread(target=_server.shutdown)
        shutdown_thread.start()
        shutdown_thread.join(timeout=5.0)

    print("Server stopped.", flush=True)
    sys.exit(0)

class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2', '/')

class NanoOwlService:
    """
    XML-RPC service class that manages a NanoOwl instance.

    Starts in disabled (low power) mode. Call enable() to start processing.
    """

    # Default configuration
    DEFAULT_IMAGE_ENCODE_ENGINE = "/opt/nanoowl/data/owl_image_encoder_patch32.engine"
    DEFAULT_VIDEO_INPUT = 0
    DEFAULT_RESOLUTION = (640, 480)

    def __init__(self):
        self.instance = None
        self.thread = None
        self._instance_lock = threading.Lock()

    def _create_instance(self, params=None):
        """Create a NanoOwl instance. Called automatically on startup."""
        if params is None:
            params = {}

        defaults = {
            'image_encode_engine': self.DEFAULT_IMAGE_ENCODE_ENGINE,
            'frame_source': 'camera',
            'video_input': self.DEFAULT_VIDEO_INPUT,
            'resolution': self.DEFAULT_RESOLUTION,
        }
        config = {**defaults, **params}

        with self._instance_lock:
            if self.instance is not None:
                self._shutdown_internal()

            self.instance = NanoOwl(
                image_encode_engine=config['image_encode_engine'],
                frame_source=config['frame_source'],
                video_input=config['video_input'],
                resolution=config['resolution'],
            )
            return True

    def _start_internal(self):
        """Start the NanoOwl processing thread."""
        with self._instance_lock:
            if self.instance is None:
                return False
            if self.thread is not None and self.thread.is_alive():
                return True

            self.thread = threading.Thread(
                target=self.instance.start_up,
                daemon=True,
                name="NanoOwlThread"
            )
            self.thread.start()
            return True

    def _shutdown_internal(self):
        """Internal shutdown (called from within locked context or signal handler)."""
        if self.instance is not None:
            self.instance.shut_down()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=10.0)
            if self.thread.is_alive():
                print("Warning: Thread did not terminate within timeout", flush=True)

        self.thread = None
        return True

    def enable(self):
        """Enable detection processing."""
        with self._instance_lock:
            if self.instance is None:
                return False
            self.instance.enable()
            return True

    def disable(self):
        """Disable detection processing."""
        with self._instance_lock:
            if self.instance is None:
                return False
            self.instance.disable()
            return True

    def is_enabled(self):
        with self._instance_lock:
            if self.instance is None:
                return False
            return self.instance.is_enabled()

    def push_frame(self, frame_data, seq):
        """
        Push a JPEG-encoded frame for detection (network mode only).

        Args:
            frame_data: xmlrpc.client.Binary containing JPEG bytes.
            seq: Integer sequence number assigned by the client.

        Returns:
            True on success, False on error.
        """
        with self._instance_lock:
            if self.instance is None:
                return False
            return self.instance.push_frame(frame_data.data, int(seq))

    def reboot(self):
        """Reboot the Jetson system."""
        print("Reboot requested via XML-RPC. Initiating system reboot...", flush=True)

        def do_reboot():
            import time
            time.sleep(1)
            subprocess.run(['sudo', 'reboot'], check=False)

        reboot_thread = threading.Thread(target=do_reboot, daemon=True)
        reboot_thread.start()

        return "Reboot initiated. System will restart shortly."

    def is_running(self):
        with self._instance_lock:
            if self.instance is None:
                return False
            return self.instance.is_running()

    def set_prompt(self, prompt):
        """
        Set the NanoOWL tree prompt.

        Uses NanoOWL tree syntax, e.g.: "[a person [a face, a hand], a dog]"
        Auto-enables processing if currently disabled.

        Returns:
            True on success, False on error.
        """
        with self._instance_lock:
            if self.instance is None:
                return False
            return self.instance.set_prompt(prompt)

    def get_prompt(self):
        with self._instance_lock:
            if self.instance is None:
                return ""
            return self.instance.get_prompt()

    def get_detections(self):
        """
        Get the latest detections (non-blocking).

        Returns:
            Dict with detection results, or None if no detections available:
            {
                "prompt": str,
                "detections": [
                    {
                        "label": str,
                        "label_id": int,
                        "parent_label": str,
                        "parent_id": int,
                        "box": [x1, y1, x2, y2],
                        "scores": [float, ...],
                        "labels": [int, ...],
                    },
                    ...
                ],
                "inference_time": float,
                "frame_seq": int or None
            }
        """
        with self._instance_lock:
            if self.instance is None:
                return None
            return self.instance.get_detections()

    def ping(self):
        return "pong"

    def get_status(self):
        """Get comprehensive status of the service."""
        with self._instance_lock:
            has_instance = self.instance is not None
            is_running = has_instance and self.instance.is_running()
            is_enabled = has_instance and self.instance.is_enabled()
            thread_alive = self.thread is not None and self.thread.is_alive()
            prompt = self.instance.get_prompt() if has_instance else ""
            frame_source = self.instance.get_frame_source() if has_instance else ""

            return {
                'has_instance': has_instance,
                'is_running': is_running,
                'is_enabled': is_enabled,
                'thread_alive': thread_alive,
                'prompt': prompt,
                'frame_source': frame_source,
            }

def main():
    global _server, _service

    host = "0.0.0.0"
    port = 8000

    # Parse --frame-source argument
    frame_source = "network"
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--frame-source" and i < len(sys.argv) - 1:
            frame_source = sys.argv[i + 1]
        elif arg.startswith("--frame-source="):
            frame_source = arg.split("=", 1)[1]

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    _server = SimpleXMLRPCServer(
        (host, port),
        requestHandler=RequestHandler,
        allow_none=True,
        logRequests=True
    )

    _service = NanoOwlService()
    _server.register_instance(_service)
    _server.register_introspection_functions()

    print(f"NanoOWL XML-RPC Server", flush=True)
    print(f"Listening on {host}:{port}", flush=True)
    print(f"PID: {os.getpid()}", flush=True)
    print(f"Frame source: {frame_source}", flush=True)

    print("Creating NanoOwl instance...", flush=True)
    _service._create_instance({'frame_source': frame_source})
    print("Starting processing thread (in disabled/low-power mode)...", flush=True)
    _service._start_internal()
    print("Processing thread started. Call enable() to start detection.", flush=True)
    print(flush=True)

    print(f"Available methods:", flush=True)
    print(f"  - enable()               - Start detection (exit low power mode)", flush=True)
    print(f"  - disable()              - Stop detection (enter low power mode)", flush=True)
    print(f"  - is_enabled()           - Check if detection is enabled", flush=True)
    print(f"  - push_frame(data, seq)  - Push a JPEG frame (network mode)", flush=True)
    print(f"  - reboot()               - Reboot the system", flush=True)
    print(f"  - is_running()           - Check if processing thread is running", flush=True)
    print(f"  - set_prompt(prompt)     - Set NanoOWL tree prompt", flush=True)
    print(f"  - get_prompt()           - Get current prompt", flush=True)
    print(f"  - get_detections()       - Get latest detections", flush=True)
    print(f"  - get_status()           - Get comprehensive status", flush=True)
    print(f"  - ping()                 - Check server responsiveness", flush=True)
    print(flush=True)
    print("Note: Auto-disables after 30s of no get_detections() calls.", flush=True)
    print("Note: get_detections() and set_prompt() auto-enable processing.", flush=True)
    print("Server ready. SIGTERM/SIGINT will trigger graceful shutdown.", flush=True)

    try:
        _server.serve_forever()
    except Exception as e:
        print(f"Server error: {e}", flush=True)
    finally:
        if _service is not None:
            _service._shutdown_internal()
        print("Server stopped.", flush=True)

if __name__ == "__main__":
    main()
