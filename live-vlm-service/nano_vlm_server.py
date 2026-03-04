#!/usr/bin/env python3
"""
NanoVlm XML-RPC Server

This module provides an XML-RPC server that allows remote clients to
create and control NanoVlm instances for vision-language model inference.

Server listens on 0.0.0.0:8000 by default.

Handles SIGTERM gracefully for systemctl stop/restart operations.

Supports enable/disable for low power mode - starts disabled by default.
Auto-disables after 20s of no get_output() calls.
"""

import os
import signal
import subprocess
import sys
import threading
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler

from nano_vlm import NanoVlm

# Global references for signal handler
_server = None
_service = None
_shutdown_event = threading.Event()

def signal_handler(signum, frame):
    """
    Handle SIGTERM and SIGINT signals for graceful shutdown.
    
    This allows systemctl stop to cleanly terminate the server.
    """
    global _server, _service, _shutdown_event
    
    sig_name = signal.Signals(signum).name
    print(f"\nReceived {sig_name} signal. Initiating graceful shutdown...", flush=True)
    
    # Set shutdown event to break out of serve_forever
    _shutdown_event.set()
    
    # Shutdown the service (NanoVlm instance and thread)
    if _service is not None:
        print("Shutting down NanoVlm service...", flush=True)
        _service._shutdown_internal()
    
    # Shutdown the XML-RPC server
    if _server is not None:
        print("Shutting down XML-RPC server...", flush=True)
        # Use shutdown in a separate thread to avoid deadlock
        shutdown_thread = threading.Thread(target=_server.shutdown)
        shutdown_thread.start()
        shutdown_thread.join(timeout=5.0)
    
    print("Server stopped.", flush=True)
    sys.exit(0)

class RequestHandler(SimpleXMLRPCRequestHandler):
    """Custom request handler with CORS-like path restrictions."""
    rpc_paths = ('/RPC2', '/')

class NanoVlmService:
    """
    XML-RPC service class that manages a NanoVlm instance.
    
    Provides methods to interact with the vision-language model processing.
    The instance is automatically created and started when the server runs,
    but starts in disabled (low power) mode. Call enable() to start processing.
    """
    
    def __init__(self):
        """Initialize the service with no active instance."""
        self.instance = None
        self.thread = None
        self._instance_lock = threading.Lock()
    
    def _create_instance(self, params=None):
        """
        Create a new NanoVlm instance with the specified parameters.
        Internal method - called automatically on startup.
        
        Args:
            params: Dictionary of parameters (see defaults below)
            
        Returns:
            True if instance was created successfully
        """
        if params is None:
            params = {}
        
        # Set defaults
        defaults = {
            'model': "Efficient-Large-Model/VILA1.5-3b",
            'api': "mlc",
            'quantization': None,
            'max_context_len': 256,
            'vision_model': None,
            'vision_scaling': None,
            'chat_template': None,
            'system_prompt': None,
            'video_input': "/dev/video0",
            'max_new_tokens': 32,
            'min_new_tokens': None,
            'do_sample': None,
            'repetition_penalty': None,
            'temperature': None,
            'top_p': None,
            'prompts': ["Describe the image concisely."],
            'output_stack_size': 25,
        }
        
        # Merge params with defaults
        config = {**defaults, **params}
        
        with self._instance_lock:
            # Shutdown existing instance if any
            if self.instance is not None:
                self._shutdown_internal()
            
            self.instance = NanoVlm(
                model=config['model'],
                api=config['api'],
                quantization=config['quantization'],
                max_context_len=config['max_context_len'],
                vision_model=config['vision_model'],
                vision_scaling=config['vision_scaling'],
                chat_template=config['chat_template'],
                system_prompt=config['system_prompt'],
                video_input=config['video_input'],
                max_new_tokens=config['max_new_tokens'],
                min_new_tokens=config['min_new_tokens'],
                do_sample=config['do_sample'],
                repetition_penalty=config['repetition_penalty'],
                temperature=config['temperature'],
                top_p=config['top_p'],
                prompts=config['prompts'],
                output_stack_size=config['output_stack_size'],
            )
            return True
    
    def _start_internal(self):
        """
        Start the NanoVlm processing thread.
        Internal method - called automatically on startup.
        
        Note: The NanoVlm instance starts in disabled mode.
        Call enable() to start actual frame processing.
        
        Returns:
            True if started successfully, False if no instance exists
        """
        with self._instance_lock:
            if self.instance is None:
                return False
            
            if self.thread is not None and self.thread.is_alive():
                return True  # Already running
            
            self.thread = threading.Thread(
                target=self.instance.startUp,
                daemon=True,
                name="NanoVlmThread"
            )
            self.thread.start()
            return True
    
    def _shutdown_internal(self):
        """Internal shutdown without lock (called from within locked context or signal handler)."""
        if self.instance is not None:
            self.instance.shutDown()
        
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=10.0)
            if self.thread.is_alive():
                print("Warning: Thread did not terminate within timeout", flush=True)
        
        self.thread = None
        return True
    
    def enable(self):
        """
        Enable frame processing (exit low power mode).
        
        When enabled, frames will be captured, embedded, and text generated.
        Also resets the auto-disable timer.
        
        Returns:
            True if enabled successfully, False if no instance exists
        """
        with self._instance_lock:
            if self.instance is None:
                return False
            self.instance.enable()
            return True
    
    def disable(self):
        """
        Disable frame processing (enter low power mode).
        
        When disabled, no frames are captured or processed, reducing power consumption.
        
        Returns:
            True if disabled successfully, False if no instance exists
        """
        with self._instance_lock:
            if self.instance is None:
                return False
            self.instance.disable()
            return True
    
    def is_enabled(self):
        """
        Check if frame processing is currently enabled.
        
        Returns:
            True if enabled, False if disabled or no instance
        """
        with self._instance_lock:
            if self.instance is None:
                return False
            return self.instance.is_enabled()
    
    def reboot(self):
        """
        Reboot the Jetson system.
        
        This will trigger a system reboot. The server will stop and
        the system will restart.
        
        Returns:
            String message (though the system will reboot before this returns)
        """
        print("Reboot requested via XML-RPC. Initiating system reboot...", flush=True)
        
        # Schedule reboot in a separate thread to allow response to be sent
        def do_reboot():
            import time
            time.sleep(1)  # Give time for XML-RPC response
            subprocess.run(['sudo', 'reboot'], check=False)
        
        reboot_thread = threading.Thread(target=do_reboot, daemon=True)
        reboot_thread.start()
        
        return "Reboot initiated. System will restart shortly."
    
    def is_running(self):
        """
        Check if the NanoVlm instance is currently running (thread active).
        
        Note: This checks if the main loop thread is running, not whether
        frame processing is enabled. Use is_enabled() to check that.
        
        Returns:
            True if running, False otherwise
        """
        with self._instance_lock:
            if self.instance is None:
                return False
            return self.instance.is_running()
    
    def set_prompts(self, prompts):
        """
        Update the prompts for the current instance.
        
        Args:
            prompts: List of prompt strings
            
        Returns:
            True if prompts were set, False if no instance exists
        """
        with self._instance_lock:
            if self.instance is None:
                return False
            self.instance.set_prompts(prompts)
            return True
    
    def get_prompts(self):
        """
        Get the current prompts.
        
        Returns:
            List of prompt strings, or empty list if no instance
        """
        with self._instance_lock:
            if self.instance is None:
                return []
            return self.instance.get_prompts()
    
    def get_output(self, index=0, timeout=30.0, prompt=None):
        """
        Get output at the specified index, optionally waiting for a specific prompt.
        
        Auto-enables processing if disabled, and waits for output to be available.
        Also resets the auto-disable timer (prevents auto-disable for 30s).
        
        Args:
            index: Stack index (0 = most recent)
            timeout: Max seconds to wait for output (default: 30.0)
            prompt: If provided, wait until entry at index has this prompt
            
        Returns:
            Dict with keys: 'prompt', 'output', and optionally 'sameAsNext'
            Returns {'prompt': None, 'output': None} if no instance or timeout
        """
        # Check instance exists (quick check with lock)
        with self._instance_lock:
            if self.instance is None:
                return {'prompt': None, 'output': None}
            instance = self.instance
        
        # Call get_output without holding the lock (it does its own waiting)
        result = instance.get_output(index, timeout, prompt)
        return result
    
    def get_output_stack_size(self):
        """
        Get the current number of items in the output stack.
        
        Returns:
            Number of items in stack, or 0 if no instance
        """
        with self._instance_lock:
            if self.instance is None:
                return 0
            return self.instance.get_output_stack_size()
    
    def clear_output_stack(self):
        """
        Clear all outputs from the stack.
        
        Returns:
            True if cleared, False if no instance
        """
        with self._instance_lock:
            if self.instance is None:
                return False
            self.instance.clear_output_stack()
            return True
    
    def ping(self):
        """
        Simple ping method to check if server is responsive.
        
        Returns:
            "pong"
        """
        return "pong"
    
    def get_status(self):
        """
        Get comprehensive status of the service.
        
        Returns:
            Dictionary with status information including:
            - has_instance: Whether NanoVlm instance exists
            - is_running: Whether the main loop thread is running
            - is_enabled: Whether frame processing is enabled
            - thread_alive: Whether the thread is alive
            - output_stack_size: Number of outputs in stack
            - prompts: Current prompt list
        """
        with self._instance_lock:
            has_instance = self.instance is not None
            is_running = has_instance and self.instance.is_running()
            is_enabled = has_instance and self.instance.is_enabled()
            thread_alive = self.thread is not None and self.thread.is_alive()
            stack_size = self.instance.get_output_stack_size() if has_instance else 0
            prompts = self.instance.get_prompts() if has_instance else []
            
            return {
                'has_instance': has_instance,
                'is_running': is_running,
                'is_enabled': is_enabled,
                'thread_alive': thread_alive,
                'output_stack_size': stack_size,
                'prompts': prompts,
            }

def main():
    """Main function to start the XML-RPC server."""
    global _server, _service
    
    host = "0.0.0.0"
    port = 8000
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    _server = SimpleXMLRPCServer(
        (host, port),
        requestHandler=RequestHandler,
        allow_none=True,
        logRequests=True
    )
    
    # Register the service instance
    _service = NanoVlmService()
    _server.register_instance(_service)
    
    # Also register introspection functions
    _server.register_introspection_functions()
    
    print(f"NanoVLM XML-RPC Server", flush=True)
    print(f"Listening on {host}:{port}", flush=True)
    print(f"PID: {os.getpid()}", flush=True)
    
    # Automatically create instance and start processing thread
    print("Creating NanoVlm instance with default parameters...", flush=True)
    _service._create_instance()
    print("Starting processing thread (in disabled/low-power mode)...", flush=True)
    _service._start_internal()
    print("Processing thread started. Call enable() to start frame processing.", flush=True)
    print(flush=True)
    
    print(f"Available methods:", flush=True)
    print(f"  - enable()               - Start frame processing (exit low power mode)", flush=True)
    print(f"  - disable()              - Stop frame processing (enter low power mode)", flush=True)
    print(f"  - is_enabled()           - Check if frame processing is enabled", flush=True)
    print(f"  - reboot()               - Reboot the system", flush=True)
    print(f"  - is_running()           - Check if main loop thread is running", flush=True)
    print(f"  - set_prompts(prompts)   - Update prompts", flush=True)
    print(f"  - get_prompts()          - Get current prompts", flush=True)
    print(f"  - get_output(index)      - Get output (also resets auto-disable timer)", flush=True)
    print(f"  - get_output_stack_size()- Get number of outputs in stack", flush=True)
    print(f"  - clear_output_stack()   - Clear output stack", flush=True)
    print(f"  - get_status()           - Get comprehensive status", flush=True)
    print(f"  - ping()                 - Check server responsiveness", flush=True)
    print(flush=True)
    print("Note: Auto-disables after 30s of no get_output() calls.", flush=True)
    print("Note: get_output() and set_prompts() auto-enable processing.", flush=True)
    print("Server ready. SIGTERM/SIGINT will trigger graceful shutdown.", flush=True)
    
    try:
        _server.serve_forever()
    except Exception as e:
        print(f"Server error: {e}", flush=True)
    finally:
        # Ensure cleanup if serve_forever exits unexpectedly
        if _service is not None:
            _service._shutdown_internal()
        print("Server stopped.", flush=True)

if __name__ == "__main__":
    main()
