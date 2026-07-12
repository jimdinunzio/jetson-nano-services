#!/usr/bin/env python3
# coding: utf-8
"""
Arm XML-RPC Server

Provides an XML-RPC server for controlling the YB-SD15M serial bus-servo arm
via Arm_Driver (pyserial, /dev/ttyTHS1 by default).

Server listens on 0.0.0.0:8001 by default.

Handles SIGTERM gracefully for systemctl stop/restart operations.

First-version commands:
  - wave()          - run a short wave gesture, return to rest
  - read_angles()   - return the current angles of the six servos
plus the usual ping()/get_status()/reboot() helpers.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler

from arm_driver import Arm_Driver

# Global references for the signal handler
_server = None
_service = None

# Servo 1-6 angles for each wave pose (from wave_arm.py).
REST = [88, 147, 0, 8, 2, 133]
RAISED = [88, 147, 11, 28, 2, 131]
WAVE_A = [126, 162, 11, 73, 43, 109]
WAVE_B = [43, 166, 11, 73, 2, 109]

# Gripper (servo 6) closed angle = 180
GRIPPER_CLOSED = 180

# Stow pose: arm folded with the gripper closed. Servos 1-5 captured from a
# hand-set pose (values fall outside 0-180 because the arm sits past the
# nominal range). Recapture after any recalibration of POS_MIN/POS_MAX.
STOWED = [-2, 186, -15, -13, 2, GRIPPER_CLOSED]

# Named full-arm poses, addressable by lowercase string (see move_to/list_poses).
POSES = {
    'rest': REST,
    'raised': RAISED,
    'stowed': STOWED,
}


def signal_handler(signum, frame):
    """Handle SIGTERM and SIGINT for graceful shutdown."""
    global _server, _service

    sig_name = signal.Signals(signum).name
    print(f"\nReceived {sig_name} signal. Initiating graceful shutdown...", flush=True)

    if _service is not None:
        _service.close()

    if _server is not None:
        print("Shutting down XML-RPC server...", flush=True)
        shutdown_thread = threading.Thread(target=_server.shutdown)
        shutdown_thread.start()
        shutdown_thread.join(timeout=5.0)

    print("Server stopped.", flush=True)
    sys.exit(0)


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2', '/')


class ArmService:
    """
    XML-RPC service class that manages a single Arm_Driver instance.

    Commands that move the arm are serialized with a lock so two clients can't
    drive the bus at the same time.
    """

    def __init__(self, port="/dev/ttyTHS1", baudrate=115200):
        self.arm = Arm_Driver(com=port, baudrate=baudrate)
        self.port = port
        self._lock = threading.Lock()

    def _move(self, joints, move_secs, sleep_secs=None):
        """Command all six servos to `joints` over `move_secs` seconds, then
        wait `sleep_secs` (defaults to `move_secs`) for the motion to settle."""
        if sleep_secs is None:
            sleep_secs = move_secs
        self.arm.Arm_serial_servo_write6_array(joints, int(move_secs * 1000))
        time.sleep(sleep_secs)

    def wave(self):
        """
        Run a short wave gesture (mirrors wave_arm.py): raise from rest, wave
        A->B->A, return to the raised pose, then back to rest. Blocks until the
        gesture cjjjjjjjjjompletes.

        Returns:
            True on success.
        """
        with self._lock:
            print("Wave requested.", flush=True)
            # Raise from rest.
            self._move(RAISED, 1.0)

            # One wave: A -> B -> back to A.
            self._move(WAVE_A, 0.5)
            self._move(WAVE_B, 0.5)
            self._move(WAVE_A, 0.5)

            # Return to the raised pose, then back to rest.
            self._move(RAISED, 1.0)
            self._move(REST, 1.0)
            return True

    def read_angles(self):
        """
        Read the current angle of each of the six servos.

        Returns:
            List of six values (degrees), or None for any servo that did not
            reply.
        """
        with self._lock:
            return [self.arm.Arm_serial_servo_read(sid) for sid in range(1, 7)]

    def move_servo(self, id, angle, time_ms=1000):
        """Move a single servo (id 1-6, 0 = all) to `angle` over `time_ms`."""
        with self._lock:
            self.arm.Arm_serial_servo_write(id, angle, time_ms)
            return True

    def move_servo_any(self, id, angle, time_ms=1000):
        """Move any bus servo (id 1-250, no per-joint inversion)."""
        with self._lock:
            self.arm.Arm_serial_servo_write_any(id, angle, time_ms)
            return True

    def move_all(self, s1, s2, s3, s4, s5, s6, time_ms=1000):
        """Move all six servos to the given angles over `time_ms`."""
        with self._lock:
            self.arm.Arm_serial_servo_write6(s1, s2, s3, s4, s5, s6, time_ms)
            return True

    def move_joints(self, joints, time_ms=1000):
        """Move all six servos from a 6-element [s1..s6] angle array."""
        with self._lock:
            self.arm.Arm_serial_servo_write6_array(list(joints), time_ms)
            return True

    def move_to(self, name, time_ms=1500):
        """Move to a named full-arm pose (case-insensitive): 'rest', 'raised',
        'stowed'. Returns True on success, False if the name is unknown."""
        key = str(name).strip().lower()
        pose = POSES.get(key)
        if pose is None:
            print("Unknown pose: %r (known: %s)" % (name, ", ".join(sorted(POSES))), flush=True)
            return False
        with self._lock:
            print("Move to pose '%s' requested." % key, flush=True)
            self.arm.Arm_serial_servo_write6_array(pose, time_ms)
            time.sleep(time_ms / 1000.0)
            return True

    def list_poses(self):
        """Return the named poses as {name: [s1..s6]}."""
        return {name: list(joints) for name, joints in POSES.items()}

    def read_servo(self, id):
        """Read one servo's angle (id 1-6). Returns None on no reply."""
        with self._lock:
            return self.arm.Arm_serial_servo_read(id)

    def read_servo_any(self, id):
        """Read any bus servo's angle (id 1-250, no inversion)."""
        with self._lock:
            return self.arm.Arm_serial_servo_read_any(id)

    def ping_servo(self, id):
        """Ping a servo by id. Returns 0xDA on success, None if unreachable."""
        with self._lock:
            return self.arm.Arm_ping_servo(id)

    def set_torque(self, onoff):
        """Torque switch for all six servos. 1: enabled, 0: disabled (free to move by hand)."""
        with self._lock:
            self.arm.Arm_serial_set_torque(onoff)
            return True

    def servo_control(self, id, num, time_ms=1000):
        """Move one servo to a raw position `num` (900-3100, servo 5: 900-4200)."""
        with self._lock:
            self.arm.bus_servo_control(id, num, time_ms)
            return True

    def servo_control_array(self, array, time_ms=1000):
        """Move all six servos from a 6-element raw-position array."""
        with self._lock:
            self.arm.bus_servo_control_array6(list(array), time_ms)
            return True

    def get_serial_port(self):
        """Return the serial device path the driver has open."""
        return self.arm.get_i2c_bus_num()

    def ping(self):
        return "pong"

    def get_status(self):
        """Get a basic status snapshot of the service."""
        return {
            'port': self.port,
            'has_arm': self.arm is not None,
        }

    def reboot(self):
        """Reboot the Jetson system."""
        print("Reboot requested via XML-RPC. Initiating system reboot...", flush=True)

        def do_reboot():
            time.sleep(1)
            subprocess.run(['sudo', 'reboot'], check=False)

        threading.Thread(target=do_reboot, daemon=True).start()
        return "Reboot initiated. System will restart shortly."

    def close(self):
        """Release the serial port."""
        with self._lock:
            try:
                self.arm.ser.close()
            except Exception:
                pass


def main():
    global _server, _service

    host = "0.0.0.0"
    port = 8001

    # Parse --serial-port argument (the serial device for the arm).
    serial_port = "/dev/ttyTHS1"
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--serial-port" and i < len(sys.argv) - 1:
            serial_port = sys.argv[i + 1]
        elif arg.startswith("--serial-port="):
            serial_port = arg.split("=", 1)[1]

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    _server = SimpleXMLRPCServer(
        (host, port),
        requestHandler=RequestHandler,
        allow_none=True,
        logRequests=True
    )

    print(f"Arm XML-RPC Server", flush=True)
    print(f"Listening on {host}:{port}", flush=True)
    print(f"PID: {os.getpid()}", flush=True)
    print(f"Serial port: {serial_port}", flush=True)

    print("Opening arm driver...", flush=True)
    _service = ArmService(port=serial_port)
    _server.register_instance(_service)
    _server.register_introspection_functions()
    print(flush=True)

    print(f"Available methods:", flush=True)
    print(f"  - wave()                              - Run a short wave gesture, return to rest", flush=True)
    print(f"  - read_angles()                       - Read the six servo angles", flush=True)
    print(f"  - move_servo(id, angle, time_ms)       - Move one servo (id 1-6, 0=all)", flush=True)
    print(f"  - move_servo_any(id, angle, time_ms)   - Move any bus servo (id 1-250)", flush=True)
    print(f"  - move_all(s1..s6, time_ms)            - Move all six servos", flush=True)
    print(f"  - move_joints(joints, time_ms)         - Move all six servos from an array", flush=True)
    print(f"  - move_to(name, time_ms)               - Move to a named pose: rest, raised, stowed", flush=True)
    print(f"  - list_poses()                         - List named poses and their angles", flush=True)
    print(f"  - read_servo(id)                       - Read one servo's angle (id 1-6)", flush=True)
    print(f"  - read_servo_any(id)                   - Read any bus servo's angle (id 1-250)", flush=True)
    print(f"  - ping_servo(id)                       - Ping a servo by id", flush=True)
    print(f"  - set_torque(onoff)                    - Enable/disable torque on all servos", flush=True)
    print(f"  - servo_control(id, num, time_ms)      - Move one servo to a raw position", flush=True)
    print(f"  - servo_control_array(array, time_ms)  - Move all six servos, raw positions", flush=True)
    print(f"  - get_serial_port()                    - Get the serial device path", flush=True)
    print(f"  - get_status()                         - Get service status", flush=True)
    print(f"  - reboot()                             - Reboot the system", flush=True)
    print(f"  - ping()                               - Check server responsiveness", flush=True)
    print(flush=True)
    print("Server ready. SIGTERM/SIGINT will trigger graceful shutdown.", flush=True)

    try:
        _server.serve_forever()
    except Exception as e:
        print(f"Server error: {e}", flush=True)
    finally:
        if _service is not None:
            _service.close()
        print("Server stopped.", flush=True)


if __name__ == "__main__":
    main()
