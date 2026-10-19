#!/usr/bin/env python3
# coding: utf-8
"""
Service Supervisor XML-RPC Server

Arbitrates the two GPU-contended services on this Jetson: NanoOWL (`nano_owl`)
and Live-VLM (`live_vlm`). Both bind port 8000 and both need the single GPU, so
only one may run at a time. This supervisor lets a remote host switch between
them on demand.

Listens on 0.0.0.0:8002 by default. Controls the existing systemd units via
`sudo systemctl` (passwordless sudo assumed, as on this box).

Key behavior:
  - switch_to(name, callback_url=None) is ASYNCHRONOUS. It kicks off a background
    switch and returns immediately with {accepted: True}. This matters because
    the model load is memory-heavy: the caller can disconnect (freeing memory)
    and poll get_status() later, or register a callback_url to be pushed the
    outcome.
  - A service is only declared "up" once its READY MARKER appears in its log
    (NOT merely when the unit is active or port 8000 pings). Markers:
        owl -> "[NanoOwl] Predictor loaded."
        vlm -> "[NanoVlm] Model loaded"
    Waiting for the marker fixes the old false-success where VLM reported up and
    then OOM'd during model load.
  - The model is loaded exactly ONCE per switch (like the old start scripts).
    It is NOT retried in place: repeated back-to-back model loads stack current
    spikes and brown out the 25W Jetson rail, latching the board OFF.
  - THE SUPERVISOR NEVER REBOOTS THE BOARD. A failed load ends at phase
    'needs_reboot' with the box still up and the reason reported; rebooting is
    a person's decision, taken with the robot's power source in mind. Only the
    reboot() RPC reboots, and only because someone asked it to.
  - A load that has not reached its marker in time ends at phase 'timeout' and
    the unit is LEFT RUNNING, because a timeout means "not seen yet", not
    "failed" -- the marker has twice arrived seconds after the old code gave up.
    get_status() keeps scanning the log, and moves the phase on to 'up' or
    'needs_reboot' once the load finishes either way.
  - If callback_url is given, the supervisor pushes progress events
    (attempt / up / timeout / needs_reboot) to that URL's switch_progress(event)
    XML-RPC method. Callbacks are best-effort and never block the switch.
  - The requested service is persisted to a state file and restored on boot.

The arm service (port 8001) is independent and is NOT managed here.
"""

import os
import re
import signal
import subprocess
import sys
import threading
import time
import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler

# --- Configuration ----------------------------------------------------------

HOST = "0.0.0.0"
PORT = 8002

# Logical name -> systemd unit name.
GPU_SERVICES = {"owl": "nano_owl", "vlm": "live_vlm"}

# Logical name -> the log file its start script writes (scanned for markers).
LOG_FILES = {
    "owl": "/tmp/nano_owl_server.log",
    "vlm": "/tmp/live_vlm_server.log",
}

# Definitive "the model is loaded and serving" marker printed by each service.
# A service is only "up" once this text appears in its log for the current start.
READY_MARKERS = {
    "owl": "[NanoOwl] Predictor loaded.",
    "vlm": "[NanoVlm] Model loaded",
}

# Service both units bind; used only for the informational app_port_ready flag.
APP_PORT = 8000

# Load-failure policy: a model load is the single biggest power-draw event on
# the board. Do NOT retry a failed load in-place -- repeated back-to-back loads
# stack current spikes and brown out the 25W Jetson rail, latching the board
# OFF (a power-down, not a reboot). So: load exactly ONCE, and if it fails,
# STOP AND SAY SO.
#
# NOTHING HERE REBOOTS THE BOARD BY ITSELF, and that is a hard rule. It used to:
# a failed load rebooted, boot-restore brought the same service back, and it
# failed the same way. Worse, an automatic reboot does not reliably return --
# 2026-09-10 22:01:57 ordered one and the board stayed dark for 18 hours until
# wall power was reconnected by hand. A robot that strands itself in the field
# is a far worse outcome than a service that is down and says it is down.
GPU_SETTLE = 8.0  # after stopping the other unit, before starting the target

# Persisted "what the host last asked for", restored on boot.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, "desired_service.txt")
DEFAULT_SERVICE = "owl"

# Same CUDA/OOM patterns the start scripts watch for. An allocation failure
# surfaces as the cuDNN/cuBLAS handle it could not create, not as the ENOMEM
# underneath.
ERROR_PATTERNS = re.compile(
    r"CUDA out of memory"
    r"|CUDA: out of memory"
    r"|OutOfMemoryError"
    r"|NVML_SUCCESS.*INTERNAL ASSERT FAILED"
    r"|RuntimeError.*CUDACachingAllocator"
    r"|cuda runtime error"
    r"|CUDA error"
    r"|cuDNN error"
    r"|CUDNN_STATUS_INTERNAL_ERROR"
    r"|CUDNN_STATUS_NOT_INITIALIZED"
    r"|CUDNN_STATUS_ALLOC_FAILED"
    r"|CUBLAS_STATUS_ALLOC_FAILED"
    r"|CUBLAS_STATUS_NOT_INITIALIZED"
)

# How long to wait for a unit to stop. The units now SIGINT the container and
# then kill it rather than asking docker to stop it politely, so a stop is a
# couple of seconds; this only has to outlast a container that ignores both.
STOP_TIMEOUT = 10.0

# How long to wait for the READY MARKER, per service. NOT one number: OWL and
# VLM load different models and the gap between them is most of a minute.
#
# 240s was one global and it was a hair short for OWL, which is how the reboot
# loop started. Twice the marker landed just past the deadline and a healthy
# load was declared failed:
#
#     2026-09-10 22:01:57  gave up      22:01:58  [NanoOwl] Predictor loaded.
#     2026-09-11 16:41:50  gave up      16:41:54  [NanoOwl] Predictor loaded.
#
# So OWL's real load is AT LEAST 240s and these are lower bounds, not the
# measurement -- the load was never allowed to finish and be timed. Until it
# is, leave generous headroom. Nothing is lost by waiting: a slow load that
# succeeds costs seconds, and calling it failed costs a service that is down.
START_TIMEOUT = {"owl": 420.0, "vlm": 240.0}
DEFAULT_START_TIMEOUT = 240.0

# Global references for the signal handler.
_server = None
_service = None


def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global _server
    sig_name = signal.Signals(signum).name
    print(f"\nReceived {sig_name}. Shutting down supervisor...", flush=True)
    if _server is not None:
        # shutdown() blocks until serve_forever returns, and serve_forever is
        # this very thread -- so ask from another one and do not join it, or
        # every stop waits out the join.
        threading.Thread(target=_server.shutdown).start()
    sys.exit(0)


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ("/RPC2", "/")


class SupervisorService:
    """
    XML-RPC service that switches between the NanoOWL and Live-VLM systemd units.

    Switching runs in a background worker thread so switch_to() returns
    immediately. A new switch_to() preempts any in-progress switch.
    """

    def __init__(self):
        self._lock = threading.Lock()       # guards self._switch / self._last_error
        self._launch_lock = threading.Lock()  # serializes starting workers
        self._worker = None                 # current switch worker thread
        self._cancel = threading.Event()    # set to preempt the current worker
        self._last_error = None             # dict from the most recent failure
        self._switch = {
            "desired": None, "phase": "idle", "attempt": 0,
            "error": None, "detail": None,
            "needs_reboot": False, "updated": time.time(),
        }

    # --- systemctl / log helpers -------------------------------------------

    def _systemctl(self, action, unit):
        """Run `sudo systemctl <action> <unit>`; return (rc, output)."""
        try:
            r = subprocess.run(
                ["sudo", "systemctl", action, unit],
                capture_output=True, text=True, timeout=60,
            )
            return r.returncode, (r.stdout + r.stderr).strip()
        except subprocess.TimeoutExpired:
            return 1, f"systemctl {action} {unit} timed out"

    def _is_active(self, unit):
        """Raw `systemctl is-active` state (active/inactive/failed/...)."""
        r = subprocess.run(
            ["systemctl", "is-active", unit], capture_output=True, text=True,
        )
        return r.stdout.strip() or "unknown"

    def _wait_inactive(self, unit, timeout):
        """Wait until `unit` is no longer active. Return True if it stopped."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._is_active(unit) != "active":
                return True
            time.sleep(1.0)
        return self._is_active(unit) != "active"

    def _log_offset(self, name):
        """Current byte size of the target's log, so we only scan new lines."""
        try:
            return os.path.getsize(LOG_FILES[name])
        except OSError:
            return 0

    def _scan_log(self, name, from_offset):
        """Scan the target's log from `from_offset`. Return
        (error_line_or_None, ready_line_or_None) for the current start."""
        error_hit = None
        ready_hit = None
        marker = READY_MARKERS[name]
        try:
            with open(LOG_FILES[name], "r", errors="replace") as f:
                f.seek(from_offset)
                for line in f:
                    if error_hit is None and ERROR_PATTERNS.search(line):
                        error_hit = line.strip()
                    if ready_hit is None and marker in line:
                        ready_hit = line.strip()
        except OSError:
            pass
        return error_hit, ready_hit

    def _port_ready(self):
        """True if something answers ping() on the app port (informational)."""
        try:
            proxy = xmlrpc.client.ServerProxy(
                f"http://127.0.0.1:{APP_PORT}/", allow_none=True
            )
            proxy.ping()
            return True
        except Exception:
            return False

    def _current_locked(self):
        for name, unit in GPU_SERVICES.items():
            if self._is_active(unit) == "active":
                return name
        return "none"

    def _read_desired(self):
        try:
            with open(STATE_FILE) as f:
                val = f.read().strip()
            if val in GPU_SERVICES or val == "none":
                return val
        except OSError:
            pass
        return DEFAULT_SERVICE

    def _write_desired(self, name):
        try:
            with open(STATE_FILE, "w") as f:
                f.write(name)
        except OSError as e:
            print(f"Warning: could not persist desired service: {e}", flush=True)

    # --- state + callback ---------------------------------------------------

    def _set_state(self, is_error=False, **fields):
        """Update the switch-state snapshot; optionally mark it as last_error."""
        with self._lock:
            self._switch.update(fields)
            self._switch["updated"] = time.time()
            if is_error:
                self._last_error = {
                    "service": self._switch.get("desired"),
                    "error": self._switch.get("error"),
                    "detail": self._switch.get("detail"),
                    "needs_reboot": self._switch.get("needs_reboot"),
                    "when": time.time(),
                }

    def _emit(self, callback_url, event):
        """Best-effort push of a progress event to the client's callback URL.

        Runs in its own daemon thread with a short socket timeout so a slow or
        dead callback endpoint never blocks (or breaks) the switch."""
        print(f"[event] {event}", flush=True)
        if not callback_url:
            return

        def _send():
            try:
                tr = xmlrpc.client.Transport()
                proxy = xmlrpc.client.ServerProxy(
                    callback_url, allow_none=True, transport=tr,
                )
                # Give the connection a bounded timeout.
                try:
                    proxy._ServerProxy__transport.timeout = 5  # noqa (best-effort)
                except Exception:
                    pass
                proxy.switch_progress(event)
            except Exception as e:
                print(f"Warning: callback to {callback_url} failed: {e}", flush=True)

        threading.Thread(target=_send, daemon=True).start()

    # --- the background switch worker --------------------------------------

    def _stop_others(self, keep, cancel):
        """Stop every GPU unit except `keep` (None => stop all). Returns True
        unless a stop timed out."""
        ok = True
        for name, unit in GPU_SERVICES.items():
            if name == keep:
                continue
            if cancel.is_set():
                return ok
            if self._is_active(unit) == "active":
                print(f"Stopping {unit}...", flush=True)
                self._systemctl("stop", unit)
                if not self._wait_inactive(unit, STOP_TIMEOUT):
                    ok = False
        return ok

    def _check_load(self, name, unit, offset):
        """One look at a load in progress. Returns ('ready' | 'cuda_oom' |
        'start_failed', detail), or (None, None) while it is still loading."""
        error_hit, ready_hit = self._scan_log(name, offset)
        if error_hit:
            return "cuda_oom", error_hit
        if ready_hit:
            return "ready", ready_hit

        state = self._is_active(unit)
        if state in ("failed", "inactive"):
            # Unit exited before the marker; re-scan for a cause.
            error_hit, _ = self._scan_log(name, offset)
            if error_hit:
                return "cuda_oom", error_hit
            return "start_failed", f"{unit} entered state '{state}' before ready"
        return None, None

    def _wait_ready(self, name, unit, offset, cancel):
        """Wait for the READY MARKER for this start. Returns one of:
        'ready' | 'cuda_oom' | 'start_failed' | 'timeout' | 'cancelled',
        plus a detail string."""
        limit = START_TIMEOUT.get(name, DEFAULT_START_TIMEOUT)
        deadline = time.time() + limit
        while time.time() < deadline:
            if cancel.is_set():
                return "cancelled", "preempted by a newer switch"

            outcome, detail = self._check_load(name, unit, offset)
            if outcome:
                return outcome, detail

            time.sleep(1.5)

        # One more look before giving up. The loop sleeps 1.5s between scans and
        # exits on the deadline, so without this the last sleep's worth of log is
        # never read at all -- a marker that lands in it is missed and a service
        # that loaded fine is reported failed.
        outcome, detail = self._check_load(name, unit, offset)
        if outcome:
            return outcome, detail

        return "timeout", f"{unit}: ready marker not seen within {limit:.0f}s"

    def _recheck_timeout(self):
        """Settle a switch left at phase 'timeout' if its load has since
        finished. The unit was left running, so the marker (or a CUDA error)
        can land after the deadline; get_status() calls this so the phase
        catches up instead of saying 'timeout' forever."""
        with self._lock:
            sw = dict(self._switch)
        if sw.get("phase") != "timeout":
            return
        name = sw["desired"]
        outcome, detail = self._check_load(name, GPU_SERVICES[name],
                                           sw["log_offset"])
        if outcome is None:
            return

        with self._lock:
            # A newer switch_to() may have replaced this one while we scanned.
            if self._switch.get("updated") != sw["updated"]:
                return
        unit = GPU_SERVICES[name]
        if outcome == "ready":
            print(f"{unit} came up after its timeout.", flush=True)
            self._set_state(phase="up", error=None, detail=detail,
                            needs_reboot=False)
        else:
            print(f"{unit} failed after its timeout: {outcome} - {detail}. "
                  f"NOT rebooting.", flush=True)
            self._set_state(is_error=True, phase="needs_reboot", error=outcome,
                            detail=detail, needs_reboot=True)

    def _sleep_cancellable(self, seconds, cancel):
        """Sleep up to `seconds`, waking early if a newer switch preempts us.
        Returns True if it was cancelled."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            if cancel.is_set():
                return True
            time.sleep(0.5)
        return cancel.is_set()

    def _run_switch(self, name, callback_url, cancel):
        """Background worker: bring up `name` with one load attempt, reporting
        the outcome and emitting events. Never reboots."""
        try:
            if name == "none":
                self._stop_others(None, cancel)
                self._set_state(phase="idle", desired="none", attempt=0,
                                error=None, detail=None, needs_reboot=False)
                self._emit(callback_url, {"event": "up", "service": "none",
                                          "attempt": 0})
                return

            unit = GPU_SERVICES[name]

            if cancel.is_set():
                self._set_state(phase="cancelled")
                return

            self._set_state(phase="switching", attempt=1)
            self._emit(callback_url, {"event": "attempt", "service": name,
                                      "attempt": 1})

            # Free the GPU: stop the other service(s) first, then let the GPU/PSU
            # settle so this load's current spike doesn't overlap the previous
            # unit's release.
            self._stop_others(name, cancel)
            if cancel.is_set():
                self._set_state(phase="cancelled")
                return
            if self._sleep_cancellable(GPU_SETTLE, cancel):
                self._set_state(phase="cancelled")
                return

            # Single load attempt; watch only lines written from here on.
            offset = self._log_offset(name)
            print(f"Starting {unit}...", flush=True)
            self._systemctl("start", unit)

            outcome, detail = self._wait_ready(name, unit, offset, cancel)

            if outcome == "cancelled":
                self._set_state(phase="cancelled", detail=detail)
                return

            if outcome == "ready":
                self._set_state(phase="up", desired=name, error=None,
                                detail=detail, needs_reboot=False)
                self._emit(callback_url, {"event": "up", "service": name,
                                          "attempt": 1, "detail": detail})
                print(f"{unit} is UP.", flush=True)
                return

            # The load did not reach its marker. Do NOT retry in-place (stacked
            # load spikes brown out the board) and do NOT reboot -- see the
            # load-failure policy at the top of this file. Stop here and report.
            #
            # A TIMEOUT IS NOT A FAILURE. It means the marker was not seen yet,
            # and the load may well still be running: twice the marker arrived
            # seconds after the old code gave up and rebooted a healthy service.
            # So a timeout leaves the unit alone -- it may still come up, and
            # get_status() re-scans from log_offset and moves the phase on once
            # it does -- while a confirmed CUDA error is a real failure that the
            # board will not recover from on its own.
            if outcome == "timeout":
                print(f"{unit}: {detail}. Leaving it running -- it may still "
                      f"come up; poll get_status().", flush=True)
                self._set_state(is_error=True, phase="timeout", error=outcome,
                                detail=detail, needs_reboot=False,
                                log_offset=offset)
                self._emit(callback_url, {"event": "timeout", "service": name,
                                          "error": outcome, "detail": detail})
                return

            print(f"{unit} failed to load: {outcome} - {detail}. NOT rebooting; "
                  f"reboot by hand once you have looked.", flush=True)
            self._set_state(is_error=True, phase="needs_reboot", error=outcome,
                            detail=detail, needs_reboot=True)
            self._emit(callback_url, {"event": "needs_reboot", "service": name,
                                      "error": outcome, "detail": detail})
        except Exception as e:
            self._set_state(is_error=True, phase="failed", error="exception",
                            detail=str(e), needs_reboot=False)
            self._emit(callback_url, {"event": "failed", "service": name,
                                      "error": "exception", "detail": str(e)})

    # --- RPC methods --------------------------------------------------------

    def switch_to(self, name, callback_url=None):
        """
        Request a switch to 'owl', 'vlm', or 'none'. ASYNCHRONOUS.

        Returns immediately: {accepted: True, desired: name}
        or {accepted: False, error: 'invalid'} for a bad name.

        The switch loads the model ONCE and never retries it in place. It also
        never reboots the board: a CUDA failure ends at phase 'needs_reboot'
        with the box still up, for a person to look at and restart. A load that
        simply has not finished ends at phase 'timeout' with the unit LEFT
        RUNNING, because it may still come up.

        Poll get_status() for progress/outcome, or pass callback_url (an
        XML-RPC endpoint exposing switch_progress(event)) to be pushed events.
        """
        if name not in GPU_SERVICES and name != "none":
            return {"accepted": False, "error": "invalid",
                    "detail": f"unknown service '{name}'"}

        with self._launch_lock:
            # Preempt any in-flight switch and wait for it to notice.
            self._cancel.set()
            w = self._worker
            if w is not None and w.is_alive():
                w.join()
            self._cancel = threading.Event()
            cancel = self._cancel

            # Persist intent (restored on boot) and reset the switch snapshot.
            self._write_desired(name)
            self._set_state(desired=name, phase="switching", attempt=0,
                            error=None, detail=None, needs_reboot=False)
            with self._lock:
                self._last_error = None

            self._worker = threading.Thread(
                target=self._run_switch, args=(name, callback_url, cancel),
                daemon=True,
            )
            self._worker.start()

        return {"accepted": True, "desired": name}

    def current(self):
        """Return the currently active GPU service: 'owl', 'vlm', or 'none'."""
        with self._lock:
            return self._current_locked()

    def stop_all(self):
        """Stop both GPU units, freeing the GPU (async)."""
        return self.switch_to("none")

    def get_status(self):
        """Snapshot of the supervisor, the in-progress/last switch, and units."""
        self._recheck_timeout()
        with self._lock:
            sw = dict(self._switch)
            last_error = self._last_error
        return {
            "active": self.current(),
            "desired": self._read_desired(),
            "owl": self._is_active(GPU_SERVICES["owl"]),
            "vlm": self._is_active(GPU_SERVICES["vlm"]),
            "app_port_ready": self._port_ready(),
            "phase": sw.get("phase"),
            "attempt": sw.get("attempt"),
            "error": sw.get("error"),
            "detail": sw.get("detail"),
            "needs_reboot": sw.get("needs_reboot"),
            "last_error": last_error,
        }

    def ping(self):
        return "pong"

    def _reboot_now(self):
        """Reboot the Jetson. Reached ONLY from the reboot() RPC -- a person
        asking for it. No automatic path calls this; see the load-failure
        policy at the top of the file.
        Runs `sudo reboot` in a daemon thread so callers/RPC replies aren't
        blocked by the shutdown sequence."""
        print("Rebooting the Jetson...", flush=True)

        def do_reboot():
            time.sleep(1)
            subprocess.run(["sudo", "reboot"], check=False)

        threading.Thread(target=do_reboot, daemon=True).start()

    def reboot(self):
        """Reboot the Jetson on demand.

        CHECK WHAT THE ROBOT IS RUNNING ON FIRST. A reboot has been seen not to
        come back: 2026-09-10 the board went dark and stayed dark for 18 hours
        until wall power was reconnected. On wall power a reboot behaves; on
        battery it may end the session until someone walks over to it.

        Nothing calls this automatically -- that is the whole point of it being
        an RPC.
        """
        print("Reboot requested via XML-RPC.", flush=True)
        self._reboot_now()
        return ("Reboot initiated. If this board is on battery it may not come "
                "back without wall power.")


def _restore_on_boot(service):
    """Background: bring up whatever was last requested, without blocking RPC."""
    desired = service._read_desired()
    print(f"Boot restore: bringing up '{desired}'...", flush=True)
    service.switch_to(desired)


def main():
    global _server, _service

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    _server = SimpleXMLRPCServer(
        (HOST, PORT),
        requestHandler=RequestHandler,
        allow_none=True,
        logRequests=True,
    )
    _service = SupervisorService()
    _server.register_instance(_service)
    _server.register_introspection_functions()

    print("Service Supervisor XML-RPC Server", flush=True)
    print(f"Listening on {HOST}:{PORT}", flush=True)
    print(f"PID: {os.getpid()}", flush=True)
    print("Managed GPU services: " + ", ".join(
        f"{k} -> {v}" for k, v in GPU_SERVICES.items()), flush=True)
    print(f"Ready markers: {READY_MARKERS}", flush=True)
    print("Methods: switch_to(name, callback_url=None), current(), stop_all(), "
          "get_status(), ping(), reboot()", flush=True)
    print("Server ready. SIGTERM/SIGINT will trigger graceful shutdown.",
          flush=True)

    # Restore the last-requested service in the background so the RPC port comes
    # up immediately.
    threading.Thread(
        target=_restore_on_boot, args=(_service,), daemon=True
    ).start()

    try:
        _server.serve_forever()
    except Exception as e:
        print(f"Server error: {e}", flush=True)
    finally:
        print("Supervisor stopped.", flush=True)


if __name__ == "__main__":
    main()
