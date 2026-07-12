#!/bin/bash
#
# Service Supervisor Startup Script
#
# Runs the XML-RPC supervisor directly (no Docker) using the system python.
# Switches the Jetson between the nano_owl and live_vlm systemd services.
#

set -u

LOG_FILE="/tmp/supervisor_server.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"

# Fall back to the system python if there is no venv (stdlib only, so fine).
if [ ! -x "$VENV_PYTHON" ]; then
    VENV_PYTHON="$(command -v python3)"
fi

SUP_PID=""

cleanup() {
    echo "$(date): Received shutdown signal, stopping supervisor..." | tee -a "$LOG_FILE"
    if [ -n "$SUP_PID" ]; then
        kill -TERM "$SUP_PID" 2>/dev/null || true
        wait "$SUP_PID" 2>/dev/null || true
    fi
    echo "$(date): Cleanup complete." | tee -a "$LOG_FILE"
    exit 0
}

trap cleanup SIGTERM SIGINT SIGHUP

echo "$(date): Starting Service Supervisor..." | tee -a "$LOG_FILE"
echo "$(date): Python: $VENV_PYTHON" | tee -a "$LOG_FILE"

cd "$SCRIPT_DIR"

# Run in the background so the trap can forward SIGTERM for graceful shutdown.
"$VENV_PYTHON" service_supervisor.py 2>&1 | tee -a "$LOG_FILE" &
SUP_PID=$!

wait "$SUP_PID"
EXIT_CODE=$?

echo "$(date): Supervisor exited with code $EXIT_CODE" | tee -a "$LOG_FILE"
exit $EXIT_CODE
