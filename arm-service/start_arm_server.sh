#!/bin/bash
#
# Arm Server Startup Script
#
# Runs the XML-RPC arm server directly (no Docker) using the service's venv.
# Talks to the YB-SD15M serial bus-servo arm over /dev/ttyTHS1.
#

set -u

LOG_FILE="/tmp/arm_server.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
SERIAL_PORT="${ARM_SERIAL_PORT:-/dev/ttyTHS1}"

# Fall back to the system python if the venv is missing.
if [ ! -x "$VENV_PYTHON" ]; then
    VENV_PYTHON="$(command -v python3)"
fi

ARM_PID=""

cleanup() {
    echo "$(date): Received shutdown signal, stopping arm server..." | tee -a "$LOG_FILE"
    if [ -n "$ARM_PID" ]; then
        kill -TERM "$ARM_PID" 2>/dev/null || true
        wait "$ARM_PID" 2>/dev/null || true
    fi
    echo "$(date): Cleanup complete." | tee -a "$LOG_FILE"
    exit 0
}

trap cleanup SIGTERM SIGINT SIGHUP

echo "$(date): Starting Arm Server..." | tee -a "$LOG_FILE"
echo "$(date): Python: $VENV_PYTHON  Serial: $SERIAL_PORT" | tee -a "$LOG_FILE"

cd "$SCRIPT_DIR"

# Run in the background so the trap can forward SIGTERM for graceful shutdown.
"$VENV_PYTHON" arm_server.py --serial-port "$SERIAL_PORT" 2>&1 | tee -a "$LOG_FILE" &
ARM_PID=$!

wait "$ARM_PID"
EXIT_CODE=$?

echo "$(date): Arm server exited with code $EXIT_CODE" | tee -a "$LOG_FILE"
exit $EXIT_CODE
