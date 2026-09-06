#!/bin/bash
#
# DOFBOT Arm Server startup script.
#
# Sources ROS and the colcon workspace, then runs the XML-RPC server. Sourcing
# here rather than inside the server means every `ros2` command the server
# spawns inherits the environment directly -- the server's own bash fallback
# never has to fire, and each command starts half a second sooner.
#
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This service lives outside the workspace it drives, so there is nothing
# above it to derive the path from -- DOFBOT_WS is the way to move it.
WORKSPACE="${DOFBOT_WS:-$HOME/ros2_ws}"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
WS_SETUP="$WORKSPACE/install/setup.bash"
LOG_FILE="${DOFBOT_ARM_LOG:-/tmp/dofbot_arm_server.log}"
PORT="${DOFBOT_ARM_PORT:-8001}"

echo "$(date): Starting DOFBOT Arm Server..." | tee -a "$LOG_FILE"

# ROS's own setup scripts read unset variables, so -u has to come off around
# them or sourcing dies on AMENT_TRACE_SETUP_FILES.
set +u
for setup in "$ROS_SETUP" "$WS_SETUP"; do
    if [ ! -f "$setup" ]; then
        echo "$(date): missing $setup -- build the workspace first" | tee -a "$LOG_FILE"
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$setup"
done
set -u

echo "$(date): workspace=$WORKSPACE ROS_DISTRO=$ROS_DISTRO port=$PORT" | tee -a "$LOG_FILE"

SERVER_PID=""

cleanup() {
    echo "$(date): Received shutdown signal, stopping arm server..." | tee -a "$LOG_FILE"
    if [ -n "$SERVER_PID" ]; then
        # The server's own SIGTERM handler stops the ROS launch before it
        # exits; wait for it rather than killing the process group, or
        # move_group outlives us.
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo "$(date): Cleanup complete." | tee -a "$LOG_FILE"
    exit 0
}

trap cleanup SIGTERM SIGINT SIGHUP

cd "$SCRIPT_DIR"

# Backgrounded so the trap above can run while the server is up.
python3 -u arm_server.py --port "$PORT" --workspace "$WORKSPACE" 2>&1 | tee -a "$LOG_FILE" &
SERVER_PID=$!

wait "$SERVER_PID"
EXIT_CODE=$?

echo "$(date): Arm server exited with code $EXIT_CODE" | tee -a "$LOG_FILE"
exit $EXIT_CODE
