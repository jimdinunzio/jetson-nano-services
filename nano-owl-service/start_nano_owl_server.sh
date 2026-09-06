#!/bin/bash
#
# NanoOWL Server Startup Script
# Monitors for CUDA/memory errors and reboots if detected
#

LOG_FILE="/tmp/nano_owl_server.log"
CONTAINER_NAME="nano_owl_server"
REBOOT_FLAG="/tmp/nano_owl_needs_reboot"
# A GPU/host allocation failure reaches the log as a cuDNN/cuBLAS init or
# "CUDA error" line: the driver reports the handle it could not create, not the
# ENOMEM underneath. NvMapMemAllocInternalTagged "error 12" lines are not
# listed -- they also appear on healthy loads and mid-run, so they are noise.
ERROR_PATTERNS="CUDA out of memory|CUDA: out of memory|OutOfMemoryError|NVML_SUCCESS.*INTERNAL ASSERT FAILED|RuntimeError.*CUDACachingAllocator|cuda runtime error|CUDA error|cuDNN error|CUDNN_STATUS_INTERNAL_ERROR|CUDNN_STATUS_NOT_INITIALIZED|CUDNN_STATUS_ALLOC_FAILED|CUBLAS_STATUS_ALLOC_FAILED|CUBLAS_STATUS_NOT_INITIALIZED"

# Remove any stale reboot flag
rm -f "$REBOOT_FLAG"

# Function to trigger immediate reboot
trigger_reboot() {
    echo "$(date): Triggering immediate system reboot..." | tee -a "$LOG_FILE"
    # Use nohup and disown to ensure reboot happens even if script is killed
    nohup sudo reboot &>/dev/null &
    disown
    exit 1
}

# Cleanup function to stop the container
cleanup() {
    echo "$(date): Received shutdown signal, stopping container..." | tee -a "$LOG_FILE"

    # Check if we need to reboot due to CUDA error
    if [ -f "$REBOOT_FLAG" ]; then
        echo "$(date): Reboot flag detected, initiating reboot..." | tee -a "$LOG_FILE"
        rm -f "$REBOOT_FLAG"
        trigger_reboot
    fi

    # Stop the Docker container
    docker stop -t 10 "$CONTAINER_NAME" 2>/dev/null || true

    # Kill the monitor process if running
    if [ -n "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi

    echo "$(date): Cleanup complete." | tee -a "$LOG_FILE"
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup SIGTERM SIGINT SIGHUP

# Stop any existing container with this name
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

echo "$(date): Starting NanoOWL Server..." | tee -a "$LOG_FILE"

# Start the error monitoring in the background
# When error is detected, it creates a flag file and stops the container
# which will cause the main process to exit and check for the flag
(
    sleep 5  # Wait for log to start populating
    tail -f "$LOG_FILE" 2>/dev/null | while read line; do
        if echo "$line" | grep -qE "$ERROR_PATTERNS"; then
            # The supervisor (service_supervisor.py) owns reboot decisions now,
            # so we only log the error and stop the container. The supervisor
            # scans this log, sees the error, and reports needs_reboot to the
            # remote host instead of rebooting the box mid-switch.
            echo "$(date): CUDA/Memory error detected! Stopping container..." | tee -a "$LOG_FILE"
            echo "$(date): Error line: $line" | tee -a "$LOG_FILE"

            # Stop the container to trigger exit.
            docker stop -t 5 "$CONTAINER_NAME" 2>/dev/null || true
            exit 0
        fi
    done
) &
MONITOR_PID=$!

# Run the container in foreground (not detached) with a fixed name
jetson-containers run \
  --no-tty \
  --name "$CONTAINER_NAME" \
  -e PIP_INDEX_URL=https://pypi.org/simple \
  --workdir /data/nano_owl \
  my_nano_owl \
  python3 nano_owl_server.py --frame-source network 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# Kill the monitor process
kill $MONITOR_PID 2>/dev/null || true

echo "$(date): Container exited with code $EXIT_CODE" | tee -a "$LOG_FILE"

# Clean up the container
docker rm "$CONTAINER_NAME" 2>/dev/null || true

# Reboot decisions are owned by the supervisor (service_supervisor.py), which
# scans this log for the error patterns and reports needs_reboot to the remote
# host. This script no longer reboots on its own.

exit $EXIT_CODE
