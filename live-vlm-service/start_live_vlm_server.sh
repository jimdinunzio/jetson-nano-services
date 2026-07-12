#!/bin/bash
#
# Live VLM Server Startup Script
# Monitors for CUDA/memory errors and reboots if detected
#

LOG_FILE="/tmp/live_vlm_server.log"
CONTAINER_NAME="live_vlm_server"
REBOOT_FLAG="/tmp/live_vlm_needs_reboot"
ERROR_PATTERNS="CUDA out of memory|CUDA: out of memory|OutOfMemoryError|NVML_SUCCESS.*INTERNAL ASSERT FAILED|RuntimeError.*CUDACachingAllocator|cuda runtime error|CUDA error"

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

echo "$(date): Starting Live VLM Server..." | tee -a "$LOG_FILE"

# Get the container image
IMAGE=$(autotag nano_llm)
echo "$(date): Using image: $IMAGE" | tee -a "$LOG_FILE"

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
  --ipc=host \
  --ulimit memlock=-1 \
  -e PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128' \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -e CUDA_CACHE_DISABLE=1 \
  "$IMAGE" \
  python3 data/nano_vlm/nano_vlm_server.py 2>&1 | tee -a "$LOG_FILE"

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
