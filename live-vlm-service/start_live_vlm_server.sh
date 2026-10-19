#!/bin/bash
#
# Live VLM Server Startup Script
#
# Watches the log for CUDA/memory errors and stops the container when one
# appears. IT DOES NOT REBOOT: the supervisor scans this log, reports the
# failure and leaves the box up for a person. See service_supervisor.py.
#
# Stopping is abrupt on purpose -- SIGINT, the signal Ctrl-C sends, then SIGKILL
# if it has not exited within 2s. `docker stop` spends a grace period per call,
# and a switch that is already stopping one model to load another has no
# seconds to spend being polite.
#

LOG_FILE="/tmp/live_vlm_server.log"
CONTAINER_NAME="live_vlm_server"
# A GPU/host allocation failure reaches the log as a cuDNN/cuBLAS init or
# "CUDA error" line: the driver reports the handle it could not create, not the
# ENOMEM underneath. NvMapMemAllocInternalTagged "error 12" lines are not
# listed -- they also appear on healthy loads and mid-run, so they are noise.
ERROR_PATTERNS="CUDA out of memory|CUDA: out of memory|OutOfMemoryError|NVML_SUCCESS.*INTERNAL ASSERT FAILED|RuntimeError.*CUDACachingAllocator|cuda runtime error|CUDA error|cuDNN error|CUDNN_STATUS_INTERNAL_ERROR|CUDNN_STATUS_NOT_INITIALIZED|CUDNN_STATUS_ALLOC_FAILED|CUBLAS_STATUS_ALLOC_FAILED|CUBLAS_STATUS_NOT_INITIALIZED"

# Interrupt the container, give it up to 2s to exit, then kill it. `docker wait`
# returns the moment it exits, and the SIGINT fails at once when there is no
# container -- so neither a clean exit nor a fresh start pays the 2s.
stop_container() {
    if docker kill --signal=SIGINT "$CONTAINER_NAME" >/dev/null 2>&1; then
        timeout 2 docker wait "$CONTAINER_NAME" >/dev/null 2>&1
        docker kill "$CONTAINER_NAME" >/dev/null 2>&1
    fi
    return 0
}

# Cleanup function to stop the container
cleanup() {
    echo "$(date): Received shutdown signal, stopping container..." | tee -a "$LOG_FILE"

    stop_container

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
stop_container
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
            # Nothing here reboots. We log the error and stop the
            # container; the supervisor scans this log, sees the error and
            # reports needs_reboot to the remote host, leaving the board up.
            echo "$(date): CUDA/Memory error detected! Stopping container..." | tee -a "$LOG_FILE"
            echo "$(date): Error line: $line" | tee -a "$LOG_FILE"

            # Stop the container to trigger exit.
            stop_container
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
