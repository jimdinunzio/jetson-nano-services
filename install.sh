#!/bin/bash
#
# Meta install script for jetson-nano-services.
#
# Runs each service's own install.sh. ollama-service is skipped (it has no
# startup shell and is managed separately).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVICES=(
    arm-service
    supervisor-service
    nano-owl-service
    live-vlm-service
)

for svc in "${SERVICES[@]}"; do
    installer="$SCRIPT_DIR/$svc/install.sh"
    if [ ! -x "$installer" ]; then
        echo "ERROR: $installer not found or not executable" >&2
        exit 1
    fi
    echo "========================================"
    echo "Installing $svc ..."
    echo "========================================"
    "$installer"
    echo
done

echo "All services installed."
