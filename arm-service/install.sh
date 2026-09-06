#!/bin/bash
#
# DOFBOT Arm Service install script.
#
# Installs + enables the systemd unit. ExecStart points into this checkout, so
# nothing is copied and `git pull` is enough to update the server.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="dofbot-arm.service"

if [ "$SCRIPT_DIR" != "/home/jim/Documents/git/jetson-nano-services/arm-service" ]; then
    echo "Note: $SERVICE_FILE has an absolute ExecStart; edit it to point at"
    echo "      $SCRIPT_DIR/start_arm_server.sh before starting the service."
fi

echo "Installing $SERVICE_FILE to /etc/systemd/system ..."
sudo install -m 0644 "$SCRIPT_DIR/$SERVICE_FILE" "/etc/systemd/system/$SERVICE_FILE"

echo "Reloading systemd daemon ..."
sudo systemctl daemon-reload

echo "Enabling $SERVICE_FILE ..."
sudo systemctl enable "$SERVICE_FILE"

echo "Done. Start now with: sudo systemctl start $SERVICE_FILE"
