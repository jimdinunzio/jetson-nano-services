#!/bin/bash
#
# Arm Service install script.
#
# Installs + enables the systemd unit. The startup shell runs directly from
# this checkout (ExecStart points here), so nothing is copied to $HOME.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="arm.service"

echo "Installing $SERVICE_FILE to /etc/systemd/system ..."
sudo install -m 0644 "$SCRIPT_DIR/$SERVICE_FILE" "/etc/systemd/system/$SERVICE_FILE"

echo "Reloading systemd daemon ..."
sudo systemctl daemon-reload

echo "Enabling $SERVICE_FILE ..."
sudo systemctl enable "$SERVICE_FILE"

echo "Done. Start now with: sudo systemctl start $SERVICE_FILE"
