#!/bin/bash
#
# NanoOWL Service install script.
#
# Installs the systemd unit. The startup shell runs directly from this checkout
# (ExecStart points here), so nothing is copied to $HOME. The unit is not
# enabled: the supervisor enables/disables it at runtime.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="nano_owl.service"

echo "Installing $SERVICE_FILE to /etc/systemd/system ..."
sudo install -m 0644 "$SCRIPT_DIR/$SERVICE_FILE" "/etc/systemd/system/$SERVICE_FILE"

echo "Reloading systemd daemon ..."
sudo systemctl daemon-reload

# Not enabled here: the supervisor enables/disables this unit at runtime.
echo "Done."
