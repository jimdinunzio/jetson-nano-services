#!/bin/bash
#
# Power probe install script. TEMPORARY: not part of the top-level install.sh;
# undo it with uninstall.sh once the power losses are understood.
#
# Installs + enables + starts the unit, and adds a shell banner to ~/.bashrc
# that warns while the probe is running.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="power-probe.service"
BASHRC_LINE="[ -r $SCRIPT_DIR/banner.sh ] && . $SCRIPT_DIR/banner.sh  # power-probe"

if [ "$SCRIPT_DIR" != "/home/jim/Documents/git/jetson-nano-services/power-probe-service" ]; then
    echo "Note: $SERVICE_FILE has an absolute ExecStart; edit it to point at"
    echo "      $SCRIPT_DIR/power_probe.py before starting the service."
fi

echo "Installing $SERVICE_FILE to /etc/systemd/system ..."
sudo install -m 0644 "$SCRIPT_DIR/$SERVICE_FILE" "/etc/systemd/system/$SERVICE_FILE"
sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_FILE"

if ! grep -qF "# power-probe" "$HOME/.bashrc"; then
    echo "Adding the power-probe banner to ~/.bashrc ..."
    echo "$BASHRC_LINE" >> "$HOME/.bashrc"
fi

echo "Done. Log: /var/log/power-probe/power.log   Remove: $SCRIPT_DIR/uninstall.sh"
