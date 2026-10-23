#!/bin/bash
#
# Removes the power probe: stops and deletes the unit and the ~/.bashrc banner.
# The logs stay in /var/log/power-probe unless --purge is given.
#
set -euo pipefail

SERVICE_FILE="power-probe.service"

echo "Stopping and removing $SERVICE_FILE ..."
sudo systemctl disable --now "$SERVICE_FILE" 2>/dev/null || true
sudo rm -f "/etc/systemd/system/$SERVICE_FILE"
sudo systemctl daemon-reload

if grep -qF "# power-probe" "$HOME/.bashrc"; then
    echo "Removing the power-probe banner from ~/.bashrc ..."
    sed -i '/# power-probe$/d' "$HOME/.bashrc"
fi

if [ "${1:-}" = "--purge" ]; then
    echo "Deleting /var/log/power-probe ..."
    sudo rm -rf /var/log/power-probe
else
    echo "Logs kept in /var/log/power-probe (rerun with --purge to delete them)."
fi
echo "Done."
