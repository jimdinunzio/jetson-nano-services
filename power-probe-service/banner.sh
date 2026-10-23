# Sourced from ~/.bashrc by install.sh: warns in every interactive shell while
# the temporary power probe is running.
if [ -n "${PS1:-}" ] && systemctl is-active --quiet power-probe.service 2>/dev/null; then
    printf '\033[1;33m*** POWER PROBE ENABLED (temporary debug): logging power rails to /var/log/power-probe\n'
    printf '*** remove with: ~/Documents/git/jetson-nano-services/power-probe-service/uninstall.sh\033[0m\n'
fi
