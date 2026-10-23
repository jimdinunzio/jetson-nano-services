#!/usr/bin/env python3
"""
Power probe -- TEMPORARY debug logger for the Jetson's power rails.

The Jetson has been losing power outright while serving (no shutdown, no
panic; the battery and buck/boost are shared with the LattePanda). This logs
the on-module INA3221 rails plus GPU load to disk, every sample synced, so the
seconds before the next cut survive it.

Remove it when the debugging is done: power-probe-service/uninstall.sh.

Columns (one line per sample):
    wall      ISO time; stale until NTP syncs after a power loss
    uptime_s  seconds since boot; trustworthy across clock jumps
    vin_mV    VDD_IN, the module's 5V input (after the carrier's regulator)
    in_mA     VDD_IN current
    cg_mA     VDD_CPU_GPU_CV current
    soc_mA    VDD_SOC current
    gpu_pct   GPU load
"""

import datetime
import glob
import os
import signal
import sys
import time

# The INA3221 averages internally and its readings move every ~70 ms at best;
# 5 Hz still puts 10+ samples across the 2-3 s GPU ramp that preceded each cut.
# Each sample is four I2C reads (~3 ms) plus one fdatasync.
RATE_HZ = float(os.environ.get("POWER_PROBE_HZ", "5"))

# Rotation caps the disk use at MAX_BYTES * KEEP (32 MiB, ~1.5 days at 5 Hz).
MAX_BYTES = 8 * 1024 * 1024
KEEP = 4

LOG_DIR = os.environ.get("LOGS_DIRECTORY", "/var/log/power-probe")
LOG_PATH = os.path.join(LOG_DIR, "power.log")
GPU_LOAD = "/sys/devices/platform/bus@0/17000000.gpu/load"
REMIND_EVERY_S = 3600

WARNING = (f"POWER PROBE ENABLED (temporary debug): logging power rails at "
           f"{RATE_HZ:g} Hz to {LOG_PATH}; remove with "
           f"power-probe-service/uninstall.sh")


def warn(msg):
    # The <4> prefix makes journald (and so syslog) record it at warning level.
    print(f"<4>{msg}", flush=True)


def find_rails():
    """Map label -> (voltage path, current path) for the INA3221 channels."""
    for hw in glob.glob("/sys/class/hwmon/hwmon*"):
        with open(os.path.join(hw, "name")) as f:
            if f.read().strip() != "ina3221":
                continue
        rails = {}
        for label_path in glob.glob(os.path.join(hw, "in[0-9]_label")):
            n = os.path.basename(label_path)[2]
            with open(label_path) as f:
                label = f.read().strip()
            curr = os.path.join(hw, f"curr{n}_input")
            if os.path.exists(curr):
                rails[label] = (os.path.join(hw, f"in{n}_input"), curr)
        return rails
    sys.exit("power_probe: no ina3221 hwmon device found")


def read_int(path):
    try:
        with open(path) as f:
            return int(f.read())
    except (OSError, ValueError):
        return -1


def boot_id():
    with open("/proc/sys/kernel/random/boot_id") as f:
        return f.read().strip()


def reset_reason():
    """Why the SoC last came out of reset: a watchdog, thermal trip or software
    reboot each record their own reason; SYS_RESET_N is the external reset line
    (power-on, input power-good dropping, or the J14 reset pin)."""
    base = "/sys/devices/platform/bus@0/c360000.pmc"
    try:
        with open(f"{base}/reset_reason") as f:
            reason = f.read().strip()
        with open(f"{base}/reset_level") as f:
            return f"{reason} {f.read().strip()}"
    except OSError:
        return "unknown"


def uptime():
    with open("/proc/uptime") as f:
        return float(f.read().split()[0])


def rotate():
    for i in range(KEEP - 1, 0, -1):
        src = LOG_PATH if i == 1 else f"{LOG_PATH}.{i - 1}"
        if os.path.exists(src):
            os.replace(src, f"{LOG_PATH}.{i}")


def open_log():
    return os.open(LOG_PATH, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)


def write(fd, line):
    os.write(fd, line.encode())
    os.fdatasync(fd)


def main():
    rails = find_rails()
    vin_v, vin_i = rails["VDD_IN"]
    cg_i = rails["VDD_CPU_GPU_CV"][1]
    soc_i = rails["VDD_SOC"][1]

    os.makedirs(LOG_DIR, exist_ok=True)
    if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) >= MAX_BYTES:
        rotate()
    fd = open_log()

    def stop(signum, _frame):
        write(fd, f"# stopped by signal {signum} at uptime {uptime():.1f}\n")
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    warn(WARNING)
    write(fd, f"# {WARNING}\n"
              f"# boot {boot_id()} at uptime {uptime():.1f}, "
              f"reset reason {reset_reason()}\n"
              f"# wall uptime_s vin_mV in_mA cg_mA soc_mA gpu_pct\n")

    period = 1.0 / RATE_HZ
    next_tick = time.monotonic()
    next_remind = next_tick + REMIND_EVERY_S
    while True:
        wall = datetime.datetime.now().isoformat(timespec="milliseconds")
        gpu = read_int(GPU_LOAD)
        write(fd, f"{wall} {uptime():.2f} {read_int(vin_v)} {read_int(vin_i)} "
                  f"{read_int(cg_i)} {read_int(soc_i)} "
                  f"{gpu / 10 if gpu >= 0 else -1:.1f}\n")

        if os.fstat(fd).st_size >= MAX_BYTES:
            os.close(fd)
            rotate()
            fd = open_log()
            write(fd, f"# {WARNING}\n# continued after rotation, boot {boot_id()}\n")

        now = time.monotonic()
        if now >= next_remind:
            warn(WARNING)
            next_remind = now + REMIND_EVERY_S
        next_tick += period
        if next_tick < now:  # fell behind (suspend, I2C stall): resync
            next_tick = now + period
        time.sleep(next_tick - now)


if __name__ == "__main__":
    main()
