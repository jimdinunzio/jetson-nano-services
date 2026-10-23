# power-probe-service — TEMPORARY power-rail logger

**Debug only. Remove it when done:** `./uninstall.sh` (add `--purge` to delete
the logs too).

The Jetson has been losing power outright while serving: no shutdown, no panic,
just a fresh boot whose first syslog line starts with NUL bytes. It shares one
12 V battery and a 72 W buck/boost with the LattePanda. This service logs the
on-module rails so the seconds before the next cut survive it.

## What it records

`/var/log/power-probe/power.log`, 5 lines a second, each one synced to disk:

```
# wall uptime_s vin_mV in_mA cg_mA soc_mA gpu_pct
2026-10-18T17:40:01.204 512.33 5104 1240 360 312 0.0
```

- `vin_mV` / `in_mA`: VDD_IN, the module's 5 V input. It is downstream of the
  carrier's regulator, so a 12 V sag shows up as a current step the regulator
  rides out (or doesn't), not as the battery voltage itself.
- `cg_mA` / `soc_mA`: the CPU+GPU+CV and SOC rails.
- `uptime_s` is the column to trust: after a power loss the wall clock comes
  back stale and jumps 15-25 min at the NTP sync.

Each boot starts with a `# boot <id>` header. A boot whose log just stops,
with no `# stopped` line, is a power cut; its last lines are the evidence.

## Cost

About 3 ms of I2C reads and one small `fdatasync` per sample, run at nice 10.
Rotation caps disk use at 4 x 8 MiB (about 1.5 days of history).
`POWER_PROBE_HZ` in the unit changes the rate.

## Reminders that it is on

- A warning in syslog when it starts and every hour after.
- A yellow banner in every interactive shell (`banner.sh`, sourced from
  `~/.bashrc`; `uninstall.sh` removes the line).
- The first line of each boot's log.
