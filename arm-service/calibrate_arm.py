#!/usr/bin/env python3
# coding: utf-8
"""
Interactive calibration helper. Run this ON THE JETSON with the servos powered.

Disables torque so you can move each joint by hand, then continuously prints the
RAW position (96..4000) of every servo along with the running min/max seen so
far. Move each joint slowly to both mechanical limits; the min/max it captures
are the numbers to plug into arm_driver.py:

  standard servos (1-4, 6):  POS_MIN / POS_MAX
  servo 5:                   POS5_MIN / POS5_MAX

Ctrl-C to exit. Torque is re-enabled on exit so the arm holds its pose.

  python3 calibrate_arm.py
  python3 calibrate_arm.py --port /dev/ttyTHS1 --hz 10
"""

import argparse
import time

from Arm_Lib import Arm_Driver

ap = argparse.ArgumentParser()
ap.add_argument('--port', default='/dev/ttyTHS1')
ap.add_argument('--hz', type=float, default=5.0, help='refresh rate (samples/sec)')
args = ap.parse_args()

arm = Arm_Driver(com=args.port)

print("Disabling torque - you can now move the arm by hand.")
print("Move each joint to both limits, then Ctrl-C.\n")
arm.Arm_serial_set_torque(0)

mins = [None] * 6
maxs = [None] * 6
period = 1.0 / args.hz if args.hz > 0 else 0.2


def fmt(v):
    return '----' if v is None else '%4d' % v


try:
    while True:
        raw = [arm.Arm_serial_servo_read_raw(sid) for sid in range(1, 7)]
        for i, r in enumerate(raw):
            if r is not None:
                mins[i] = r if mins[i] is None else min(mins[i], r)
                maxs[i] = r if maxs[i] is None else max(maxs[i], r)
        line = "  ".join(
            "s%d=%s[%s..%s]" % (i + 1, fmt(raw[i]), fmt(mins[i]), fmt(maxs[i]))
            for i in range(6)
        )
        print("\r" + line, end="", flush=True)
        time.sleep(period)
except KeyboardInterrupt:
    print("\n\nObserved raw ranges:")
    for i in range(6):
        print("  servo %d:  min=%s  max=%s" % (i + 1, fmt(mins[i]), fmt(maxs[i])))
    print("\nRe-enabling torque.")
    arm.Arm_serial_set_torque(1)
