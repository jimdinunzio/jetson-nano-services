#!/usr/bin/env python3
# coding: utf-8
"""
Write-then-read-back check for servo angles. Run this ON THE JETSON with the
servos powered.

Moves each servo to a target angle, waits for the motion to settle, then reads
the angle back and reports the error.

  python3 test_arm_write_read.py 120
  python3 test_arm_write_read.py 120 --port /dev/ttyTHS1 --time 1000
"""

import argparse
import time

from Arm_Lib import Arm_Driver

ap = argparse.ArgumentParser()
ap.add_argument('angle', type=int, help='target angle in degrees (0-180)')
ap.add_argument('--port', default='/dev/ttyTHS1')
ap.add_argument('--time', type=int, default=1000, help='move duration in ms')
args = ap.parse_args()

arm = Arm_Driver(com=args.port)

for i in range(1, 6):
    arm.Arm_serial_servo_write(i, args.angle, args.time)
    time.sleep(args.time / 1000.0 + 0.2)
    back = arm.Arm_serial_servo_read(i)
    if back is None:
        print('servo %s -> wrote %s deg, read back None (no reply)' % (i, args.angle))
    else:
        print('servo %s -> wrote %s deg, read %s deg (err %+d)'
              % (i, args.angle, back, back - args.angle))
