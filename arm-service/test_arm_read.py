#!/usr/bin/env python3
# coding: utf-8
"""
Minimal read check for servo angles. Run this ON THE JETSON with the servos powered.

  python3 test_arm_.py 120
  python3 test_arm_read.py 120 --port /dev/ttyTHS1
"""

import argparse

from Arm_Lib import Arm_Driver

ap = argparse.ArgumentParser()
ap.add_argument('--port', default='/dev/ttyTHS1')
args = ap.parse_args()

arm = Arm_Driver(com='/dev/ttyTHS1')

for i in range(1,7):
    print('ping servo %s -> %s' % (i, arm.Arm_ping_servo(i) == 0xDA and 'alive' or 'no reply'))
    print('read servo %s -> %s deg' % (i, arm.Arm_serial_servo_read(i)))
