#!/usr/bin/env python3
# coding: utf-8
"""
wave_arm.py - make the arm wave.

Assumes the arm starts close to the REST pose below. Moves through a short
wave sequence, then holds the raised pose until Ctrl-C, at which point it
returns to REST.

  python3 wave_arm.py
  python3 wave_arm.py --port /dev/ttyTHS1   # explicit port
"""

import argparse
import time

from arm_driver import Arm_Driver

# Servo 1-6 angles for each pose.
REST    = [88, 147,  0,  8,  2, 133]
RAISED  = [88, 147, 11, 28,  2, 131]
WAVE_A  = [126, 162, 11, 73, 43, 109]
WAVE_B  = [43, 166, 11, 73,  2, 109]


def move(arm, joints, secs):
    """Command all six servos to `joints` over `secs` seconds, then wait it out."""
    ms = int(secs * 1000)
    arm.Arm_serial_servo_write6_array(joints, ms)
    time.sleep(secs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='/dev/ttyTHS1')
    args = ap.parse_args()

    arm = Arm_Driver(com=args.port)

    try:
        # Raise from rest.
        move(arm, RAISED, 1.0)
        time.sleep(1.0)

        # One wave: A -> B -> back to A.
        move(arm, WAVE_A, 0.5)
        time.sleep(0.5)
        move(arm, WAVE_B, 0.5)
        time.sleep(0.5)
        move(arm, WAVE_A, 0.5)
        time.sleep(0.5)

        # Return to the raised pose and hold until interrupted.
        move(arm, RAISED, 1.0)

    finally:
        move(arm, REST, 1.0)


if __name__ == '__main__':
    main()
