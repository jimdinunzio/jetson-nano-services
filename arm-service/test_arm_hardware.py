#!/usr/bin/env python3
# coding: utf-8
"""
Simple on-hardware smoke test for arm_driver.Arm_Driver. Run this ON THE JETSON
with the servos powered.

  python3 test_arm_hardware.py
  python3 test_arm_hardware.py --port /dev/ttyTHS1 --no-move

It pings each servo, moves gently to the safe full_vertical pose, reads every joint back,
does a small base sweep and a gripper open/close, then drops torque so the arm
can be moved by hand. Moves use a slow duration; keep a hand near the power
switch the first time you run it.

  --port PORT   serial port (default /dev/ttyTHS1)
  --time MS     move duration in ms (default 1500)
  --no-move     comms only: ping + read, no motion
"""

import argparse
import time

from Arm_Lib import Arm_Driver

# Safe poses from arm_notes.md (angles: s1..s6).
full_vertical = (90, 90, 90, 90, 90, 50)     # upright / centred, gripper open
REST = (90, 156, 0, 2, 89, 90)      # curled tight over base, gripper half open


def settle(time_ms):
    time.sleep(time_ms / 1000.0 + 0.4)


def main():
    ap = argparse.ArgumentParser(description="Arm_Driver hardware smoke test")
    ap.add_argument('--port', default='/dev/ttyTHS1')
    ap.add_argument('--time', type=int, default=1500, help='move duration ms')
    ap.add_argument('--no-move', action='store_true',
                    help='comms only - ping and read, no motion')
    args = ap.parse_args()

    arm = Arm_Driver(com=args.port)
    print('Opened %s' % args.port)

    # 1. Comms check - ping each servo.
    print('\nPing servos (expect 0xDA / 218 if alive):')
    for sid in range(1, 7):
        print('  servo %d -> %s' % (sid, arm.Arm_ping_servo(sid)))

    # 2. Read current angles before moving.
    print('\nCurrent angles:')
    for sid in range(1, 7):
        print('  servo %d -> %s deg' % (sid, arm.Arm_serial_servo_read(sid)))

    if args.no_move:
        print('\n--no-move set; skipping motion. Done.')
        return

    # 3. Move gently to full_vertical, then read back.
    print('\nMoving to full_vertical %s over %dms ...' % (full_vertical, args.time))
    arm.Arm_serial_servo_write6(*(full_vertical + (args.time,)))
    settle(args.time)
    print('Read back at full_vertical:')
    for sid in range(1, 7):
        print('  servo %d -> %s deg' % (sid, arm.Arm_serial_servo_read(sid)))

    # 4. Small base sweep to confirm direction (90 -> 60 -> 120 -> 90).
    print('\nBase (servo 1) sweep ...')
    for ang in (60, 120, 90):
        arm.Arm_serial_servo_write(1, ang, args.time)
        settle(args.time)
        print('  commanded %3d -> reads %s' % (ang, arm.Arm_serial_servo_read(1)))

    # 5. Gripper open / close (servo 6). Move away promptly so a closed gripper
    #    never stalls long enough to trip servo protection.
    print('\nGripper (servo 6) open -> close -> half ...')
    for ang in (0, 150, 50):
        arm.Arm_serial_servo_write(6, ang, 800)
        time.sleep(1.0)
        print('  commanded %3d -> reads %s' % (ang, arm.Arm_serial_servo_read(6)))

    # 6. Drop torque so the arm can be repositioned by hand.
    print('\nTorque OFF - arm should now move freely. Done.')
    arm.Arm_serial_set_torque(0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted.')
