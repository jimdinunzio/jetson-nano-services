#!/usr/bin/env python3
# coding: utf-8
"""
Offline self-test for arm_driver.Arm_Driver - no hardware required.

Replaces serial.Serial with a fake that records transmitted bytes and replays
canned reply bytes, so we can verify:
  * write/raw-control packets match the known-good golden vector
  * the reply parser decodes a 0xF5 frame to the right angle
  * a bad reply checksum yields None
  * stubbed methods warn once and return their neutral value

Run:  python test_arm_driver.py
"""

import sys
import types


class FakeSerial(object):
    """Minimal stand-in for serial.Serial. Records writes and replays canned
    read bytes. Constructing it opens nothing - it never touches real hardware
    and ignores the port/baud args entirely."""
    def __init__(self, *args, **kwargs):
        self.written = bytearray()
        self.to_read = bytearray()  # bytes the "device" will hand back on read()

    def write(self, data):
        self.written += bytes(data)
        return len(data)

    def read(self, n=1):
        chunk = self.to_read[:n]
        del self.to_read[:n]
        return bytes(chunk)

    def close(self):
        pass

    def reset_input_buffer(self):
        pass


# Force-replace the 'serial' module with a stub BEFORE importing arm_driver, so
# the driver binds serial.Serial -> FakeSerial and can NEVER open a real port -
# even if pyserial happens to be installed on this machine. Unconditional on
# purpose: do not fall through to real pyserial under any circumstance.
_stub = types.ModuleType('serial')
_stub.Serial = FakeSerial
sys.modules['serial'] = _stub

import arm_driver

# Hard guarantees that the offline test is fully isolated from real hardware.
assert arm_driver.serial is _stub, 'arm_driver did not bind to the stub serial module'
assert arm_driver.serial.Serial is FakeSerial, 'serial.Serial is not FakeSerial'


GOLDEN_WRITE = bytes.fromhex('ffff0107032a0c1c03e8b7')   # servo1 -> raw 3100, 1000ms
READ_REQUEST = bytes.fromhex('ffff0104023802be')          # read servo 1
# Reply: FF F5 | ID | LEN=04 | CMD=02 | DATA_H DATA_L | CHK   (raw 3100 -> angle 180)
GOOD_REPLY = bytes.fromhex('fff5' '01' '04' '02' '0c1c' 'd0')
BAD_REPLY = bytes.fromhex('fff5' '01' '04' '02' '0c1c' '00')  # wrong checksum


def new_driver():
    arm = arm_driver.Arm_Driver()
    assert isinstance(arm.ser, FakeSerial), 'driver opened a non-fake serial port'
    return arm, arm.ser


def expect(label, got, want):
    ok = got == want
    print(('PASS' if ok else 'FAIL'), label, '->', got if not ok else '')
    assert ok, '%s: got %r want %r' % (label, got, want)


def main():
    # 1. Golden vector via the angle API: Arm_serial_servo_write(1, 180, 1000).
    arm, ser = new_driver()
    arm.Arm_serial_servo_write(1, 180, 1000)
    expect('write(1,180,1000) frame', bytes(ser.written), GOLDEN_WRITE)

    # 1b. Same golden vector via raw-unit control: bus_servo_control(1, 3100, 1000).
    arm, ser = new_driver()
    arm.bus_servo_control(1, 3100, 1000)
    expect('bus_servo_control(1,3100,1000) frame', bytes(ser.written), GOLDEN_WRITE)

    # 2. Read parse: request frame correct, reply decodes to the right angle.
    arm, ser = new_driver()
    ser.to_read = bytearray(GOOD_REPLY)
    ang = arm.Arm_serial_servo_read(1)
    expect('read(1) request frame', bytes(ser.written), READ_REQUEST)
    expect('read(1) angle', ang, 180)

    # 3. Bad checksum -> None.
    arm, ser = new_driver()
    ser.to_read = bytearray(BAD_REPLY)
    expect('read(1) bad checksum', arm.Arm_serial_servo_read(1), None)

    # 4. No reply -> None.
    arm, ser = new_driver()
    expect('read(1) no reply', arm.Arm_serial_servo_read(1), None)

    # 5. Stubs: neutral returns; warning printed once.
    arm, ser = new_driver()
    expect('Arm_get_hardversion', arm.Arm_get_hardversion(), 'serial-driver')
    expect('Arm_Read_Action_Num', arm.Arm_Read_Action_Num(), 0)
    expect('Arm_serial_servo_write_offset_state', arm.Arm_serial_servo_write_offset_state(), 0)
    expect('Arm_RGB_set returns None', arm.Arm_RGB_set(255, 0, 0), None)
    arm.Arm_RGB_set(0, 0, 0)  # second call should NOT warn again
    expect('warn-once bookkeeping', 'Arm_RGB_set' in arm._warned, True)

    print('\nAll offline tests passed.')


if __name__ == '__main__':
    main()
