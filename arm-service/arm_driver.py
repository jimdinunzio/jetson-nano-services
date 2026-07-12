#!/usr/bin/env python3
# coding: utf-8
"""
Arm_Driver - drop-in replacement for Arm_Lib.Arm_Device that speaks the
YB-SD15M smart serial bus-servo UART protocol directly, instead of going
through the Yahboom expansion board over I2C.

Host: Jetson Orin Nano, talking to the servo chain over its hardware UART
(e.g. /dev/ttyTHS1). pyserial only - no RPi.GPIO.

All method signatures match Arm_Lib.Arm_Device so existing callers work
unchanged. Expansion-board-only features (RGB, buzzer, button, PWM, action
groups, product select, etc.) have no serial equivalent and are stubbed: they
warn once and return a neutral value.

Protocol (YB-SD15M_Bus_Servo_Protocol.xlsx):
  Command frame (host->servo):  0xFF 0xFF | ID | LEN | CMD | ADDR | DATA... | CHK
    LEN = len(DATA) + 3            (counts CMD + ADDR + DATA + CHK)
    CHK = (~(ID + LEN + CMD + ADDR + sum(DATA))) & 0xFF
  Reply frame (servo->host):    0xFF 0xF5 | ID | LEN(0x04) | CMD(0x02) | DATA_H | DATA_L | CHK
    Parsed by keying on the 0xF5 header, then the following 6 bytes:
    [id, len, cmd, val_H, val_L, chk]
    chk   = (~(id + len + cmd + val_H + val_L)) & 0xFF
    value = (val_H << 8) | val_L          (raw position 96..4000)
"""

import time
import threading

import serial


# Command opcodes / register addresses
_CMD_READ = 0x02
_CMD_WRITE = 0x03
_ADDR_POS = 0x2A      # write target position
_ADDR_READ_POS = 0x38  # read current position
_ADDR_TORQUE = 0x28    # torque enable/disable
_ADDR_SET_ID = 0x05    # set servo ID (broadcast only)
_BROADCAST_ID = 0xFE
_REPLY_HEADER = 0xF5


# ------------------------------------------------------------------ #
# Servo calibration - raw position <-> API angle mapping.
# EDIT THESE to re-range the arm for a new installation. Standard servos
# (1-4, 6) span ANGLE_MIN..ANGLE_MAX degrees over POS_MIN..POS_MAX raw units;
# servo 5 has its own wider range.
# ------------------------------------------------------------------ #
ANGLE_MIN = 0
ANGLE_MAX = 180
POS_MIN = 900          # raw position at ANGLE_MIN
POS_MAX = 3100         # raw position at ANGLE_MAX

ANGLE5_MIN = 0
ANGLE5_MAX = 270
POS5_MIN = 380         # raw position at ANGLE5_MIN
POS5_MAX = 3700        # raw position at ANGLE5_MAX

# Servos whose API angle runs opposite to raw position (mechanically inverted).
_INVERTED_IDS = (2, 3, 4)

# Absolute servo position limits (protocol raw range 96..4000). Angle inputs
# outside the nominal 0-180 range are allowed and converted; only the final raw
# position is clamped to these, to protect the servo and keep frames valid.
RAW_MIN = 96
RAW_MAX = 4000

# bus_servo_control() raw-unit convention (from Arm_Lib): servo 5 is written at
# pos = num - _CTRL5_OFFSET.
_CTRL5_OFFSET = 514


class Arm_Driver(object):

    def __init__(self, com="/dev/ttyTHS1", baudrate=115200):
        """
        com       serial port (Jetson hardware UART, e.g. /dev/ttyTHS1).
        baudrate  115200 (8N1) for YB-SD15M.
        """
        self.port = com
        # timeout >= 10ms: full 8-byte reply ~= 0.7ms at 115200, plus servo
        # processing headroom.
        self.ser = serial.Serial(com, baudrate, timeout=0.02)
        # The serial port is a single shared resource; serialize a write and its
        # echo/reply so concurrent threads cannot interleave bus traffic.
        self.lock = threading.Lock()
        self._warned = set()  # names of stubs already warned about (warn once)

    # ------------------------------------------------------------------ #
    # Low-level framing helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _checksum(parts):
        return (~sum(parts)) & 0xFF

    def _build_frame(self, id, cmd, addr, data):
        length = len(data) + 3  # CMD + ADDR + DATA + CHK
        body = [id & 0xFF, length, cmd, addr] + [d & 0xFF for d in data]
        chk = self._checksum(body)
        return bytes([0xFF, 0xFF] + body + [chk])

    def _send(self, id, cmd, addr, data):
        """Transmit one write-only command frame (no reply expected)."""
        frame = self._build_frame(id, cmd, addr, data)
        try:
            with self.lock:
                self.ser.write(frame)
        except Exception:
            print('Arm_Driver serial write error')

    def _query(self, id, cmd, addr, data, expect_id=None):
        """Transmit a command and read its reply as one atomic transaction (held
        under a single lock so the write and its reply cannot interleave with
        another thread). Returns the raw position value (96..4000) or None."""
        frame = self._build_frame(id, cmd, addr, data)
        try:
            with self.lock:
                self.ser.write(frame)
                buf = self.ser.read(32)
        except Exception:
            print('Arm_Driver serial query error')
            return None
        return self._parse_reply(buf, expect_id)

    def _parse_reply(self, buf, expect_id=None):
        """Scan the RX stream for the 0xF5-framed reply, verify checksum, return
        raw position value (96..4000) or None. Any leading echo / stray bytes are
        skipped."""
        idx = buf.find(_REPLY_HEADER)
        while idx != -1 and idx + 7 <= len(buf):
            # bytes after 0xF5: id, len(0x04), cmd(0x02), val_H, val_L, chk
            s_id, length, cmd, val_h, val_l, chk = buf[idx + 1:idx + 7]
            if self._checksum([s_id, length, cmd, val_h, val_l]) == chk:
                if expect_id is None or s_id == (expect_id & 0xFF):
                    return (val_h << 8) | val_l
            idx = buf.find(_REPLY_HEADER, idx + 1)
        return None

    # ------------------------------------------------------------------ #
    # Angle <-> raw position conversion (verbatim from Arm_Lib)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _angle_to_pos(id, angle):
        """Convert an API angle (deg) to a raw servo position, applying the
        per-servo inversion / range exactly as Arm_Lib.Arm_Device does."""
        if id == 5:
            return int((POS5_MAX - POS5_MIN) * (angle - ANGLE5_MIN) / (ANGLE5_MAX - ANGLE5_MIN) + POS5_MIN)
        if id in _INVERTED_IDS:
            angle = ANGLE_MIN + ANGLE_MAX - angle
        return int((POS_MAX - POS_MIN) * (angle - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN) + POS_MIN)

    @staticmethod
    def _pos_to_angle(id, pos):
        """Inverse of _angle_to_pos. Returns the raw position mapped to degrees,
        even if it falls outside the nominal range, so callers see the true
        physical pose."""
        if id == 5:
            return int((ANGLE5_MAX - ANGLE5_MIN) * (pos - POS5_MIN) / (POS5_MAX - POS5_MIN) + ANGLE5_MIN)
        angle = int((ANGLE_MAX - ANGLE_MIN) * (pos - POS_MIN) / (POS_MAX - POS_MIN) + ANGLE_MIN)
        if id in _INVERTED_IDS:
            angle = ANGLE_MIN + ANGLE_MAX - angle
        return angle

    def _write_pos(self, id, pos, time_ms):
        # Clamp to the servo's absolute raw limits (96..4000). This keeps the
        # 12-bit position field valid even when the angle maps out of the
        # nominal 0-180 range (e.g. writing back a pose read past nominal).
        pos = max(RAW_MIN, min(RAW_MAX, int(pos)))
        data = [(pos >> 8) & 0xFF, pos & 0xFF,
                (time_ms >> 8) & 0xFF, time_ms & 0xFF]
        self._send(id, _CMD_WRITE, _ADDR_POS, data)

    # ------------------------------------------------------------------ #
    # Public API - motion writes
    # ------------------------------------------------------------------ #
    # Set bus servo angle: id 1-6 (0 broadcasts to all 6 servos), angle 0-180
    # (servo 5: 0-270), time = move duration in ms.
    def Arm_serial_servo_write(self, id, angle, time):
        if id == 0:  # all servos
            self.Arm_serial_servo_write6(angle, angle, angle, angle, angle, angle, time)
            return
        self._write_pos(id, self._angle_to_pos(id, angle), time)

    # Set any bus servo angle: id 1-250 (0 broadcasts), angle 0-180 -> 900-3100.
    # No per-joint inversion (matches Arm_Lib.Arm_serial_servo_write_any).
    def Arm_serial_servo_write_any(self, id, angle, time):
        pos = int((POS_MAX - POS_MIN) * (angle - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN) + POS_MIN)
        target = (id & 0xFF) if id != 0 else _BROADCAST_ID
        self._write_pos(target, pos, time)

    # Set all six servo angles. s1-s4 and s6 use 0-180, s5 uses 0-270. Angles
    # outside the nominal range are accepted (so a pose read past nominal can be
    # written back); the resulting raw position is clamped in _write_pos.
    # The protocol has no multi-position broadcast, so send one packet per servo.
    def Arm_serial_servo_write6(self, s1, s2, s3, s4, s5, s6, time):
        for sid, ang in zip((1, 2, 3, 4, 5, 6), (s1, s2, s3, s4, s5, s6)):
            self._write_pos(sid, self._angle_to_pos(sid, ang), time)

    # Set all six servo angles from an array.
    def Arm_serial_servo_write6_array(self, joints, time):
        s1, s2, s3, s4, s5, s6 = joints[0], joints[1], joints[2], joints[3], joints[4], joints[5]
        self.Arm_serial_servo_write6(s1, s2, s3, s4, s5, s6, time)

    # ------------------------------------------------------------------ #
    # Public API - reads
    # ------------------------------------------------------------------ #
    # Read specified servo angle, id 1-6 returns 0-180 (servo 5: 0-270); None on fail.
    def Arm_serial_servo_read(self, id):
        if id < 1 or id > 6:
            print("id must be 1 - 6")
            return None
        # The gripper (servo 6) sits at the far end of the chain and can drop a
        # single query; retry a few times, the way Arm_ping_servo does.
        for _ in range(5):
            pos = self._query(id, _CMD_READ, _ADDR_READ_POS, [0x02], expect_id=id)
            if pos:  # not None and not 0
                return self._pos_to_angle(id, pos)
            time.sleep(0.003)
        return None

    # Read bus servo angle, id 1-250 returns 0-180 (no inversion).
    def Arm_serial_servo_read_any(self, id):
        if id < 1 or id > 250:
            print("id must be 1 - 250")
            return None
        pos = self._query(id, _CMD_READ, _ADDR_READ_POS, [0x02], expect_id=id)
        if pos is None:
            return None
        return int((ANGLE_MAX - ANGLE_MIN) * (pos - POS_MIN) / (POS_MAX - POS_MIN) + ANGLE_MIN)

    # Read a servo's RAW position (96..4000) with no angle conversion. Returns
    # None only on no reply. Intended for calibration (see calibrate_arm.py).
    def Arm_serial_servo_read_raw(self, id):
        if id < 1 or id > 250:
            print("id must be 1 - 250")
            return None
        for _ in range(5):
            pos = self._query(id, _CMD_READ, _ADDR_READ_POS, [0x02], expect_id=id)
            if pos:
                return pos
            time.sleep(0.003)
        return None

    # Ping a servo. Returns 0xDA on success (mirrors Arm_Lib's "alive" sentinel),
    # None if no valid reply after retries.
    def Arm_ping_servo(self, id):
        data = int(id)
        if data <= 0 or data > 250:
            return None
        for _ in range(5):
            if self._query(data, _CMD_READ, _ADDR_READ_POS, [0x02], expect_id=data) is not None:
                return 0xDA
            time.sleep(0.003)
        return None

    # ------------------------------------------------------------------ #
    # Public API - configuration
    # ------------------------------------------------------------------ #
    # Torque switch. 1: enable torque, 0: disable (servo can be moved by hand).
    def Arm_serial_set_torque(self, onoff):
        val = 0x01 if onoff == 1 else 0x00
        for sid in range(1, 7):
            self._send(sid, _CMD_WRITE, _ADDR_TORQUE, [val])

    # Set bus servo ID (broadcast). Connect only ONE servo at a time when doing this.
    def Arm_serial_set_id(self, id):
        self._send(_BROADCAST_ID, _CMD_WRITE, _ADDR_SET_ID, [id & 0xFF])

    # ------------------------------------------------------------------ #
    # Public API - raw-unit position control (reuses Arm_Lib remapping)
    # ------------------------------------------------------------------ #
    def bus_servo_control(self, id, num, time=1000):
        # Raw-unit control. Out-of-range num is accepted; the resulting raw
        # position is clamped in _write_pos.
        if id == 1 or id == 6:
            pos = int(num)
        elif id == 2 or id == 3 or id == 4:  # invert actual angle
            pos = int(POS_MIN + POS_MAX - num)
        elif id == 5:
            pos = int(num - _CTRL5_OFFSET)
        else:
            print("bus_servo_control error, id must be [1, 6]")
            return
        self._write_pos(id, pos, time)

    def bus_servo_control_array6(self, array, time=1000):
        if len(array) != 6:
            print("bus_servo_control_array6 input error")
            return
        s1, s2, s3, s4, s5, s6 = array[0], array[1], array[2], array[3], array[4], array[5]
        # Out-of-range values are accepted; raw positions are clamped in _write_pos.
        positions = [int(s1),
                     int(POS_MIN + POS_MAX - s2),
                     int(POS_MIN + POS_MAX - s3),
                     int(POS_MIN + POS_MAX - s4),
                     int(s5 - _CTRL5_OFFSET),
                     int(s6)]
        for sid, pos in zip((1, 2, 3, 4, 5, 6), positions):
            self._write_pos(sid, pos, time)

    # Compatibility shim - there is no I2C bus number for the serial driver.
    def get_i2c_bus_num(self):
        return self.port

    # ------------------------------------------------------------------ #
    # Stubs - expansion-board-only features, no serial equivalent.
    # Warn once, return a neutral value.
    # ------------------------------------------------------------------ #
    def _stub(self, name):
        if name not in self._warned:
            self._warned.add(name)
            print('%s: not supported on the serial bus-servo driver' % name)

    def Arm_get_hardversion(self):
        self._stub('Arm_get_hardversion')
        return 'serial-driver'

    def Arm_Read_Action_Num(self):
        self._stub('Arm_Read_Action_Num')
        return 0

    def Arm_serial_servo_write_offset_state(self):
        self._stub('Arm_serial_servo_write_offset_state')
        return 0

    def Arm_serial_servo_write_offset_switch(self, id):
        self._stub('Arm_serial_servo_write_offset_switch')

    def Arm_Product_Select(self, index):
        self._stub('Arm_Product_Select')

    def Arm_RGB_set(self, red, green, blue):
        self._stub('Arm_RGB_set')

    def Arm_Button_Mode(self, mode):
        self._stub('Arm_Button_Mode')

    def Arm_reset(self):
        self._stub('Arm_reset')

    def Arm_PWM_servo_write(self, id, angle):
        self._stub('Arm_PWM_servo_write')

    def Arm_Clear_Action(self):
        self._stub('Arm_Clear_Action')

    def Arm_Action_Study(self):
        self._stub('Arm_Action_Study')

    def Arm_Action_Mode(self, mode):
        self._stub('Arm_Action_Mode')

    def Arm_Buzzer_On(self, delay=0xff):
        self._stub('Arm_Buzzer_On')

    def Arm_Buzzer_Off(self):
        self._stub('Arm_Buzzer_Off')
