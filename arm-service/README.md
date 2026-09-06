# arm-service — XML-RPC front door to the DOFBOT stack

A remote caller (the robot's higher-level brain, a laptop, a script) says
`pick_can(x, y, z)`; this service turns that into the `ros2` commands
`dofbot_ctrl` already provides.

**The arm itself lives in another tree**: `~/ros2_ws/src/dofbot_ros2`, a colcon
workspace, and every file named here without a path is in it —
`ARCHITECTURE.md`, `gripper.py`, `chassis.xacro`. This service drives that
workspace from outside it: `DOFBOT_WS` says where it is, defaulting to
`~/ros2_ws`, and `start_arm_server.sh` sources it.

This service imports no rclpy and is not a ROS package. It is a supervisor: one
long-lived `ros2 launch` it keeps alive, one short-lived `ros2 run` per motion
command. Everything that is hard (planning, IK, the planning scene, the gripper
model) stays in `dofbot_ctrl`, where the tests are.

**It drives the servos through ROS, not over serial.** The version of this
service that opened `/dev/ttyTHS1` itself is gone; its bus-servo driver is now
`dofbot_arm_lib/Arm_Lib` in the workspace, and `moveit_bridge` is the only
thing that writes to the port. Two things must never hold it at once, so
nothing here touches serial.

## Commands

| method | what it runs |
|---|---|
| `enable_arm()` | `ros2 launch dofbot_ctrl pick_place.launch.py rviz:=false` |
| `disable_arm()` | stops that launch (whole process group) |
| `pick_can(x, y, z)` | `ros2 run dofbot_ctrl pick_place -- --pick x y z` |
| `place_can()` | `ros2 run dofbot_ctrl pick_place -- --place` |
| `move_to_state(name)` | `ros2 run dofbot_ctrl move_to_state -- NAME` |
| `reset_arm(state, force)` | `ros2 run dofbot_ctrl pick_place -- --reset STATE` |
| `wave_arm(waves, finish, seconds)` | `ros2 run dofbot_ctrl wave_arm -- --waves N` |
| `is_holding(object)` | `ros2 run dofbot_ctrl vision_check -- --held` |

Plus `list_states()`, `stop()`, `tail_log()`, `get_status()`, `ping()`.

Every command returns the same dict:

```python
{'ok': bool, 'command': str, 'returncode': int,
 'output': str,      # the ros2 command's console output
 'error': str, 'seconds': float}
```

`output` is the tail of what the node printed, which is where a MoveIt failure
explains itself — hand it to a human, don't parse it.

`is_holding()` adds three keys to that dict: `held` (**True / False / None**),
`verdict` (the same three as text) and `reason`.

## Running it

```bash
./start_arm_server.sh            # foreground, sources ROS + the workspace
sudo ./install.sh                # or as a systemd unit, enabled at boot
sudo systemctl start dofbot-arm
```

Port 8001 by default (`DOFBOT_ARM_PORT`) — the port the pre-ROS arm service
used, and which it no longer needs. Not 8002: that belongs to
`supervisor-service`.

**The unit is `dofbot-arm.service`, and the old one was `arm.service`.** A
machine that ran the serial version still has that unit installed; it points at
a startup script that no longer exists, so remove it rather than leave it to be
started by mistake:

```bash
sudo systemctl disable --now arm.service
sudo rm /etc/systemd/system/arm.service && sudo systemctl daemon-reload
```

`start_arm_server.sh` runs the system `python3` with ROS sourced, not a venv.
There is nothing to pip-install: the server is stdlib only and everything it
drives is a `ros2` command.

## Using it

```bash
python3 arm_client.py status
python3 arm_client.py enable
python3 arm_client.py state ready
python3 arm_client.py pick 0.30 0.0 0.039
python3 arm_client.py place
python3 arm_client.py held                     # is the can still in the jaws?
python3 arm_client.py wave 3                   # a greeting, then stow at init
python3 arm_client.py reset                   # after a pick that failed partway
python3 arm_client.py disable --park init
python3 arm_client.py -i                     # interactive
```

From code:

```python
from arm_client import ArmClient
arm = ArmClient('http://192.168.55.1:8001/')
arm.enable_arm()
if arm.pick_can(0.30, 0.0, 0.039)['ok']:
    arm.place_can()
```

## Where to put the robot

`pick_can()` takes the object where it is; it does not drive. Whoever does drive
gets to choose where the can ends up relative to the arm, and that choice decides
whether the pick is comfortable or knife-edge. **Aim for 0.30 m.**

```python
GRASP_STANDOFF = 0.30    # drive until the object sits here, base_link x
ARM_MIN_REACH  = 0.24    # closer than this, reposition rather than reach
ARM_MAX_REACH  = 0.36    # further than this, drive in
ARM_MAX_YAW    = 100.0   # degrees off centre, either side
```

**Test the radius, `hypot(x, y)`, not x.** The arm is rotationally symmetric
about its base yaw, so a can at (0.24, 0.20) is 0.31 m out and is a good target;
an x-only test reads it as too close and drives away from a comfortable pick.

Measured offline against the same pitch/standoff/height search `pick_place`
runs, soda can, extended fingers, carpet:

| object x | what the pick gets |
|---|---|
| < 0.20 | nothing solves at any pitch |
| 0.20–0.23 | solves, but off the proven grip height, standoff down to its 20 mm floor, under 0.05 rad of joint margin |
| 0.24–0.27 | works; standoff climbing 45 → 80 mm |
| **0.28–0.32** | **the sweet spot** — full 80 mm standoff, the 80 mm grip height proven on hardware, `grasp_pitch` landing on its preferred 2.2, > 0.12 rad of margin, and 20–50 mm of Cartesian lift |
| 0.33–0.36 | still full standoff and full margin, but no straight-up lift left; `move_named('carry')` does the clearing |
| 0.37–0.39 | degrades into shallow-pitch side grabs at a 20–45 mm standoff |
| ≥ 0.40 | nothing solves at any pitch |

**The near edge is the one that bites, and it bites suddenly.** A ±25 mm base
error either side of 0.30 stays inside the sweet spot; the same error from 0.22
lands at 0.195, where nothing solves. That asymmetry is why `GRASP_STANDOFF`
sits mid-band rather than at the closest workable distance — closer is not
safer here. See ARCHITECTURE.md, "The approach sweep": the standoff pose is the
grasp pulled *back along the tool axis*, which at a steep pitch is up and
**inward**, so it runs out before the reach does.

**Yaw is nearly free.** Nothing degrades out to 90°; margin falls to 0.14 rad at
100° and 0.06 at 105°, and 110° is past `arm1_Joint`'s stop. The chassis is not
what limits it — the arm's closest approach to the chassis cylinder is the
shoulder, at a fixed 60 mm, and swinging sideways only opens that up. 100° is
the gate; prefer to turn the base anyway, since a square approach costs one
cheap rotation.

**These numbers are for the EXTENDED fingers.** They move the whole working ring
outward. On the stock jaws with the 30 mm test block the same sweep gives a band
of 0.13–0.31 and a sweet spot of 0.20–0.29 — different constants entirely, so
anything that hard-codes the values above has to read `DOFBOT_GRIPPER` the way
`gripper.py` and `dofbot.urdf` both do.

## Things worth knowing

- **x, y, z is the object's CENTRE in `base_link`, in metres** — for something
  standing on the floor that is the floor height plus half the object's HEIGHT,
  not half its width and not zero. Same convention as `pick_place` itself. An
  upright 122 mm can on office carpet is `z = -0.022 + 0.061 = 0.039`; on a hard
  floor `-0.026 + 0.061 = 0.035`. Both ground figures are measured, and which
  one applies is what `DOFBOT_FLOOR` selects (see `chassis.xacro`).
- **The arm must be enabled first.** Motion commands do not auto-enable; being
  explicit is what stops a stray call from starting a stack nobody expected.
- **One motion at a time.** A second request comes back `busy` rather than
  queueing: a queued arm command would execute against a world that has moved
  on since it was sent.
- **`stop()` aborts the command, not the servos.** It SIGINTs the running node;
  the arm settles wherever the waypoint already in flight leaves it.
- **`moveit_bridge` needs `/dev/ttyTHS1` to itself.** Stop `gui_teleop` and
  `joint_state_mirror` before `enable_arm()`.
- **A failed `enable_arm()` tears its own launch down.** A leftover half-stack
  would put a second `move_group` on the domain at the next attempt, and two
  action servers driving one arm is a far worse failure than a clean retry.
- **After an aborted pick the planning scene still holds the object**, and
  moves out of it fail instantly with `INVALID_MOTION_PLAN`. `reset_arm()` is
  the way out: it clears the scene, opens the gripper and goes home, in that
  order, because planning cannot get out of a hole the scene is holding it in.
  `move_to_state()` will not do it — that deliberately does not touch the
  scene. `reset_arm(force=True)` is the last resort for when MoveIt will not
  plan from where the arm is; the blind move it makes is not collision checked.
- **`reset_arm()` drops what is held, where it is**, before the arm moves.
  Carrying it home first would only drop it from higher up.
- **`place_can()` drops at a named state, not at a coordinate.** `over_trash`
  is nominally straight in front of the robot and is a placeholder for a bin
  perception has yet to find; when it does, this grows the coordinates
  `pick_can()` already takes.
- **`wave_arm()` is a gesture, not a pose command.** It plans into a raised
  pose, sends the swings as one timed trajectory so the arm does not stop dead
  at each end of the swing, and stows at `init`. It goes through
  `moveit_bridge` like everything else, so the arm stays enabled and the whole
  path is collision-checked against the live scene — a wave refuses rather than
  sweeping through something the scene knows about.
- **One wave takes 3 s, and `seconds` sets that pace for any wave count.** The
  swings run well above `max_joint_speed`, deliberately: that 30 deg/s is a
  *picking* speed, set so the servos do not trail the plan by the bridge's
  200 ms `track_time_ms` and put the gripper somewhere the plan did not. A wave
  has nothing to arrive at, so the same lag only softens the ends of the swing.
- **`is_holding()` LOOKS, it does not remember.** It grabs a frame from the
  wrist camera and puts it to nanoOWL. The point of the question is exactly the
  case where a live look and the planning scene disagree — a can that was
  picked and has since slipped out.
- **`held` has three values and `None` is not `False`.** `True` seen, `False`
  the camera had a good look and there was nothing there, `None` the question
  could not be asked: nanoOWL not running, or a dark frame. `ok` says only that
  the question got asked, so `ok` with `held` False is a working camera
  reporting an empty gripper. Treating `None` as `False` turns "the detector is
  busy elsewhere" into "you dropped it".
- **It is a classification, not a detection.** The wrist camera is aimed about
  five degrees past its own grip point, so a held can reaches the frame only as
  a sliver and no detector will name it. nanoOWL is asked to choose between
  *an empty gripper* and *a gripper holding a soda can* instead, which needs no
  detectable object: measured, 0.99 holding and 0.93 empty.
- **It reads the WHOLE frame, so it means what it says at `carry` and nowhere
  else.** Carry points the tool well up and shows no floor, so the only can
  that can appear is the one being held. Asked from a pose that CAN see the
  floor, a can lying there reads as held at 1.000 — a confident wrong answer,
  not a weak one. See ARCHITECTURE.md, *Seeing the pick*.
- **It does not need the arm enabled**, since a camera and a detector are not
  `move_group` — but it does take the motion lock, so it comes back `busy`
  during a pick rather than photographing a moving arm.
- **A pick asks the same two questions itself**, at the standoff and at carry,
  and aborts on a definite no. It never aborts on `unknown`, so a pick on a
  machine with nanoOWL stopped runs exactly as it always did. `reset_arm()` is
  the way out of one that aborted after the grasp.
- **`pick_can()` and `place_can()` are two processes.** The carried object lives
  in the planning scene between them, and that is where `--place` reads it
  from — so a server restart between the two halves is survivable, and
  `reset_arm()` is what tidies up when something else is not.
