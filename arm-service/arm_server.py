#!/usr/bin/env python3
# coding: utf-8
"""
DOFBOT Arm XML-RPC Server (ROS 2 edition).

Exposes the dofbot_ros2 stack over XML-RPC so a remote client -- typically the
robot's higher-level brain on another machine -- can enable the arm and run
pick/place without knowing anything about ROS.

Every command is a `ros2` CLI invocation, deliberately: the ROS entry points in
dofbot_ctrl already own the hard parts (planning, IK, the planning scene), and
shelling out to them keeps this server a thin, restartable supervisor with no
rclpy state of its own to get out of sync. It runs in ONE ros2 process it keeps
alive (the launch), plus one short-lived process per motion command.

Commands:
  - enable_arm()            - start the pick_place stack (move_group + bridge)
  - disable_arm()           - shut that stack down
  - pick_can(x, y, z)       - pick the object at x,y,z and carry it
  - place_can()             - place what is being carried
  - move_to_state(name)     - move to a named state ('ready', 'init', ...)
  - reset_arm(state)        - recover: clear the scene, let go, go home
  - wave_arm(waves)         - wave hello, then stow
  - is_holding()            - look: is the object still in the jaws?
plus list_states(), stop(), tail_log(), get_status() and ping().

is_holding() is the odd one out: it needs no move_group, only the wrist camera
and the nanoOWL detector beside it, so it answers whether or not the arm is
enabled.

The arm must be enabled before any motion command; the motion commands are
serialized, and a second one arrives back as busy rather than queueing.

Server listens on 0.0.0.0:8001 by default. SIGTERM/SIGINT shut the launch down
before exiting, so `systemctl stop` does not orphan move_group.
"""

import argparse
import errno
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from socketserver import ThreadingMixIn
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler

DEFAULT_HOST = '0.0.0.0'
# 8001, the port the pre-ROS arm service used and no longer needs. NOT 8002 --
# that is supervisor-service's.
DEFAULT_PORT = 8001

# This file lives in jetson-nano-services, beside the other services, and
# drives a colcon workspace somewhere else entirely -- so the workspace is
# a setting, not something derivable from where this file sits. The startup
# script passes --workspace; DOFBOT_WS overrides the default for a direct
# run.
DEFAULT_WORKSPACE = os.environ.get(
    'DOFBOT_WS', os.path.expanduser('~/ros2_ws'))
DEFAULT_ROS_DISTRO = os.environ.get('ROS_DISTRO') or 'humble'

# Where the launch's own stdout goes. It is the only long-lived process here,
# and when a start fails this file is the evidence -- tail_log() serves it.
DEFAULT_LAUNCH_LOG = os.environ.get('DOFBOT_LAUNCH_LOG',
                                    '/tmp/dofbot_arm_launch.log')

# Nodes that must appear before enable_arm() calls the stack up. move_group is
# what plans; robot_state_publisher is what everything else reads the model
# from; moveit_bridge is the only thing that actually moves servos, so without
# it "enabled" would mean a simulation that silently does nothing.
CORE_NODES = ('/move_group', '/robot_state_publisher')
BRIDGE_NODE = '/moveit_bridge'

# move_group appears in the node list the moment it constructs its node, which
# is well before it has loaded the planning pipeline and is willing to be
# planned with. /move_action is what every motion command here actually talks
# to, so that action server -- not the node -- is the readiness signal.
MOVE_ACTION = '/move_action'

# Seconds. Picks are slow -- approach, grasp, lift and carry are each a planned
# move -- so the ceiling is generous; it exists to stop a wedged command from
# holding the motion lock forever, not to bound normal operation.
ENABLE_TIMEOUT = 90
PICK_TIMEOUT = 240
PLACE_TIMEOUT = 240
STATE_TIMEOUT = 120
RESET_TIMEOUT = 120

# A camera check is a frame grab plus one detector round trip -- under a second
# of real work. The ceiling is for the process around it: `ros2 run` has to
# start a Python interpreter and import the workspace, and a Jetson under load
# takes its time over that.
VISION_TIMEOUT = 60

# The line vision_check prints its answer on. It is the ONE line of that
# program's output meant to be parsed, and it is our own node's deliberate
# machine-readable output -- not MoveIt's console chatter, which is handed to a
# human unparsed and always will be.
VERDICT_PREFIX = 'VERDICT: '

# One wave is 3s of gesture, but the planned moves in and out are ordinary
# collision-checked moves and are the slow part. Generous: this exists to stop a
# wedged command holding the motion lock, not to bound normal operation.
WAVE_TIMEOUT = 180

# Command output is echoed back to the client. Keep the tail: MoveIt failures
# say what went wrong on the last few lines and log banners on the first few.
MAX_OUTPUT = 8000

_server = None
_service = None

# `ros2 launch` colours its output even when it is redirected to a file, and ESC
# is not a legal XML 1.0 character -- so an un-stripped log came back as a
# response the client could not parse at all ("not well-formed (invalid
# token)"). Strip the sequences for readability, then drop anything else XML
# cannot carry, because the text is whatever the nodes chose to print.
_ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
_NOT_XML = re.compile(
    '[^\t\n\r\u0020-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]')


def _printable(text):
    """Text that survives XML-RPC. See _ANSI above."""
    return _NOT_XML.sub('', _ANSI.sub('', text))


def _tail(text, limit=MAX_OUTPUT):
    if text is None:
        return ''
    text = _printable(text)
    if len(text) <= limit:
        return text
    return '...[%d bytes truncated]...\n' % (len(text) - limit) + text[-limit:]


def _result(ok, command='', returncode=-1, output='', error='', seconds=0.0):
    """The shape every command returns. XML-RPC has no None worth sending, so
    absent fields are empty strings rather than nil."""
    return {
        'ok': bool(ok),
        'command': command,
        'returncode': int(returncode),
        'output': _tail(output),
        'error': error or '',
        'seconds': round(float(seconds), 2),
    }


def _verdict(output):
    """The camera's answer, pulled out of vision_check's output.

    vision_check prints exactly one machine-readable line, `VERDICT: {json}`,
    and everything else it prints is prose for a human. Parsing that one line
    is not the thing the rest of this server refuses to do: the README's "hand
    `output` to a human, don't parse it" is about MoveIt's console, which owes
    us no format. This line is our own node's, versioned in the same tree, and
    written to be read here.

    The LAST such line wins, so a rerun inside one invocation cannot be
    answered by its first attempt. Anything unparseable comes back as the
    unknown answer rather than a guess -- a client that reads `held` as a
    boolean must see None, not False, when nothing was actually decided.
    """
    blank = {'held': None, 'verdict': 'unknown', 'reason': '', 'sighting': {}}
    for line in reversed((output or '').splitlines()):
        line = line.strip()
        if not line.startswith(VERDICT_PREFIX):
            continue
        try:
            data = json.loads(line[len(VERDICT_PREFIX):])
        except ValueError:
            break
        verdict = str(data.get('verdict') or 'unknown')
        return {
            'held': {'present': True, 'absent': False}.get(verdict),
            'verdict': verdict,
            'reason': str(data.get('reason') or ''),
            'sighting': data,
        }
    return dict(blank, reason='vision_check printed no verdict line')


def _b(value):
    """Python truth -> a launch-argument literal."""
    return 'true' if value else 'false'


def _group_empty(pgid):
    """True once no process is left in the group -- zombies included."""
    try:
        os.killpg(pgid, 0)
    except OSError as exc:
        return exc.errno == errno.ESRCH
    return False


def _kill_group(proc, first=signal.SIGINT, grace=15.0):
    """Signal a child's whole process group and wait for the group to empty.

    ros2 launch fans out into move_group, ros2_control_node, the spawners and
    moveit_bridge; killing only the parent leaves every one of them running and
    the next enable_arm() then has two of everything. Each child here is started
    with start_new_session=True precisely so the group can be addressed.

    Waiting on the parent is not enough either: `ros2 launch` returns from SIGINT
    a good few seconds before the nodes it spawned are actually gone, and a
    disable_arm() that answers during that window invites the caller to enable a
    second stack on top of the dying one. So this waits on the GROUP, reaping the
    parent as it goes -- an unreaped zombie is still a member of it.
    """
    if proc.poll() is not None:
        return proc.returncode
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None

    for sig, wait in ((first, grace), (signal.SIGTERM, 10.0),
                      (signal.SIGKILL, 5.0)):
        try:
            if pgid is None:
                proc.send_signal(sig)
            else:
                os.killpg(pgid, sig)
        except OSError as exc:
            if exc.errno != errno.ESRCH:
                raise

        deadline = time.time() + wait
        while time.time() < deadline:
            code = proc.poll()          # also reaps the parent
            if pgid is None:
                if code is not None:
                    return code
            elif _group_empty(pgid):
                return code
            time.sleep(0.25)
    return proc.poll()


class RosRunner:
    """Builds argv for `ros2 ...` calls, sourcing the overlays if we must.

    Under the supplied start script the environment is already sourced and the
    argv is used as-is. Run the server by hand from an unsourced shell (or from
    a systemd unit that lost the environment) and it falls back to wrapping each
    call in a bash that sources ROS and the workspace first -- correct either
    way, just half a second slower per call.
    """

    def __init__(self, workspace=DEFAULT_WORKSPACE, distro=DEFAULT_ROS_DISTRO):
        self.workspace = workspace
        self.distro = distro
        self.install = os.path.join(workspace, 'install')
        self.setup_files = [p for p in (
            '/opt/ros/%s/setup.bash' % distro,
            os.path.join(self.install, 'setup.bash'),
        ) if os.path.exists(p)]

    def sourced(self):
        """True when this process already has the workspace overlay."""
        return (shutil.which('ros2') is not None
                and self.install in os.environ.get('AMENT_PREFIX_PATH', ''))

    def argv(self, args):
        args = [str(a) for a in args]
        if self.sourced():
            return args
        script = ''.join('. %s\n' % shlex.quote(f) for f in self.setup_files)
        # exec "$@" rather than an interpolated command line: the arguments stay
        # a real argv and never go back through the shell's word splitting.
        return ['bash', '-c', script + 'exec "$@"', 'dofbot-arm'] + args

    def describe(self):
        return {
            'workspace': self.workspace,
            'ros_distro': self.distro,
            'sourced_in_process': self.sourced(),
            'setup_files': list(self.setup_files),
        }


class ArmService:
    """The XML-RPC surface. Owns the launch process and the motion lock."""

    def __init__(self, runner, launch_log=DEFAULT_LAUNCH_LOG):
        self._runner = runner
        self._launch_log = launch_log
        self._launch = None
        self._launch_started = 0.0
        self._launch_argv = []
        self._launch_fh = None
        self._launch_lock = threading.Lock()
        # Non-reentrant and never blocked on: a motion request that arrives
        # while another is running is answered 'busy', because a queued arm
        # command executes against a world that has moved on since it was sent.
        self._motion = threading.Lock()
        self._task = None
        self._last = None
        self._states = None
        self._closed = False

    # ---------------------------------------------------------------- helpers

    def _launch_alive(self):
        return self._launch is not None and self._launch.poll() is None

    def _run(self, command, args, timeout):
        """Run one short-lived ros2 command under the motion lock."""
        if not self._motion.acquire(blocking=False):
            running = self._task
            busy = ('busy: %s has been running for %.0fs'
                    % (running['name'], time.time() - running['started'])
                    if running else 'busy')
            return _result(False, command, error=busy)
        started = time.time()
        try:
            argv = self._runner.argv(args)
            print('[%s] %s' % (command, ' '.join(shlex.quote(a) for a in argv)),
                  flush=True)
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, start_new_session=True)
            self._task = {'name': command, 'pid': proc.pid, 'started': started,
                          'proc': proc}
            timed_out = False
            try:
                output = proc.communicate(timeout=timeout)[0]
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_group(proc)
                try:
                    output = proc.communicate(timeout=10)[0]
                except subprocess.TimeoutExpired:
                    # A surviving grandchild still holds the pipe. The command
                    # is dead either way; report it without hanging the caller.
                    output = '(output unavailable: the pipe stayed open)'
            elapsed = time.time() - started

            if timed_out:
                result = _result(False, command, proc.returncode or -1, output,
                                 'timed out after %.0fs' % timeout, elapsed)
            elif proc.returncode == 0:
                result = _result(True, command, 0, output, '', elapsed)
            else:
                result = _result(False, command, proc.returncode, output,
                                 '%s exited %d' % (command, proc.returncode),
                                 elapsed)
            self._last = dict(result, finished=time.time())
            print('[%s] %s in %.1fs (rc=%s)'
                  % (command, 'ok' if result['ok'] else result['error'],
                     elapsed, result['returncode']), flush=True)
            return result
        except OSError as exc:
            return _result(False, command, error='could not run %s: %s'
                                                 % (command, exc),
                           seconds=time.time() - started)
        finally:
            self._task = None
            self._motion.release()

    def _require_enabled(self, command):
        """None when the stack is up, else the failure to hand back."""
        if self._launch_alive():
            return None
        return _result(False, command,
                       error='arm is not enabled -- call enable_arm() first')

    def _graph_list(self, what, timeout=15):
        """`ros2 <what> list` as a set. Empty if the call fails -- callers
        poll, so a flaky discovery round just costs another iteration."""
        try:
            out = subprocess.run(self._runner.argv(['ros2', what, 'list']),
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL,
                                 text=True, timeout=timeout).stdout
        except (subprocess.TimeoutExpired, OSError):
            return set()
        out = _printable(out)
        return {line.strip() for line in out.splitlines() if line.strip()}

    def _wait_ready(self, wanted, timeout):
        """Poll the graph until every wanted node is up. (ok, detail)."""
        deadline = time.time() + timeout
        missing = set(wanted)
        while time.time() < deadline:
            if not self._launch_alive():
                return False, ('launch exited with code %s before the stack '
                               'came up' % self._launch.returncode)
            missing = set(wanted) - self._graph_list('node')
            if not missing and MOVE_ACTION in self._graph_list('action'):
                return True, ('stack up: %s, %s'
                              % (', '.join(sorted(wanted)), MOVE_ACTION))
            if not missing:
                missing = {MOVE_ACTION}
            time.sleep(2.0)
        return False, ('timed out after %ds waiting for %s'
                       % (timeout, ', '.join(sorted(missing))))

    def _stop_launch(self):
        """Bring the stack down. Returns a human-readable outcome."""
        with self._launch_lock:
            if self._launch is None:
                return 'arm was not enabled'
            if self._launch.poll() is not None:
                detail = 'launch had already exited (code %s)' % self._launch.returncode
            else:
                code = _kill_group(self._launch)
                detail = 'stack stopped (launch exit code %s)' % code
            self._launch = None
            self._launch_argv = []
            self._launch_started = 0.0
            if self._launch_fh is not None:
                try:
                    self._launch_fh.close()
                except OSError:
                    pass
                self._launch_fh = None
            return detail

    # --------------------------------------------------------------- commands

    def enable_arm(self, timeout=ENABLE_TIMEOUT, bridge=True, rviz=False,
                   port=''):
        """Start the pick_place stack and wait for it to come up.

        Runs `ros2 launch dofbot_ctrl pick_place.launch.py rviz:=false`, which
        is move_group, ros2_control on mock joints, robot_state_publisher and
        moveit_bridge. The launch stays running until disable_arm().

        Args:
            timeout: seconds to wait for the nodes to appear.
            bridge:  run moveit_bridge, i.e. actually drive the servos. False
                     is simulation only and touches no serial port.
            rviz:    start RViz on the robot. Normally false -- run RViz on a
                     laptop with `ros2 launch dofbot_moveit moveit_rviz.launch.py`.
            port:    serial port override, '' for the launch default.

        Returns a result dict. Enabling twice is not an error; the second call
        reports the stack that is already up.

        moveit_bridge needs exclusive access to the serial port: stop
        gui_teleop and joint_state_mirror before calling this.
        """
        with self._launch_lock:
            if self._launch_alive():
                return _result(True, 'enable_arm', 0,
                               'already enabled (pid %d, up %.0fs)'
                               % (self._launch.pid,
                                  time.time() - self._launch_started))
            args = ['ros2', 'launch', 'dofbot_ctrl', 'pick_place.launch.py',
                    'rviz:=%s' % _b(rviz), 'bridge:=%s' % _b(bridge)]
            if port:
                args.append('port:=%s' % port)
            argv = self._runner.argv(args)
            print('[enable_arm] %s' % ' '.join(shlex.quote(a) for a in argv),
                  flush=True)
            try:
                # Truncated per launch: the log is evidence for THIS start, and
                # a stale tail from the previous one is worse than no tail.
                self._launch_fh = open(self._launch_log, 'w')
                self._launch_fh.write('=== %s: %s\n'
                                      % (time.strftime('%Y-%m-%d %H:%M:%S'),
                                         ' '.join(args)))
                self._launch_fh.flush()
                self._launch = subprocess.Popen(
                    argv, stdout=self._launch_fh, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                    # Belt and braces with _printable: this keeps the escapes
                    # out of the log file too, where they are just noise.
                    env=dict(os.environ, RCUTILS_COLORIZED_OUTPUT='0'))
            except OSError as exc:
                self._launch = None
                return _result(False, 'enable_arm',
                               error='could not start the launch: %s' % exc)
            self._launch_argv = args
            self._launch_started = time.time()

        wanted = list(CORE_NODES) + ([BRIDGE_NODE] if bridge else [])
        started = time.time()
        ok, detail = self._wait_ready(wanted, timeout)
        if not ok:
            # Do not leave half a stack up. A second move_group joining the
            # domain on the next attempt is a much nastier failure than this
            # one: two action servers, CONTROL_FAILED, an arm driven twice.
            self._stop_launch()
            return _result(False, 'enable_arm', error=detail,
                           output=self.tail_log(40),
                           seconds=time.time() - started)
        return _result(True, 'enable_arm', 0, detail,
                       seconds=time.time() - started)

    def disable_arm(self, park=''):
        """Shut the pick_place stack down.

        Args:
            park: optionally move to this named state first (e.g. 'init'),
                  while the stack is still up to plan the move. Ignored if the
                  arm is not enabled; a failed park does not stop the shutdown.

        The servos hold their last position after the nodes exit -- this cuts
        control, not torque.
        """
        parked = ''
        if park and self._launch_alive():
            result = self.move_to_state(park)
            parked = ('parked at %r; ' % park if result['ok']
                      else 'park at %r failed (%s); ' % (park, result['error']))
        return _result(True, 'disable_arm', 0, parked + self._stop_launch())

    def pick_can(self, x, y, z, object='', timeout=PICK_TIMEOUT):
        """Pick the object centred at (x, y, z) and carry it.

        Runs `ros2 run dofbot_ctrl pick_place -- --pick x y z`: approach,
        grasp, attach, lift and carry, stopping short of the place. Call
        place_can() to finish, or move_to_state() to go somewhere else first.

        Args:
            x, y, z: the object's CENTRE in base_link, metres. For something
                     sitting on the floor that is half its height, not zero.
            object:  catalogue entry to grasp as; '' uses the node's default.
            timeout: seconds before the command is killed.
        """
        blocked = self._require_enabled('pick_can')
        if blocked:
            return blocked
        args = ['ros2', 'run', 'dofbot_ctrl', 'pick_place', '--', '--pick']
        if object:
            args += ['--object', str(object)]
        args += ['%g' % float(x), '%g' % float(y), '%g' % float(z)]
        return self._run('pick_can', args, timeout)

    def place_can(self, timeout=PLACE_TIMEOUT):
        """Place whatever the gripper is carrying.

        Runs `ros2 run dofbot_ctrl pick_place -- --place`, the second half of
        the sequence pick_can() stops in the middle of: over the bin, open,
        detach, back to carry.

        Where the bin is is a named state ('over_trash') for now, nominally
        straight in front of the robot. When perception starts finding the bin
        this grows the coordinates pick_can() already takes.
        """
        blocked = self._require_enabled('place_can')
        if blocked:
            return blocked
        args = ['ros2', 'run', 'dofbot_ctrl', 'pick_place', '--', '--place']
        return self._run('place_can', args, timeout)

    def move_to_state(self, name, timeout=STATE_TIMEOUT):
        """Move to one of the saved states -- see list_states().

        Runs `ros2 run dofbot_ctrl move_to_state NAME`: a collision-checked
        joint-space plan, the same one RViz's MoveIt panel makes.

        This does not touch the planning scene. A move that fails instantly
        after an aborted pick usually means the scene still holds the object;
        that is what `pick_place --reset` is for.
        """
        name = str(name).strip()
        states = self.list_states()
        if states and name not in states:
            return _result(False, 'move_to_state',
                           error='unknown state %r; have %s'
                                 % (name, ', '.join(states)))
        blocked = self._require_enabled('move_to_state')
        if blocked:
            return blocked
        args = ['ros2', 'run', 'dofbot_ctrl', 'move_to_state', '--', name]
        return self._run('move_to_state', args, timeout)

    def reset_arm(self, state='ready', force=False, timeout=RESET_TIMEOUT):
        """Recover from a run that died partway: clear the scene, let go, go home.

        Runs `ros2 run dofbot_ctrl pick_place -- --reset STATE`. This is the
        only command here that touches the planning scene, and the only way out
        of the wedged state a failed pick leaves behind: the object stays in the
        scene -- attached to the gripper if the run died after the grasp -- and
        every move after that fails instantly with INVALID_MOTION_PLAN, because
        the plan's own first waypoint is inside something. move_to_state cannot
        fix that; it does not clear the scene.

        Opening the gripper DROPS whatever is held, where it is, before the arm
        moves. That is the intent -- carrying it to `state` first would drop it
        from up there instead.

        Args:
            state: where to leave the arm afterwards (default 'ready').
            force: if MoveIt will not plan out of where the arm is, drive out
                   blind -- joint interpolation, NOT COLLISION CHECKED. Watch
                   the arm and have the power switch to hand.
        """
        state = str(state).strip() or 'ready'
        states = self.list_states()
        if states and state not in states:
            return _result(False, 'reset_arm',
                           error='unknown state %r; have %s'
                                 % (state, ', '.join(states)))
        blocked = self._require_enabled('reset_arm')
        if blocked:
            return blocked
        args = ['ros2', 'run', 'dofbot_ctrl', 'pick_place', '--',
                '--reset', state]
        if force:
            args.append('--force')
        return self._run('reset_arm', args, timeout)

    def wave_arm(self, waves=1, finish='', seconds=0.0,
                 timeout=WAVE_TIMEOUT):
        """Wave hello, then stow the arm.

        Runs `ros2 run dofbot_ctrl wave_arm`: move_group plans the way into the
        gesture, the swings themselves go to the arm controller as one timed
        trajectory, and moveit_bridge drives the servos throughout. Like every
        other command here it needs the arm enabled, and it is collision-checked
        against the live planning scene -- a wave will refuse rather than sweep
        the arm through something the scene knows about.

        Args:
            waves:   back-and-forth swings (default 1).
            finish:  named state to stow at afterwards; '' uses the node's
                     default ('init').
            seconds: how long ONE wave takes, so the pace holds whatever `waves`
                     says; 0 uses the node's default (3 s).
        """
        try:
            waves = int(waves)
            seconds = float(seconds)
        except (TypeError, ValueError):
            return _result(False, 'wave_arm',
                           error='waves and seconds must be numbers')
        if waves < 1:
            return _result(False, 'wave_arm', error='waves must be at least 1')
        if seconds < 0:
            # 0 is 'use the node's default'; a negative would silently become
            # that too, and the caller would never learn it asked for nonsense.
            return _result(False, 'wave_arm',
                           error='seconds cannot be negative')
        finish = str(finish).strip()
        if finish:
            # Only when one was asked for: '' means the node's own default, and
            # checking it would cost a subprocess to learn nothing.
            states = self.list_states()
            if states and finish not in states:
                return _result(False, 'wave_arm',
                               error='unknown state %r; have %s'
                                     % (finish, ', '.join(states)))
        blocked = self._require_enabled('wave_arm')
        if blocked:
            return blocked
        args = ['ros2', 'run', 'dofbot_ctrl', 'wave_arm', '--',
                '--waves', str(waves)]
        if finish:
            args += ['--finish', finish]
        if seconds > 0:
            args += ['--seconds', '%g' % seconds]
        return self._run('wave_arm', args, timeout)

    def is_holding(self, object='', timeout=VISION_TIMEOUT):
        """Is the gripper holding the object? Looks, rather than remembers.

        Runs `ros2 run dofbot_ctrl vision_check -- --held`: a frame from the
        wrist camera, put to nanoOWL, judged against the pixel region a held
        object appears in. It is a LIVE look, not a record of what the last
        pick did -- the point of the question is the case where the two
        disagree, a can that was picked and has since slipped out.

        THREE ANSWERS, and `held` is the one to read:

            held True   an object was seen in the jaws
            held False  the camera had a good look and there was nothing there
            held None   the question could not be asked -- nanoOWL not running,
                        a dark frame, or the pixel region not yet calibrated.
                        NOT the same as False, and must not be treated as it

        `ok` says whether the question was ASKED, not what the answer
        was: a can that is not there is a successful query. `verdict` carries the same
        three states as text and `reason` says why in English.

        Meaningful at 'carry', which is where a pick leaves the arm and where
        the pixel region was measured. Asked with the arm somewhere else it is
        answering about a view nobody calibrated, and will mostly say no.

        Does not need the arm enabled -- a camera and a detector are not
        move_group -- but it does take the motion lock, because a frame grabbed
        halfway through a moving arm answers about nowhere in particular.

        Args:
            object:  catalogue entry to look for; '' means the node's default.
            timeout: seconds before the command is killed.
        """
        args = ['ros2', 'run', 'dofbot_ctrl', 'vision_check', '--', '--held']
        if object:
            args += ['--object', str(object)]
        result = self._run('is_holding', args, timeout)
        return dict(result, **_verdict(result['output']))

    def list_states(self):
        """The saved state names, read from the node itself and cached.

        Asks `move_to_state --list`, which answers without connecting to
        move_group, so this works whether or not the arm is enabled. Returns []
        if the workspace cannot be reached -- the caller should then let the
        real command report the error rather than guessing.
        """
        if self._states is not None:
            return list(self._states)
        try:
            out = subprocess.run(
                self._runner.argv(['ros2', 'run', 'dofbot_ctrl',
                                   'move_to_state', '--', '--list']),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, timeout=60).stdout
        except (subprocess.TimeoutExpired, OSError):
            return []
        names = []
        for line in _printable(out).splitlines():
            fields = line.split()
            # The first column is the state name; the header row starts with
            # the literal 'state', and every other column is a number.
            if fields and fields[0] != 'state' and not fields[0][0].isdigit():
                names.append(fields[0])
        if names:
            self._states = names
        return list(names)

    def stop(self):
        """Abort the motion command in flight, if any. Leaves the stack up.

        SIGINT first, so the node gets to unwind; escalating only if it will
        not. The arm stops where the last waypoint it was already executing
        leaves it -- this cancels the command, not the servos.
        """
        task = self._task
        if task is None:
            return _result(True, 'stop', 0, 'nothing running')
        code = _kill_group(task['proc'], grace=5.0)
        return _result(True, 'stop', 0,
                       'stopped %s (pid %d, exit %s)'
                       % (task['name'], task['pid'], code))

    def tail_log(self, lines=40):
        """The last `lines` of the launch's output -- why enable_arm failed."""
        try:
            with open(self._launch_log, 'r', errors='replace') as fh:
                return _tail(''.join(fh.readlines()[-int(lines):]))
        except OSError as exc:
            return 'could not read %s: %s' % (self._launch_log, exc)

    def get_status(self):
        """A snapshot: whether the arm is enabled, and what it is doing."""
        task = self._task
        status = {
            'arm_enabled': self._launch_alive(),
            'launch_pid': self._launch.pid if self._launch_alive() else 0,
            'launch_uptime': (round(time.time() - self._launch_started, 1)
                              if self._launch_alive() else 0.0),
            'launch_args': list(self._launch_argv),
            'launch_log': self._launch_log,
            'busy': task is not None,
            'running': task['name'] if task else '',
            'running_for': round(time.time() - task['started'], 1) if task else 0.0,
            'states': self.list_states(),
        }
        status.update(self._runner.describe())
        if self._last is not None:
            status['last'] = {k: self._last[k]
                              for k in ('command', 'ok', 'returncode',
                                        'error', 'seconds')}
        return status

    def ping(self):
        return 'pong'

    def close(self):
        """Called on shutdown. Never leave move_group orphaned.

        Idempotent: the signal handler and the serve_forever teardown both
        reach here on the way out.
        """
        if self._closed:
            return
        self._closed = True
        task = self._task
        if task is not None:
            _kill_group(task['proc'], grace=5.0)
        print(self._stop_launch(), flush=True)


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2', '/')


class ThreadingXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    """Threaded so ping/get_status/stop still answer during a 4-minute pick.

    Concurrency the arm cares about is handled a layer down, by the motion
    lock; this only stops the socket from being the bottleneck.
    """
    daemon_threads = True
    allow_reuse_address = True


def signal_handler(signum, frame):
    print('\nReceived %s. Shutting down...' % signal.Signals(signum).name,
          flush=True)
    if _service is not None:
        _service.close()
    if _server is not None:
        # shutdown() blocks until serve_forever returns, and serve_forever is
        # this very thread -- so it has to be asked from another one.
        threading.Thread(target=_server.shutdown).start()
    sys.exit(0)


METHOD_HELP = [
    ('enable_arm(timeout, bridge, rviz, port)', 'Start the pick_place stack'),
    ('disable_arm(park)', 'Stop the stack, optionally parking first'),
    ('pick_can(x, y, z, object)', 'Pick the object at x,y,z and carry it'),
    ('place_can()', 'Place what the gripper is carrying'),
    ('move_to_state(name)', 'Move to a saved state (ready, init, carry, ...)'),
    ('reset_arm(state, force)', 'Clear the scene, let go, go home'),
    ('wave_arm(waves, finish, seconds)', 'Wave hello, then stow'),
    ('list_states()', 'Names move_to_state accepts'),
    ('stop()', 'Abort the motion in flight; leave the stack up'),
    ('tail_log(lines)', "Tail the launch's output"),
    ('get_status()', 'Enabled? busy? what ran last?'),
    ('ping()', 'Check server responsiveness'),
]


def main():
    global _server, _service

    parser = argparse.ArgumentParser(
        prog='arm_server', description=__doc__.split('\n\n')[0])
    parser.add_argument('--host', default=DEFAULT_HOST)
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--workspace', default=DEFAULT_WORKSPACE,
                        help='colcon workspace holding dofbot_ctrl '
                             '(default: %(default)s)')
    parser.add_argument('--ros-distro', default=DEFAULT_ROS_DISTRO)
    parser.add_argument('--launch-log', default=DEFAULT_LAUNCH_LOG)
    cli = parser.parse_args()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    runner = RosRunner(workspace=cli.workspace, distro=cli.ros_distro)
    _service = ArmService(runner, launch_log=cli.launch_log)
    _server = ThreadingXMLRPCServer((cli.host, cli.port),
                                    requestHandler=RequestHandler,
                                    allow_none=True, logRequests=True)
    _server.register_instance(_service)
    _server.register_introspection_functions()

    print('DOFBOT Arm XML-RPC Server', flush=True)
    print('Listening on %s:%d' % (cli.host, cli.port), flush=True)
    print('PID: %d' % os.getpid(), flush=True)
    print('Workspace: %s (ROS %s, %s)'
          % (runner.workspace, runner.distro,
             'sourced' if runner.sourced() else 'sourcing per command'),
          flush=True)
    print('Launch log: %s' % cli.launch_log, flush=True)
    print(flush=True)
    print('Available methods:', flush=True)
    for name, blurb in METHOD_HELP:
        print('  - %-42s - %s' % (name, blurb), flush=True)
    print(flush=True)
    print('Server ready. SIGTERM/SIGINT stop the arm stack before exiting.',
          flush=True)

    try:
        _server.serve_forever()
    except Exception as exc:
        print('Server error: %s' % exc, flush=True)
    finally:
        _service.close()
        print('Server stopped.', flush=True)


if __name__ == '__main__':
    main()
