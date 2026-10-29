#!/usr/bin/env python3
"""Standalone one- or two-arm ALOHA teleoperation without data recording. Pass "left", "right",
or "dual" (both arms at once, sharing one keyboard) as the `side` argument.

Keys (read from the terminal this runs in):
    RIGHT arrow  start, and resume after a pause
    LEFT arrow   pause AND lock - the follower(s) freeze, haptics drop to 0 PWM, and every
                 leader is re-torqued rigid at once (session.lock())
    b            re-zero the tactile baseline(s) (do it with nothing grasped)
    ESC or q     stop AND lock immediately (every leader re-torqued rigid); after one second
                 every arm walks to its sleep position and torque cuts, with no further
                 confirmation. See shutdown_and_sleep().

With tactile force feedback on (default), each leader gripper is driven in PWM mode so that
puppet-side contact force pushes back on the operator's hand. See GripperForceFeedback.

===========================================================================================
CHANGES since the pre-force-feedback version (kept alongside as "one_side_teleop copy.py")
===========================================================================================

1. Start/stop is on the keyboard, not the gripper
   press_to_start() (squeeze the leader gripper past a threshold) is gone, replaced by
   KeyboardControl + wait_for_start(): RIGHT starts, LEFT pauses and locks, ESC/q stops and
   powers down, b re-zeroes. Reads the tty in cbreak mode - not raw - so Ctrl+C still works,
   and needs no X server. Dropped with it: --skip_start_wait, --start_trigger_ratio,
   --puppet_start_ratio, and --torque_on_exit - a paused or stopped leader is always torqued.

1b. --side dual and shutdown_and_sleep()
   `side` also accepts "dual": both arms run in the same process, one ArmSession per side,
   driven by one shared teleop_loop() and one shared KeyboardControl so LEFT/RIGHT/ESC need
   pressing only once for both. ESC/q no longer quits outright - it stops the loop and locks
   every leader rigid immediately (so nothing is ever torque-free while waiting), pauses one
   second, then walks every arm through sleep_plus.py's own two-stage shutdown sequence (old
   way-point, then the measured rest pose - see that file for why not straight to either pose,
   and why the target is a measured rest pose rather than the arm model's factory one),
   with no confirmation prompt. This exists because going straight to a single "sleep" target,
   or cutting torque before the arm actually arrived, both let an arm drop hard under its own
   weight - see sleep_plus.py's module comment for the full story. LEFT is the separate control
   for "I need to let go for a moment but I'm not done": it pauses AND locks (leader rigid)
   without touching the sleep position at all; RIGHT hands control back.

2. Both grippers park at one mapped pose
   prep_robots() puts the master at --start_open_ratio (default 1.0, fully open) and the
   puppet at exactly what that maps to through MASTER2PUPPET_*_JOINT_FN, so the two already
   agree and nothing jumps when teleoperation begins.

3. Tactile force feedback on the leader gripper (the bulk of the new code)
   GripperForceFeedback subscribes to /tactile/<side>_gripper_outer/raw (the pickled bundle
   from 4arms_tactile_teleop.launch, needs extra_outputs containing "force") and maps
   frames["force"][..., 2] - the normal component - onto a leader-gripper Goal_PWM.
   The leader gripper motor is an XC430-W150-T: PWM mode only, no current control, and
   interbotix's own helper opens with a positive command, so positive PWM is the direction
   that resists the operator closing.

   The mapping is a wall, not a force follower (--feedback_mode wall):
       contact latches an anchor at the leader gripper's current position
       depth     = max(0, anchor - position)
       stiffness = k_min + (k_max - k_min) * x**curve      (x = normalised force)
       pwm       = min(stiffness * depth, max_pwm)
   At the anchor the command is 0, so the feedback can only resist further closing and never
   drives the handle open. Commanding force directly (--feedback_mode force, kept for
   comparison) pushes the leader open, which opens the follower, which drops the force, which
   lets the operator squeeze again - a limit cycle felt as the gripper juddering back.

   Guards around it, each one there for a symptom seen on the bench:
     - capture_baseline() averages the resting reading at start (and on 'b'), so a sensor
       offset cannot show up as constant resistance
     - --force_ramp_s fades the feedback in from 0 at every start and resume
     - --force_curve > 1 keeps light contact nearly free
     - stale tactile data or a pause commands 0 PWM, never a frozen command
     - contact releases with hysteresis (--force_deadband * DEFAULT_CONTACT_RELEASE_FRACTION)
     - the command is zeroed at the gripper's own travel stops

4. Torque gating on the leader gripper (MasterGripperEffort)
   A Goal_PWM of 0 is not a free motor: torque stays on and the driver brakes the windings,
   which the operator feels as constant drag. Torque is dropped once the command has been 0
   for DEFAULT_TORQUE_GATE_DELAY_S and restored before the next non-zero command.
   --no_torque_gate keeps the old always-torqued behaviour.

5. Closing speed brake (ClosingSpeedBrake)
   A fast snap of the leader carried the follower past the object on inertia. This resists
   only the excess above --close_speed_limit, and only while closing:
       pwm = min(gain * max(0, closing_speed - limit), cap)
   Below the limit it is exactly 0, so a normal close is untouched - deliberately not a
   damper, which would drag at every speed in both directions. --close_brake_gain 0 disables.
   Needs no tactile data; it works during a free approach.

6. Pause/resume blends instead of snapping
   The operator can move the leader while paused, so resuming interpolates the follower from
   where it froze onto the leader over --resume_blend_s. The same blend runs at startup.

7. Status line
   Once a second: contact state, raw and baseline-corrected force, wall depth, closing speed
   with its one-second peak (use it to calibrate --close_speed_limit), brake PWM, total PWM
   and tactile age.

Tuning order that worked here: leave --wall_stiffness_min at 1500 (it sets how crisp first
contact feels), tune --close_brake_gain against the printed vclose peak, and only then touch
--max_feedback_pwm. Every knob is a CLI flag; the defaults in this file are the settings that
were tuned on the bench.
"""

import argparse
import os
import pickle
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import rospy
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from std_msgs.msg import UInt8MultiArray

# Dual-mode: works both run directly (python aloha_scripts/one_side_teleop.py ..., aloha_scripts/
# itself on sys.path, flat imports resolve) and imported as aloha_scripts.one_side_teleop from a
# repo-root script (e.g. imitate_episodes_multi_gpu.py), where only the dotted form resolves -
# same pattern already used by sleep_plus.py and visualize_episodes_multi.py.
try:
    from aloha_scripts.constants import DT, START_ARM_POSE
    from aloha_scripts.constants import MASTER2PUPPET_LEFT_JOINT_FN, MASTER2PUPPET_RIGHT_JOINT_FN
    from aloha_scripts.constants import MASTER_LEFT_GRIPPER_JOINT_CLOSE, MASTER_LEFT_GRIPPER_JOINT_OPEN
    from aloha_scripts.constants import MASTER_RIGHT_GRIPPER_JOINT_CLOSE, MASTER_RIGHT_GRIPPER_JOINT_OPEN
    from aloha_scripts.constants import PUPPET_LEFT_GRIPPER_JOINT_CLOSE, PUPPET_LEFT_GRIPPER_JOINT_OPEN
    from aloha_scripts.constants import PUPPET_RIGHT_GRIPPER_JOINT_CLOSE, PUPPET_RIGHT_GRIPPER_JOINT_OPEN
    from aloha_scripts.robot_utils import torque_on, torque_off, move_arms, move_grippers, get_arm_gripper_positions
    # Sleep-position constants and the two-stage, correctly-timed shutdown move: shared with
    # sleep_plus.py so a change to either the measured rest pose or the shutdown speed only has to
    # be made once. See sleep_plus.py's own module comment for why these two stages exist (a hard
    # drop was observed going straight to a single "sleep" target) and why the target is a measured
    # resting pose rather than each arm model's factory-default one (inaccurate for this particular
    # pair of arms - added tactile hardware shifts the balance point).
    from aloha_scripts.sleep_plus import (
        OLD_PUPPET_SLEEP_POSITION, OLD_MASTER_SLEEP_POSITION,
        PUPPET_SLEEP_POSITION, MASTER_SLEEP_POSITION,
        SHUTDOWN_SPEED_SCALE, move_shut_robot,
    )
except ImportError:
    from constants import DT, START_ARM_POSE
    from constants import MASTER2PUPPET_LEFT_JOINT_FN, MASTER2PUPPET_RIGHT_JOINT_FN
    from constants import MASTER_LEFT_GRIPPER_JOINT_CLOSE, MASTER_LEFT_GRIPPER_JOINT_OPEN
    from constants import MASTER_RIGHT_GRIPPER_JOINT_CLOSE, MASTER_RIGHT_GRIPPER_JOINT_OPEN
    from constants import PUPPET_LEFT_GRIPPER_JOINT_CLOSE, PUPPET_LEFT_GRIPPER_JOINT_OPEN
    from constants import PUPPET_RIGHT_GRIPPER_JOINT_CLOSE, PUPPET_RIGHT_GRIPPER_JOINT_OPEN
    from robot_utils import torque_on, torque_off, move_arms, move_grippers, get_arm_gripper_positions
    from sleep_plus import (
        OLD_PUPPET_SLEEP_POSITION, OLD_MASTER_SLEEP_POSITION,
        PUPPET_SLEEP_POSITION, MASTER_SLEEP_POSITION,
        SHUTDOWN_SPEED_SCALE, move_shut_robot,
    )

DEFAULT_ARM_COMMAND_MOVE_TIME = DT
DEFAULT_PREP_ARM_MOVE_TIME = 2.0
DEFAULT_PREP_ARM_ACCEL_TIME = 0.3
DEFAULT_GRIPPER_STALL_ERROR = 0.04
DEFAULT_GRIPPER_STALL_MOTION = 0.001
DEFAULT_GRIPPER_STALL_CYCLES = 8
DEFAULT_GRIPPER_RELEASE_MARGIN = 0.05
DEFAULT_PUPPET_GRIPPER_CURRENT_LIMIT = 300
# Gripper start and trigger poses, as a fraction of the master gripper travel
# (0.0 = fully closed, 1.0 = fully open). The grippers park at the start ratio and
# teleoperation begins once the master is squeezed past the trigger ratio, so the gap
# between the two is how far you have to squeeze.
DEFAULT_START_OPEN_RATIO = 1.0
# Where the puppet gripper parks, on the same scale. Set it equal to the trigger ratio for a
# seamless handover.
DEFAULT_RESUME_BLEND_S = 1.5

# ---- leader-gripper force feedback -------------------------------------------------
# The leader gripper motor is an XC430-W150-T. In PWM operating mode it takes Goal_PWM in
# [-885, 885]; interbotix's own gripper helper opens with a positive command and closes with a
# negative one (interbotix_xs_modules/gripper.py: open() -> +value, close() -> -value), so a
# POSITIVE PWM is the direction that pushes the leader gripper open, i.e. resists the operator
# squeezing it further. That is the direction this feedback always commands.
XC430_PWM_LIMIT = 885
DEFAULT_TACTILE_TOPIC_TEMPLATE = "/tactile/{side}_gripper_outer/raw"
# Which field of the tactile bundle the feedback reads, and which of its channels. In "force"
# (35, 20, 3) channel 2 is the normal component and channels 0/1 are the two shear components.
# The SDK also publishes "force_norm", its own normal-only copy of the same quantity - measured
# over 30 frames the two agree exactly on most frames but not all, so they are computed
# separately and only one of them should be read. Its unit is whatever the XENSE SDK returns;
# at rest the whole-field |sum| sits around 0.01-0.02, hence the deadband below.
DEFAULT_FORCE_FIELD = "force"
FORCE_NORMAL_CHANNEL = 2
DEFAULT_FORCE_DEADBAND = 0.05
# Force that saturates the feedback at max_pwm, and the shape of the ramp up to it. An exponent
# above 1 keeps the onset soft, so light contact is barely felt and only a real grasp pushes back.
DEFAULT_FORCE_FULL_SCALE = 0.6
DEFAULT_FORCE_CURVE = 2.0
DEFAULT_MAX_FEEDBACK_PWM = 250
DEFAULT_FORCE_TIMEOUT_S = 0.3
DEFAULT_FORCE_FILTER = 0.3
# Resting offset removal: average this many frames right after the feedback goes live (nothing
# grasped yet) and subtract it, so a sensor bias never becomes constant resistance.
DEFAULT_FORCE_BASELINE_SAMPLES = 30
# Seconds spent fading the feedback in from 0 after start or resume.
DEFAULT_FORCE_RAMP_S = 1.0
# Contact is dropped again once the net force falls below this fraction of the deadband, so a
# reading hovering around the threshold cannot chatter the wall on and off.
DEFAULT_CONTACT_RELEASE_FRACTION = 0.5
# "wall" mode stiffness in PWM per radian of leader-gripper travel past the contact anchor, at
# just-detected contact and at full-scale force. The leader gripper's whole travel is ~0.87 rad,
# so 4000 PWM/rad reaches the 250 PWM cap after squeezing ~0.06 rad past the anchor.
DEFAULT_WALL_STIFFNESS_MIN = 1500.0
DEFAULT_WALL_STIFFNESS_MAX = 6000.0
# A Goal_PWM of 0 still brakes the motor, so torque is dropped after the command has been 0 this
# long - short enough to feel immediate, long enough that a flickering command cannot chatter it.
DEFAULT_TORQUE_GATE_DELAY_S = 0.25
# Soft speed governor on the leader gripper. Closing slower than the limit is completely free -
# this is not viscous damping, which would drag at every speed and in both directions. Only the
# excess above the limit is resisted, which is what keeps a fast snap from carrying the follower
# past the object on inertia while leaving a normal, deliberate close untouched.
DEFAULT_CLOSE_SPEED_LIMIT = 3.5 # 2.0 modified by gnq
DEFAULT_CLOSE_BRAKE_GAIN = 120.0
DEFAULT_CLOSE_BRAKE_MAX_PWM = 400
# The velocity register is quantised at 0.229 rpm = 0.024 rad/s per count, and is noisier than
# position, so it gets its own light filter.
DEFAULT_VELOCITY_FILTER = 0.4
DEFAULT_FEEDBACK_TEST_DURATION_S = 2.0
# Stop pushing this close to the leader gripper's hard stops: holding PWM against a stop just
# stalls the motor (heat, overload shutdown). interbotix's own gripper helper guards the same way.
FEEDBACK_LIMIT_MARGIN = 0.03

# How a (35, 20) field of per-taxel normal force becomes one scalar.
FORCE_METRICS = {
    "abs_sum": lambda normal: abs(float(normal.sum())),      # net normal force, sign-cancelling
    "sum_abs": lambda normal: float(np.abs(normal).sum()),   # total contact regardless of sign
    "mean_abs": lambda normal: float(np.abs(normal).mean()),
    "max_abs": lambda normal: float(np.abs(normal).max()),   # peak taxel
}


class KeyboardControl:
    """Arrow-key / ESC reader on the controlling terminal, polled without blocking.

    cbreak mode, not raw: Ctrl+C keeps working. Right arrow starts or resumes, left arrow
    pauses and locks (see ArmSession.lock()/.unlock()), ESC (or q) stops and, after a delay the
    caller controls, powers down (see shutdown_and_sleep()), b re-zeroes the tactile baseline,
    r emits a 'reset' event (meaning is caller-defined - e.g. run_vtla_policy.py uses it, along
    with left arrow, to abort the current rollout and re-home for the next one).
    When stdin is not a tty there is nothing to read from, so `enabled` is False and the caller
    runs without key control.
    """

    # A lone ESC and the start of an arrow-key sequence look identical until the next bytes
    # arrive, so an unaccompanied ESC only counts once nothing followed it for this long.
    ESC_HOLDOFF_S = 0.05

    def __init__(self):
        self.enabled = False
        self._fd = None
        self._saved = None
        self._buffer = b""
        self._buffer_time = 0.0
        try:
            if sys.stdin.isatty():
                self._fd = sys.stdin.fileno()
                self._saved = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                self.enabled = True
        except (termios.error, ValueError, OSError) as exc:
            print(f"Keyboard control unavailable ({exc}).")

    def poll(self):
        """Return the key events seen since the last call: 'right', 'left', 'esc', 'rezero',
        'reset' - zero or more, in the order the corresponding bytes arrived."""
        if not self.enabled:
            return []
        while select.select([self._fd], [], [], 0)[0]:
            chunk = os.read(self._fd, 64)
            if not chunk:
                break
            self._buffer += chunk
            self._buffer_time = time.time()
        return self._parse()

    def _parse(self):
        """Drain the byte buffer into key events, leaving any incomplete sequence behind."""
        events = []
        while self._buffer:
            if not self._buffer.startswith(b"\x1b"):
                key = self._buffer[:1]
                self._buffer = self._buffer[1:]
                if key in (b"q", b"Q"):
                    events.append("esc")
                elif key in (b"b", b"B"):
                    events.append("rezero")
                elif key in (b"r", b"R"):
                    events.append("reset")
                continue
            # CSI ("\x1b[C") and SS3 ("\x1bOC") both show up depending on terminal mode
            if len(self._buffer) >= 3 and self._buffer[1:2] in (b"[", b"O"):
                code = self._buffer[2:3]
                self._buffer = self._buffer[3:]
                if code == b"C":
                    events.append("right")
                elif code == b"D":
                    events.append("left")
                continue
            if len(self._buffer) == 1 and time.time() - self._buffer_time > self.ESC_HOLDOFF_S:
                self._buffer = b""
                events.append("esc")
                continue
            break  # incomplete sequence: leave it buffered for the next poll
        return events

    def drain(self):
        """Throw away anything typed so far (e.g. before asking for a fresh keypress)."""
        self.poll()
        self._buffer = b""

    def close(self):
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._saved = None
        self.enabled = False


class GripperForceFeedback:
    """Turns the puppet-side tactile normal force into a leader-gripper Goal_PWM.

    Subscribes to the pickled /tactile/<sensor>/raw bundle published by
    4arms_tactile_teleop.launch and reads frames[field][..., 2] - the normal component; channels
    0 and 1 are the shear components. The scalar it reduces that field to becomes PWM in these
    steps:

        force = EMA(metric(field), filter_alpha)
        net   = force - baseline
        x     = clamp((net - deadband) / (full_scale - deadband), 0, 1)

    and then, depending on `mode`:

        "wall" (default)
            The moment `net` crosses the deadband the leader gripper's current position is
            latched as `anchor`, and the command renders a one-way spring against it:

                depth     = max(0, anchor - position)        # how far past contact you squeezed
                stiffness = k_min + (k_max - k_min) * x**curve
                pwm       = sign * min(stiffness * depth, max_pwm) * ramp

            At the anchor the command is 0, so the feedback can only ever resist the operator
            closing further - it never drives the handle open. That distinction matters: a
            command proportional to force alone pushes the leader open, which opens the
            follower, which drops the force, which lets the operator squeeze again - a limit
            cycle the operator feels as the gripper juddering back at them. A spring anchored
            at the contact point has no such loop; squeezing harder just makes the wall stiffer.

        "force"
            The original absolute mapping, pwm = sign * max_pwm * x**curve * ramp. Kept for
            comparison; it is the one that can judder.

    `baseline` is the resting reading captured by capture_baseline() while the follower is
    holding nothing, so a sensor offset cannot show up as constant resistance. `ramp` fades in
    over ramp_s from each activate(), so neither the start nor a resume kicks the operator's
    hand. `sign` is +1 by default: the opening direction, the one that resists closing. Stale
    tactile data commands 0 PWM - nobody should be left pushing against a frozen command from a
    dead publisher - and so does being paused (deactivate()).
    """

    def __init__(self, topic, deadband, full_scale, curve, max_pwm, timeout_s, filter_alpha,
                 metric, sign, ramp_s, mode="wall", field=DEFAULT_FORCE_FIELD,
                 stiffness_min=DEFAULT_WALL_STIFFNESS_MIN,
                 stiffness_max=DEFAULT_WALL_STIFFNESS_MAX,
                 release_fraction=DEFAULT_CONTACT_RELEASE_FRACTION):
        self.topic = topic
        self.field = field
        self.deadband = deadband
        self.full_scale = full_scale
        self.curve = curve
        self.mode = mode
        self.stiffness_min = stiffness_min
        self.stiffness_max = stiffness_max
        self.release_fraction = release_fraction
        self.contact = False
        self.anchor = None
        self.max_pwm = min(int(max_pwm), XC430_PWM_LIMIT)
        self.timeout_s = timeout_s
        self.filter_alpha = filter_alpha
        self.metric_name = metric
        self._metric = FORCE_METRICS[metric]
        self.sign = 1 if sign >= 0 else -1
        self.ramp_s = ramp_s
        self.baseline = 0.0
        self._lock = threading.Lock()
        self._force = 0.0
        self._stamp = 0.0
        self._frames = 0
        self._samples = []
        self._active_since = None
        self._warned_missing = False
        # The payload also carries a (700, 400, 3) preview image, so give the transport a big
        # buffer - a small one queues frames and the haptics start lagging seconds behind.
        self._sub = rospy.Subscriber(
            topic, UInt8MultiArray, self._callback, queue_size=1, buff_size=2 ** 24, tcp_nodelay=True
        )

    def _callback(self, msg):
        try:
            payload = pickle.loads(bytes(msg.data))
        except Exception as exc:
            rospy.logwarn_throttle(5.0, f"tactile feedback: cannot unpickle {self.topic}: {exc}")
            return
        values = payload.get("frames", {}).get(self.field)
        if values is None:
            if not self._warned_missing:
                self._warned_missing = True
                print(f"Warning: {self.topic} carries no {self.field!r}. Relaunch with "
                      f"extra_outputs containing it, or force feedback stays at 0 PWM.")
            return
        array = np.asarray(values, dtype=np.float64)
        normal = (array[..., FORCE_NORMAL_CHANNEL]
                  if array.ndim == 3 and array.shape[-1] == 3 else array)
        value = self._metric(normal)
        with self._lock:
            alpha = self.filter_alpha
            self._force = value if self._frames == 0 else (1 - alpha) * self._force + alpha * value
            self._stamp = time.time()
            self._frames += 1
            self._samples.append(value)
            if len(self._samples) > 200:
                del self._samples[:-200]

    @property
    def frames(self):
        with self._lock:
            return self._frames

    def capture_baseline(self, samples, timeout_s=2.0):
        """Average the resting force and subtract it from every later reading.

        Call this with nothing in the follower gripper. Returns the baseline it settled on.
        """
        if samples <= 0:
            self.baseline = 0.0
            return 0.0
        target = self.frames + samples
        deadline = time.time() + timeout_s
        while self.frames < target and time.time() < deadline:
            time.sleep(0.01)
        with self._lock:
            collected = self._samples[-samples:]
        if not collected:
            print("Warning: no tactile frames to measure the resting force; baseline stays 0.")
            self.baseline = 0.0
        else:
            self.baseline = float(np.mean(collected))
            print(f"Force feedback: resting baseline {self.baseline:.4f} over {len(collected)} frames.")
        return self.baseline

    def activate(self):
        """Start (or restart) the fade-in. Feedback stays at 0 PWM until this is called."""
        self._active_since = time.time()
        self._release_contact()

    def deactivate(self):
        """Hold 0 PWM until the next activate() - used while paused."""
        self._active_since = None
        self._release_contact()

    def _release_contact(self):
        self.contact = False
        self.anchor = None

    def command(self, position=None):
        """Map the latest force to a leader-gripper PWM.

        `position` is the leader gripper joint angle, needed to anchor the wall. Returns
        (pwm, filtered_force, age_s, depth) where depth is how far past the contact anchor the
        operator has squeezed (0 in "force" mode or with no contact). The command is 0 whenever
        the feedback is paused, the tactile data is stale, or nothing is being touched.
        """
        with self._lock:
            force, stamp, frames = self._force, self._stamp, self._frames
        age = time.time() - stamp if frames else float("inf")
        if self._active_since is None or frames == 0 or age > self.timeout_s:
            self._release_contact()
            return 0, force, age, 0.0

        net = force - self.baseline
        if self.contact:
            if net < self.deadband * self.release_fraction:
                self._release_contact()
        elif net > self.deadband:
            self.contact = True
            self.anchor = position
        if not self.contact:
            return 0, force, age, 0.0

        span = max(self.full_scale - self.deadband, 1e-9)
        x = min(max((net - self.deadband) / span, 0.0), 1.0)
        ramp = 1.0 if self.ramp_s <= 0 else min((time.time() - self._active_since) / self.ramp_s, 1.0)

        depth = 0.0
        if self.mode == "wall":
            if self.anchor is None or position is None:
                return 0, force, age, 0.0
            depth = max(0.0, self.anchor - position)
            stiffness = self.stiffness_min + (self.stiffness_max - self.stiffness_min) * (x ** self.curve)
            magnitude = min(stiffness * depth, self.max_pwm)
        else:
            magnitude = self.max_pwm * (x ** self.curve)
        return int(round(self.sign * magnitude * ramp)), force, age, depth

    def describe(self):
        shape = (f"stiffness={self.stiffness_min:.0f}..{self.stiffness_max:.0f} PWM/rad"
                 if self.mode == "wall" else f"max_pwm={self.max_pwm}")
        return (f"{self.topic} -> {self.field}[..., {FORCE_NORMAL_CHANNEL}] ({self.metric_name}), "
                f"mode={self.mode}, "
                f"deadband={self.deadband}, full_scale={self.full_scale}, curve={self.curve}, "
                f"{shape}, cap={self.max_pwm}, ramp={self.ramp_s}s, sign={self.sign:+d} "
                f"({'resists closing' if self.sign > 0 else 'ASSISTS closing'})")


class ClosingSpeedBrake:
    """Resists only the fast part of a leader-gripper close.

        excess = max(0, closing_speed - speed_limit)
        pwm    = min(gain * excess, max_pwm)

    Below `speed_limit` the command is exactly 0, so a normal close feels the same as it does
    with no brake at all. Opening is never resisted. This is deliberately not a damper: a damper
    resists every motion at every speed in both directions, which is what the motor's own
    dynamic braking does and what makes the handle feel draggy.

    It needs no tactile data - it is a local effect on the leader, so it works during a free
    approach, before anything is touched.
    """

    def __init__(self, speed_limit, gain, max_pwm, filter_alpha):
        self.speed_limit = speed_limit
        self.gain = gain
        self.max_pwm = min(int(max_pwm), XC430_PWM_LIMIT)
        self.filter_alpha = filter_alpha
        self.speed = 0.0          # filtered closing speed, positive while closing
        self.peak_speed = 0.0     # peak since the last peek(), for tuning the limit
        self._seen = False

    def command(self, velocity):
        """Return the braking magnitude (>= 0) for the current leader gripper velocity."""
        if velocity is None or self.gain <= 0:
            return 0
        closing = max(0.0, -float(velocity))
        alpha = self.filter_alpha
        self.speed = closing if not self._seen else (1 - alpha) * self.speed + alpha * closing
        self._seen = True
        self.peak_speed = max(self.peak_speed, self.speed)
        excess = self.speed - self.speed_limit
        if excess <= 0:
            return 0
        return int(round(min(excess * self.gain, self.max_pwm)))

    def peek_peak(self):
        """Peak closing speed since the last call - print it to calibrate the limit."""
        peak, self.peak_speed = self.peak_speed, 0.0
        return peak


class GripperContactGuard:
    """Stop closing the puppet gripper after current rises or position stalls."""

    def __init__(self, name, stall_error, stall_motion, stall_cycles, release_margin, effort_threshold=None):
        self.name = name
        self.stall_error = stall_error
        self.stall_motion = stall_motion
        self.stall_cycles = stall_cycles
        self.release_margin = release_margin
        self.effort_threshold = effort_threshold
        self.previous_position = None
        self.stall_count = 0
        self.hold_position = None

    def filter_command(self, desired_position, actual_position, effort=None):
        desired_position = float(desired_position)
        actual_position = float(actual_position)

        if self.hold_position is not None:
            if desired_position > self.hold_position + self.release_margin:
                self.hold_position = None
                self.stall_count = 0
            else:
                self.previous_position = actual_position
                return self.hold_position

        if self.previous_position is None:
            self.previous_position = actual_position
            return desired_position

        closing_error = actual_position - desired_position
        actual_motion = abs(actual_position - self.previous_position)
        effort_contact = (
            self.effort_threshold is not None
            and effort is not None
            and closing_error > 0
            and abs(float(effort)) >= self.effort_threshold
        )
        if closing_error > self.stall_error and actual_motion < self.stall_motion:
            self.stall_count += 1
        else:
            self.stall_count = 0

        self.previous_position = actual_position
        if effort_contact or self.stall_count >= self.stall_cycles:
            self.hold_position = actual_position
            self.stall_count = 0
            source = "effort/current" if effort_contact else "position stall"
            print(f"{self.name} gripper contact detected by {source}; holding at {self.hold_position:.4f}.")
            return self.hold_position

        return desired_position


def make_bot(robot_model, robot_name, init_node):
    return InterbotixManipulatorXS(
        robot_model=robot_model,
        group_name="arm",
        gripper_name="gripper",
        robot_name=robot_name,
        init_node=init_node,
    )


def set_arm_position_time_profile(bot, moving_time, accel_time=None):
    if accel_time is None:
        accel_time = min(DT, moving_time / 2)
    bot.dxl.robot_set_operating_modes(
        "group",
        "arm",
        "position",
        "time",
        int(moving_time * 1000),
        int(accel_time * 1000),
    )
    bot.arm.moving_time = None
    bot.arm.accel_time = None
    bot.arm.set_trajectory_time(moving_time=moving_time, accel_time=accel_time)


def move_arms_to_start(master_bot, puppet_bot):
    start_arm_qpos = START_ARM_POSE[:6]
    for bot in (master_bot, puppet_bot):
        set_arm_position_time_profile(bot, DEFAULT_PREP_ARM_MOVE_TIME, DEFAULT_PREP_ARM_ACCEL_TIME)
        commanded = bot.arm.set_joint_positions(
            start_arm_qpos,
            moving_time=DEFAULT_PREP_ARM_MOVE_TIME,
            accel_time=DEFAULT_PREP_ARM_ACCEL_TIME,
            blocking=False,
        )
        if not commanded:
            print(f"Warning: {bot.dxl.robot_name} arm start-pose command was rejected by joint/velocity limits.")
    time.sleep(DEFAULT_PREP_ARM_MOVE_TIME)


def clamp_arm_joint_positions(bot, joint_positions):
    clamped_positions = []
    limit_violations = []
    lower_limits = bot.arm.group_info.joint_lower_limits
    upper_limits = bot.arm.group_info.joint_upper_limits
    joint_names = bot.arm.group_info.joint_names

    for index, position in enumerate(joint_positions):
        position = float(position)
        lower = lower_limits[index]
        upper = upper_limits[index]
        if position < lower:
            clamped_positions.append(lower)
            limit_violations.append((index, joint_names[index], position, lower, upper))
        elif position > upper:
            clamped_positions.append(upper)
            limit_violations.append((index, joint_names[index], position, lower, upper))
        else:
            clamped_positions.append(position)

    return clamped_positions, limit_violations


def get_arm_gripper_effort(bot):
    efforts = getattr(bot.gripper.core.joint_states, "effort", None)
    if efforts is None or len(efforts) <= 6:
        return None
    return float(efforts[6])


def get_arm_gripper_velocity(bot):
    """Leader/follower gripper joint velocity in rad/s (negative closes), or None if unpublished."""
    velocities = getattr(bot.gripper.core.joint_states, "velocity", None)
    if velocities is None or len(velocities) <= 6:
        return None
    return float(velocities[6])


def set_puppet_gripper_current_limit(puppet_bot, current_limit):
    if current_limit is None:
        return
    last_error = None
    for register_name in ("current_limit", "Current_Limit"):
        try:
            puppet_bot.dxl.robot_set_motor_registers("single", "gripper", register_name, int(current_limit))
            print(f"Set puppet gripper current limit to {current_limit}.")
            return
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to set puppet gripper current limit to {current_limit}") from last_error


def apply_puppet_gripper_current_limit(puppet_bot, current_limit, cycle_gripper_torque=False):
    if current_limit is None:
        return
    if cycle_gripper_torque:
        puppet_bot.dxl.robot_torque_enable("single", "gripper", False)
    set_puppet_gripper_current_limit(puppet_bot, current_limit)
    if cycle_gripper_torque:
        puppet_bot.dxl.robot_torque_enable("single", "gripper", True)


def publish_master_gripper_pwm(master_bot, command, pwm):
    """Send one Goal_PWM to the leader gripper (only meaningful while it is in 'pwm' mode)."""
    command.cmd = float(pwm)
    master_bot.gripper.core.pub_single.publish(command)


class MasterGripperEffort:
    """Sends Goal_PWM to the leader gripper, dropping motor torque whenever the command is 0.

    A Goal_PWM of 0 is not a free motor: torque stays enabled and the driver holds the windings
    at 0% duty, which brakes the handle. The operator feels that as constant drag even with
    nothing grasped. Torquing the motor off is what makes the handle genuinely free, so that is
    what happens once the command has been 0 for `release_delay_s` - the delay keeps a command
    flickering around zero from chattering the torque service. Torque comes back immediately,
    before the next non-zero command goes out.
    """

    def __init__(self, master_bot, torque_gate=True, release_delay_s=DEFAULT_TORQUE_GATE_DELAY_S):
        self._bot = master_bot
        self._command = enable_master_gripper_pwm(master_bot)
        self._torque_gate = torque_gate
        self._release_delay_s = release_delay_s
        self._torque_on = True  # the mode switch left it on
        self._zero_since = None
        self._last_pwm = 0

    def send(self, pwm):
        pwm = int(pwm)
        now = time.time()
        if pwm == 0:
            if self._last_pwm != 0:
                self._publish(0)
            if self._torque_gate:
                if self._zero_since is None:
                    self._zero_since = now
                if self._torque_on and now - self._zero_since >= self._release_delay_s:
                    self._bot.dxl.robot_torque_enable("single", "gripper", False)
                    self._torque_on = False
            return
        self._zero_since = None
        if not self._torque_on:
            self._bot.dxl.robot_torque_enable("single", "gripper", True)
            self._torque_on = True
            self._last_pwm = None  # force a resend: the register may not have survived
        if pwm != self._last_pwm:
            self._publish(pwm)

    def _publish(self, pwm):
        publish_master_gripper_pwm(self._bot, self._command, pwm)
        self._last_pwm = pwm

    def restore(self):
        """Zero the effort, re-enable torque, and hand the gripper back to position mode."""
        if not self._torque_on:
            try:
                self._bot.dxl.robot_torque_enable("single", "gripper", True)
                self._torque_on = True
            except Exception as exc:
                print(f"Warning: could not re-enable the leader gripper torque: {exc}")
        release_master_gripper_pwm(self._bot, self._command)


def enable_master_gripper_pwm(master_bot):
    """Switch the leader gripper to PWM mode, parked at 0 effort so it stays free to move.

    xs_sdk torques the motor off, writes the mode, then torques it back on, so the very first
    thing we do afterwards is command 0 - otherwise a stale Goal_PWM could bite immediately.
    """
    master_bot.dxl.robot_set_operating_modes("single", "gripper", "pwm")
    command = JointSingleCommand(name="gripper")
    publish_master_gripper_pwm(master_bot, command, 0)
    return command


def release_master_gripper_pwm(master_bot, command):
    """Zero the effort and hand the leader gripper back to position mode."""
    try:
        publish_master_gripper_pwm(master_bot, command, 0)
        time.sleep(0.05)
        master_bot.dxl.robot_set_operating_modes("single", "gripper", "position")
    except Exception as exc:
        print(f"Warning: could not restore the leader gripper to position mode: {exc}")


def feedback_direction_test(master_bot, pwm, duration_s):
    """Hold a fixed PWM so the operator can confirm which way the leader gripper is pushed."""
    command = enable_master_gripper_pwm(master_bot)
    print(f"Direction test: holding {pwm:+d} PWM on the leader gripper for {duration_s:.1f}s.")
    print("Hold the handle loosely. A correct sign pushes the gripper OPEN (resisting a squeeze).")
    deadline = time.time() + duration_s
    try:
        while time.time() < deadline:
            publish_master_gripper_pwm(master_bot, command, pwm)
            time.sleep(DT)
    finally:
        release_master_gripper_pwm(master_bot, command)
    print("Direction test done.")


def master_gripper_pose(robot_side, ratio):
    """Master gripper joint angle at `ratio` of its travel (0.0 = closed, 1.0 = open)."""
    if robot_side == "left":
        gripper_open = MASTER_LEFT_GRIPPER_JOINT_OPEN
        gripper_close = MASTER_LEFT_GRIPPER_JOINT_CLOSE
    else:
        gripper_open = MASTER_RIGHT_GRIPPER_JOINT_OPEN
        gripper_close = MASTER_RIGHT_GRIPPER_JOINT_CLOSE
    return gripper_close + ratio * (gripper_open - gripper_close)


def prep_robots(
    master_bot,
    puppet_bot,
    robot_side,
    arm_command_move_time,
    puppet_gripper_current_limit,
    start_open_ratio=DEFAULT_START_OPEN_RATIO,
):
    # reboot gripper motors, and set operating modes for all motors
    puppet_bot.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot.dxl.robot_set_operating_modes("single", "gripper", "position")
    set_puppet_gripper_current_limit(puppet_bot, puppet_gripper_current_limit)
    torque_on(puppet_bot)
    torque_on(master_bot)

    # move arms to starting position
    move_arms_to_start(master_bot, puppet_bot)
    # move_arms_to_start already left the puppet arm in position + "time" profile mode, so we
    # only need to shrink the per-command trajectory time to the teleop cadence. Do it with
    # set_trajectory_time (Profile_Velocity/Acceleration are RAM registers) instead of
    # set_arm_position_time_profile(), which re-writes the EEPROM Operating_Mode and briefly
    # drops the puppet arm's torque -> the arm used to sag ("drop down") right after homing.
    _accel_time = min(DT, arm_command_move_time / 2)
    puppet_bot.arm.moving_time = None
    puppet_bot.arm.accel_time = None
    puppet_bot.arm.set_trajectory_time(moving_time=arm_command_move_time, accel_time=_accel_time)
    # move grippers to starting position: the master parks at start_open_ratio and the puppet at
    # exactly what that maps to, so the two already agree and nothing jumps when teleop begins.
    master_start = master_gripper_pose(robot_side, start_open_ratio)
    puppet_start = master_to_puppet_gripper(robot_side, master_start)
    print(f"Gripper start poses: master {master_start:.4f}, puppet {puppet_start:.4f} (mapped).")
    move_grippers([master_bot, puppet_bot], [master_start, puppet_start], move_time=0.5)


def wait_for_start(keyboard):
    """Block until the operator presses the right arrow. Returns False if they pressed ESC."""
    if not keyboard.enabled:
        print("No tty on stdin, so there are no key controls: starting teleoperation now.")
        return True
    print("Press the RIGHT arrow to start teleoperation, ESC (or q) to quit.")
    keyboard.drain()
    while True:
        for event in keyboard.poll():
            if event == "right":
                return True
            if event == "esc":
                print("Quit before starting.")
                return False
        time.sleep(0.02)


def master_to_puppet_gripper(robot_side, master_gripper_joint):
    if robot_side == "left":
        return MASTER2PUPPET_LEFT_JOINT_FN(master_gripper_joint)
    return MASTER2PUPPET_RIGHT_JOINT_FN(master_gripper_joint)


class ArmSession:
    """Per-side teleop state: which bots, which haptics, and the pause/resume blend.

    One instance per side. teleop_loop() drives a list of these each tick - one for single-side
    teleop, two (left and right) for --side dual, sharing one keyboard and one control loop so
    LEFT/RIGHT/ESC act on both arms at once instead of needing to be pressed twice.
    """

    def __init__(self, robot_side, master_bot, puppet_bot, gripper_stall_error, gripper_stall_motion,
                gripper_stall_cycles, gripper_release_margin, gripper_effort_threshold):
        self.robot_side = robot_side
        self.master_bot = master_bot
        self.puppet_bot = puppet_bot
        self.gripper_command = JointSingleCommand(name="gripper")
        self.guard = GripperContactGuard(
            robot_side.capitalize(),
            gripper_stall_error,
            gripper_stall_motion,
            gripper_stall_cycles,
            gripper_release_margin,
            effort_threshold=gripper_effort_threshold,
        )
        self.feedback_open_limit = master_gripper_pose(robot_side, 1.0) - FEEDBACK_LIMIT_MARGIN
        self.feedback_close_limit = master_gripper_pose(robot_side, 0.0) + FEEDBACK_LIMIT_MARGIN
        self.force_feedback = None
        self.close_brake = None
        self.master_effort = None
        self.paused = False
        self.reported_clamped_joints = set()
        self.last_feedback_report = 0.0
        self.feedback_stale_since = None
        self.blend_from_arm = None
        self.blend_from_gripper = None
        self.blend_end = 0.0

    def reset_blend(self, now, resume_blend_s):
        """(Re)anchor the blend at the puppet's actual current pose - called at teleop start and
        on every resume, so the follower always eases from where it really is, never snaps."""
        self.blend_from_arm = np.array(self.puppet_bot.dxl.joint_states.position[:6], dtype=np.float64)
        self.blend_from_gripper = get_arm_gripper_positions(self.puppet_bot)
        self.blend_end = now + resume_blend_s

    def stop_haptics(self):
        if self.force_feedback is not None and self.master_effort is not None:
            self.force_feedback.deactivate()
            self.master_effort.send(0)

    def lock(self):
        """Re-torque the leader arm rigid (gripper back to position mode) and silence haptics.
        Used by a LEFT-arrow pause and by the ESC/q shutdown path. master_effort is set to None
        afterwards since restore() has already taken the gripper out of PWM mode - the object
        would no longer be valid to keep sending through; unlock() builds a fresh one on resume.
        """
        self.stop_haptics()
        if self.master_effort is not None:
            self.master_effort.restore()
            self.master_effort = None
        torque_on(self.master_bot)

    def unlock(self, torque_gate):
        """Undo lock(): hand the leader back to the operator (torque-free arm; gripper back into
        PWM mode with feedback active again). Only for a RIGHT-arrow resume after a LEFT-arrow
        pause - the very first hand-off at teleop start does this same torque_off/PWM/activate
        sequence itself, in teleop(), because it also has a one-time baseline capture to run
        between switching to PWM mode and activating the ramp, which this simpler path skips
        (the baseline from teleop start is still good; 'b' recaptures it on demand any time).
        """
        torque_off(self.master_bot)
        if self.force_feedback is not None:
            self.master_effort = MasterGripperEffort(self.master_bot, torque_gate=torque_gate)
            self.force_feedback.activate()


def _drive_one_tick(session, now, resume_blend_s, gripper_contact_guard):
    """One control-loop tick for a single (non-paused) session: sync arm+gripper, run haptics."""
    blend_end = session.blend_end
    blend = 1.0 if now >= blend_end or resume_blend_s <= 0 else 1.0 - (blend_end - now) / resume_blend_s

    # sync joint positions
    master_state_joints = session.master_bot.dxl.joint_states.position[:6]
    puppet_arm_joints, limit_violations = clamp_arm_joint_positions(session.puppet_bot, master_state_joints)
    new_violations = [v for v in limit_violations if v[0] not in session.reported_clamped_joints]
    if new_violations:
        details = ", ".join(
            f"{name} {value:.3f}->[{lower:.3f}, {upper:.3f}]"
            for _, name, value, lower, upper in new_violations
        )
        print(f"Warning: {session.robot_side} clamping puppet arm target to joint limits "
              f"({details}); repeated warnings suppressed.")
        session.reported_clamped_joints.update(v[0] for v in new_violations)
    if blend < 1.0:
        puppet_arm_joints = list(session.blend_from_arm + blend * (np.array(puppet_arm_joints) - session.blend_from_arm))
    session.puppet_bot.arm.publish_positions(puppet_arm_joints, blocking=False)

    # sync gripper positions
    master_gripper_joint = session.master_bot.dxl.joint_states.position[6]
    puppet_gripper_joint_target = master_to_puppet_gripper(session.robot_side, master_gripper_joint)
    if blend < 1.0:
        puppet_gripper_joint_target = (
            session.blend_from_gripper + blend * (puppet_gripper_joint_target - session.blend_from_gripper)
        )
    if gripper_contact_guard:
        puppet_gripper_joint_target = session.guard.filter_command(
            puppet_gripper_joint_target,
            get_arm_gripper_positions(session.puppet_bot),
            effort=get_arm_gripper_effort(session.puppet_bot),
        )

    session.gripper_command.cmd = puppet_gripper_joint_target
    session.puppet_bot.gripper.core.pub_single.publish(session.gripper_command)

    # haptics: resist the operator closing further, anchored at where contact began, plus
    # the speed governor, which needs no contact at all
    if session.force_feedback is not None and session.master_effort is not None:
        pwm, force, age, depth = session.force_feedback.command(master_gripper_joint)
        gripper_velocity = get_arm_gripper_velocity(session.master_bot)
        brake_pwm = session.close_brake.command(gripper_velocity) if session.close_brake is not None else 0
        pwm += session.force_feedback.sign * brake_pwm
        # Guard the hard stops, but only when the handle is not already moving away from
        # them: braking a fast close that starts at the open stop must not be suppressed.
        moving_open = gripper_velocity is not None and gripper_velocity > 0
        if pwm > 0 and master_gripper_joint >= session.feedback_open_limit and not moving_open:
            pwm = 0
        elif pwm < 0 and master_gripper_joint <= session.feedback_close_limit:
            pwm = 0
        pwm = int(max(-XC430_PWM_LIMIT, min(XC430_PWM_LIMIT, pwm)))
        session.master_effort.send(pwm)
        if age > session.force_feedback.timeout_s:
            if session.feedback_stale_since is None:
                session.feedback_stale_since = now
                print(f"{session.robot_side} force feedback: tactile data stale ({age:.2f}s), holding 0 PWM.")
        elif session.feedback_stale_since is not None:
            session.feedback_stale_since = None
            print(f"{session.robot_side} force feedback: tactile data back, resuming.")
        if now - session.last_feedback_report >= 1.0:
            session.last_feedback_report = now
            state = "contact" if session.force_feedback.contact else "free   "
            speed = f"{session.close_brake.speed:5.2f}" if session.close_brake is not None else "  -  "
            peak = f"{session.close_brake.peek_peak():5.2f}" if session.close_brake is not None else "  -  "
            print(f"{session.robot_side:5s} {state}  force={force:8.4f}  net={force - session.force_feedback.baseline:+8.4f}  "
                  f"depth={depth:6.4f}rad  vclose={speed}(peak {peak})rad/s  "
                  f"brake={brake_pwm:+4d}  pwm={pwm:+5d}  age={age * 1000:5.1f}ms")


def teleop_loop(
    sessions,
    keyboard,
    duration_s=None,
    gripper_contact_guard=True,
    force_baseline_samples=DEFAULT_FORCE_BASELINE_SAMPLES,
    resume_blend_s=DEFAULT_RESUME_BLEND_S,
    torque_gate=True,
):
    """Drive every session in `sessions` from its leader until stopped, the duration, or Ctrl+C.

    One shared keyboard controls all of them together:
        LEFT    pauses every session at once AND locks it (leader torqued rigid, gripper back
                to position mode, haptics silenced) - session.lock(). Nothing moves while
                paused; there is no "paused but the leader is still free" in-between state.
        RIGHT   resumes every paused session (session.unlock(): leader torque-free again,
                gripper back into PWM mode, feedback reactivated), blending each follower back
                onto its own leader over resume_blend_s.
        ESC/q   stops the loop immediately and returns "shutdown" - the caller (teleop()) waits
                a second and then walks every arm to its sleep position and cuts torque with no
                further confirmation; see shutdown_and_sleep().
    Returns "duration" if duration_s elapses first.
    """
    start_time = time.time()
    for session in sessions:
        session.reset_blend(start_time, resume_blend_s)

    while duration_s is None or time.time() - start_time < duration_s:
        for event in keyboard.poll():
            if event == "esc":
                for session in sessions:
                    session.lock()  # rigid immediately - the leader must not sit torque-free
                                    # during teleop()'s one-second pause before it walks to sleep
                return "shutdown"
            if event == "rezero":
                for session in sessions:
                    if session.force_feedback is not None:
                        session.force_feedback.capture_baseline(force_baseline_samples)
                continue
            if event == "left":
                newly_paused = [s for s in sessions if not s.paused]
                for session in newly_paused:
                    session.paused = True
                    session.lock()
                if newly_paused:
                    print("Paused and locked (leader rigid). RIGHT to resume, ESC to stop and power off.")
            elif event == "right":
                now = time.time()
                resumed = [s for s in sessions if s.paused]
                for session in resumed:
                    session.paused = False
                    session.unlock(torque_gate)
                    session.reset_blend(now, resume_blend_s)
                if resumed:
                    print(f"Resumed; leader(s) free again, blending the follower(s) back over "
                          f"{resume_blend_s:.1f}s.")

        now = time.time()
        for session in sessions:
            if session.paused:
                continue
            _drive_one_tick(session, now, resume_blend_s, gripper_contact_guard)
        time.sleep(DT)

    return "duration"


def shutdown_and_sleep(sessions):
    """Called after teleop_loop() returns "shutdown" (ESC/q) and teleop()'s one-second pause.

    Every leader was already re-torqued rigid (gripper back to position mode) the instant ESC
    was pressed - teleop_loop()'s "esc" branch calls session.lock() itself, precisely so the
    leader is never left torque-free during the pause that follows. The lock() call here just
    re-asserts that (harmless/idempotent) before every arm - leader and follower, both sides if
    --side dual - walks through the same two-stage shutdown sleep_plus.py uses (old way-point,
    then the measured rest pose - see that file's module comment for why not the factory pose),
    grippers open, torque off. No confirmation prompt: by the time this runs the operator has
    already committed to shutting down.

    Reuses sleep_plus.move_shut_robot() directly so this and `python sleep_plus.py --shut_down`
    can never drift apart on timing or target pose.
    """
    for session in sessions:
        session.lock()
    print("Moving to the sleep position...")
    puppet_waypoints = (OLD_PUPPET_SLEEP_POSITION, PUPPET_SLEEP_POSITION)
    master_waypoints = (OLD_MASTER_SLEEP_POSITION, MASTER_SLEEP_POSITION)
    puppet_move_times = tuple(t * SHUTDOWN_SPEED_SCALE for t in (1, 1))
    master_move_times = tuple(t * SHUTDOWN_SPEED_SCALE for t in (1, 1.5))
    puppet_gripper_open = {"left": PUPPET_LEFT_GRIPPER_JOINT_OPEN, "right": PUPPET_RIGHT_GRIPPER_JOINT_OPEN}
    master_gripper_open = {"left": MASTER_LEFT_GRIPPER_JOINT_OPEN, "right": MASTER_RIGHT_GRIPPER_JOINT_OPEN}

    threads = []
    for session in sessions:
        threads.append(threading.Thread(
            target=move_shut_robot,
            args=(session.puppet_bot, puppet_gripper_open[session.robot_side], puppet_waypoints, puppet_move_times)))
        threads.append(threading.Thread(
            target=move_shut_robot,
            args=(session.master_bot, master_gripper_open[session.robot_side], master_waypoints, master_move_times)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("At the sleep position; torque off.")
    return True


def teleop(
    robot_side,
    skip_prep=False,
    arm_command_move_time=DEFAULT_ARM_COMMAND_MOVE_TIME,
    duration_s=None,
    gripper_contact_guard=True,
    gripper_stall_error=DEFAULT_GRIPPER_STALL_ERROR,
    gripper_stall_motion=DEFAULT_GRIPPER_STALL_MOTION,
    gripper_stall_cycles=DEFAULT_GRIPPER_STALL_CYCLES,
    gripper_release_margin=DEFAULT_GRIPPER_RELEASE_MARGIN,
    puppet_gripper_current_limit=DEFAULT_PUPPET_GRIPPER_CURRENT_LIMIT,
    gripper_effort_threshold=None,
    start_open_ratio=DEFAULT_START_OPEN_RATIO,
    resume_blend_s=DEFAULT_RESUME_BLEND_S,
    force_feedback_configs=None,
    close_brake_config=None,
    force_baseline_samples=DEFAULT_FORCE_BASELINE_SAMPLES,
    torque_gate=True,
    feedback_test_pwm=None,
    feedback_test_duration=DEFAULT_FEEDBACK_TEST_DURATION_S,
):
    """Run one- or two-arm teleoperation. No data recording.

    robot_side is "left", "right", or "dual" (both arms at once, sharing one keyboard so a
    single LEFT/RIGHT/ESC acts on both together instead of needing to be pressed per side).

    Right arrow starts, and resumes after a pause. Left arrow pauses AND locks every leader
    rigid (session.lock()/.unlock() - see ArmSession). ESC/q stops AND locks every leader rigid
    immediately too, then after a one-second pause walks every arm to its sleep position and
    cuts torque with no further confirmation - see shutdown_and_sleep().

    force_feedback_configs is a dict keyed by side, e.g. {"left": {...}, "right": {...}} - only
    the sides actually running need a key; a missing or None entry disables feedback for that
    side. close_brake_config is a single dict shared by every side (it carries no per-side
    topic, unlike force feedback).
    """
    sides = ("left", "right") if robot_side == "dual" else (robot_side,)
    force_feedback_configs = force_feedback_configs or {}

    sessions = []
    keyboard = KeyboardControl()
    skip_final_lock = False
    try:
        bots = {}
        for index, side in enumerate(sides):
            bots[side] = (
                make_bot("wx250s", f"master_{side}", init_node=(index == 0)),
                make_bot("vx300s", f"puppet_{side}", init_node=False),
            )

        for side in sides:
            master_bot, puppet_bot = bots[side]
            if not skip_prep:
                prep_robots(
                    master_bot,
                    puppet_bot,
                    side,
                    arm_command_move_time,
                    puppet_gripper_current_limit,
                    start_open_ratio=start_open_ratio,
                )
            else:
                apply_puppet_gripper_current_limit(
                    puppet_bot,
                    puppet_gripper_current_limit,
                    cycle_gripper_torque=True,
                )
                set_arm_position_time_profile(puppet_bot, arm_command_move_time)

        if feedback_test_pwm is not None:
            for side in sides:
                feedback_direction_test(bots[side][0], int(feedback_test_pwm), feedback_test_duration)
            # Left torqued on from prep above; a one-shot direction test needs nothing further.
            skip_final_lock = True
            return

        for side in sides:
            master_bot, puppet_bot = bots[side]
            session = ArmSession(
                side, master_bot, puppet_bot,
                gripper_stall_error, gripper_stall_motion, gripper_stall_cycles,
                gripper_release_margin, gripper_effort_threshold,
            )
            config = force_feedback_configs.get(side)
            if config is not None:
                session.force_feedback = GripperForceFeedback(**config)
                print(f"{side} force feedback: {session.force_feedback.describe()}")
            if close_brake_config:
                session.close_brake = ClosingSpeedBrake(**close_brake_config)
            sessions.append(session)

        for session in sessions:
            if session.force_feedback is not None:
                deadline = time.time() + 2.0
                while session.force_feedback.frames == 0 and time.time() < deadline:
                    time.sleep(0.05)
                if session.force_feedback.frames == 0:
                    print(f"Warning: no tactile data on {session.force_feedback.topic} yet. "
                          f"{session.robot_side} teleoperation will run with 0 PWM until it arrives "
                          f"(check the launch and 'rostopic hz').")
                session.force_feedback.capture_baseline(force_baseline_samples)

        if not wait_for_start(keyboard):
            for session in sessions:
                torque_off(session.master_bot)  # never started: let the leader(s) go limp, as before
            skip_final_lock = True
            return

        # Hand the leader(s) over to the operator: each arm goes torque-free, and each gripper
        # goes to PWM mode parked at 0 effort, which is just as free until feedback ramps in.
        for session in sessions:
            torque_off(session.master_bot)
            if session.force_feedback is not None:
                session.master_effort = MasterGripperEffort(session.master_bot, torque_gate=torque_gate)
                session.force_feedback.activate()
                print(f"{session.robot_side} leader gripper is in PWM mode; force feedback fades in now.")
                if session.close_brake is not None and session.close_brake.gain > 0:
                    print(f"{session.robot_side} closing speed brake: free below "
                          f"{session.close_brake.speed_limit} rad/s, then {session.close_brake.gain} "
                          f"PWM per rad/s over, capped at {session.close_brake.max_pwm}.")
        print("Started. RIGHT = resume, LEFT = pause & lock, b = re-zero baseline(s), "
              "ESC = stop and power off.")

        reason = teleop_loop(
            sessions,
            keyboard,
            duration_s=duration_s,
            gripper_contact_guard=gripper_contact_guard,
            force_baseline_samples=force_baseline_samples,
            resume_blend_s=resume_blend_s,
            torque_gate=torque_gate,
        )
        if reason == "shutdown":
            print("Stopped and locked (leader(s) rigid). Moving to the sleep position and "
                  "powering off in 1 second...")
            time.sleep(1.0)
            shutdown_and_sleep(sessions)
            skip_final_lock = True
        else:
            print("Duration reached.")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if not skip_final_lock:
            for session in sessions:
                try:
                    session.lock()
                except Exception as exc:
                    print(f"Warning: could not lock the {session.robot_side} leader: {exc}")
        keyboard.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("side", choices=("left", "right", "dual"),
                        help="'dual' runs both arms at once, sharing one keyboard: LEFT/RIGHT/ESC "
                             "act on both together.")
    parser.add_argument(
        "--skip_prep",
        action="store_true",
        help="Skip homing/start-pose setup and immediately enter teleoperation.",
    )
    parser.add_argument(
        "--arm_command_move_time",
        type=float,
        default=DEFAULT_ARM_COMMAND_MOVE_TIME,
        help="Time-profile seconds for puppet arm position commands.",
    )
    parser.add_argument(
        "--duration_s",
        type=float,
        default=None,
        help="Optional teleoperation duration in seconds. Default runs until Ctrl+C.",
    )
    parser.add_argument(
        "--disable_gripper_contact_guard",
        action="store_true",
        help="Allow the puppet gripper to keep closing even after stall/contact detection.",
    )
    parser.add_argument(
        "--gripper_stall_error",
        type=float,
        default=DEFAULT_GRIPPER_STALL_ERROR,
        help="Minimum actual-vs-target closing error before contact detection can trigger.",
    )
    parser.add_argument(
        "--gripper_stall_motion",
        type=float,
        default=DEFAULT_GRIPPER_STALL_MOTION,
        help="Maximum per-cycle actual gripper motion considered stalled.",
    )
    parser.add_argument(
        "--gripper_stall_cycles",
        type=int,
        default=DEFAULT_GRIPPER_STALL_CYCLES,
        help="Consecutive stalled control cycles required before holding the gripper.",
    )
    parser.add_argument(
        "--gripper_release_margin",
        type=float,
        default=DEFAULT_GRIPPER_RELEASE_MARGIN,
        help="How far the master must command opening before a held gripper follows again.",
    )
    parser.add_argument(
        "--puppet_gripper_current_limit",
        type=int,
        default=DEFAULT_PUPPET_GRIPPER_CURRENT_LIMIT,
        help="Raw Dynamixel current limit for the puppet gripper. Use 0 to skip setting it.",
    )
    parser.add_argument(
        "--start_open_ratio",
        type=float,
        default=DEFAULT_START_OPEN_RATIO,
        help="Where both grippers park before teleoperation, as a fraction of the master travel "
             "(1.0 = fully open). The puppet parks at whatever this maps to.",
    )
    parser.add_argument(
        "--resume_blend_s",
        type=float,
        default=DEFAULT_RESUME_BLEND_S,
        help="Seconds to blend the follower back onto the leader after a pause. Default %(default)s.",
    )
    parser.add_argument(
        "--no_force_feedback",
        action="store_true",
        help="Disable leader-gripper force feedback (leave the leader gripper torque-free).",
    )
    parser.add_argument(
        "--tactile_topic",
        default=None,
        help="Tactile /raw topic driving the feedback. Default: /tactile/<side>_gripper_outer/raw. "
             "Ignored when side is 'dual' (each side always uses its own default topic there, "
             "since a single override could not name two different topics).",
    )
    parser.add_argument(
        "--force_metric",
        default="abs_sum",
        choices=sorted(FORCE_METRICS),
        help="How the (35, 20) normal-force field becomes one scalar. Default abs_sum.",
    )
    parser.add_argument(
        "--force_deadband",
        type=float,
        default=DEFAULT_FORCE_DEADBAND,
        help="Force below this commands no PWM at all. Default %(default)s (sensor noise is ~0.02).",
    )
    parser.add_argument(
        "--feedback_mode",
        default="wall",
        choices=("wall", "force"),
        help="wall (default): resist closing past where contact began, so the leader is never "
             "driven open. force: raw force-proportional PWM, which can judder against the hand.",
    )
    parser.add_argument(
        "--wall_stiffness_min",
        type=float,
        default=DEFAULT_WALL_STIFFNESS_MIN,
        help="wall mode PWM per radian at just-detected contact. Default %(default)s.",
    )
    parser.add_argument(
        "--wall_stiffness_max",
        type=float,
        default=DEFAULT_WALL_STIFFNESS_MAX,
        help="wall mode PWM per radian at --force_full_scale. Default %(default)s.",
    )
    parser.add_argument(
        "--close_speed_limit",
        type=float,
        default=DEFAULT_CLOSE_SPEED_LIMIT,
        help="Leader gripper closing speed (rad/s) below which nothing is resisted at all. "
             "Watch the vclose peak in the status line to set it above your normal close. "
             "Default %(default)s.",
    )
    parser.add_argument(
        "--close_brake_gain",
        type=float,
        default=DEFAULT_CLOSE_BRAKE_GAIN,
        help="PWM per rad/s of closing speed above the limit. 0 disables the brake entirely. "
             "Default %(default)s.",
    )
    parser.add_argument(
        "--close_brake_max_pwm",
        type=int,
        default=DEFAULT_CLOSE_BRAKE_MAX_PWM,
        help="Cap on the speed brake alone, separate from --max_feedback_pwm. Default %(default)s.",
    )
    parser.add_argument(
        "--velocity_filter",
        type=float,
        default=DEFAULT_VELOCITY_FILTER,
        help="EMA factor on the leader gripper speed; 1.0 disables filtering. Default %(default)s.",
    )
    parser.add_argument(
        "--no_torque_gate",
        action="store_true",
        help="Keep the leader gripper torqued even at 0 PWM. Leaves the motor braking the "
             "handle, which feels like constant drag - only useful for debugging.",
    )
    parser.add_argument(
        "--force_full_scale",
        type=float,
        default=DEFAULT_FORCE_FULL_SCALE,
        help="Force at which the feedback reaches --max_feedback_pwm. Default %(default)s.",
    )
    parser.add_argument(
        "--force_curve",
        type=float,
        default=DEFAULT_FORCE_CURVE,
        help="Exponent shaping deadband->full_scale. 1.0 is linear; higher is a softer onset. "
             "Default %(default)s.",
    )
    parser.add_argument(
        "--force_ramp_s",
        type=float,
        default=DEFAULT_FORCE_RAMP_S,
        help="Seconds to fade the feedback in after start or resume. Default %(default)s.",
    )
    parser.add_argument(
        "--force_baseline_samples",
        type=int,
        default=DEFAULT_FORCE_BASELINE_SAMPLES,
        help="Frames averaged at start to cancel the sensor's resting offset. 0 disables. "
             "Default %(default)s.",
    )
    parser.add_argument(
        "--max_feedback_pwm",
        type=int,
        default=DEFAULT_MAX_FEEDBACK_PWM,
        help=f"Hard cap on the feedback PWM. Default %(default)s of the XC430's {XC430_PWM_LIMIT}.",
    )
    parser.add_argument(
        "--force_timeout",
        type=float,
        default=DEFAULT_FORCE_TIMEOUT_S,
        help="Tactile data older than this commands 0 PWM. Default %(default)s s.",
    )
    parser.add_argument(
        "--force_filter",
        type=float,
        default=DEFAULT_FORCE_FILTER,
        help="EMA factor on the force signal; 1.0 disables filtering. Default %(default)s.",
    )
    parser.add_argument(
        "--feedback_sign",
        type=int,
        default=1,
        choices=(1, -1),
        help="+1 (default) pushes the leader gripper OPEN, resisting a squeeze. -1 inverts it.",
    )
    parser.add_argument(
        "--feedback_test_pwm",
        type=int,
        default=None,
        help="Hold this fixed PWM on the leader gripper for a few seconds, then exit. "
             "Use it once to confirm which sign resists closing.",
    )
    parser.add_argument(
        "--feedback_test_duration",
        type=float,
        default=DEFAULT_FEEDBACK_TEST_DURATION_S,
        help="Seconds for --feedback_test_pwm. Default %(default)s.",
    )
    parser.add_argument(
        "--gripper_effort_threshold",
        type=float,
        default=None,
        help="Absolute puppet gripper effort/current threshold for contact detection. Default disables effort/current detection.",
    )
    args = parser.parse_args()
    if args.arm_command_move_time <= 0:
        raise ValueError("--arm_command_move_time must be positive")
    if args.duration_s is not None and args.duration_s <= 0:
        raise ValueError("--duration_s must be positive")
    if args.gripper_stall_error <= 0:
        raise ValueError("--gripper_stall_error must be positive")
    if args.gripper_stall_motion < 0:
        raise ValueError("--gripper_stall_motion must be non-negative")
    if args.gripper_stall_cycles <= 0:
        raise ValueError("--gripper_stall_cycles must be positive")
    if args.gripper_release_margin <= 0:
        raise ValueError("--gripper_release_margin must be positive")
    if args.puppet_gripper_current_limit < 0:
        raise ValueError("--puppet_gripper_current_limit must be non-negative")
    if args.gripper_effort_threshold is not None and args.gripper_effort_threshold <= 0:
        raise ValueError("--gripper_effort_threshold must be positive")
    if not 0.0 <= args.start_open_ratio <= 1.0:
        raise ValueError("--start_open_ratio must be within [0.0, 1.0]")
    if args.force_deadband < 0:
        raise ValueError("--force_deadband must be non-negative")
    if args.force_full_scale <= args.force_deadband:
        raise ValueError("--force_full_scale must be greater than --force_deadband")
    if args.force_curve <= 0:
        raise ValueError("--force_curve must be positive")
    if args.wall_stiffness_min < 0 or args.wall_stiffness_max < args.wall_stiffness_min:
        raise ValueError("--wall_stiffness_max must be >= --wall_stiffness_min >= 0")
    if args.close_speed_limit < 0:
        raise ValueError("--close_speed_limit must be non-negative")
    if args.close_brake_gain < 0:
        raise ValueError("--close_brake_gain must be non-negative")
    if not 0 < args.close_brake_max_pwm <= XC430_PWM_LIMIT:
        raise ValueError(f"--close_brake_max_pwm must be within (0, {XC430_PWM_LIMIT}]")
    if not 0 < args.velocity_filter <= 1.0:
        raise ValueError("--velocity_filter must be within (0, 1]")
    if args.force_ramp_s < 0:
        raise ValueError("--force_ramp_s must be non-negative")
    if args.force_baseline_samples < 0:
        raise ValueError("--force_baseline_samples must be non-negative")
    if args.resume_blend_s < 0:
        raise ValueError("--resume_blend_s must be non-negative")
    if not 0 < args.max_feedback_pwm <= XC430_PWM_LIMIT:
        raise ValueError(f"--max_feedback_pwm must be within (0, {XC430_PWM_LIMIT}]")
    if args.force_timeout <= 0:
        raise ValueError("--force_timeout must be positive")
    if not 0 < args.force_filter <= 1.0:
        raise ValueError("--force_filter must be within (0, 1]")
    if args.feedback_test_pwm is not None and abs(args.feedback_test_pwm) > XC430_PWM_LIMIT:
        raise ValueError(f"--feedback_test_pwm must be within +-{XC430_PWM_LIMIT}")
    if args.feedback_test_duration <= 0:
        raise ValueError("--feedback_test_duration must be positive")

    close_brake_config = dict(
        speed_limit=args.close_speed_limit,
        gain=args.close_brake_gain,
        max_pwm=args.close_brake_max_pwm,
        filter_alpha=args.velocity_filter,
    )

    # One config per side (topic differs; every other knob is shared). --tactile_topic only
    # makes sense for a single named side, since it cannot name two different topics for 'dual'.
    force_feedback_configs = None
    if not args.no_force_feedback:
        sides = ("left", "right") if args.side == "dual" else (args.side,)
        force_feedback_configs = {
            side: dict(
                topic=(args.tactile_topic if args.side != "dual" and args.tactile_topic
                       else DEFAULT_TACTILE_TOPIC_TEMPLATE.format(side=side)),
                deadband=args.force_deadband,
                full_scale=args.force_full_scale,
                curve=args.force_curve,
                max_pwm=args.max_feedback_pwm,
                timeout_s=args.force_timeout,
                filter_alpha=args.force_filter,
                metric=args.force_metric,
                sign=args.feedback_sign,
                ramp_s=args.force_ramp_s,
                mode=args.feedback_mode,
                stiffness_min=args.wall_stiffness_min,
                stiffness_max=args.wall_stiffness_max,
            )
            for side in sides
        }

    puppet_gripper_current_limit = args.puppet_gripper_current_limit or None
    gripper_effort_threshold = args.gripper_effort_threshold
    if puppet_gripper_current_limit is not None:
        print(f"Puppet gripper current limit: {puppet_gripper_current_limit}")
    if gripper_effort_threshold is not None:
        print(f"Gripper effort/current contact threshold: {gripper_effort_threshold:.2f}")
    else:
        print("Gripper effort/current contact detection: disabled")

    teleop(
        args.side,
        skip_prep=args.skip_prep,
        arm_command_move_time=args.arm_command_move_time,
        duration_s=args.duration_s,
        gripper_contact_guard=not args.disable_gripper_contact_guard,
        gripper_stall_error=args.gripper_stall_error,
        gripper_stall_motion=args.gripper_stall_motion,
        gripper_stall_cycles=args.gripper_stall_cycles,
        gripper_release_margin=args.gripper_release_margin,
        puppet_gripper_current_limit=puppet_gripper_current_limit,
        gripper_effort_threshold=gripper_effort_threshold,
        start_open_ratio=args.start_open_ratio,
        resume_blend_s=args.resume_blend_s,
        force_feedback_configs=force_feedback_configs,
        close_brake_config=close_brake_config,
        force_baseline_samples=args.force_baseline_samples,
        torque_gate=not args.no_torque_gate,
        feedback_test_pwm=args.feedback_test_pwm,
        feedback_test_duration=args.feedback_test_duration,
    )


if __name__ == "__main__":
    main()
