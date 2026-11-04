"""Hardware pieces of the human-in-the-loop rollout (hil_rollout.py).

ClutchLeader / ClutchLeaderPair
    The leader arms with the clutch-style takeover, ported from lerobot's WidowXRos teleop
    (lerobot branch aloha2, src/lerobot/teleoperators/widowx_ros/widowx_ros.py):
      1. pause: the (torqued) leader is driven onto the follower's last command, which becomes the
         clutch reference
      2. release: the arm's Position_P_Gain is lowered to 0 over release_ramp_s before the torque
         is cut, so the arm softens into the operator's hand instead of dropping; the gains are
         restored once the torque is off
      3. settle: wait until every arm joint stayed within settle_tolerance for settle_s (at most
         settle_timeout_s), then offset = leader - reference per arm joint
      4. correcting: the follower is commanded with leader - offset (gripper absolute)
      5. lock: offset cleared, the follower's last command becomes the next reference, goal
         registers are aimed at the present pose before the torque comes back on
    Unlike the lerobot version every step is driven by tick() from the 50 Hz control loop instead
    of blocking, so the follower keeps being commanded and recorded while the leader moves,
    softens and settles. Leader-gripper haptics (record_episodes_aloha2.ArmHaptics) run while the
    arm is released and are restored to position mode before it is locked.

FollowerCommander
    Sends one 14-dim command per tick to both followers (RealEnv.step's convention: arm joints in
    rad, grippers as the normalized opening) with the safety layer: joint-limit clamp, a per-tick
    joint step limit, and an optional blend from the measured pose (correction start / resume).

HilInput
    Terminal keys (one_side_teleop.KeyboardControl plus 'e'/Enter) and an optional evdev foot
    pedal, merged into one event stream.
"""

import concurrent.futures
import logging
import queue
import threading
import time

import numpy as np

from constants import (DT, MASTER_LEFT_GRIPPER_JOINT_NORMALIZE_FN, MASTER_RIGHT_GRIPPER_JOINT_NORMALIZE_FN,
                       PUPPET_LEFT_GRIPPER_JOINT_CLOSE, PUPPET_LEFT_GRIPPER_JOINT_OPEN,
                       PUPPET_LEFT_GRIPPER_JOINT_UNNORMALIZE_FN, PUPPET_RIGHT_GRIPPER_JOINT_CLOSE,
                       PUPPET_RIGHT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN,
                       MASTER_LEFT_GRIPPER_JOINT_CLOSE, MASTER_LEFT_GRIPPER_JOINT_OPEN,
                       MASTER_RIGHT_GRIPPER_JOINT_CLOSE, MASTER_RIGHT_GRIPPER_JOINT_OPEN)
from interbotix_xs_msgs.msg import JointSingleCommand
from robot_utils import torque_off, torque_on

SIDES = ("left", "right")
MASTER_GRIPPER_RANGE = {
    "left": (MASTER_LEFT_GRIPPER_JOINT_CLOSE, MASTER_LEFT_GRIPPER_JOINT_OPEN),
    "right": (MASTER_RIGHT_GRIPPER_JOINT_CLOSE, MASTER_RIGHT_GRIPPER_JOINT_OPEN),
}
MASTER_GRIPPER_NORMALIZE = {"left": MASTER_LEFT_GRIPPER_JOINT_NORMALIZE_FN,
                            "right": MASTER_RIGHT_GRIPPER_JOINT_NORMALIZE_FN}
PUPPET_GRIPPER_RANGE = {
    "left": (PUPPET_LEFT_GRIPPER_JOINT_CLOSE, PUPPET_LEFT_GRIPPER_JOINT_OPEN),
    "right": (PUPPET_RIGHT_GRIPPER_JOINT_CLOSE, PUPPET_RIGHT_GRIPPER_JOINT_OPEN),
}
PUPPET_GRIPPER_UNNORMALIZE = {"left": PUPPET_LEFT_GRIPPER_JOINT_UNNORMALIZE_FN,
                              "right": PUPPET_RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN}
P_GAIN_STEP_S = 0.04  # one Position_P_Gain write per arm every this many seconds while releasing
ALIGN_MAX_SPEED = 1.0  # rad/s: the leader alignment takes at least (largest joint distance) / this
MIN_P_GAIN = 100  # a leader gain below this means an earlier run was killed mid-release
SETTLE_FRACTION = 0.9  # a settled window only needs this share of settle_s: tick jitter must not cost one more tick


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


class ClutchLeader:
    LOCKED, MOVING, RELEASING, SETTLING, FREE = "locked", "moving", "releasing", "settling", "free"

    def __init__(self, bot, side, release_ramp_s=0.5, relative=True, settle_s=0.5, settle_tolerance=0.01,
                 settle_timeout_s=2.0, max_offset_warn=0.25, haptics=None, control_dt=DT, io_executor=None):
        if side not in SIDES:
            raise ValueError("side must be 'left' or 'right'")
        if release_ramp_s < 0 or settle_s <= 0 or settle_tolerance <= 0 or settle_timeout_s < settle_s:
            raise ValueError("need release_ramp_s >= 0, settle_s > 0, settle_tolerance > 0, "
                             "settle_timeout_s >= settle_s")
        self.bot = bot
        self.side = side
        self.release_ramp_s = release_ramp_s
        self.relative = relative
        self.settle_s = settle_s
        self.settle_tolerance = settle_tolerance
        self.settle_timeout_s = settle_timeout_s
        self.max_offset_warn = max_offset_warn
        self.haptics = haptics
        self.control_dt = control_dt
        self.state = self.LOCKED
        self.p_gains = []
        self.reference = None  # follower arm pose the clutch offset is taken against
        self.offset = np.zeros(6)
        self.engaged = False  # offset applies (clutch-style correction running)
        self.last_arm_action = None
        self.gripper_command = JointSingleCommand(name="gripper")
        self._move = None
        self._ramp = None
        self._window = []
        self._settle_start = None
        self._settle_stats = None  # numbers behind takeover_progress() while settling
        # Register writes (Position_P_Gain, torque) are ROS service calls - with per-joint gains 6 per
        # step. They run on this arm's own worker thread, so the control loop never waits for them
        # and the two leaders release in parallel.
        self._own_io = io_executor is None
        self._io = io_executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="leader-" + side)

    # -- setup ---------------------------------------------------------------------------------

    def configure(self):
        """Read the arm's Position_P_Gain registers (restored after every release ramp). Refuses
        gains that look like the remains of an interrupted ramp, which would leave the arm limp."""
        response = self.bot.dxl.robot_get_motor_registers("group", "arm", "Position_P_Gain")
        gains = [int(gain) for gain in response.values]
        if not gains or min(gains) < MIN_P_GAIN:
            raise RuntimeError(
                "{} leader Position_P_Gain is {} - an earlier run was probably killed while releasing the "
                "arm. Power-cycle the leader arms (the gain is a servo RAM register) and restart the launch."
                .format(self.side, gains))
        self.p_gains = gains

    def _set_p_gains(self, gains):
        if len(set(gains)) == 1:
            self.bot.dxl.robot_set_motor_registers("group", "arm", "Position_P_Gain", int(gains[0]))
            return
        for joint, gain in zip(self.bot.arm.group_info.joint_names, gains):
            self.bot.dxl.robot_set_motor_registers("single", joint, "Position_P_Gain", int(gain))

    # -- readings ------------------------------------------------------------------------------

    def read(self):
        """(six arm joint angles, gripper joint angle) as measured."""
        position = self.bot.dxl.joint_states.position
        return np.array(position[:6], dtype=np.float64), float(position[6])

    def raw_action(self):
        """Absolute leader reading in the hdf5 /action layout: 6 joints + normalized gripper."""
        arm, gripper = self.read()
        return np.concatenate([arm, [MASTER_GRIPPER_NORMALIZE[self.side](gripper)]])

    def action(self):
        """Follower command from this leader: arm - clutch offset, gripper absolute (normalized)."""
        arm, gripper = self.read()
        arm = arm - self.offset
        self.last_arm_action = arm.copy()
        return np.concatenate([arm, [MASTER_GRIPPER_NORMALIZE[self.side](gripper)]])

    @property
    def busy(self):
        return self.state in (self.MOVING, self.RELEASING, self.SETTLING)

    # -- transitions ---------------------------------------------------------------------------

    def begin_move(self, arm_target, gripper_opening, duration_s, now):
        """Drive the locked leader onto the follower's pose (arm rad, gripper as normalized opening)
        over duration_s; that pose becomes the clutch reference."""
        if self.state != self.LOCKED:
            raise RuntimeError("{} leader must be locked to be moved (state {})".format(self.side, self.state))
        arm_start, gripper_start = self.read()
        close, open_ = MASTER_GRIPPER_RANGE[self.side]
        gripper_target = close + float(np.clip(gripper_opening, 0.0, 1.0)) * (open_ - close)
        self.reference = np.array(arm_target, dtype=np.float64)
        distance = float(np.abs(self.reference - arm_start).max())
        duration = max(duration_s, distance / ALIGN_MAX_SPEED, 2 * self.control_dt)
        self._move = {"last": now, "progress": 0.0, "duration": duration, "arm": (arm_start, self.reference.copy()),
                      "gripper": (gripper_start, gripper_target)}
        self.state = self.MOVING

    def stop_move(self):
        if self.state == self.MOVING:
            self._move = None
            self.state = self.LOCKED

    def begin_release(self, now):
        """Soften and release the arm (P-gain ramp, torque off, haptics on); with relative control
        and a reference, the clutch offset is taken once the arm has settled."""
        if self.state != self.LOCKED:
            raise RuntimeError("{} leader must be locked to be released (state {})".format(self.side, self.state))
        steps = max(int(round(self.release_ramp_s / P_GAIN_STEP_S)), 1) if self.release_ramp_s > 0 else 0
        self._ramp = {"next": now, "steps": steps, "done": 0, "pending": None, "released": False,
                      "gains_restored": False}
        self.state = self.RELEASING
        self.tick(now)

    def tick(self, now):
        if self.state == self.MOVING:
            self._tick_move(now)
        elif self.state == self.RELEASING:
            self._tick_release(now)
        elif self.state == self.SETTLING:
            self._tick_settle(now)
        elif self.state == self.FREE and self.haptics is not None:
            self.haptics.update()

    def _tick_move(self, now):
        move = self._move
        # Progress by elapsed time, but at most two control periods per tick: a stalled loop (slow
        # inference, a blocking call) makes the motion slower, never a jump.
        move["progress"] += min(max(now - move["last"], 0.0), 2 * self.control_dt) / move["duration"]
        move["last"] = now
        t = min(move["progress"] + 1e-6, 1.0)
        (arm_start, arm_target), (gripper_start, gripper_target) = move["arm"], move["gripper"]
        w = _smoothstep(t)
        arm = arm_start + w * (arm_target - arm_start)
        move_time = max(2 * self.control_dt, 0.05)
        self.bot.arm.publish_positions(list(arm), moving_time=move_time, accel_time=move_time / 2, blocking=False)
        self.gripper_command.cmd = float(gripper_start + w * (gripper_target - gripper_start))
        self.bot.gripper.core.pub_single.publish(self.gripper_command)
        if t >= 1.0:
            self._move = None
            self.state = self.LOCKED

    def _release_torque(self):
        torque_off(self.bot)
        if self._ramp["steps"] and self.p_gains:
            # The torque is off, so restoring the gains moves nothing; the next torque_on() finds
            # the arm stiff again.
            self._set_p_gains(self.p_gains)
            self._ramp["gains_restored"] = True

    def _tick_release(self, now):
        ramp = self._ramp
        if ramp["pending"] is not None:  # previous register write still running on the worker
            if not ramp["pending"].done():
                return
            ramp["pending"].result()  # re-raise a failed service call here
            ramp["pending"] = None
        if not ramp["released"]:
            if ramp["steps"] and self.p_gains:
                # One gain step per interval, never two at once: slow service calls or a stalled
                # loop stretch the ramp instead of dropping the gain to 0; after the last (0) step,
                # wait one more interval.
                if now + 1e-6 < ramp["next"]:
                    return
                if ramp["done"] < ramp["steps"]:
                    ramp["done"] += 1
                    scale = 1.0 - ramp["done"] / ramp["steps"]
                    ramp["pending"] = self._io.submit(self._set_p_gains,
                                                      [round(gain * scale) for gain in self.p_gains])
                    ramp["next"] = now + P_GAIN_STEP_S
                    return
            ramp["released"] = True
            ramp["pending"] = self._io.submit(self._release_torque)
            return
        if self.haptics is not None:
            self.haptics.start()
        self._ramp = None
        if self.relative and self.reference is not None:
            self._window = []
            self._settle_start = now
            self._settle_stats = {"steady_s": 0.0, "span": 0.0, "joint": 0, "elapsed_s": 0.0}
            self.state = self.SETTLING
        else:
            self.state = self.FREE

    def _tick_settle(self, now):
        arm, _ = self.read()
        self._window = [(t, q) for t, q in self._window if now - t <= self.settle_s] + [(now, arm)]
        span = np.ptp(np.stack([q for _, q in self._window]), axis=0)
        settled = (now - self._window[0][0] >= self.settle_s * SETTLE_FRACTION
                   and bool((span <= self.settle_tolerance).all()))
        timed_out = now - self._settle_start >= self.settle_timeout_s
        joint = int(np.argmax(span))
        self._settle_stats = {"steady_s": now - self._steady_since(), "span": float(span[joint]), "joint": joint,
                              "elapsed_s": now - self._settle_start}
        if self.haptics is not None:
            self.haptics.update()
        if not (settled or timed_out):
            return
        self.offset = arm - self.reference
        self.last_arm_action = self.reference.copy()
        self.engaged = True
        self.state = self.FREE
        worst = int(np.argmax(np.abs(self.offset)))
        message = "{} leader: relative control from here ({} after {:.2f}s, largest offset {:+.3f} rad on joint {})".format(
            self.side, "settled" if settled else "settle timeout", now - self._settle_start, self.offset[worst], worst)
        if abs(self.offset[worst]) > self.max_offset_warn:
            logging.warning(message + " - leader and follower poses differ visibly; pause to re-align.")
        else:
            logging.info(message)

    def _steady_since(self):
        """Time of the oldest reading of the newest run in which every joint stayed within
        settle_tolerance (the window holds at most settle_s of readings)."""
        low = high = None
        since = self._window[-1][0]
        for stamp, arm in reversed(self._window):
            low = arm if low is None else np.minimum(low, arm)
            high = arm if high is None else np.maximum(high, arm)
            if bool((high - low > self.settle_tolerance).any()):
                break
            since = stamp
        return since

    def _joint_name(self, joint):
        names = getattr(getattr(self.bot.arm, "group_info", None), "joint_names", None) or ()
        return names[joint] if joint < len(names) else "joint {}".format(joint)

    def takeover_progress(self):
        """For the operator display while this leader gets ready for a take-over: None unless it is
        releasing or settling, else a dict with "stage", "side", "share" (0..1 within the stage) and
        the stage's numbers. Display only - nothing here feeds back into the control."""
        if self.state == self.RELEASING and self._ramp is not None:
            ramp = self._ramp
            share = 1.0 if ramp["released"] else (ramp["done"] / ramp["steps"] if ramp["steps"] else 0.0)
            return {"stage": "releasing", "side": self.side, "share": share,
                    "done": ramp["done"], "steps": ramp["steps"]}
        if self.state == self.SETTLING and self._settle_stats is not None:
            stats = self._settle_stats
            return {"stage": "settling", "side": self.side,
                    "share": min(1.0, stats["steady_s"] / self.settle_s),
                    "steady_s": stats["steady_s"], "need_s": self.settle_s,
                    "span": stats["span"], "tolerance": self.settle_tolerance,
                    "joint_name": self._joint_name(stats["joint"]),
                    "elapsed_s": stats["elapsed_s"], "timeout_s": self.settle_timeout_s}
        return None

    def lock(self):
        """Torque the leader on where it is. Ends a clutch-style correction: the offset is
        cleared and the follower's last command becomes the reference for the next release."""
        self._move = None
        ramp, self._ramp = self._ramp, None
        if ramp is not None and ramp["pending"] is not None:
            try:
                ramp["pending"].result(timeout=5.0)  # finish the write in flight before re-aiming
            except Exception as exc:
                logging.warning("%s leader: register write failed during release: %s", self.side, exc)
        softened = ramp is not None and ramp["done"] > 0 and bool(self.p_gains) and not ramp["gains_restored"]
        if self.engaged and self.last_arm_action is not None:
            self.reference = self.last_arm_action.copy()
        self.engaged = False
        self.offset = np.zeros(6)
        if self.haptics is not None:
            self.haptics.restore()
        # The goal registers still hold the pose the arm was released at; aim them at the present
        # pose first so torquing on cannot pull the arm back there.
        arm, gripper = self.read()
        move_time = max(2 * self.control_dt, 0.05)
        self.bot.arm.publish_positions(list(arm), moving_time=move_time, accel_time=move_time / 2, blocking=False)
        self.gripper_command.cmd = gripper
        self.bot.gripper.core.pub_single.publish(self.gripper_command)
        if softened:
            # Interrupted mid-ramp (gains lowered, arm sagging): stiffen only after the goal above
            # points at the present pose, or the arm would snap back to the old one.
            time.sleep(2 * self.control_dt)  # let the driver apply the goal topic first
            self._set_p_gains(self.p_gains)
        torque_on(self.bot)
        self.state = self.LOCKED

    def close(self):
        """Stop this arm's register-write worker (call after lock())."""
        if self._own_io:
            self._io.shutdown(wait=True)


class ClutchLeaderPair:
    """Both leaders, driven together (follower action layout: left 7 then right 7)."""

    def __init__(self, left, right):
        self.arms = (left, right)

    def configure(self):
        for arm in self.arms:
            arm.configure()

    @property
    def busy(self):
        return any(arm.busy for arm in self.arms)

    @property
    def released(self):
        return all(arm.state == ClutchLeader.FREE for arm in self.arms)

    @property
    def locked(self):
        return all(arm.state == ClutchLeader.LOCKED for arm in self.arms)

    def begin_move(self, follower_command, duration_s, now):
        for index, arm in enumerate(self.arms):
            part = follower_command[index * 7:(index + 1) * 7]
            arm.begin_move(part[:6], part[6], duration_s, now)

    def stop_move(self):
        for arm in self.arms:
            arm.stop_move()

    def begin_release(self, now):
        for arm in self.arms:
            arm.begin_release(now)

    def tick(self, now):
        for arm in self.arms:
            arm.tick(now)

    def lock(self):
        for arm in self.arms:
            arm.lock()

    def close(self):
        for arm in self.arms:
            arm.close()

    def action(self):
        return np.concatenate([arm.action() for arm in self.arms])

    def raw_action(self):
        return np.concatenate([arm.raw_action() for arm in self.arms])

    def offsets(self):
        return np.concatenate([arm.offset for arm in self.arms])

    def takeover_progress(self):
        """The arm furthest from ready (still releasing before settling, then the smaller steady
        share): the take-over only starts once both are free. None when neither is preparing."""
        stage_order = {"releasing": 0, "settling": 1}
        shots = [shot for shot in (arm.takeover_progress() for arm in self.arms) if shot is not None]
        return min(shots, key=lambda shot: (stage_order[shot["stage"]], shot["share"])) if shots else None


class FollowerCommander:
    """One command per tick to both followers, with joint-limit clamp, per-tick step limit and an
    optional blend that starts from the last command sent (the measured pose only before any)."""

    def __init__(self, puppet_bots, max_joint_step=0.1, clip_gripper=False, clock=time.monotonic):
        self.bots = tuple(puppet_bots)
        self.max_joint_step = max_joint_step
        self.clip_gripper = clip_gripper
        self.clock = clock
        self.last = None  # last command sent (14, follower convention)
        self._last_arm_joints = None  # last arm targets actually published (rad, 2 x 6)
        self._last_gripper_joints = None  # last gripper joint targets published (rad, 2)
        self._blend_s = 0.0
        self._blend_pending = False
        self._blend_from = None
        self._blend_end = 0.0
        self._gripper_commands = [JointSingleCommand(name="gripper") for _ in self.bots]
        self._reported_limits = set()
        self._last_step_warning = 0.0

    def measured(self):
        """The followers' measured pose in the command convention (gripper normalized by the same
        endpoints the command uses)."""
        out = np.empty(14)
        for index, (bot, side) in enumerate(zip(self.bots, SIDES)):
            position = bot.dxl.joint_states.position
            close, open_ = PUPPET_GRIPPER_RANGE[side]
            out[index * 7:index * 7 + 6] = position[:6]
            out[index * 7 + 6] = (position[6] - close) / (open_ - close)
        return out

    def current(self):
        """What the followers hold: the last command, or the measured pose before any command."""
        return self.last.copy() if self.last is not None else self.measured()

    def begin_blend(self, duration_s):
        """Ease the next commands in over duration_s, starting from the last command sent.

        Not from the measured pose: the position servos hold the arm a little below its goal against
        gravity (P control keeps a steady-state error, ~0.03 rad on a shoulder), so a blend that
        starts at the measured pose first commands "stay where you are", the holding error vanishes
        and the arm sags before the blend lifts it again. The measured pose is only used before any
        command was sent."""
        if duration_s > 0:
            self._blend_s = duration_s
            self._blend_pending = True

    def send(self, command):
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (14,) or not np.isfinite(command).all():
            raise ValueError("follower command must be 14 finite values, got {}".format(command))
        now = self.clock()
        arms = [command[0:6].copy(), command[7:13].copy()]
        grippers = [float(command[6]), float(command[13])]
        if self.clip_gripper:
            grippers = [min(max(g, 0.0), 1.0) for g in grippers]
        gripper_joints = [PUPPET_GRIPPER_UNNORMALIZE[side](g) for side, g in zip(SIDES, grippers)]

        for index, bot in enumerate(self.bots):
            lower = np.asarray(bot.arm.group_info.joint_lower_limits[:6])
            upper = np.asarray(bot.arm.group_info.joint_upper_limits[:6])
            clamped = np.clip(arms[index], lower, upper)
            new = [j for j in np.flatnonzero(clamped != arms[index]) if (index, j) not in self._reported_limits]
            if new:
                logging.warning("%s follower target clamped to joint limits on joints %s (repeats suppressed)",
                                SIDES[index], new)
                self._reported_limits.update((index, j) for j in new)
            arms[index] = clamped

        if self._blend_pending:
            if self._last_arm_joints is not None and self._last_gripper_joints is not None:
                self._blend_from = [np.concatenate([arm, [gripper]])
                                    for arm, gripper in zip(self._last_arm_joints, self._last_gripper_joints)]
            else:
                self._blend_from = [np.array(bot.dxl.joint_states.position[:7], dtype=np.float64)
                                    for bot in self.bots]
            self._blend_end = now + self._blend_s
            self._blend_pending = False
        if self._blend_from is not None and now < self._blend_end:
            w = 1.0 - (self._blend_end - now) / self._blend_s
            for index in range(2):
                arms[index] = self._blend_from[index][:6] + w * (arms[index] - self._blend_from[index][:6])
                gripper_joints[index] = self._blend_from[index][6] + w * (gripper_joints[index] - self._blend_from[index][6])
        else:
            self._blend_from = None

        if self.max_joint_step is not None:
            if self._last_arm_joints is None:
                self._last_arm_joints = [np.array(bot.dxl.joint_states.position[:6], dtype=np.float64)
                                         for bot in self.bots]
            for index in range(2):
                delta = arms[index] - self._last_arm_joints[index]
                limited = np.clip(delta, -self.max_joint_step, self.max_joint_step)
                if (limited != delta).any() and now - self._last_step_warning > 1.0:
                    self._last_step_warning = now
                    logging.warning("%s follower: command jumped %.3f rad in one tick; limited to %.3f",
                                    SIDES[index], float(np.abs(delta).max()), self.max_joint_step)
                arms[index] = self._last_arm_joints[index] + limited

        for index, bot in enumerate(self.bots):
            bot.arm.publish_positions(list(arms[index]), blocking=False)
            self._gripper_commands[index].cmd = float(gripper_joints[index])
            bot.gripper.core.pub_single.publish(self._gripper_commands[index])
        self._last_arm_joints = [a.copy() for a in arms]
        self._last_gripper_joints = [float(g) for g in gripper_joints]

        sent = np.empty(14)
        for index, side in enumerate(SIDES):
            close, open_ = PUPPET_GRIPPER_RANGE[side]
            sent[index * 7:index * 7 + 6] = arms[index]
            sent[index * 7 + 6] = (gripper_joints[index] - close) / (open_ - close)
        self.last = sent
        return sent.copy()


class HilInput:
    """Key/pedal events: 'pause' (RIGHT), 'correct' (LEFT), 'end' (e/Enter), 'esc' (ESC/q),
    'rezero' (b). The pedal maps evdev key codes onto the same events."""

    def __init__(self, pedal_device=None, pedal_keys=None):
        from one_side_teleop import KeyboardControl

        self.keyboard = _make_keyboard(KeyboardControl)
        self._pedal_events = queue.Queue()
        self.pedal_thread = None
        if pedal_device:
            self.pedal_thread = _start_pedal(pedal_device, pedal_keys or {}, self._pedal_events)

    @property
    def enabled(self):
        return self.keyboard.enabled or self.pedal_thread is not None

    def poll(self):
        events = []
        for event in self.keyboard.poll():
            events.append({"right": "pause", "left": "correct"}.get(event, event))
        while True:
            try:
                events.append(self._pedal_events.get_nowait())
            except queue.Empty:
                return events

    def drain(self):
        self.keyboard.drain()
        while not self._pedal_events.empty():
            self._pedal_events.get_nowait()

    def close(self):
        self.keyboard.close()


def _make_keyboard(keyboard_class):
    """KeyboardControl that also reports 'end' for e/E/Enter (its parser, plus those keys)."""

    class HilKeyboard(keyboard_class):
        def _parse(self):
            events = []
            while self._buffer:
                if not self._buffer.startswith(b"\x1b"):
                    key = self._buffer[:1]
                    self._buffer = self._buffer[1:]
                    if key in (b"q", b"Q"):
                        events.append("esc")
                    elif key in (b"b", b"B"):
                        events.append("rezero")
                    elif key in (b"e", b"E", b"\n", b"\r"):
                        events.append("end")
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
                if len(self._buffer) >= 2 and self._buffer[1:2] not in (b"[", b"O"):
                    # ESC followed by anything else (ESC then a key within the hold-off, Alt+key):
                    # a real ESC (same as KeyboardControl._parse).
                    self._buffer = self._buffer[1:]
                    events.append("esc")
                    continue
                if len(self._buffer) == 1 and time.time() - self._buffer_time > self.ESC_HOLDOFF_S:
                    self._buffer = b""
                    events.append("esc")
                    continue
                break  # incomplete sequence: leave it buffered for the next poll
            return events

    return HilKeyboard()


def _start_pedal(device_path, key_to_event, events):
    try:
        from evdev import InputDevice, categorize, ecodes
    except ImportError:
        logging.warning("evdev is not installed; the foot pedal is disabled")
        return None

    def reader():
        try:
            device = InputDevice(device_path)
            try:
                # Exclusive: otherwise the pedal also types its keys (e.g. 'b' = re-zero) into the
                # focused terminal, where the keyboard reader picks them up a second time.
                device.grab()
            except OSError as exc:
                logging.warning("Could not grab the pedal (%s); its keys also reach the terminal", exc)
            logging.info("Pedal connected: %s", device.name)
            for event in device.read_loop():
                if event.type != ecodes.EV_KEY:
                    continue
                key = categorize(event)
                if key.keystate != 1:  # key-down only
                    continue
                codes = key.keycode if isinstance(key.keycode, (list, tuple)) else [key.keycode]
                for code in codes:
                    if code in key_to_event:
                        events.put(key_to_event[code])
                        break
        except (OSError, PermissionError) as exc:
            logging.warning("Pedal %s unavailable: %s", device_path, exc)

    thread = threading.Thread(target=reader, name="hil-pedal", daemon=True)
    thread.start()
    return thread
