import os,sys
import pickle
import threading
import time
import h5py
import argparse
import h5py_cache
import numpy as np
from tqdm import tqdm
import cv2

from constants import DT, START_ARM_POSE, TASK_CONFIGS, DT
from constants import MASTER_LEFT_GRIPPER_JOINT_MID, MASTER_LEFT_GRIPPER_JOINT_CLOSE
from constants import MASTER_RIGHT_GRIPPER_JOINT_MID, MASTER_RIGHT_GRIPPER_JOINT_CLOSE
from constants import PUPPET_LEFT_GRIPPER_JOINT_CLOSE, PUPPET_LEFT_GRIPPER_JOINT_OPEN
from constants import PUPPET_RIGHT_GRIPPER_JOINT_CLOSE, PUPPET_RIGHT_GRIPPER_JOINT_OPEN
from robot_utils import Recorder, ImageRecorder, get_arm_gripper_positions
from robot_utils import move_arms, torque_on, torque_off, move_grippers
from real_env import make_real_env, get_action

import rospy
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from std_msgs.msg import UInt8MultiArray
from sleep_plus import sleep_all_robots,shut_down_all_robots

# Leader-gripper haptics, imported rather than copied so both scripts share one implementation
# and one set of tuned defaults: retune one_side_teleop.py and recording follows automatically.
from one_side_teleop import (
    ClosingSpeedBrake,
    GripperForceFeedback,
    KeyboardControl,
    MasterGripperEffort,
    get_arm_gripper_velocity,
    master_gripper_pose,
    master_to_puppet_gripper,
    set_arm_position_time_profile,
    wait_for_start,
    DEFAULT_CLOSE_BRAKE_GAIN,
    DEFAULT_CLOSE_BRAKE_MAX_PWM,
    DEFAULT_CLOSE_SPEED_LIMIT,
    DEFAULT_FORCE_BASELINE_SAMPLES,
    DEFAULT_FORCE_CURVE,
    DEFAULT_FORCE_DEADBAND,
    DEFAULT_FORCE_FIELD,
    DEFAULT_FORCE_FILTER,
    DEFAULT_FORCE_FULL_SCALE,
    DEFAULT_FORCE_RAMP_S,
    DEFAULT_FORCE_TIMEOUT_S,
    DEFAULT_MAX_FEEDBACK_PWM,
    DEFAULT_PREP_ARM_ACCEL_TIME,
    DEFAULT_PREP_ARM_MOVE_TIME,
    DEFAULT_START_OPEN_RATIO,
    DEFAULT_TACTILE_TOPIC_TEMPLATE,
    DEFAULT_VELOCITY_FILTER,
    DEFAULT_WALL_STIFFNESS_MAX,
    DEFAULT_WALL_STIFFNESS_MIN,
    FEEDBACK_LIMIT_MARGIN,
    XC430_PWM_LIMIT,
)
import IPython
e = IPython.embed


class ArmHaptics:
    """Force feedback and closing-speed brake for one leader gripper.

    Bundles what one_side_teleop drives inline - GripperForceFeedback (tactile normal force
    rendered as a wall anchored at the contact point), ClosingSpeedBrake (resists only the fast
    part of a close) and MasterGripperEffort (PWM out, torque dropped when the command is 0) -
    so the bimanual recording loop can carry one of these per arm.
    """

    def __init__(self, master_bot, robot_side, feedback_config, brake_config, torque_gate=True):
        self.master_bot = master_bot
        self.robot_side = robot_side
        self.feedback = GripperForceFeedback(**feedback_config)
        self.brake = ClosingSpeedBrake(**brake_config)
        self.torque_gate = torque_gate
        self.effort = None
        self.open_limit = master_gripper_pose(robot_side, 1.0) - FEEDBACK_LIMIT_MARGIN
        self.close_limit = master_gripper_pose(robot_side, 0.0) + FEEDBACK_LIMIT_MARGIN
        self.pwm = 0
        self.force = 0.0
        self.age = float("inf")
        self.depth = 0.0
        self.brake_pwm = 0

    def wait_for_data(self, timeout_s=2.0):
        """Give the tactile stream a moment to arrive; warn rather than block if it does not."""
        deadline = time.time() + timeout_s
        while self.feedback.frames == 0 and time.time() < deadline:
            time.sleep(0.05)
        if self.feedback.frames == 0:
            print(f"Warning: no tactile data on {self.feedback.topic} yet. The {self.robot_side} "
                  f"leader gripper will stay at 0 PWM until it arrives.")
            return False
        return True

    def calibrate(self, baseline_samples):
        """Zero the resting force. Call this before the leader is handed to the operator."""
        self.feedback.capture_baseline(baseline_samples)

    def start(self):
        """Switch the leader gripper to PWM mode and fade the wall in. Call calibrate() first."""
        self.effort = MasterGripperEffort(self.master_bot, torque_gate=self.torque_gate)
        self.feedback.activate()

    def update(self):
        """One control cycle: tactile + speed -> Goal_PWM on the leader gripper."""
        if self.effort is None:
            return
        position = self.master_bot.dxl.joint_states.position[6]
        velocity = get_arm_gripper_velocity(self.master_bot)
        pwm, force, age, depth = self.feedback.command(position)
        brake_pwm = self.brake.command(velocity)
        pwm += self.feedback.sign * brake_pwm
        # Guard the hard stops, but not while the handle is already moving away from them.
        moving_open = velocity is not None and velocity > 0
        if pwm > 0 and position >= self.open_limit and not moving_open:
            pwm = 0
        elif pwm < 0 and position <= self.close_limit:
            pwm = 0
        pwm = int(max(-XC430_PWM_LIMIT, min(XC430_PWM_LIMIT, pwm)))
        self.effort.send(pwm)
        self.pwm, self.force, self.age, self.depth, self.brake_pwm = pwm, force, age, depth, brake_pwm

    def pause(self):
        self.feedback.deactivate()
        if self.effort is not None:
            self.effort.send(0)

    def resume(self):
        self.feedback.activate()

    def rezero(self, baseline_samples):
        self.feedback.capture_baseline(baseline_samples)

    def restore(self):
        """Zero the effort and hand the leader gripper back to position mode."""
        if self.effort is not None:
            self.effort.restore()
            self.effort = None

    def status(self):
        state = "contact" if self.feedback.contact else "free"
        return (f"{self.robot_side[:1].upper()}[{state:7s} f={self.force:6.3f} "
                f"d={self.depth:5.3f} v={self.brake.speed:4.2f} pwm={self.pwm:+4d}]")


class TactileRecorder:
    """Buffers selected fields from selected XENSE sensors, one sample per recorded timestep.

    Each /tactile/<name>/raw message is a pickled {"frames": {...}, "meta": {...}} bundle from
    4arms_tactile_teleop.launch. Only the fields the task config asks for are kept, because the
    cost of them differs by three orders of magnitude: the 35x20 fields (force, mesh3d, ...) are
    a few KB per frame, while rectify/difference/depth are ~1 MB per frame per sensor and
    dominate both RAM during the episode and the file afterwards. estimate() spells that out
    before recording starts so the operator sees the bill first.

    Sampling is pull-based: the callback only keeps the newest bundle, and sample() copies it
    into the buffers once per recorded timestep, so the tactile stream (60Hz) and the recording
    loop (~50Hz) do not have to agree on a rate.
    """

    # uint8 BGR frames, JPEG'd on write exactly like the camera images
    IMAGE_FIELDS = ("rectify", "difference")

    def __init__(self, sensor_names, field_names):
        self.sensor_names = list(sensor_names)
        self.field_names = list(field_names)
        self.buffers = {name: {field: [] for field in self.field_names} for name in self.sensor_names}
        self.missing = {name: 0 for name in self.sensor_names}
        self._latest = {name: None for name in self.sensor_names}
        self._lock = threading.Lock()
        self._subs = []
        for name in self.sensor_names:
            topic = f"/tactile/{name}/raw"
            self._subs.append(rospy.Subscriber(
                topic, UInt8MultiArray, self._callback, callback_args=name,
                queue_size=1, buff_size=2 ** 24, tcp_nodelay=True))
            print(f"Recording tactile {name} -> {topic}: {', '.join(self.field_names)}")

    def _callback(self, msg, name):
        try:
            payload = pickle.loads(bytes(msg.data))
        except Exception as exc:
            rospy.logwarn_throttle(5.0, f"tactile recorder: cannot unpickle {name}: {exc}")
            return
        frames = payload.get("frames", {})
        # Copy only what is wanted; the bundle's other entries can be ~1MB each.
        sample = {}
        for field in self.field_names:
            value = frames.get(field)
            if value is None:
                continue
            array = np.array(value, copy=True)
            if field in self.IMAGE_FIELDS and array.ndim == 3 and array.shape[-1] == 3:
                # The XENSE SDK's own type stub claims "rectify"/"difference" are BGR, but the
                # actual bytes off the publisher are RGB (confirmed against XenseStudio's
                # reference rendering). cv2.imencode('.jpg', ...) below assumes BGR, so convert
                # here or every saved tactile frame comes out R/B-swapped.
                array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            sample[field] = array
        with self._lock:
            self._latest[name] = sample

    def wait_for_data(self, timeout_s=3.0):
        """Return the sensors that produced nothing within the timeout."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                silent = [name for name, sample in self._latest.items() if sample is None]
            if not silent:
                return []
            time.sleep(0.05)
        return silent

    def check_fields(self):
        """Report fields that were asked for but never showed up in a bundle."""
        problems = {}
        with self._lock:
            for name, sample in self._latest.items():
                if sample is None:
                    continue
                absent = [field for field in self.field_names if field not in sample]
                if absent:
                    problems[name] = absent
        return problems

    def sample(self):
        """Take one sample per sensor. Call once per recorded timestep."""
        with self._lock:
            for name in self.sensor_names:
                latest = self._latest[name]
                if latest is None:
                    self.missing[name] += 1
                for field in self.field_names:
                    self.buffers[name][field].append(None if latest is None else latest.get(field))

    def estimate(self, n_steps):
        """Human-readable RAM/disk estimate for an episode of n_steps, from the live shapes."""
        with self._lock:
            samples = {name: sample for name, sample in self._latest.items() if sample}
        if not samples:
            return "tactile size unknown (no data yet)"
        total = 0
        parts = []
        for name, sample in samples.items():
            for field, value in sample.items():
                total += value.nbytes * n_steps
            parts.append(name)
        return (f"tactile buffer for {n_steps} steps x {len(parts)} sensors: "
                f"~{total / 1e9:.2f} GB before compression")

    def pack(self, episode_len, jpeg_quality=50, compress=True):
        """Turn the buffers into HDF5-ready arrays.

        Returns (packed, lengths):
          packed  = {sensor: {field: {"data": ndarray, "compressed": bool, "shape": tuple}}}
          lengths = {"entries": ["<sensor>/<field>", ...], "lens": (n_entries, T) int64}

        With compress=True the image fields are JPEG-encoded and zero-padded to ONE width shared
        by every tactile image dataset, with the true byte counts gathered into a single table -
        the same layout the camera images use, so both can be read the same way. JPEG output is
        variable length while HDF5 rows are not, hence the padding and the table. With
        compress=False they are stored as raw (T, H, W, 3) uint8 instead, which is ~140x larger
        and needs a second full-size array while packing. Everything else becomes float32.
        Timesteps where a sensor had not published yet are zero-filled.
        """
        packed = {}
        pending = []  # (sensor, field, encoded frames, lens, original shape)
        for name in self.sensor_names:
            packed[name] = {}
            for field in self.field_names:
                frames = self.buffers[name][field][:episode_len]
                template = next((f for f in frames if f is not None), None)
                if template is None:
                    print(f"Warning: tactile {name}/{field} never produced data; skipping it.")
                    continue
                filled = [f if f is not None else np.zeros_like(template) for f in frames]
                if field in self.IMAGE_FIELDS and template.dtype == np.uint8 and compress:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                    encoded = [cv2.imencode('.jpg', f, encode_param)[1] for f in filled]
                    lens = np.array([len(e) for e in encoded], dtype=np.int64)
                    pending.append((name, field, encoded, lens, template.shape))
                elif field in self.IMAGE_FIELDS and template.dtype == np.uint8:
                    packed[name][field] = {
                        "data": np.asarray(filled, dtype=np.uint8),
                        "compressed": False,
                        "shape": template.shape,
                    }
                else:
                    packed[name][field] = {
                        "data": np.asarray(filled, dtype=np.float32),
                        "compressed": False,
                        "shape": template.shape,
                    }

        padded_size = int(max((entry[3].max() for entry in pending), default=0))
        entries, all_lens = [], []
        for name, field, encoded, lens, shape in pending:
            padded = np.zeros((len(encoded), padded_size), dtype='uint8')
            for i, e in enumerate(encoded):
                padded[i, :len(e)] = e.reshape(-1)
            packed[name][field] = {"data": padded, "compressed": True, "shape": shape}
            entries.append(f"{name}/{field}")
            all_lens.append(lens)
        lengths = {
            "entries": entries,
            "lens": np.stack(all_lens) if all_lens else np.zeros((0, episode_len), dtype=np.int64),
        }
        return packed, lengths

    def close(self):
        for sub in self._subs:
            sub.unregister()
        self._subs = []


def wait_for_decision(keyboard, prompt):
    """Block until the operator says what to do with this episode.

    RIGHT saves it and moves on, LEFT throws it away and records the same index again, ESC
    aborts the whole run. Without a tty there is nobody to ask, so the episode is saved.
    """
    if not keyboard.enabled:
        print("No tty on stdin, so there are no key controls: saving this episode.")
        return "save"
    print(prompt)
    keyboard.drain()
    while True:
        for event in keyboard.poll():
            if event == "right":
                return "save"
            if event == "left":
                return "redo"
            if event == "esc":
                return "quit"
        time.sleep(0.02)


def build_haptics_config(args, robot_side):
    """Feedback and brake constructor kwargs for one side, from the CLI plus tuned defaults."""
    feedback_config = dict(
        topic=DEFAULT_TACTILE_TOPIC_TEMPLATE.format(side=robot_side),
        # Normal force comes from force[..., 2], not from the SDK's separate force_norm copy.
        field=DEFAULT_FORCE_FIELD,
        deadband=args["force_deadband"],
        full_scale=args["force_full_scale"],
        curve=DEFAULT_FORCE_CURVE,
        max_pwm=args["max_feedback_pwm"],
        timeout_s=DEFAULT_FORCE_TIMEOUT_S,
        filter_alpha=DEFAULT_FORCE_FILTER,
        metric="abs_sum",
        sign=1,
        ramp_s=DEFAULT_FORCE_RAMP_S,
        mode="wall",
        stiffness_min=args["wall_stiffness_min"],
        stiffness_max=args["wall_stiffness_max"],
    )
    brake_config = dict(
        speed_limit=args["close_speed_limit"],
        gain=args["close_brake_gain"],
        max_pwm=DEFAULT_CLOSE_BRAKE_MAX_PWM,
        filter_alpha=DEFAULT_VELOCITY_FILTER,
    )
    return feedback_config, brake_config


def opening_ceremony(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right,
                     dataset_name, start_open_ratio=DEFAULT_START_OPEN_RATIO):
    """ Move all 4 robots to a pose where it is easy to start demonstration """
    # reboot gripper motors, and set operating modes for all motors

    puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)

    # puppet_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    # puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    # master_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    # master_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
    # # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit
    # puppet_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    # puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    # master_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    # master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
    # # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit

    # threading
    import threading
    def set_operating_modes(robot, arm_mode, gripper_mode):
        # arm_mode is always "position" at every call site below; what actually matters here is
        # the profile TYPE. robot_set_operating_modes() defaults profile_type to "velocity" with
        # profile_velocity=0 (i.e. no speed limit) when not told otherwise, so leaving it at that
        # default put every arm into raw max-speed velocity-profile mode right before move_arms()
        # below - and, since nothing later in this process ever switches it back, every later
        # move_arms() call in this run (including the sleep/shutdown ones in sleep_plus.py) kept
        # running unthrottled too. one_side_teleop.py avoids this by forcing time-based-profile
        # mode via set_arm_position_time_profile() before it ever calls set_joint_positions();
        # doing the same here is what actually fixes the "moves too fast" report, not just
        # matching its move_time value.
        set_arm_position_time_profile(robot, DEFAULT_PREP_ARM_MOVE_TIME, DEFAULT_PREP_ARM_ACCEL_TIME)
        robot.dxl.robot_set_operating_modes("single", "gripper", gripper_mode)
    def configure_robots(puppet_bot_left, master_bot_left, puppet_bot_right, master_bot_right):
        threads = [
            threading.Thread(target=set_operating_modes, args=(puppet_bot_left, "position", "current_based_position")),
            threading.Thread(target=set_operating_modes, args=(master_bot_left, "position", "position")),
            threading.Thread(target=set_operating_modes, args=(puppet_bot_right, "position", "current_based_position")),
            threading.Thread(target=set_operating_modes, args=(master_bot_right, "position", "position"))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    configure_robots(puppet_bot_left, master_bot_left, puppet_bot_right, master_bot_right)

    torque_on(puppet_bot_left)
    torque_on(master_bot_left)
    torque_on(puppet_bot_right)
    torque_on(master_bot_right)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [start_arm_qpos] * 4, move_time=DEFAULT_PREP_ARM_MOVE_TIME)
    # move grippers to starting position, the same way one_side_teleop does it: each master parks
    # at start_open_ratio (fully open by default) and each puppet at exactly what that maps to, so
    # the two already agree and nothing jumps when teleoperation begins.
    master_left_start = master_gripper_pose("left", start_open_ratio)
    master_right_start = master_gripper_pose("right", start_open_ratio)
    puppet_left_start = master_to_puppet_gripper("left", master_left_start)
    puppet_right_start = master_to_puppet_gripper("right", master_right_start)
    print(f"Gripper start poses (mapped): master L {master_left_start:.4f} / R {master_right_start:.4f}, "
          f"puppet L {puppet_left_start:.4f} / R {puppet_right_start:.4f}.")
    move_grippers(
        [master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right],
        [master_left_start, puppet_left_start, master_right_start, puppet_right_start],
        move_time=0.5,
    )
    # move_arms() above left every puppet arm's Profile_Velocity/Acceleration registers at the
    # slow 2.0s "walk to the start pose" duration. Teleoperation drives the puppet with per-tick
    # RealEnv.step() -> set_joint_positions(..., moving_time=None) calls (see real_env.py), which
    # reuse whatever moving_time is already cached - so without this, every 20ms tick would ask
    # the servo for a fresh "reach this in 2 seconds" plan and the arm would barely creep, unable
    # to track the master in real time. one_side_teleop.py's prep_robots() shrinks this back down
    # to teleop cadence (DT) right after reaching the start pose for exactly this reason; do the
    # same here, with set_trajectory_time() (a RAM register write) rather than
    # set_arm_position_time_profile() (which rewrites the EEPROM Operating_Mode and briefly drops
    # torque, letting the arm sag).
    for puppet_bot in (puppet_bot_left, puppet_bot_right):
        puppet_bot.arm.moving_time = None
        puppet_bot.arm.accel_time = None
        puppet_bot.arm.set_trajectory_time(moving_time=DT, accel_time=min(DT, DT / 2))
    # Starting is now a keypress, not a gripper squeeze - the caller does that with a
    # KeyboardControl so the same RIGHT/LEFT/ESC keys work here as in one_side_teleop.


def capture_one_episode(dt, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite,
                        haptics_args=None, tactile_sensors=(), tactile_outputs=()):
    print("*"*100)
    print(f'Dataset name: {dataset_name}')
    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    env = make_real_env(init_node=False, setup_robots=False)

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    haptics_args = haptics_args or {}
    start_open_ratio = haptics_args.get("start_open_ratio", DEFAULT_START_OPEN_RATIO)
    baseline_samples = haptics_args.get("force_baseline_samples", DEFAULT_FORCE_BASELINE_SAMPLES)

    # move all 4 robots to a starting pose where it is easy to start teleoperation
    opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right,
                     dataset_name, start_open_ratio=start_open_ratio)

    keyboard = KeyboardControl()
    haptics = []
    tactile = None
    decision = None
    try:
        if not haptics_args.get("no_force_feedback", False):
            for robot_side, master_bot in (("left", master_bot_left), ("right", master_bot_right)):
                feedback_config, brake_config = build_haptics_config(haptics_args, robot_side)
                arm = ArmHaptics(master_bot, robot_side, feedback_config, brake_config,
                                 torque_gate=not haptics_args.get("no_torque_gate", False))
                haptics.append(arm)
                print(f"Force feedback ({robot_side}): {arm.feedback.describe()}")
            for arm in haptics:
                arm.wait_for_data()
            for arm in haptics:
                arm.calibrate(baseline_samples)

        if tactile_sensors and tactile_outputs:
            tactile = TactileRecorder(tactile_sensors, tactile_outputs)
            silent = tactile.wait_for_data()
            if silent:
                print(f"Warning: no tactile data from {', '.join(silent)}. Those timesteps will "
                      f"be zero-filled - check the launch and 'rostopic hz'.")
            for name, absent in tactile.check_fields().items():
                print(f"Warning: {name} publishes no {', '.join(absent)}. Relaunch with "
                      f'extra_outputs containing them, or drop them from tactile_outputs.')
            print(tactile.estimate(max_timesteps))

        print(f"Ready to record {dataset_name}.")
        if not wait_for_start(keyboard):
            return "quit"

        # Hand both leaders over to the operator, then bring the haptics up.
        torque_off(master_bot_left)
        torque_off(master_bot_right)
        for arm in haptics:
            arm.start()
        if haptics:
            print("Leader grippers are in PWM mode; force feedback fades in now.")
        print("Recording. RIGHT = stop and save, LEFT = stop and re-record, "
              "b = re-zero baselines, ESC = abort the run.")

        # Data collection
        ts = env.reset(fake=True)
        timesteps = [ts]
        actions = []
        actual_dt_history = []
        time0 = time.time()
        last_status = 0.0
        progress = tqdm(total=max_timesteps)
        t = 0
        while t < max_timesteps:
            for event in keyboard.poll():
                if event == "right":
                    decision = "save"
                elif event == "left":
                    decision = "redo"
                elif event == "esc":
                    decision = "quit"
                elif event == "rezero":
                    for arm in haptics:
                        arm.rezero(baseline_samples)
                if decision is not None:
                    break
            if decision is not None:
                progress.write(f"Stopped early at {t}/{max_timesteps} timesteps -> {decision}.")
                break

            t0 = time.time() #
            action = get_action(master_bot_left, master_bot_right)
            t1 = time.time() #
            ts = env.step(action)
            t2 = time.time() #
            timesteps.append(ts)
            actions.append(action)
            actual_dt_history.append([t0, t1, t2])
            if tactile is not None:
                tactile.sample()
            for arm in haptics:
                arm.update()
            if haptics and t2 - last_status >= 1.0:
                last_status = t2
                progress.write("  ".join(arm.status() for arm in haptics))
            time.sleep(max(0, DT - (time.time() - t0)))
            t += 1
            progress.update(1)
        progress.close()
        elapsed = time.time() - time0
        print(f'Avg fps: {t / elapsed if elapsed else 0:.2f} over {t} timesteps')

        # Put the hardware in a safe state before asking anything: haptics off, leaders holding,
        # follower grippers open. Deciding here, rather than after the write, means a re-record
        # costs nothing - the compression and the HDF5 write never happen.
        for arm in haptics:
            arm.restore()
        haptics.clear()
        torque_on(master_bot_left)
        torque_on(master_bot_right)
        env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
        env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
        move_grippers([env.puppet_bot_left, env.puppet_bot_right],
                      [PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN],
                      move_time=0.5)

        if decision is None:
            decision = wait_for_decision(
                keyboard,
                f"{dataset_name} finished ({t} timesteps). "
                f"RIGHT = save and continue, LEFT = re-record, ESC = abort the run.")
    finally:
        for arm in haptics:
            arm.restore()
        keyboard.close()
        if tactile is not None:
            # Unsubscribe here whatever happens: pack() only reads the buffers, so nothing below
            # needs the stream alive, and every early return would otherwise leak a subscriber.
            tactile.close()

    if decision == "quit":
        print(f"Aborted: {dataset_name} was not saved.")
        return "quit"
    if decision == "redo":
        print(f"Discarding {dataset_name}; recording it again.")
        return "redo"

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 25:
        print(f'\n\nfreq_mean is {freq_mean}, lower than 25, re-collecting... \n\n\n\n')
        return "redo"

    # RIGHT can stop an episode before max_timesteps, so everything below is sized to what was
    # actually recorded. The training loader reads each episode's length from the file
    # (utils.py: episode_len = action.shape[0]) and pads to the longest, so short episodes are fine.
    episode_len = len(actions)

    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'
    
    action                  (14,)         'float64'
    base_action             (2,)          'float64'
    """

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        # '/base_action': [],
        # '/base_action_t265': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)

        # data_dict['/base_action'].append(ts.observation['base_vel'])
        # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

    # plot /base_action vs /base_action_t265
    # import matplotlib.pyplot as plt
    # plt.plot(np.array(data_dict['/base_action'])[:, 0], label='base_action_linear')
    # plt.plot(np.array(data_dict['/base_action'])[:, 1], label='base_action_angular')
    # plt.plot(np.array(data_dict['/base_action_t265'])[:, 0], '--', label='base_action_t265_linear')
    # plt.plot(np.array(data_dict['/base_action_t265'])[:, 1], '--', label='base_action_t265_angular')
    # plt.legend()
    # plt.savefig('record_episodes_vel_debug.png', dpi=300)

    COMPRESS = True

    # Defined here, not inside the branch: they are passed to save_hdf5 either way.
    padded_size = None
    compressed_len = None

    if COMPRESS:
        # JPEG compression
        t0 = time.time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] # mobilealoha 50, yay 90, tried as low as 20, seems fine
        compressed_len = []
        for cam_name in camera_names:
            image_list = data_dict[f'/observations/images/{cam_name}']
            compressed_list = []
            compressed_len.append([])
            for image in image_list:
                result, encoded_image = cv2.imencode('.jpg', image, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))
            data_dict[f'/observations/images/{cam_name}'] = compressed_list
        print(f'compression: {time.time() - t0:.2f}s')

        # pad so it has same length
        t0 = time.time()
        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()
        for cam_name in camera_names:
            compressed_image_list = data_dict[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
        print(f'padding: {time.time() - t0:.2f}s')

    tactile_packed, tactile_lengths = {}, {"entries": [], "lens": None}
    if tactile is not None:
        t0 = time.time()
        tactile_packed, tactile_lengths = tactile.pack(episode_len, compress=COMPRESS)
        packed_bytes = sum(entry["data"].nbytes
                           for fields in tactile_packed.values() for entry in fields.values())
        print(f'tactile pack: {time.time() - t0:.2f}s, {packed_bytes / 1e6:.1f} MB')

    # HDF5
    t0 = time.time()

    def save_hdf5(dataset_path, camera_names, episode_len, padded_size, COMPRESS, data_dict,
                  compressed_len, tactile_packed, tactile_lengths):
        """在后台线程中保存 HDF5 数据"""
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = False
            root.attrs['compress'] = COMPRESS
            obs = root.create_group('observations')
            image = obs.create_group('images')

            for cam_name in camera_names:
                if COMPRESS:
                    _ = image.create_dataset(cam_name, (episode_len, padded_size), dtype='uint8',
                                             chunks=(1, padded_size))
                else:
                    _ = image.create_dataset(cam_name, (episode_len, 480, 640, 3), dtype='uint8',
                                             chunks=(1, 480, 640, 3))

            _ = obs.create_dataset('qpos', (episode_len, 14))
            _ = obs.create_dataset('qvel', (episode_len, 14))
            _ = obs.create_dataset('effort', (episode_len, 14))
            _ = root.create_dataset('action', (episode_len, 14))
            _ = root.create_dataset('base_action', (episode_len, 2))
            # _ = root.create_dataset('base_action_t265', (max_timesteps, 2))

            for name, array in data_dict.items():
                root[name][...] = array

            if COMPRESS:
                _ = root.create_dataset('compress_len', (len(camera_names), episode_len))
                root['/compress_len'][...] = compressed_len

            # /observations/tactile/<sensor>/<field>, laid out like the camera images: JPEG
            # frames zero-padded to one shared width, with the true byte counts in a single
            # /observations/tactile/compress_len table whose row order is in its 'entries' attr.
            if tactile_packed:
                tactile_group = obs.create_group('tactile')
                for sensor_name, fields in tactile_packed.items():
                    sensor_group = tactile_group.create_group(sensor_name)
                    for field_name, entry in fields.items():
                        dataset = sensor_group.create_dataset(
                            field_name, data=entry["data"],
                            chunks=(1,) + entry["data"].shape[1:] if entry["data"].ndim > 1 else None)
                        dataset.attrs['compressed'] = entry["compressed"]
                        dataset.attrs['frame_shape'] = np.array(entry["shape"], dtype='int64')
                if tactile_lengths["entries"]:
                    lengths_dataset = tactile_group.create_dataset(
                        'compress_len', data=tactile_lengths["lens"])
                    lengths_dataset.attrs['entries'] = tactile_lengths["entries"]

    # 启动后台线程
    thread = threading.Thread(target=save_hdf5, args=(
    dataset_path, camera_names, episode_len, padded_size, COMPRESS, data_dict, compressed_len,
    tactile_packed, tactile_lengths))
    thread.start()

    # print("HDF5 data storage has started, and the main thread continues executing other tasks...")

    # Wait for the HDF5 thread to complete at some point
    thread.join()
    # print("HDF5 data storage is complete, and the main thread continues executing subsequent code.")

    print(f'Saving: {time.time() - t0:.1f} secs ({episode_len} timesteps)')

    return "save"


def main(args):
    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']
    tactile_sensors = task_config.get('tactile_sensors', [])
    tactile_outputs = task_config.get('tactile_outputs', [])

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True

    dataset_name = f'episode_{episode_idx}'
    print(dataset_name + '\n')
    while True:
        status = capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name,
                                     overwrite, haptics_args=args,
                                     tactile_sensors=tactile_sensors,
                                     tactile_outputs=tactile_outputs)
        if status in ("quit", "save"):
            return status
        # "redo" - chosen with LEFT, or forced by the low-fps check - records the same index again.
        sleep_all_robots()


def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
    return freq_mean

def debug():
    print(f'====== Debug mode ======')
    recorder = Recorder('right', is_debug=True)
    image_recorder = ImageRecorder(init_node=False, is_debug=True)
    while True:
        time.sleep(1)
        recorder.print_diagnostics()
        image_recorder.print_diagnostics()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
#     parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
#     main(vars(parser.parse_args())) # TODO
    # debug()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    parser.add_argument('--start_idx', action='store', type=int, help='Start index.', required=True) # @gnq
    parser.add_argument('--end_idx', action='store', type=int, help='Start index.', required=True)
    # Leader-gripper force feedback. On by default; the defaults come from one_side_teleop.py,
    # so retuning there carries over to recording without touching this file.
    parser.add_argument('--no_force_feedback', action='store_true',
                        help='Disable leader-gripper force feedback for this run.')
    parser.add_argument('--no_torque_gate', action='store_true',
                        help="Keep the leader grippers torqued at 0 PWM (they then brake the handle).")
    parser.add_argument('--start_open_ratio', type=float, default=DEFAULT_START_OPEN_RATIO,
                        help='Where both grippers park before recording, as a fraction of master travel.')
    parser.add_argument('--force_deadband', type=float, default=DEFAULT_FORCE_DEADBAND,
                        help='Force below this commands no PWM at all.')
    parser.add_argument('--force_full_scale', type=float, default=DEFAULT_FORCE_FULL_SCALE,
                        help='Force at which the wall reaches its stiffest.')
    parser.add_argument('--wall_stiffness_min', type=float, default=DEFAULT_WALL_STIFFNESS_MIN,
                        help='PWM per radian at just-detected contact.')
    parser.add_argument('--wall_stiffness_max', type=float, default=DEFAULT_WALL_STIFFNESS_MAX,
                        help='PWM per radian at full-scale force.')
    parser.add_argument('--max_feedback_pwm', type=int, default=DEFAULT_MAX_FEEDBACK_PWM,
                        help=f"Cap on the wall PWM, out of the XC430's {XC430_PWM_LIMIT}.")
    parser.add_argument('--close_speed_limit', type=float, default=DEFAULT_CLOSE_SPEED_LIMIT,
                        help='Closing speed (rad/s) below which nothing is resisted.')
    parser.add_argument('--close_brake_gain', type=float, default=DEFAULT_CLOSE_BRAKE_GAIN,
                        help='PWM per rad/s of closing speed above the limit. 0 disables the brake.')
    parser.add_argument('--force_baseline_samples', type=int, default=DEFAULT_FORCE_BASELINE_SAMPLES,
                        help="Frames averaged to cancel each sensor's resting offset. 0 disables.")
    # main(vars(parser.parse_args()))
    # debug()

    args = vars(parser.parse_args())

    current_episode = args['start_idx']

    while current_episode <= args['end_idx']:
        args['episode_idx'] = current_episode
        decision = main(args)

        if decision == "quit":
            # ESC: nothing was written for this episode, so there is nothing to clean up.
            shut_down_all_robots()
            print(f"Aborted by ESC. The data is saved up to episode {current_episode - 1}.")
            sys.exit("Quitting the process.")

        print("*" * 100)
        print(f"Episode {current_episode} saved.")
        current_episode += 1
        if current_episode <= args['end_idx']:
            sleep_all_robots()

    shut_down_all_robots()
    print("Completed all episodes.")
    print(f"The data is saved to {current_episode - 1} episode.")
