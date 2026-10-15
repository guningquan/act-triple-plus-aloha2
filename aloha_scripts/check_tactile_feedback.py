#!/usr/bin/env python3
"""Live gripper-only teleoperation for tuning leader-gripper force feedback by hand.

Drives ONE side's gripper (master_<side> -> puppet_<side>), exactly like one_side_teleop.py's
haptics stack, but with every "feel" parameter exposed as an on-screen slider so you can dial it
in while actually squeezing the gripper, instead of guessing values and restarting. The arms are
never touched (not moved, not torqued on/off) - make sure they are already resting safely before
running this. Once the feel is right, press 'p' (or just read the "controls" window) to get a
ready-to-paste `--force_deadband ... --force_full_scale ...` flag string for
record_episodes_aloha2.py / record_episodes_aloha2_vtla.py / one_side_teleop.py, which all share
these exact flag names.

Only subscribes to the "force" field - "force_norm" is not needed: it is just
frames["force"][..., 2] (the normal component) published a second time, so the "normal only"
heat map here is sliced from the same "force" payload instead of a second subscription.

    roslaunch aloha 4arms_tactile_teleop.launch extra_outputs:="force"
    python3 check_tactile.py left                     # tune the left gripper
    python3 check_tactile.py right --no_torque_gate    # keep the leader gripper always torqued

Keys (in the image window): q or ESC = quit (restores the leader gripper to position mode),
b = re-zero the baseline (nothing touching the sensor), p = print the current settings.
"""

import argparse
import pickle
import threading
import time
from collections import deque

import cv2
import numpy as np
import rospy
from std_msgs.msg import UInt8MultiArray

from constants import DT
from robot_utils import get_arm_gripper_positions, move_grippers
from one_side_teleop import (
    make_bot, set_puppet_gripper_current_limit, master_gripper_pose, master_to_puppet_gripper,
    get_arm_gripper_effort, get_arm_gripper_velocity,
    GripperForceFeedback, ClosingSpeedBrake, MasterGripperEffort, GripperContactGuard,
    DEFAULT_TACTILE_TOPIC_TEMPLATE, DEFAULT_FORCE_FIELD, DEFAULT_FORCE_TIMEOUT_S,
    DEFAULT_FORCE_FILTER, DEFAULT_FORCE_DEADBAND, DEFAULT_FORCE_FULL_SCALE, DEFAULT_FORCE_CURVE,
    DEFAULT_MAX_FEEDBACK_PWM, DEFAULT_FORCE_RAMP_S, DEFAULT_FORCE_BASELINE_SAMPLES,
    DEFAULT_WALL_STIFFNESS_MIN, DEFAULT_WALL_STIFFNESS_MAX, DEFAULT_CLOSE_SPEED_LIMIT,
    DEFAULT_CLOSE_BRAKE_GAIN, DEFAULT_CLOSE_BRAKE_MAX_PWM, DEFAULT_VELOCITY_FILTER,
    DEFAULT_PUPPET_GRIPPER_CURRENT_LIMIT, DEFAULT_GRIPPER_STALL_ERROR, DEFAULT_GRIPPER_STALL_MOTION,
    DEFAULT_GRIPPER_STALL_CYCLES, DEFAULT_GRIPPER_RELEASE_MARGIN, FEEDBACK_LIMIT_MARGIN,
    XC430_PWM_LIMIT, JointSingleCommand,
)

PANEL_HEIGHT = 480
CONTROL_WINDOW = "haptics controls"
IMAGE_WINDOW = "tactile"
CURVE_WINDOW = "tactile curves (channel means)"
DISPLAY_HZ = 20.0

# One "force" field only - see the module docstring for why force_norm is not needed.
CURVE_FIELDS = ("force",)
CURVE_COLORS = {"force": (80, 255, 80)}  # BGR

# (target attribute owner, attribute name, trackbar label, trackbar max, value = ticks / scale, hint)
SLIDER_SPECS = (
    ("feedback", "deadband", "deadband x1000", 3000, 1000.0,
     "力低于这个值完全不给反馈；越小越灵敏，太小会把噪声当接触"),
    ("feedback", "full_scale", "full_scale x1000", 5000, 1000.0,
     "力到这个值时刚度封顶（不是牛顿，是sensor原始量级）；越小越容易封顶变硬"),
    ("feedback", "curve", "curve x100", 800, 100.0,
     "deadband到full_scale之间变硬的曲线指数；越大轻触越软，重捏才明显变硬"),
    ("feedback", "stiffness_min", "stiffness_min", 20000, 1.0,
     "刚接触瞬间的墙硬度(PWM/rad)；越大一碰就顶手，越小刚接触更柔和"),
    ("feedback", "stiffness_max", "stiffness_max", 30000, 1.0,
     "力到full_scale时的墙硬度上限(PWM/rad)；越大捏到底越像撞硬墙"),
    ("feedback", "max_pwm", "max_feedback_pwm", XC430_PWM_LIMIT, 1.0,
     "反馈PWM硬上限，超过此值不再增大；防止电机堵转过热"),
    ("feedback", "ramp_s", "ramp_s x100", 500, 100.0,
     "每次开始/恢复时反馈淡入的秒数；越大手感越柔和不会突然顶手"),
    ("brake", "speed_limit", "close_speed_limit x100", 1000, 100.0,
     "低于这个闭合速度完全不刹车；正常慢慢捏合不受影响"),
    ("brake", "gain", "close_brake_gain", 3000, 1.0,
     "超过限速后，每超1rad/s多刹多少PWM；0=关闭防甩刹车"),
    ("brake", "max_pwm", "close_brake_max_pwm", XC430_PWM_LIMIT, 1.0,
     "防甩刹车单独的PWM上限，跟max_feedback_pwm是分开算的"),
)


class RawSubscriber:
    """Keeps the newest /raw payload plus a rolling history of the force channel means.

    Purely for display - the actual haptics run through their own GripperForceFeedback
    subscription to the same topic (ROS allows any number of subscribers on one topic).
    """

    def __init__(self, topic, history):
        self._lock = threading.Lock()
        self._payload = None
        self._seq = 0
        self._stamps = []
        self._errors = 0
        self._history = {field: deque(maxlen=history) for field in CURVE_FIELDS}
        self.topic = topic
        self._sub = rospy.Subscriber(
            topic, UInt8MultiArray, self._callback, queue_size=1, buff_size=2 ** 24, tcp_nodelay=True
        )

    def _callback(self, msg):
        now = time.time()
        try:
            payload = pickle.loads(bytes(msg.data))
        except Exception as exc:
            with self._lock:
                self._errors += 1
            rospy.logwarn_throttle(5.0, f"cannot unpickle {self.topic}: {exc}")
            return
        frames = payload.get("frames", {})
        means = {}
        for field in CURVE_FIELDS:
            value = frames.get(field)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float64)
            if array.ndim == 3 and array.shape[-1] == 3:
                means[field] = array.reshape(-1, 3).mean(axis=0)
        with self._lock:
            self._payload = payload
            self._seq += 1
            self._stamps.append(now)
            if len(self._stamps) > 60:
                del self._stamps[:-60]
            for field, value in means.items():
                self._history[field].append(value)

    def take(self):
        """Return (payload, seq, measured_hz, unpickle_errors)."""
        with self._lock:
            stamps = self._stamps
            hz = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
            return self._payload, self._seq, hz, self._errors

    def curves(self):
        with self._lock:
            return {field: np.array(samples, dtype=np.float64) if samples else np.empty((0, 3))
                    for field, samples in self._history.items()}


def field_magnitude(array):
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] in (2, 3):
        return np.linalg.norm(array, axis=-1)
    return np.abs(array.reshape(array.shape[0], -1)) if array.ndim > 2 else np.abs(array)


def heatmap(array, title, scale, height=PANEL_HEIGHT, draw_arrows=False):
    magnitude = field_magnitude(np.asarray(array))
    ceiling = scale if scale is not None else max(float(magnitude.max()), 1e-6)
    normalized = np.clip(magnitude / ceiling, 0.0, 1.0)
    image = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    factor = max(1, int(round(height / image.shape[0])))
    image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)

    if draw_arrows and np.ndim(array) == 3 and np.shape(array)[-1] == 3:
        shear = np.asarray(array, dtype=np.float64)[..., :2]
        arrow_scale = 0.45 * factor / ceiling
        for row in range(shear.shape[0]):
            for col in range(shear.shape[1]):
                cx, cy = int((col + 0.5) * factor), int((row + 0.5) * factor)
                dx, dy = shear[row, col] * arrow_scale
                if abs(dx) + abs(dy) > 1.0:
                    cv2.arrowedLine(image, (cx, cy), (int(cx + dx), int(cy + dy)),
                                    (255, 255, 255), 1, tipLength=0.35)

    cv2.putText(image, f"{title}  max={magnitude.max():.3f}  scale={ceiling:.3f}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if image.shape[0] != height:
        width = max(1, int(round(image.shape[1] * height / image.shape[0])))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    return image


def strip_charts(history, width=560, chart_height=160, y_scale=None):
    charts = []
    for channel in range(3):
        chart = np.full((chart_height, width, 3), 24, dtype=np.uint8)
        series = {field: samples[:, channel] for field, samples in history.items() if len(samples)}

        span = max((float(np.abs(values).max()) for values in series.values()), default=0.0)
        ceiling = y_scale if y_scale is not None else max(span * 1.25, 1e-6)
        middle = chart_height // 2

        def to_y(value):
            return int(np.clip(middle - value / ceiling * (chart_height / 2 - 12), 1, chart_height - 2))

        cv2.line(chart, (0, middle), (width, middle), (70, 70, 70), 1)
        cv2.line(chart, (0, to_y(ceiling)), (width, to_y(ceiling)), (45, 45, 45), 1)
        cv2.line(chart, (0, to_y(-ceiling)), (width, to_y(-ceiling)), (45, 45, 45), 1)

        for field, values in series.items():
            if len(values) < 2:
                continue
            xs = np.linspace(0, width - 1, len(values))
            points = np.stack([xs, [to_y(v) for v in values]], axis=1).astype(np.int32)
            cv2.polylines(chart, [points], False, CURVE_COLORS[field], 1, cv2.LINE_AA)

        label = f"ch{channel} {'(normal)' if channel == 2 else '(shear)'}  +-{ceiling:.2e}"
        cv2.putText(chart, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        offset = 0
        for field, values in series.items():
            text = f"{field}={values[-1]:+.2e}"
            cv2.putText(chart, text, (width - 170 + offset, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        CURVE_COLORS[field], 1, cv2.LINE_AA)
            offset += 170
        charts.append(chart)
        charts.append(np.full((2, width, 3), 60, dtype=np.uint8))
    return np.vstack(charts[:-1])


def to_panel(image, height=PANEL_HEIGHT):
    image = np.asarray(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def slider_initial_values(args):
    return {
        ("feedback", "deadband"): args.force_deadband,
        ("feedback", "full_scale"): args.force_full_scale,
        ("feedback", "curve"): args.force_curve,
        ("feedback", "stiffness_min"): args.wall_stiffness_min,
        ("feedback", "stiffness_max"): args.wall_stiffness_max,
        ("feedback", "max_pwm"): args.max_feedback_pwm,
        ("feedback", "ramp_s"): args.force_ramp_s,
        ("brake", "speed_limit"): args.close_speed_limit,
        ("brake", "gain"): args.close_brake_gain,
        ("brake", "max_pwm"): args.close_brake_max_pwm,
    }


PANEL_LINE_H = 20
PANEL_ROWS_PER_SLIDER = 2  # value line + Chinese hint line


def control_panel_height():
    return 110 + PANEL_LINE_H * PANEL_ROWS_PER_SLIDER * len(SLIDER_SPECS)


def build_control_window(args):
    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW, 720, control_panel_height())
    initial = slider_initial_values(args)
    for target, attr, label, hi, scale, _hint in SLIDER_SPECS:
        pos = int(round(initial[(target, attr)] * scale))
        pos = min(max(pos, 0), hi)
        cv2.createTrackbar(label, CONTROL_WINDOW, pos, hi, lambda _pos: None)


def read_sliders(feedback, brake):
    for target, attr, label, _hi, scale, _hint in SLIDER_SPECS:
        pos = cv2.getTrackbarPos(label, CONTROL_WINDOW)
        value = pos / scale
        obj = feedback if target == "feedback" else brake
        if attr == "max_pwm":
            value = int(value)
        setattr(obj, attr, value)


def commit_typed_value(index, text):
    """Apply a keyboard-typed value to slider `index` by moving its trackbar; read_sliders()
    picks it up on the very next tick, so this never has to touch feedback/brake directly."""
    target, attr, label, hi, scale, _hint = SLIDER_SPECS[index]
    try:
        value = float(text)
    except ValueError:
        print(f"Invalid number {text!r} for {target}.{attr}; input discarded.")
        return
    pos = int(round(value * scale))
    clamped = min(max(pos, 0), hi)
    if clamped != pos:
        print(f"{target}.{attr}: {value:g} is outside the slider range, "
              f"clamped to {clamped / scale:.4f}.")
    cv2.setTrackbarPos(label, CONTROL_WINDOW, clamped)


def render_control_panel(feedback, brake, status_lines, edit_index, edit_buffer):
    height = control_panel_height()
    panel = np.full((height, 720, 3), 24, dtype=np.uint8)
    cv2.putText(panel, "Drag a slider, or press its number (1-9,0) then type a value + Enter.",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, "Esc cancels typing (or quits when not typing). b = re-zero. p = print flags.",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    y = 64
    for i, (target, attr, label, _hi, scale, hint) in enumerate(SLIDER_SPECS):
        obj = feedback if target == "feedback" else brake
        value = getattr(obj, attr)
        key_label = str((i + 1) % 10)  # 1..9 then 0 for the 10th
        editing = i == edit_index
        color = (120, 220, 255) if editing else (120, 255, 120)
        value_text = f"[{key_label}] {target}.{attr:<13s} = {value:10.4f}   ({label})"
        if editing:
            value_text += f"   -> {edit_buffer}_"
        cv2.putText(panel, value_text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y += PANEL_LINE_H
        cv2.putText(panel, "      " + hint, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
        y += PANEL_LINE_H
    y += 8
    for line in status_lines:
        cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 120), 1, cv2.LINE_AA)
        y += PANEL_LINE_H
    return panel


def print_current_settings(feedback, brake):
    print("\nCurrent haptics feel - paste into record_episodes_aloha2.py / "
          "record_episodes_aloha2_vtla.py / one_side_teleop.py (same flag names there):")
    print(f"  --force_deadband {feedback.deadband:.4f} --force_full_scale {feedback.full_scale:.4f} "
          f"--force_curve {feedback.curve:.3f} --wall_stiffness_min {feedback.stiffness_min:.1f} "
          f"--wall_stiffness_max {feedback.stiffness_max:.1f} --max_feedback_pwm {feedback.max_pwm} "
          f"--force_ramp_s {feedback.ramp_s:.3f} --close_speed_limit {brake.speed_limit:.3f} "
          f"--close_brake_gain {brake.gain:.2f} --close_brake_max_pwm {brake.max_pwm}\n")


def setup_hardware(side, args):
    """Bring up master_<side>/puppet_<side>'s GRIPPERS only. The arms are never touched."""
    master_bot = make_bot("wx250s", f"master_{side}", init_node=True)
    puppet_bot = make_bot("vx300s", f"puppet_{side}", init_node=False)

    puppet_bot.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    set_puppet_gripper_current_limit(puppet_bot, args.puppet_gripper_current_limit)
    puppet_bot.dxl.robot_torque_enable("single", "gripper", True)
    master_bot.dxl.robot_torque_enable("single", "gripper", True)

    deadline = time.time() + 3.0
    while len(master_bot.dxl.joint_states.position) < 7 and time.time() < deadline:
        time.sleep(0.02)
    if len(master_bot.dxl.joint_states.position) < 7:
        raise RuntimeError(f"No joint_states from master_{side} - is it powered and connected?")

    master_start = master_bot.dxl.joint_states.position[6]
    puppet_start = master_to_puppet_gripper(side, master_start)
    print(f"Syncing puppet_{side} gripper to the leader's current position ({puppet_start:.4f})...")
    move_grippers([puppet_bot], [puppet_start], move_time=1.0)

    return master_bot, puppet_bot


def gripper_tick(side, master_bot, puppet_bot, gripper_command, guard, feedback, brake, master_effort,
                 open_limit, close_limit, disable_guard):
    """One control-cycle tick: mirror the gripper, then run the haptics on the leader."""
    master_gripper_joint = master_bot.dxl.joint_states.position[6]
    puppet_target = master_to_puppet_gripper(side, master_gripper_joint)
    if not disable_guard:
        puppet_target = guard.filter_command(
            puppet_target, get_arm_gripper_positions(puppet_bot),
            effort=get_arm_gripper_effort(puppet_bot),
        )
    gripper_command.cmd = puppet_target
    puppet_bot.gripper.core.pub_single.publish(gripper_command)

    pwm, force, age, depth = feedback.command(master_gripper_joint)
    velocity = get_arm_gripper_velocity(master_bot)
    brake_pwm = brake.command(velocity)
    pwm += feedback.sign * brake_pwm
    moving_open = velocity is not None and velocity > 0
    if pwm > 0 and master_gripper_joint >= open_limit and not moving_open:
        pwm = 0
    elif pwm < 0 and master_gripper_joint <= close_limit:
        pwm = 0
    pwm = int(max(-XC430_PWM_LIMIT, min(XC430_PWM_LIMIT, pwm)))
    master_effort.send(pwm)
    return pwm, force, age, depth, brake_pwm, velocity


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("side", choices=("left", "right"), help="Which gripper to tune.")
    parser.add_argument("--puppet_gripper_current_limit", type=int, default=DEFAULT_PUPPET_GRIPPER_CURRENT_LIMIT)
    parser.add_argument("--no_torque_gate", action="store_true",
                        help="Keep the leader gripper torqued even at 0 PWM (debugging only).")
    parser.add_argument("--disable_gripper_contact_guard", action="store_true",
                        help="Allow the puppet gripper to keep closing even after stall/contact detection.")
    parser.add_argument("--gripper_stall_error", type=float, default=DEFAULT_GRIPPER_STALL_ERROR)
    parser.add_argument("--gripper_stall_motion", type=float, default=DEFAULT_GRIPPER_STALL_MOTION)
    parser.add_argument("--gripper_stall_cycles", type=int, default=DEFAULT_GRIPPER_STALL_CYCLES)
    parser.add_argument("--gripper_release_margin", type=float, default=DEFAULT_GRIPPER_RELEASE_MARGIN)
    parser.add_argument("--gripper_effort_threshold", type=float, default=None)
    # Initial slider positions - the sliders then let you move away from these live.
    parser.add_argument("--force_deadband", type=float, default=DEFAULT_FORCE_DEADBAND)
    parser.add_argument("--force_full_scale", type=float, default=DEFAULT_FORCE_FULL_SCALE)
    parser.add_argument("--force_curve", type=float, default=DEFAULT_FORCE_CURVE)
    parser.add_argument("--wall_stiffness_min", type=float, default=DEFAULT_WALL_STIFFNESS_MIN)
    parser.add_argument("--wall_stiffness_max", type=float, default=DEFAULT_WALL_STIFFNESS_MAX)
    parser.add_argument("--max_feedback_pwm", type=int, default=DEFAULT_MAX_FEEDBACK_PWM)
    parser.add_argument("--force_ramp_s", type=float, default=DEFAULT_FORCE_RAMP_S)
    parser.add_argument("--close_speed_limit", type=float, default=DEFAULT_CLOSE_SPEED_LIMIT)
    parser.add_argument("--close_brake_gain", type=float, default=DEFAULT_CLOSE_BRAKE_GAIN)
    parser.add_argument("--close_brake_max_pwm", type=int, default=DEFAULT_CLOSE_BRAKE_MAX_PWM)
    parser.add_argument("--force_baseline_samples", type=int, default=DEFAULT_FORCE_BASELINE_SAMPLES)
    # Display.
    parser.add_argument("--force_scale", type=float, default=None,
                        help="Fixed colour-scale ceiling for the combined force heat map.")
    parser.add_argument("--normal_scale", type=float, default=None,
                        help="Fixed colour-scale ceiling for the normal-only (force[...,2]) heat map.")
    parser.add_argument("--history", type=int, default=300)
    parser.add_argument("--curve_scale", type=float, default=None)
    parser.add_argument("--print_hz", type=float, default=1.0)
    parser.add_argument("--no_curves", action="store_true")
    parser.add_argument("--no_images", action="store_true")
    parser.add_argument("--no_display", action="store_true",
                        help="No windows or sliders - just runs the haptics with the CLI values above.")
    parser.add_argument("--no_arrows", action="store_true")
    args = parser.parse_args()

    side = args.side
    topic = DEFAULT_TACTILE_TOPIC_TEMPLATE.format(side=side)
    rospy.init_node("check_tactile", anonymous=True, disable_signals=True)

    master_bot, puppet_bot = setup_hardware(side, args)
    gripper_command = JointSingleCommand(name="gripper")
    guard = GripperContactGuard(
        side.capitalize(), args.gripper_stall_error, args.gripper_stall_motion,
        args.gripper_stall_cycles, args.gripper_release_margin,
        effort_threshold=args.gripper_effort_threshold,
    )
    open_limit = master_gripper_pose(side, 1.0) - FEEDBACK_LIMIT_MARGIN
    close_limit = master_gripper_pose(side, 0.0) + FEEDBACK_LIMIT_MARGIN

    feedback = GripperForceFeedback(
        topic=topic, field=DEFAULT_FORCE_FIELD, deadband=args.force_deadband,
        full_scale=args.force_full_scale, curve=args.force_curve, max_pwm=args.max_feedback_pwm,
        timeout_s=DEFAULT_FORCE_TIMEOUT_S, filter_alpha=DEFAULT_FORCE_FILTER, metric="abs_sum",
        sign=1, ramp_s=args.force_ramp_s, mode="wall", stiffness_min=args.wall_stiffness_min,
        stiffness_max=args.wall_stiffness_max,
    )
    brake = ClosingSpeedBrake(
        speed_limit=args.close_speed_limit, gain=args.close_brake_gain,
        max_pwm=args.close_brake_max_pwm, filter_alpha=DEFAULT_VELOCITY_FILTER,
    )

    print(f"Waiting for tactile data on {topic} ...")
    deadline = time.time() + 3.0
    while feedback.frames == 0 and time.time() < deadline:
        time.sleep(0.05)
    if feedback.frames == 0:
        print(f"Warning: no tactile data on {topic} yet - relaunch with extra_outputs:=\"force\".")
    print("Capturing resting baseline - make sure nothing is touching the sensor...")
    feedback.capture_baseline(args.force_baseline_samples)

    master_effort = MasterGripperEffort(master_bot, torque_gate=not args.no_torque_gate)
    feedback.activate()
    print(f"Live. {side} leader gripper is torque-free with force feedback active.")
    print("Keys (in the controls window): q/ESC = quit, b = re-zero baseline, p = print settings, "
          "1-9/0 = select a slider then type a number + Enter to set it exactly (Esc cancels typing).")

    show = not args.no_display
    if show:
        build_control_window(args)

    source = RawSubscriber(topic, history=max(args.history, 2)) if show and not args.no_images else None
    announced = False
    last_display = 0.0
    last_print = 0.0
    print_period = 1.0 / max(args.print_hz, 0.01)
    display_period = 1.0 / DISPLAY_HZ
    # Typed-value entry: press a digit to pick a slider (edit_index), then type into edit_buffer.
    edit_index = None
    edit_buffer = ""
    digit_to_index = {ord(str((i + 1) % 10)): i for i in range(len(SLIDER_SPECS))}

    try:
        while not rospy.is_shutdown():
            now = time.time()
            if show:
                read_sliders(feedback, brake)

            pwm, force, age, depth, brake_pwm, velocity = gripper_tick(
                side, master_bot, puppet_bot, gripper_command, guard, feedback, brake, master_effort,
                open_limit, close_limit, args.disable_gripper_contact_guard,
            )

            if now - last_print >= print_period:
                last_print = now
                state = "contact" if feedback.contact else "free   "
                print(f"{side:5s} {state}  force={force:8.4f}  net={force - feedback.baseline:+8.4f}  "
                      f"depth={depth:6.4f}rad  vclose={velocity if velocity is not None else 0.0:+6.3f}rad/s  "
                      f"brake={brake_pwm:+4d}  pwm={pwm:+5d}  age={age * 1000:5.1f}ms")

            if show and now - last_display >= display_period:
                last_display = now
                status_lines = [
                    f"force={force:8.4f}  net={force - feedback.baseline:+8.4f}  depth={depth:6.4f}rad",
                    f"pwm={pwm:+5d}  brake={brake_pwm:+4d}  vclose={velocity if velocity is not None else 0.0:+6.3f}rad/s",
                    f"contact={'yes' if feedback.contact else 'no'}  baseline={feedback.baseline:+8.4f}",
                ]
                cv2.imshow(CONTROL_WINDOW, render_control_panel(feedback, brake, status_lines,
                                                                 edit_index, edit_buffer))

                if source is not None:
                    payload, seq, hz, errors = source.take()
                    if payload is not None:
                        frames = payload.get("frames", {})
                        if not announced:
                            announced = True
                            if "force" not in frames:
                                print(f'!! "force" is NOT in the payload - relaunch with '
                                      f'extra_outputs:="force"')
                        if not args.no_curves:
                            history = source.curves()
                            if any(len(samples) for samples in history.values()):
                                cv2.imshow(CURVE_WINDOW, strip_charts(history, y_scale=args.curve_scale))
                        panels = []
                        if frames.get("rectify") is not None:
                            # XENSE's own type stub claims BGR; the actual bytes are RGB.
                            panels.append(to_panel(cv2.cvtColor(frames["rectify"], cv2.COLOR_RGB2BGR)))
                        if frames.get("force") is not None:
                            force_field = np.asarray(frames["force"])
                            panels.append(heatmap(force_field, "force", args.force_scale,
                                                  draw_arrows=not args.no_arrows))
                            panels.append(heatmap(force_field[..., 2], "force[...,2] normal", args.normal_scale))
                        if panels:
                            cv2.imshow(IMAGE_WINDOW, np.hstack(panels))

                key = cv2.waitKey(1) & 0xFF
                if edit_index is not None:
                    # Typing a value for slider `edit_index`; everything else is suspended.
                    if key == 27:  # Esc cancels typing only, does not quit
                        edit_index, edit_buffer = None, ""
                    elif key in (13, 10):  # Enter commits
                        commit_typed_value(edit_index, edit_buffer)
                        edit_index, edit_buffer = None, ""
                    elif key in (8, 127):  # Backspace
                        edit_buffer = edit_buffer[:-1]
                    elif key in (ord("."), ord("-")) or ord("0") <= key <= ord("9"):
                        edit_buffer += chr(key)
                elif key in (ord("q"), 27):
                    break
                elif key == ord("b"):
                    feedback.capture_baseline(args.force_baseline_samples)
                elif key == ord("p"):
                    print_current_settings(feedback, brake)
                elif key in digit_to_index:
                    edit_index, edit_buffer = digit_to_index[key], ""
            elif not show:
                time.sleep(0.001)

            elapsed = time.time() - now
            if elapsed < DT:
                time.sleep(DT - elapsed)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        feedback.deactivate()
        master_effort.send(0)
        master_effort.restore()
        if show:
            cv2.destroyAllWindows()
        print_current_settings(feedback, brake)


if __name__ == "__main__":
    main()
