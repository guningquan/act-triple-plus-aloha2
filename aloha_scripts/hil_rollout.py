#!/usr/bin/env python3
"""Human-in-the-loop rollout of a remote policy on ALOHA, recording in the raw training formats.

The policy runs on the GPU server inside its own repo (openpi or N0-VTLA scripts/serve_policy.py, or
lerobot's examples/aloha_ros/serve_policy_websocket.py for a LeRobot checkpoint, --policy lerobot);
this script is the robot side: it captures observations, executes the returned action chunks,
lets the operator take over with the leader arms (clutch-style, see hil_devices.ClutchLeader) and
records every tick with phase/intervention labels (see hil_recorder.py).

    Server (in the openpi repo on the GPU box):
        uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
            --policy.config=pi05_packaging_simple_2_full \
            --policy.dir=<openpi_checkpoints>/pi05_packaging_simple_2_full/packaging_simple_2_pt/20000
    Robot (this repo's ROS env, after `roslaunch aloha 4arms_teleop.launch` or the tactile launch):
        python3 aloha_scripts/hil_rollout.py --task_name packaging_simple_2 --policy openpi \
            --host 10.240.77.20 --port 8000 --prompt "packaging the items" --num_episodes 20
    LeRobot checkpoint (pi0.5, MolmoAct2, ACT, GR00T, ... trained on these hdf5 episodes):
        python examples/aloha_ros/serve_policy_websocket.py --policy_path <.../pretrained_model> \
            --task "packaging the items" --port 8000                      (server, lerobot repo)
        python3 aloha_scripts/hil_rollout.py --task_name packaging_simple_2 --policy lerobot \
            --host 10.240.77.20 --port 8000 --num_episodes 20              (robot)
    Evaluation without takeover or recording: run_pi_policy.py (openpi, lerobot) / run_vtla_policy.py (N0).
    Before the first episode one warm-up inference is sent (openpi compiles the model on its first call; --warmup_timeout).
    Step-by-step guide: README_run_robot.md.

Keys (terminal; a foot pedal that types arrow keys works the same, or use --pedal_device):
    RIGHT   start the episode; then pause / resume the policy
    LEFT    while paused: take over (leaders soften, settle, clutch engages); again: hand back (pause)
    e/Enter end the episode -> RIGHT save, LEFT discard, ESC discard and quit
    ESC/q   end the episode and quit after the save/discard question
    b       re-zero the leader-gripper force feedback baseline

Phases: autonomous (policy drives the followers, leaders locked) -RIGHT-> paused (followers hold,
leaders driven onto the followers' pose over --align_s) -LEFT-> leaders released (P-gain ramp over
--release_ramp_s, settle, offset = leader - follower) -> correcting (followers follow leader -
offset, eased in over --correction_blend_s; the first --correction_skip_s of each correction is not
labelled as intervention) -LEFT-> paused (leaders locked, offset cleared) -RIGHT-> autonomous.

Output: --format hdf5 (openpi / ACT / lerobot converters; default for --policy openpi / lerobot) writes
<output_dir>/episode_N.hdf5, --format vtla (N0-VTLA's convert_canonical_data.py; required for N0)
writes <output_dir>/episode_N/. Default output_dir: the task's dataset_dir + "_hil".
hil_split_episodes.py turns them into corrections-only (or autonomous-only) episodes.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from chunk_executor import ChunkExecutor, PolicyError
from hil_recorder import PHASE_AUTONOMOUS, PHASE_CORRECTING, PHASE_PAUSED
from tactile_stream import (DEFAULT_MAX_TACTILE_AGE_S, TactileStale, calibrate_sensors, current_runtime_config,
                            topic_for)

PHASE_NAMES = {PHASE_AUTONOMOUS: "autonomous", PHASE_PAUSED: "paused", PHASE_CORRECTING: "correcting"}
# Progress-line colour by phase: green while the policy drives, yellow while paused/waiting for a
# decision, red once a human has taken over (PHASE_CORRECTING - the followers are tracking the
# leaders). ANSI_PHASE colours the phase word in the postfix text (bold + bright SGR codes, not the
# dim/dull standard 30-37 ANSI hues most terminal themes render for plain "green"/"yellow"); tqdm's
# own `colour=` bar-fill knob is given the same hues as saturated truecolor hex instead of the named
# "green"/"yellow" (which map to those same dull standard codes), so the bar and the text match and
# both actually look green/yellow/red instead of olive/muddy.
ANSI_RESET = "\033[0m"
ANSI_PHASE = {PHASE_AUTONOMOUS: "\033[1;92m", PHASE_PAUSED: "\033[1;93m", PHASE_CORRECTING: "\033[1;91m"}
TQDM_PHASE_COLOUR = {PHASE_AUTONOMOUS: "#00FF00", PHASE_PAUSED: "#FFFF00", PHASE_CORRECTING: "#FF0000"}

# Take-over preparation (leaders releasing / settling) is still PHASE_PAUSED, so it stays in the yellow family:
# bright-yellow bar fill, dim empty part. Green and red are left out on purpose - they already mean "the policy
# drives" and "you are live" - the line turns red only when the correction really starts.
ANSI_PREPARE = "\033[93m"
ANSI_DIM = "\033[2m"
ANSI_BOLD = "\033[1m"
PREPARE_BAR_WIDTH = 10
PREPARE_REFRESH_S = 0.05  # redraw period of the preparation line (a paused tqdm bar is otherwise redrawn on events only)


def _bar_chars():
    """Block characters when stdout can print them (tqdm falls back the same way), else ASCII."""
    try:
        "█░".encode(sys.stdout.encoding or "ascii")
        return "█", "░"
    except (UnicodeEncodeError, LookupError):
        return "#", "-"


def format_takeover_progress(shot, chars=None):
    """One preparation snapshot (ClutchLeaderPair.takeover_progress) as the text for the progress line:
        releasing [██████░░░░] 7/12
        settling [████░░░░░░] steady 0.08/0.20s | move 0.014/0.010 rad (R elbow) | wait 0.4/1.0s
    The bar fills while the leader stays within tolerance and empties as soon as a joint moves; the
    moving joint is named only while it is over the tolerance."""
    fill, empty = chars or _bar_chars()
    filled = int(round(max(0.0, min(1.0, shot["share"])) * PREPARE_BAR_WIDTH))
    bar = "[{}{}{}{}{}{}]".format(ANSI_PREPARE, fill * filled, ANSI_RESET, ANSI_DIM,
                                  empty * (PREPARE_BAR_WIDTH - filled), ANSI_RESET)
    if shot["stage"] == "releasing":
        text = "releasing " + bar
        return text + " {}/{}".format(shot["done"], shot["steps"]) if shot["steps"] else text
    move = "{:.3f}/{:.3f} rad".format(shot["span"], shot["tolerance"])
    if shot["span"] > shot["tolerance"]:
        move = ANSI_BOLD + move + ANSI_RESET + " ({} {})".format(shot["side"][0].upper(), shot["joint_name"])
    return "settling {} steady {:.2f}/{:.2f}s | move {} | wait {:.1f}/{:.1f}s".format(
        bar, shot["steady_s"], shot["need_s"], move, shot["elapsed_s"], shot["timeout_s"])


class StaleStream(RuntimeError):
    """A camera/tactile/joint stream is missing or older than --max_stream_age."""


class HilSession:
    """One episode's control loop and state machine. Hardware comes in as objects so the logic runs
    without ROS (see tests/test_hil_session.py):
        follower  hil_devices.FollowerCommander     leaders  hil_devices.ClutchLeaderPair
        executor  chunk_executor.ChunkExecutor      capture  callable(require_fresh) -> Frame
        recorder  hil_recorder.*Episode             inputs   hil_devices.HilInput
    """

    def __init__(self, follower, leaders, executor, capture, recorder, inputs, control_dt, max_steps,
                 align_s=2.0, correction_blend_s=0.0, correction_skip_s=0.2, resume_blend_s=0.0,
                 on_rezero=None, clock=time.monotonic, sleep=time.sleep, show_progress=True):
        self.follower = follower
        self.leaders = leaders
        self.executor = executor
        self.capture = capture
        self.recorder = recorder
        self.inputs = inputs
        self.control_dt = control_dt
        self.max_steps = max_steps
        self.align_s = align_s
        self.correction_blend_s = correction_blend_s
        self.skip_ticks = int(round(correction_skip_s / control_dt))
        self.resume_blend_s = resume_blend_s
        self.on_rezero = on_rezero
        self.clock = clock
        self.sleep = sleep
        self.show_progress = show_progress
        # The episode ends after max_steps ticks of policy driving + human correcting; paused time
        # (waiting, aligning, releasing) does not count. The total, pauses included, is capped at 4x.
        self.max_total_steps = 4 * max_steps
        self.phase = PHASE_AUTONOMOUS
        self.releasing = False  # leaders softening/settling, CORRECTING starts when they are free
        self.release_pending = False  # take-over asked while the leaders were still aligning
        self.correction_ticks = 0
        self.corrections = 0
        self.steps = 0  # every tick
        self.active_steps = 0  # autonomous + correcting ticks, counted against max_steps
        self.skipped_frames = 0
        self.end_reason = None
        self._limit_warned = False
        self._bar = None
        self._last_stale_warning = 0.0
        self._last_prepare_refresh = float("-inf")

    # -- transitions -----------------------------------------------------------------------------

    def _pause(self, now, reason):
        self.executor.reset()
        self.phase = PHASE_PAUSED
        self._update_progress()
        hold = self.follower.current()
        self.leaders.begin_move(hold, self.align_s, now)
        self._say("Paused ({}): followers hold, leaders moving onto them. LEFT = take over, RIGHT = resume."
              .format(reason))

    def _resume(self):
        self.leaders.stop_move()
        self.executor.reset()
        self.follower.begin_blend(self.resume_blend_s)
        self.phase = PHASE_AUTONOMOUS
        self._update_progress()
        self._say("Resumed: policy driving.")

    def _start_correction(self):
        self.releasing = False
        self.phase = PHASE_CORRECTING
        self.correction_ticks = 0
        self.corrections += 1
        self._update_progress()
        self.follower.begin_blend(self.correction_blend_s)
        self._say("Correcting (#{}): followers follow the leaders. LEFT = hand back.".format(self.corrections))

    def _end_correction(self):
        self.leaders.lock()
        self.phase = PHASE_PAUSED
        self._update_progress()
        if self.active_steps >= self.max_steps:
            self.end_reason = "max_steps"
            return
        self._say("Correction #{} done; leaders locked. LEFT = take over again, RIGHT = resume the policy."
              .format(self.corrections))

    def handle(self, event, now):
        if event in ("end", "esc"):
            self.end_reason = event
        elif event == "rezero":
            if self.phase == PHASE_CORRECTING or self.releasing or self.release_pending:
                self._say("Re-zero the force feedback while not taking over (it pauses the control loop).")
            elif self.on_rezero is not None:
                self.on_rezero()
        elif event == "pause":
            if self.phase == PHASE_AUTONOMOUS:
                self._pause(now, "operator")
            elif self.phase == PHASE_PAUSED:
                if self.releasing or self.release_pending:
                    self._say("Leaders are being released; wait for the correction to start.")
                else:
                    self._resume()
            else:
                self._say("Hand back first (LEFT), then RIGHT resumes the policy.")
        elif event == "correct":
            if self.phase == PHASE_PAUSED:
                if self.releasing or self.release_pending:
                    return
                if self.leaders.busy:
                    self.release_pending = True  # released once they reach the followers
                else:
                    self.leaders.begin_release(now)
                    self.releasing = True
            elif self.phase == PHASE_CORRECTING:
                self._end_correction()
            else:
                self._say("Pause first (RIGHT), then LEFT takes over.")

    # -- loop --------------------------------------------------------------------------------------

    def _capture(self, now):
        try:
            return self.capture(self.phase == PHASE_AUTONOMOUS)
        except StaleStream as exc:
            if self.phase == PHASE_AUTONOMOUS:
                self._pause(now, "stale stream: {}".format(exc))
            elif now - self._last_stale_warning > 1.0:
                self._last_stale_warning = now
                self._say("Warning: {}; not recording this tick.".format(exc))
            self.skipped_frames += 1
            return None

    def tick(self):
        now = self.clock()
        for event in self.inputs.poll():
            self.handle(event, now)
        if self.end_reason is not None:
            return False

        frame = self._capture(now)
        intervention = False
        if self.phase == PHASE_PAUSED:
            self.leaders.tick(now)
            if self.release_pending and not self.leaders.busy:
                self.release_pending = False
                self.leaders.begin_release(now)
                self.releasing = True
            if self.releasing and self.leaders.released:
                self._start_correction()
            elif self.releasing:
                self._refresh_takeover_progress(now)

        if self.phase == PHASE_AUTONOMOUS:
            try:
                action = self.executor.step(lambda: frame)
            except PolicyError as exc:
                # self.clock(), not `now`: a sync inference may have blocked for seconds, and a stale
                # start time would make the leader alignment think it is already finished
                self._pause(self.clock(), str(exc))
                action = None
            sent = self.follower.send(action) if action is not None else self.follower.current()
        elif self.phase == PHASE_CORRECTING:
            self.leaders.tick(now)
            sent = self.follower.send(self.leaders.action())
            intervention = self.correction_ticks >= self.skip_ticks
            self.correction_ticks += 1
        else:
            sent = self.follower.current()

        if frame is not None:
            self.recorder.add(frame, sent, self.phase, intervention, self.leaders.raw_action())
        self.steps += 1
        if self.phase != PHASE_PAUSED:
            self.active_steps += 1
            if self._bar is not None:
                self._bar.update(1)
        if self.steps % 10 == 0:
            self._update_progress()
        if self.active_steps >= self.max_steps:
            if self.phase != PHASE_CORRECTING:
                self.end_reason = "max_steps"
            elif not self._limit_warned:
                self._limit_warned = True
                self._say("Time limit reached: hand back (LEFT) to end the episode.")
        if self.steps >= self.max_total_steps:
            self.end_reason = "max_steps"
        return self.end_reason is None

    def _say(self, message):
        """Event messages go above the progress bar instead of breaking it."""
        if self._bar is not None:
            self._bar.write(message)
        else:
            print(message)

    def _refresh_takeover_progress(self, now):
        """Redraw the preparation line now (the paused phase never calls bar.update), at most every
        PREPARE_REFRESH_S."""
        if self._bar is None or now - self._last_prepare_refresh < PREPARE_REFRESH_S - 1e-9:
            return
        self._last_prepare_refresh = now
        self._update_progress(refresh=True)

    def _update_progress(self, refresh=False):
        if self._bar is None:
            return
        phase_word = ANSI_PHASE[self.phase] + PHASE_NAMES[self.phase] + ANSI_RESET
        shot = self.leaders.takeover_progress() if self.releasing else None
        parts = [phase_word]
        if shot is not None:
            parts.append(format_takeover_progress(shot))
        parts += ["rec {}".format(len(self.recorder)), "corr {}".format(self.corrections)]
        latencies = getattr(self.executor, "latencies", None)
        if latencies and shot is None:
            parts.append("infer {:.0f} ms".format(1000 * latencies[-1]))
        if self.phase == PHASE_CORRECTING:
            parts.append("offset {:.3f}".format(float(np.abs(self.leaders.offsets()).max())))
        self._bar.colour = TQDM_PHASE_COLOUR[self.phase]
        self._bar.set_postfix_str(" | ".join(parts), refresh=refresh)

    def run(self):
        """Run until e/ESC or the time limit; returns "end", "esc" or "max_steps". The leaders are
        locked and the policy stopped on the way out. Progress: one tqdm line (policy + correction
        time against max_steps, phase, recorded frames, corrections, last inference time)."""
        if self.show_progress:
            from tqdm import tqdm

            self._bar = tqdm(total=self.max_steps, unit="tick", dynamic_ncols=True, leave=True,
                             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} active ticks [{elapsed}{postfix}]")
            self._update_progress()
        next_tick = self.clock()
        try:
            while True:
                if not self.tick():
                    break
                next_tick += self.control_dt
                delay = next_tick - self.clock()
                if delay > 0:
                    self.sleep(delay)
                elif delay < -self.control_dt:
                    next_tick = self.clock()  # overran (blocking inference): no catch-up burst
        finally:
            self.executor.reset()
            if not self.leaders.locked:
                self.leaders.lock()
            if self._bar is not None:
                self._update_progress()
                self._bar.close()
                self._bar = None
        return self.end_reason or "max_steps"


# ------------------------------------------------------------------------------------------------
# Hardware entry point
# ------------------------------------------------------------------------------------------------

HAPTICS_TUNING = ("start_open_ratio", "force_deadband", "force_full_scale", "wall_stiffness_min",
                  "wall_stiffness_max", "max_feedback_pwm", "close_speed_limit", "close_brake_gain")


# Servers whose policies take the hdf5 recorder's observation (RealEnv); recorded as hdf5 by default
HDF5_POLICIES = ("openpi", "lerobot")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task_name", required=True, help="Entry of constants.TASK_CONFIGS (cameras, length, dir)")
    parser.add_argument("--policy", choices=HDF5_POLICIES + ("n0_eef", "n0_joint"), required=True,
                        help="Server type: openpi ALOHA config, LeRobot checkpoint (serve_policy_websocket.py), "
                             "or N0-VTLA eef/joint checkpoint")
    parser.add_argument("--host", required=True, help="Policy server host, e.g. 10.240.77.20")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default=None,
                        help="Task prompt sent with every inference (openpi: omitted -> the config's "
                             "default prompt; lerobot: omitted -> the server's --task; "
                             "N0: default 'Perform the task.')")
    parser.add_argument("--format", choices=("hdf5", "vtla"), default=None,
                        help="Recording format; default hdf5 for openpi / lerobot, vtla for N0 (which needs it)")
    parser.add_argument("--output_dir", type=Path, default=None, help="Default: task dataset_dir + '_hil'")
    parser.add_argument("--num_episodes", type=int, default=1, help="Episodes to save before exiting")
    parser.add_argument("--max_timesteps", type=int, default=None,
                        help="Ticks of policy driving + correcting per episode, paused time not counted "
                             "(default: task episode_len); a correction running at the limit ends on hand-back")

    inference = parser.add_argument_group("inference")
    inference.add_argument("--action_horizon", type=int, default=50,
                           help="Chunk steps executed per inference (blocking, like openpi aloha_real)")
    inference.add_argument("--infer_timeout", type=float, default=10.0, help="Seconds before a reply counts as lost")
    inference.add_argument("--warmup_timeout", type=float, default=900.0,
                           help="Timeout of the one warm-up inference sent before the first episode (openpi "
                                "compiles the model on its first call, which can take minutes)")
    inference.add_argument("--policy_hz", type=float, default=None,
                           help="Rate the chunk steps are played at (default: openpi / lerobot 50, N0 30)")
    inference.add_argument("--policy_cameras", nargs="+", default=None,
                           help="Cameras sent to the server (openpi default: cam_high cam_left_wrist "
                                "cam_right_wrist; lerobot default: the checkpoint's cameras)")
    inference.add_argument("--resolution", type=int, default=224, help="openpi: client-side resize_with_pad size")
    inference.add_argument("--jpeg_quality", type=int, default=90,
                           help="lerobot: JPEG quality of the camera images sent (90 = the recorder's)")

    hil = parser.add_argument_group("takeover")
    hil.add_argument("--absolute_takeover", action="store_true",
                     help="Disable the clutch offset (follower jumps to the leader's absolute pose)")
    hil.add_argument("--align_s", type=float, default=2.0, help="Leader move onto the followers on pause")
    hil.add_argument("--release_ramp_s", type=float, default=0.5, help="Leader P-gain ramp to 0 before torque off")
    hil.add_argument("--settle_s", type=float, default=0.5,
                     help="Leader must stay within --settle_tolerance for this long before the offset is taken")
    hil.add_argument("--settle_tolerance", type=float, default=0.01, help="rad, per joint, within the settle window")
    hil.add_argument("--settle_timeout_s", type=float, default=2.0,
                     help="Longest wait for the leader to settle (must be >= --settle_s); then the offset is taken anyway")
    hil.add_argument("--max_offset_warn", type=float, default=0.25)
    hil.add_argument("--correction_blend_s", type=float, default=0.0,
                     help="Follower ease-in at correction start, from its last command (0: none - with the clutch "
                          "the first correction command already equals the held one)")
    hil.add_argument("--correction_skip_s", type=float, default=0.2,
                     help="Start of each correction not labelled as intervention")
    hil.add_argument("--resume_blend_s", type=float, default=0.0,
                     help="Follower ease-in from its last command when the policy resumes")

    safety = parser.add_argument_group("safety")
    safety.add_argument("--max_joint_step", type=float, default=0.1,
                        help="Per-tick follower arm joint change limit (rad, 50 Hz); 0 disables")
    safety.add_argument("--max_stream_age", type=float, default=0.5, help="Pause the policy on older camera data")
    safety.add_argument("--max_tactile_age", type=float, default=DEFAULT_MAX_TACTILE_AGE_S,
                        help="Pause the policy on tactile frames older than this (s), by the data's own age")
    safety.add_argument("--stream_timeout", type=float, default=5.0)
    safety.add_argument("--tactile_jpeg_quality", type=int, default=90,
                        help="hdf5 format: JPEG quality for tactile rectify/difference sent raw by the publisher")
    safety.add_argument("--no_tactile_calibration", action="store_true",
                        help="Do not re-zero the tactile sensors (/tactile/calibrate) before each episode")

    haptics = parser.add_argument_group("leader-gripper force feedback (record_episodes_aloha2.py defaults)")
    haptics.add_argument("--no_force_feedback", action="store_true")
    haptics.add_argument("--no_torque_gate", action="store_true")
    for name in HAPTICS_TUNING:
        haptics.add_argument("--" + name, type=float, default=argparse.SUPPRESS)
    haptics.add_argument("--force_baseline_samples", type=int, default=argparse.SUPPRESS)

    vtla = parser.add_argument_group("VTLA streams (N0 / --format vtla)")
    vtla.add_argument("--third_camera", choices=("cam_high", "cam_low"), default="cam_high")
    vtla.add_argument("--aloha_action_space", choices=("eef", "joint_position"), default=None,
                      help="VTLA export action space (default: eef for n0_eef, joint_position otherwise)")
    vtla.add_argument("--tactile_map", nargs=4, default=None, metavar="SENSOR",
                      help="Sensors for left_wrist_left, left_wrist_right, right_wrist_left, right_wrist_right")

    pedal = parser.add_argument_group("foot pedal (evdev)")
    pedal.add_argument("--pedal_device", default=None, help="e.g. /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd")
    pedal.add_argument("--pedal_pause", default="KEY_A")
    pedal.add_argument("--pedal_correct", default="KEY_B")
    pedal.add_argument("--pedal_end", default="KEY_C")
    return parser


def resolve_args(args, parser):
    if args.format is None:
        args.format = "hdf5" if args.policy in HDF5_POLICIES else "vtla"
    if args.policy not in HDF5_POLICIES and args.format != "vtla":
        parser.error("N0-VTLA needs --format vtla (its tactile streams come from the VTLA capture)")
    if args.policy_cameras is None and args.policy == "openpi":
        args.policy_cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    if args.aloha_action_space is None:
        args.aloha_action_space = "eef" if args.policy == "n0_eef" else "joint_position"
    if args.policy not in HDF5_POLICIES and args.prompt is None:
        args.prompt = "Perform the task."
    if args.max_joint_step is not None and args.max_joint_step <= 0:
        args.max_joint_step = None
    if args.num_episodes < 1:
        parser.error("--num_episodes must be >= 1")
    for name in ("align_s", "correction_blend_s", "correction_skip_s", "resume_blend_s", "release_ramp_s"):
        if getattr(args, name) < 0:
            parser.error("--{} must be >= 0".format(name))
    return args


def wait_for(inputs, choices, prompt, sleep=time.sleep):
    """Block until the keyboard or the pedal gives one of `choices` ({event: result})."""
    print(prompt)
    inputs.drain()
    while True:
        for event in inputs.poll():
            if event in choices:
                return choices[event]
        sleep(0.02)


def next_index(root):
    index = 0
    while (root / "episode_{}".format(index)).exists() or (root / "episode_{}.hdf5".format(index)).exists():
        index += 1
    return index


def main(argv=None):
    parser = build_parser()
    args = resolve_args(parser.parse_args(argv), parser)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Hardware imports here, so --help and the tests work without ROS.
    import rospy
    import record_episodes_aloha2 as legacy
    from constants import (DT, PUPPET_LEFT_GRIPPER_JOINT_CLOSE, PUPPET_LEFT_GRIPPER_JOINT_OPEN,
                           PUPPET_RIGHT_GRIPPER_JOINT_CLOSE, PUPPET_RIGHT_GRIPPER_JOINT_OPEN, TASK_CONFIGS)
    from hil_devices import ClutchLeader, ClutchLeaderPair, FollowerCommander, HilInput
    from hil_recorder import Hdf5HilEpisode, VtlaHilEpisode
    from policy_client import Frame, LerobotAlohaAdapter, N0VtlaAdapter, OpenpiAlohaAdapter, PolicyClient

    for name in HAPTICS_TUNING + ("force_baseline_samples",):
        if not hasattr(args, name):
            setattr(args, name, getattr(legacy, "DEFAULT_" + name.upper()))
    task = TASK_CONFIGS[args.task_name]
    camera_names = task["camera_names"]
    max_steps = args.max_timesteps or task["episode_len"]
    root = args.output_dir or Path(task["dataset_dir"] + "_hil")
    root.mkdir(parents=True, exist_ok=True)
    print("HIL output: {} ({})".format(root, args.format))

    client = PolicyClient(args.host, args.port, infer_timeout_s=args.infer_timeout)
    logging.info("Connected to %s. Server metadata: %s", client.uri, client.metadata)

    master_left = legacy.InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                                 robot_name="master_left", init_node=True)
    master_right = legacy.InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                                  robot_name="master_right", init_node=False)
    env = legacy.make_real_env(init_node=False, setup_robots=False)
    puppets = (env.puppet_bot_left, env.puppet_bot_right)

    streams = sources = None
    calibration = [{"closed": PUPPET_LEFT_GRIPPER_JOINT_CLOSE, "open": PUPPET_LEFT_GRIPPER_JOINT_OPEN},
                   {"closed": PUPPET_RIGHT_GRIPPER_JOINT_CLOSE, "open": PUPPET_RIGHT_GRIPPER_JOINT_OPEN}]
    if args.format == "vtla":
        import record_episodes_aloha2_vtla as vtla

        tactile_map = (dict(zip(vtla.DEFAULT_TACTILE_MAP, args.tactile_map)) if args.tactile_map
                       else dict(vtla.DEFAULT_TACTILE_MAP))
        sources = vtla.video_sources(args.third_camera, tactile_map)
        vtla.check_video_tools()
        arm_names = (env.puppet_bot_left.arm.group_info.joint_names, env.puppet_bot_right.arm.group_info.joint_names)
        streams = vtla.LiveStreams(sources, arm_names, calibration, save_force=False,
                                   max_tactile_age=args.max_tactile_age)
        streams.wait_ready(args.stream_timeout, args.max_stream_age)

    if args.policy == "openpi":
        adapter = OpenpiAlohaAdapter(args.prompt, cameras=args.policy_cameras, resolution=args.resolution,
                                     policy_hz=args.policy_hz or 50.0)
    elif args.policy == "lerobot":
        adapter = LerobotAlohaAdapter.from_metadata(client.metadata, args.prompt, cameras=args.policy_cameras,
                                                    policy_hz=args.policy_hz or 50.0, jpeg_quality=args.jpeg_quality)
    else:
        adapter = N0VtlaAdapter(args.prompt, args.policy[len("n0_"):], sources, puppets, policy_hz=args.policy_hz)

    def capture(require_fresh):
        vtla_snapshot = None
        stamp = time.monotonic()
        if streams is not None:
            try:
                stamp, vtla_snapshot = streams.snapshot(args.max_stream_age)
            except RuntimeError as exc:
                raise StaleStream(str(exc))
        try:
            # With a TactileRecorder attached (hdf5 format with tactile), obs['tactile'] is taken
            # here, with the images, and a too-old tactile frame raises TactileStale.
            obs = env.get_observation()
        except TactileStale as exc:
            raise StaleStream(str(exc))
        if require_fresh and args.policy in HDF5_POLICIES:
            recorder = env.image_recorder
            now = rospy.get_time()
            for camera in adapter.cameras:
                secs = getattr(recorder, camera + "_secs", None)
                if secs is None:
                    raise StaleStream("no {} image yet".format(camera))
                age = now - (secs + getattr(recorder, camera + "_nsecs") * 1e-9)
                if age > args.max_stream_age:
                    raise StaleStream("{} image is {:.2f}s old".format(camera, age))
        return Frame(stamp, obs["qpos"], obs["qvel"], obs["effort"], obs["images"], vtla_snapshot,
                     tactile=obs.get("tactile"), timestamps=obs.get("timestamps"))

    def infer(frame, timeout=None):
        obs = adapter.prepare(frame)
        if obs.get("prompt", "") is None:
            del obs["prompt"]  # openpi / lerobot fall back to the config's / server's default prompt
        return adapter.decode(client.infer(obs, timeout=timeout)["actions"], frame)

    pedal_keys = {args.pedal_pause: "pause", args.pedal_correct: "correct", args.pedal_end: "end"}
    inputs = HilInput(args.pedal_device, pedal_keys)
    if not inputs.keyboard.enabled:
        parser.error("hil_rollout.py needs a terminal (stdin is not a tty)")
    settings = {key: getattr(args, key) for key in (
        "action_horizon", "absolute_takeover", "align_s", "release_ramp_s",
        "settle_s", "settle_tolerance", "settle_timeout_s", "correction_blend_s", "correction_skip_s",
        "resume_blend_s", "max_joint_step")}
    settings["policy_hz"] = adapter.policy_hz
    saved = 0
    warmed_up = False
    index = next_index(root)
    try:
        while saved < args.num_episodes:
            name = "episode_{}".format(index)
            print("*" * 100 + "\n{} ({} of {} to save)".format(name, saved + 1, args.num_episodes))
            legacy.opening_ceremony(master_left, master_right, env.puppet_bot_left, env.puppet_bot_right, name,
                                    start_open_ratio=args.start_open_ratio)
            haptics = [None, None]
            if not args.no_force_feedback:
                for i, (side, bot) in enumerate((("left", master_left), ("right", master_right))):
                    feedback, brake = legacy.build_haptics_config(vars(args), side)
                    haptics[i] = legacy.ArmHaptics(bot, side, feedback, brake, torque_gate=not args.no_torque_gate)
                    haptics[i].wait_for_data()
            tactile = None
            if args.format == "hdf5" and task.get("tactile_sensors") and task.get("tactile_outputs"):
                tactile = legacy.TactileRecorder(task["tactile_sensors"], task["tactile_outputs"],
                                                 max_age_s=args.max_tactile_age, jpeg_quality=args.tactile_jpeg_quality)
                tactile.wait_for_data()
            # As record_episodes_aloha2.py: re-zero every tactile sensor in use (followers at the start
            # pose, grippers open, so nothing touches the gels) before the force baseline is taken, and
            # keep the runtime config that zeroing exports for the episode file.
            in_use = {topic_for(name) for name in getattr(tactile, "sensor_names", None) or ()}
            in_use |= {topic_for(name) for name in getattr(streams, "tactile_names", None) or ()}
            in_use |= {topic_for(arm.feedback.topic) for arm in haptics if getattr(arm, "feedback", None) is not None}
            tactile_config = None
            if in_use:
                if args.no_tactile_calibration:
                    tactile_config = current_runtime_config()
                else:
                    print("Zeroing the tactile sensors - nothing may touch them...")
                    tactile_config = calibrate_sensors(sorted(in_use))
            for arm in haptics:
                if arm is not None:
                    arm.calibrate(args.force_baseline_samples)
            clutch = dict(release_ramp_s=args.release_ramp_s, relative=not args.absolute_takeover,
                          settle_s=args.settle_s, settle_tolerance=args.settle_tolerance,
                          settle_timeout_s=args.settle_timeout_s, max_offset_warn=args.max_offset_warn, control_dt=DT)
            leaders = ClutchLeaderPair(ClutchLeader(master_left, "left", haptics=haptics[0], **clutch),
                                       ClutchLeader(master_right, "right", haptics=haptics[1], **clutch))
            leaders.configure()
            follower = FollowerCommander(puppets, max_joint_step=args.max_joint_step,
                                         clip_gripper=args.format == "vtla")
            executor = ChunkExecutor(infer, adapter.policy_hz, DT, action_horizon=args.action_horizon)
            if args.format == "hdf5":
                recorder = Hdf5HilEpisode(camera_names, tactile=tactile)
                if tactile is not None:
                    # capture() -> env.get_observation() now carries this episode's tactile snapshots
                    tactile.reset()
                    env.tactile = tactile
            else:
                models = puppets[0].arm.robot_des, puppets[1].arm.robot_des
                recorder = VtlaHilEpisode(sources, models, calibration, args.aloha_action_space,
                                          args.prompt or args.task_name)

            reason, decision = None, "redo"
            try:
                if not warmed_up:
                    frame = capture(False)
                    adapter.start_episode(frame)
                    print("Warming up the policy server (its first inference compiles the model and can take "
                          "minutes; the robot stays still)...")
                    start = time.monotonic()
                    infer(frame, timeout=args.warmup_timeout)
                    print("Policy server ready ({:.0f}s).".format(time.monotonic() - start))
                    warmed_up = True
                if wait_for(inputs, {"pause": "start", "esc": "quit"},
                            "Ready. RIGHT = start the policy, ESC = quit.") == "quit":
                    reason, decision = "esc", "quit"
                else:
                    adapter.start_episode(capture(False))
                    session = HilSession(
                        follower, leaders, executor, capture, recorder, inputs, DT, max_steps,
                        align_s=args.align_s, correction_blend_s=args.correction_blend_s,
                        correction_skip_s=args.correction_skip_s, resume_blend_s=args.resume_blend_s,
                        on_rezero=lambda: [arm.rezero(args.force_baseline_samples) for arm in haptics if arm])
                    print("Policy running. RIGHT = pause, e = end episode, ESC = quit.")
                    inputs.drain()  # nothing typed or pedalled before the start counts
                    from tqdm.contrib.logging import logging_redirect_tqdm

                    with logging_redirect_tqdm():  # warnings print above the progress bar
                        reason = session.run()
                    print("Episode ended ({}): {} ticks ({} active), {} recorded, {} corrections, {} unrecorded; {}"
                          .format(reason, session.steps, session.active_steps, len(recorder), session.corrections,
                                  session.skipped_frames, executor.stats()))
            finally:
                env.tactile = None
                executor.close()
                if not leaders.locked:
                    leaders.lock()
                leaders.close()
                for arm in haptics:
                    if arm is not None:
                        arm.restore()
            # As record_episodes_aloha2.py: followers' grippers open before asking.
            for bot in puppets:
                bot.dxl.robot_set_operating_modes("single", "gripper", "position")
            legacy.move_grippers(list(puppets), [PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN],
                                 move_time=0.5)

            if len(recorder):
                prompt = "{} ({} frames): RIGHT = save, LEFT = discard".format(name, len(recorder))
                prompt += " (then quit)." if reason == "esc" else ", ESC = discard and quit."
                decision = wait_for(inputs, {"pause": "save", "correct": "redo", "esc": "quit"}, prompt)
            if decision == "save":
                attrs = {"policy": args.policy, "prompt": args.prompt or "", "server": client.uri,
                         "server_metadata": client.metadata, "settings": settings, "task_name": args.task_name}
                try:
                    count = recorder.save(str(root / name) if args.format == "hdf5" else root / name, attrs,
                                          tactile_config=tactile_config)
                except Exception as exc:
                    print("Saving {} failed ({}); recording it again.".format(name, exc))
                else:
                    saved += 1
                    index = next_index(root)
                    print("Saved {} ({} frames).".format(root / name, count))
            else:
                print("{} discarded.".format(name))
            recorder.close()
            if reason == "esc" or decision == "quit":
                break
            if saved < args.num_episodes:
                # opening_ceremony() rewrites operating modes, which briefly drops arm torque: start
                # every episode from the rest pose, as record_episodes_aloha2.py does.
                legacy.sleep_all_robots()
    finally:
        inputs.close()
        if streams is not None:
            streams.close()
        client.close()
        legacy.shut_down_all_robots()
    print("Saved {} episode(s) to {}.".format(saved, root))


if __name__ == "__main__":
    main()
