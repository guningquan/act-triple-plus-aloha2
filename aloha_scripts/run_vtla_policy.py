#!/usr/bin/env python3
"""Run a trained N0-VTLA policy on the real ALOHA robot (client side).

Server (remote GPU box, inside the N0-VTLA repo/env):
    python scripts/serve_policy.py \
        --policy.config=vtla_tactile_posttrain \
        --policy.dir=checkpoints/vtla_tactile_posttrain/my_experiment/5

Client (this script, the machine wired to the robot, this repo's ROS env):
    roslaunch aloha 4arms_tactile_teleop.launch
    python3 aloha_scripts/run_vtla_policy.py --host 10.240.77.20 --port 8000 \
        --prompt "Perform the task."

This is NOT the openpi aloha_real example adapted in place - that example's server expects a
plain {"state": qpos(14), "images": {cam_name: HWC}} joint-space observation with no tactile
input, which is a different schema from what vtla_tactile_posttrain was trained on. The
observation/action contract below was read directly out of the N0-VTLA server it talks to:
  - n0vtla/policies/canonical_tactile_policy.py (CanonicalTactileInputs: the input schema)
  - n0vtla/training/config.py (LeRobotCanonicalTaskTactileDataConfig: repack + delta-action mask)
  - n0vtla/policies/policy.py + policy_config.py (exact transform order run by policy.infer())
using the SAME kinematics this repo's own recorder already established
(record_episodes_aloha2_vtla.py: joints_to_eef, via each puppet arm's own robot_des M/Slist).

Wire protocol
-------------
websocket + msgpack, numpy arrays packed as {"__ndarray__": True, "data", "dtype", "shape"}.
This is byte-for-byte what openpi_client/n0vtla_client speak (see their websocket_client_policy.py
+ msgpack_numpy.py), reimplemented in policy_client.py (shared with hil_rollout.py) instead of
depending on either package: this script runs under the ROS box's system Python 3.8, which has no
reason to carry jax/torch just to pack a dict over a socket. Needs websockets + msgpack (both in
the `aloha` conda env).

--action_space is REQUIRED (no default) and must match how the CONNECTED CHECKPOINT was
trained - it picks both the wire shape/dimensionality sent and how the returned action is
interpreted:
    eef    server config vtla_tactile_posttrain (LeRobotCanonicalTaskTactileDataConfig ->
           CanonicalTactileInputs). observation.state is (32,) - EEF_LAYOUT occupies [0:20],
           dims [20:32] are zero padding (confirmed against
           Dataset/board_insertion_vtla/meta/info.json: observation.state/action are both
           shape [32] on disk, NOT 20 - convert_canonical_data.py pads every robot/mode to the
           canonical 32-dim container). Server returns (horizon, 32) absolute EEF poses,
           IK-solved into joint targets client-side (solve_arm()).
    joint  server config vtla_joint_posttrain (LeRobotAlohaTactileJointDataConfig ->
           FlexivTactileInputs / FlexivJointOutputs). observation.state is ALSO (32,), not
           14 - JOINT_LAYOUT occupies [0:14] (matches record_episodes_aloha2_vtla.JOINT_COLS),
           dims [14:32] are zero padding (same on-disk confirmation, against
           Dataset/board_insertion_vtla_joint/meta/info.json). Server returns (horizon, 14)
           absolute joint targets directly (FlexivJointOutputs trims the model's internal
           32-dim action to the real 14 before replying) - no IK, just clamped to each arm's
           own limits (clamp_joint_target()) and sent to set_joint_positions as-is.

Observation sent per inference - the wire SHAPE is different per --action_space, because the
two server configs run different input transforms and `create_trained_policy` does NOT run
either DataConfig's RepackTransform (only the data_transforms.inputs the config sets are
applied at serve time; the repack that maps raw dotted LeRobot column names -> these keys is a
training-time-only step). Sending the wrong shape to the wrong server fails with
`KeyError: 'images'` (sent flat keys to a joint/FlexivTactileInputs server) or a shape/key
mismatch the other way around.

  --action_space eef  (CanonicalTactileInputs: flat, dotted-key dict at the top level):
    observation.image.third_view                HWC uint8 RGB          (cam_high or cam_low)
    observation.image.left_wrist_view           HWC uint8 RGB          (cam_left_wrist)
    observation.image.right_wrist_view          HWC uint8 RGB          (cam_right_wrist)
    observation.image.left_wrist_left_tactile     (2,H,W,C) uint8 RGB   [baseline, current] (XENSE rectify, native 700x400)
    observation.image.left_wrist_right_tactile    (2,H,W,C) uint8 RGB   [baseline, current]
    observation.image.right_wrist_left_tactile    (2,H,W,C) uint8 RGB   [baseline, current]
    observation.image.right_wrist_right_tactile   (2,H,W,C) uint8 RGB   [baseline, current]
    observation.state                             float32 (32,)
    prompt                                        str

  --action_space joint  (FlexivTactileInputs/_build_inputs: nested dict - see
  n0vtla/policies/flexiv_policy.py::_build_inputs, which does `data["images"]["cam_high"]` etc.
  directly, not a dotted top-level key):
    images: {cam_high, cam_left_wrist, cam_right_wrist}   each HWC uint8 RGB
    tactile: {left_wrist_left_tactile, left_wrist_right_tactile,
              right_wrist_left_tactile, right_wrist_right_tactile}   each (2,H,W,C) uint8 RGB
              [baseline, current] - SHORT keys (no "observation.image." prefix), matching
              N0VTLAConfig.tactile_image_keys = canonical_schema.TACTILE_SHORT_KEYS.
    state                                          float32 (32,)
    prompt                                         str

No resize is done client-side - the server's model_transforms do that (ResizeImages(224,224)),
so this script sends native-resolution frames, same as training saw before that step.

Tactile baseline (T_0): CanonicalTactileInputs reads each tactile view as a stacked array
-- index 0 = baseline, index 1 = current -- and diffs them in pixel space before the tactile
encoder. Handing it a bare (H,W,C) frame instead is a documented fallback that substitutes an
all-BLACK placeholder for T_0; it does not error, it silently feeds the model a raw tactile
frame as if it were a contact-change signal, which the policy never saw in training.
capture_tactile_baseline() grabs T_0 once per episode (right after homing, before any
contact), and build_observation() pairs it with the live frame on every subsequent call.

Color order - the one detail in this file that is NOT "the obvious thing":
    record_episodes_aloha2_vtla.py deliberately converts XENSE's rectify bytes from RGB to BGR
    before handing them to ffmpeg's "-pixel_format bgr24", because that is what makes a *video
    player* show true colors (see that file's tactile_callback for the full story). But the
    training dataloader (lerobot's decode_video_frames -> torchvision.io.VideoReader) always
    decodes video back out to true RGB - which is what the model actually trained on, regardless
    of the intermediate video codec's internal colorspace bookkeeping. This live path has no
    video round-trip to undo, so it must reproduce "true RGB" directly:
      - camera images: cv_bridge with desired_encoding="passthrough" (usb_cam natively publishes
        rgb8 - see openpi's own examples/aloha_real/robot_utils.py, which does exactly this).
      - tactile images: the SDK's raw "rectify" bytes, UNCHANGED. Do not apply the
        RGB2BGR fix that the three recording-path files use - that fix is specifically there to
        cancel out on a *video decode*, and applying it here would feed the model backwards.

Action received per inference, already de-delta'd by the server's own AbsoluteActions transform
(so this is the state to command, not a state to add to something first):
    --action_space eef (response["actions"]: (action_horizon=50, 32) float32):
        [0:3]   left  xyz (m, puppet_left/base_link)      [3:9]   left  rot6d
        [9]     left  gripper, 0=closed..1=open (unclipped by the model; clipped here)
        [10:13] right xyz                                  [13:19] right rot6d
        [19]    right gripper
        [20:32] unused padding (all-zero action_mask on this task's data; ignored)
    --action_space joint (response["actions"]: (action_horizon=50, 14) float32):
        [0:6]   left  joints j1..6 (rad)                   [6]     left  gripper, 0..1
        [7:13]  right joints j1..6 (rad)                    [13]    right gripper, 0..1
Only --action_horizon of the 50 returned steps are executed before re-inferring (matching
openpi's ActionChunkBroker), at 1/30s per step - the training resample rate, NOT this repo's
usual 50Hz teleop/recording DT. Executing the chunk at the wrong rate plays the trajectory back
at the wrong speed relative to what the model's dynamics were conditioned on.
"""

import argparse
import logging
import time

import cv2
import numpy as np

from constants import (PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_LEFT_GRIPPER_JOINT_UNNORMALIZE_FN,
                      PUPPET_RIGHT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN,
                      START_ARM_POSE)
from one_side_teleop import KeyboardControl
from policy_client import PolicyClient
from real_env import make_real_env
from robot_utils import move_arms, move_grippers
from sleep_plus import shut_down_puppet_robots
from tactile_stream import DEFAULT_MAX_TACTILE_AGE_S, TactileStream
from record_episodes_aloha2_vtla import (
    DEFAULT_TACTILE_MAP,
    FPS,
    JOINT_COLS,
    TACTILE_SENSORS,
    joint_state_vector,
    joints_to_eef,
    video_sources,
)

# EEF_LAYOUT documents the 20-dim observation.state / first 20 dims of the 32-dim action, both
# in the layout joints_to_eef() already produces and canonical_schema.ACTION_SLOTS expects.
# Only used when --action_space eef.
EEF_LAYOUT = {
    "left_xyz": slice(0, 3), "left_rot6d": slice(3, 9), "left_gripper": 9,
    "right_xyz": slice(10, 13), "right_rot6d": slice(13, 19), "right_gripper": 19,
}
ACTION_DIM = 32

# JOINT_LAYOUT documents the 14-dim observation.state / action layout for --action_space
# joint - matches record_episodes_aloha2_vtla.JOINT_COLS exactly (a vtla_joint_posttrain
# -style checkpoint's data config reads that same recorder output). Unlike the eef action, a
# joint-mode server response is (horizon, 14), not (horizon, 32) - there is no shared 32-dim
# padded wire format between the two modes (confirmed against n0vtla's own
# FlexivJointOutputs/canonical_schema.py - joint mode is a separate, unpadded output transform).
JOINT_LAYOUT = {
    "left_joints": slice(0, 6), "left_gripper": 6,
    "right_joints": slice(7, 13), "right_gripper": 13,
}
STALE_STREAM_S = 0.5  # refuse to act on a camera/tactile frame older than this


# --------------------------------------------------------------------------------------------
# Observation capture: cameras + tactile, kept in true RGB (see module docstring). This is a
# deliberately separate, smaller class from record_episodes_aloha2_vtla.LiveStreams - that one
# converts to BGR for the video encoder, which is exactly the one thing this script must not do.
# --------------------------------------------------------------------------------------------

class RGBStreams:
    """Cameras by receipt age (max_age); tactile through the shared tactile_stream.TactileStream,
    by the data's own age (now - the publisher's local_read_end, max_tactile_age), with rectify in
    true RGB prepared in the stream's callback thread (decoded first if the publisher sends JPEG)."""

    def __init__(self, sources, tactile_map, max_tactile_age=DEFAULT_MAX_TACTILE_AGE_S):
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self._rospy = rospy
        self._bridge = CvBridge()
        self._latest = {}
        self._subs = []
        self.max_tactile_age = float(max_tactile_age)
        self.camera_names = sorted({source for key, source in sources.items() if source.startswith("cam_")})
        self.tactile_names = sorted(set(tactile_map.values()))
        if set(self.tactile_names) != set(TACTILE_SENSORS):
            raise ValueError("Tactile mapping must cover all four sensors: {}".format(TACTILE_SENSORS))
        for camera in self.camera_names:
            topic = "/usb_{}/image_raw".format(camera)
            self._subs.append(rospy.Subscriber(topic, Image, self._camera_cb, callback_args=camera,
                                               queue_size=1, buff_size=2 ** 24, tcp_nodelay=True))
        # Deliberately "rgb": the raw publisher's rectify bytes are already the true RGB the model
        # trained on (see module docstring); only the recording path flips to BGR.
        self._tactile = {sensor: TactileStream.acquire(sensor, prefetch=(("rectify", "rgb"),))
                         for sensor in self.tactile_names}

    def _camera_cb(self, message, name):
        try:
            # passthrough, NOT bgr8: usb_cam natively publishes rgb8, and the model wants true
            # RGB (see module docstring). Converting here would be correct for a human viewer
            # and wrong for this model.
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            self._latest[name] = (np.array(frame, copy=True, order="C"), time.monotonic())
        except Exception as exc:
            self._rospy.logwarn_throttle(5, "camera {}: {}".format(name, exc))

    def snapshot(self, max_age, max_tactile_age=None):
        tactile_limit = self.max_tactile_age if max_tactile_age is None else max_tactile_age
        now, now_wall = time.monotonic(), time.time()
        frames = {n: self._tactile[n].latest() for n in self.tactile_names}
        missing = [n for n in self.camera_names if n not in self._latest] + [n for n, f in frames.items() if f is None]
        if missing:
            raise RuntimeError("no data yet from: {}".format(missing))
        stale = [n for n in self.camera_names if now - self._latest[n][1] > max_age]
        stale += [n for n, f in frames.items() if f.age(now_wall) > tactile_limit]
        if stale:
            raise RuntimeError("stale (camera > {}s, tactile > {}s old): {}".format(max_age, tactile_limit, stale))
        out = {n: self._latest[n][0] for n in self.camera_names}
        for name, frame in frames.items():
            image = frame.field("rectify", "rgb")
            if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise RuntimeError("tactile {}: rectify must be a uint8 HxWx3 image".format(name))
            out[name] = np.array(image, copy=True)
        return out

    def wait_fresh_tactile(self, after, timeout_s=1.0):
        """Wait until every tactile sensor has a frame the SDK delivered after host time `after`."""
        for name in self.tactile_names:
            if self._tactile[name].wait_newer_than(after, timeout_s) is None:
                raise RuntimeError("no fresh tactile frame from {} within {:.1f} s".format(name, timeout_s))

    def wait_ready(self, timeout_s, max_age):
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                self.snapshot(max_age)
                return
            except RuntimeError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(str(exc))
                time.sleep(0.05)

    def close(self):
        for sub in self._subs:
            sub.unregister()
        self._subs = []
        for stream in self._tactile.values():
            stream.release()
        self._tactile = {}


# --------------------------------------------------------------------------------------------
# rot6d <-> matrix. The decode side is copied verbatim (same Gram-Schmidt convention, same
# degenerate-input guard) from n0vtla/transforms.py:_rot6d_to_matrix so a policy that hiccups on
# an OOD frame and emits a near-zero/collinear rot6d cannot NaN its way into commanding a joint
# angle - it falls back to holding the arm's current orientation for that one control tick.
# --------------------------------------------------------------------------------------------

def rot6d_to_matrix(d6):
    a1, a2 = np.asarray(d6[0:3], dtype=np.float64), np.asarray(d6[3:6], dtype=np.float64)
    n1 = np.linalg.norm(a1)
    b1 = a1 / n1 if n1 >= 1e-8 else a1
    a2p = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(a2p)
    b2 = a2p / n2 if n2 >= 1e-8 else a2p
    if n1 < 1e-6 or n2 < 1e-6:
        return np.eye(3), False
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1), True  # columns are b1, b2, b3


def solve_arm(puppet_bot, xyz, rot6d, guess):
    """IK for one arm's predicted absolute EEF pose. Returns (theta_list, ok, new_guess)."""
    rotation, rot_ok = rot6d_to_matrix(rot6d)
    if not rot_ok or not np.isfinite(xyz).all():
        return None, False, guess
    target = np.eye(4)
    target[:3, :3] = rotation
    target[:3, 3] = xyz
    theta_list, ok = puppet_bot.arm.set_ee_pose_matrix(target, custom_guess=list(guess), execute=False)
    if not ok:
        return None, False, guess
    return theta_list, True, theta_list


def clamp_joint_target(puppet_bot, joints):
    """Clamp a predicted joint-position action to this arm's own limits before sending it.

    joint mode has no IK step, so it also has none of set_ee_pose_matrix's built-in
    position/velocity gate (see solve_arm) - an OOD/noisy prediction would otherwise go straight
    to set_joint_positions with no safety net at all. Position limits are checked here (cheap,
    and the arm will refuse an out-of-range goal anyway); velocity is intentionally left to
    set_trajectory_time's Profile_Velocity/Profile_Acceleration registers already set once in
    run_episode(), the same as every other joint command this script sends.
    """
    lower = puppet_bot.arm.group_info.joint_lower_limits
    upper = puppet_bot.arm.group_info.joint_upper_limits
    return [float(np.clip(theta, lo, hi)) for theta, lo, hi in zip(joints, lower, upper)]


def home_puppets(env):
    """Move both puppet arms to START_ARM_POSE and open both grippers.

    setup_puppet_bot() (called by make_real_env) only tourques the arms on and sets their
    operating mode - it never commands a position, so without this the arm starts wherever it
    physically happens to be (wherever teleop, a previous rollout, or gravity left it). Every
    recorded episode this checkpoint trained on began from this same pose
    (record_episodes_aloha2.py's opening_ceremony uses the same START_ARM_POSE), so starting a
    rollout from anywhere else feeds the model an out-of-distribution first observation - and
    since this script drives the arm from whatever the model predicts for that first frame with
    no other sanity check, an OOD state is also a safety concern, not just an accuracy one.
    """
    start_pose = START_ARM_POSE[:6]
    move_arms([env.puppet_bot_left, env.puppet_bot_right], [start_pose, start_pose], move_time=2.0)
    move_grippers([env.puppet_bot_left, env.puppet_bot_right],
                  [PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN], move_time=0.5)


def park_puppets_on_quit():
    """ESC/q was pressed: open both grippers, walk both puppets to the sleep pose, cut torque.

    Reuses sleep_plus's own two-stage shutdown sequence (shared with record_episodes_aloha2.py /
    one_side_teleop.py) rather than a straight-line move, since going directly from a rollout's
    working pose to the sleep pose has been observed to drop the arm when torque cuts.
    """
    print("ESC/q: stopping. Opening grippers and moving both puppets to the sleep position...")
    shut_down_puppet_robots()
    print("Puppets are at the sleep position with torque off.")


# --------------------------------------------------------------------------------------------
# Main control loop
# --------------------------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="Policy server host, e.g. 10.240.77.20")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action-space", "--action_space", dest="action_space",
                        choices=("eef", "joint"), required=True,
                        help="Must match how the CONNECTED CHECKPOINT was trained/converted "
                             "(record_episodes_aloha2_vtla.py's --aloha-action-space eef / "
                             "joint_position - 'joint' here is that same joint_position mode, "
                             "just spelled shorter) - there is no default on purpose, so a "
                             "mismatched checkpoint/mode pairing has to be a deliberate choice, "
                             "not a silent one. 'eef': server returns (horizon, 32) absolute EEF "
                             "poses, IK-solved into joint targets here (see EEF_LAYOUT). "
                             "'joint': server returns (horizon, 14) absolute joint targets "
                             "directly (see JOINT_LAYOUT) - no IK, sent to the arms as-is "
                             "(clamped to each arm's own joint limits first).")
    parser.add_argument("--prompt", default="Perform the task.",
                        help="Sent as the 'prompt' key every inference. The server also has a "
                             "training-config fallback prompt, but relying on it silently ties "
                             "every rollout to whatever task the checkpoint's config hardcodes.")
    parser.add_argument("--third_camera", choices=("cam_high", "cam_low"), default="cam_high",
                        help="Must match --third_camera used when this checkpoint's data was "
                             "recorded (record_episodes_aloha2_vtla.py's own default).")
    for slot, sensor in DEFAULT_TACTILE_MAP.items():
        parser.add_argument("--" + slot, choices=TACTILE_SENSORS, default=sensor)
    parser.add_argument("--action_horizon", type=int, default=50,
                        help="How many of the server's 50-step chunk to execute before "
                             "re-inferring. Lower = more reactive to new images, more network "
                             "round trips. Default 50 (~0.33s of open-loop execution at 30Hz).")
    parser.add_argument("--max_episode_steps", type=int, default=2000)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--gripper_moving_time", type=float, default=None,
                        help="Arm trajectory time per executed chunk step. Default 1/30s, "
                             "matching the training resample rate; only override to smooth out "
                             "visible per-tick jerk at the cost of tracking lag.")
    parser.add_argument("--stream_timeout", type=float, default=5.0)
    parser.add_argument("--max_stream_age", type=float, default=STALE_STREAM_S)
    parser.add_argument("--max_tactile_age", type=float, default=DEFAULT_MAX_TACTILE_AGE_S,
                        help="Refuse tactile frames older than this (s), by the data's own age "
                             "(the publisher's local_read_end), not by receipt")
    parser.add_argument("--debug", action="store_true",
                        help="Inspect mode: home the arms, then on each ENTER capture one "
                             "observation, print its shapes, show its 7 images (q = next image, "
                             "ESC = skip the rest), send it to the server, print the returned "
                             "action chunk's shape, and IK-solve the whole chunk into a joint-space "
                             "plan whose shape is also printed - all WITHOUT commanding the arms. "
                             "Type 'q' at the ENTER prompt (or Ctrl+C) to quit.")
    return parser


def capture_tactile_baseline(streams, max_age):
    """Snapshot every tactile view once, to serve as this episode's zero-contact baseline T_0.

    Call exactly once per episode, right after homing and before any contact - build_observation()
    pairs every later live frame against whatever this returns, so a stale or mid-contact
    baseline here silently poisons the tactile diff for the whole episode. Keyed by tactile
    *source* name (e.g. "left_gripper_outer"), matching RGBStreams.tactile_names / snapshot().
    Every tactile view is a frame the SDK delivered after this call started, i.e. after homing.
    """
    if hasattr(streams, "wait_fresh_tactile"):
        streams.wait_fresh_tactile(time.time())
    frames = streams.snapshot(max_age)
    return {source: frames[source] for source in streams.tactile_names}


# Model-side camera key for each canonical view name video_sources() produces
# ("observation.image.<CAM_MODEL_KEY key>" -> the FlexivTactileInputs/_build_inputs key).
CAM_MODEL_KEY = {
    "third_view": "cam_high",
    "left_wrist_view": "cam_left_wrist",
    "right_wrist_view": "cam_right_wrist",
}


def build_observation(action_space, streams, sources, prompt, max_age, state, tactile_baseline):
    """`state` is the 32-dim padded vector from compute_observation_state() for BOTH modes.

    The wire SHAPE differs by server/config, not just `state`'s width - see the module
    docstring's "Observation sent per inference" section:
      eef   -> vtla_tactile_posttrain / CanonicalTactileInputs: flat dotted-key dict.
      joint -> vtla_joint_posttrain / FlexivTactileInputs: nested {images, tactile, state, prompt}.
    """
    frames = streams.snapshot(max_age)
    return observation_from_frames(action_space, frames, streams.tactile_names, sources, prompt, state,
                                   tactile_baseline)


def observation_from_frames(action_space, frames, tactile_names, sources, prompt, state, tactile_baseline):
    """build_observation() on already-captured true-RGB frames ({source name: HWC uint8}), so
    hil_rollout.py (via policy_client.N0VtlaAdapter) sends the exact same wire layout."""
    state = np.asarray(state, dtype=np.float32)

    if action_space == "joint":
        images, tactile = {}, {}
        for key, source in sources.items():
            short = key[len("observation.image."):]
            if source in tactile_names:
                # index 0 = baseline T_0, index 1 = current T_t - see module docstring.
                tactile[short] = np.stack([tactile_baseline[source], frames[source]], axis=0)
            else:
                images[CAM_MODEL_KEY[short]] = frames[source]
        return {"images": images, "tactile": tactile, "state": state, "prompt": prompt}

    obs = {}
    for key, source in sources.items():
        if source in tactile_names:
            obs[key] = np.stack([tactile_baseline[source], frames[source]], axis=0)
        else:
            obs[key] = frames[source]
    obs["observation.state"] = state
    obs["prompt"] = prompt
    return obs


def compute_observation_state(action_space, left_joints, right_joints, left_names, right_names,
                              calibration, models):
    """left_joints/right_joints are live dxl.joint_states; returns the 32-dim, zero-padded
    canonical vector build_observation() sends as `state`/`observation.state` for EITHER mode -
    both on-disk datasets (Dataset/board_insertion_vtla*/meta/info.json) store observation.state
    as shape [32], confirming convert_canonical_data.py pads every robot/action-space to the
    canonical 32-dim container, not just the EEF one.
      eef:   dims [0:20] = joints_to_eef()'s 20-dim EEF pose (EEF_LAYOUT), [20:32] = 0.
      joint: dims [0:14] = the raw JOINT_COLS-ordered joint vector, [14:32] = 0.
    """
    left_state = joint_state_vector(left_joints, left_names, *calibration[0])
    right_state = joint_state_vector(right_joints, right_names, *calibration[1])
    joints14 = np.concatenate([left_state, right_state])
    state = np.zeros(32, dtype=np.float32)
    if action_space == "eef":
        state[:20] = joints_to_eef(joints14[np.newaxis, :], models)[0]
    else:
        state[:14] = joints14
    return state


# --------------------------------------------------------------------------------------------
# --debug: inspect one observation/action pair interactively, without ever moving the arms.
# --------------------------------------------------------------------------------------------

def _flatten_obs_images(obs):
    """obs.items(), but one level deep into a nested {"images": {...}, "tactile": {...}} dict
    (--action_space joint's wire shape) so the debug helpers below don't need to special-case it.
    Flat (--action_space eef) observations pass through unchanged."""
    for key, value in obs.items():
        if key in ("images", "tactile") and isinstance(value, dict):
            for sub_key, sub_value in value.items():
                yield "{}.{}".format(key, sub_key), sub_value
        else:
            yield key, value


def print_observation(obs):
    print("observation:")
    for key, value in _flatten_obs_images(obs):
        if isinstance(value, np.ndarray):
            print("  {:42s} shape={} dtype={}".format(key, value.shape, value.dtype))
        else:
            print("  {:42s} = {!r}".format(key, value))


def show_observation_images(obs):
    """cv2.imshow each image key in turn; 'q' advances, ESC skips the rest. Blocks on each frame.

    Captured frames are true RGB (see module docstring) so this converts to BGR for the window -
    display-only, the array sent to the server in `obs` is never touched. Tactile keys are the
    (2,H,W,C) [baseline, current] stack build_observation() sends, so each is shown as two
    separate frames instead of one.
    """
    images = []
    for key, value in _flatten_obs_images(obs):
        if not isinstance(value, np.ndarray):
            continue
        if value.ndim == 3:
            images.append((key, value))
        elif value.ndim == 4:
            images.append((key + " [baseline T_0]", value[0]))
            images.append((key + " [current T_t]", value[1]))
    if not images:
        return
    window = "obs image (q = next, ESC = skip remaining)"
    try:
        for label, image in images:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(window, bgr)
            cv2.setWindowTitle(window, label)
            print("  showing {} - press q for next, ESC to skip the rest".format(label))
            while True:
                pressed = cv2.waitKey(0) & 0xFF
                if pressed in (ord("q"), 27):
                    break
            if pressed == 27:
                break
    except cv2.error as exc:
        print("  display unavailable ({}); skipping image preview.".format(exc))
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass  # never got shown (e.g. no display), or already gone


def plan_joint_trajectory(env, models, actions, horizon, left_seed, right_seed):
    """IK-solve a whole action chunk into a joint-space plan. Never calls set_joint_positions /
    set_gripper_pose - `solve_arm` itself only ever calls set_ee_pose_matrix(execute=False), so
    nothing here can command the real arms.

    Returns (targets, delta, ok) where:
      targets: (horizon, 14) float64, JOINT_COLS layout [left_j1..6, left_gripper,
               right_j1..6, right_gripper] - left_gripper/right_gripper here are UNNORMALIZED
               joint angles (constants.PUPPET_*_GRIPPER_JOINT_UNNORMALIZE_FN), not the raw
               0..1 fraction the model predicts, so this array is directly comparable to a
               puppet arm's own joint_states.
      delta:   targets - the seed joint state broadcast over the horizon (the "joint change").
      ok:      (horizon, 2) bool, whether IK found a limits-respecting solution for [left, right]
               at each step. A False step repeats the previous step's joint target (hold), same
               fallback run_episode() uses live.
    """
    seed0 = np.concatenate([left_seed, [0.0], right_seed, [0.0]])  # gripper slots unused as a seed
    targets = np.empty((horizon, 14), dtype=np.float64)
    ok = np.zeros((horizon, 2), dtype=bool)
    left_guess, right_guess = list(left_seed), list(right_seed)
    prev_left, prev_right = list(left_seed), list(right_seed)
    for h in range(horizon):
        row = actions[h]
        left_theta, left_ok, left_guess = solve_arm(
            env.puppet_bot_left, row[EEF_LAYOUT["left_xyz"]], row[EEF_LAYOUT["left_rot6d"]], left_guess)
        right_theta, right_ok, right_guess = solve_arm(
            env.puppet_bot_right, row[EEF_LAYOUT["right_xyz"]], row[EEF_LAYOUT["right_rot6d"]], right_guess)
        prev_left = left_theta if left_ok else prev_left
        prev_right = right_theta if right_ok else prev_right
        left_gripper = PUPPET_LEFT_GRIPPER_JOINT_UNNORMALIZE_FN(float(np.clip(row[EEF_LAYOUT["left_gripper"]], 0.0, 1.0)))
        right_gripper = PUPPET_RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN(float(np.clip(row[EEF_LAYOUT["right_gripper"]], 0.0, 1.0)))
        targets[h] = np.concatenate([prev_left, [left_gripper], prev_right, [right_gripper]])
        ok[h] = (left_ok, right_ok)
    delta = targets - seed0[np.newaxis, :]
    return targets, delta, ok


def debug_loop(client, env, streams, sources, models, args):
    print("Homing both puppet arms to START_ARM_POSE...")
    home_puppets(env)
    print("Homed.\n")
    left_names = env.puppet_bot_left.arm.group_info.joint_names
    right_names = env.puppet_bot_right.arm.group_info.joint_names
    calibration = (args._left_calibration, args._right_calibration)

    tactile_baseline = capture_tactile_baseline(streams, args.max_stream_age)
    print("Captured tactile baseline (T_0) for: {}\n".format(sorted(tactile_baseline)))

    while True:
        try:
            typed = input("Press ENTER to capture+infer ('q' to quit): ").strip().lower()
        except EOFError:
            break
        if typed in ("q", "quit", "exit"):
            break

        left_joints = env.puppet_bot_left.dxl.joint_states
        right_joints = env.puppet_bot_right.dxl.joint_states
        state = compute_observation_state(args.action_space, left_joints, right_joints,
                                          left_names, right_names, calibration, models)

        obs = build_observation(args.action_space, streams, sources, args.prompt,
                                args.max_stream_age, state, tactile_baseline)
        print()
        print_observation(obs)
        show_observation_images(obs)

        print("\nRunning inference...")
        t0 = time.monotonic()
        result = client.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float64)
        print("actions: shape={} dtype={}  (server {:.0f}ms)".format(
            actions.shape, actions.dtype, 1000 * (time.monotonic() - t0)))

        horizon = min(args.action_horizon, actions.shape[0])
        if args.action_space == "eef":
            targets, delta, ok = plan_joint_trajectory(
                env, models, actions, horizon, left_joints.position[:6], right_joints.position[:6])
            print("joint-space plan (IK only, arms NOT commanded):")
            print("  targets: shape={} dtype={}  [left_j1..6, left_gripper, right_j1..6, right_gripper]"
                  .format(targets.shape, targets.dtype))
            print("  delta (targets - current): shape={} dtype={}".format(delta.shape, delta.dtype))
            print("  IK ok per step [left, right]: {}/{} left ok, {}/{} right ok".format(
                int(ok[:, 0].sum()), horizon, int(ok[:, 1].sum()), horizon))
            if not ok.all():
                first_bad = np.argwhere(~ok)[0]
                print("  first IK failure at step {}, arm={}".format(
                    first_bad[0], "left" if first_bad[1] == 0 else "right"))
        else:
            targets = actions[:horizon, :14]
            print("joint-space plan (no IK needed, arms NOT commanded):")
            print("  targets: shape={} dtype={}  [left_j1..6, left_gripper, right_j1..6, right_gripper]"
                  .format(targets.shape, targets.dtype))
            print("  min={} max={}".format(targets.min(axis=0), targets.max(axis=0)))
        print()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def run_episode(client, env, streams, sources, models, args, keyboard):
    """Returns "done" (ran max_episode_steps), "quit" (ESC/q - caller parks+torques off), or
    "reset" (LEFT arrow/r - abort THIS rollout only; the arms are simply left where they are and
    main()'s loop re-homes and waits for RIGHT again for the next round, same as after "done")."""
    moving_time = args.gripper_moving_time or (1.0 / FPS)
    accel_time = min(moving_time / 2.0, 1.0 / FPS)
    for bot in (env.puppet_bot_left, env.puppet_bot_right):
        bot.arm.set_trajectory_time(moving_time=moving_time, accel_time=accel_time)

    left_names = env.puppet_bot_left.arm.group_info.joint_names
    right_names = env.puppet_bot_right.arm.group_info.joint_names
    calibration = [args._left_calibration, args._right_calibration]

    # T_0: captured once, right after homing (this function is only ever called immediately
    # after home_puppets() in main()) and before the arm has touched anything - the same
    # "first frame of the episode" convention the recorder's delta_timestamps stack used at
    # train time. Re-used for every build_observation() call below; never recaptured mid-episode.
    tactile_baseline = capture_tactile_baseline(streams, args.max_stream_age)
    logging.info("Captured tactile baseline (T_0) for: %s", sorted(tactile_baseline))

    keyboard.drain()
    step = 0
    while step < args.max_episode_steps:
        for event in keyboard.poll():
            if event == "esc":
                return "quit"
            if event in ("left", "reset"):
                return "reset"

        left_joints = env.puppet_bot_left.dxl.joint_states
        right_joints = env.puppet_bot_right.dxl.joint_states
        state = compute_observation_state(args.action_space, left_joints, right_joints,
                                          left_names, right_names, calibration, models)

        obs = build_observation(args.action_space, streams, sources, args.prompt,
                                args.max_stream_age, state, tactile_baseline)
        result = client.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float64)
        min_dim = 20 if args.action_space == "eef" else 14
        if actions.ndim != 2 or actions.shape[1] < min_dim:
            raise RuntimeError("Unexpected action shape from server for --action_space "
                               "{}: {}".format(args.action_space, actions.shape))

        left_guess = list(left_joints.position[:6])
        right_guess = list(right_joints.position[:6])
        horizon = min(args.action_horizon, actions.shape[0])
        for h in range(horizon):
            for event in keyboard.poll():
                if event == "esc":
                    return "quit"
                if event in ("left", "reset"):
                    return "reset"
            tick = time.monotonic()
            row = actions[h]

            if args.action_space == "eef":
                left_theta, left_ok, left_guess = solve_arm(
                    env.puppet_bot_left, row[EEF_LAYOUT["left_xyz"]], row[EEF_LAYOUT["left_rot6d"]], left_guess)
                if left_ok:
                    env.puppet_bot_left.arm.set_joint_positions(left_theta, moving_time=moving_time,
                                                                accel_time=accel_time, blocking=False)
                else:
                    logging.warning("left arm: no IK solution this tick, holding position")

                right_theta, right_ok, right_guess = solve_arm(
                    env.puppet_bot_right, row[EEF_LAYOUT["right_xyz"]], row[EEF_LAYOUT["right_rot6d"]], right_guess)
                if right_ok:
                    env.puppet_bot_right.arm.set_joint_positions(right_theta, moving_time=moving_time,
                                                                 accel_time=accel_time, blocking=False)
                else:
                    logging.warning("right arm: no IK solution this tick, holding position")

                left_gripper = float(np.clip(row[EEF_LAYOUT["left_gripper"]], 0.0, 1.0))
                right_gripper = float(np.clip(row[EEF_LAYOUT["right_gripper"]], 0.0, 1.0))
            else:
                # joint: the server already returns absolute joint targets (de-delta'd
                # the same way the eef path's poses are, see module docstring) - no IK, just
                # clamp to this arm's own limits and send. Held (not commanded) if non-finite,
                # the same fallback the eef path takes on a degenerate/OOD prediction.
                left_row = row[JOINT_LAYOUT["left_joints"]]
                if np.isfinite(left_row).all():
                    env.puppet_bot_left.arm.set_joint_positions(
                        clamp_joint_target(env.puppet_bot_left, left_row),
                        moving_time=moving_time, accel_time=accel_time, blocking=False)
                else:
                    logging.warning("left arm: non-finite joint prediction this tick, holding position")

                right_row = row[JOINT_LAYOUT["right_joints"]]
                if np.isfinite(right_row).all():
                    env.puppet_bot_right.arm.set_joint_positions(
                        clamp_joint_target(env.puppet_bot_right, right_row),
                        moving_time=moving_time, accel_time=accel_time, blocking=False)
                else:
                    logging.warning("right arm: non-finite joint prediction this tick, holding position")

                left_gripper = float(np.clip(row[JOINT_LAYOUT["left_gripper"]], 0.0, 1.0))
                right_gripper = float(np.clip(row[JOINT_LAYOUT["right_gripper"]], 0.0, 1.0))

            env.set_gripper_pose(left_gripper, right_gripper)

            step += 1
            if step >= args.max_episode_steps:
                break
            time.sleep(max(0.0, (1.0 / FPS) - (time.monotonic() - tick)))
    return "done"


def main():
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    print("--action_space = {} (must match the connected checkpoint's training data - "
          "see --help if unsure).".format(args.action_space))

    from constants import PUPPET_LEFT_GRIPPER_JOINT_CLOSE, PUPPET_RIGHT_GRIPPER_JOINT_CLOSE
    args._left_calibration = (PUPPET_LEFT_GRIPPER_JOINT_CLOSE, PUPPET_LEFT_GRIPPER_JOINT_OPEN)
    args._right_calibration = (PUPPET_RIGHT_GRIPPER_JOINT_CLOSE, PUPPET_RIGHT_GRIPPER_JOINT_OPEN)

    tactile_map = {slot: getattr(args, slot) for slot in DEFAULT_TACTILE_MAP}
    sources = video_sources(args.third_camera, tactile_map)

    client = PolicyClient(args.host, args.port)
    logging.info("Connected. Server metadata: %s", client.metadata)

    env = make_real_env(init_node=True, setup_robots=True)
    models = (env.puppet_bot_left.arm.robot_des, env.puppet_bot_right.arm.robot_des)
    streams = RGBStreams(sources, tactile_map, max_tactile_age=args.max_tactile_age)

    if args.debug:
        # debug_loop() uses input(), a line-based blocking read - KeyboardControl's cbreak mode
        # would break normal line editing/echo, so it is deliberately not constructed here.
        try:
            logging.info("Waiting for camera/tactile streams...")
            streams.wait_ready(args.stream_timeout, args.max_stream_age)
            logging.info("Streams ready.")
            debug_loop(client, env, streams, sources, models, args)
        finally:
            streams.close()
            client.close()
        return

    keyboard = KeyboardControl()
    try:
        logging.info("Waiting for camera/tactile streams...")
        streams.wait_ready(args.stream_timeout, args.max_stream_age)
        logging.info("Streams ready.")
        episode = 0
        while episode < args.num_episodes:
            logging.info("Homing both puppet arms to START_ARM_POSE...")
            home_puppets(env)
            if not keyboard.enabled:
                logging.info("Homed. No tty on stdin: starting immediately.")
            else:
                print("Homed. Press the RIGHT arrow key to start manipulation (ESC to quit).")
                started = False
                keyboard.drain()
                while not started:
                    for event in keyboard.poll():
                        if event == "right":
                            started = True
                        elif event == "esc":
                            park_puppets_on_quit()
                            return
                    time.sleep(0.02)
            logging.info("Episode %d/%d: running.", episode + 1, args.num_episodes)
            if keyboard.enabled:
                print("LEFT arrow or r = abort this round and re-home for the next one, "
                      "ESC/q = stop and power down.")
            outcome = run_episode(client, env, streams, sources, models, args, keyboard)
            logging.info("Episode %d/%d: %s.", episode + 1, args.num_episodes, outcome)
            if outcome == "quit":
                park_puppets_on_quit()
                break
            if outcome == "reset":
                # Does NOT advance `episode` or count against --num_episodes - re-homes and
                # retries the SAME round. Otherwise a reset on the last (or only) round would
                # just run the while condition out and silently exit main(), which is exactly
                # the "everything stopped and the program exited" bug this fixes.
                print("Reset: re-homing and retrying this round...")
                continue
            episode += 1
    finally:
        streams.close()
        keyboard.close()
        client.close()


if __name__ == "__main__":
    main()
