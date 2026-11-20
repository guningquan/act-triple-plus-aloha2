"""Robot side of serve_forceguide: session protocol, tactile pipeline and the in-chunk refresh executor.

serve_forceguide (forceguide repo, scripts/forceguide/serve_forceguide.py) serves the frozen pi0.5 P0 and every
guided condition of a campaign as ONE stateful session behind the unchanged openpi WebsocketPolicyServer
(ForceGuide real-robot design, section 6.2). Every request carries "session" and a per-session monotonic "req":

    reset    {"forceguide": "reset", "condition", "policy_hz"}           -> {"forceguide": {"req", "ok", "condition",
                                                                              "fingerprint"}}
    plan     openpi observation + "noise" [H, model_action_dim] + tactile payload + "tactile_step" (+ "ack")
                                                                         -> {"actions": [H, A_out], "forceguide":
                                                                              {"req", "chunk", "version", "K", ...}}
    refresh  {"forceguide": "refresh", "chunk", "k", "k_pin", "tactile_step", tactile payload, "ack"}
                                                                         -> full re-guided [H, A_out] plan whose
                                                                            positions < k_pin equal the executed plan,
                                                                            or a K = 0 reply without actions
    observe  {"forceguide": "observe", tactile payload}                 -> {"forceguide": {"req", "ok",
                                                                              "missing_steps"}}
    tactile payload = "tactile_steps": [n] global steps (ascending), "tactile_frames": {key: uint8 [n, 224, 224, 3]}

What lives here (run_forceguide_policy.py wires it to the robot):
    ServerInfo / ConditionInfo / TactileSpec   the metadata "forceguide" block, the only client configuration source
    ForceGuideAdapter    cameras (letterbox 224 CHW, BGR swap if the P0 trained on BGR), state (RealEnv qpos or VTLA
                         motor-angle grippers via FollowerCommander.measured()), single-arm expansion to 14 dims with
                         the idle arm held at the homing pose and its gripper commanded OPEN (1.0)
    prepare_live_tactile apply_live_codec + prepare_tactile: copies of the forceguide repo's features.py functions
                         ("pad_pil" = openpi_client resize_with_pad, "stretch_pil"), bit-identical (golden test)
    TactileHistory       exact-step frames: step g = the frame captured at the FIRST tick whose policy-step clock
                         position >= g (before that tick's command), never overwritten; bounded ring + step 0
    TactilePayload       per trial: needed_steps() of an anchor step that the server has not confirmed yet
    ForceGuideConnection PolicyClient that reconnects after server errors and checks the server fingerprint
    RefreshWorker        daemon thread owning the second (refresh/observe) connection; replies come back tagged
    RefreshingChunkExecutor  ChunkExecutor + refresh scheduling, splice / reject rules and ack bookkeeping
    required_lead()      the splice-timing simulation behind the --lead choice

Why two connections: WebsocketPolicyServer handles requests synchronously inside its event loop, so the server
serializes everything anyway; a socket dropped on the refresh path (timeout close can stall ~1 s) must never block
the blocking chunk-start plan, which has its own connection. Why acks: the server keeps traj_base = exactly what
the robot executes (the chunk-start plan plus every APPLIED refresh) and pins positions < k_pin to it, so every
refresh reply is acknowledged (applied from s / rejected / dropped) in the next refresh or plan.
"""

import collections
import hashlib
import itertools
import json
import logging
import math
import queue
import threading
import time
import uuid

import numpy as np
from PIL import Image

from chunk_executor import ChunkExecutor, PolicyError, _Chunk
from policy_client import PolicyClient, packb, resize_with_pad, unpackb

EPS = 1e-9
PREPARED_SIZE = 224
KNOWN_SENSORS = ("left_gripper_inner", "left_gripper_outer", "right_gripper_inner", "right_gripper_outer")
TACTILE_FIELD = ("rectify", "rgb")  # TactileStream field/order: the publisher's true-RGB rectify image
RESIZE_MODES = ("pad", "pad_pil", "stretch_pil")
LIVE_CODECS = ("none", "yuv420")
ENCODERS = ("dinov2", "sparsh")
TACTILE_INPUTS = ("raw", "diff")
MODES = ("chunk", "refresh")
ARMS = ("both", "left", "right")
STATE_GRIPPERS = ("realenv", "vtla")
DEFAULT_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ARM_JOINTS = {"both": (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12), "left": (0, 1, 2, 3, 4, 5),
              "right": (7, 8, 9, 10, 11, 12)}
ARM_GRIPPERS = {"both": (6, 13), "left": (6,), "right": (13,)}
# board_insertion raw metadata.json "gripper_calibration" ([left, right] follower motor angle endpoints, rad): the
# VTLA state gripper is (angle - closed) / (open - closed) with these endpoints.
DATASET_GRIPPER_CALIBRATION = ({"closed": -1.5, "open": 0.0767}, {"closed": -1.5, "open": 0.0782})
DEFAULT_MAX_SPLICE_JUMP = 0.075  # rad, about the per-30-Hz-step p99 of the board_insertion demonstrations
DEFAULT_MAX_PLAN_STEP = 0.2  # rad, at least the demonstrations' largest per-step change
DEFAULT_MAX_GRIPPER_JUMP = 0.25
DEFAULT_MAX_TACTILE_AGE_S = 0.2  # tactile_stream.DEFAULT_MAX_TACTILE_AGE_S
HISTORY_MARGIN_STEPS = 16  # steps kept beyond the widest needed_steps() span (anchor lag + slack)
# A refresh reply's positions < k_pin come from the server's traj_base through elementwise output transforms, so they
# equal the executed plan bit for bit; anything above float noise means the two plans drifted apart.
PIN_TOLERANCE = 1e-5
LATE_FLAG_FRAC = 0.05  # robot review finding 2: flag trials (and the session) with more than 5 % late refreshes

# Refresh outcomes (robot review findings 3 / exp 1, intention to treat):
# - INFRA_REFRESH_STATUSES are real infrastructure faults: a transport or protocol failure ("failed"), a server error
#   reply ("error"), stale tactile data ("skipped-stale") and the slots skipped because one of those disabled the
#   rest of the chunk ("skipped-disabled"; a chunk is disabled only by a failure, an error reply or a stopped worker).
#   Malformed replies (no actions, error or K = 0; undecodable actions) keep the status "rejected" but are counted as
#   "malformed", an infrastructure fault too. Only these can make a trial an infrastructure_failure (re-run).
# - PROTOCOL_REFRESH_STATUSES (and "late": applied from s > k_pin) are outcomes of the method and of its compute time:
#   the splice guard rejecting a large plan change, deadline / past-end / stale-chunk drops, a slot skipped while the
#   previous refresh was still in flight. They grow with eta, K and contact, so they are recorded per trial as
#   protocol flags and never exclude or re-run a trial (that would censor the trials guidance changes most).
# "skipped-K0" (dynamic K chose not to refresh) is the policy's own decision and counts in neither.
INFRA_REFRESH_STATUSES = ("failed", "error", "skipped-stale", "skipped-disabled")
PROTOCOL_REFRESH_STATUSES = ("rejected", "dropped", "skipped-in-flight")
BAD_REFRESH_STATUSES = PROTOCOL_REFRESH_STATUSES + INFRA_REFRESH_STATUSES  # bad_frac: every lost or late slot


# --------------------------------------------------------------------------------------------
# Logging (blinded console)
# --------------------------------------------------------------------------------------------

def console_text(text):
    """logging extra for a record that could tell the blinded operator a trial's condition (the refresh connection
    or worker, server text, a traceback): run_forceguide_policy's console shows the neutral `text` in its place, and
    text=None (FILE_ONLY) keeps the record in the session's client.log only. Without the runner's handlers (tests,
    other scripts) such records log as usual."""
    return {"console_text": text}


FILE_ONLY = console_text(None)
# The client's DEBUG records go through this logger: the runner sets it to DEBUG for client.log and keeps the root
# logger at INFO, so the libraries' DEBUG output (websockets logs every frame) is never produced.
DEBUG_LOG = logging.getLogger("forceguide")


# --------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------

class InfrastructureFailure(PolicyError):
    """Server, network or sensor trouble rather than a policy decision: the trial is re-run, not counted."""


class StaleTactile(InfrastructureFailure):
    """A tactile sensor stalled (data too old or seq frozen), or a needed step is missing from the history."""


class ServerChanged(InfrastructureFailure):
    """After a reconnect the server instance or a condition fingerprint differs from the session's."""


class ConnectFailed(InfrastructureFailure):
    """No connection within the budget."""


class ConnectionLost(InfrastructureFailure):
    """The websocket closed under us (server crash, network, or the server's error path)."""


class ServerError(InfrastructureFailure):
    """The server replied with a traceback string (it closes the connection right after)."""


class ProtocolError(InfrastructureFailure):
    """A reply that does not answer the request (wrong req, missing fields, refused reset)."""


# --------------------------------------------------------------------------------------------
# Metadata: the server's "forceguide" block
# --------------------------------------------------------------------------------------------

class TactileSpec:
    """The serving-side tactile spec of one condition (cpm.json contact.tactile), as far as the client needs it.

    The client prepares every frame itself (live_codec, rotate, resize to 224), so these fields must match what the
    CPM was trained on; the encoder/model/grid fields are only logged. Defaults are features.TactileSpec's (resize
    "pad" included: a spec without a resize field means the legacy letterbox, which the client refuses, rather than
    a silently different preparation).
    """

    def __init__(self, keys, encoder="dinov2", model="", grid=4, input="raw", resize="pad", rotate=0,
                 gap_s=0.0, live_codec="none"):
        keys = tuple(str(key) for key in keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("a tactile spec needs distinct keys, got {}".format(keys))
        if encoder not in ENCODERS:
            raise ValueError("unknown tactile encoder {!r}".format(encoder))
        if input not in TACTILE_INPUTS:
            raise ValueError("unknown tactile input {!r}".format(input))
        if resize not in RESIZE_MODES:
            raise ValueError("unknown tactile resize {!r}".format(resize))
        if live_codec not in LIVE_CODECS:
            raise ValueError("unknown live codec {!r}".format(live_codec))
        self.keys = keys
        self.encoder = encoder
        self.model = str(model or "")
        self.grid = int(grid)
        self.input = input
        self.resize = resize
        self.rotate = int(rotate) % 4
        self.gap_s = float(gap_s or 0.0)
        self.live_codec = live_codec

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return None
        if not isinstance(data, dict) or not data.get("keys"):
            raise ValueError("tactile spec must be a dict with 'keys', got {!r}".format(data))
        fields = ("encoder", "model", "grid", "input", "resize", "rotate", "gap_s", "live_codec")
        return cls(data["keys"], **{name: data[name] for name in fields if data.get(name) is not None})

    def to_dict(self):
        return {"keys": list(self.keys), "encoder": self.encoder, "model": self.model, "grid": self.grid,
                "input": self.input, "resize": self.resize, "rotate": self.rotate, "gap_s": self.gap_s,
                "live_codec": self.live_codec}

    @property
    def needs_background(self):
        """diff inputs and Sparsh subtract the episode's step-0 frame, so step 0 must reach the server."""
        return self.input == "diff" or self.encoder == "sparsh"

    def __eq__(self, other):
        return isinstance(other, TactileSpec) and self.to_dict() == other.to_dict()

    def __repr__(self):
        return "TactileSpec({})".format(self.to_dict())


class ConditionInfo:
    """One served condition: mode, tactile spec, window and cadence (metadata conditions[name])."""

    def __init__(self, name, mode="chunk", tactile=None, window=1, stride=1, gap_frames=0, refresh_every=None,
                 eta=None, fingerprint=None, raw=None, lead=None, min_lead=None):
        if mode not in MODES:
            raise ValueError("condition {!r}: unknown mode {!r}".format(name, mode))
        self.name = str(name)
        self.mode = mode
        self.tactile = tactile
        self.window = int(window)
        self.stride = int(stride)
        self.gap_frames = int(gap_frames)
        self.refresh_every = None if refresh_every is None else int(refresh_every)
        self.eta = eta
        self.fingerprint = fingerprint
        self.raw = dict(raw or {})
        # Refresh lead (k_pin - k): `lead` is the lead the condition's eta was calibrated with (calibration.json via
        # offline_guidance_check --lead L; the server refuses refreshes with any other lead), `min_lead` the smallest
        # lead its dynamic-K table covers (dynamic_k.json). None: no constraint (old servers send neither).
        self.lead = None if lead is None else int(lead)
        self.min_lead = None if min_lead is None else int(min_lead)
        if self.window < 1 or self.stride < 1 or self.gap_frames < 0:
            raise ValueError("condition {!r}: need window >= 1, stride >= 1, gap_frames >= 0".format(name))
        if mode == "refresh" and (self.refresh_every is None or self.refresh_every < 1):
            raise ValueError("condition {!r}: refresh mode needs refresh_every >= 1".format(name))
        if (self.lead is not None and self.lead < 1) or (self.min_lead is not None and self.min_lead < 1):
            raise ValueError("condition {!r}: lead and min_lead must be >= 1".format(name))

    @classmethod
    def from_dict(cls, name, data, policy_hz=None):
        data = dict(data or {})
        tactile_raw = data.get("tactile", data.get("tactile_spec"))
        spec = TactileSpec.from_dict(tactile_raw) if tactile_raw else None
        nested = tactile_raw if isinstance(tactile_raw, dict) else {}

        def pick(key, default):
            value = data.get(key)
            if value is None:
                value = nested.get(key)
            return default if value is None else value

        gap = pick("gap_frames", None)
        if gap is None:
            gap = (int(round(spec.gap_s * float(policy_hz)))
                   if spec is not None and spec.encoder == "sparsh" and policy_hz else 0)
        return cls(name, mode=data.get("mode") or "chunk", tactile=spec, window=pick("window", 1),
                   stride=pick("stride", 1), gap_frames=gap, refresh_every=data.get("refresh_every"),
                   eta=data.get("eta"), fingerprint=data.get("fingerprint"), raw=data, lead=data.get("lead"),
                   min_lead=data.get("min_lead"))

    @property
    def gap(self):
        """Frames between a Sparsh frame and its previous frame (0 for DINOv2 inputs, and for the gap-0 control)."""
        return self.gap_frames if self.tactile is not None and self.tactile.encoder == "sparsh" else 0

    @property
    def span(self):
        """How far back from its anchor step needed_steps() reaches (step 0 aside)."""
        return (self.window - 1) * self.stride + self.gap


def fingerprint_of(metadata):
    """(server_instance_id, sorted (condition, fingerprint) pairs): what must not change within a session."""
    block = (metadata or {}).get("forceguide") or {}
    conditions = block.get("conditions") or {}
    return (block.get("server_instance_id"),
            tuple(sorted((str(name), str((entry or {}).get("fingerprint"))) for name, entry in conditions.items())))


class ServerInfo:
    """The metadata "forceguide" block: the client's single source of configuration (P0 facts + conditions)."""

    def __init__(self, raw, version, model_action_dim, action_horizon, policy_hz, arm, cameras, image_bgr,
                 state_gripper, server_instance_id, git_rev, conditions):
        self.raw = raw
        self.version = version
        self.model_action_dim = int(model_action_dim)
        self.action_horizon = int(action_horizon)
        self.policy_hz = float(policy_hz)
        self.arm = arm
        self.cameras = tuple(cameras)
        self.image_bgr = bool(image_bgr)
        self.state_gripper = state_gripper
        self.server_instance_id = server_instance_id
        self.git_rev = git_rev
        self.conditions = conditions

    @classmethod
    def from_metadata(cls, metadata):
        block = (metadata or {}).get("forceguide")
        if not isinstance(block, dict):
            raise RuntimeError("The server sent no 'forceguide' metadata block - is it serve_forceguide.py? "
                               "Metadata: {}".format(metadata))
        missing = [key for key in ("model_action_dim", "action_horizon", "policy_hz", "conditions", "cameras")
                   if block.get(key) is None]
        if missing:
            raise RuntimeError("forceguide metadata lacks {}".format(missing))
        arm = block.get("arm") or "both"
        if arm not in ARMS:
            raise RuntimeError("forceguide metadata: unknown arm {!r}".format(arm))
        state_gripper = block.get("state_gripper") or "realenv"
        if state_gripper not in STATE_GRIPPERS:
            raise RuntimeError("forceguide metadata: unknown state_gripper {!r}".format(state_gripper))
        policy_hz = float(block["policy_hz"])
        conditions = {str(name): ConditionInfo.from_dict(name, entry, policy_hz)
                      for name, entry in dict(block["conditions"]).items()}
        if not conditions:
            raise RuntimeError("forceguide metadata lists no conditions")
        return cls(block, block.get("version"), block["model_action_dim"], block["action_horizon"], policy_hz, arm,
                   block["cameras"], block.get("image_bgr", False), state_gripper,
                   block.get("server_instance_id"), block.get("git_rev"), conditions)

    def fingerprint(self):
        return fingerprint_of({"forceguide": self.raw})

    def condition(self, name):
        if name not in self.conditions:
            raise KeyError("condition {!r} is not served (served: {})".format(name, sorted(self.conditions)))
        return self.conditions[name]


def sensor_for_key(key, overrides=None):
    """Tactile topic name for a dataset feature key: the converter names tactile keys by sensor, so the key's last
    dotted component (observation.images.right_gripper_outer -> right_gripper_outer) is the sensor."""
    if overrides and key in overrides:
        return overrides[key]
    sensor = str(key).rsplit(".", 1)[-1]
    if sensor not in KNOWN_SENSORS:
        raise ValueError("cannot tell which tactile sensor feeds {!r}; pass --tactile_sensor {}=<sensor> "
                         "(sensors: {})".format(key, key, ", ".join(KNOWN_SENSORS)))
    return sensor


# --------------------------------------------------------------------------------------------
# Tactile preparation: copies of the forceguide repo's features.py (to_uint8_hwc, prepare_tactile for pad_pil /
# stretch_pil, apply_live_codec, window_offsets, needed_steps). The server encodes what the client prepares, so these
# must give the same bits as training's precompute; tests/test_forceguide_client.py compares them with a golden file
# written by F's own functions. Keep them line-for-line in sync with features.py.
# --------------------------------------------------------------------------------------------

def to_uint8_hwc(image):
    """An image as uint8 [..., H, W, 3]; accepts float in [0, 1], channel-first (LeRobot) and single-channel input."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * np.clip(image, 0, 1)).round().astype(np.uint8)
    if image.shape[-1] not in (1, 3) and image.shape[-3] in (1, 3):
        image = np.moveaxis(image, -3, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image


def resize_stretch_pil(images, height=PREPARED_SIZE, width=PREPARED_SIZE):
    """PIL BILINEAR resize of [..., H, W, 3] uint8 images to (height, width), aspect ratio not kept: what
    torchvision's Resize((224, 224)) does to a PIL image in Sparsh's pipeline (features._resize_stretch_pil)."""
    images = np.asarray(images)
    flat = images.reshape(-1, *images.shape[-3:])
    resized = np.stack([np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR)) for image in flat])
    return resized.reshape(*images.shape[:-3], *resized.shape[1:])


def prepare_tactile(image, spec, allow_legacy_pad=False):
    """Tactile frames [..., H, W, C] -> uint8 [..., 224, 224, 3] per the spec (features.prepare_tactile).

    Rotation (axes -3, -2, so batch axes are left alone) comes before resizing. "pad_pil" is openpi_client's
    resize_with_pad (policy_client.resize_with_pad is a copy), "stretch_pil" a PIL bilinear resize. "pad" is the
    legacy JAX letterbox of old NeoSim CPMs, which cannot be reproduced bit-exactly without JAX: refused unless
    allow_legacy_pad (then the PIL letterbox, at most a grey level away, stands in). Colour order is never changed.
    """
    image = to_uint8_hwc(image)
    if image.ndim < 3 or image.shape[-1] != 3:
        raise ValueError("expected images [..., H, W, 3], got shape {}".format(image.shape))
    if spec.resize == "pad" and not allow_legacy_pad:
        raise ValueError("tactile resize 'pad' (legacy JAX letterbox) cannot be reproduced on the robot client; "
                         "train the CPM with pad_pil or stretch_pil")
    if spec.rotate:
        image = np.rot90(image, spec.rotate, axes=(-3, -2))
    image = np.ascontiguousarray(image)
    if spec.resize in ("pad_pil", "pad"):
        return resize_with_pad(image, PREPARED_SIZE, PREPARED_SIZE)
    if spec.resize == "stretch_pil":
        return resize_stretch_pil(image)
    raise ValueError("unknown tactile resize {!r}".format(spec.resize))


def apply_live_codec(native_rgb, codec):
    """Live native RGB frames uint8 [..., H, W, 3] through the codec loss the dataset frames carry
    (features.apply_live_codec).

    The recorder stored tactile video as H.264 yuv420p, so dataset frames have half-resolution chroma; raw live frames
    do not. "yuv420" is OpenCV's RGB -> YUV_I420 -> RGB round trip at native resolution (even height and width).
    Live frames only, before prepare_tactile.
    """
    images = np.asarray(native_rgb)
    if codec == "none":
        return images
    if codec != "yuv420":
        raise ValueError("live_codec must be one of {}, got {!r}".format(LIVE_CODECS, codec))
    if images.dtype != np.uint8 or images.ndim < 3 or images.shape[-1] != 3:
        raise ValueError("expected uint8 frames [..., H, W, 3], got {} {}".format(images.dtype, images.shape))
    height, width = images.shape[-3:-1]
    if height % 2 or width % 2:
        raise ValueError("yuv420 needs even height and width, got {} x {}".format(height, width))
    import cv2

    flat = images.reshape(-1, height, width, 3)
    out = np.stack([cv2.cvtColor(cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2YUV_I420),
                                 cv2.COLOR_YUV2RGB_I420) for image in flat])
    return out.reshape(images.shape)


def prepare_live_tactile(native_rgb, spec, allow_legacy_pad=False):
    """Live tactile frame(s) [..., H, W, 3] uint8 RGB -> what the server encodes, [..., 224, 224, 3] uint8:
    apply_live_codec at native resolution (the dataset frames carry the codec at native resolution), then
    prepare_tactile (rotate, resize)."""
    return prepare_tactile(apply_live_codec(native_rgb, spec.live_codec), spec, allow_legacy_pad=allow_legacy_pad)


# --------------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------------

def window_offsets(window, stride):
    """Step offsets of a causal tactile window, oldest first: [-(W - 1) * stride, ..., -stride, 0] (train_cpm's rows;
    features.window_offsets, as a list)."""
    if window < 1 or stride < 1:
        raise ValueError("window {} and stride {} must be >= 1".format(window, stride))
    return [int(stride) * offset for offset in range(-(int(window) - 1), 1)]


def needed_steps(step, window=1, stride=1, gap=0, *, background=False):
    """The episode steps whose tactile frames a query at `step` reads, sorted and unique (features.needed_steps):
    the window entries max(step + o, 0), each entry's Sparsh previous frame max(u - gap, 0), and step 0 when the
    encoder needs the background (diff input, Sparsh). The server looks them up exactly."""
    if gap < 0:
        raise ValueError("gap must be >= 0, got {}".format(gap))
    steps = {max(int(step) + int(offset), 0) for offset in window_offsets(window, stride)}
    if gap:
        steps |= {max(u - int(gap), 0) for u in steps}
    if background:
        steps.add(0)
    return sorted(steps)


def global_step(position):
    """Policy step of a policy-step clock position (the same rounding _Chunk.sample uses)."""
    return int(math.floor(position + EPS))


def chunk_k(offset):
    return int(math.floor(max(offset, 0.0) + EPS))


def max_index_read(offset, length):
    """Highest chunk index _Chunk.sample reads at this offset: it interpolates a[low] and a[low + 1] at fractional
    offsets, so with a 30 Hz policy in the 50 Hz loop a command already depends on the NEXT step."""
    offset = max(offset, 0.0)
    low = int(math.floor(offset + EPS))
    if offset - low < EPS or low + 1 >= length:
        return min(low, length - 1)
    return low + 1


def sample_actions(actions, offset, length):
    """_Chunk.sample at a chunk offset, for an arbitrary action array."""
    offset = max(offset, 0.0)
    low = int(math.floor(offset + EPS))
    frac = offset - low
    if frac < EPS or low + 1 >= length:
        return np.asarray(actions[min(low, length - 1)], dtype=np.float64).copy()
    return (1.0 - frac) * actions[low] + frac * actions[low + 1]


def ticks_per_chunk(length, advance):
    """Control ticks one chunk of `length` playable steps lasts (offsets 0, a, 2a, ... <= length - 1)."""
    return int(math.floor((length - 1) / advance + EPS)) + 1


# --------------------------------------------------------------------------------------------
# Adapter: observation in, 14-dim follower commands out
# --------------------------------------------------------------------------------------------

class ForceGuideAdapter:
    """The P0's observation and action conventions, from the server metadata (identical for all conditions).

    cameras: the P0's cameras only (a single-arm P0 trained with a black, masked-out second wrist image, so that
        camera must not be sent), letterboxed to 224 (openpi_client resize_with_pad) and sent CHW uint8; the
        board_insertion LeRobot images are cv2 BGR arrays, so image_bgr swaps the live RGB frames first.
    state: "realenv" = RealEnv qpos (gripper = normalized finger position); "vtla" = FollowerCommander.measured()
        read in the same tick (gripper = follower motor angle normalized by the calibrated endpoints, what the VTLA
        recorder stored). float32 like the dataset.
    actions: a single-arm reply [H, 7] becomes [H, 14] with the idle arm at hold_pose (the homing target) and its
        gripper commanded 1.0 (open). Never a qpos gripper value: FollowerCommander reads the gripper field as a
        normalized JOINT command, and qpos's open finger position (~0.3) would close it to ~70 % of its stroke.
    """

    name = "forceguide"

    def __init__(self, cameras=DEFAULT_CAMERAS, arm="both", image_bgr=False, state_gripper="realenv", prompt=None,
                 resolution=PREPARED_SIZE, hold_pose=None, action_horizon=50, model_action_dim=32):
        if arm not in ARMS:
            raise ValueError("arm must be one of {}".format(ARMS))
        if state_gripper not in STATE_GRIPPERS:
            raise ValueError("state_gripper must be one of {}".format(STATE_GRIPPERS))
        if arm != "both" and (hold_pose is None or len(hold_pose) < 6):
            raise ValueError("a single-arm P0 needs the idle arm's hold pose (6 joints)")
        self.cameras = tuple(cameras)
        self.arm = arm
        self.image_bgr = bool(image_bgr)
        self.state_gripper = state_gripper
        self.prompt = prompt
        self.resolution = int(resolution)
        self.hold_pose = None if hold_pose is None else np.asarray(hold_pose[:6], dtype=np.float64)
        self.action_horizon = int(action_horizon)
        self.model_action_dim = int(model_action_dim)
        self.out_dim = 14 if arm == "both" else 7

    @classmethod
    def from_info(cls, info, prompt=None, hold_pose=None, resolution=PREPARED_SIZE):
        return cls(cameras=info.cameras, arm=info.arm, image_bgr=info.image_bgr, state_gripper=info.state_gripper,
                   prompt=prompt, resolution=resolution, hold_pose=hold_pose, action_horizon=info.action_horizon,
                   model_action_dim=info.model_action_dim)

    @property
    def active_joints(self):
        return ARM_JOINTS[self.arm]

    @property
    def active_grippers(self):
        return ARM_GRIPPERS[self.arm]

    def state(self, frame, measured=None):
        if self.state_gripper == "vtla":
            if measured is None:
                raise RuntimeError("state_gripper 'vtla' needs FollowerCommander.measured() from the same tick")
            full = np.asarray(measured, dtype=np.float64)
        else:
            full = np.asarray(frame.qpos, dtype=np.float64)
        if full.shape != (14,) or not np.isfinite(full).all():
            raise RuntimeError("robot state must be 14 finite values, got {}".format(full))
        if self.arm == "left":
            full = full[0:7]
        elif self.arm == "right":
            full = full[7:14]
        return full.astype(np.float32)

    def images(self, frame):
        images = {}
        for camera in self.cameras:
            image = frame.images.get(camera)
            if image is None:
                raise RuntimeError("camera {} has not published yet".format(camera))
            image = np.asarray(image)
            if self.image_bgr:
                image = image[..., ::-1]
            image = resize_with_pad(np.ascontiguousarray(image), self.resolution, self.resolution)
            images[camera] = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
        return images

    def prepare(self, frame, measured=None):
        obs = {"state": self.state(frame, measured), "images": self.images(frame)}
        if self.prompt is not None:
            obs["prompt"] = self.prompt  # otherwise the server injects the training config's default prompt
        return obs

    def decode(self, actions):
        """[H, A_out] server actions -> [H, 14] follower commands (float64)."""
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != self.out_dim or len(actions) == 0:
            raise RuntimeError("expected [H, {}] actions for arm={}, got {}".format(self.out_dim, self.arm,
                                                                                   actions.shape))
        if self.arm == "both":
            return actions.copy()
        out = np.empty((len(actions), 14), dtype=np.float64)
        active = slice(0, 7) if self.arm == "left" else slice(7, 14)
        idle = 7 if self.arm == "left" else 0
        out[:, active] = actions
        out[:, idle:idle + 6] = self.hold_pose
        out[:, idle + 6] = 1.0  # idle gripper open
        return out


def gripper_calibration_from_metadata(metadata):
    """[left, right] {"closed", "open"} from a raw episode metadata.json dict or the forceguide metadata block."""
    value = (metadata or {}).get("gripper_calibration")
    if value is None:
        return None
    if isinstance(value, dict):
        value = [value["left"], value["right"]]
    if len(value) != 2:
        raise ValueError("gripper_calibration must list [left, right], got {!r}".format(value))
    return tuple({"closed": float(arm["closed"]), "open": float(arm["open"])} for arm in value)


def check_gripper_calibration(expected=None, actual=None, atol=1e-6):
    """Raise unless the follower gripper endpoints FollowerCommander.measured() normalizes with (constants.py)
    equal the ones the VTLA training data was normalized with, so live state grippers mean what training's did."""
    expected = tuple(expected or DATASET_GRIPPER_CALIBRATION)
    if actual is None:
        from constants import (PUPPET_LEFT_GRIPPER_JOINT_CLOSE, PUPPET_LEFT_GRIPPER_JOINT_OPEN,
                               PUPPET_RIGHT_GRIPPER_JOINT_CLOSE, PUPPET_RIGHT_GRIPPER_JOINT_OPEN)
        actual = ({"closed": PUPPET_LEFT_GRIPPER_JOINT_CLOSE, "open": PUPPET_LEFT_GRIPPER_JOINT_OPEN},
                  {"closed": PUPPET_RIGHT_GRIPPER_JOINT_CLOSE, "open": PUPPET_RIGHT_GRIPPER_JOINT_OPEN})
    for side, want, have in zip(("left", "right"), expected, actual):
        for key in ("closed", "open"):
            if abs(float(want[key]) - float(have[key])) > atol:
                raise RuntimeError("{} follower gripper {} endpoint is {} in constants.py but {} in the training data; "
                                   "VTLA state grippers would not match training".format(side, key, have[key],
                                                                                        want[key]))
    return actual


# --------------------------------------------------------------------------------------------
# Tactile history, staleness, payload
# --------------------------------------------------------------------------------------------

class _TactileEntry:
    __slots__ = ("image", "seq", "read_end", "age", "captured")

    def __init__(self, image, seq, read_end, age, captured):
        self.image = image  # native RGB array, a reference into the TactileStream frame (read-only)
        self.seq = seq
        self.read_end = read_end
        self.age = age  # data age when captured (s)
        self.captured = captured  # host monotonic capture time


def _entry_from_frame(frame, now_wall, now_mono):
    if frame is None:
        return None
    try:
        image = frame.field(*TACTILE_FIELD)
    except Exception:  # noqa: BLE001 - a frame that cannot be decoded counts as missing
        return None
    if image is None:
        return None
    read_end = float(frame.read_end)
    return _TactileEntry(image, int(frame.seq), read_end, now_wall - read_end, now_mono)


class TactileHistory:
    """Exact-step tactile frames for one episode, per sensor.

    record(position, snapshot) stores, for every not yet recorded global step g <= floor(position), the frames of
    this tick: history[g] = the frame captured at the FIRST tick whose policy-step clock position >= g, taken before
    that tick's command is sent (the recorder sampled dataset row g just before action g). A step is never
    overwritten, so the chunk-start frame stays exactly the one the plan sent. A ring of `capacity` steps (plus
    step 0, the background) bounds memory: native frames are ~0.84 MB each.

    prepared(step, sensor) applies the episode's TactileSpec once per (step, sensor) and caches the 224 frame.
    Thread-safe: the control loop records, the refresh worker prepares.
    """

    def __init__(self, sensors, capacity, wall_clock=time.time, mono_clock=time.monotonic):
        if int(capacity) < 1:
            raise ValueError("history capacity must be >= 1")
        self.sensors = tuple(sensors)
        self.capacity = int(capacity)
        self._wall = wall_clock
        self._mono = mono_clock
        self._lock = threading.Lock()
        self._entries = {}
        self._prepared = {}
        self._next = 0
        self._spec = None
        self._allow_legacy_pad = False

    def reset(self, spec=None, allow_legacy_pad=False):
        if spec is not None and spec.resize == "pad" and not allow_legacy_pad:
            raise ValueError("tactile resize 'pad' (legacy JAX letterbox) cannot be reproduced on the client")
        with self._lock:
            self._entries = {}
            self._prepared = {}
            self._next = 0
            self._spec = spec
            self._allow_legacy_pad = allow_legacy_pad

    @property
    def spec(self):
        return self._spec

    @property
    def next_step(self):
        with self._lock:
            return self._next

    def steps(self):
        with self._lock:
            return sorted(self._entries)

    def record(self, position, snapshot):
        """Store this tick's frames for every step first reached now; returns those steps."""
        newest = global_step(position)
        with self._lock:
            first = self._next
        if newest < first:
            return []
        now_wall, now_mono = self._wall(), self._mono()
        entries = {sensor: _entry_from_frame((snapshot or {}).get(sensor), now_wall, now_mono)
                   for sensor in self.sensors}
        steps = list(range(first, newest + 1))
        with self._lock:
            for step in steps:
                self._entries[step] = entries
            self._next = newest + 1
            oldest = newest - self.capacity + 1
            for step in [step for step in self._entries if 0 < step < oldest]:
                del self._entries[step]
                for sensor in self.sensors:
                    self._prepared.pop((step, sensor), None)
        return steps

    def entry(self, step, sensor):
        with self._lock:
            entries = self._entries.get(step)
        if entries is None:
            raise KeyError("step {} is not in the tactile history".format(step))
        entry = entries.get(sensor)
        if entry is None:
            raise KeyError("no {} frame was available for step {}".format(sensor, step))
        return entry

    def fresh(self, steps, sensors, max_age):
        """(ok, reason): every (step, sensor) frame exists and was at most max_age old when captured."""
        for step in steps:
            for sensor in sensors:
                try:
                    entry = self.entry(step, sensor)
                except KeyError as exc:
                    return False, str(exc.args[0])
                if max_age is not None and entry.age > max_age:
                    return False, "{} frame for step {} was {:.3f} s old when captured".format(sensor, step,
                                                                                            entry.age)
        return True, None

    def prepared(self, step, sensor):
        key = (step, sensor)
        with self._lock:
            cached = self._prepared.get(key)
            spec, allow = self._spec, self._allow_legacy_pad
        if cached is not None:
            return cached
        if spec is None:
            raise RuntimeError("TactileHistory.reset(spec) must set this episode's tactile spec first")
        image = prepare_live_tactile(self.entry(step, sensor).image, spec, allow_legacy_pad=allow)
        with self._lock:
            if step in self._entries:
                image = self._prepared.setdefault(key, image)
        return image


class TactileMonitor:
    """Per-sensor staleness: data age (now - the publisher's read time) and how long seq has not changed. A stalled
    sensor keeps handing out its last frame with a low receipt age, so both are checked."""

    def __init__(self, sensors, max_age_s=DEFAULT_MAX_TACTILE_AGE_S, wall_clock=time.time, mono_clock=time.monotonic):
        self.sensors = tuple(sensors)
        self.max_age_s = float(max_age_s)
        self._wall = wall_clock
        self._mono = mono_clock
        self._seq = {}
        self._changed = {}
        self.state = {sensor: None for sensor in self.sensors}

    def reset(self):
        self._seq = {}
        self._changed = {}
        self.state = {sensor: None for sensor in self.sensors}

    def update(self, snapshot):
        now_wall, now_mono = self._wall(), self._mono()
        for sensor in self.sensors:
            frame = (snapshot or {}).get(sensor)
            if frame is None:
                self.state[sensor] = None
                continue
            seq = int(frame.seq)
            if self._seq.get(sensor) != seq:
                self._seq[sensor] = seq
                self._changed[sensor] = now_mono
            self.state[sensor] = {"seq": seq, "age": now_wall - float(frame.read_end),
                                  "unchanged_s": now_mono - self._changed[sensor]}
        return self.state

    def stale(self, sensors=None):
        """Reasons, one per stale sensor (empty list: all fresh)."""
        reasons = []
        for sensor in self.sensors if sensors is None else sensors:
            state = self.state.get(sensor)
            if state is None:
                reasons.append("{}: no frame".format(sensor))
            elif state["age"] > self.max_age_s:
                reasons.append("{}: data {:.3f} s old".format(sensor, state["age"]))
            elif state["unchanged_s"] > self.max_age_s:
                reasons.append("{}: seq {} unchanged for {:.3f} s".format(sensor, state["seq"],
                                                                          state["unchanged_s"]))
        return reasons


class TactileRig:
    """The campaign's tactile streams, history and staleness monitor, ticked once per control tick."""

    def __init__(self, streams, capacity, max_age_s=DEFAULT_MAX_TACTILE_AGE_S, wall_clock=time.time,
                 mono_clock=time.monotonic):
        self.streams = dict(streams)
        self.sensors = tuple(sorted(self.streams))
        self.history = TactileHistory(self.sensors, capacity, wall_clock, mono_clock)
        self.monitor = TactileMonitor(self.sensors, max_age_s, wall_clock, mono_clock)

    def snapshot(self):
        return {sensor: stream.latest() for sensor, stream in self.streams.items()}

    def reset(self, spec=None, allow_legacy_pad=False):
        self.history.reset(spec, allow_legacy_pad)
        self.monitor.reset()

    def tick(self, position):
        """Capture before this tick's command; returns per-sensor {seq, age} for the tick log."""
        snapshot = self.snapshot()
        self.monitor.update(snapshot)
        self.history.record(position, snapshot)
        return {sensor: (None if state is None else {"seq": state["seq"], "age": state["age"]})
                for sensor, state in self.monitor.state.items()}

    def close(self):
        for stream in self.streams.values():
            release = getattr(stream, "release", None)
            if release is not None:
                try:
                    release()
                except Exception:  # noqa: BLE001
                    pass


class TactilePayload:
    """One trial's tactile upload bookkeeping: which steps an anchor needs and which the server has confirmed.

    A step counts as sent only once a reply confirmed it (not when a message was built): the plan (main thread)
    and a refresh (worker) may be in flight together, and a plan that left out a step only "sent" by a refresh
    that has not reached the server yet would fail with missing steps. Duplicates are harmless (first write wins
    on the server). Cleared on reset and after any refresh-connection failure.
    """

    def __init__(self, history, key_sensors, window=1, stride=1, gap=0, background=False, proxy=False):
        self.history = history
        self.key_sensors = collections.OrderedDict(key_sensors)
        self.window = int(window)
        self.stride = int(stride)
        self.gap = int(gap)
        self.background = bool(background)
        self.proxy = bool(proxy)  # condition without a tactile CPM: same upload, ignored by the server
        self._lock = threading.Lock()
        self._confirmed = set()

    @classmethod
    def empty(cls):
        return cls(None, {})

    @property
    def keys(self):
        return tuple(self.key_sensors)

    @property
    def sensors(self):
        return tuple(dict.fromkeys(self.key_sensors.values()))

    def needed(self, step):
        if not self.key_sensors:
            return []
        return needed_steps(step, self.window, self.stride, self.gap, background=self.background)

    def unconfirmed(self, steps):
        with self._lock:
            return [step for step in steps if step not in self._confirmed]

    def confirmed(self):
        with self._lock:
            return set(self._confirmed)

    def confirm(self, steps):
        with self._lock:
            self._confirmed.update(int(step) for step in steps)

    def forget(self, steps):
        with self._lock:
            self._confirmed.difference_update(int(step) for step in steps)

    def clear(self):
        with self._lock:
            self._confirmed.clear()

    def fresh(self, step, max_age):
        if not self.key_sensors:
            return True, None
        return self.history.fresh(self.needed(step), self.sensors, max_age)

    def build(self, step):
        """(steps, {key: uint8 [n, 224, 224, 3]}, prepare_ms) for the not yet confirmed needed steps of `step`."""
        start = time.perf_counter()
        steps = self.unconfirmed(self.needed(step))
        frames = {}
        for key, sensor in self.key_sensors.items():
            if steps:
                frames[key] = np.ascontiguousarray(np.stack([self.history.prepared(g, sensor) for g in steps]))
            else:
                frames[key] = np.zeros((0, PREPARED_SIZE, PREPARED_SIZE, 3), dtype=np.uint8)
        return [int(g) for g in steps], frames, 1000.0 * (time.perf_counter() - start)

    def describe(self, step):
        """For the request log: the anchor's needed steps with the host time each was captured and every sensor's
        data time (read_end), so the analysis can see the real time a window or a Sparsh pair spans (a window that
        reaches back over a blocking chunk-start pause compresses that pause into one step)."""
        steps = self.needed(step)
        if not steps:
            return None
        captured, read_end = [], {sensor: [] for sensor in self.sensors}
        for g in steps:
            first = None
            for sensor in self.sensors:
                try:
                    entry = self.history.entry(g, sensor)
                except KeyError:
                    entry = None
                read_end[sensor].append(None if entry is None else entry.read_end)
                if first is None and entry is not None:
                    first = entry.captured
            captured.append(first)
        return {"anchor": int(step), "steps": [int(g) for g in steps], "captured": captured, "read_end": read_end}


# --------------------------------------------------------------------------------------------
# Connections
# --------------------------------------------------------------------------------------------

def _close_quietly(ws):
    if ws is not None:
        try:
            ws.close()
        except Exception:  # noqa: BLE001
            pass


class ForceGuideConnection(PolicyClient):
    """PolicyClient for one serve_forceguide connection, made robust for a long multi-trial session.

    PolicyClient keeps a dead socket after the server's error path (traceback string, then close 1011), so every
    later request would fail; here any failure other than a reply timeout closes and clears the socket and the next
    request reconnects. Reconnecting takes a budget (the refresh path must give up within one refresh period, the
    plan path may wait 30 s), and after every reconnect the server fingerprint (instance id + condition
    fingerprints) is compared with the session's: a restarted or reconfigured server raises ServerChanged instead of
    silently mixing conditions. request() also returns the client time split (pack/send/wait) and payload size.
    """

    def __init__(self, host, port, connect_timeout_s=30.0, infer_timeout_s=None, close_timeout_s=1.0, name="main",
                 expected=None):
        self.close_timeout_s = float(close_timeout_s)
        self.name = name
        self.connects = 0
        self.expected = expected
        self.metadata = None
        super().__init__(host, port, connect_timeout_s=connect_timeout_s, infer_timeout_s=infer_timeout_s)

    def _connect(self, budget_s=None):
        import websockets.sync.client as ws_client
        from websockets.exceptions import WebSocketException

        budget = self.connect_timeout_s if budget_s is None else float(budget_s)
        deadline = time.monotonic() + budget
        last_error = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ws = None
            try:
                # Handshake and metadata both wait inside the budget: the server answers neither while its event loop
                # computes another request, and the refresh path must give up within one refresh period.
                ws = ws_client.connect(self.uri, compression=None, max_size=None, close_timeout=self.close_timeout_s,
                                       open_timeout=max(remaining, 0.05))
                metadata = unpackb(ws.recv(timeout=max(deadline - time.monotonic(), 0.05)))
            except (OSError, WebSocketException) as exc:  # refused, handshake/metadata timeout, closed
                _close_quietly(ws)
                last_error = exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                # The console shows the session's first connections (before any trial), without the connection's
                # name; a reconnect happens mid-session, and which connection waits tells a refresh condition.
                logging.info("%s: waiting for the policy server at %s (%s)", self.name, self.uri, exc,
                             extra=console_text(None if self.connects else "waiting for the policy server at {} "
                                                "({})".format(self.uri, exc)))
                time.sleep(min(2.0, remaining))
                continue
            self._ws = ws
            self.metadata = metadata
            self.connects += 1
            if self.expected is not None:
                found = fingerprint_of(metadata)
                if found != self.expected:
                    self.drop()
                    raise ServerChanged("{}: the server at {} changed during the session (expected {}, got {})".format(
                        self.name, self.uri, self.expected, found))
            return
        raise ConnectFailed("{}: could not connect to {} within {:.2f} s: {}".format(self.name, self.uri, budget,
                                                                                   last_error))

    def connected(self):
        return self._ws is not None

    def drop(self):
        ws, self._ws = self._ws, None
        _close_quietly(ws)

    def close(self):
        self.drop()

    def request(self, obs, timeout=None, connect_budget_s=None):
        """(reply, timing). Raises TimeoutError (socket dropped: the late reply dies with it), ServerError (str
        reply), ConnectionLost, ConnectFailed or ServerChanged."""
        from websockets.exceptions import ConnectionClosed

        timing = {"connect_ms": 0.0}
        if self._ws is None:
            start = time.perf_counter()
            self._connect(connect_budget_s)
            timing["connect_ms"] = 1000.0 * (time.perf_counter() - start)
        t0 = time.perf_counter()
        data = packb(obs)
        t1 = time.perf_counter()
        try:
            self._ws.send(data)
        except (ConnectionClosed, OSError) as exc:
            self.drop()
            raise ConnectionLost("{}: send failed: {}".format(self.name, exc)) from exc
        t2 = time.perf_counter()
        try:
            response = self._ws.recv(timeout=self.infer_timeout_s if timeout is None else timeout)
        except TimeoutError:
            self.drop()
            raise
        except (ConnectionClosed, OSError) as exc:
            self.drop()
            raise ConnectionLost("{}: connection closed: {}".format(self.name, exc)) from exc
        t3 = time.perf_counter()
        if isinstance(response, str):
            self.drop()
            raise ServerError("{}: policy server error:\n{}".format(self.name, response))
        reply = unpackb(response)
        t4 = time.perf_counter()
        timing.update(pack_ms=1000.0 * (t1 - t0), send_ms=1000.0 * (t2 - t1), wait_ms=1000.0 * (t3 - t2),
                      unpack_ms=1000.0 * (t4 - t3), bytes=len(data), reply_bytes=len(response))
        return reply, timing


# --------------------------------------------------------------------------------------------
# Refresh worker
# --------------------------------------------------------------------------------------------

class RefreshRequest:
    """One refresh (or observe) job, tagged so its reply is applied only to the round and chunk it was made for."""

    def __init__(self, kind="refresh", round_id=None, generation=None, chunk_id=None, server_chunk=None, k=None,
                 k_pin=None, tactile_step=0, tick_time=None, submit_time=None, ack=None, payload=None, timeout=None,
                 connect_budget_s=None):
        self.kind = kind
        self.round_id = round_id
        self.generation = generation
        self.chunk_id = chunk_id
        self.server_chunk = server_chunk
        self.k = k
        self.k_pin = k_pin
        self.tactile_step = int(tactile_step)
        self.tick_time = tick_time
        self.submit_time = submit_time
        self.ack = ack
        self.payload = payload
        self.timeout = timeout
        self.connect_budget_s = connect_budget_s
        self.req = None  # assigned when the worker builds the message
        self.steps = []  # tactile steps the message carried
        self.window = None  # the anchor's needed steps and their capture times (TactilePayload.describe)
        self.prepare_ms = None


class RefreshResult:
    """status: "ok" (reply holds the server's answer, which may itself be an error or K = 0), or a transport
    failure: "timeout", "connect", "connection", "server_error", "server_changed", "error"."""

    def __init__(self, request):
        self.request = request
        self.status = None
        self.reply = None
        self.error = None
        self.timing = {}
        self.finish_time = None
        self.reconnects = 0


class RefreshWorker:
    """Daemon thread that owns the refresh/observe connection.

    The control loop only hands over a job (references to frames, no packing): building the message (tactile
    preparation, msgpack) and the round trip happen here. One job at a time; `busy` stays set until the control loop
    has polled the result, so a new refresh never goes out before the previous reply was decided (its ack rides on
    the next message). Every exception is caught; a hung server never blocks shutdown (daemon thread, join timeout,
    and closing the socket interrupts a pending recv).
    """

    def __init__(self, connection, build_fn, timeout_s, after=None, clock=time.monotonic, connect_budget_s=None,
                 name="forceguide-refresh"):
        self._conn = connection
        self._build = build_fn
        self.timeout_s = float(timeout_s)
        self.connect_budget_s = connect_budget_s
        self._after = after
        self._clock = clock
        self._jobs = queue.Queue()
        self._results = queue.Queue()
        self._busy = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    @property
    def busy(self):
        return self._busy

    @property
    def alive(self):
        return self._thread.is_alive() and not self._stop.is_set()

    def submit(self, job):
        if self._busy or not self.alive:
            return False
        if job.submit_time is None:
            job.submit_time = self._clock()
        self._busy = True
        self._jobs.put(job)
        return True

    def poll(self):
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                break
        if results:
            self._busy = False
        return results

    def call(self, job, wait_s):
        """Synchronous round trip through the worker (warm-up, observe), only while no executor uses the worker:
        results of earlier jobs are discarded."""
        if self._busy:
            self.poll()
        if not self.submit(job):
            raise InfrastructureFailure("refresh worker is busy or stopped")
        deadline = time.monotonic() + wait_s
        while True:
            try:
                result = self._results.get(timeout=max(deadline - time.monotonic(), 0.0))
            except queue.Empty:
                raise InfrastructureFailure("no result from the refresh worker within {:.1f} s".format(
                    wait_s)) from None
            if result.request is job:
                self._busy = False
                return result

    def _run(self):
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                break
            result = RefreshResult(job)
            try:
                message = self._build(job)
                budget = job.connect_budget_s if job.connect_budget_s is not None else self.connect_budget_s
                reply, timing = self._conn.request(message, timeout=job.timeout or self.timeout_s,
                                                   connect_budget_s=budget)
                result.status, result.reply, result.timing = "ok", reply, timing
            except ServerChanged as exc:
                result.status, result.error = "server_changed", str(exc)
            except ConnectFailed as exc:
                result.status, result.error = "connect", str(exc)
            except ConnectionLost as exc:
                result.status, result.error = "connection", str(exc)
            except ServerError as exc:
                result.status, result.error = "server_error", str(exc)
            except TimeoutError as exc:
                result.status, result.error = "timeout", "no reply within {:.2f} s ({})".format(
                    job.timeout or self.timeout_s, exc)
            except Exception as exc:  # noqa: BLE001 - the worker must survive anything
                result.status, result.error = "error", "{}: {}".format(type(exc).__name__, exc)
            result.finish_time = self._clock()
            result.reconnects = getattr(self._conn, "connects", 0)
            if self._after is not None:
                try:
                    self._after(job, result)
                except Exception:  # noqa: BLE001
                    logging.exception("refresh worker: post-reply bookkeeping failed", extra=FILE_ONLY)
            self._results.put(result)

    def close(self, timeout=2.0):
        self._stop.set()
        self._jobs.put(None)
        self._thread.join(timeout)
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------------------------
# Session client
# --------------------------------------------------------------------------------------------

def _reply_block(reply):
    block = (reply or {}).get("forceguide")
    return block if isinstance(block, dict) else {}


def _int_list(values):
    if values is None:
        return []
    return [int(v) for v in np.asarray(values).reshape(-1)]


def _is_stale_chunk(error):
    return error is not None and str(error).strip().lower().startswith("stale chunk")


class ForceGuideClient:
    """One serve_forceguide session: the main connection (reset, plan) in the control thread and the refresh
    connection inside a RefreshWorker. Requests of both share one session id and one monotonic req counter."""

    def __init__(self, main, refresh=None, info=None, session_id=None, hard_timeout_s=10.0, refresh_budget_s=None,
                 clock=time.monotonic):
        self.main = main
        self.info = info or ServerInfo.from_metadata(main.metadata)
        self.session = session_id or "fg-{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:8])
        self.clock = clock
        self._req_lock = threading.Lock()
        self._req = 0
        fingerprint = self.info.fingerprint()
        main.expected = fingerprint
        self.worker = None
        if refresh is not None:
            refresh.expected = fingerprint
            if refresh.metadata is not None and fingerprint_of(refresh.metadata) != fingerprint:
                raise ServerChanged("the refresh connection reached a different server instance")
            self.worker = RefreshWorker(refresh, self.build_message, hard_timeout_s, after=self.after_reply,
                                        clock=clock, connect_budget_s=refresh_budget_s)

    def next_req(self):
        with self._req_lock:
            self._req += 1
            return self._req

    def _main_request(self, message, timeout):
        try:
            return self.main.request(message, timeout)
        except TimeoutError as exc:
            raise InfrastructureFailure("no {} reply within {} s".format(message.get("forceguide"), timeout)) from exc

    def reset(self, condition, policy_hz, timeout=None, trial=None, attempt=None):
        """trial (the schedule's trial_id) seeds the server's random-K and shuffle-history draws per decision, so a
        trial's draws do not depend on the session order (warm-up, earlier trials, re-runs) and a re-run replays
        them; attempt is logged only. Both are omitted when None (warm-up, preflight)."""
        req = self.next_req()
        message = {"forceguide": "reset", "session": self.session, "req": req, "condition": condition,
                   "policy_hz": float(policy_hz)}
        if trial is not None:
            message["trial"] = str(trial)
        if attempt is not None:
            message["attempt"] = int(attempt)
        reply, timing = self._main_request(message, timeout)
        block = _reply_block(reply)
        # The messages below never name the condition: they reach the console, and the operator is blinded.
        if block.get("error") is not None:
            raise InfrastructureFailure("reset failed: {}".format(block["error"]))
        if block.get("req") is not None and int(block["req"]) != req:
            raise ProtocolError("reset reply answers req {} instead of {}".format(block.get("req"), req))
        if not block.get("ok") or block.get("condition") != condition:
            raise ProtocolError("reset was not acknowledged (ok={!r}, same condition: {})".format(
                block.get("ok"), block.get("condition") == condition))
        expected = self.info.condition(condition).fingerprint
        if expected is not None and block.get("fingerprint") != expected:
            raise ServerChanged("the condition fingerprint is {} now, {} at the handshake".format(
                block.get("fingerprint"), expected))
        return {"type": "reset", "req": req, "condition": condition, "trial": message.get("trial"),
                "attempt": message.get("attempt"), "reply": block, "client": timing,
                "server_timing": reply.get("server_timing")}

    def plan(self, obs, plan_step, payload, noise, ack=None, timeout=None):
        """(reply, record) for one chunk-start plan; raises on any failure (the round is aborted)."""
        payload = payload if payload is not None else TactilePayload.empty()
        start = self.clock()
        req = self.next_req()
        steps, frames, prepare_ms = payload.build(plan_step)
        message = dict(obs)
        message.update({"forceguide": "plan", "session": self.session, "req": req,
                        "noise": np.asarray(noise, dtype=np.float32), "tactile_step": int(plan_step),
                        "tactile_steps": steps, "tactile_frames": frames})
        if ack is not None:
            message["ack"] = dict(ack)
        reply, timing = self._main_request(message, timeout)
        block = _reply_block(reply)
        if block.get("error") is not None:
            raise InfrastructureFailure("plan failed: {} (missing steps {})".format(block["error"],
                                                                                   block.get("missing_steps")))
        if block.get("req") is not None and int(block["req"]) != req:
            raise ProtocolError("plan reply answers req {} instead of {}".format(block.get("req"), req))
        if "actions" not in reply:
            raise ProtocolError("plan reply has no actions: {}".format(sorted(reply)))
        payload.confirm(steps)
        payload.forget(_int_list(block.get("missing_steps")))
        timing = dict(timing, prepare_ms=prepare_ms)
        record = {"type": "plan", "req": req, "plan_step": int(plan_step), "tactile_steps": steps,
                  "window": payload.describe(plan_step), "ack_sent": ack, "server_chunk": block.get("chunk"),
                  "K": block.get("K"), "version": block.get("version"), "timing_ms": block.get("timing_ms"),
                  "diag": block.get("diag"), "server_timing": reply.get("server_timing"), "client": timing,
                  "rtt_ms": 1000.0 * (self.clock() - start)}
        return reply, record

    def build_message(self, job):
        """Worker thread: the refresh/observe message for a job (assigns its req, prepares its tactile frames)."""
        job.req = self.next_req()
        payload = job.payload if job.payload is not None else TactilePayload.empty()
        steps, frames, prepare_ms = payload.build(job.tactile_step)
        job.steps, job.prepare_ms = steps, prepare_ms
        job.window = payload.describe(job.tactile_step)
        message = {"forceguide": job.kind, "session": self.session, "req": job.req,
                   "tactile_step": int(job.tactile_step), "tactile_steps": steps, "tactile_frames": frames}
        if job.kind == "refresh":
            message.update({"chunk": job.server_chunk, "k": int(job.k), "k_pin": int(job.k_pin)})
            if job.ack is not None:
                message["ack"] = dict(job.ack)
        return message

    def after_reply(self, job, result):
        """Worker thread: confirm the frames the server cached; resend everything after a connection failure."""
        payload = job.payload
        if payload is None:
            return
        if result.status != "ok":
            payload.clear()
            return
        block = _reply_block(result.reply)
        error = block.get("error")
        if error is None or _is_stale_chunk(error):  # a stale-chunk refresh's tactile is still cached
            payload.confirm(job.steps)
        payload.forget(_int_list(block.get("missing_steps")))

    def observe(self, payload, tactile_step, timeout):
        """Synchronous tactile-only upload (warm-up); returns the RefreshResult."""
        if self.worker is None:
            raise RuntimeError("observe needs the refresh connection")
        job = RefreshRequest(kind="observe", payload=payload, tactile_step=tactile_step, timeout=timeout)
        return self.worker.call(job, timeout + 5.0)

    def close(self, timeout=2.0):
        if self.worker is not None:
            self.worker.close(timeout)
        try:
            self.main.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------------------------
# Splice checks and the refreshing executor
# --------------------------------------------------------------------------------------------

class SpliceGuard:
    """Decides whether a refresh reply may replace the plan from chunk index s on. Only the active arm counts
    (the idle arm of a single-arm P0 is held identically in every reply).

    Rejected when: (a) wrong shape or non-finite; (b) the splice point jumps: max |new[s] - old[s]| over the arm
    joints > max_splice_jump, or, when the splice already changes this tick's command, that command vs the last
    command actually sent (after FollowerCommander's clamps); (c) a step > max_plan_step in the reply from s - 1 on
    (max_{i >= s-1} |new[i+1] - new[i]|) or in the executed transition old[s - 1] -> new[s]; (d) a gripper jumps more
    than max_gripper_jump at s. The largest change over the spliced suffix is only reported: it measures guidance
    strength, not a discontinuity.
    """

    def __init__(self, arm_joints, grippers, max_splice_jump=DEFAULT_MAX_SPLICE_JUMP,
                 max_plan_step=DEFAULT_MAX_PLAN_STEP, max_gripper_jump=DEFAULT_MAX_GRIPPER_JUMP):
        self.arm_joints = list(arm_joints)
        self.grippers = list(grippers)
        self.max_splice_jump = float(max_splice_jump)
        self.max_plan_step = float(max_plan_step)
        self.max_gripper_jump = float(max_gripper_jump)

    def check(self, old, new, s, length, last_sent=None, offset=None):
        old = np.asarray(old, dtype=np.float64)
        new = np.asarray(new, dtype=np.float64)
        if new.shape != old.shape:
            return False, "shape {} != plan {}".format(new.shape, old.shape), {}
        if not np.isfinite(new).all():
            return False, "non-finite actions", {}
        joints, grippers = self.arm_joints, self.grippers
        spliced = old.copy()
        spliced[s:] = new[s:]
        jump = float(np.abs(new[s, joints] - old[s, joints]).max()) if joints else 0.0
        gripper_jump = float(np.abs(new[s, grippers] - old[s, grippers]).max()) if grippers else 0.0
        max_step = 0.0
        if joints:
            for segment in (new[max(s - 1, 0):length][:, joints], spliced[max(s - 1, 0):s + 1][:, joints]):
                if len(segment) > 1:
                    max_step = max(max_step, float(np.abs(np.diff(segment, axis=0)).max()))
        suffix = np.abs(new[s:length][:, joints] - old[s:length][:, joints]) if joints else np.zeros((0, 0))
        jump_sent = None
        if last_sent is not None and offset is not None and max_index_read(offset, length) >= s and joints:
            command = sample_actions(spliced, offset, length)
            jump_sent = float(np.abs(command[joints] - np.asarray(last_sent, dtype=np.float64)[joints]).max())
        metrics = {"splice_jump": jump, "splice_jump_sent": jump_sent, "max_plan_step": max_step,
                   "gripper_jump": gripper_jump, "suffix_max": float(suffix.max()) if suffix.size else 0.0,
                   "suffix_mean": float(suffix.mean()) if suffix.size else 0.0}
        if jump > self.max_splice_jump:
            return False, "splice jump {:.3f} rad > {:.3f}".format(jump, self.max_splice_jump), metrics
        if jump_sent is not None and jump_sent > self.max_splice_jump:
            return False, "jump vs last sent command {:.3f} rad > {:.3f}".format(jump_sent,
                                                                                self.max_splice_jump), metrics
        if max_step > self.max_plan_step:
            return False, "plan step {:.3f} rad > {:.3f}".format(max_step, self.max_plan_step), metrics
        if gripper_jump > self.max_gripper_jump:
            return False, "gripper jump {:.3f} > {:.3f}".format(gripper_jump, self.max_gripper_jump), metrics
        return True, None, metrics


class PlanResult:
    def __init__(self, actions, server_chunk=None, req=None, record=None):
        self.actions = actions
        self.server_chunk = server_chunk
        self.req = req
        self.record = record


class _RefreshChunk(_Chunk):
    def __init__(self, actions, start, length, chunk_id, server_chunk, plan_step, plan_req):
        super().__init__(actions, start, length)
        self.id = chunk_id
        self.server_chunk = server_chunk
        self.plan_step = plan_step
        self.plan_req = plan_req
        self.max_read = -1  # highest chunk index read by any command already sampled
        self.attempted = set()  # refresh slots (k) already used: at most one refresh per (chunk, k)
        self.disabled = None  # reason once refreshes stop for the rest of this chunk
        self.source = np.full(len(actions), -1 if plan_req is None else int(plan_req), dtype=np.int64)


class RefreshingChunkExecutor(ChunkExecutor):
    """ChunkExecutor with serve_forceguide's in-chunk refresh (mode "refresh"); mode "chunk" behaves like
    ChunkExecutor apart from the plan call's extra arguments and the bookkeeping.

    Per tick (step): (1) apply the refresh replies that arrived (main thread only, at the top of the tick);
    (2) plan when the chunk is exhausted: plan_fn(frame_provider, plan_step=floor(position), ack=...); (3) sample;
    (4) submit at most one refresh per (chunk, k) when k = floor(offset) > 0, k % refresh_every == 0,
    k_pin = k + lead < chunk length, no request in flight and fresh tactile; tactile_step = plan_step + k (the
    anchor train_cpm uses: c = r + k).

    A reply is spliced from s = max(k_pin, s_min) with s_min = 1 + the highest chunk index any already sampled
    command read, dropped when s >= length, too old (refresh_timeout_s since submission: a client-side apply
    deadline, not a socket timeout) or from another round/chunk, and rejected by the SpliceGuard. Every decided reply
    is acknowledged ({"req", "status", "s"}) in the next refresh or plan so the server's traj_base stays what the
    robot executes. After a transport failure or a server error reply no further refresh is sent in that chunk.
    """

    _instances = itertools.count(1)

    def __init__(self, plan_fn, policy_hz, control_dt, action_horizon=50, clock=time.monotonic, mode="chunk",
                 refresh_every=None, lead=None, transport=None, decode=None, guard=None, can_refresh=None,
                 refresh_timeout_s=None, log_fn=None, sidecar=None, payload=None, connect_budget_s=None,
                 request_timeout_s=None):
        super().__init__(plan_fn, policy_hz, control_dt, action_horizon=action_horizon, clock=clock)
        if mode not in MODES:
            raise ValueError("mode must be one of {}".format(MODES))
        if mode == "refresh":
            if transport is None or decode is None:
                raise ValueError("refresh mode needs a transport and a decode function")
            if refresh_every is None or int(refresh_every) < 1 or lead is None or int(lead) < 1:
                raise ValueError("refresh mode needs refresh_every >= 1 and lead >= 1")
            if self.advance > 1.0 + EPS:
                raise ValueError("refresh mode needs policy_hz * control_dt <= 1 so that every k is reached")
        self.mode = mode
        self.refresh_every = None if refresh_every is None else int(refresh_every)
        self.lead = None if lead is None else int(lead)
        self.transport = transport
        self.decode = decode
        self.guard = guard or SpliceGuard(ARM_JOINTS["both"], ARM_GRIPPERS["both"])
        self.can_refresh = can_refresh
        self.refresh_timeout_s = refresh_timeout_s
        self.payload = payload
        self.connect_budget_s = connect_budget_s
        self.request_timeout_s = request_timeout_s  # the worker's hard reply timeout for these jobs (None: its own)
        self.round_id = next(self._instances)
        self.generation = 0
        self.pending_ack = None
        self.last_sent = None
        self.last_tick = {}
        self.counts = collections.Counter()
        self.sidecar = sidecar if sidecar is not None else []
        self._log = log_fn or (lambda record: None)
        self._chunk_counter = 0
        self._tick_time = None
        self._worker_warned = False

    # -- public helpers -------------------------------------------------------------------------

    def reset(self):
        """Forget the chunk (abort/pause): replies to requests made before this are dropped on arrival."""
        super().reset()
        self.generation += 1
        self.pending_ack = None

    def close(self):
        pass

    def note_sent(self, command):
        """The command FollowerCommander actually sent this tick (after clamps), for the splice check."""
        self.last_sent = None if command is None else np.asarray(command, dtype=np.float64).copy()

    def finish(self, wait_s=0.0, sleep=time.sleep):
        """End of the round: nothing is applied any more. Waits up to wait_s for the request in flight and logs its
        reply as dropped ("round-ended", not counted against the round) so it does not leak into the next round."""
        self.generation += 1
        self.pending_ack = None
        if self.transport is None:
            return
        deadline = self.clock() + wait_s
        while True:
            self._tick_time = self.clock()
            for result in self.transport.poll():
                self._finish(result.request, "dropped", "round-ended", count=False, result=result)
            if not self.transport.busy or self.clock() >= deadline:
                return
            sleep(0.005)

    @property
    def chunk(self):
        return self._chunk

    def summary(self):
        """Refresh outcome counts of the round (see INFRA_REFRESH_STATUSES / PROTOCOL_REFRESH_STATUSES).

        infra = slots lost to infrastructure faults (failed / error / skipped-stale / skipped-disabled / malformed
        replies): infra_frac above the campaign threshold makes the trial an infrastructure failure (re-run).
        protocol = rejected / dropped / skipped-in-flight / late slots: method and timing outcomes, recorded as
        protocol flags only. bad = both (not K = 0, the policy's own choice)."""
        counts = dict(self.counts)
        slots = self.counts["slots"]
        infra = sum(self.counts[status] for status in INFRA_REFRESH_STATUSES) + self.counts["malformed"]
        protocol = (sum(self.counts[status] for status in PROTOCOL_REFRESH_STATUSES) - self.counts["malformed"]
                    + self.counts["late"])
        bad = infra + protocol
        late_frac = self.counts["late_any"] / slots if slots else 0.0

        def frac(n):
            return n / slots if slots else 0.0

        return {"slots": slots, "counts": counts, "bad": bad, "bad_frac": frac(bad), "infra": infra,
                "infra_frac": frac(infra), "protocol": protocol, "protocol_frac": frac(protocol),
                "accept_frac": frac(self.counts["applied"]), "late_frac": late_frac,
                "late_flag": late_frac > LATE_FLAG_FRAC, "pin_mismatch": self.counts["pin_mismatch"],
                "plans": self._chunk_counter}

    # -- the tick ----------------------------------------------------------------------------------

    def step(self, frame_provider):
        self._tick_time = self.clock()
        if self.transport is not None:
            for result in self.transport.poll():
                self._resolve(result)
        planned = False
        if self._chunk is None or self._chunk.sample(self.position) is None:
            self._plan(frame_provider)
            planned = True
        chunk = self._chunk
        offset = self.position - chunk.start
        action = chunk.sample(self.position)
        low = min(chunk_k(offset), chunk.length - 1)
        high = max_index_read(offset, chunk.length)
        chunk.max_read = max(chunk.max_read, high)
        k = chunk_k(offset)
        self.last_tick = {"chunk": chunk.id, "server_chunk": chunk.server_chunk, "position": self.position,
                          "offset": offset, "step": global_step(self.position), "k": k, "planned": planned,
                          "plan_req": chunk.plan_req if planned else None,
                          "src_req": sorted({int(chunk.source[low]), int(chunk.source[high])})}
        self._maybe_submit(chunk, k)
        self.position += self.advance
        return action

    def _plan(self, frame_provider):
        self._chunk_counter += 1
        plan_step = global_step(self.position)
        ack, self.pending_ack = self.pending_ack, None
        start = self.clock()
        try:
            result = self.infer_fn(frame_provider, plan_step=plan_step, ack=ack)
        except InfrastructureFailure:
            raise
        except Exception as exc:
            raise PolicyError("inference failed: {}".format(exc)) from exc
        self.latencies.append(self.clock() - start)
        if not isinstance(result, PlanResult):
            result = PlanResult(result)
        actions = np.asarray(result.actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) == 0 or not np.isfinite(actions).all():
            raise PolicyError("policy returned an invalid action chunk {}".format(actions.shape))
        self._chunk = _RefreshChunk(actions.copy(), self.position, min(self.action_horizon, len(actions)),
                                    self._chunk_counter, result.server_chunk, plan_step, result.req)

    # -- refresh submission ------------------------------------------------------------------------

    def _slot_request(self, chunk, k):
        return RefreshRequest(kind="refresh", round_id=self.round_id, generation=self.generation, chunk_id=chunk.id,
                              server_chunk=chunk.server_chunk, k=k, k_pin=k + self.lead,
                              tactile_step=chunk.plan_step + k, tick_time=self._tick_time, ack=self.pending_ack,
                              payload=self.payload, connect_budget_s=self.connect_budget_s,
                              timeout=self.request_timeout_s)

    def _maybe_submit(self, chunk, k):
        if self.mode != "refresh" or k <= 0 or k % self.refresh_every or k in chunk.attempted:
            return
        if k + self.lead >= chunk.length:
            return
        chunk.attempted.add(k)
        self.counts["slots"] += 1
        request = self._slot_request(chunk, k)
        if chunk.disabled is None and not getattr(self.transport, "alive", True):
            chunk.disabled = "refresh worker stopped"
            if not self._worker_warned:  # the operator sees the trial's outcome (console_reason), not this
                logging.error("forceguide: the refresh worker is not running; refreshes are skipped", extra=FILE_ONLY)
                self._worker_warned = True
        if chunk.disabled is not None:
            self._finish(request, "skipped-disabled", chunk.disabled)
            return
        if self.transport.busy:
            self._finish(request, "skipped-in-flight", "previous request still in flight")
            return
        if self.can_refresh is not None:
            ok, reason = self.can_refresh(request.tactile_step)
            if not ok:
                self._finish(request, "skipped-stale", reason)
                return
        request.submit_time = self.clock()
        if not self.transport.submit(request):
            self._finish(request, "skipped-in-flight", "worker refused the job")
            return
        self.pending_ack = None
        self.counts["submitted"] += 1

    # -- reply handling ----------------------------------------------------------------------------

    def _finish(self, request, status, reason=None, count=True, result=None, **fields):
        if count:
            self.counts[status] += 1
            if fields.get("late"):
                # "late" (applied past k_pin) is a bad outcome of its own; "late_any" also counts the late replies
                # that were then dropped or rejected, for the late-fraction flag.
                self.counts["late_any"] += 1
                if status == "applied":
                    self.counts["late"] += 1
        record = {"type": request.kind, "status": status, "reason": reason, "round": request.round_id,
                  "chunk": request.chunk_id, "server_chunk": request.server_chunk, "k": request.k,
                  "k_pin": request.k_pin, "tactile_step": request.tactile_step, "req": request.req,
                  "tactile_steps": list(request.steps), "window": request.window, "ack_sent": request.ack,
                  "t_tick": request.tick_time, "t_submit": request.submit_time, "prepare_ms": request.prepare_ms}
        if result is not None:
            record.update({"t_finish": result.finish_time, "client": result.timing, "reconnects": result.reconnects,
                           "transport": result.status, "transport_error": result.error})
            if result.finish_time is not None and request.submit_time is not None:
                record["rtt_ms"] = 1000.0 * (result.finish_time - request.submit_time)
            if result.finish_time is not None and request.tick_time is not None:
                record["rtt_tick_ms"] = 1000.0 * (result.finish_time - request.tick_time)
            record["apply_delay_ms"] = (None if result.finish_time is None
                                        else 1000.0 * (self._tick_time - result.finish_time))
            reply = result.reply or {}
            block = _reply_block(reply)
            if block:
                record.update({"K": block.get("K"), "signal": block.get("signal"), "timing_ms": block.get("timing_ms"),
                               "diag": block.get("diag"), "missing_steps": block.get("missing_steps"),
                               "reply_k": block.get("k"), "reply_k_pin": block.get("k_pin")})
            if reply.get("server_timing") is not None:
                record["server_timing"] = reply.get("server_timing")
        record.update(fields)
        self._log(record)
        return record

    def _ack(self, request, status, s):
        self.pending_ack = {"req": int(request.req), "status": status, "s": int(s)}

    def _resolve(self, result):
        request = result.request
        chunk = self._chunk
        if (request.round_id != self.round_id or request.generation != self.generation or chunk is None
                or request.chunk_id != chunk.id):
            # The chunk (or round) it was made for is gone; the server dropped that chunk too, so no ack.
            same_round = request.round_id == self.round_id
            self._finish(request, "dropped", "stale-chunk" if same_round else "stale-round", count=same_round,
                         result=result)
            return
        if result.status != "ok":
            chunk.disabled = "transport {}".format(result.status)
            self.pending_ack = None
            self._finish(request, "failed", result.status, result=result)
            if result.status == "server_changed":
                raise ServerChanged(result.error or "server changed")
            return
        reply = result.reply or {}
        block = _reply_block(reply)
        if block.get("req") is not None and int(block["req"]) != request.req:
            chunk.disabled = "protocol"
            self.pending_ack = None
            self._finish(request, "failed", "protocol: reply req {} != {}".format(block.get("req"), request.req),
                         result=result)
            return
        if block.get("error") is not None:
            chunk.disabled = "server error"  # the server invalidates the chunk on errors
            self.pending_ack = None
            self._finish(request, "error", str(block["error"]), result=result)
            return
        if block.get("K") is not None and int(block["K"]) == 0:
            # Dynamic K chose not to refresh: nothing to apply, nothing pending on the server.
            self._ack(request, "dropped", -1)
            self._finish(request, "skipped-K0", None, result=result)
            return
        if "actions" not in reply:
            self._ack(request, "rejected", -1)
            self.counts["malformed"] += 1  # a server fault, not a splice-guard decision: counts as infrastructure
            self._finish(request, "rejected", "protocol: refresh reply without actions, error or K = 0",
                         result=result, malformed=True)
            return
        if (self.refresh_timeout_s is not None and request.submit_time is not None
                and self._tick_time - request.submit_time > self.refresh_timeout_s):
            self._ack(request, "dropped", -1)
            self._finish(request, "dropped", "deadline", result=result,
                         age_ms=1000.0 * (self._tick_time - request.submit_time))
            return
        try:
            new = self.decode(reply["actions"])
        except Exception as exc:  # noqa: BLE001
            self._ack(request, "rejected", -1)
            self.counts["malformed"] += 1
            self._finish(request, "rejected", "decode: {}".format(exc), result=result, malformed=True)
            return
        # s_min: the first chunk index no command sent so far has read (a fractional offset reads a[low] and
        # a[low + 1]); the reply may only replace the plan from there and from k_pin on.
        s_min = chunk.max_read + 1
        s = max(int(request.k_pin), s_min)
        old = chunk.actions
        active = sorted(set(self.guard.arm_joints) | set(self.guard.grippers))
        pin_mismatch = None
        if new.shape == old.shape and request.k_pin > 0 and active:
            # The server pins positions < k_pin to its traj_base, which the acks keep equal to this plan.
            pin_mismatch = float(np.abs(new[:request.k_pin][:, active] - old[:request.k_pin][:, active]).max())
            if pin_mismatch > PIN_TOLERANCE:
                self.counts["pin_mismatch"] += 1
                # client.log only: a console line would tell the (blinded) operator that this is a refresh condition
                DEBUG_LOG.debug("refresh req %s: reply differs from the executed plan by %.2e below k_pin",
                                request.req, pin_mismatch, extra=FILE_ONLY)
        common = dict(s=s, s_min=s_min, late=max(s - int(request.k_pin), 0), pin_mismatch=pin_mismatch)
        if s >= chunk.length:
            self._ack(request, "dropped", s)
            self._remember(request, s, "dropped", old, new)
            self._finish(request, "dropped", "past-end", result=result, **common)
            return
        ok, reason, metrics = self.guard.check(old, new, s, chunk.length, self.last_sent,
                                               self.position - chunk.start)
        if not ok:
            self._ack(request, "rejected", s)
            self._remember(request, s, "rejected", old, new)
            self._finish(request, "rejected", reason, result=result, **dict(common, **metrics))
            return
        self._remember(request, s, "applied", old, new)
        chunk.actions = old.copy()
        chunk.actions[s:] = new[s:]
        chunk.source[s:] = int(request.req)
        self._ack(request, "applied", s)
        self._finish(request, "applied", None, result=result, **dict(common, **metrics))

    def _remember(self, request, s, status, old, new):
        if new.shape == old.shape:
            self.sidecar.append({"req": int(request.req), "k": int(request.k), "k_pin": int(request.k_pin),
                                 "s": int(s), "status": status, "old": old.copy(), "new": np.asarray(new).copy()})


# --------------------------------------------------------------------------------------------
# Splice timing simulation (the --lead choice)
# --------------------------------------------------------------------------------------------

def simulate_refresh_timing(advance, rtt_s, control_dt, lead, horizon=50, refresh_every=5):
    """The executor's refresh timing in one chunk, with every reply taking rtt_s (from the tick it was submitted in)
    and applied at the top of the first tick at or after it arrives. Returns (decided, skipped): decided is a list
    of {k, k_pin, s, late} (late = s - k_pin when the reply could only be spliced past k_pin), skipped the ks whose
    slot found the previous request still in flight."""
    decided, skipped = [], []
    in_flight = None
    attempted = set()
    max_read = -1
    for tick in range(ticks_per_chunk(horizon, advance)):
        now = tick * control_dt
        offset = tick * advance
        if in_flight is not None and in_flight["arrival"] <= now + EPS:
            s = max(in_flight["k_pin"], max_read + 1)
            decided.append({"k": in_flight["k"], "k_pin": in_flight["k_pin"], "s": s,
                            "late": max(s - in_flight["k_pin"], 0), "dropped": s >= horizon})
            in_flight = None
        max_read = max(max_read, max_index_read(offset, horizon))
        k = chunk_k(offset)
        if k > 0 and k % refresh_every == 0 and k + lead < horizon and k not in attempted:
            attempted.add(k)
            if in_flight is None:
                in_flight = {"k": k, "k_pin": k + lead, "arrival": now + rtt_s}
            else:
                skipped.append(k)
    return decided, skipped


def required_lead(advance, rtt_s, control_dt, horizon=50, refresh_every=5):
    """Smallest lead (policy steps) for which every refresh reply of a chunk is spliced exactly at k_pin, i.e. arrives
    before the commands reach k_pin (interpolation included), for replies taking rtt_s; None if no lead works.

    Reproduces the review simulation's table (minimal splice rule, 50 Hz loop): 30 Hz policy needs lead 2 up to
    40 ms, 3 up to 60 ms, 4 up to 100 ms; 50 Hz policy 2 up to 40 ms, 3 up to 60, 4 up to 80, 5 up to 100. Use the
    p95 refresh RTT measured with the real payload at the campaign's largest K (--preflight).
    """
    for lead in range(1, horizon):
        decided, _ = simulate_refresh_timing(advance, rtt_s, control_dt, lead, horizon, refresh_every)
        if decided and all(item["late"] == 0 for item in decided):
            return lead
    return None


# --------------------------------------------------------------------------------------------
# Campaign, trials, noise
# --------------------------------------------------------------------------------------------

class PayloadSettings:
    def __init__(self, spec, key_sensors, window, stride, gap, background, proxy):
        self.spec = spec
        self.key_sensors = key_sensors
        self.window = window
        self.stride = stride
        self.gap = gap
        self.background = background
        self.proxy = proxy

    def make_payload(self, history):
        return TactilePayload(history, self.key_sensors, self.window, self.stride, self.gap, self.background,
                              self.proxy)


class CampaignConfig:
    """Settings shared by every trial of a campaign, so that all conditions run the identical client pipeline:
    one policy rate, one refresh_every and one lead for every refresh condition, the same tactile sensors
    subscribed and uploaded (conditions without a tactile CPM - P0, vision-only twins - upload the frames of a
    proxy tactile condition, which the server ignores)."""

    def __init__(self, info, conditions, lead=None, policy_hz=None, refresh_every=None, force=False,
                 sensor_overrides=None, history_margin=HISTORY_MARGIN_STEPS, require_lead=True):
        names = list(dict.fromkeys(conditions))
        if not names:
            raise ValueError("no conditions to run")
        for name in names:
            info.condition(name)
        self.info = info
        self.names = names
        self.policy_hz = info.policy_hz
        if policy_hz is not None and abs(float(policy_hz) - info.policy_hz) > 1e-9:
            if not force:
                raise ValueError("--policy_hz {} differs from the server's {} (the CPM's data rate); pass --force to "
                                 "override".format(policy_hz, info.policy_hz))
            self.policy_hz = float(policy_hz)
        refresh = [name for name in names if info.conditions[name].mode == "refresh"]
        self.refresh_conditions = refresh
        cadences = sorted({info.conditions[name].refresh_every for name in refresh})
        self.refresh_every = cadences[0] if cadences else None
        if len(cadences) > 1 and not force:
            raise ValueError("refresh conditions use different refresh_every {}; one cadence per campaign".format(
                cadences))
        if refresh_every is not None and cadences and int(refresh_every) != self.refresh_every:
            if not force:
                raise ValueError("--refresh_every {} differs from the server's {}; pass --force to override".format(
                    refresh_every, self.refresh_every))
            self.refresh_every = int(refresh_every)
        if lead is not None and int(lead) < 1:
            raise ValueError("--lead must be >= 1")
        # Robot review finding 1 (contract C1): the lead k_pin - k decides which steps a refresh puts into execution,
        # so a condition's eta calibration (calibration.json, metadata "lead") is valid only for that lead, and the
        # server refuses refreshes with any other lead (and below a dynamic-K table's "min_lead").
        self.calibrated_leads = {name: info.conditions[name].lead for name in refresh
                                 if info.conditions[name].lead is not None}
        leads = sorted(set(self.calibrated_leads.values()))
        min_leads = [info.conditions[name].min_lead for name in refresh if info.conditions[name].min_lead is not None]
        self.min_lead = max(min_leads) if min_leads else None
        if len(leads) > 1:
            message = ("the refresh conditions were calibrated with different leads {}: one lead per campaign; re-run "
                       "offline_guidance_check --lead <L> for all of them and serve that calibration".format(
                           self.calibrated_leads))
            if not force:
                raise ValueError(message)
            logging.warning("%s - continuing because of --force", message)
        if lead is None and len(leads) == 1 and not require_lead:
            lead = leads[0]  # preflight mode: measure with the lead the conditions were calibrated with
        if refresh and lead is None and require_lead:
            hint = " (the refresh conditions were calibrated with lead {})".format(leads[0]) if len(leads) == 1 else ""
            raise ValueError("refresh conditions need --lead (policy steps, from --preflight){}".format(hint))
        self.lead = None if lead is None else int(lead)
        if self.lead is not None and refresh:
            other = {name: value for name, value in self.calibrated_leads.items() if value != self.lead}
            if other:
                message = ("--lead {} differs from the lead the eta of {} was calibrated with ({}): re-run "
                           "offline_guidance_check --lead {} and serve the new calibration.json, or run the campaign "
                           "with the calibrated lead".format(self.lead, sorted(other), sorted(set(other.values())),
                                                             self.lead))
                if not force:
                    raise ValueError(message)
                logging.warning("%s - continuing because of --force", message)
            if self.min_lead is not None and self.lead < self.min_lead:
                message = ("--lead {} is below min_lead {} of the dynamic-K tables served (dynamic_k.json): the server "
                           "refuses these refreshes; raise --lead or re-run calibrate_dynamic_k".format(
                               self.lead, self.min_lead))
                if not force:
                    raise ValueError(message)
                logging.warning("%s - continuing because of --force", message)
        # The lead a preflight / warm-up measures with when none is set (at least every served min_lead).
        self.default_lead = self.lead if self.lead is not None else max(3, self.min_lead or 1)
        for name in names:
            spec = info.conditions[name].tactile
            if spec is not None and spec.resize == "pad" and not force:
                raise ValueError("condition {!r} uses the legacy 'pad' tactile resize, which the client cannot "
                                 "reproduce".format(name))
        self.allow_legacy_pad = bool(force)
        tactile = sorted(name for name in names if info.conditions[name].tactile is not None)
        self.proxy = tactile[0] if tactile else None
        self.sensor_overrides = dict(sensor_overrides or {})
        key_sensors = collections.OrderedDict()
        for name in tactile:
            for key in info.conditions[name].tactile.keys:
                key_sensors[key] = sensor_for_key(key, self.sensor_overrides)
        self.key_sensors = key_sensors
        self.sensors = tuple(sorted(set(key_sensors.values())))
        span = max([info.conditions[name].span for name in tactile] or [0])
        self.history_capacity = span + int(history_margin)
        # Robot review finding 1(b): with stride and gap multiples of refresh_every, each refresh uploads about one
        # new step (the older window entries and previous frames are cached); otherwise payloads (and RTT) grow.
        self.notes = []
        for name in refresh:
            condition = info.conditions[name]
            for what, value in (("stride", condition.stride if condition.window > 1 else 0), ("gap", condition.gap)):
                if value and self.refresh_every and value % self.refresh_every:
                    self.notes.append("condition {!r}: {} {} is not a multiple of refresh_every {} (larger "
                                      "refresh payloads)".format(name, what, value, self.refresh_every))
        for note in self.notes:
            logging.warning(note)

    def settings(self, name):
        condition = self.info.condition(name)
        source = condition if condition.tactile is not None else (
            self.info.conditions[self.proxy] if self.proxy is not None else None)
        if source is None:
            return PayloadSettings(None, collections.OrderedDict(), 1, 1, 0, False, proxy=True)
        spec = source.tactile
        key_sensors = collections.OrderedDict((key, self.key_sensors[key]) for key in spec.keys)
        return PayloadSettings(spec, key_sensors, source.window, source.stride, source.gap, spec.needs_background,
                               proxy=condition.tactile is None)

    def to_dict(self):
        return {"conditions": self.names, "policy_hz": self.policy_hz, "refresh_every": self.refresh_every,
                "lead": self.lead, "calibrated_leads": dict(self.calibrated_leads), "min_lead": self.min_lead,
                "default_lead": self.default_lead, "refresh_conditions": self.refresh_conditions, "proxy": self.proxy,
                "sensors": list(self.sensors), "key_sensors": dict(self.key_sensors),
                "history_capacity": self.history_capacity, "notes": list(self.notes)}


class NoiseSchedule:
    """Paired flow noise: plan i of a trial gets default_rng([seed, noise_id, i]).standard_normal((H, D)), with
    noise_id = the initial condition by default, so every condition starts from the same noise at the same initial
    condition. Restarts at 0 for each attempt (re-runs reproduce the noise); refreshes consume none."""

    def __init__(self, seed, noise_id, horizon, dim):
        if int(seed) < 0 or int(noise_id) < 0:
            raise ValueError("seed and noise_id must be non-negative integers")
        self.seed = int(seed)
        self.noise_id = int(noise_id)
        self.horizon = int(horizon)
        self.dim = int(dim)
        self.plan_index = 0

    def next(self):
        rng = np.random.default_rng([self.seed, self.noise_id, self.plan_index])
        self.plan_index += 1
        return rng.standard_normal((self.horizon, self.dim)).astype(np.float32)


def make_schedule(conditions, init_condition_ids, seed=0, repeats=1):
    """Trial rows: blocks of one initial condition each, every condition once per block in a random order.

    noise_id equals the initial condition (repeats == 1) or initial condition + 1e6 * repeat, so repeated blocks of
    one initial condition do not reuse noise while the conditions inside a block share it."""
    conditions = list(dict.fromkeys(conditions))
    inits = [int(init) for init in init_condition_ids]
    if len(set(inits)) != len(inits):
        raise ValueError("initial condition ids repeat ({}): use repeats for more blocks per initial condition".format(
            sorted({init for init in inits if inits.count(init) > 1})))
    rows, index, block = [], 0, 0
    for repeat in range(int(repeats)):
        for init in inits:
            init = int(init)
            order = list(conditions)
            np.random.default_rng([int(seed), repeat, init]).shuffle(order)
            for condition in order:
                rows.append({"trial_id": "t{:04d}".format(index), "block": block, "init_condition_id": init,
                             "condition": condition, "noise_id": init + 1000000 * repeat})
                index += 1
            block += 1
    return rows


def write_schedule(path, rows):
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def file_sha256(path):
    """sha256 of a file's bytes (the schedule's identity in every episodes.jsonl row)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_sha256(rows):
    """sha256 of a generated schedule (rows as canonical JSON), for runs without a schedule file."""
    return hashlib.sha256(json.dumps(to_jsonable(list(rows)), sort_keys=True).encode()).hexdigest()


def load_schedule(path, info=None):
    """Rows of trials.jsonl (trial_id, init_condition_id, condition[, noise_id, block]), validated: trial ids are
    unique, and so is every (condition, noise_id) pair (a repeated pair would run the same condition on the same
    paired noise twice, and the pairing on (block, noise_id) could not tell the two apart)."""
    rows, seen, pairs = [], set(), {}
    with open(path) as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("trial_id", "init_condition_id", "condition"):
                if key not in row:
                    raise ValueError("{}:{}: missing {!r}".format(path, number, key))
            row["trial_id"] = str(row["trial_id"])
            if row["trial_id"] in seen:
                raise ValueError("{}:{}: duplicate trial_id {!r}".format(path, number, row["trial_id"]))
            seen.add(row["trial_id"])
            try:
                row["init_condition_id"] = int(row["init_condition_id"])
                row["noise_id"] = int(row.get("noise_id", row["init_condition_id"]))
            except (TypeError, ValueError):
                raise ValueError("{}:{}: init_condition_id / noise_id must be integers (they seed the paired "
                                 "noise)".format(path, number)) from None
            if row["noise_id"] < 0:
                raise ValueError("{}:{}: noise_id must be >= 0".format(path, number))
            pair = (str(row["condition"]), row["noise_id"])
            if pair in pairs:
                raise ValueError("{}:{}: condition {!r} with noise_id {} repeats trial {!r} (each condition runs "
                                 "once per noise_id)".format(path, number, pair[0], pair[1], pairs[pair]))
            pairs[pair] = row["trial_id"]
            if info is not None:
                info.condition(row["condition"])
            rows.append(row)
    if not rows:
        raise ValueError("{} lists no trials".format(path))
    return rows


# --------------------------------------------------------------------------------------------
# One trial attempt
# --------------------------------------------------------------------------------------------

class TrialContext:
    """What changes per trial attempt: condition, tactile payload, noise and the logs; provides the executor's
    plan function and its refresh gate."""

    def __init__(self, client, adapter, condition, payload, monitor=None, noise=None, measured_fn=None,
                 plan_timeout=None, max_tactile_age=DEFAULT_MAX_TACTILE_AGE_S, labels=None):
        self.client = client
        self.adapter = adapter
        self.condition = condition
        self.payload = payload
        self.monitor = monitor
        self.noise = noise
        self.measured_fn = measured_fn
        self.plan_timeout = plan_timeout
        self.max_tactile_age = max_tactile_age
        self.labels = dict(labels or {})
        self.requests = []
        self.ticks = []
        self.sidecar = []
        self.plan_records = {}

    def log_request(self, record):
        """Every request record carries the session id and the trial labels: client and server logs join on
        (session, req), never on time."""
        record = dict(record)
        record.update(self.labels)
        if self.client is not None:
            record.setdefault("session", self.client.session)
        self.requests.append(record)

    def stale_reasons(self, step):
        reasons = list(self.monitor.stale(self.payload.sensors)) if self.monitor is not None else []
        if not reasons:
            ok, reason = self.payload.fresh(step, self.max_tactile_age)
            if not ok:
                reasons.append(reason)
        return reasons

    def can_refresh(self, tactile_step):
        reasons = self.stale_reasons(tactile_step)
        return (False, "; ".join(reasons)) if reasons else (True, None)

    def plan(self, frame_provider, plan_step, ack=None):
        reasons = self.stale_reasons(plan_step)
        if reasons:
            raise StaleTactile("stale tactile at the chunk start: {}".format("; ".join(reasons)))
        frame = frame_provider()
        measured = self.measured_fn() if self.adapter.state_gripper == "vtla" and self.measured_fn else None
        obs = self.adapter.prepare(frame, measured)
        noise = self.noise.next() if self.noise is not None else np.zeros(
            (self.adapter.action_horizon, self.adapter.model_action_dim), dtype=np.float32)
        noise_index = None if self.noise is None else self.noise.plan_index - 1
        reply, record = self.client.plan(obs, plan_step, self.payload, noise, ack=ack, timeout=self.plan_timeout)
        actions = self.adapter.decode(reply["actions"])
        record["noise_index"] = noise_index
        self.log_request(record)
        self.plan_records[record["req"]] = self.requests[-1]
        return PlanResult(actions, server_chunk=record["server_chunk"], req=record["req"], record=record)

    def annotate_plan(self, req, **fields):
        record = self.plan_records.get(req)
        if record is not None:
            record.update(fields)


# --------------------------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------------------------

def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_jsonl(path, records, mode="w"):
    with open(path, mode) as handle:
        for record in records:
            handle.write(json.dumps(to_jsonable(record)) + "\n")


def summarize_ms(values):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size), "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "max": float(values.max())}


def save_sidecar(path, entries):
    """npz of every refresh reply's old/new plan (decided against), for smoothness analysis."""
    if not entries:
        return False
    np.savez_compressed(path, req=np.array([e["req"] for e in entries]), k=np.array([e["k"] for e in entries]),
                        k_pin=np.array([e["k_pin"] for e in entries]), s=np.array([e["s"] for e in entries]),
                        status=np.array([e["status"] for e in entries]),
                        old=np.stack([e["old"] for e in entries]), new=np.stack([e["new"] for e in entries]))
    return True
