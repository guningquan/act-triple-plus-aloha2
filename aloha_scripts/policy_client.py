"""Policy-server client shared by every script that runs a remote policy on this robot.

openpi (scripts/serve_policy.py) and N0-VTLA (scripts/serve_policy.py) serve their policies with
the same websocket + msgpack protocol: on connect the server sends one packed metadata dict, then
every send/recv pair is one inference. PolicyClient speaks that protocol without importing
openpi_client / n0vtla_client, so it runs under the ROS box's system Python 3.8 with only
`websockets` and `msgpack` installed.

Each model needs its own observation layout, so the model-specific part lives in an adapter:
    OpenpiAlohaAdapter   openpi ALOHA configs (e.g. pi05_packaging_simple_2_full): the observation
                         openpi's examples/aloha_real client sends, {"state", "images", "prompt"}
    N0VtlaAdapter        N0-VTLA vtla_tactile_posttrain (eef) / vtla_joint_posttrain (joint):
                         the observation run_vtla_policy.py sends, including the four tactile views
An adapter turns one captured Frame into the server's observation (prepare) and the returned
action chunk into (H, 14) follower commands (decode): [left arm 6 (rad), left gripper, right arm 6,
right gripper], grippers as the normalized opening RealEnv.step() takes (0 = closed, 1 = open).
Everything model-specific beyond that (joint flips, gripper conversions, normalization, resizing
to the model's resolution, delta actions) stays in the server's own transforms.
"""

import logging
import time

import msgpack
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------------------------
# msgpack-numpy, wire-compatible with openpi_client/n0vtla_client msgpack_numpy.py
# --------------------------------------------------------------------------------------------

def pack_array(obj):
    # Wrapper-dict keys are bytes literals, not str, on purpose: msgpack's "bin" wire type
    # always round-trips as python bytes regardless of the unpacker's `raw` setting, while a
    # str key's wire type ("str") decodes to str or bytes depending on that setting and on the
    # msgpack-python version's default. openpi_client/n0vtla_client build this same dict with
    # bytes keys for exactly that reason - match it exactly, or a server pinned to a different
    # msgpack-python version silently fails to recognise the wrapper and hands the model a raw
    # {"data":...,"dtype":...} dict instead of an ndarray.
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError("Unsupported dtype for the wire: {}".format(obj.dtype))
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": list(obj.shape)}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj or "__ndarray__" in obj:
        data = obj.get(b"data", obj.get("data"))
        dtype = obj.get(b"dtype", obj.get("dtype"))
        shape = obj.get(b"shape", obj.get("shape"))
        return np.ndarray(buffer=data, dtype=np.dtype(dtype), shape=tuple(shape))
    if b"__npgeneric__" in obj or "__npgeneric__" in obj:
        data = obj.get(b"data", obj.get("data"))
        dtype = obj.get(b"dtype", obj.get("dtype"))
        return np.dtype(dtype).type(data)
    return obj


def packb(obj):
    return msgpack.packb(obj, default=pack_array, use_bin_type=True)


def unpackb(data):
    return msgpack.unpackb(data, object_hook=unpack_array, raw=False)


class PolicyClient:
    """Blocking websocket client for the openpi/n0vtla WebsocketPolicyServer protocol.

    `websockets.sync.client` gives a plain blocking socket - no asyncio needed. `infer_timeout_s`
    (or infer()'s `timeout`) bounds each round trip, so a hung server surfaces as TimeoutError
    instead of freezing the control loop forever. The server answers every request in order, so
    after a timeout the late reply would be read as the answer to the NEXT request: the connection
    is therefore dropped before the TimeoutError is raised and reopened by the next infer().
    """

    def __init__(self, host, port, connect_timeout_s=30.0, infer_timeout_s=None):
        uri = host if host.startswith("ws") else "ws://{}".format(host)
        if port is not None:
            uri = "{}:{}".format(uri, port)
        self.uri = uri
        self.connect_timeout_s = connect_timeout_s
        self.infer_timeout_s = infer_timeout_s
        self._ws = None
        self._connect()

    def _connect(self):
        import websockets.sync.client as ws_client

        deadline = time.monotonic() + self.connect_timeout_s
        last_error = None
        while time.monotonic() < deadline:
            try:
                self._ws = ws_client.connect(self.uri, compression=None, max_size=None, close_timeout=1.0)
                break
            except (ConnectionRefusedError, OSError) as exc:
                last_error = exc
                logging.info("Waiting for policy server at %s ...", self.uri)
                time.sleep(2.0)
        else:
            raise RuntimeError("Could not connect to policy server at {}: {}".format(self.uri, last_error))
        self.metadata = unpackb(self._ws.recv())

    def infer(self, obs, timeout=None):
        if self._ws is None:
            self._connect()
        self._ws.send(packb(obs))
        try:
            response = self._ws.recv(timeout=self.infer_timeout_s if timeout is None else timeout)
        except TimeoutError:
            ws, self._ws = self._ws, None  # the reply still on its way dies with this connection
            try:
                ws.close()
            except Exception:
                pass
            raise
        if isinstance(response, str):
            raise RuntimeError("Policy server error:\n{}".format(response))
        return unpackb(response)

    def close(self):
        if self._ws is not None:
            self._ws.close()
            self._ws = None


# --------------------------------------------------------------------------------------------
# openpi_client.image_tools.resize_with_pad, copied in behaviour (PIL bilinear, centered zero pad)
# --------------------------------------------------------------------------------------------

def resize_with_pad(images, height, width, method=Image.BILINEAR):
    """Resize [..., H, W, C] uint8 images to (height, width) without distortion, padding with
    zeros - the exact PIL path openpi's aloha_real client runs before sending (its training data
    went through the same function, so the pixels match what the model was trained on)."""
    if images.shape[-3:-1] == (height, width):
        return images
    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image, height, width, method):
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.asarray(image)
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)
    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return np.asarray(zero_image)


# --------------------------------------------------------------------------------------------
# Observation capture and model adapters
# --------------------------------------------------------------------------------------------

class Frame:
    """One control tick's observation, captured before that tick's command is sent.

    qpos/qvel/effort/images: RealEnv.get_observation() - the hdf5 recorder's layout (qpos grippers
        are the normalized finger POSITION, images are the camera topics' own rgb8 bytes).
    vtla: record_episodes_aloha2_vtla.LiveStreams snapshot ({name: {"image" BGR, ...}} for the
        cameras and tactile sensors, {"state": 7-vector} for joint_left/joint_right), or None when
        the VTLA streams are not running.
    tactile: RealEnv.get_observation()'s obs['tactile'] (a TactileRecorder snapshot taken with the
        images), or None when no TactileRecorder is attached.
    timestamps: RealEnv.get_observation()'s obs['timestamps'], or None.
    """

    def __init__(self, time, qpos, qvel, effort, images, vtla=None, tactile=None, timestamps=None):
        self.time = time
        self.qpos = qpos
        self.qvel = qvel
        self.effort = effort
        self.images = images
        self.vtla = vtla
        self.tactile = tactile
        self.timestamps = timestamps


def solve_ik(arm, xyz, rotation, guess):
    """Joint angles reaching the pose (xyz, 3x3 rotation) in the arm's base frame, or None.

    InterbotixArmXSInterface.set_ee_pose_matrix(execute=False) without its velocity check: that
    check divides the distance to the arm's CURRENT command by one control period, so it rejects
    every step of a chunk that lies more than ~0.06 rad ahead. The same wrapping into the joint
    range and the same position-limit check are kept.
    """
    import modern_robotics as mr

    target = np.eye(4)
    target[:3, :3] = rotation
    target[:3, 3] = xyz
    theta, success = mr.IKinSpace(arm.robot_des.Slist, arm.robot_des.M, target, list(guess), 0.001, 0.001)
    if not success:
        return None
    rev = getattr(arm, "rev", 2 * np.pi)
    lower, upper = arm.group_info.joint_lower_limits, arm.group_info.joint_upper_limits
    theta = list(theta)
    for x in range(len(theta)):
        if theta[x] <= -rev:
            theta[x] %= -rev
        elif theta[x] >= rev:
            theta[x] %= rev
        if round(theta[x], 3) < round(lower[x], 3):
            theta[x] += rev
        elif round(theta[x], 3) > round(upper[x], 3):
            theta[x] -= rev
        if not lower[x] <= int(theta[x] * 1000) / 1000.0 <= upper[x]:
            return None
    return theta


def _finite_chunk(actions, min_dim):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < min_dim or len(actions) == 0:
        raise RuntimeError("Unexpected action chunk shape from the server: {}".format(actions.shape))
    return actions


class OpenpiAlohaAdapter:
    """openpi ALOHA policy (AlohaInputs/AlohaOutputs on the server, adapt_to_pi included).

    Sends exactly what openpi's examples/aloha_real client sends: qpos (14,) as state, each camera
    resized with pad to 224x224 (PIL) and transposed to CHW uint8, and the prompt. The server
    returns (action_horizon, 14) absolute actions in the hdf5 /action layout, which is already the
    follower command (RealEnv.step convention).
    """

    name = "openpi"
    needs_vtla = False

    def __init__(self, prompt, cameras=("cam_high", "cam_left_wrist", "cam_right_wrist"), resolution=224,
                 policy_hz=50.0):
        self.prompt = prompt
        self.cameras = tuple(cameras)
        self.resolution = resolution
        self.policy_hz = float(policy_hz)

    def start_episode(self, frame):
        pass

    def prepare(self, frame):
        images = {}
        for camera in self.cameras:
            image = frame.images.get(camera)
            if image is None:
                raise RuntimeError("camera {} has not published yet".format(camera))
            image = resize_with_pad(np.asarray(image), self.resolution, self.resolution)
            images[camera] = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
        return {"state": np.asarray(frame.qpos, dtype=np.float64), "images": images, "prompt": self.prompt}

    def decode(self, actions, frame):
        actions = _finite_chunk(actions, 14)
        return actions[:, :14].copy()


class N0VtlaAdapter:
    """N0-VTLA policy, eef (vtla_tactile_posttrain) or joint (vtla_joint_posttrain).

    The observation is run_vtla_policy.py's (observation_from_frames): true-RGB camera and tactile
    images (the LiveStreams snapshot is BGR, so it is converted back here), [baseline, current]
    tactile pairs with the baseline captured at episode start, and the 32-dim padded state. eef
    chunks are IK-solved into joint targets per step with solve_ik (seeded by the previous step; a
    failed step holds the previous target), grippers are clipped to [0, 1] as run_vtla_policy does.
    """

    name = "n0"
    needs_vtla = True

    def __init__(self, prompt, action_space, sources, puppet_bots, policy_hz=None):
        import run_vtla_policy as vtla_client
        from record_episodes_aloha2_vtla import FPS, TACTILE_SENSORS

        if action_space not in ("eef", "joint"):
            raise ValueError("action_space must be 'eef' or 'joint', got {!r}".format(action_space))
        self._client = vtla_client
        self.prompt = prompt
        self.action_space = action_space
        self.sources = dict(sources)
        self.tactile_names = sorted(TACTILE_SENSORS)
        self.puppet_bots = puppet_bots
        self.models = (puppet_bots[0].arm.robot_des, puppet_bots[1].arm.robot_des)
        self.policy_hz = float(policy_hz or FPS)
        self.tactile_baseline = None

    def _rgb_frames(self, frame):
        import cv2

        if frame.vtla is None:
            raise RuntimeError("N0-VTLA needs the VTLA camera/tactile streams (use --format vtla)")
        return {name: cv2.cvtColor(frame.vtla[name]["image"], cv2.COLOR_BGR2RGB)
                for name in set(self.sources.values())}

    def start_episode(self, frame):
        """Capture this episode's zero-contact tactile baseline T_0 (call before any contact)."""
        rgb = self._rgb_frames(frame)
        self.tactile_baseline = {name: rgb[name] for name in self.tactile_names}

    def prepare(self, frame):
        if self.tactile_baseline is None:
            raise RuntimeError("start_episode() must capture the tactile baseline first")
        joints14 = np.concatenate([frame.vtla["joint_left"]["state"], frame.vtla["joint_right"]["state"]])
        state = np.zeros(32, dtype=np.float32)
        if self.action_space == "eef":
            from record_episodes_aloha2_vtla import joints_to_eef
            state[:20] = joints_to_eef(joints14[np.newaxis, :], self.models)[0]
        else:
            state[:14] = joints14
        return self._client.observation_from_frames(
            self.action_space, self._rgb_frames(frame), self.tactile_names, self.sources, self.prompt,
            state, self.tactile_baseline)

    def decode(self, actions, frame):
        layout = self._client.EEF_LAYOUT if self.action_space == "eef" else self._client.JOINT_LAYOUT
        actions = _finite_chunk(actions, 20 if self.action_space == "eef" else 14)
        out = np.empty((len(actions), 14), dtype=np.float64)
        out[:, 6] = np.clip(actions[:, layout["left_gripper"]], 0.0, 1.0)
        out[:, 13] = np.clip(actions[:, layout["right_gripper"]], 0.0, 1.0)
        if self.action_space == "joint":
            out[:, 0:6] = actions[:, layout["left_joints"]]
            out[:, 7:13] = actions[:, layout["right_joints"]]
            return out
        seeds = [list(frame.qpos[0:6]), list(frame.qpos[7:13])]
        for step, row in enumerate(actions):
            for arm, (bot, xyz, rot6d) in enumerate((
                    (self.puppet_bots[0], row[layout["left_xyz"]], row[layout["left_rot6d"]]),
                    (self.puppet_bots[1], row[layout["right_xyz"]], row[layout["right_rot6d"]]))):
                rotation, rot_ok = self._client.rot6d_to_matrix(rot6d)
                theta = solve_ik(bot.arm, xyz, rotation, seeds[arm]) if rot_ok and np.isfinite(xyz).all() else None
                if theta is not None:
                    seeds[arm] = theta
                elif step == 0:
                    logging.warning("%s arm: no IK solution for the first chunk step, holding position",
                                    ("left", "right")[arm])
                out[step, arm * 7:arm * 7 + 6] = seeds[arm]
        return out
