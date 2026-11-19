"""ZMQ + msgpack client for the N0-VTLA fuse server (N0-VTLA-fuse ``scripts/serve_zmq.py``), real ALOHA side.

Used by run_vtla_fused.py (fused models). Same wire protocol as the NeoSim client that was
tested against this server (N0-Foundation remote_infer_lib/n0vtla_zmq_client.py), written for this box's Python 3.8;
needs only numpy, cv2, PIL, msgpack and pyzmq (``pip install pyzmq msgpack`` in the aloha env if missing).

Wire protocol (msgpack over a ZMQ REQ socket; the server is one REP socket)
---------------------------------------------------------------------------
request  {"cmd": "reset"}                                   -> {"status": "ok"}
request  {"cmd": "predict", "state": [32 floats], "prompt",  -> {"status": "ok", "actions": [[32 floats] x H],
          <camera keys>, <tactile keys>, <history keys>}         "infer_time_ms"}
request  {"cmd": "seg_step", "k", "p", "force",             -> {"status": "ok", "fresh": bool (VTLA_DIR2_C_FRESH),
          <tactile keys>, <history keys>}                        "replanned", "e", "delta", "phase", ...}
request  {"cmd": "prepare", <same fields as predict>}      -> {"status": "ok", "actions", "plan": int, "infer_time_ms"}
request  {"cmd": "refine", "plan", "m", "executed" [[14] x m], -> {"status": "ok", "plan", "m", "stale", "fresh",
          "state" (optional), "force" (optional),                 "actions" (absent if stale or fresh), "surprise" [4],
          <tactile keys>, <history keys>}                         "cusum" [4], "pad", "e", "delta", "delta_max",
                                                                  "saturation", "n_reread", "ms", "infer_time_ms"}
camera keys   observation/image (cam_high), observation/wrist_image (cam_left_wrist),
              observation/secondary_image (cam_right_wrist)
tactile keys  observation/<view>  for the 4 model views (left_wrist_left_tactile, ...), the CURRENT frames
history keys  observation/<view>_hist<k>  the frame k control steps before (Direction 3; the first frame of the episode
              while fewer than k steps have passed)
Images go as PNG bytes of the BGR image (lossless; the server cv2-decodes and flips back to RGB), or JPEG with
``codec="jpg"`` (smaller and faster, for the per-step probes). predict / prepare send the NATIVE tactile frames (the
original N0-VTLA client does the same; the server resizes them). seg_step / refine resize the tactile frames to the
model size (224 x 224, ``MODEL_TACTILE_SIZE``) BEFORE encoding by default, with the same function the server's input
transforms use (``resize_with_pad``, copied from n0vtla_client.image_tools), so the model sees the same tensors while
the robot encodes, sends and the server decodes / resizes ~10x fewer pixels per probe; ``tactile_size=None`` sends
native frames (needed only for a server older than 2026-11-18). All images are TRUE RGB on this side (cameras: usb_cam
rgb8 passthrough; tactile: the Xense SDK's rectify bytes unchanged), exactly what the model trained on.

prepare / refine (in-chunk tactile correction, server started with VTLA_REFINE_CKPT): ``prepare`` is a predict that
also builds the server's correction cache for this chunk and names it with a plan id; ``refine`` sends the m rows
actually commanded so far in this chunk plus the live touch and gets the full corrected chunk back (absolute joint
targets, always the ORIGINAL plan + a fresh correction, never cumulative). A refine for a plan the server no longer
holds comes back as status "ok" with ``stale: True`` (not an error), and the reply also carries the C-fresh decision
(``fresh``: end the chunk, no actions). Refines use a short per-call timeout so a lost reply cannot hold the shared
socket (and with it the next prepare) for the full --timeout.

The server keeps ONE tactile baseline per episode: ``reset()`` clears it and the first ``predict()`` after it captures
it, so call reset() right after homing, before any contact (as run_vtla_policy.capture_tactile_baseline does).
The REQ socket is not thread-safe: every call holds ``self.lock``.
"""

import threading

import cv2
import msgpack
import numpy as np
from PIL import Image
import zmq

CAM_WIRE_KEYS = {
    "cam_high": "observation/image",
    "cam_left_wrist": "observation/wrist_image",
    "cam_right_wrist": "observation/secondary_image",
}
TAC_VIEWS = ("left_wrist_left_tactile", "left_wrist_right_tactile",
             "right_wrist_left_tactile", "right_wrist_right_tactile")
STATE_DIM = 32       # canonical padded layout: 14 joint values in [0:14], zeros after (convert_canonical_data.py)
MODEL_TACTILE_SIZE = (224, 224)   # (height, width) every tactile frame reaches the model at (ResizeImages(224, 224))
ACTION_DIM = 14      # real dims of the returned (H, 32) actions: [left j1..6, left gripper, right j1..6, right gripper]


def encode_image(img, codec="png", jpeg_quality=95):
    """HxWx3 RGB uint8 -> encoded bytes of the BGR image (what serve_zmq.py::_decode_image expects)."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("expected an HxWx3 uint8 RGB image, got {} {}".format(arr.dtype, arr.shape))
    bgr = cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2BGR)
    if codec == "png":
        ok, buf = cv2.imencode(".png", bgr)
    elif codec == "jpg":
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    else:
        raise ValueError("codec must be 'png' or 'jpg', got {!r}".format(codec))
    if not ok:
        raise ValueError("image encode failed for shape {}".format(arr.shape))
    return buf.tobytes()


def resize_with_pad(img, height, width, method=Image.BILINEAR):
    """One HWC uint8 RGB frame -> (height, width), aspect kept, zero padding: line for line the server's
    n0vtla_client.image_tools.resize_with_pad / _resize_with_pad_pil (what ResizeImages applies to every model image),
    so a frame resized here is the frame the server would have produced. A frame already at the size is returned as is.
    """
    img = np.asarray(img)
    if img.shape[-3:-1] == (height, width):
        return img
    image = Image.fromarray(img)
    cur_width, cur_height = image.size
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)
    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return np.asarray(zero_image)


def tactile_fields(tacs, hist, codec="png", size=None):
    """{view: RGB} current frames + {k: {view: RGB}} history frames -> wire fields; all 4 views required.
    size (height, width): resize every frame to it before encoding (probes / refines); None = native frames."""
    missing = [v for v in TAC_VIEWS if v not in tacs]
    if missing:
        raise ValueError("tactile views missing: {}".format(missing))

    def enc(img):
        return encode_image(img if size is None else resize_with_pad(img, int(size[0]), int(size[1])), codec)

    out = {"observation/" + v: enc(tacs[v]) for v in TAC_VIEWS}
    for k, frames in (hist or {}).items():
        for v in TAC_VIEWS:
            if v in frames:
                out["observation/{}_hist{}".format(v, int(k))] = enc(frames[v])
    return out


class VtlaZmqClient:
    """REQ client for serve_zmq.py: reset() once per episode, predict() per chunk, seg_step() for Direction 2;
    prepare() per chunk + refine() within it for the in-chunk tactile correction (--refine)."""

    def __init__(self, host, port, timeout_s=30.0):
        self.addr = "tcp://{}:{}".format(host, int(port))
        self._timeout_ms = max(1, int(timeout_s * 1000))
        self._ctx = zmq.Context()
        self._sock = None
        self.lock = threading.Lock()
        self.last_infer_time_ms = None   # the server's inference time of the last predict
        self._open()

    def _open(self):
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        sock.connect(self.addr)
        self._sock = sock

    def _reopen(self):
        if self._sock is not None:
            self._sock.close(linger=0)
        self._open()

    def _set_timeout(self, timeout_ms):
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, timeout_ms)

    def call(self, msg, timeout_s=None):
        """One request / reply under the lock; a timeout recreates the (then stuck) REQ socket and raises.

        ``timeout_s`` overrides the socket timeout for this call only (refines: a lost reply must not hold the lock,
        and with it the next chunk's prepare, for the full default timeout); the default is restored afterwards (a
        timeout reopens the socket, which also restores it)."""
        timeout_ms = self._timeout_ms if timeout_s is None else max(1, int(timeout_s * 1000))
        with self.lock:
            try:
                if timeout_ms != self._timeout_ms:
                    self._set_timeout(timeout_ms)
                self._sock.send(msgpack.packb(msg, use_bin_type=True))
                raw = self._sock.recv()
                if timeout_ms != self._timeout_ms:
                    self._set_timeout(self._timeout_ms)
            except zmq.Again:
                self._reopen()
                raise TimeoutError("no reply from {} within {:.1f} s (cmd={})".format(
                    self.addr, timeout_ms / 1000.0, msg.get("cmd")))
            except zmq.ZMQError:
                self._reopen()
                raise
        reply = msgpack.unpackb(raw, raw=False)
        if not isinstance(reply, dict) or reply.get("status") != "ok":
            message = reply.get("message") if isinstance(reply, dict) else reply
            raise RuntimeError("N0-VTLA server error (cmd={}): {}".format(msg.get("cmd"), message))
        return reply

    def reset(self):
        self.call({"cmd": "reset"})

    @staticmethod
    def _padded_state(state):
        s = np.zeros(STATE_DIM, dtype=np.float32)
        raw = np.asarray(state, dtype=np.float32).reshape(-1)
        s[: min(raw.size, STATE_DIM)] = raw[:STATE_DIM]
        return s.tolist()

    @classmethod
    def _obs_msg(cls, cmd, state, prompt, cams, tacs, tacs_hist, codec):
        """The full chunk-start observation message (predict / prepare)."""
        msg = {"cmd": cmd, "state": cls._padded_state(state)}
        if prompt:
            msg["prompt"] = prompt
        for cam, key in CAM_WIRE_KEYS.items():
            if cam not in cams:
                raise ValueError("camera {} missing".format(cam))
            msg[key] = encode_image(cams[cam], codec)
        msg.update(tactile_fields(tacs, tacs_hist, codec))
        return msg

    @staticmethod
    def _parse_actions(reply):
        """reply["actions"] [[32] x H] -> (H, 14) float64; raises on a wrong shape or non-finite joint values."""
        actions = np.asarray(reply.get("actions"), dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] < ACTION_DIM or not np.isfinite(actions[:, :ACTION_DIM]).all():
            raise RuntimeError("bad actions from server: shape {}".format(actions.shape))
        return actions[:, :ACTION_DIM]

    def predict(self, state, prompt, cams, tacs, tacs_hist=None, codec="png"):
        """-> (H, 14) float64 absolute joint targets. state: the 32-dim padded vector; cams: {cam name: RGB}."""
        reply = self.call(self._obs_msg("predict", state, prompt, cams, tacs, tacs_hist, codec))
        actions = self._parse_actions(reply)
        self.last_infer_time_ms = reply.get("infer_time_ms")
        return actions

    def prepare(self, state, prompt, cams, tacs, tacs_hist=None, codec="png"):
        """predict() that also makes the server cache this chunk for refine() -> ((H, 14) actions, plan id)."""
        reply = self.call(self._obs_msg("prepare", state, prompt, cams, tacs, tacs_hist, codec))
        if "plan" not in reply or reply["plan"] is None:
            raise RuntimeError("prepare reply without a plan id: the server has no corrector (start serve_zmq.py "
                               "with VTLA_REFINE_CKPT, or run without --refine)")
        actions = self._parse_actions(reply)
        self.last_infer_time_ms = reply.get("infer_time_ms")
        return actions, int(reply["plan"])

    def refine(self, plan_id, m, executed, tacs, tacs_hist=None, state=None, force=False, codec="jpg",
               timeout_s=None, tactile_size=MODEL_TACTILE_SIZE):
        """In-chunk correction after m commanded steps of plan ``plan_id`` -> the server's reply dict, with
        ``actions`` parsed to the full corrected (H, 14) chunk, or None when the reply carries none (stale plan or
        C-fresh fired). executed: the (m, 14) rows actually commanded in this chunk; state: the current robot state
        (32-dim padded like predict's), optional. tactile_size: resize the tactile frames to the model size before
        encoding (default; None = native frames)."""
        rows = np.asarray(executed, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[0] != int(m) or rows.shape[1] < ACTION_DIM:
            raise ValueError("executed must be ({}, >= {}) rows, got {}".format(int(m), ACTION_DIM, rows.shape))
        msg = {"cmd": "refine", "plan": int(plan_id), "m": int(m), "executed": rows[:, :ACTION_DIM].tolist()}
        if state is not None:
            msg["state"] = self._padded_state(state)
        if force:
            msg["force"] = True
        msg.update(tactile_fields(tacs, tacs_hist, codec, tactile_size))
        reply = self.call(msg, timeout_s=timeout_s)
        if reply.get("stale") or reply.get("fresh") or reply.get("actions") is None:
            reply["actions"] = None
        else:
            reply["actions"] = self._parse_actions(reply)
        return reply

    def seg_step(self, k, p, tacs, tacs_hist=None, force=False, codec="jpg", tactile_size=MODEL_TACTILE_SIZE):
        """Direction 2 probe at k control steps after the chunk start (plan index p) -> the server's reply dict.
        tactile_size: resize the tactile frames to the model size before encoding (default; None = native frames)."""
        msg = {"cmd": "seg_step", "k": int(k), "p": int(p), "force": bool(force)}
        msg.update(tactile_fields(tacs, tacs_hist, codec, tactile_size))
        return self.call(msg)

    def close(self):
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
        self._ctx.term()
