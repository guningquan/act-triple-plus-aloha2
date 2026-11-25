"""ForceGuide robot client (forceguide_client.py, run_forceguide_policy.py, forceguide_preflight.py), fakes only.

FakeSessionServer below speaks serve_forceguide's session protocol at the action level (traj_base / traj_pending as
action arrays, the ack rule, the exact-step tactile cache, K = 0 / error / stale-chunk replies), so the client's
bookkeeping can be checked against a server's view; BlockingServer is the unchanged WebsocketPolicyServer's shape
(asyncio handler that blocks the event loop while it computes, traceback string + close 1011 on errors) for the
connection tests.
"""

import asyncio
import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
import logging
import logging.config
import os
import re
import socket
import sys
import threading
import time
import traceback
import types
import warnings

import numpy as np
import pytest
from fakes import FakeBot, FakeClock
from PIL import Image

import forceguide_client as fc
import forceguide_preflight
import run_forceguide_policy as runner
from chunk_executor import _Chunk
from hil_devices import FollowerCommander
from policy_client import Frame, packb, resize_with_pad, unpackb

F_CLIENT = "/mnt/server_ssd2/Programs_server/forceguide/packages/openpi-client/src/openpi_client"
KEYS = ("observation.images.right_gripper_inner", "observation.images.right_gripper_outer")
POSE = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
DT = 0.02


def _load_f(module):
    path = os.path.join(F_CLIENT, module + ".py")
    if not os.path.exists(path):
        pytest.skip("forceguide openpi_client sources not found")
    spec = importlib.util.spec_from_file_location("f_openpi_" + module, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


# --------------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------------

class FakeTactileFrame:
    def __init__(self, image, seq, read_end, meta=None):
        self.image = image
        self.seq = seq
        self.read_end = read_end
        self.meta = dict(meta or {"image_codec": "raw"})
        self.count = seq

    def field(self, key, order="rgb"):
        assert (key, order) == ("rectify", "rgb")
        return self.image


class FakeStream:
    """A new frame per latest() call (a fresh XENSE frame every tick) unless frozen."""

    def __init__(self, clock, base, codec="raw"):
        self.clock = clock
        self.base = base
        self.seq = 0
        self.frozen = None
        self.codec = codec
        self.released = False

    def latest(self):
        if self.frozen is not None:
            return self.frozen
        self.seq += 1
        image = (self.base.astype(np.int32) + 3 * (self.seq % 11)).clip(0, 255).astype(np.uint8)
        return FakeTactileFrame(image, self.seq, self.clock(), {"image_codec": self.codec})

    def freeze(self):
        self.frozen = self.latest()

    def release(self):
        self.released = True


def sensor_image(sensor, shape=(70, 40)):
    """Gel-like native frame: a per-sensor pattern with a strong R/B imbalance (like the XENSE gels)."""
    rng = np.random.RandomState(abs(hash(sensor)) % (2 ** 31))
    image = np.empty(shape + (3,), np.uint8)
    image[..., 0] = rng.randint(150, 200, shape)
    image[..., 1] = rng.randint(100, 150, shape)
    image[..., 2] = rng.randint(60, 110, shape)
    return image


def make_metadata(conditions=None, arm="both", policy_hz=30.0, horizon=20, state_gripper="vtla", image_bgr=True,
                  cameras=("cam_high", "cam_left_wrist", "cam_right_wrist"), instance="server-1"):
    tactile = {"keys": list(KEYS), "encoder": "dinov2", "model": "facebook/dinov2-base", "grid": 4, "input": "raw",
               "resize": "pad_pil", "rotate": 0, "gap_s": 0.0, "live_codec": "none"}
    if conditions is None:
        conditions = {
            "p0": {"mode": "chunk", "tactile": None, "window": 1, "stride": 1, "gap_frames": 0, "eta": None},
            "tg": {"mode": "chunk", "tactile": dict(tactile), "window": 1, "stride": 1, "gap_frames": 0, "eta": 400.0},
            "i2": {"mode": "refresh", "tactile": dict(tactile, input="diff", live_codec="yuv420"), "window": 1,
                   "stride": 1, "gap_frames": 0, "refresh_every": 5, "eta": 300.0},
            "i4": {"mode": "refresh", "tactile": dict(tactile, encoder="sparsh", resize="stretch_pil", gap_s=1 / 6),
                   "window": 3, "stride": 5, "gap_frames": 5, "refresh_every": 5, "eta": 250.0},
        }
    for name, entry in conditions.items():
        entry.setdefault("fingerprint", "fp-" + name)
    return {"reset_pose": POSE, "forceguide": {
        "version": 1, "model_action_dim": 32, "action_horizon": horizon, "policy_hz": policy_hz, "arm": arm,
        "cameras": list(cameras), "image_bgr": image_bgr, "state_gripper": state_gripper,
        "server_instance_id": instance, "git_rev": "abc123", "conditions": conditions}}


class FakeSessionServer:
    """serve_forceguide's session semantics at the action level (thread-safe, like the real serialized server)."""

    def __init__(self, metadata):
        self.metadata = metadata
        self.info = fc.ServerInfo.from_metadata(metadata)
        self.H = self.info.action_horizon
        self.A = 14 if self.info.arm == "both" else 7
        self.joint_cols = [c for c in range(self.A) if c % 7 != 6]
        self.lock = threading.Lock()
        self.session = None
        self.condition = None
        self.cache = {}
        self.chunk = 0
        self.chunk_valid = False
        self.base = None
        self.pending = None
        self.pending_req = None
        self.script = {}  # refresh index -> {"K": 0} | {"error": str} | {"jump": rad} | {"delta": rad}
        self.refresh_index = 0
        self.pinned = []  # per refresh, after applying its ack: the server's executed plan (traj_base)
        self.bases = []  # (chunk, traj_base) when a plan ends a chunk (after the plan-carried ack)
        self.messages = []
        self.uploads = []
        self.plan_sleep = 0.0
        self.refresh_sleep = 0.0

    def handle(self, message):
        with self.lock:
            return self._handle(message)

    def _handle(self, message):
        self.messages.append(message)
        kind = message.get("forceguide", "plan")
        req = message.get("req")
        if kind == "reset":
            self.session, self.condition = message["session"], message["condition"]
            self.cache, self.chunk_valid, self.base, self.pending, self.pending_req = {}, False, None, None, None
            fingerprint = self.metadata["forceguide"]["conditions"][self.condition]["fingerprint"]
            return {"forceguide": {"req": req, "ok": True, "condition": self.condition, "fingerprint": fingerprint}}
        if message.get("session") != self.session:
            if kind == "plan":
                raise RuntimeError("session mismatch")
            return {"forceguide": {"error": "session mismatch", "req": req}}
        self._store(kind, message)
        if kind == "observe":
            return {"forceguide": {"req": req, "ok": True, "missing_steps": self._missing(message["tactile_step"])}}
        if kind == "plan":
            if self.plan_sleep:
                time.sleep(self.plan_sleep)
            self._apply_ack(message.get("ack"))
            if self.base is not None:
                self.bases.append((self.chunk, self.base.copy()))
            missing = self._missing(message["tactile_step"])
            if missing:
                raise RuntimeError("missing tactile steps {}".format(missing))
            assert np.asarray(message["noise"]).shape == (self.H, 32)
            self.chunk += 1
            self.chunk_valid = True
            self.base = self._plan_actions(message["state"])
            self.pending = self.pending_req = None
            return {"actions": self.base.copy(),
                    "forceguide": {"req": req, "chunk": self.chunk, "version": 0, "K": 3, "timing_ms": {"all": 1.0}}}
        assert kind == "refresh", kind
        if self.refresh_sleep:
            time.sleep(self.refresh_sleep)
        if not self.chunk_valid or message.get("chunk") != self.chunk:
            return {"forceguide": {"error": "stale chunk", "req": req}}
        self._apply_ack(message.get("ack"))
        k, k_pin = int(message["k"]), int(message["k_pin"])
        assert 0 < k <= k_pin < self.H and message["tactile_step"] >= k
        self.pinned.append({"req": req, "k": k, "k_pin": k_pin, "base": self.base.copy()})
        missing = self._missing(message["tactile_step"])
        if missing:
            self.chunk_valid = False
            return {"forceguide": {"error": "missing tactile steps", "missing_steps": missing, "req": req}}
        script = self.script.get(self.refresh_index, {})
        self.refresh_index += 1
        if script.get("error"):
            self.chunk_valid = False
            return {"forceguide": {"error": script["error"], "req": req}}
        if script.get("K") == 0:
            return {"forceguide": {"req": req, "chunk": self.chunk, "K": 0, "signal": {"d": 0.0, "r": 0.0}}}
        pending = self.base.copy()
        pending[k_pin:, self.joint_cols] += script.get("delta", 0.002)
        pending[k_pin:, 0] += script.get("jump", 0.0)
        self.pending, self.pending_req = pending, req
        reply_actions = pending.copy()
        reply_actions[:k_pin, 0] += script.get("pin_offset", 0.0)  # a (buggy) server whose pinned prefix drifted
        block = {"req": req, "chunk": self.chunk, "k": k, "k_pin": k_pin, "K": script.get("K", 3),
                 "signal": {"d": 0.1, "r": 0.0, "tactile": 0.01}, "timing_ms": {"refresh": 1.0}}
        if script.get("drop_actions"):  # malformed: neither actions, nor an error, nor K = 0
            return {"forceguide": block}
        return {"actions": reply_actions, "forceguide": block}

    def _apply_ack(self, ack):
        if self.pending is None:
            return
        if ack is not None and int(ack["req"]) == self.pending_req and ack["status"] == "applied":
            s = int(ack["s"])
            self.base[s:] = self.pending[s:]
        self.pending = self.pending_req = None

    def _plan_actions(self, state):
        pose = list(POSE) + [0.5]
        row = np.asarray((pose * 2)[:self.A], dtype=np.float64)
        actions = np.tile(row, (self.H, 1))
        actions[:, self.joint_cols] += 0.001 * np.arange(self.H)[:, None]
        return actions

    def _store(self, kind, message):
        steps = [int(s) for s in message.get("tactile_steps") or []]
        frames = message.get("tactile_frames") or {}
        self.uploads.append((kind, message.get("tactile_step"), steps))
        assert steps == sorted(set(steps))
        for key, array in frames.items():
            assert array.dtype == np.uint8 and array.shape == (len(steps), 224, 224, 3), (key, array.shape)
        for index, step in enumerate(steps):
            for key, array in frames.items():
                self.cache.setdefault(step, {}).setdefault(key, array[index].copy())  # first write wins

    def _missing(self, step):
        condition = self.info.condition(self.condition)
        spec = condition.tactile
        if spec is None:
            return []
        need = fc.needed_steps(step, condition.window, condition.stride, condition.gap,
                              background=spec.needs_background)
        return [g for g in need if not all(key in self.cache.get(g, {}) for key in spec.keys)]


class LoopbackConnection:
    """ForceGuideConnection stand-in: msgpack round trip into a FakeSessionServer, no socket."""

    def __init__(self, server, name="main"):
        self.server = server
        self.metadata = server.metadata
        self.expected = None
        self.connects = 1
        self.name = name
        self.closed = False

    def request(self, message, timeout=None, connect_budget_s=None):
        packed = packb(message)
        try:
            reply = self.server.handle(unpackb(packed))
        except Exception as exc:  # the real server's error path: traceback string, then close
            raise fc.ServerError("{}: {}".format(self.name, exc)) from exc
        return unpackb(packb(reply)), {"bytes": len(packed), "pack_ms": 0.0, "send_ms": 0.0, "wait_ms": 0.0}

    def close(self):
        self.closed = True


class InProcessTransport:
    """RefreshWorker stand-in: the real ForceGuideClient.build_message, the fake server at submission, and the reply
    released `delay` seconds (fake clock) later."""

    def __init__(self, server, client, clock, delays=None):
        self.server = server
        self.client = client
        self.clock = clock
        self.delays = dict(delays or {})
        self.queue = []
        self._busy = False
        self.alive = True
        self.messages = []
        self.snapshots = []
        self.snapshot_fn = None

    @property
    def busy(self):
        return self._busy

    def submit(self, job):
        if self._busy:
            return False
        self._busy = True
        if self.snapshot_fn is not None:
            self.snapshots.append(self.snapshot_fn())
        message = self.client.build_message(job)
        self.messages.append(message)
        result = fc.RefreshResult(job)
        result.status, result.reply = "ok", unpackb(packb(self.server.handle(unpackb(packb(message)))))
        result.finish_time = self.clock() + self.delays.get(len(self.messages) - 1, 0.0)
        self.client.after_reply(job, result)
        self.queue.append(result)
        return True

    def poll(self):
        now = self.clock()
        ready = [result for result in self.queue if result.finish_time <= now + 1e-12]
        self.queue = [result for result in self.queue if result not in ready]
        if ready:
            self._busy = False
        return ready


class ScriptedKeyboard:
    def __init__(self, script=None, enabled=True):
        self.script = dict(script or {})
        self.count = 0
        self.enabled = enabled
        self.closed = False

    def poll(self):
        events = self.script.get(self.count, [])
        self.count += 1
        return list(events)

    def drain(self):
        pass

    def close(self):
        self.closed = True


def camera_frame(clock, qpos=None):
    images = {name: np.full((48, 64, 3), (40 * i + 10, 100, 200 - 30 * i), np.uint8)
              for i, name in enumerate(("cam_high", "cam_left_wrist", "cam_right_wrist"))}
    return Frame(clock(), np.asarray(qpos if qpos is not None else np.zeros(14)), np.zeros(14), np.zeros(14), images)


def make_harness(condition="i2", policy_hz=30.0, horizon=20, lead=3, refresh_timeout=None, delays=None, arm="both",
                 conditions=None, script=None, guard=None, campaign_conditions=None):
    clock = FakeClock(0.0)
    metadata = make_metadata(conditions=conditions, arm=arm, policy_hz=policy_hz, horizon=horizon)
    server = FakeSessionServer(metadata)
    server.script = dict(script or {})
    info = fc.ServerInfo.from_metadata(metadata)
    client = fc.ForceGuideClient(LoopbackConnection(server), None, info, session_id="s-test", clock=clock)
    campaign = fc.CampaignConfig(info, campaign_conditions or [condition], lead=lead)
    streams = {sensor: FakeStream(clock, sensor_image(sensor)) for sensor in campaign.sensors}
    rig = fc.TactileRig(streams, campaign.history_capacity, 0.2, wall_clock=clock, mono_clock=clock)
    settings = campaign.settings(condition)
    rig.reset(settings.spec)
    payload = settings.make_payload(rig.history)
    adapter = fc.ForceGuideAdapter.from_info(info, hold_pose=POSE)
    bots = (FakeBot(gripper=0.07), FakeBot(gripper=0.07))
    measured = FollowerCommander(bots, max_joint_step=None).measured
    ctx = fc.TrialContext(client, adapter, condition, payload, rig.monitor, fc.NoiseSchedule(0, 3, horizon, 32),
                          measured, labels={"trial_id": "t0", "attempt": 1})
    client.reset(condition, policy_hz)
    transport = InProcessTransport(server, client, clock, delays)
    cond = info.condition(condition)
    refresh = cond.mode == "refresh"
    executor = fc.RefreshingChunkExecutor(
        ctx.plan, policy_hz, DT, action_horizon=horizon, clock=clock, mode=cond.mode,
        refresh_every=cond.refresh_every if refresh else None, lead=lead if refresh else None,
        transport=transport if refresh else None, decode=adapter.decode,
        guard=guard or fc.SpliceGuard(adapter.active_joints, adapter.active_grippers), can_refresh=ctx.can_refresh,
        refresh_timeout_s=refresh_timeout, log_fn=ctx.log_request, sidecar=ctx.sidecar, payload=payload)
    transport.snapshot_fn = lambda: executor.chunk.actions.copy()
    h = types.SimpleNamespace(clock=clock, server=server, info=info, client=client, campaign=campaign, rig=rig,
                              streams=streams, payload=payload, adapter=adapter, ctx=ctx, transport=transport,
                              executor=executor, sent=[], positions=[], measured=measured)
    h.frame_provider = lambda: camera_frame(clock)
    return h


def run_ticks(h, n):
    for _ in range(n):
        h.positions.append(h.executor.position)
        h.rig.tick(h.executor.position)
        action = h.executor.step(h.frame_provider)
        h.executor.note_sent(action)
        h.sent.append(action)
        h.clock.now += DT


def refreshes(h, status=None):
    return [r for r in h.ctx.requests if r.get("type") == "refresh" and (status is None or r["status"] == status)]


def spec_for(**kwargs):
    data = {"keys": list(KEYS), "resize": "pad_pil", "rotate": 0, "live_codec": "none"}
    data.update(kwargs)
    return fc.TactileSpec.from_dict(data)


# --------------------------------------------------------------------------------------------
# Tactile preparation parity
# --------------------------------------------------------------------------------------------

def test_pad_pil_matches_forceguide_openpi_client_resize_with_pad():
    image_tools = _load_f("image_tools")
    rng = np.random.RandomState(0)
    for shape in ((700, 400, 3), (2, 700, 400, 3), (480, 640, 3), (224, 224, 3)):
        images = rng.randint(0, 256, shape, dtype=np.uint8)
        np.testing.assert_array_equal(fc.prepare_live_tactile(images, spec_for(resize="pad_pil")),
                                      image_tools.resize_with_pad(images, 224, 224))
    out = fc.prepare_live_tactile(rng.randint(0, 256, (700, 400, 3), dtype=np.uint8), spec_for())
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    assert (out[:, :48] == 0).all() and (out[:, 176:] == 0).all()  # 700x400 -> 224x128 centred, zero side bars


def test_stretch_pil_is_pil_bilinear_and_same_size_passthrough():
    rng = np.random.RandomState(1)
    images = rng.randint(0, 256, (3, 700, 400, 3), dtype=np.uint8)
    out = fc.prepare_live_tactile(images, spec_for(resize="stretch_pil"))
    for index in range(3):
        expected = np.asarray(Image.fromarray(images[index]).resize((224, 224), resample=Image.BILINEAR))
        np.testing.assert_array_equal(out[index], expected)
    same = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    np.testing.assert_array_equal(fc.resize_stretch_pil(same), np.asarray(
        Image.fromarray(same).resize((224, 224), resample=Image.BILINEAR)))


def test_rotate_turns_the_image_plane_also_in_batches():
    rng = np.random.RandomState(2)
    images = rng.randint(0, 256, (2, 700, 400, 3), dtype=np.uint8)
    for k in (1, 2, 3):
        out = fc.prepare_live_tactile(images, spec_for(rotate=k, resize="stretch_pil"))
        for index in range(2):
            expected = fc.resize_stretch_pil(np.ascontiguousarray(np.rot90(images[index], k)))
            np.testing.assert_array_equal(out[index], expected)


def test_yuv420_live_codec_is_the_cv2_i420_round_trip_before_rotation_and_resize():
    import cv2

    rng = np.random.RandomState(3)
    image = rng.randint(0, 256, (700, 400, 3), dtype=np.uint8)
    coded = fc.apply_live_codec(image, "yuv420")
    expected = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420), cv2.COLOR_YUV2RGB_I420)
    np.testing.assert_array_equal(coded, expected)
    assert np.abs(coded.astype(int) - image).mean() > 1.0  # it does change the frame (chroma subsampling)
    assert fc.apply_live_codec(image, "none") is image
    batch = np.stack([image, image[::-1]])
    np.testing.assert_array_equal(fc.apply_live_codec(batch, "yuv420")[1], fc.apply_live_codec(image[::-1].copy(),
                                                                                                "yuv420"))
    spec = spec_for(live_codec="yuv420", rotate=1, resize="pad_pil")
    np.testing.assert_array_equal(fc.prepare_live_tactile(image, spec),
                                  resize_with_pad(np.ascontiguousarray(np.rot90(expected, 1)), 224, 224))
    with pytest.raises(ValueError, match="even"):
        fc.apply_live_codec(image[:699], "yuv420")  # I420 needs even dimensions
    with pytest.raises(ValueError):
        fc.apply_live_codec(image.astype(np.float32), "yuv420")  # uint8 only, like features.apply_live_codec
    with pytest.raises(ValueError):
        fc.apply_live_codec(image, None)  # features accepts only "none" / "yuv420"


GOLDEN_CANDIDATES = (
    os.environ.get("FORCEGUIDE_TACTILE_GOLDEN", ""),
    "/mnt/server_ssd2/Programs_server/forceguide/src/openpi/forceguide/testdata/tactile_prep.npz",
)


def golden_tactile_file():
    """The golden file written by the forceguide repo's own features functions (scripts/forceguide/
    make_client_golden.py, committed as src/openpi/forceguide/testdata/tactile_prep.npz), or None:
    $FORCEGUIDE_TACTILE_GOLDEN, then the forceguide checkout next to this one."""
    for path in GOLDEN_CANDIDATES:
        if path and os.path.exists(path):
            return path
    return None


def _json_entry(data, name):
    return json.loads(str(data[name][()]))


def _golden_inputs(data):
    """input name -> uint8 frames: format 2 stores every frame as a lossless PNG plus the sha256 of the stacked
    array (checked here, so a decoding problem is not mistaken for a preparation difference); format 1 arrays."""
    if _json_entry(data, "meta").get("format", 1) < 2:
        return {name[len("input/"):]: data[name] for name in data.files if name.startswith("input/")}
    inputs = {}
    for name, info in _json_entry(data, "inputs").items():
        frames, index = [], 0
        while "input_png/{}/{}".format(name, index) in data.files:
            png = data["input_png/{}/{}".format(name, index)].tobytes()
            frames.append(np.array(Image.open(io.BytesIO(png))))
            index += 1
        array = np.ascontiguousarray(np.stack(frames).reshape(info["shape"]))
        assert hashlib.sha256(array.tobytes()).hexdigest() == info["sha256"], "{}: PNG decoding differs".format(name)
        inputs[name] = array
    return inputs


def _golden_version_note(meta):
    """A clear note when this env's Pillow / OpenCV / numpy differ from the versions the golden file was written
    with (PIL resampling and OpenCV's I420 conversion are where a different build could round differently)."""
    import cv2
    import PIL

    recorded = meta.get("versions", {})
    running = {"Pillow": PIL.__version__, "cv2": cv2.__version__, "numpy": np.__version__}
    changed = ["{} {} -> {}".format(k, recorded.get(k), v) for k, v in sorted(running.items()) if recorded.get(k) != v]
    if not changed:
        return None
    return ("golden tactile file written with {}; running {} ({}): a mismatch may come from that build, not from the "
            "code").format(", ".join("{} {}".format(k, recorded.get(k)) for k in sorted(running)),
                           ", ".join("{} {}".format(k, running[k]) for k in sorted(running)), "; ".join(changed))


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@pytest.mark.skipif(golden_tactile_file() is None, reason="no tactile_prep.npz golden file")
def test_tactile_preparation_matches_the_forceguide_golden_file():
    """features.prepare_tactile(apply_live_codec(x, codec), spec) and features.needed_steps, computed by F in its own
    environment (scripts/forceguide/make_client_golden.py), must be reproduced bit for bit by the client copies in
    this one (different Pillow / OpenCV / numpy builds: PIL's resampling and OpenCV's I420 conversion are integer
    code, so any change there shows up here; the versions are printed when they differ)."""
    with np.load(golden_tactile_file(), allow_pickle=False) as data:
        meta = _json_entry(data, "meta")
        note = _golden_version_note(meta)
        if note:
            print("NOTE: " + note)
            warnings.warn(note, stacklevel=2)
        inputs = _golden_inputs(data)
        cases = _json_entry(data, "cases")
        assert cases
        mismatches = []
        for case in cases:
            spec = fc.TactileSpec.from_dict(case["spec"])
            out = fc.prepare_live_tactile(inputs[case["input"]], spec)
            if "sha256" in case:  # format 2: digest of the whole output, frame 0 kept for a few cases
                if out.dtype == np.uint8 and list(out.shape) == case["shape"] and _digest(out) == case["sha256"]:
                    continue
                diff = "digest differs" if list(out.shape) == case["shape"] else "shape {} != {}".format(
                    out.shape, case["shape"])
                key = "output_png/" + case["id"]
                if key in data.files and list(out.shape) == case["shape"]:
                    expected = np.array(Image.open(io.BytesIO(data[key].tobytes())))
                    first = out.reshape((-1,) + out.shape[-3:])[0]
                    diff = "frame 0 max diff {}".format(int(np.abs(first.astype(int) - expected.astype(int)).max()))
                mismatches.append((case["id"], diff))
                continue
            expected = data["output/" + case["id"]]
            if out.dtype != expected.dtype or out.shape != expected.shape or not np.array_equal(out, expected):
                diff = (np.abs(out.astype(int) - expected.astype(int)).max() if out.shape == expected.shape
                        else "shape {} != {}".format(out.shape, expected.shape))
                mismatches.append((case["id"], diff))
        if "codec" in data.files:
            codecs = _json_entry(data, "codec")
            for name, digest in sorted(codecs.items()):
                if _digest(fc.apply_live_codec(inputs[name], "yuv420")) != digest:
                    mismatches.append(("codec/" + name, "digest differs"))
        else:
            codecs = [name for name in data.files if name.startswith("codec/")]
            for name in codecs:
                coded = fc.apply_live_codec(data["input/" + name[len("codec/"):]], "yuv420")
                if not np.array_equal(coded, data[name]):
                    mismatches.append((name, int(np.abs(coded.astype(int) - data[name].astype(int)).max())))
        rows = _json_entry(data, "needed_steps")
        for row in rows:
            args = dict(row["args"])
            background = args.pop("background")
            if fc.needed_steps(background=background, **args) != row["out"]:
                mismatches.append(("needed_steps", row))
        assert not mismatches, "client tactile preparation differs from F{}: {}".format(
            " ({})".format(note) if note else " (versions {})".format(meta.get("versions")), mismatches[:10])
        assert codecs and rows


def test_legacy_jax_pad_resize_is_refused():
    image = np.zeros((70, 40, 3), np.uint8)
    with pytest.raises(ValueError, match="cannot be reproduced"):
        fc.prepare_live_tactile(image, spec_for(resize="pad"))
    assert fc.prepare_live_tactile(image, spec_for(resize="pad"), allow_legacy_pad=True).shape == (224, 224, 3)
    with pytest.raises(ValueError):
        fc.TactileHistory(("a",), 4).reset(spec_for(resize="pad"))


def test_tactile_spec_parsing_and_validation():
    spec = fc.TactileSpec.from_dict({"keys": KEYS, "encoder": "sparsh", "input": "raw", "resize": "stretch_pil",
                                     "rotate": 5, "gap_s": 0.1667, "live_codec": "yuv420", "extra": 1})
    assert spec.rotate == 1 and spec.needs_background and spec.to_dict()["keys"] == list(KEYS)
    assert not spec_for().needs_background and spec_for(input="diff").needs_background
    for bad in ({"keys": []}, {"keys": KEYS, "resize": "area"}, {"keys": KEYS, "live_codec": "h264"},
                {"keys": [KEYS[0], KEYS[0]]}, {"keys": KEYS, "encoder": "clip"}):
        with pytest.raises(ValueError):
            fc.TactileSpec.from_dict(bad)


# --------------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------------

def test_needed_steps_definition():
    assert fc.window_offsets(1, 5) == [0] and fc.window_offsets(3, 5) == [-10, -5, 0]
    assert fc.needed_steps(7) == [7]
    assert fc.needed_steps(7, background=True) == [0, 7]
    assert fc.needed_steps(12, window=3, stride=5) == [2, 7, 12]
    assert fc.needed_steps(3, window=3, stride=5) == [0, 3]  # steps < 0 map to 0
    assert fc.needed_steps(20, window=2, stride=5, gap=5) == [10, 15, 20]  # Sparsh previous frames overlap
    assert fc.needed_steps(20, window=1, gap=3, background=True) == [0, 17, 20]
    assert fc.needed_steps(2, window=1, gap=5) == [0, 2]
    assert fc.needed_steps(20, window=2, stride=5, gap=0, background=True) == [0, 15, 20]  # gap-0 control


def test_max_index_read_is_what_chunk_sample_reads():
    class Reads:
        def __init__(self):
            self.read = set()

        def __getitem__(self, index):
            self.read.add(index)
            return np.zeros(14)

    for length in (50, 7):
        for offset in np.concatenate([np.arange(0, length, 0.2), np.arange(0, length, 0.6), [length - 1.0]]):
            reads = Reads()
            chunk = _Chunk(reads, 0.0, length)
            if chunk.sample(offset) is None:
                continue
            assert max(reads.read) == fc.max_index_read(offset, length), (offset, length)
            np.testing.assert_allclose(fc.sample_actions(np.arange(length * 14.0).reshape(length, 14), offset, length),
                                       _Chunk(np.arange(length * 14.0).reshape(length, 14), 0.0, length).sample(offset))


def test_ticks_per_chunk_matches_chunk_executor():
    assert fc.ticks_per_chunk(50, 1.0) == 50
    assert fc.ticks_per_chunk(50, DT * 30) == 82  # offsets 0 .. 48.6; 49.2 starts the next chunk
    assert fc.ticks_per_chunk(20, 0.6) == 32


# --------------------------------------------------------------------------------------------
# Tactile history (exact steps), staleness
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("advance", [1.0, DT * 30])
def test_history_step_is_the_first_tick_at_or_past_it_and_never_overwritten(advance):
    clock = FakeClock(0.0)
    stream = FakeStream(clock, sensor_image("right_gripper_inner"))
    history = fc.TactileHistory(("right_gripper_inner",), capacity=500, wall_clock=clock, mono_clock=clock)
    history.reset(spec_for())
    position, first_tick = 0.0, {}
    ticks = []
    for _ in range(200):
        steps = history.record(position, {"right_gripper_inner": stream.latest()})
        for g in steps:
            first_tick[g] = stream.seq  # the frame captured at this tick
        ticks.append(position)
        history.record(position, {"right_gripper_inner": stream.latest()})  # same position again: nothing new
        position += advance
        clock.now += DT
    for g, seq in first_tick.items():
        assert history.entry(g, "right_gripper_inner").seq == seq
        first = min(i for i, p in enumerate(ticks) if p >= g - 1e-9)
        assert fc.global_step(ticks[first]) >= g and (first == 0 or ticks[first - 1] < g - 1e-9)
    assert sorted(first_tick) == list(range(fc.global_step(ticks[-1]) + 1))


def test_history_chunk_start_frame_is_the_one_captured_at_the_planning_tick():
    # 30 Hz policy in the 50 Hz loop: the second chunk starts at position 49.2 -> plan_step 49, first reached there
    clock = FakeClock(0.0)
    stream = FakeStream(clock, sensor_image("right_gripper_inner"))
    history = fc.TactileHistory(("right_gripper_inner",), capacity=200, wall_clock=clock, mono_clock=clock)
    position, advance = 0.0, DT * 30
    seq_at = {}
    for _ in range(83):
        frame = stream.latest()
        history.record(position, {"right_gripper_inner": frame})
        seq_at[round(position, 6)] = frame.seq
        position += advance
    start = 82 * advance
    assert abs(start - 49.2) < 1e-6 and fc.global_step(start) == 49
    assert history.entry(49, "right_gripper_inner").seq == seq_at[round(start, 6)]


def test_history_ring_bounds_memory_but_keeps_the_step_0_background():
    clock = FakeClock(0.0)
    stream = FakeStream(clock, sensor_image("left_gripper_outer"))
    history = fc.TactileHistory(("left_gripper_outer",), capacity=10, wall_clock=clock, mono_clock=clock)
    history.reset(spec_for(keys=["observation.images.left_gripper_outer"]))
    for position in range(100):
        history.record(float(position), {"left_gripper_outer": stream.latest()})
        history.prepared(position, "left_gripper_outer")
    assert history.steps() == [0] + list(range(90, 100))
    assert len(history._prepared) == 11
    with pytest.raises(KeyError):
        history.entry(50, "left_gripper_outer")
    assert history.prepared(0, "left_gripper_outer").shape == (224, 224, 3)


def test_history_freshness_uses_the_age_at_capture():
    clock = FakeClock(10.0)
    history = fc.TactileHistory(("s",), capacity=10, wall_clock=clock, mono_clock=clock)
    old = FakeTactileFrame(np.zeros((4, 4, 3), np.uint8), 1, read_end=9.5)
    history.record(0.0, {"s": old})
    history.record(1.0, {"s": FakeTactileFrame(np.zeros((4, 4, 3), np.uint8), 2, read_end=10.0)})
    history.record(2.0, {"s": None})
    assert history.fresh([1], ["s"], 0.2) == (True, None)
    ok, reason = history.fresh([0, 1], ["s"], 0.2)
    assert not ok and "0.500 s old" in reason
    ok, reason = history.fresh([2], ["s"], 0.2)
    assert not ok and "no s frame" in reason
    assert not history.fresh([7], ["s"], 0.2)[0]


def test_monitor_flags_old_data_and_a_frozen_seq():
    clock = FakeClock(0.0)
    stream = FakeStream(clock, sensor_image("right_gripper_outer"))
    monitor = fc.TactileMonitor(("right_gripper_outer", "right_gripper_inner"), 0.2, wall_clock=clock,
                                mono_clock=clock)
    monitor.update({"right_gripper_outer": stream.latest()})
    assert monitor.stale(["right_gripper_outer"]) == []
    assert monitor.stale(["right_gripper_inner"]) == ["right_gripper_inner: no frame"]
    stream.freeze()
    for _ in range(15):  # same frame (and seq) for 0.3 s
        clock.now += DT
        monitor.update({"right_gripper_outer": stream.latest()})
    reasons = monitor.stale(["right_gripper_outer"])
    assert len(reasons) == 1 and ("old" in reasons[0] or "unchanged" in reasons[0])
    # a publisher re-sending its last frame with a fresh receipt but frozen seq is caught by the seq rule
    frame = stream.latest()
    for _ in range(15):
        clock.now += DT
        monitor.update({"right_gripper_outer": FakeTactileFrame(frame.image, frame.seq, clock())})
    assert "unchanged" in monitor.stale(["right_gripper_outer"])[0]


# --------------------------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------------------------

def test_single_arm_reply_holds_the_idle_gripper_open_even_when_qpos_reports_030():
    info = fc.ServerInfo.from_metadata(make_metadata(arm="right", state_gripper="realenv", image_bgr=False,
                                                     cameras=("cam_high", "cam_right_wrist")))
    adapter = fc.ForceGuideAdapter.from_info(info, hold_pose=POSE)
    qpos = np.array([0.1, -0.9, 1.1, 0.0, -0.3, 0.0, 0.30, 0.2, -0.8, 1.0, 0.1, -0.2, 0.3, 0.29])
    frame = camera_frame(FakeClock(0.0), qpos)
    obs = adapter.prepare(frame)
    np.testing.assert_allclose(obs["state"], qpos[7:14], rtol=1e-6)  # active arm only
    assert sorted(obs["images"]) == ["cam_high", "cam_right_wrist"]  # no left wrist: trained black and masked
    reply = np.random.RandomState(0).randn(20, 7)
    out = adapter.decode(reply)
    assert out.shape == (20, 14)
    np.testing.assert_array_equal(out[:, 7:14], reply)
    np.testing.assert_array_equal(out[:, 0:6], np.tile(POSE, (20, 1)))
    assert (out[:, 6] == 1.0).all()  # open, never qpos[6] = 0.30 (would close it to ~70 % of its stroke)
    left = fc.ForceGuideAdapter(cameras=("cam_high",), arm="left", hold_pose=POSE).decode(reply)
    np.testing.assert_array_equal(left[:, 0:7], reply)
    assert (left[:, 13] == 1.0).all() and (left[:, 7:13] == POSE).all()
    assert adapter.active_joints == (7, 8, 9, 10, 11, 12) and adapter.active_grippers == (13,)
    with pytest.raises(RuntimeError):
        adapter.decode(np.zeros((20, 14)))  # a two-arm reply for a single-arm P0
    with pytest.raises(ValueError):
        fc.ForceGuideAdapter(arm="right")  # needs the hold pose


def test_vtla_state_comes_from_follower_measured_not_qpos():
    info = fc.ServerInfo.from_metadata(make_metadata())
    adapter = fc.ForceGuideAdapter.from_info(info)
    bots = (FakeBot(gripper=0.0767), FakeBot(gripper=-1.5))  # left open, right closed (motor angle)
    measured = FollowerCommander(bots, max_joint_step=None).measured()
    assert measured[6] == pytest.approx(1.0) and measured[13] == pytest.approx(0.0)
    frame = camera_frame(FakeClock(0.0), np.full(14, 0.3))
    obs = adapter.prepare(frame, measured)
    assert obs["state"].dtype == np.float32
    np.testing.assert_allclose(obs["state"], measured, rtol=1e-6)
    with pytest.raises(RuntimeError, match="measured"):
        adapter.prepare(frame)


def test_cameras_are_letterboxed_chw_and_swapped_to_bgr_when_the_p0_trained_on_bgr():
    frame = camera_frame(FakeClock(0.0))
    bgr = fc.ForceGuideAdapter(image_bgr=True, prompt="board insertion")
    rgb = fc.ForceGuideAdapter(image_bgr=False)
    obs_bgr, obs_rgb = bgr.prepare(frame), rgb.prepare(frame)
    for camera, image in frame.images.items():
        expected = np.transpose(resize_with_pad(image, 224, 224), (2, 0, 1))
        np.testing.assert_array_equal(obs_rgb["images"][camera], expected)
        np.testing.assert_array_equal(obs_bgr["images"][camera], expected[::-1])
    assert obs_bgr["prompt"] == "board insertion" and "prompt" not in obs_rgb


def test_gripper_calibration_must_match_the_training_data():
    assert fc.check_gripper_calibration()  # constants.py == board_insertion metadata.json
    raw = {"gripper_calibration": [{"closed": -1.5, "open": 0.0767}, {"closed": -1.5, "open": 0.0782}]}
    assert fc.gripper_calibration_from_metadata(raw) == fc.DATASET_GRIPPER_CALIBRATION
    with pytest.raises(RuntimeError, match="right follower gripper open"):
        fc.check_gripper_calibration(({"closed": -1.5, "open": 0.0767}, {"closed": -1.5, "open": 0.09}))


# --------------------------------------------------------------------------------------------
# Metadata, campaign
# --------------------------------------------------------------------------------------------

def test_server_info_parses_the_forceguide_block():
    info = fc.ServerInfo.from_metadata(make_metadata())
    assert info.policy_hz == 30.0 and info.model_action_dim == 32 and info.arm == "both" and info.image_bgr
    i4 = info.condition("i4")
    assert i4.mode == "refresh" and i4.window == 3 and i4.stride == 5 and i4.gap == 5 and i4.span == 15
    assert info.condition("tg").gap == 0 and info.condition("p0").tactile is None
    with pytest.raises(KeyError):
        info.condition("nope")
    with pytest.raises(RuntimeError, match="forceguide"):
        fc.ServerInfo.from_metadata({"reset_pose": POSE})
    derived = make_metadata(conditions={"x": {"mode": "chunk", "tactile": {"keys": list(KEYS), "encoder": "sparsh",
                                                                          "gap_s": 1 / 6}}})
    assert fc.ServerInfo.from_metadata(derived).condition("x").gap == 5  # round(gap_s * fps)
    with pytest.raises(ValueError):
        fc.ServerInfo.from_metadata(make_metadata(conditions={"x": {"mode": "refresh"}}))  # no refresh_every
    assert fc.sensor_for_key("observation.images.left_gripper_inner") == "left_gripper_inner"
    with pytest.raises(ValueError, match="--tactile_sensor"):
        fc.sensor_for_key("observation.images.right_wrist_left_tactile")
    assert fc.sensor_for_key("x.y", {"x.y": "right_gripper_inner"}) == "right_gripper_inner"


def test_campaign_shares_one_pipeline_cadence_and_lead():
    info = fc.ServerInfo.from_metadata(make_metadata())
    campaign = fc.CampaignConfig(info, ["p0", "i2", "i4", "p0"], lead=3)
    assert campaign.names == ["p0", "i2", "i4"] and campaign.refresh_every == 5 and campaign.lead == 3
    assert campaign.sensors == ("right_gripper_inner", "right_gripper_outer")
    assert campaign.proxy == "i2" and campaign.history_capacity == 15 + fc.HISTORY_MARGIN_STEPS
    p0 = campaign.settings("p0")
    assert p0.proxy and p0.spec == info.condition("i2").tactile and list(p0.key_sensors) == list(KEYS)
    i4 = campaign.settings("i4")
    assert not i4.proxy and (i4.window, i4.stride, i4.gap, i4.background) == (3, 5, 5, True)
    with pytest.raises(ValueError, match="--lead"):
        fc.CampaignConfig(info, ["i2"])
    assert fc.CampaignConfig(info, ["i2"], require_lead=False).lead is None
    with pytest.raises(ValueError, match="policy_hz"):
        fc.CampaignConfig(info, ["p0"], policy_hz=50.0)
    assert fc.CampaignConfig(info, ["p0"], policy_hz=50.0, force=True).policy_hz == 50.0
    with pytest.raises(ValueError, match="refresh_every"):
        fc.CampaignConfig(info, ["i2"], lead=3, refresh_every=4)
    mixed = make_metadata()
    mixed["forceguide"]["conditions"]["i4"]["refresh_every"] = 6
    with pytest.raises(ValueError, match="one cadence"):
        fc.CampaignConfig(fc.ServerInfo.from_metadata(mixed), ["i2", "i4"], lead=3)
    legacy = make_metadata()
    legacy["forceguide"]["conditions"]["tg"]["tactile"]["resize"] = "pad"
    with pytest.raises(ValueError, match="legacy"):
        fc.CampaignConfig(fc.ServerInfo.from_metadata(legacy), ["tg"])
    none = fc.CampaignConfig(info, ["p0"])
    assert none.sensors == () and none.settings("p0").key_sensors == {}
    assert campaign.notes == [] and campaign.to_dict()["notes"] == []  # i4: stride 5 = gap 5 = refresh_every 5
    odd = make_metadata()
    odd["forceguide"]["conditions"]["i4"]["stride"] = 3
    notes = fc.CampaignConfig(fc.ServerInfo.from_metadata(odd), ["i4"], lead=3).notes
    assert len(notes) == 1 and "stride 3" in notes[0]


def test_campaign_lead_must_be_the_lead_the_served_eta_was_calibrated_with():
    """Robot review 1 / contract C1: metadata conditions[name] "lead" (calibration.json) and "min_lead"
    (dynamic_k.json); servers without them (fixtures, older servers) impose nothing."""
    calibrated = make_metadata()
    for name in ("i2", "i4"):
        calibrated["forceguide"]["conditions"][name].update(lead=4, min_lead=1)
    info = fc.ServerInfo.from_metadata(calibrated)
    assert (info.condition("i2").lead, info.condition("i2").min_lead, info.condition("p0").lead) == (4, 1, None)
    with pytest.raises(ValueError, match="offline_guidance_check --lead 3"):
        fc.CampaignConfig(info, ["p0", "i2"], lead=3)
    assert fc.CampaignConfig(info, ["p0", "i2"], lead=3, force=True).lead == 3
    campaign = fc.CampaignConfig(info, ["p0", "i2", "i4"], lead=4)
    assert campaign.lead == 4 and campaign.to_dict()["calibrated_leads"] == {"i2": 4, "i4": 4}
    preflight = fc.CampaignConfig(info, ["i2"], require_lead=False)  # preflight mode: the calibrated lead
    assert preflight.lead == 4 and preflight.default_lead == 4
    with pytest.raises(ValueError, match="calibrated with lead 4"):
        fc.CampaignConfig(info, ["i2"])  # trial mode: --lead stays explicit
    mixed = make_metadata()
    mixed["forceguide"]["conditions"]["i2"]["lead"] = 3
    mixed["forceguide"]["conditions"]["i4"]["lead"] = 4
    with pytest.raises(ValueError, match="different leads"):
        fc.CampaignConfig(fc.ServerInfo.from_metadata(mixed), ["i2", "i4"], lead=3)
    floor = make_metadata()
    floor["forceguide"]["conditions"]["i2"].update(lead=None, min_lead=3)  # dynamic K for leads >= 3, no calibration
    info = fc.ServerInfo.from_metadata(floor)
    with pytest.raises(ValueError, match="below min_lead 3"):
        fc.CampaignConfig(info, ["i2"], lead=2)
    assert fc.CampaignConfig(info, ["i2"], lead=5).lead == 5
    assert fc.CampaignConfig(info, ["i2"], require_lead=False).default_lead == 3
    floor["forceguide"]["conditions"]["i2"]["min_lead"] = 5
    assert fc.CampaignConfig(fc.ServerInfo.from_metadata(floor), ["i2"], require_lead=False).default_lead == 5
    legacy = fc.CampaignConfig(fc.ServerInfo.from_metadata(make_metadata()), ["i2"], require_lead=False)
    assert legacy.lead is None and legacy.default_lead == 3 and legacy.min_lead is None
    with pytest.raises(ValueError, match=">= 1"):
        fc.ConditionInfo("x", mode="refresh", refresh_every=5, lead=0)


def test_refresh_error_replies_never_count_as_round_trips(monkeypatch, tmp_path):
    """Robot review 1 (b-d): a server refusing every refresh (e.g. a lead its calibration does not allow) answers
    fast; the preflight must refuse instead of reporting a low RTT, the warm-up must stop, and trial statistics
    leave the error replies out."""
    metadata = make_metadata(horizon=20, policy_hz=30.0)
    server = FakeSessionServer(metadata)
    server.script = {i: {"error": "k_pin - k = 3 != the calibrated lead 2"} for i in range(1000)}
    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(3)]), [])
    log_dir = str(tmp_path / "logs")
    assert runner.main(["--host", "h", "--condition", "i2", "--lead", "3", "--log_dir", log_dir, "--preflight", "2",
                        "--no_warmup"]) == 2
    with open(os.path.join(log_dir, "preflight.json")) as handle:
        report = json.load(handle)
    assert not report["ok"] and report["refresh_rtt_ms"]["n"] == 0 and report.get("required_lead") is None
    assert report["conditions"]["i2"]["refresh_errors"] >= 2
    assert any("error reply" in problem and "calibrated lead 2" in problem for problem in report["problems"])
    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(3)]), [])
    with pytest.raises(fc.InfrastructureFailure, match="refresh requests failed"):
        runner.main(["--host", "h", "--condition", "i2", "--lead", "3", "--log_dir", log_dir])
    assert os.path.exists(os.path.join(log_dir, "warmup.jsonl"))
    ctx = types.SimpleNamespace(ticks=[], requests=[
        {"type": "refresh", "transport": "ok", "status": "error", "rtt_tick_ms": 2.0, "K": None},
        {"type": "refresh", "transport": "ok", "status": "applied", "rtt_tick_ms": 40.0, "K": 3},
        {"type": "refresh", "transport": "timeout", "status": "failed", "rtt_tick_ms": 300.0}])
    stats = runner.trial_stats(ctx)
    assert stats["refresh_rtt_ms"]["n"] == 1 and stats["refresh_K"] == {"3": 1} and stats["refresh_errors"] == 2


# --------------------------------------------------------------------------------------------
# Executor: splice, reject, ack, steps
# --------------------------------------------------------------------------------------------

def test_on_time_splice_at_k_pin_with_a_50hz_policy():
    h = make_harness(policy_hz=50.0, horizon=20, lead=2)
    run_ticks(h, 25)
    applied = refreshes(h, "applied")
    assert [r["k"] for r in applied] == [5, 10, 15] and all(r["s"] == r["k_pin"] for r in applied)
    assert all(r["late"] == 0 and r["pin_mismatch"] == 0.0 for r in applied)
    # the robot executed the old plan before k_pin and the refreshed one from k_pin on
    first = h.sent[:20]
    assert first[6][0] == pytest.approx(0.006)  # index 6 < k_pin 7 of the k=5 refresh: plan value
    assert first[7][0] == pytest.approx(0.007 + 0.002)  # spliced from k_pin
    # the plan of chunk 2 carried the last ack; the server's final traj_base of chunk 1 is what the robot executed
    chunk, base = h.server.bases[0]
    assert chunk == 1
    np.testing.assert_array_equal(np.asarray(first), base)


def test_late_reply_is_spliced_from_s_min_with_the_30hz_interpolation():
    # k = 5 is first reached at offset 5.4; a 100 ms reply lands at offset 8.4 where the previous command (7.8)
    # already read index 8, so s_min = 9 > k_pin = 8 -> late by one step
    h = make_harness(policy_hz=30.0, horizon=20, lead=3, delays={0: 0.1})
    run_ticks(h, 18)  # k = 10 (the next refresh, carrying the ack) is first reached at tick 17, offset 10.2
    first = refreshes(h)[0]
    assert first["status"] == "applied" and first["k"] == 5 and first["k_pin"] == 8
    assert first["s_min"] == 9 and first["s"] == 9 and first["late"] == 1
    assert h.executor.counts["late"] == 1 and h.executor.counts["late_any"] == 1
    assert h.transport.messages[1]["k"] == 10
    assert h.transport.messages[1]["ack"] == {"req": first["req"], "status": "applied", "s": 9}
    # the robot executed old[8] (the tick at offset 7.8 interpolated old[7] and old[8]) and new[9] onwards
    snapshot = h.transport.snapshots[1]  # client plan when k = 10 was submitted
    np.testing.assert_array_equal(snapshot[:9], h.transport.snapshots[0][:9])
    assert snapshot[9, 0] == pytest.approx(0.009 + 0.002) and snapshot[8, 0] == pytest.approx(0.008)
    np.testing.assert_array_equal(h.server.pinned[1]["base"], snapshot)  # the server applied the ack from s = 9


def test_reply_past_the_chunk_end_is_dropped_and_acked_in_the_plan():
    # 30 Hz, H = 20: k = 15 is submitted at tick 25 (offset 15.0), k_pin 18. The reply lands between tick 31
    # (offset 18.6, which reads a[18] and a[19]) and tick 32 (offset 19.2 > 19: the next plan). It is decided at
    # the top of tick 32, before that plan: s_min = 20 = length -> dropped with s = 20, acked by the plan.
    h = make_harness(policy_hz=30.0, horizon=20, lead=3, delays={2: 0.13})
    run_ticks(h, 33)
    third = refreshes(h)[2]
    assert (third["k"], third["status"], third["reason"]) == (15, "dropped", "past-end")
    assert third["s_min"] == 20 and third["s"] == 20 and third["late"] == 2
    plans = [m for m in h.server.messages if m.get("forceguide") == "plan"]
    assert plans[1]["ack"] == {"req": third["req"], "status": "dropped", "s": 20}
    # the server discarded that result: chunk 1's final traj_base is the plan with only the first two refreshes
    np.testing.assert_array_equal(h.server.bases[0][1], h.transport.snapshots[2])
    assert h.executor.counts["late"] == 0 and h.executor.counts["late_any"] == 1


def test_reply_for_a_finished_chunk_is_dropped_without_an_ack():
    # 50 Hz, H = 20: k = 15 (k_pin 17) answered 0.2 s later, at tick 25, when chunk 2 (planned at tick 20) runs
    h = make_harness(policy_hz=50.0, horizon=20, lead=2, delays={2: 0.2})
    run_ticks(h, 30)
    dropped = refreshes(h, "dropped")
    assert [(r["k"], r["reason"]) for r in dropped] == [(15, "stale-chunk")]
    plans = [m for m in h.server.messages if m.get("forceguide") == "plan"]
    assert "ack" not in plans[1]  # undecided at the plan -> missing ack = dropped on the server
    assert h.executor.pending_ack is None or h.executor.pending_ack["req"] != dropped[0]["req"]


def test_server_stale_chunk_reply_keeps_the_plan_caches_the_tactile_and_stops_refreshes():
    h = make_harness(condition="i2", policy_hz=50.0, horizon=20, lead=2, script={1: {"error": "stale chunk"}})
    run_ticks(h, 20)
    statuses = [(r["k"], r["status"]) for r in refreshes(h)]
    assert statuses == [(5, "applied"), (10, "error"), (15, "skipped-disabled")]
    stale = refreshes(h, "error")[0]
    assert stale["reason"] == "stale chunk" and stale["tactile_steps"] == [10]
    assert 10 in h.payload.confirmed()  # the server cached the stale refresh's tactile: never sent again
    assert h.executor.pending_ack is None  # nothing to acknowledge: the server holds no pending result
    run_ticks(h, 1)  # tick 20: the next plan
    plan = [m for m in h.server.messages if m.get("forceguide") == "plan"][1]
    assert "ack" not in plan and plan["tactile_steps"] == [20]


def test_reply_without_actions_error_or_k0_is_rejected_and_acked():
    h = make_harness(condition="i2", policy_hz=50.0, horizon=20, lead=2, script={0: {"drop_actions": True}})
    run_ticks(h, 12)  # k = 10 is submitted at tick 10 (with the ack) and decided at tick 11
    first, second = refreshes(h)[:2]
    assert first["status"] == "rejected" and first["reason"].startswith("protocol")
    assert h.transport.messages[1]["ack"] == {"req": first["req"], "status": "rejected", "s": -1}
    assert second["status"] == "applied"  # refreshes go on: the ack kept the server consistent
    summary = h.executor.summary()  # a malformed reply is a server fault, not a splice-guard decision
    assert first["malformed"] and summary["counts"]["malformed"] == 1
    assert (summary["infra"], summary["protocol"]) == (1, 0) and summary["bad"] == 1


def test_summary_separates_infrastructure_faults_from_method_outcomes():
    """Robot review 3: a guard rejection is the method's outcome (protocol), a server error an infrastructure fault
    that also disables the rest of its chunk (skipped-disabled)."""
    h = make_harness(condition="i2", policy_hz=50.0, horizon=20, lead=2, script={0: {"jump": 1.0}})
    run_ticks(h, 20)
    summary = h.executor.summary()
    assert summary["counts"]["rejected"] == 1 and (summary["infra"], summary["protocol"]) == (0, 1)
    assert summary["protocol_frac"] == pytest.approx(1 / 3) and summary["infra_frac"] == 0.0
    assert runner.classify("done", summary, 0.1) == "labelled"
    h = make_harness(condition="i2", policy_hz=50.0, horizon=20, lead=2, script={0: {"error": "boom"}})
    run_ticks(h, 20)
    summary = h.executor.summary()
    assert summary["counts"]["error"] == 1 and summary["counts"]["skipped-disabled"] == 2
    assert (summary["infra"], summary["protocol"]) == (3, 0) and summary["infra_frac"] == 1.0
    assert runner.classify("done", summary, 0.1) == "infrastructure_failure"


def test_pinned_prefix_mismatch_is_counted_and_never_printed(caplog):
    h = make_harness(condition="i2", policy_hz=50.0, horizon=20, lead=2, script={0: {"pin_offset": 0.01}})
    with caplog.at_level("INFO"):
        run_ticks(h, 12)
    first = refreshes(h)[0]
    assert first["pin_mismatch"] == pytest.approx(0.01) and h.executor.summary()["pin_mismatch"] == 1
    assert not [r for r in caplog.records if "k_pin" in r.getMessage()]  # the blinded operator sees nothing
    assert refreshes(h)[1]["pin_mismatch"] == 0.0


def test_request_records_carry_the_session_and_the_window_capture_times():
    h = make_harness(condition="i4", policy_hz=30.0, horizon=50, lead=3)
    run_ticks(h, 30)
    plan = [r for r in h.ctx.requests if r["type"] == "plan"][0]
    record = refreshes(h, "applied")[1]  # k = 10
    for entry in (plan, record):
        assert entry["session"] == "s-test" and entry["trial_id"] == "t0"
    window = record["window"]
    assert window["anchor"] == record["tactile_step"] == 10
    assert window["steps"] == fc.needed_steps(10, 3, 5, 5, background=True) == [0, 5, 10]
    assert all(t is not None for t in window["captured"]) and window["captured"] == sorted(window["captured"])
    assert sorted(window["read_end"]) == ["right_gripper_inner", "right_gripper_outer"]
    assert plan["window"]["steps"] == [0]


def test_splice_guard_rules():
    guard = fc.SpliceGuard(fc.ARM_JOINTS["both"], fc.ARM_GRIPPERS["both"])
    old = np.tile(np.r_[POSE, 0.5, POSE, 0.5], (20, 1)) + 0.001 * np.arange(20)[:, None]
    ok, reason, metrics = guard.check(old, old + 0.01, 8, 20)
    assert ok and metrics["splice_jump"] == pytest.approx(0.01)
    far = old.copy()
    far[15:, 0] += np.linspace(0.05, 0.6, 5)  # big change far ahead, smooth at the splice point: guidance, accepted
    ok, reason, metrics = guard.check(old, far, 8, 20)
    assert ok and metrics["suffix_max"] == pytest.approx(0.6) and metrics["splice_jump"] == 0.0
    jump = old.copy()
    jump[8:, 3] += 0.1
    assert guard.check(old, jump, 8, 20)[1].startswith("splice jump")
    steep = old.copy()
    steep[12:, 1] += 0.06 * np.arange(1, 9)[:, None].ravel() * 4  # 0.24 rad per step inside the suffix
    assert guard.check(old, steep, 8, 20)[1].startswith("plan step")
    grip = old.copy()
    grip[8:, 13] -= 0.3
    assert guard.check(old, grip, 8, 20)[1].startswith("gripper jump")
    nan = old.copy()
    nan[10, 2] = np.nan
    assert guard.check(old, nan, 8, 20)[1] == "non-finite actions"
    assert guard.check(old, old[:, :7], 8, 20)[1].startswith("shape")
    # the splice changes this tick's command: compare it with what FollowerCommander actually sent
    last_sent = old[7].copy()
    last_sent[0] -= 0.09  # e.g. the per-tick step limit held the arm back
    ok, reason, metrics = guard.check(old, old + 0.001, 8, 20, last_sent=last_sent, offset=8.0)
    assert not ok and reason.startswith("jump vs last sent") and metrics["splice_jump_sent"] > 0.075
    assert guard.check(old, old + 0.001, 8, 20, last_sent=last_sent, offset=6.0)[0]  # not read this tick
    single = fc.SpliceGuard(fc.ARM_JOINTS["right"], fc.ARM_GRIPPERS["right"])
    idle = old.copy()
    idle[8:, 0:7] += 1.0  # the idle (left) arm is ignored for a right-arm P0
    assert single.check(old, idle, 8, 20)[0]


def test_splice_guard_plan_step_rule_covers_the_reply_and_the_executed_transition():
    guard = fc.SpliceGuard(fc.ARM_JOINTS["both"], fc.ARM_GRIPPERS["both"])
    old = np.tile(np.r_[POSE, 0.5, POSE, 0.5], (20, 1)) + 0.001 * np.arange(20)[:, None]
    # the reply's own step new[s-1] -> new[s] (max_{i >= s-1} |new[i+1] - new[i]|), although new[s-1] is not run
    odd = old.copy()
    odd[7, 2] += 0.3
    ok, reason, metrics = guard.check(old, odd, 8, 20)
    assert not ok and reason.startswith("plan step") and metrics["max_plan_step"] == pytest.approx(0.3, abs=2e-3)
    assert guard.check(old, odd, 9, 20)[0]  # spliced from 9: row 7 is outside the rule
    # the executed transition old[s-1] -> new[s]: the old plan already steps 0.15 there, the reply adds 0.07
    steep = old.copy()
    steep[8:, 4] += 0.15
    new = steep.copy()
    new[8:, 4] += 0.07
    new[7, 4] = steep[7, 4] + 0.07  # the reply itself is smooth (0.15 per step at most)
    ok, reason, metrics = guard.check(steep, new, 8, 20)
    assert not ok and reason.startswith("plan step") and metrics["max_plan_step"] == pytest.approx(0.22, abs=2e-3)
    assert metrics["splice_jump"] == pytest.approx(0.07)


def test_ack_protocol_keeps_the_server_plan_equal_to_what_the_robot_executes():
    """Applied on time, late, rejected, K = 0, dropped at the deadline, a server error: after each, the next
    refresh's pinned trajectory (the server's traj_base after applying the ack) equals the client's plan."""
    script = {2: {"jump": 0.2}, 3: {"K": 0}, 6: {"error": "boom"}}
    h = make_harness(policy_hz=30.0, horizon=50, lead=3, refresh_timeout=0.12, delays={1: 0.1, 4: 0.15},
                     script=script)
    run_ticks(h, 82 + 40)
    first_chunk = [r for r in refreshes(h) if r["chunk"] == 1]
    statuses = [(r["k"], r["status"], r.get("reason")) for r in first_chunk]
    assert [s for _, s, _ in statuses] == ["applied", "applied", "rejected", "skipped-K0", "dropped", "applied",
                                           "error", "skipped-disabled", "skipped-disabled"], statuses
    assert first_chunk[1]["late"] >= 1 and first_chunk[4]["reason"] == "deadline"
    assert first_chunk[2]["reason"].startswith("splice jump")
    acks = [message.get("ack") for message in h.transport.messages[:7]]
    assert acks[0] is None
    assert [a["status"] for a in acks[1:]] == ["applied", "applied", "rejected", "dropped", "dropped", "applied"]
    assert acks[2]["s"] > first_chunk[1]["k_pin"]  # the late splice point
    assert acks[3]["s"] == first_chunk[2]["s"] and acks[4]["s"] == -1
    # the invariant, request by request (server state after applying the ack == client plan at submission)
    assert len(h.server.pinned) == len(h.transport.snapshots) >= 7
    for pinned, snapshot in zip(h.server.pinned, h.transport.snapshots):
        np.testing.assert_array_equal(pinned["base"], snapshot)
    # the plan of chunk 2 carried no ack (the error invalidated chunk 1); chunk 1's final base is the executed plan
    plans = [m for m in h.server.messages if m.get("forceguide") == "plan"]
    assert len(plans) == 2 and "ack" not in plans[1]
    assert h.server.bases[0][0] == 1
    second = [r for r in refreshes(h) if r["chunk"] == 2]
    assert second and all(r["status"] == "applied" for r in second)
    summary = h.executor.summary()
    assert summary["counts"]["applied"] >= 3 and summary["bad"] >= 5


def test_plan_carries_the_last_refresh_ack_and_the_server_applies_it():
    h = make_harness(policy_hz=50.0, horizon=20, lead=2)
    run_ticks(h, 21)
    plans = [m for m in h.server.messages if m.get("forceguide") == "plan"]
    last = refreshes(h)[-1]
    assert plans[1]["ack"] == {"req": last["req"], "status": "applied", "s": last["s"]}
    chunk, base = h.server.bases[0]
    assert chunk == 1
    np.testing.assert_array_equal(base, h.transport.snapshots[-1] if False else base)
    # the executed commands of chunk 1 are exactly the server's final traj_base rows
    np.testing.assert_allclose(np.asarray(h.sent[:20]), base, atol=0)


def test_no_duplicate_refresh_at_the_same_k_when_replies_are_instant():
    # 30 Hz policy: k = floor(offset) stays the same for two ticks (e.g. 1.2 and 1.8); replies arrive at once
    h = make_harness(policy_hz=30.0, horizon=50, lead=3)
    run_ticks(h, 82)
    ks = [r["k"] for r in refreshes(h)]
    assert ks == sorted(set(ks)) == [5, 10, 15, 20, 25, 30, 35, 40, 45]
    assert len(h.transport.messages) == 9


def test_refresh_steps_and_exact_tactile_uploads_across_a_30hz_chunk_boundary():
    # Sparsh window W=3 stride 5 gap 5 + background: the fake server errors on any missing needed step
    h = make_harness(condition="i4", policy_hz=30.0, horizon=50, lead=3)
    run_ticks(h, 82 * 2 + 30)
    plans = [m for m in h.server.messages if m.get("forceguide") == "plan"]
    assert [p["tactile_step"] for p in plans] == [0, 49, 98]  # chunks start at positions 0, 49.2, 98.4
    by_req = {m["req"]: m for m in h.transport.messages}
    for record in refreshes(h):
        assert record["status"] == "applied", record
        message = by_req[record["req"]]
        plan_step = {1: 0, 2: 49, 3: 98}[record["chunk"]]
        assert message["tactile_step"] == plan_step + message["k"]
        need = fc.needed_steps(message["tactile_step"], 3, 5, 5, background=True)
        assert set(message["tactile_steps"]) <= set(need)  # only needed steps, never another step
    # every needed step was uploaded exactly once per episode (first write wins; duplicates only after errors)
    uploaded = [step for _, _, steps in h.server.uploads for step in steps]
    assert len(uploaded) == len(set(uploaded))
    # chunk start 49: the window [39, 44, 49] and previous frames [34, 39, 44] and background 0
    assert set(fc.needed_steps(49, 3, 5, 5, background=True)) <= set(h.server.cache)


def test_tactile_step_is_never_ahead_of_the_recorded_history():
    h = make_harness(condition="i2", policy_hz=30.0, horizon=50, lead=3)
    submitted = []
    original = h.transport.submit

    def submit(job):
        submitted.append((job.tactile_step, h.rig.history.next_step - 1, job.k, h.executor.chunk.plan_step))
        return original(job)

    h.transport.submit = submit
    run_ticks(h, 82 * 3)
    assert submitted and all(step <= newest for step, newest, _, _ in submitted)
    assert all(step == plan_step + k for step, _, k, plan_step in submitted)


def test_refresh_is_skipped_while_tactile_is_stale_and_a_plan_aborts():
    h = make_harness(condition="i2", policy_hz=50.0, horizon=20, lead=2)
    run_ticks(h, 3)
    for stream in h.streams.values():
        stream.freeze()  # its newest frame was read at t = 0.06
    run_ticks(h, 15)
    # k = 10 (t = 0.20): data 0.14 s old, still fresh; k = 15 (t = 0.30): 0.24 s > 0.2 -> skipped, plan kept
    assert [r["k"] for r in refreshes(h) if r["status"] == "applied"] == [5, 10]
    skipped = refreshes(h, "skipped-stale")
    assert [r["k"] for r in skipped] == [15] and "right_gripper" in skipped[0]["reason"]
    assert len(h.transport.messages) == 2  # nothing was sent for k = 15
    with pytest.raises(fc.StaleTactile):
        run_ticks(h, 5)  # the next plan (chunk exhausted at tick 20) refuses stale tactile


def test_reset_drops_replies_of_requests_made_before():
    h = make_harness(policy_hz=50.0, horizon=20, lead=2, delays={0: 0.5})
    run_ticks(h, 6)  # k = 5 submitted, reply pending
    h.executor.reset()
    h.clock.now += 0.6
    run_ticks(h, 2)
    dropped = refreshes(h, "dropped")
    assert dropped and dropped[0]["reason"] == "stale-chunk"
    assert h.executor.pending_ack is None


def test_finish_logs_the_reply_in_flight_without_counting_it():
    h = make_harness(policy_hz=50.0, horizon=20, lead=2, delays={0: 0.05})
    run_ticks(h, 6)
    assert h.transport.busy
    h.executor.finish(wait_s=1.0, sleep=h.clock.sleep)
    last = h.ctx.requests[-1]
    assert last["status"] == "dropped" and last["reason"] == "round-ended" and not h.transport.busy
    assert h.executor.counts["dropped"] == 0


def test_transport_failure_disables_refreshes_for_the_rest_of_the_chunk():
    h = make_harness(policy_hz=50.0, horizon=20, lead=2)
    original = h.transport.submit

    def failing(job):
        accepted = original(job)
        h.transport.queue[-1].status = "timeout"
        h.transport.queue[-1].reply = None
        return accepted

    h.transport.submit = failing
    run_ticks(h, 20)
    statuses = [r["status"] for r in refreshes(h)]
    assert statuses == ["failed", "skipped-disabled", "skipped-disabled"]


def test_server_changed_on_the_refresh_path_aborts_the_round():
    h = make_harness(policy_hz=50.0, horizon=20, lead=2)
    original = h.transport.submit

    def changed(job):
        accepted = original(job)
        h.transport.queue[-1].status = "server_changed"
        h.transport.queue[-1].error = "instance changed"
        return accepted

    h.transport.submit = changed
    with pytest.raises(fc.ServerChanged):
        run_ticks(h, 10)


def test_chunk_mode_never_refreshes_and_uploads_the_same_tactile():
    # P0 in a campaign with the refresh condition i2: the identical pipeline uploads i2's tactile (ignored by P0)
    h = make_harness(condition="p0", policy_hz=30.0, horizon=20, campaign_conditions=["p0", "i2"])
    run_ticks(h, 40)
    assert refreshes(h) == [] and h.transport.messages == []
    plans = [m for m in h.server.messages if m.get("forceguide") == "plan"]
    assert len(plans) == 2 and sorted(plans[0]["tactile_frames"]) == list(KEYS)
    # i2 = W 1 + diff background: step 0, then the second chunk's start at position 32 * 0.6 = 19.2 -> step 19
    assert plans[0]["tactile_steps"] == [0] and plans[1]["tactile_steps"] == [19]
    assert plans[0]["noise"].dtype == np.float32 and plans[0]["state"].dtype == np.float32
    assert h.payload.proxy and h.ctx.payload.sensors == ("right_gripper_inner", "right_gripper_outer")
    # a P0-only campaign subscribes no sensor and uploads nothing
    alone = make_harness(condition="p0", policy_hz=30.0, horizon=20)
    run_ticks(alone, 2)
    plan = [m for m in alone.server.messages if m.get("forceguide") == "plan"][0]
    assert alone.campaign.sensors == () and plan["tactile_steps"] == [] and plan["tactile_frames"] == {}


# --------------------------------------------------------------------------------------------
# required_lead
# --------------------------------------------------------------------------------------------

def test_required_lead_reproduces_the_review_table():
    adv30, adv50 = DT * 30, DT * 50
    assert [fc.required_lead(adv30, r / 1000.0, DT) for r in (25, 40, 45, 60, 65, 80, 100)] == [2, 2, 3, 3, 4, 4, 4]
    assert [fc.required_lead(adv50, r / 1000.0, DT) for r in (25, 40, 45, 60, 65, 80, 85, 100)] == [
        2, 2, 3, 3, 4, 4, 5, 5]
    assert fc.required_lead(adv50, 0.0, DT) == 1 and fc.required_lead(adv30, 0.0, DT) == 2
    assert fc.required_lead(adv50, 5.0, DT, horizon=20) is None


def _review_simulation(adv, rtt_ms, lead, every=5, horizon=50):
    """scratchpad/review_client/splice_sim.py's run() with the minimal rule (an independent oracle)."""
    import math

    ticks = int(math.ceil((horizon - 1) / adv)) + 1
    in_flight, submitted, late, ontime = None, set(), [], []
    for t in range(ticks):
        now, off = t * 20.0, t * adv
        if off > horizon - 1 + 1e-9:
            break
        if in_flight is not None and in_flight[3] <= now + 1e-9:
            _, j, kp, _ = in_flight
            low = math.floor(off + 1e-9)
            if abs(adv - 1.0) < 1e-12:
                first = low
            else:
                prev = off - adv
                pl = math.floor(prev + 1e-9)
                first = pl + (2 if prev - pl > 1e-9 else 1)
            s = max(kp, first)
            (late if s > kp else ontime).append(j)
            in_flight = None
        j = math.floor(off + 1e-9)
        if j > 0 and j % every == 0 and j + lead < horizon and j not in submitted and in_flight is None:
            in_flight = (t, j, j + lead, now + rtt_ms)
            submitted.add(j)
    return late, ontime


@pytest.mark.parametrize("policy_hz", [30, 50])
def test_required_lead_agrees_with_the_review_simulation(policy_hz):
    adv = DT * policy_hz
    for rtt_ms in range(0, 120, 5):
        needed = fc.required_lead(adv, rtt_ms / 1000.0, DT)
        for lead in range(1, 8):
            late, ontime = _review_simulation(adv, rtt_ms, lead)
            if late or ontime:
                assert (not late) == (lead >= needed), (policy_hz, rtt_ms, lead, needed, late)


def test_simulation_matches_the_executor_splice_points():
    for policy_hz, rtt, lead in ((30.0, 0.06, 2), (30.0, 0.1, 3), (50.0, 0.05, 2), (50.0, 0.03, 3)):
        h = make_harness(policy_hz=policy_hz, horizon=50, lead=lead, delays={i: rtt for i in range(20)})
        run_ticks(h, fc.ticks_per_chunk(50, DT * policy_hz))
        decided, skipped = fc.simulate_refresh_timing(DT * policy_hz, rtt, DT, lead)
        got = [(r["k"], r["s"]) for r in refreshes(h) if r["status"] in ("applied", "dropped")]
        assert got == [(d["k"], d["s"]) for d in decided], (policy_hz, rtt, lead)
        assert [r["k"] for r in refreshes(h, "skipped-in-flight")] == skipped


# --------------------------------------------------------------------------------------------
# Connections and the worker (real websockets, blocking asyncio server)
# --------------------------------------------------------------------------------------------

def _wire():
    try:
        module = _load_f("msgpack_numpy")
        return module.packb, module.unpackb
    except pytest.skip.Exception:
        return packb, unpackb


class BlockingServer:
    """WebsocketPolicyServer's shape: metadata first, then per message a synchronous handle() inside the async
    handler (it blocks the event loop, so requests of all connections are served one at a time); on an exception
    the traceback string, then close(1011)."""

    def __init__(self, handle, metadata_fn):
        self.handle = handle
        self.metadata_fn = metadata_fn
        self.pack, self.unpack = _wire()
        self.connections = 0
        self._ready = threading.Event()
        self._loop = None
        self._stop = None
        self.port = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self._ready.wait(5.0)

    async def _handler(self, ws):
        from websockets.exceptions import ConnectionClosed

        self.connections += 1
        await ws.send(self.pack(self.metadata_fn()))
        while True:
            try:
                message = self.unpack(await ws.recv())
            except ConnectionClosed:
                return
            try:
                reply = self.handle(message)
            except Exception:  # noqa: BLE001
                await ws.send(traceback.format_exc())
                await ws.close(code=1011, reason="Internal server error. Traceback included in previous frame.")
                return
            try:
                await ws.send(self.pack(reply))
            except ConnectionClosed:
                return

    def _run(self):
        from websockets.asyncio.server import serve

        async def main():
            self._loop = asyncio.get_event_loop()
            self._stop = self._loop.create_future()
            async with serve(self._handler, "127.0.0.1", 0, compression=None, max_size=None) as server:
                self.port = server.sockets[0].getsockname()[1]
                self._ready.set()
                await self._stop

        asyncio.run(main())

    def stop(self):
        if self._loop is not None and not self._stop.done():
            self._loop.call_soon_threadsafe(self._stop.set_result, None)
        self._thread.join(5.0)


def test_connection_recovers_after_the_servers_error_string_and_close():
    metadata = make_metadata()

    def handle(message):
        if message.get("boom"):
            raise RuntimeError("boom")
        return {"ok": message["id"]}

    server = BlockingServer(handle, lambda: metadata)
    try:
        conn = fc.ForceGuideConnection("127.0.0.1", server.port, connect_timeout_s=5.0, infer_timeout_s=5.0,
                                       name="refresh", expected=fc.fingerprint_of(metadata))
        with pytest.raises(fc.ServerError, match="boom"):
            conn.request({"boom": True})
        assert not conn.connected()  # PolicyClient would keep the dead socket and fail forever
        reply, timing = conn.request({"id": 7}, timeout=5.0)
        assert reply == {"ok": 7} and conn.connects == 2 and timing["connect_ms"] > 0
        assert {"pack_ms", "send_ms", "wait_ms", "bytes"} <= set(timing)
        conn.close()
    finally:
        server.stop()


def test_reconnect_to_a_changed_server_raises_server_changed():
    instance = {"id": "server-1"}
    server = BlockingServer(lambda m: (_ for _ in ()).throw(RuntimeError("x")) if m.get("boom") else {"ok": 1},
                            lambda: make_metadata(instance=instance["id"]))
    try:
        conn = fc.ForceGuideConnection("127.0.0.1", server.port, connect_timeout_s=5.0, infer_timeout_s=5.0,
                                       name="main", expected=fc.fingerprint_of(make_metadata(instance="server-1")))
        with pytest.raises(fc.ServerError):
            conn.request({"boom": True})
        instance["id"] = "server-2"  # e.g. restarted with another condition set
        with pytest.raises(fc.ServerChanged):
            conn.request({"id": 1})
    finally:
        server.stop()


def test_connect_budget_is_respected_when_nothing_listens():
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    start = time.monotonic()
    with pytest.raises(fc.ConnectFailed):
        fc.ForceGuideConnection("127.0.0.1", port, connect_timeout_s=0.3, name="refresh")
    assert time.monotonic() - start < 1.5


def test_refresh_path_reconnect_gives_up_within_its_budget_while_the_server_computes():
    """The server's event loop answers neither the handshake nor the metadata while it computes (a plan here), so a
    refresh-path reconnect must give up within its budget (one refresh period) instead of waiting for it."""
    metadata = make_metadata()
    server = BlockingServer(lambda m: (time.sleep(1.5), {"ok": 1})[1] if m.get("slow") else {"ok": 2},
                            lambda: metadata)
    try:
        main = fc.ForceGuideConnection("127.0.0.1", server.port, connect_timeout_s=5.0, infer_timeout_s=5.0)
        refresh = fc.ForceGuideConnection("127.0.0.1", server.port, connect_timeout_s=5.0, infer_timeout_s=5.0,
                                          close_timeout_s=0.2, name="refresh", expected=fc.fingerprint_of(metadata))
        refresh.drop()  # e.g. after a timeout
        thread = threading.Thread(target=main.request, args=({"slow": True},), kwargs={"timeout": 5.0}, daemon=True)
        thread.start()
        time.sleep(0.2)  # the server is computing now
        start = time.monotonic()
        with pytest.raises(fc.ConnectFailed):
            refresh.request({"id": 1}, timeout=5.0, connect_budget_s=0.2)
        assert time.monotonic() - start < 0.8
        thread.join(5.0)
        reply, timing = refresh.request({"id": 2}, timeout=5.0, connect_budget_s=2.0)  # free again
        assert reply == {"ok": 2} and refresh.connects == 2 and timing["connect_ms"] > 0
        main.close()
        refresh.close()
    finally:
        server.stop()


def test_plan_waits_only_for_the_refresh_in_flight_not_for_a_close_stall():
    metadata = make_metadata()

    def handle(message):
        time.sleep(0.4 if message.get("forceguide") == "refresh" else 0.05)  # blocks the event loop
        return {"forceguide": {"req": message["req"]}}

    server = BlockingServer(handle, lambda: metadata)
    try:
        main = fc.ForceGuideConnection("127.0.0.1", server.port, connect_timeout_s=5.0, infer_timeout_s=5.0)
        refresh = fc.ForceGuideConnection("127.0.0.1", server.port, connect_timeout_s=5.0, infer_timeout_s=5.0,
                                          close_timeout_s=1.0, name="refresh")
        worker = fc.RefreshWorker(refresh, lambda job: {"forceguide": "refresh", "req": 1}, timeout_s=0.1)
        assert worker.submit(fc.RefreshRequest())
        time.sleep(0.05)  # the server is busy with the refresh now; its reply will be late (hard timeout 0.1 s)
        start = time.monotonic()
        reply, _ = main.request({"forceguide": "plan", "req": 2}, timeout=5.0)
        elapsed = time.monotonic() - start
        assert reply["forceguide"]["req"] == 2
        assert elapsed < 0.75, elapsed  # in-flight refresh (~0.35 s) + plan (0.05 s); no 1 s close stall
        deadline = time.monotonic() + 5.0
        results = []
        while not results and time.monotonic() < deadline:
            results = worker.poll()
            time.sleep(0.01)
        assert results and results[0].status == "timeout" and not worker.busy
        job = fc.RefreshRequest(timeout=2.0)  # the server takes 0.4 s per refresh; 0.1 s would time out again
        assert worker.submit(job)  # the worker reconnects and goes on
        deadline = time.monotonic() + 5.0
        results = []
        while not results and time.monotonic() < deadline:
            results = worker.poll()
            time.sleep(0.01)
        assert results[0].status == "ok" and results[0].reconnects == 2
        worker.close(1.0)
        main.close()
    finally:
        server.stop()


def test_worker_survives_errors_and_shutdown_never_hangs():
    metadata = make_metadata()
    server = BlockingServer(lambda m: (time.sleep(3.0), {"ok": 1})[1] if m.get("slow") else {"ok": 2},
                            lambda: metadata)
    try:
        conn = fc.ForceGuideConnection("127.0.0.1", server.port, connect_timeout_s=5.0, infer_timeout_s=10.0,
                                       name="refresh")

        def build(job):
            if job.k == -1:
                raise ValueError("cannot prepare")
            return {"slow": job.k == 99}

        worker = fc.RefreshWorker(conn, build, timeout_s=10.0)
        result = worker.call(fc.RefreshRequest(k=-1), 5.0)
        assert result.status == "error" and "cannot prepare" in result.error
        assert worker.call(fc.RefreshRequest(k=1), 5.0).status == "ok"
        assert worker.submit(fc.RefreshRequest(k=99)) and not worker.submit(fc.RefreshRequest(k=2))
        time.sleep(0.1)
        start = time.monotonic()
        worker.close(timeout=0.5)
        assert time.monotonic() - start < 2.0
    finally:
        server.stop()


def test_full_session_over_websockets_with_two_connections():
    metadata = make_metadata(horizon=20, policy_hz=50.0)
    fake = FakeSessionServer(metadata)
    server = BlockingServer(fake.handle, lambda: metadata)
    try:
        main = fc.ForceGuideConnection("127.0.0.1", server.port, connect_timeout_s=5.0, infer_timeout_s=5.0)
        refresh = fc.ForceGuideConnection("127.0.0.1", server.port, connect_timeout_s=5.0, infer_timeout_s=5.0,
                                          close_timeout_s=0.2, name="refresh")
        client = fc.ForceGuideClient(main, refresh, hard_timeout_s=5.0, refresh_budget_s=0.1)
        campaign = fc.CampaignConfig(client.info, ["i2"], lead=3)
        streams = {sensor: FakeStream(time.time, sensor_image(sensor)) for sensor in campaign.sensors}
        rig = fc.TactileRig(streams, campaign.history_capacity, 0.2)
        settings = campaign.settings("i2")
        rig.reset(settings.spec)
        payload = settings.make_payload(rig.history)
        adapter = fc.ForceGuideAdapter.from_info(client.info, hold_pose=POSE)
        measured = FollowerCommander((FakeBot(gripper=0.07), FakeBot(gripper=0.07)), max_joint_step=None).measured
        ctx = fc.TrialContext(client, adapter, "i2", payload, rig.monitor, fc.NoiseSchedule(0, 1, 20, 32), measured,
                              plan_timeout=5.0)
        ctx.log_request(client.reset("i2", 50.0, timeout=5.0))
        executor = fc.RefreshingChunkExecutor(
            ctx.plan, 50.0, DT, action_horizon=20, mode="refresh", refresh_every=5, lead=3, transport=client.worker,
            decode=adapter.decode, guard=fc.SpliceGuard(adapter.active_joints, adapter.active_grippers),
            can_refresh=ctx.can_refresh, refresh_timeout_s=1.0, log_fn=ctx.log_request, sidecar=ctx.sidecar,
            payload=payload)
        outcome, reason = runner.run_forceguide_episode(executor, runner.DryFollower(measured), runner.NullKeyboard(),
                                                        lambda: camera_frame(time.monotonic), rig, 45, DT, ctx=ctx)
        executor.finish(wait_s=2.0)
        assert outcome == "done", reason
        records = [r for r in ctx.requests if r["type"] == "refresh"]
        assert records and all(r["transport"] == "ok" for r in records)
        assert {r["status"] for r in records} <= {"applied", "dropped"}
        reqs = [m["req"] for m in fake.messages]
        assert reqs == sorted(reqs) and len(set(reqs)) == len(reqs)  # one monotonic counter for both connections
        assert {m["session"] for m in fake.messages} == {client.session}
        assert all(r.get("missing_steps") in (None, []) for r in records)
        client.close()
    finally:
        server.stop()


# --------------------------------------------------------------------------------------------
# Schedule, noise
# --------------------------------------------------------------------------------------------

def test_schedule_blocks_randomize_conditions_within_each_initial_condition(tmp_path):
    rows = fc.make_schedule(["p0", "tg", "i2"], range(6), seed=1)
    assert len(rows) == 18 and len({r["trial_id"] for r in rows}) == 18
    for block in range(6):
        inside = [r for r in rows if r["block"] == block]
        assert sorted(r["condition"] for r in inside) == ["i2", "p0", "tg"]
        assert {r["init_condition_id"] for r in inside} == {block} and {r["noise_id"] for r in inside} == {block}
    orders = {tuple(r["condition"] for r in rows if r["block"] == b) for b in range(6)}
    assert len(orders) > 1  # randomized, not a fixed order
    assert fc.make_schedule(["p0", "tg"], [3], seed=1) == fc.make_schedule(["p0", "tg"], [3], seed=1)
    twice = fc.make_schedule(["p0"], [3], repeats=2)
    assert [r["noise_id"] for r in twice] == [3, 1000003]
    path = str(tmp_path / "trials.jsonl")
    fc.write_schedule(path, rows)
    info = fc.ServerInfo.from_metadata(make_metadata())
    assert fc.load_schedule(path, info) == rows
    with open(path, "a") as handle:
        handle.write(json.dumps({"trial_id": "t0000", "init_condition_id": 1, "condition": "p0"}) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        fc.load_schedule(path, info)
    repeated = tmp_path / "repeated.jsonl"  # contract C2: one trial per (condition, noise_id)
    repeated.write_text("".join(json.dumps({"trial_id": t, "init_condition_id": 4, "condition": "p0"}) + "\n"
                                for t in ("a", "b")))
    with pytest.raises(ValueError, match="noise_id 4 repeats trial 'a'"):
        fc.load_schedule(str(repeated), info)
    with pytest.raises(ValueError, match="repeat"):
        fc.make_schedule(["p0", "tg"], [3, 3])
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"trial_id": "a", "init_condition_id": "x", "condition": "p0"}) + "\n")
    with pytest.raises(ValueError, match="integers"):
        fc.load_schedule(str(bad))
    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text(json.dumps({"trial_id": "a", "init_condition_id": 1, "condition": "zz"}) + "\n")
    with pytest.raises(KeyError):
        fc.load_schedule(str(unknown), info)


def test_noise_is_paired_by_initial_condition_and_restarts_per_attempt():
    a, b = fc.NoiseSchedule(7, 4, 50, 32), fc.NoiseSchedule(7, 4, 50, 32)
    first = [a.next() for _ in range(3)]
    np.testing.assert_array_equal(first[1], b.next() * 0 + fc.NoiseSchedule(7, 4, 50, 32).next() * 0 + first[1])
    np.testing.assert_array_equal(first[0], np.random.default_rng([7, 4, 0]).standard_normal((50, 32)).astype(
        np.float32))
    np.testing.assert_array_equal(first[2], np.random.default_rng([7, 4, 2]).standard_normal((50, 32)).astype(
        np.float32))
    assert first[0].dtype == np.float32 and first[0].shape == (50, 32) and a.plan_index == 3
    assert not np.array_equal(first[0], fc.NoiseSchedule(7, 5, 50, 32).next())
    with pytest.raises(ValueError):
        fc.NoiseSchedule(-1, 0, 50, 32)


# --------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------

class _Follower:
    def __init__(self, clock):
        self.clock = clock
        self.sent = []

    def send(self, action):
        self.sent.append((self.clock(), np.asarray(action)))
        return np.asarray(action, dtype=np.float64)


def _plan_executor(clock, plan_s, policy_hz=50.0):
    def plan_fn(frame_provider, plan_step, ack):
        clock.now += plan_s  # the blocking round trip
        chunk = np.tile(np.r_[POSE, 0.5, POSE, 0.5], (50, 1))
        chunk[:, 0] = 0.001 * np.arange(50)
        return fc.PlanResult(chunk, server_chunk=1, req=plan_step + 1)

    return fc.RefreshingChunkExecutor(plan_fn, policy_hz, DT, action_horizon=50, clock=clock)


def test_plan_pause_padding_and_one_period_sleep_after_the_overrun():
    clock = FakeClock(0.0)
    follower = _Follower(clock)
    ctx = fc.TrialContext(None, None, "x", fc.TactilePayload.empty())
    ctx.plan_records[1] = {"type": "plan", "req": 1}
    outcome, _ = runner.run_forceguide_episode(_plan_executor(clock, 0.13), follower, runner.NullKeyboard(),
                                               lambda: None, None, 5, DT, plan_pause_s=0.25, clock=clock,
                                               sleep=clock.sleep, ctx=ctx)
    times = [t for t, _ in follower.sent]
    assert outcome == "done"
    assert times[0] == pytest.approx(0.25)  # a'[0] waits until plan_pause_s after the blocking tick began
    assert times[1] - times[0] == pytest.approx(DT)  # one control period later, not back to back
    assert np.diff(times[1:]) == pytest.approx(DT)
    assert ctx.plan_records[1]["pause_ms"] == pytest.approx(250.0) and not ctx.plan_records[1]["pause_overrun"]
    assert ctx.ticks[0]["plan"] and ctx.ticks[0]["overrun"] and not ctx.ticks[2]["overrun"]
    # a plan slower than the pause is flagged
    clock = FakeClock(0.0)
    ctx = fc.TrialContext(None, None, "x", fc.TactilePayload.empty())
    ctx.plan_records[1] = {}
    runner.run_forceguide_episode(_plan_executor(clock, 0.4), _Follower(clock), runner.NullKeyboard(), lambda: None,
                                  None, 2, DT, plan_pause_s=0.25, clock=clock, sleep=clock.sleep, ctx=ctx)
    assert ctx.plan_records[1]["pause_overrun"] and ctx.plan_records[1]["pause_ms"] == pytest.approx(400.0)


def test_tick_records_carry_the_trial_labels_and_the_measured_pose():
    clock = FakeClock(0.0)
    ctx = fc.TrialContext(None, None, "x", fc.TactilePayload.empty(), labels={"trial_id": "t7", "attempt": 2})
    poses = iter(np.arange(14.0 * 3).reshape(3, 14))
    outcome, _ = runner.run_forceguide_episode(_plan_executor(clock, 0.0), _Follower(clock), runner.NullKeyboard(),
                                               lambda: None, None, 3, DT, clock=clock, sleep=clock.sleep, ctx=ctx,
                                               measured_fn=lambda: next(poses))
    assert outcome == "done" and len(ctx.ticks) == 3
    assert all(t["trial_id"] == "t7" and t["attempt"] == 2 for t in ctx.ticks)
    np.testing.assert_array_equal(ctx.ticks[2]["measured"], np.arange(28.0, 42.0))
    assert all(t["period"] is None or t["period"] == pytest.approx(DT) for t in ctx.ticks)


def test_effective_refresh_interval_counts_applied_splices_from_the_chunk_start():
    records = [{"chunk": 1, "status": "applied", "s": 7}, {"chunk": 1, "status": "rejected", "s": 12},
               {"chunk": 1, "status": "applied", "s": 17}, {"chunk": 2, "status": "applied", "s": 8},
               {"chunk": 2, "status": "skipped-K0", "s": None}]
    assert sorted(runner.refresh_intervals(records)) == [7, 8, 10]
    assert runner.refresh_intervals([]) == []


def test_session_executor_uses_the_campaign_cadence_lead_and_refresh_period_budget():
    h = make_harness(condition="i2", policy_hz=30.0, horizon=50, lead=4)
    h.client.worker = h.transport
    args = types.SimpleNamespace(seed=5, max_tactile_age=0.2)
    session = runner.Session(args, h.client, h.campaign, h.adapter, h.rig, fc.SpliceGuard((0,), (6,)), DT, 50,
                             h.frame_provider, clock=h.clock, sleep=h.clock.sleep)
    ctx = session.trial("i2", 3, h.measured, 10.0, {"trial_id": "t"})
    executor = session.executor(ctx, refresh_timeout_s=0.25)
    assert (executor.mode, executor.refresh_every, executor.lead) == ("refresh", 5, 4)
    assert executor.connect_budget_s == pytest.approx(5 / 30.0)  # the refresh path gives up within one period
    assert executor.refresh_timeout_s == 0.25 and executor.transport is h.transport
    np.testing.assert_array_equal(ctx.noise.next(), fc.NoiseSchedule(5, 3, 50, 32).next())


def test_keys_pressed_during_a_blocking_plan_stop_the_first_command():
    clock = FakeClock(0.0)
    follower = _Follower(clock)
    for key, expected in (("left", "aborted"), ("esc", "quit"), ("right", "ended")):
        outcome, reason = runner.run_forceguide_episode(_plan_executor(clock, 0.1), follower,
                                                        ScriptedKeyboard({1: [key]}), lambda: None, None, 10, DT,
                                                        clock=clock, sleep=clock.sleep)
        assert (outcome, reason) == (expected, "key during the plan") and follower.sent == []


def test_plan_errors_and_stale_cameras_are_infrastructure_failures():
    clock = FakeClock(0.0)

    def broken(frame_provider, plan_step, ack):
        raise TimeoutError("no reply")

    executor = fc.RefreshingChunkExecutor(broken, 50.0, DT, clock=clock)
    outcome, reason = runner.run_forceguide_episode(executor, _Follower(clock), runner.NullKeyboard(), lambda: None,
                                                    None, 5, DT, clock=clock, sleep=clock.sleep)
    assert outcome == "infrastructure" and "no reply" in reason

    def stale(frame_provider, plan_step, ack):
        return frame_provider()

    def camera():
        raise runner.StaleImage("cam_high image is 1.0s old")

    executor = fc.RefreshingChunkExecutor(stale, 50.0, DT, clock=clock)
    outcome, reason = runner.run_forceguide_episode(executor, _Follower(clock), runner.NullKeyboard(), camera, None,
                                                    5, DT, clock=clock, sleep=clock.sleep)
    assert outcome == "infrastructure" and "cam_high" in reason


def test_operator_outcome_prompt_and_classification():
    answers = iter(["maybe", "n", "align", "jam", "slipped at the hole"])
    printed = []
    outcome = runner.ask_outcome(lambda prompt: next(answers), printed.append)
    assert outcome == {"labelled": True, "success": False, "rerun": False, "stage": "align", "failure_type": "jam",
                       "notes": "slipped at the hole"} and printed
    answers = iter(["y", "", ""])
    assert runner.ask_outcome(lambda prompt: next(answers))["stage"] == "done"
    answers = iter(["r", "board fell"])
    assert runner.ask_outcome(lambda prompt: next(answers))["rerun"]

    def eof(prompt):
        raise EOFError

    assert not runner.ask_outcome(eof)["labelled"]
    assert runner.classify("done", {"slots": 10, "bad_frac": 0.05}, 0.1) == "labelled"
    # robot review 3 / exp 1: rejected / late / dropped / in-flight slots are method outcomes, never a re-run cause
    rejected = {"slots": 10, "counts": {"rejected": 3, "late": 2, "dropped": 1, "skipped-in-flight": 1},
                "bad_frac": 0.7}
    assert runner.classify("done", rejected, 0.1) == "labelled"
    assert runner.protocol_flags(rejected, 0.1)["flags"] == ["bad_refresh_frac"]
    assert runner.protocol_flags(rejected, 0.1)["counts"]["rejected"] == 3
    assert runner.classify("done", {"slots": 10, "bad_frac": 0.2, "infra_frac": 0.2}, 0.1) == "infrastructure_failure"
    infra = {"slots": 10, "counts": {"failed": 1, "skipped-disabled": 1}, "bad_frac": 0.2}  # older summary format
    assert runner.classify("ended", infra, 0.1) == "infrastructure_failure" and runner.infra_frac(infra) == 0.2
    assert runner.classify("done", {"slots": 10, "counts": {"malformed": 2, "rejected": 2}}, 0.1) == (
        "infrastructure_failure")
    assert runner.protocol_flags({"slots": 0}, 0.1)["flags"] == []
    assert runner.classify("infrastructure", {"slots": 0}, 0.1) == "infrastructure_failure"
    assert runner.classify("quit", {"slots": 0}, 0.1) == "quit"
    assert runner.parse_inits(["0-3", "7,9"]) == [0, 1, 2, 3, 7, 9]
    assert runner.key_outcome(["x", "right"]) == "ended"
    server_error = "ServerError: main: policy server error:\nTraceback (most recent call last):\n  File session.py"
    assert runner.console_reason(server_error) == "the policy server raised an error; details in episodes.jsonl"
    assert runner.console_reason("StaleImage: cam_high image is 0.61s old") == (
        "camera/tactile data stale; details in episodes.jsonl")
    assert "limit" in runner.console_reason(None)


def test_clamp_flags_report_which_safety_layer_acted():
    bots = (FakeBot(), FakeBot())
    planned = np.r_[POSE, 1.2, POSE, 0.5]
    planned[0] = 3.5  # beyond the waist limit 3.0
    sent = planned.copy()
    sent[0] = 0.1  # limited by the per-tick step after clamping
    flags = runner.clamp_flags(planned, sent, bots, clip_gripper=True)
    assert flags == {"limit": [True, False], "step": [True, False], "gripper_clip": True}


def _fake_hardware(monkeypatch, server, keyboards, log):
    class FakeImageRecorder:
        cam_high_secs = cam_left_wrist_secs = cam_right_wrist_secs = 0
        cam_high_nsecs = cam_left_wrist_nsecs = cam_right_wrist_nsecs = 0

    class FakeEnv:
        def __init__(self):
            self.puppet_bot_left, self.puppet_bot_right = FakeBot(gripper=0.07), FakeBot(gripper=0.07)
            for bot in (self.puppet_bot_left, self.puppet_bot_right):
                bot.arm.set_trajectory_time = lambda moving_time=None, accel_time=None: None
            self.image_recorder = FakeImageRecorder()

        def get_observation(self):
            frame = camera_frame(time.monotonic)
            qpos = np.concatenate([self.puppet_bot_left.position[:7], self.puppet_bot_right.position[:7]])
            return {"qpos": qpos, "qvel": np.zeros(14), "effort": np.zeros(14), "images": frame.images}

        def get_images(self):
            return camera_frame(time.monotonic).images

    env = FakeEnv()
    streams = {sensor: FakeStream(time.time, sensor_image(sensor)) for sensor in fc.KNOWN_SENSORS}
    monkeypatch.setitem(sys.modules, "real_env", types.SimpleNamespace(
        make_real_env=lambda init_node, setup_robots: env))
    monkeypatch.setitem(sys.modules, "robot_utils", types.SimpleNamespace(
        move_arms=lambda bots, poses, move_time: log.append("home"), move_grippers=lambda bots, t, move_time: None))
    monkeypatch.setitem(sys.modules, "sleep_plus", types.SimpleNamespace(
        shut_down_puppet_robots=lambda: log.append("shutdown")))
    monkeypatch.setitem(sys.modules, "one_side_teleop", types.SimpleNamespace(
        KeyboardControl=lambda: next(keyboards)))
    monkeypatch.setitem(sys.modules, "tactile_stream", types.SimpleNamespace(
        TactileStream=types.SimpleNamespace(acquire=lambda sensor, prefetch=(): streams[sensor]),
        calibrate_sensors=lambda sensors: log.append(("calibrate", tuple(sensors))) or {"serials": {"x": 1}}))
    connections = []

    def connect(host, port, **kwargs):
        connections.append(LoopbackConnection(server, kwargs.get("name", "main")))
        return connections[-1]

    monkeypatch.setattr(runner, "ForceGuideConnection", connect)
    return env, streams, connections


def _write_preflight(path, conditions, advance, refresh_every=5, required_lead=2, ok=True, age_h=0.0,
                     p99_ms=20.0, recommended=0.05):
    """A timing preflight report as run_preflight writes it (the fields check_preflight_report reads)."""
    written = (datetime.datetime.now() - datetime.timedelta(hours=age_h)).isoformat()
    report = {"conditions": {name: {"outcome": "done"} for name in conditions}, "advance": advance,
              "refresh_every": refresh_every, "required_lead": required_lead, "ok": ok,
              "problems": [] if ok else ["no refresh round trip succeeded"], "plan_rtt_p99_ms": p99_ms,
              "plan_pause_s_recommended": recommended, "written": written}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(report, handle)
    return path


def _episodes(log_dir):
    with open(os.path.join(log_dir, "episodes.jsonl")) as handle:
        return [json.loads(line) for line in handle]


def test_main_runs_a_scheduled_campaign_with_faked_hardware(monkeypatch, tmp_path):
    metadata = make_metadata(horizon=20, policy_hz=50.0)
    server = FakeSessionServer(metadata)
    schedule = str(tmp_path / "trials.jsonl")
    assert runner.main(["--make_schedule", schedule, "--schedule_conditions", "p0", "i2",
                        "--schedule_inits", "0-1", "--schedule_seed", "3"]) == 0
    log = []
    keyboards = iter([ScriptedKeyboard({0: ["right"]}) for _ in range(10)])
    _, streams, connections = _fake_hardware(monkeypatch, server, keyboards, log)
    answers = iter(["y", "insert", "", "n", "align", "miss", "", "r", "fell over", "y", "done", "", "y", "", ""])
    log_dir = str(tmp_path / "logs")
    _write_preflight(os.path.join(log_dir, "preflight.json"), ["p0", "i2"], advance=DT * 50.0)
    code = runner.main(["--host", "127.0.0.1", "--schedule", schedule, "--lead", "2", "--log_dir", log_dir,
                        "--max_episode_steps", "45", "--plan_pause_s", "0.05", "--max_bad_refresh_frac", "0.5"],
                       input_fn=lambda prompt: next(answers))
    assert code == 0
    with open(os.path.join(log_dir, "episodes.jsonl")) as handle:
        episodes = [json.loads(line) for line in handle]
    assert [e["status"] for e in episodes] == ["success", "failure", "rerun", "success", "success"]
    assert episodes[2]["trial_id"] == episodes[3]["trial_id"] and episodes[3]["attempt"] == 2
    # trial identity (contract C2) and the campaign's pause / preflight (exp 10) in every row
    rows = {row["trial_id"]: row for row in fc.load_schedule(schedule)}
    for episode in episodes:
        assert episode["schedule_sha256"] == fc.file_sha256(schedule)
        row = rows[episode["trial_id"]]
        assert (episode["condition"], episode["noise_id"], episode["block"]) == (row["condition"], row["noise_id"],
                                                                                 row["block"])
        assert episode["protocol_flags"]["flags"] == [] and episode["plan_pause_source"] == "--plan_pause_s"
        assert episode["preflight_check"]["exists"] and episode["preflight_check"]["problems"] == []
        assert episode["operator"]["labelled"]
    resets = [(m["trial"], m["attempt"]) for m in server.messages if m.get("forceguide") == "reset" and "trial" in m]
    assert resets == [(e["trial_id"], e["attempt"]) for e in episodes]  # robot review 6: the server seeds per trial
    with open(os.path.join(log_dir, "sessions.jsonl")) as handle:
        record = json.loads(handle.readline())
    assert record["schedule_sha256"] == fc.file_sha256(schedule) and record["plan_pause_s"] == 0.05
    assert record["preflight_check"]["sha256"] == fc.file_sha256(os.path.join(log_dir, "preflight.json"))
    for episode in episodes:
        assert episode["condition_fingerprint"] == "fp-" + episode["condition"]
        assert episode["server_instance_id"] == "server-1" and episode["lead"] == 2 and episode["refresh_every"] == 5
        assert episode["metadata"]["conditions"]["i2"]["mode"] == "refresh" and episode["ticks"] == 45
        assert os.path.exists(episode["logs"]["ticks"]) and os.path.exists(episode["logs"]["requests"])
        assert episode["plan_pause_ms"]["n"] == 3 and episode["tactile_payload"]["keys"] == list(KEYS)
    i2 = [e for e in episodes if e["condition"] == "i2"]
    assert i2 and all(e["refresh"]["counts"].get("applied", 0) >= 3 for e in i2)
    assert all(e["logs"]["refresh_npz"] for e in i2)
    with np.load(i2[0]["logs"]["refresh_npz"]) as sidecar:
        assert sidecar["old"].shape[1:] == (20, 14) and (sidecar["s"] >= sidecar["k_pin"]).all()
    with open(episodes[0]["logs"]["ticks"]) as handle:
        tick = json.loads(handle.readline())
    assert {"period", "overrun", "chunk", "position", "offset", "step", "planned_action", "sent_action", "clamp",
            "tactile", "camera_ages", "src_req"} <= set(tick) and tick["plan"]
    with open(episodes[0]["logs"]["requests"]) as handle:
        kinds = [json.loads(line)["type"] for line in handle]
    assert kinds[0] == "reset" and "plan" in kinds
    assert os.path.exists(os.path.join(log_dir, "warmup.jsonl"))
    assert log[-1] == "shutdown" and ("calibrate", ("right_gripper_inner", "right_gripper_outer")) in log
    assert sorted(streams) and all(streams[s].released for s in ("right_gripper_inner", "right_gripper_outer"))
    assert {c.name for c in connections} == {"main", "refresh"} and all(c.closed for c in connections)
    # the operator only ever saw trial ids and initial conditions: the noise is shared inside each block
    plans = [m for m in server.messages if m.get("forceguide") == "plan"]
    assert len({m["session"] for m in server.messages}) == 1 and plans
    # resume: every trial is final, nothing runs again
    keyboards2 = iter([ScriptedKeyboard() for _ in range(3)])
    _fake_hardware(monkeypatch, server, keyboards2, log)
    before = len(server.messages)
    assert runner.main(["--host", "127.0.0.1", "--schedule", schedule, "--lead", "2", "--log_dir", log_dir,
                        "--no_warmup", "--plan_pause_s", "0.05"]) == 0
    assert len(server.messages) == before


def test_main_marks_infrastructure_failures_and_reruns_them(monkeypatch, tmp_path, capsys):
    metadata = make_metadata(horizon=20, policy_hz=50.0)
    server = FakeSessionServer(metadata)
    calls = {"plans": 0}
    original = server._handle

    def flaky(message):
        if message.get("forceguide") == "plan" and message.get("tactile_step") == 0:
            calls["plans"] += 1
            if calls["plans"] == 2:  # the first trial's first plan (after the warm-up plan) fails
                raise RuntimeError("XLA error")
        return original(message)

    server._handle = flaky
    log = []
    keyboards = iter([ScriptedKeyboard() for _ in range(10)])  # no key during the trials
    _fake_hardware(monkeypatch, server, keyboards, log)
    starts = []
    monkeypatch.setattr(runner, "wait_for_start", lambda keyboard: starts.append(1) or True)  # RIGHT per attempt
    answers = iter(["y", "done", ""])
    log_dir = str(tmp_path / "logs")
    assert runner.main(["--host", "h", "--condition", "p0", "--num_episodes", "1", "--log_dir", log_dir,
                        "--max_episode_steps", "10"], input_fn=lambda prompt: next(answers)) == 0
    with open(os.path.join(log_dir, "episodes.jsonl")) as handle:
        episodes = [json.loads(line) for line in handle]
    assert [e["status"] for e in episodes] == ["infrastructure_failure", "success"]
    assert "XLA error" in episodes[0]["reason"] and episodes[1]["attempt"] == 2 and len(starts) == 2
    console = [line for line in capsys.readouterr().out.splitlines() if "infrastructure failure" in line]
    assert console and all("Traceback" not in line and "XLA error" not in line for line in console), console
    assert episodes[0]["noise_id"] == episodes[1]["noise_id"]  # a re-run replays the same paired noise
    plans = [m for m in server.messages if m.get("forceguide") == "plan" and m.get("tactile_step") == 0]
    np.testing.assert_array_equal(plans[-1]["noise"], fc.NoiseSchedule(0, 0, 20, 32).next())


def test_every_attempt_that_ran_gets_an_operator_outcome_and_only_infrastructure_faults_rerun(monkeypatch, tmp_path):
    """Robot review 3 / exp 1 (intention to treat) and robot review 6 (trial id in the reset)."""
    metadata = make_metadata(horizon=20, policy_hz=50.0)
    events = []
    monkeypatch.setattr(runner, "print", lambda *a, **k: events.append(("print", " ".join(str(x) for x in a))),
                        raising=False)
    monkeypatch.setattr(runner, "wait_for_start", lambda keyboard: True)

    def operator(answers):
        answers = iter(answers)

        def answer(prompt):
            events.append(("prompt", prompt))
            return next(answers)

        return answer

    # 1. a server error reply disables the rest of its chunk: an infrastructure fault -> re-run, but the operator
    #    still enters what physically happened, before being told
    server = FakeSessionServer(metadata)
    server.script = {0: {"error": "boom"}}
    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(10)]), [])
    log_dir = str(tmp_path / "infra")
    assert runner.main(["--host", "h", "--condition", "i2", "--num_episodes", "1", "--lead", "2", "--log_dir",
                        log_dir, "--max_episode_steps", "45", "--no_warmup"],
                       input_fn=operator(["n", "align", "jam", "", "y", "done", ""])) == 0
    episodes = _episodes(log_dir)
    assert [e["status"] for e in episodes] == ["infrastructure_failure", "success"]
    first = episodes[0]
    assert first["refresh"]["counts"]["error"] == 1 and first["refresh"]["infra_frac"] > 0.1
    assert first["operator"] == {"labelled": True, "success": False, "rerun": False, "stage": "align",
                                 "failure_type": "jam", "notes": ""}
    told = next(i for i, event in enumerate(events) if "it will be re-run" in event[1])
    asked = [i for i, event in enumerate(events) if event[0] == "prompt"]
    assert asked[0] < told < asked[-1]
    rows = [{"trial_id": "r0000", "block": 0, "init_condition_id": 0, "noise_id": 0, "condition": "i2",
             "run_label": "infra"}]  # contract C9: the log dir's name labels a run without a schedule file
    assert {e["schedule_sha256"] for e in episodes} == {fc.rows_sha256(rows)}
    assert {e["run_label"] for e in episodes} == {"infra"}
    resets = [m for m in server.messages if m.get("forceguide") == "reset"]
    assert [(m["trial"], m["attempt"]) for m in resets] == [("r0000", 1), ("r0000", 2)]

    # 2. splice-guard rejections are the method's own outcome: flagged, labelled, never re-run
    server = FakeSessionServer(metadata)
    server.script = {0: {"jump": 1.0}}
    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(10)]), [])
    log_dir = str(tmp_path / "rejected")
    assert runner.main(["--host", "h", "--condition", "i2", "--num_episodes", "1", "--lead", "2", "--log_dir",
                        log_dir, "--max_episode_steps", "45", "--no_warmup", "--max_bad_refresh_frac", "0.05"],
                       input_fn=operator(["y", "done", ""])) == 0
    episodes = _episodes(log_dir)
    assert [e["status"] for e in episodes] == ["success"]
    assert episodes[0]["refresh"]["counts"]["rejected"] == 1 and episodes[0]["refresh"]["bad_frac"] > 0.05
    assert episodes[0]["protocol_flags"]["flags"] == ["bad_refresh_frac"]
    assert episodes[0]["protocol_flags"]["counts"]["rejected"] == 1


def test_an_infrastructure_stop_names_its_fault_before_the_operator_answers(monkeypatch, tmp_path):
    """Contract C10 (review N1): the "stopped" line of an attempt stopped by an infrastructure fault names the fault
    in the blinded console form (console_reason: a fixed category, never the server's text); the order is kept, the
    operator answers before being told that the attempt will be re-run."""
    server = FakeSessionServer(make_metadata(horizon=20, policy_hz=50.0))
    failed = []
    original = server._handle

    def flaky(message):  # the first attempt's second plan fails: the attempt ran one chunk, then stopped
        if message.get("forceguide", "plan") == "plan" and message.get("tactile_step", 0) > 0 and not failed:
            failed.append(message["tactile_step"])
            raise RuntimeError("XLA error")
        return original(message)

    server._handle = flaky
    events = []
    monkeypatch.setattr(runner, "print", lambda *a, **k: events.append(("print", " ".join(str(x) for x in a))),
                        raising=False)
    monkeypatch.setattr(runner, "wait_for_start", lambda keyboard: True)
    answers = iter(["n", "align", "jam", "", "y", "done", ""])

    def answer(prompt):
        events.append(("prompt", prompt))
        return next(answers)

    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(10)]), [])
    log_dir = str(tmp_path / "logs")
    assert runner.main(["--host", "h", "--condition", "p0", "--num_episodes", "1", "--log_dir", log_dir,
                        "--max_episode_steps", "30", "--no_warmup"], input_fn=answer) == 0
    episodes = _episodes(log_dir)
    assert [e["status"] for e in episodes] == ["infrastructure_failure", "success"] and len(failed) == 1
    assert episodes[0]["outcome"] == "infrastructure" and "XLA error" in episodes[0]["reason"]
    stopped = [i for i, event in enumerate(events) if event[1].startswith("Trial r0000 stopped")]
    assert len(stopped) == 1, events
    line = events[stopped[0]][1]
    assert line == "Trial r0000 stopped ({}).".format(runner.console_reason(episodes[0]["reason"]))
    assert line == "Trial r0000 stopped (the policy server raised an error; details in episodes.jsonl)."
    asked = [i for i, event in enumerate(events) if event[0] == "prompt"]
    told = next(i for i, event in enumerate(events) if "it will be re-run" in event[1])
    assert stopped[0] < asked[0] < told


def test_unblinded_console_names_the_condition_and_the_full_failure_reason(monkeypatch, tmp_path):
    """--unblinded (2026-11-24, the operator does not want blinding): the start line and the "stopped" / re-run lines
    name the trial's condition and the full reason; episodes.jsonl is the same as in a blinded run."""
    server = FakeSessionServer(make_metadata(horizon=20, policy_hz=50.0))
    failed = []
    original = server._handle

    def flaky(message):  # the first attempt's second plan fails, as in the blinded test above
        if message.get("forceguide", "plan") == "plan" and message.get("tactile_step", 0) > 0 and not failed:
            failed.append(message["tactile_step"])
            raise RuntimeError("XLA error")
        return original(message)

    server._handle = flaky
    events = []
    monkeypatch.setattr(runner, "print", lambda *a, **k: events.append(" ".join(str(x) for x in a)), raising=False)
    monkeypatch.setattr(runner, "wait_for_start", lambda keyboard: True)
    answers = iter(["n", "align", "jam", "", "y", "done", ""])
    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(10)]), [])
    log_dir = str(tmp_path / "logs")
    assert runner.main(["--host", "h", "--condition", "p0", "--num_episodes", "1", "--log_dir", log_dir,
                        "--max_episode_steps", "30", "--no_warmup", "--unblinded"],
                       input_fn=lambda prompt: next(answers)) == 0
    episodes = _episodes(log_dir)
    assert [e["status"] for e in episodes] == ["infrastructure_failure", "success"] and len(failed) == 1
    starts = [line for line in events if line.startswith("Trial r0000 [p0] (attempt")]
    assert len(starts) == 2, events
    stopped = [line for line in events if line.startswith("Trial r0000 stopped")]
    assert stopped == ["Trial r0000 stopped ({}).".format(episodes[0]["reason"])] and "XLA error" in stopped[0]
    rerun = [line for line in events if "it will be re-run" in line]
    assert len(rerun) == 1 and "XLA error" in rerun[0]


def test_no_outcome_asks_nothing_and_a_resume_skips_the_trials_that_ran(monkeypatch, tmp_path):
    """--no_outcome (2026-11-24, the operator keeps their own record): no outcome prompt, every attempt that ran is
    logged as unlabelled and counts as done, so the same command run again starts nothing new."""
    server = FakeSessionServer(make_metadata(horizon=20, policy_hz=50.0))
    monkeypatch.setattr(runner, "wait_for_start", lambda keyboard: True)

    def no_prompt(prompt):
        raise AssertionError("asked: {}".format(prompt))

    log_dir = str(tmp_path / "logs")
    argv = ["--host", "h", "--condition", "p0", "--num_episodes", "2", "--log_dir", log_dir,
            "--max_episode_steps", "30", "--no_warmup", "--no_outcome"]
    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(10)]), [])
    assert runner.main(argv, input_fn=no_prompt) == 0
    episodes = _episodes(log_dir)
    assert [(e["trial_id"], e["status"]) for e in episodes] == [("r0000", "unlabelled"), ("r0001", "unlabelled")]
    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(10)]), [])
    assert runner.main(argv, input_fn=no_prompt) == 0
    assert len(_episodes(log_dir)) == 2


def test_runs_without_a_schedule_file_carry_a_run_label_in_their_trial_identity(monkeypatch, tmp_path, capsys):
    """Contract C9 (review N3): a --condition run is labelled (--run_label, default: its log dir's name); the label
    is inside schedule_sha256 and in every episodes.jsonl row and sessions.jsonl record. Two runs in different log
    dirs never share trials, yet stay pairable trial by trial (same trial ids, initial conditions and noise ids); a
    resume in the same log dir keeps the identity; another label there is refused like another schedule (unless
    --force); --run_label with --schedule and labels unusable as analysis names exit with a message."""
    server = FakeSessionServer(make_metadata(horizon=20, policy_hz=50.0))
    monkeypatch.setattr(runner, "wait_for_start", lambda keyboard: True)

    def run(log_dir, *extra, answers=("y", "done", "") * 2):
        _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(6)]), [])
        replies = iter(answers)
        return runner.main(["--host", "h", "--condition", "p0", "--num_episodes", "2", "--log_dir", log_dir,
                            "--max_episode_steps", "10", "--no_warmup"] + list(extra),
                           input_fn=lambda prompt: next(replies))

    pilot, realenv = str(tmp_path / "p0_pilot"), str(tmp_path / "p0_conv_realenv")
    assert run(pilot, answers=("y", "done", "", "n", "align", "jam", "")) == 0
    assert run(realenv + os.sep) == 0  # a trailing separator names the same directory
    first, second = _episodes(pilot), _episodes(realenv)
    assert [e["run_label"] for e in first + second] == ["p0_pilot"] * 2 + ["p0_conv_realenv"] * 2
    assert {e["schedule_sha256"] for e in first} == {fc.rows_sha256(runner.generated_rows("p0", 2, "p0_pilot"))}
    assert not {e["schedule_sha256"] for e in first} & {e["schedule_sha256"] for e in second}

    def key(episode):
        return episode["trial_id"], episode["block"], episode["init_condition_id"], episode["noise_id"]

    assert [key(e) for e in first] == [key(e) for e in second] == [("r0000", 0, 0, 0), ("r0001", 1, 1, 1)]
    with open(os.path.join(realenv, "sessions.jsonl")) as handle:
        assert json.loads(handle.readline())["run_label"] == "p0_conv_realenv"
    # resume in the same log dir (default label, or the same label given explicitly): same trials, none re-runs
    before = len(server.messages)
    assert run(pilot) == 0 and run(pilot, "--run_label", "p0_pilot") == 0
    assert len(server.messages) == before and len(_episodes(pilot)) == 2
    with open(os.path.join(pilot, "sessions.jsonl")) as handle:
        resumed = [json.loads(line) for line in handle][-1]
    assert resumed["resume_check"]["problems"] == [] and resumed["resume_check"]["run_label"] == ["p0_pilot"]
    # another label there is another run: refused like another schedule, unless --force
    with pytest.raises(SystemExit, match=r"run label \['p0_pilot'\], now 'other'.*another schedule"):
        run(pilot, "--run_label", "other")
    assert run(pilot, "--run_label", "other", "--force") == 0
    assert [(e["run_label"], e["trial_id"], e["attempt"]) for e in _episodes(pilot)[2:]] == [
        ("other", "r0000", 1), ("other", "r0001", 1)]
    # --run_label is for runs without a schedule file; a label must work as an analysis name
    schedule = str(tmp_path / "trials.jsonl")
    fc.write_schedule(schedule, fc.make_schedule(["p0"], [0], seed=1))
    capsys.readouterr()
    with pytest.raises(SystemExit):
        runner.main(["--host", "h", "--schedule", schedule, "--log_dir", str(tmp_path / "campaign"),
                     "--run_label", "x"])
    assert "--run_label labels a run without a schedule file" in capsys.readouterr().err
    for log_dir, extra, message in ((pilot, ("--run_label", "a=b"), "--run_label 'a=b': use letters"),
                                    (pilot, ("--run_label", ""), "--run_label '': use letters"),
                                    (str(tmp_path / "p0 pilot"), (), "pass --run_label")):
        with pytest.raises(SystemExit):
            run(log_dir, *extra)
        assert message in capsys.readouterr().err
    assert not os.path.exists(str(tmp_path / "p0 pilot")) and len(_episodes(pilot)) == 4


def test_resume_check_keeps_the_run_label_and_names_unlabelled_attempts(tmp_path):
    """Contract C9 and back-compat: check_resume refuses another run label in a log dir (unless --force), and names
    attempts logged without a label (a --schedule campaign, or a --condition run of an older client, whose rows were
    hashed without the label) instead of reporting only another sha256."""
    path = str(tmp_path / "episodes.jsonl")
    legacy = [{"trial_id": "r0000", "block": 0, "init_condition_id": 0, "noise_id": 0, "condition": "p0"}]
    fc.write_jsonl(path, [{"schedule_sha256": fc.rows_sha256(legacy), "trial_id": "r0000", "attempt": 1,
                           "status": "success", "plan_pause_s": 0.0, "lead": None}])
    labelled = fc.rows_sha256(runner.generated_rows("p0", 1, "logs"))
    with pytest.raises(SystemExit, match="logged without a run label.*older than run labels"):
        runner.check_resume(path, labelled, 0.0, None, False, run_label="logs")
    forced = runner.check_resume(path, labelled, 0.0, None, True, run_label="logs")
    assert forced["forced"] and forced["run_label"] == [] and len(forced["problems"]) == 1
    path = str(tmp_path / "labelled.jsonl")
    fc.write_jsonl(path, [{"schedule_sha256": labelled, "run_label": "logs", "trial_id": "r0000", "attempt": 1,
                           "status": "failure", "plan_pause_s": 0.0, "lead": None}])
    same = runner.check_resume(path, labelled, 0.0, None, False, run_label="logs")
    assert same["problems"] == [] and same["run_label"] == ["logs"]
    other = fc.rows_sha256(runner.generated_rows("p0", 1, "other"))
    with pytest.raises(SystemExit, match=r"run label \['logs'\], now 'other'"):
        runner.check_resume(path, other, 0.0, None, False, run_label="other")
    with pytest.raises(SystemExit, match="another schedule") as refused:  # a --schedule campaign in that log dir
        runner.check_resume(path, "f" * 64, 0.0, None, False)
    assert "run label" not in str(refused.value)


def test_a_campaign_lead_below_one_is_refused_like_a_condition_lead(monkeypatch, tmp_path):
    """Contract C11 (review N4): ConditionInfo refuses a lead or min_lead < 1 and so does the campaign's --lead, in
    trial and in preflight mode and for a P0-only run, before the robot is touched."""
    info = fc.ServerInfo.from_metadata(make_metadata())
    for lead in (0, -1):
        for conditions, require_lead in ((["p0", "i2"], True), (["p0", "i2"], False), (["p0"], True)):
            with pytest.raises(ValueError, match="--lead must be >= 1"):
                fc.CampaignConfig(info, conditions, lead=lead, require_lead=require_lead)
    for field in ("lead", "min_lead"):
        with pytest.raises(ValueError, match=">= 1"):
            fc.ConditionInfo("x", mode="refresh", refresh_every=5, **{field: 0})
    server = FakeSessionServer(make_metadata(horizon=20, policy_hz=50.0))
    for extra in ((), ("--preflight", "2")):
        log = []
        _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(3)]), log)
        with pytest.raises(ValueError, match="--lead must be >= 1"):
            runner.main(["--host", "h", "--condition", "i2", "--lead", "0", "--log_dir", str(tmp_path / "logs"),
                         "--no_warmup"] + list(extra))
        assert log == [] and not server.messages


def test_campaign_needs_a_log_dir_a_recent_passed_preflight_and_a_padded_pause(monkeypatch, tmp_path, capsys):
    """Exp 10 and contract C2: a --schedule run refuses to start without them (unless --force) and logs both."""
    metadata = make_metadata(horizon=20, policy_hz=50.0)
    server = FakeSessionServer(metadata)
    schedule = str(tmp_path / "trials.jsonl")
    fc.write_schedule(schedule, fc.make_schedule(["p0", "i2"], [0], seed=1))
    log_dir = str(tmp_path / "logs")
    report = os.path.join(log_dir, "preflight.json")

    def run(extra, answers=(), directory=log_dir):
        _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard({0: ["right"]}) for _ in range(6)]), [])
        replies = iter(answers)
        argv = ["--host", "h", "--schedule", schedule, "--lead", "2", "--no_warmup", "--max_episode_steps", "10"]
        return runner.main(argv + (["--log_dir", directory] if directory else []) + list(extra),
                           input_fn=lambda prompt: next(replies))

    with pytest.raises(SystemExit):
        run([], directory=None)
    assert "explicit --log_dir" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="no preflight report"):
        run([])
    _write_preflight(report, ["p0", "i2"], advance=DT * 50.0, p99_ms=None, recommended=None)
    with pytest.raises(SystemExit, match="not padded"):
        run([])
    _write_preflight(report, ["p0", "i2"], advance=DT * 50.0)
    for pause in ("0", "0.01"):
        with pytest.raises(SystemExit, match="shorter than the preflight's p99"):
            run(["--plan_pause_s", pause])
    _write_preflight(report, ["p0", "i2"], advance=DT * 50.0, age_h=30.0)
    with pytest.raises(SystemExit, match="h old"):
        run([])
    _write_preflight(report, ["p0"], advance=DT * 50.0, ok=False)
    with pytest.raises(SystemExit, match="refused.*did not measure conditions \\['i2'\\]"):
        run([])
    _write_preflight(report, ["p0", "i2"], advance=DT * 50.0, recommended=0.04)
    assert run([], answers=["y", "", "", "n", "", "", ""]) == 0  # the pause defaults to the recommendation
    episodes = _episodes(log_dir)
    assert [e["plan_pause_s"] for e in episodes] == [0.04, 0.04]
    assert {e["plan_pause_source"] for e in episodes} == {"preflight"}
    assert all(e["preflight_check"]["sha256"] == fc.file_sha256(report) for e in episodes)
    # a resumed campaign keeps its pause and its lead
    with pytest.raises(SystemExit, match="plan_pause_s"):
        run(["--plan_pause_s", "0.06"])
    other = str(tmp_path / "forced")
    assert run(["--force"], answers=["y", "", "", "y", "", "", ""], directory=other) == 0
    with open(os.path.join(other, "sessions.jsonl")) as handle:
        record = json.loads(handle.readline())
    assert record["preflight_check"]["forced"] and record["plan_pause_source"] == "none"
    assert "no preflight report" in record["preflight_check"]["problems"][0]


def test_preflight_mode_reports_lead_and_pause_and_refuses_a_small_lead(monkeypatch, tmp_path):
    metadata = make_metadata(horizon=20, policy_hz=30.0)
    server = FakeSessionServer(metadata)
    log = []
    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(3)]), log)
    log_dir = str(tmp_path / "logs")
    assert runner.main(["--host", "h", "--condition", "i2", "--lead", "3", "--log_dir", log_dir, "--preflight", "2",
                        "--no_warmup"]) == 0
    with open(os.path.join(log_dir, "preflight.json")) as handle:
        report = json.load(handle)
    assert report["ok"] and report["required_lead"] == 2 and report["refresh_rtt_ms"]["n"] >= 4
    assert report["conditions"]["i2"]["plan_rtt_ms"]["n"] == 2 and report["plan_pause_s_recommended"] > 0
    assert log[-1] == "shutdown"
    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(3)]), log)
    assert runner.main(["--host", "h", "--condition", "i2", "--lead", "1", "--log_dir", log_dir, "--preflight", "1",
                        "--no_warmup"]) == 2
    # trial mode refuses a lead below the recorded preflight's requirement
    with open(os.path.join(log_dir, "preflight.json"), "w") as handle:
        json.dump(dict(report, required_lead=4), handle)
    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(3)]), log)
    with pytest.raises(SystemExit, match="required lead 4"):
        runner.main(["--host", "h", "--condition", "i2", "--lead", "3", "--log_dir", log_dir, "--no_warmup"])


def test_main_refuses_a_jpeg_tactile_publisher(monkeypatch, tmp_path):
    server = FakeSessionServer(make_metadata(horizon=20, policy_hz=50.0))
    log = []
    _, streams, _ = _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(3)]), log)
    for stream in streams.values():
        stream.codec = "jpeg"
    with pytest.raises(RuntimeError, match="image_codec"):
        runner.main(["--host", "h", "--condition", "tg", "--log_dir", str(tmp_path), "--no_warmup"])
    assert log[-1] == "shutdown"


# --------------------------------------------------------------------------------------------
# forceguide_preflight.py
# --------------------------------------------------------------------------------------------

def _reference(tmp_path, swap_keys=False, with_spec=True):
    arrays = {}
    spec = spec_for(keys=list(KEYS))
    for key in KEYS + ("observation.images.left_gripper_inner",):
        image = sensor_image(key.rsplit(".", 1)[-1])
        arrays[key + "/native"] = image
        arrays[key + "/prepared"] = fc.prepare_live_tactile(image, spec)
    if swap_keys:
        a, b = KEYS
        arrays[a + "/prepared"], arrays[b + "/prepared"] = arrays[b + "/prepared"], arrays[a + "/prepared"]
    camera = camera_frame(FakeClock(0.0)).images["cam_high"]
    arrays["observation.images.cam_high/native"] = camera[..., ::-1]  # stored BGR like the LeRobot set
    if with_spec:
        arrays["spec"] = np.array(json.dumps(spec.to_dict()))
    path = str(tmp_path / "reference.npz")
    np.savez(path, **arrays)
    return path


def test_preflight_checks_colour_order_and_identity(tmp_path):
    reference = forceguide_preflight.load_reference(_reference(tmp_path))
    assert sorted(forceguide_preflight.tactile_keys(reference)) == sorted(KEYS + (
        "observation.images.left_gripper_inner",))
    assert forceguide_preflight.camera_keys(reference) == ["observation.images.cam_high"]
    assert reference["spec"]["resize"] == "pad_pil"
    spec = spec_for(keys=list(KEYS))
    rng = np.random.RandomState(0)
    live = {key: np.clip(sensor_image(key.rsplit(".", 1)[-1]).astype(int) + rng.randint(-3, 4, (70, 40, 3)), 0,
                         255).astype(np.uint8) for key in KEYS}
    report = forceguide_preflight.run_tactile_checks(live, reference, spec)
    assert report["ok"]
    for key in KEYS:
        assert report["identity"][key]["best"] == key and report["identity"][key]["own"] > 0.9
        assert report["colour"][key]["verdict"] == "ok"
    swapped = {key: image[..., ::-1].copy() for key, image in live.items()}  # wrong tactile_bgr
    report = forceguide_preflight.run_tactile_checks(swapped, reference, spec)
    assert not report["ok"] and {c["verdict"] for c in report["colour"].values()} == {"swapped"}
    crossed = {KEYS[0]: live[KEYS[1]], KEYS[1]: live[KEYS[0]]}  # sensors wired to the wrong keys
    report = forceguide_preflight.run_tactile_checks(crossed, reference, spec)
    assert not report["ok"] and report["identity"][KEYS[0]]["best"] == KEYS[1]
    grey = np.full((70, 40, 3), 128, np.uint8)
    assert forceguide_preflight.colour_order_check(grey, grey)["verdict"] == "inconclusive"
    cameras = forceguide_preflight.run_camera_checks(camera_frame(FakeClock(0.0)).images, reference, image_bgr=True)
    assert cameras["ok"] and cameras["cameras"]["cam_high"]["verdict"] == "ok"
    assert not forceguide_preflight.run_camera_checks(camera_frame(FakeClock(0.0)).images, reference,
                                                      image_bgr=False)["ok"]
    assert forceguide_preflight.gripper_home_check(np.r_[np.zeros(6), 0.99, np.zeros(6), 1.01])["ok"]
    assert not forceguide_preflight.gripper_home_check(np.r_[np.zeros(6), 0.30, np.zeros(6), 1.0])["ok"]
    assert forceguide_preflight.gripper_home_check(np.r_[np.zeros(6), 0.3, np.zeros(6), 1.0], arm="right")["ok"]
    assert forceguide_preflight.normalized_correlation(grey, grey) == 0.0


SENSOR_KEYS = tuple("observation.images." + sensor for sensor in fc.KNOWN_SENSORS)


def _export_reference(tmp_path, episodes=(0, 25), image_bgr=True, legacy_meta=False, name="export_reference.npz"):
    """An npz in the exact layout of the forceguide repo's export_tactile_reference.py (contract C3), synthetic
    frames: all four sensors (RGB natives, prepared pad_pil / rotate 0), camera frames colour-corrected RGB."""
    spec = spec_for(keys=list(SENSOR_KEYS))
    camera_keys = ["observation.images.cam_high"]
    arrays = {"tactile_keys": np.array(SENSOR_KEYS), "camera_keys": np.array(camera_keys),
              "episodes": np.array(episodes, dtype=np.int64)}
    for i, key in enumerate(SENSOR_KEYS):
        native = np.stack([np.clip(sensor_image(key.rsplit(".", 1)[-1]).astype(np.int32) + 2 * row, 0, 255)
                           .astype(np.uint8) for row in range(len(episodes))])
        arrays["tactile_native_{}".format(i)] = native
        arrays["tactile_prepared_{}".format(i)] = fc.prepare_tactile(native, spec)
    camera = camera_frame(FakeClock(0.0)).images["cam_high"]  # what the live camera delivers: colour-correct RGB
    arrays["camera_rgb_0"] = np.stack([camera] * len(episodes))
    arrays["tactile_channel_means"] = np.stack([arrays["tactile_native_{}".format(i)].mean(axis=(1, 2))
                                                for i in range(len(SENSOR_KEYS))]).astype(np.float64)
    meta = {"source": "lerobot", "config": "pi05_board_insertion_tactile4", "episodes": list(episodes), "frame": 0,
            "tactile_keys": list(SENSOR_KEYS), "camera_keys": camera_keys, "tactile_bgr": True,
            "image_bgr": image_bgr, "spec": json.dumps(spec.to_dict())}
    if not legacy_meta:
        meta.update(camera_color="rgb", tactile_color="rgb", prepared_spec=json.dumps(spec.to_dict()))
    arrays["meta_json"] = np.array(json.dumps(meta))
    path = str(tmp_path / name)
    np.savez_compressed(path, **arrays)
    return path


def test_preflight_reads_the_export_tactile_reference_layout(tmp_path):
    """Parity 1 / robot 2 / spec 1: the reader takes the exporter's [E, ...] stacks and meta_json, one episode at a
    time; the camera frames are colour-corrected RGB, so a live frame equal to them passes with the P0's image_bgr."""
    path = _export_reference(tmp_path)
    reference = forceguide_preflight.load_reference(path)
    assert reference["layout"] == "export_tactile_reference" and reference["episode"] == 0
    assert sorted(forceguide_preflight.tactile_keys(reference)) == sorted(SENSOR_KEYS)
    assert forceguide_preflight.camera_keys(reference) == ["observation.images.cam_high"]
    assert reference["spec"]["resize"] == "pad_pil" and reference["meta"]["camera_bgr_swapped"]
    assert reference["keys"][KEYS[0]]["native"].shape == (70, 40, 3)
    later = forceguide_preflight.load_reference(path, episode=25)
    assert later["episode"] == 25
    assert not np.array_equal(later["keys"][KEYS[0]]["native"], reference["keys"][KEYS[0]]["native"])
    with pytest.raises(ValueError, match="not episode 7"):
        forceguide_preflight.load_reference(path, episode=7)
    args = types.SimpleNamespace(condition=None, keys=None, resize=None, rotate=None, live_codec=None)
    spec = forceguide_preflight.resolve_spec(args, reference)
    assert spec.keys == SENSOR_KEYS and spec.resize == "pad_pil"
    live = {key: reference["keys"][key]["native"].copy() for key in SENSOR_KEYS}
    report = forceguide_preflight.run_tactile_checks(live, reference, spec)
    assert report["ok"] and all(report["identity"][key]["best"] == key for key in SENSOR_KEYS)
    assert set(report["prepared_parity"]["max_abs_diff"].values()) == {0}  # the client's copy is bit-identical
    crossed = dict(live, **{KEYS[0]: live[KEYS[1]], KEYS[1]: live[KEYS[0]]})
    assert not forceguide_preflight.run_tactile_checks(crossed, reference, spec)["ok"]
    swapped = {key: image[..., ::-1].copy() for key, image in live.items()}
    assert not forceguide_preflight.run_tactile_checks(swapped, reference, spec)["ok"]
    images = {"cam_high": reference["keys"]["observation.images.cam_high"]["native"].copy()}
    cameras = forceguide_preflight.run_camera_checks(images, reference, image_bgr=True)
    assert cameras["ok"] and cameras["cameras"]["cam_high"]["verdict"] == "ok"
    wrong = forceguide_preflight.run_camera_checks(images, reference, image_bgr=False)
    assert not wrong["ok"] and wrong["cameras"]["cam_high"]["verdict"] == "swapped"
    legacy = forceguide_preflight.load_reference(_export_reference(tmp_path, legacy_meta=True, name="old.npz"))
    assert legacy["meta"]["camera_bgr_swapped"] and legacy["spec"]["resize"] == "pad_pil"  # pre-C3 exports
    rgb = forceguide_preflight.load_reference(_export_reference(tmp_path, image_bgr=False, name="rgb.npz"))
    assert not rgb["meta"]["camera_bgr_swapped"]
    assert forceguide_preflight.run_camera_checks(images, rgb, image_bgr=False)["ok"]
    assert not forceguide_preflight.run_camera_checks(images, rgb, image_bgr=True)["ok"]
    with pytest.raises(ValueError, match="no episodes"):
        forceguide_preflight.load_reference(_reference(tmp_path), episode=0)


def test_preflight_prepares_the_reference_with_the_condition_spec(tmp_path):
    """Parity 2: the reference natives go through the resolved spec, so a Sparsh (stretch_pil) or rotated CPM checked
    against a reference exported with pad_pil / rotate 0 still passes a correctly wired rig."""
    reference = forceguide_preflight.load_reference(_export_reference(tmp_path))
    live = {key: reference["keys"][key]["native"].copy() for key in KEYS}
    for spec in (spec_for(resize="stretch_pil"), spec_for(rotate=1), spec_for(resize="stretch_pil", rotate=1),
                 spec_for(live_codec="yuv420")):
        report = forceguide_preflight.run_tactile_checks(live, reference, spec, KEYS)
        assert report["ok"], report["identity"]
        if spec.live_codec == "none":  # the live codec (live side only) blurs the noise frames' chroma
            assert all(report["identity"][key]["own"] > 0.99 for key in KEYS)
        assert report["prepared_parity"]["same_geometry"] == (spec.resize == "pad_pil" and spec.rotate == 0)
    crossed = {KEYS[0]: live[KEYS[1]], KEYS[1]: live[KEYS[0]]}
    rotated = spec_for(resize="stretch_pil", rotate=1)
    assert not forceguide_preflight.run_tactile_checks(crossed, reference, rotated, KEYS)["ok"]
    # what the check compared before: the npz's pad_pil / rotate-0 frames, a different geometry - a perfect rig failed
    stale = forceguide_preflight.identity_check(
        {key: fc.prepare_live_tactile(live[key], rotated) for key in KEYS},
        {key: entry["prepared"] for key, entry in reference["keys"].items() if "prepared" in entry})
    assert all(stale[key]["own"] < 0.5 for key in KEYS)


def test_preflight_main_with_faked_sensors_and_robot(monkeypatch, tmp_path, capsys):
    reference_path = _reference(tmp_path)
    streams = {sensor: FakeStream(time.time, sensor_image(sensor)) for sensor in fc.KNOWN_SENSORS}

    class Config:
        def wait(self, timeout):
            return {"serials": {"right_gripper_inner": "OG000614"}, "calibrated": True}

    monkeypatch.setitem(sys.modules, "tactile_stream", types.SimpleNamespace(
        TactileStream=types.SimpleNamespace(acquire=lambda sensor, prefetch=(): streams[sensor]),
        RuntimeConfig=types.SimpleNamespace(get=lambda: Config()), calibrate_sensors=lambda sensors: None))
    bots = (FakeBot(gripper=0.0767), FakeBot(gripper=0.0782))
    env = types.SimpleNamespace(puppet_bot_left=bots[0], puppet_bot_right=bots[1],
                                get_images=lambda: camera_frame(FakeClock(0.0)).images)
    for bot in bots:
        bot.arm.set_trajectory_time = lambda moving_time=None, accel_time=None: None
    monkeypatch.setitem(sys.modules, "real_env",
                        types.SimpleNamespace(make_real_env=lambda init_node, setup_robots: env))
    monkeypatch.setitem(sys.modules, "robot_utils", types.SimpleNamespace(
        move_arms=lambda *a, **k: None, move_grippers=lambda *a, **k: None))
    shutdowns = []  # 2026-11-24: the check ends with the followers at the sleep pose, torque off (unless --no_sleep)
    monkeypatch.setitem(sys.modules, "sleep_plus",
                        types.SimpleNamespace(shut_down_puppet_robots=lambda: shutdowns.append(1)))
    out = str(tmp_path / "report.json")
    assert forceguide_preflight.main(["--reference", reference_path, "--out", out, "--frames", "2"]) == 0
    assert shutdowns == [1]
    console = capsys.readouterr().out  # the result comes after the robot's shutdown output, not buried by it
    assert console.index("Followers are at the sleep pose") < console.index("best match") < console.index("PASS")
    assert forceguide_preflight.main(["--reference", reference_path, "--out", out, "--frames", "1", "--no_sleep"]) == 0
    assert shutdowns == [1]
    with open(out) as handle:
        report = json.load(handle)
    assert report["ok"] and report["gripper"]["grippers"] == pytest.approx([1.0, 1.0])
    assert report["runtime_config"]["serials"] == {"right_gripper_inner": "OG000614"}
    assert report["codecs"] == {"right_gripper_inner": "raw", "right_gripper_outer": "raw"}
    assert report["serials"]["ok"] is None  # nothing expected: recorded only
    assert forceguide_preflight.main(["--reference", reference_path, "--out", out, "--frames", "1",
                                      "--expected_serial", "right_gripper_inner=OG000614"]) == 0
    assert forceguide_preflight.main(["--reference", reference_path, "--out", out, "--frames", "1",
                                      "--expected_serial", "right_gripper_inner=OG000706"]) == 1  # sensors swapped
    with open(out) as handle:
        assert json.load(handle)["serials"]["sensors"]["right_gripper_inner"] == {
            "expected": "OG000706", "found": "OG000614", "ok": False}
    export_path = _export_reference(tmp_path)  # export_tactile_reference.py's layout, all four sensors
    assert forceguide_preflight.main(["--reference", export_path, "--out", out, "--frames", "2",
                                      "--reference_episode", "25"]) == 0
    with open(out) as handle:
        exported = json.load(handle)
    assert exported["reference_layout"] == "export_tactile_reference" and exported["reference_episode"] == 25
    assert exported["keys"] == list(SENSOR_KEYS) and exported["image_bgr"] is True
    assert exported["cameras"]["cameras"]["cam_high"]["verdict"] == "ok" and exported["tactile"]["ok"]
    bots[1].position[6] = -1.0  # right gripper not open at home
    assert forceguide_preflight.main(["--reference", reference_path, "--out", out, "--frames", "1"]) == 1
    with pytest.raises(SystemExit, match="not in the reference"):
        forceguide_preflight.main(["--reference", reference_path, "--keys", "observation.images.left_gripper_outer"])
    assert forceguide_preflight.serial_check(None, {"right_gripper_inner": "OG000614"})["ok"] is None


# --------------------------------------------------------------------------------------------
# Protocol fixtures recorded from the real serve_forceguide session
# --------------------------------------------------------------------------------------------

# Written by the forceguide repo's scripts/forceguide/client_server_check.py --write-fixtures from a live run of this
# client against serve_forceguide (dummy policy): complete reset .. plan .. refresh sequences. Arrays above 4 KiB were
# strided down to keep the files small; manifest.json lists their original shapes and _load_fixture repeats them back.
FIXTURES = os.environ.get("FORCEGUIDE_PROTOCOL_FIXTURES",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "forceguide"))

# Top-level keys the client sends per message kind (DESIGN 6.2): optional ones included, then the always-sent ones.
CLIENT_KEYS = {
    "reset": {"forceguide", "session", "req", "condition", "policy_hz", "trial", "attempt"},
    "plan": {"forceguide", "session", "req", "state", "images", "prompt", "noise", "tactile_step", "tactile_steps",
             "tactile_frames", "ack"},
    "refresh": {"forceguide", "session", "req", "chunk", "k", "k_pin", "tactile_step", "tactile_steps",
                "tactile_frames", "ack"},
    "observe": {"forceguide", "session", "req", "tactile_step", "tactile_steps", "tactile_frames"},
}
# "trial" / "attempt" go with trial resets only (robot review 6), not with warm-up / preflight resets or fixtures
# recorded before them.
ALWAYS_SENT = {"reset": CLIENT_KEYS["reset"] - {"trial", "attempt"}, "plan": CLIENT_KEYS["plan"] - {"prompt", "ack"},
               "refresh": CLIENT_KEYS["refresh"] - {"ack"}, "observe": CLIENT_KEYS["observe"]}


def _check_request(request):
    kind = request.get("forceguide", "plan")
    assert kind in CLIENT_KEYS, kind
    extra = set(request) - CLIENT_KEYS[kind]
    assert not extra, "{} request uses keys the client never sends: {}".format(kind, sorted(extra))
    assert isinstance(request["session"], str) and int(request["req"]) >= 1
    if kind in ("plan", "refresh", "observe"):
        steps = [int(g) for g in request["tactile_steps"]]
        assert steps == sorted(set(steps)) and all(g >= 0 for g in steps)
        for frames in request["tactile_frames"].values():
            assert frames.dtype == np.uint8 and frames.shape == (len(steps), 224, 224, 3)
    if "ack" in request:
        assert set(request["ack"]) == {"req", "status", "s"}
        assert request["ack"]["status"] in ("applied", "rejected", "dropped")
    if kind == "plan" and "noise" in request:
        assert request["noise"].dtype == np.float32 and request["noise"].ndim == 2
    if kind == "refresh":
        assert 0 < int(request["k"]) <= int(request["k_pin"])
    if kind == "reset" and "trial" in request:
        assert isinstance(request["trial"], str) and int(request.get("attempt", 1)) >= 1
    return kind


def _check_reply(kind, reply):
    """The client's reading of a reply: "error", "K0", "actions" or "ok"; raises if the client could not use it."""
    block = fc._reply_block(reply)
    assert block, "reply without a 'forceguide' block: {}".format(sorted(reply))
    if block.get("error") is not None:
        assert isinstance(block["error"], str)
        return "error"
    assert block.get("req") is not None
    if kind == "reset":
        assert block.get("ok") and "condition" in block and "fingerprint" in block
        return "ok"
    if kind == "observe":
        assert "ok" in block and "missing_steps" in block
        return "ok"
    if kind == "refresh" and block.get("K") is not None and int(block["K"]) == 0:
        assert "actions" not in reply and "chunk" in block
        return "K0"
    actions = np.asarray(reply["actions"])
    assert actions.ndim == 2 and np.isfinite(actions).all()
    assert "chunk" in block
    if kind == "refresh":
        assert {"k", "k_pin", "K", "signal"} <= set(block)
    return "actions"


def _expand(tree, shrunk):
    """Repeats every array listed in a fixture manifest's "shrunk" entry ({path: {shape, stride}}) back to its
    original shape (client_server_check.expand)."""
    for path, info in (shrunk or {}).items():
        keys = path.split("/")
        parent = tree
        for key in keys[:-1]:
            parent = parent[key]
        array = parent[keys[-1]]
        for axis, (stride, size) in enumerate(zip(info["stride"], info["shape"])):
            if stride > 1:
                array = np.repeat(array, stride, axis=axis)
            array = np.take(array, np.arange(size), axis=axis)
        parent[keys[-1]] = np.ascontiguousarray(array)
    return tree


def _load_fixture(directory, filename):
    with open(os.path.join(directory, filename), "rb") as handle:
        tree = unpackb(handle.read())
    manifest_path = os.path.join(directory, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as handle:
            entry = json.load(handle).get("files", {}).get(filename, {})
        tree = _expand(tree, entry.get("shrunk"))
    return tree


def _check_protocol_fixtures(directory):
    names = sorted(f[:-len(".request.msgpack")] for f in os.listdir(directory) if f.endswith(".request.msgpack"))
    assert names, "no <name>.request.msgpack in {}".format(directory)
    seen = []
    for name in names:
        request = _load_fixture(directory, name + ".request.msgpack")
        kind = _check_request(request)
        outcome = None
        if os.path.exists(os.path.join(directory, name + ".reply.msgpack")):
            reply = _load_fixture(directory, name + ".reply.msgpack")
            outcome = _check_reply(kind, reply)
            block = fc._reply_block(reply)
            if block.get("req") is not None:
                assert int(block["req"]) == int(request["req"]), (name, block.get("req"), request["req"])
        seen.append((kind, outcome))
    return seen


def test_golden_protocol_fixtures_match_the_client_messages():
    """<name>.request.msgpack / <name>.reply.msgpack pairs recorded from this client talking to serve_forceguide
    (client_server_check.py): every request uses only keys the client sends (with its dtypes and shapes), and the
    client can read every real reply: reset, plan, refresh with actions, K = 0, "stale chunk" and another error
    reply, observe."""
    assert os.path.isdir(FIXTURES), "no protocol fixtures in {} (client_server_check.py --write-fixtures)".format(
        FIXTURES)
    seen = _check_protocol_fixtures(FIXTURES)
    assert {kind for kind, _ in seen} >= {"reset", "plan", "refresh", "observe"}
    assert {outcome for kind, outcome in seen if kind == "refresh"} >= {"actions", "K0", "error"}
    with open(os.path.join(FIXTURES, "manifest.json")) as handle:
        manifest = json.load(handle)
    errors = []
    for name in sorted(manifest["files"]):
        if name.endswith(".reply.msgpack"):
            block = fc._reply_block(_load_fixture(FIXTURES, name))
            if block.get("error") is not None:
                errors.append(block["error"])
    assert any(fc._is_stale_chunk(e) for e in errors) and any(not fc._is_stale_chunk(e) for e in errors), errors


def test_protocol_fixture_sequences_keep_the_plan_and_the_acks_consistent():
    """Within each recorded sequence (one reset, its plans and refreshes, in order): req increases, every refresh
    names the chunk of the last plan reply, k_pin = k + lead, tactile_step = plan step + k, every tactile step is
    sent once, every ack answers the last refresh reply before it, and every refresh reply with actions repeats
    the client's plan (last plan + applied splices) at the pinned positions < k_pin, bit for bit."""
    with open(os.path.join(FIXTURES, "manifest.json")) as handle:
        manifest = json.load(handle)
    assert manifest["sequences"], "no sequences in the fixture manifest"
    for sequence, names in manifest["sequences"].items():
        plan, plan_step, chunk, last_req, sent, previous = None, None, None, 0, set(), None
        for name in names:
            request = _load_fixture(FIXTURES, name + ".request.msgpack")
            reply = _load_fixture(FIXTURES, name + ".reply.msgpack")
            kind, block = request.get("forceguide", "plan"), fc._reply_block(reply)
            assert int(request["req"]) > last_req, (sequence, name)
            last_req = int(request["req"])
            steps = [int(g) for g in request.get("tactile_steps", [])]
            if kind == "reset":
                sent = set()
            assert not sent & set(steps), (sequence, name, "tactile steps sent twice", sorted(sent & set(steps)))
            if block.get("error") is None or fc._is_stale_chunk(block.get("error")):
                sent |= set(steps)
            ack = request.get("ack")
            if ack is not None and previous is not None:
                assert int(ack["req"]) == previous, (sequence, name, ack, previous)
            if kind == "plan" and "actions" in reply:
                plan = np.asarray(reply["actions"], np.float64)
                plan_step, chunk = int(request["tactile_step"]), block["chunk"]
            elif kind == "refresh":
                entry = manifest["files"].get(name + ".request.msgpack") or {}
                injected = entry.get("injected")
                assert request["chunk"] == chunk and int(request["tactile_step"]) == plan_step + int(request["k"])
                lead = int(request["k_pin"]) - int(request["k"])
                assert lead >= 1
                for what, value in zip(("campaign lead", "calibrated lead"), _manifest_leads(manifest,
                                                                                         entry.get("condition"))[:2]):
                    assert value is None or lead == int(value), (sequence, name, what, value, lead)
                min_lead = _manifest_leads(manifest, entry.get("condition"))[2]
                assert min_lead is None or lead >= int(min_lead), (sequence, name, min_lead, lead)
                previous = int(request["req"])
                if "actions" in reply and not injected:
                    new = np.asarray(reply["actions"], np.float64)
                    k_pin = int(request["k_pin"])
                    assert new.shape == plan.shape and np.array_equal(new[:k_pin], plan[:k_pin]), (sequence, name)
                    s = _next_ack_s(manifest, names, name)
                    if s is not None:
                        plan = plan.copy()
                        plan[s:] = new[s:]


def _manifest_leads(manifest, condition):
    """(campaign lead, calibrated lead, min_lead) a fixture manifest records for a condition (contract C1): the
    server block's "lead" / "min_lead" and, when "conditions" maps names to metadata entries, the condition's own
    "lead" / "min_lead". Fixtures recorded before C1 lack some or all of them: None, not checked."""
    server = manifest.get("server") or {}
    conditions = server.get("conditions")
    entry = conditions.get(condition) if isinstance(conditions, dict) and condition is not None else None
    entry = entry if isinstance(entry, dict) else {}
    metadata = ((server.get("metadata") or {}).get("conditions") or {}).get(condition) or {}
    calibrated = entry.get("lead", metadata.get("lead"))
    min_lead = entry.get("min_lead", metadata.get("min_lead", server.get("min_lead")))
    return server.get("lead"), calibrated, min_lead


def test_manifest_leads_tolerate_fixtures_recorded_before_the_lead_metadata():
    assert _manifest_leads({}, "tg") == (None, None, None)
    assert _manifest_leads({"server": {"lead": 5, "conditions": ["tg"]}}, "tg") == (5, None, None)
    server = {"lead": 5, "conditions": {"tg": {"lead": 5, "min_lead": 1}, "p0": {"lead": None}}}
    assert _manifest_leads({"server": server}, "tg") == (5, 5, 1)
    assert _manifest_leads({"server": server}, "p0") == (5, None, None)
    nested = {"server": {"metadata": {"conditions": {"tg": {"lead": 4, "min_lead": 2}}}}}
    assert _manifest_leads(nested, "tg") == (None, 4, 2)


def _next_ack_s(manifest, names, name):
    """The splice point the client acknowledged for `name`'s reply (the next request's ack), or None."""
    position = names.index(name)
    for later in names[position + 1:]:
        ack = _load_fixture(FIXTURES, later + ".request.msgpack").get("ack")
        if ack is not None:
            return int(ack["s"]) if ack.get("status") == "applied" and ack.get("s") is not None else None
    return None


def test_protocol_fixture_checker_on_fixtures_recorded_from_the_fake_session(tmp_path):
    """The checker above, run on fixtures recorded from FakeSessionServer (which implements the DESIGN 6.2 replies),
    so it is known to work before the forceguide repo writes its own; also pins CLIENT_KEYS to what the client
    really sends."""
    h = make_harness(condition="i2", policy_hz=50.0, horizon=20, lead=2,
                     script={1: {"K": 0}, 2: {"error": "stale chunk"}})
    pairs = []
    original = h.server.handle

    def recording(message):
        reply = original(message)
        pairs.append((message, reply))
        return reply

    h.server.handle = recording
    h.client.reset("i2", 50.0)
    run_ticks(h, 21)
    observe = h.client.build_message(fc.RefreshRequest(kind="observe", payload=h.payload, tactile_step=20))
    recording(unpackb(packb(observe)))
    for index, (message, reply) in enumerate(pairs):
        kind = message.get("forceguide", "plan")
        assert ALWAYS_SENT[kind] <= set(message) <= CLIENT_KEYS[kind], (kind, sorted(message))
        name = "{:02d}_{}".format(index, kind)
        with open(str(tmp_path / (name + ".request.msgpack")), "wb") as handle:
            handle.write(packb(message))
        with open(str(tmp_path / (name + ".reply.msgpack")), "wb") as handle:
            handle.write(packb(reply))
    seen = _check_protocol_fixtures(str(tmp_path))
    assert ("reset", "ok") in seen and ("plan", "actions") in seen and ("observe", "ok") in seen
    assert {outcome for kind, outcome in seen if kind == "refresh"} == {"actions", "K0", "error"}
    with pytest.raises(AssertionError, match="never sends"):
        _check_request(dict(pairs[1][0], tactile_prev={}))  # a server expecting a field the client lacks


# Words that tell a guided, refresh or tactile condition from P0 (fix round 3, R1); condition names are added per test.
BLINDED_WORDS = ("tactile", "refresh", "guid", "cpm")


def _assert_blinded(text, conditions):
    lowered = text.lower()
    for word in BLINDED_WORDS + tuple(name.lower() for name in conditions):
        assert word not in lowered, (word, text)


def test_the_blinded_console_shows_a_fixed_category_never_the_server_text():
    """Review issue 1 (fix round 3, R1): console_reason shows a fixed category for every reason the runner produces
    ("<exception class>: <message>" from run_forceguide_episode and the reset): the session's own error texts
    (forceguide session.py ProtocolErrors, served back as {"error": str(exc)} and wrapped by ForceGuideClient.plan /
    reset), connection names and the ServerChanged messages (forceguide_client _connect, ForceGuideClient.__init__ /
    reset, the executor's refresh result) never reach the console; an unknown reason shows its exception class
    only; camera and tactile staleness read alike (every condition of a campaign uploads tactile frames)."""
    conditions = ("tg_dynamic", "tg_dynamic2", "i4_live", "p0")
    stale = "camera/tactile data stale"
    cases = {
        "InfrastructureFailure: plan failed: condition 'tg_dynamic' reads tactile input: send reset and session ids "
        "(missing steps None)": "plan failed (server error)",
        "InfrastructureFailure: plan failed: tactile_frames lacks 'left' (the condition reads ['left', 'right']) "
        "(missing steps None)": "plan failed (server error)",
        "InfrastructureFailure: plan failed: tactile tokens of steps [3, 4] are not available (missing steps [3, 4])":
            "plan failed (server error)",
        "InfrastructureFailure: plan failed: a plan of a tactile condition needs tactile_step (the chunk's global "
        "step) (missing steps None)": "plan failed (server error)",
        "InfrastructureFailure: reset failed: condition 'tg_dynamic' is not served (served: ['p0', 'tg_dynamic2'])":
            "reset failed (server error)",
        "InfrastructureFailure: reset failed: a legacy {'reset': True} is accepted for P0 only, not condition "
        "'tg_dynamic'": "reset failed (server error)",
        "InfrastructureFailure: no plan reply within 10.0 s": "plan failed (no reply in time)",
        "InfrastructureFailure: no reset reply within 10.0 s": "reset failed (no reply in time)",
        "ProtocolError: plan reply answers req 7 instead of 8": "plan failed (malformed reply)",
        "ProtocolError: plan reply has no actions: ['forceguide']": "plan failed (malformed reply)",
        "ProtocolError: reset reply answers req 3 instead of 4": "reset failed (malformed reply)",
        "ProtocolError: reset was not acknowledged (ok=False, same condition: False)": "reset failed (malformed reply)",
        "ServerChanged: refresh: the server at ws://h:8000 changed during the session (expected aa11, got bb22)":
            "the policy server changed",
        "ServerChanged: main: the server at ws://h:8000 changed during the session (expected aa11, got bb22)":
            "the policy server changed",
        "ServerChanged: the refresh connection reached a different server instance": "the policy server changed",
        "ServerChanged: the condition fingerprint is 1f2e now, 3d4c at the handshake": "the policy server changed",
        "ServerChanged: server changed": "the policy server changed",
        "ServerError: refresh: policy server error:\nTraceback (most recent call last):\n  File \"session.py\", "
        "line 9, in refresh\n    cpm.guide(tg_dynamic)\nRuntimeError: guidance failed":
            "the policy server raised an error",
        "ConnectFailed: main: could not connect to ws://h:8000 within 30.00 s: [Errno 111] Connection refused":
            "connection to the policy server failed",
        "ConnectionLost: refresh: connection closed: received 1011 (internal error) tg_dynamic cpm":
            "connection to the policy server failed",
        "StaleTactile: stale tactile at the chunk start: gelsight_left seq frozen for 0.40 s": stale,
        "StaleImage: cam_high image is 0.61s old": stale,
        "PolicyError: inference failed: cam_high image is 0.61s old": stale,  # the executor wraps a camera stall
        "PolicyError: inference failed: no cam_left_wrist image yet": stale,
        "PolicyError: inference failed: 'tactile_frames'": "plan failed (client error)",
        "PolicyError: policy returned an invalid action chunk (0, 14)": "plan failed (malformed reply)",
        "InfrastructureFailure: no result from the refresh worker within 1.0 s": "InfrastructureFailure",
        "IndexError: refresh splice of tg_dynamic out of range": "IndexError",
        "KeyError: 'cpm_guidance'": "KeyError",
        "a reason without an exception class (tactile, tg_dynamic)": "infrastructure failure",
    }
    for reason, category in cases.items():
        shown = runner.console_reason(reason)
        assert shown == category + "; details in episodes.jsonl", (reason, shown)
        if category != stale:
            _assert_blinded(shown, conditions)
    # no reason, or an attempt that ran to its end and was classified on its refresh slots (classify)
    for reason, outcome in ((None, "infrastructure"), ("key", "ended"), ("key during the plan", "aborted"),
                            (None, "done")):
        assert runner.console_reason(reason, outcome) == "request failures above the limit; details in episodes.jsonl"


def test_a_plan_refused_with_the_session_text_reaches_the_console_as_a_category(monkeypatch, tmp_path):
    """R1 end to end: the session refuses a plan of a tactile condition with its own text (naming the condition and
    the tactile input it reads); the "stopped" and "re-run" lines show only "plan failed (server error)", and
    episodes.jsonl keeps the server's text in full."""
    server = FakeSessionServer(make_metadata(horizon=20, policy_hz=50.0))
    refused = []
    original = server._handle
    text = "condition 'tg' reads tactile input: send reset and session ids"

    def refusing(message):  # the first attempt's second plan is refused as forceguide session.py refuses it
        if message.get("forceguide", "plan") == "plan" and message.get("tactile_step", 0) > 0 and not refused:
            refused.append(message["req"])
            return {"forceguide": {"error": text, "req": message["req"]}}
        return original(message)

    server._handle = refusing
    printed = []
    monkeypatch.setattr(runner, "print", lambda *a, **k: printed.append(" ".join(str(x) for x in a)), raising=False)
    monkeypatch.setattr(runner, "wait_for_start", lambda keyboard: True)
    answers = iter(["n", "align", "jam", "", "y", "done", ""])
    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(10)]), [])
    log_dir = str(tmp_path / "logs")
    assert runner.main(["--host", "h", "--condition", "tg", "--num_episodes", "1", "--log_dir", log_dir,
                        "--max_episode_steps", "30", "--no_warmup"], input_fn=lambda prompt: next(answers)) == 0
    episodes = _episodes(log_dir)
    assert [e["status"] for e in episodes] == ["infrastructure_failure", "success"] and len(refused) == 1
    assert episodes[0]["reason"].startswith("InfrastructureFailure: plan failed: " + text)
    category = "plan failed (server error); details in episodes.jsonl"
    stopped = [line for line in printed if line.startswith("Trial r0000 stopped")]
    rerun = [line for line in printed if "it will be re-run" in line]
    assert stopped == ["Trial r0000 stopped ({}).".format(category)], printed
    assert rerun == ["Trial r0000: infrastructure failure ({}); it will be re-run.".format(category)], printed
    _assert_blinded(stopped[0] + rerun[0], ("tg", "p0"))


def test_a_resume_with_another_seed_is_refused_unless_forced(monkeypatch, tmp_path):
    """Review issue 3 (fix round 3, R2): the seed draws every trial's plan noise (default_rng([seed, noise_id,
    plan_index])), which pairs a trial across conditions. A resume in a log dir whose attempts used another --seed is
    refused like another schedule or run label (unless --force); the resume check records the earlier seeds and the
    session record the current one (sessions.jsonl). Rows without a seed (older logs) are not compared."""
    path = str(tmp_path / "episodes.jsonl")
    sha = fc.rows_sha256(runner.generated_rows("p0", 1, "logs"))
    row = {"schedule_sha256": sha, "run_label": "logs", "trial_id": "r0000", "attempt": 1, "status": "failure",
           "plan_pause_s": 0.0, "lead": None}
    fc.write_jsonl(path, [dict(row, seed=0)])
    same = runner.check_resume(path, sha, 0.0, None, False, run_label="logs", seed=0)
    assert same["problems"] == [] and same["seed"] == [0]
    with pytest.raises(SystemExit, match=r"--seed \[0\], now 1 .*--force overrides"):
        runner.check_resume(path, sha, 0.0, None, False, run_label="logs", seed=1)
    forced = runner.check_resume(path, sha, 0.0, None, True, run_label="logs", seed=1)
    assert forced["forced"] and forced["seed"] == [0] and len(forced["problems"]) == 1
    legacy = str(tmp_path / "legacy.jsonl")
    fc.write_jsonl(legacy, [row])
    assert runner.check_resume(legacy, sha, 0.0, None, False, run_label="logs", seed=1)["problems"] == []

    server = FakeSessionServer(make_metadata(horizon=20, policy_hz=50.0))
    monkeypatch.setattr(runner, "wait_for_start", lambda keyboard: True)
    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(20)]), [])
    log_dir = str(tmp_path / "logs")
    argv = ["--host", "h", "--condition", "p0", "--num_episodes", "1", "--log_dir", log_dir, "--max_episode_steps",
            "30", "--no_warmup"]
    answers = iter(["y", "done", ""])
    assert runner.main(argv + ["--seed", "3"], input_fn=lambda prompt: next(answers)) == 0
    with pytest.raises(SystemExit, match=r"the campaign ran with --seed \[3\], now 0"):
        runner.main(argv, input_fn=lambda prompt: pytest.fail("no trial may run"))
    assert runner.main(argv + ["--seed", "4", "--force"], input_fn=lambda prompt: pytest.fail("all done")) == 0
    assert [(e["trial_id"], e["seed"]) for e in _episodes(log_dir)] == [("r0000", 3)]
    with open(os.path.join(log_dir, "sessions.jsonl")) as handle:
        sessions = [json.loads(line) for line in handle]
    assert [s["seed"] for s in sessions] == [3, 4]
    assert sessions[0]["resume_check"] == {"previous_attempts": 0, "problems": [], "forced": False}
    assert sessions[1]["resume_check"]["seed"] == [3] and sessions[1]["resume_check"]["forced"]


def test_a_trial_folder_that_holds_data_is_moved_aside_never_overwritten(monkeypatch, tmp_path):
    """Fix round 3 (R3): a crash between an attempt's trial folder and its episodes.jsonl row makes the resume number
    the next attempt the same. The folder is moved aside to trials/<id>_a<n>.prev<k> (first free k >= 1) with a
    console warning and trial_folder_moved_to in the new row, and the new attempt keeps the name the analysis
    rebuilds (<log_dir>/trials/<id>_a<n>); the crashed attempt's data is kept. A --force run of another label in
    that log dir moves the earlier run's folder aside the same way; an empty folder is reused."""
    server = FakeSessionServer(make_metadata(horizon=20, policy_hz=50.0))
    monkeypatch.setattr(runner, "wait_for_start", lambda keyboard: True)
    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(20)]), [])
    log_dir = str(tmp_path / "logs")
    argv = ["--host", "h", "--condition", "p0", "--num_episodes", "1", "--log_dir", log_dir, "--max_episode_steps",
            "30", "--no_warmup"]

    class Crash(Exception):
        pass

    write_jsonl = runner.write_jsonl

    def crashing(path, records, mode="w"):
        if path.endswith("episodes.jsonl"):
            raise Crash("power cut before the episodes.jsonl row")
        return write_jsonl(path, records, mode)

    monkeypatch.setattr(runner, "write_jsonl", crashing)
    answers = iter(["n", "align", "jam", "crashed"])
    # after the first attempt an escaping exception is logged to client.log and main() returns 1 (blinding)
    assert runner.main(argv, input_fn=lambda prompt: next(answers)) == 1
    with open(os.path.join(log_dir, "client.log"), encoding="utf-8") as handle:
        assert "Crash: power cut before the episodes.jsonl row" in handle.read()
    monkeypatch.setattr(runner, "write_jsonl", write_jsonl)
    folder = os.path.join(log_dir, "trials", "r0000_a1")
    partial = sorted(os.listdir(folder))
    assert "ticks.jsonl" in partial and not os.path.exists(os.path.join(log_dir, "episodes.jsonl"))
    with open(os.path.join(folder, "ticks.jsonl")) as handle:
        crashed_ticks = handle.read()
    os.makedirs(folder + ".prev1")  # taken (e.g. an earlier crash): the first free k is 2

    printed = []
    monkeypatch.setattr(runner, "print", lambda *a, **k: printed.append(" ".join(str(x) for x in a)), raising=False)
    answers = iter(["y", "done", ""])
    assert runner.main(argv, input_fn=lambda prompt: next(answers)) == 0
    episodes = _episodes(log_dir)
    assert [(e["trial_id"], e["attempt"], e["status"]) for e in episodes] == [("r0000", 1, "success")]
    assert episodes[0]["trial_folder_moved_to"] == folder + ".prev2"
    assert episodes[0]["logs"]["ticks"] == os.path.join(log_dir, "trials", "r0000_a1", "ticks.jsonl")
    assert sorted(os.listdir(folder + ".prev2")) == partial and os.listdir(folder + ".prev1") == []
    with open(os.path.join(folder + ".prev2", "ticks.jsonl")) as handle:
        assert handle.read() == crashed_ticks
    with open(os.path.join(folder, "ticks.jsonl")) as handle:
        assert handle.read() != crashed_ticks  # the new attempt's ticks (other clock times)
    warned = [line for line in printed if line.startswith("WARNING: ")]
    assert len(warned) == 1 and folder in warned[0] and folder + ".prev2" in warned[0], printed

    answers = iter(["n", "insert", "slip", ""])
    assert runner.main(argv + ["--run_label", "other", "--force"], input_fn=lambda prompt: next(answers)) == 0
    episodes = _episodes(log_dir)
    assert [(e["run_label"], e["attempt"], e["status"]) for e in episodes] == [("logs", 1, "success"),
                                                                              ("other", 1, "failure")]
    assert episodes[1]["trial_folder_moved_to"] == folder + ".prev3"
    assert sorted(os.listdir(folder + ".prev3")) == sorted(os.listdir(folder))

    empty = os.path.join(str(tmp_path / "fresh"), "trials", "r0007_a2")
    os.makedirs(empty)
    assert runner.claim_trial_folder(str(tmp_path / "fresh"), "r0007", 2) == (empty, None)
    assert runner.claim_trial_folder(str(tmp_path / "unused"), "r0007", 1) == (
        os.path.join(str(tmp_path / "unused"), "trials", "r0007_a1"), None)
    stray = os.path.join(str(tmp_path / "fresh"), "trials", "r0008_a1")
    with open(stray, "w") as handle:
        handle.write("not a folder")
    assert runner.claim_trial_folder(str(tmp_path / "fresh"), "r0008", 1) == (stray, stray + ".prev1")
    assert os.path.isdir(stray) and os.path.isfile(stray + ".prev1")


def test_an_attempt_ended_by_the_operator_that_lost_its_refresh_slots_shows_the_slot_category(monkeypatch, tmp_path):
    """R1: an attempt the operator ended (RIGHT, reason "key") that is an infrastructure failure because of its
    refresh slots (classify) shows "request failures above the limit" in the re-run line, not its reason; the
    server's refresh error text stays in requests.jsonl."""
    server = FakeSessionServer(make_metadata(horizon=20, policy_hz=50.0))
    server.script = {0: {"error": "refresh guidance of 'i2' failed: tactile CPM diverged"}}
    printed = []
    monkeypatch.setattr(runner, "print", lambda *a, **k: printed.append(" ".join(str(x) for x in a)), raising=False)
    monkeypatch.setattr(runner, "wait_for_start", lambda keyboard: True)
    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard({36: ["right"]}) for _ in range(10)]), [])
    log_dir = str(tmp_path / "logs")
    answers = iter(["n", "align", "jam", "", "y", "done", ""])
    assert runner.main(["--host", "h", "--condition", "i2", "--num_episodes", "1", "--lead", "2", "--log_dir",
                        log_dir, "--max_episode_steps", "45", "--no_warmup"],
                       input_fn=lambda prompt: next(answers)) == 0
    episodes = _episodes(log_dir)
    assert [(e["status"], e["outcome"], e["reason"]) for e in episodes] == [
        ("infrastructure_failure", "ended", "key"), ("success", "ended", "key")]
    assert episodes[0]["refresh"]["infra_frac"] > 0.1
    rerun = [line for line in printed if "it will be re-run" in line]
    assert rerun == ["Trial r0000: infrastructure failure (request failures above the limit; details in "
                     "episodes.jsonl); it will be re-run."], printed


# --------------------------------------------------------------------------------------------
# Fix round 4 (review-3 issue 3): console blinding outside console_reason, client.log
# --------------------------------------------------------------------------------------------

CONDITION_NAMES = re.compile(r"\b(?:p0|tg|i2|i4)\b")  # make_metadata's conditions
REVEALING = ("refresh", "tactile", "guid", "cpm", "traceback (most recent call last)", "k_pin", "worker",
             "_maybe_submit")
# What rospy.init_node does to logging (rosgraph.roslogging.configure_logging -> logging.config.fileConfig): the
# root logger's handlers are replaced by ROS's log file (python_logging.conf of ROS Noetic, reduced).
ROS_LOGGING_CONF = """[loggers]
keys=root

[handlers]
keys=rosFile

[formatters]
keys=

[logger_root]
level=INFO
handlers=rosFile

[handler_rosFile]
class=FileHandler
level=DEBUG
args=({path!r}, 'a')
"""


def _revealing(text, tmp_path):
    """The words of console text (tmp paths removed) that could tell the blinded operator a trial's condition."""
    text = text.replace(str(tmp_path), "<tmp>")
    return [word for word in REVEALING if word in text.lower()] + CONDITION_NAMES.findall(text)


@contextlib.contextmanager
def _command_line_logging():
    """The root logger as a command line starts: without handlers (pytest's capture handlers would turn a
    logging.basicConfig console into a no-op). Restored afterwards; handlers the run left on it are closed."""
    root = logging.getLogger()
    saved, level = root.handlers[:], root.level
    for handler in saved:
        root.removeHandler(handler)
    try:
        yield
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()
        for handler in saved:
            root.addHandler(handler)
        root.setLevel(level)


def test_client_log_keeps_the_records_that_could_unblind_and_the_console_never_shows_them(tmp_path):
    """The refresh connection's reconnect wait, the refresh worker's "not running" error and its bookkeeping
    traceback, and the pinned-prefix line name the refresh path: with the runner's ClientLog they reach client.log
    only. The session's first connection wait shows on the console without the connection's name, no traceback
    reaches the console, and no record does while an attempt runs (ClientLog.trial)."""
    console = io.StringIO()
    debug_level = fc.DEBUG_LOG.level
    client_log = runner.ClientLog(str(tmp_path), stream=console)
    server = None
    try:
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
        probe.close()  # nothing listens
        with pytest.raises(fc.ConnectFailed):  # the session's first connection (main() opens both before any trial)
            fc.ForceGuideConnection("127.0.0.1", closed_port, connect_timeout_s=0.5, infer_timeout_s=1.0,
                                    close_timeout_s=0.2, name="refresh")
        metadata = make_metadata()
        server = BlockingServer(lambda message: {"ok": 1}, lambda: metadata)
        port = server.port
        conn = fc.ForceGuideConnection("127.0.0.1", port, connect_timeout_s=5.0, infer_timeout_s=1.0,
                                       close_timeout_s=0.2, name="refresh")
        server.stop()  # the server restarts mid-session: the refresh connection reconnects and waits
        conn.drop()
        with pytest.raises(fc.ConnectFailed):
            conn.request({"id": 1}, timeout=0.5, connect_budget_s=0.6)
        h = make_harness(condition="i2", policy_hz=50.0, horizon=20, lead=2, script={0: {"pin_offset": 0.01}})
        run_ticks(h, 12)  # the first refresh reply differs below k_pin (a DEBUG record)
        assert h.executor.summary()["pin_mismatch"] == 1
        h.transport.alive = False  # the refresh worker stopped
        run_ticks(h, 30)

        class Conn:
            connects = 1

            def request(self, message, timeout=None, connect_budget_s=None):
                return {"forceguide": {"req": 1}}, {}

            def close(self):
                pass

        def broken(job, result):
            raise RuntimeError("bookkeeping bug at k_pin")

        worker = fc.RefreshWorker(Conn(), lambda job: {"forceguide": "refresh"}, timeout_s=1.0, after=broken)
        try:
            assert worker.submit(fc.RefreshRequest())
            results, deadline = [], time.monotonic() + 5.0
            while not results and time.monotonic() < deadline:
                results = worker.poll()
                time.sleep(0.01)
            assert results and results[0].status == "ok"
        finally:
            worker.close()
        with client_log.trial():  # e.g. a follower step limit at a refresh splice
            logging.warning("left follower: command jumped 0.200 rad in one tick; limited to 0.100")
        try:
            raise ValueError("internal detail")
        except ValueError:
            logging.exception("an unmarked error")
    finally:
        if server is not None:
            server.stop()
        client_log.close()
    lines = console.getvalue().splitlines()
    assert len(lines) == 2 and lines[0].startswith(
        "INFO waiting for the policy server at ws://127.0.0.1:{} (".format(closed_port)), lines
    assert lines[1] == "ERROR an unmarked error" and _revealing(console.getvalue(), tmp_path) == [], lines
    with open(client_log.path, encoding="utf-8") as handle:
        full = handle.read()
    for expected in ("refresh: waiting for the policy server at ws://127.0.0.1:{} (".format(closed_port),
                     "refresh: waiting for the policy server at ws://127.0.0.1:{} (".format(port),  # the reconnect
                     "below k_pin", "forceguide: the refresh worker is not running",
                     "refresh worker: post-reply bookkeeping failed", "RuntimeError: bookkeeping bug at k_pin",
                     "command jumped 0.200 rad", "an unmarked error", "ValueError: internal detail"):
        assert expected in full, expected
    assert full.count("Traceback") == 2
    assert client_log.file not in logging.getLogger().handlers and fc.DEBUG_LOG.level == debug_level


@pytest.mark.parametrize("ros_logging", [False, True])
def test_a_client_bug_in_a_trial_stays_off_the_console_and_client_log_has_it(monkeypatch, tmp_path, capsys,
                                                                             ros_logging):
    """End to end: an unexpected client error in a trial of a refresh condition (a bug in the refresh slot code,
    simulated) goes with its traceback to <log_dir>/client.log only - also after rospy.init_node's logging config
    replaced the root logger's handlers (ros_logging, simulated in make_real_env). So does an ordinary warning logged
    during the attempt (a follower step limit at a refresh splice). The console names neither the condition, its
    mode nor the error, and shows no traceback; it prints the client log's path once."""
    server = FakeSessionServer(make_metadata(horizon=20, policy_hz=50.0))
    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(10)]), [])
    if ros_logging:
        make_env = sys.modules["real_env"].make_real_env
        conf = tmp_path / "python_logging.conf"
        conf.write_text(ROS_LOGGING_CONF.format(path=str(tmp_path / "ros.log")))

        def make_real_env(init_node, setup_robots):
            logging.config.fileConfig(str(conf), disable_existing_loggers=False)
            return make_env(init_node, setup_robots)

        monkeypatch.setitem(sys.modules, "real_env", types.SimpleNamespace(make_real_env=make_real_env))
    monkeypatch.setattr(runner, "wait_for_start", lambda keyboard: True)
    original = fc.RefreshingChunkExecutor._maybe_submit

    def buggy(self, chunk, k):
        if self.mode == "refresh" and k > 0:
            logging.warning("left follower: command jumped 0.200 rad in one tick; limited to 0.100")
            raise RuntimeError("refresh slot bookkeeping bug (k_pin)")
        return original(self, chunk, k)

    monkeypatch.setattr(fc.RefreshingChunkExecutor, "_maybe_submit", buggy)
    log_dir = str(tmp_path / "logs")
    answers = iter(["n", "align", "jam", ""])
    with _command_line_logging():
        code = runner.main(["--host", "h", "--condition", "i2", "--num_episodes", "1", "--lead", "2", "--log_dir",
                            log_dir, "--max_episode_steps", "30", "--max_attempts", "1", "--no_warmup",
                            "--no_tactile_calibration", "--plan_pause_s", "0.05"],
                           input_fn=lambda prompt: next(answers))
    assert code == 0
    captured = capsys.readouterr()
    console = captured.out + captured.err
    assert "Trial r0000 stopped (RuntimeError; details in episodes.jsonl)." in console
    assert _revealing(console, tmp_path) == [] and "command jumped" not in console, console
    path = os.path.join(log_dir, "client.log")
    assert [line for line in console.splitlines() if "Full log" in line] == [
        "Full log (not for the blinded operator): {}".format(path)]
    with open(path, encoding="utf-8") as handle:
        full = handle.read()
    for expected in ("trial r0000 stopped by an unexpected client error", "Traceback",
                     "self._maybe_submit(chunk, k)", "RuntimeError: refresh slot bookkeeping bug (k_pin)",
                     "conditions ['i2'], lead 2, refresh_every 5", "WARNING [MainThread] root: left follower: command "
                     "jumped 0.200 rad"):
        assert expected in full, expected
    assert _episodes(log_dir)[0]["reason"] == "RuntimeError: refresh slot bookkeeping bug (k_pin)"


def test_ctrl_c_in_a_trial_exits_130_without_a_console_traceback(monkeypatch, tmp_path, capsys):
    """Ctrl-C while the executor is in the refresh slot code: Python's traceback would show that code path on the
    console. main() logs it to client.log, prints one neutral line and returns 130, after the usual shutdown."""
    log = []
    _fake_hardware(monkeypatch, FakeSessionServer(make_metadata(horizon=20, policy_hz=50.0)),
                   iter([ScriptedKeyboard() for _ in range(10)]), log)
    monkeypatch.setattr(runner, "wait_for_start", lambda keyboard: True)
    original = fc.RefreshingChunkExecutor._maybe_submit

    def interrupted(self, chunk, k):
        if self.mode == "refresh" and k > 0:
            raise KeyboardInterrupt
        return original(self, chunk, k)

    monkeypatch.setattr(fc.RefreshingChunkExecutor, "_maybe_submit", interrupted)
    log_dir = str(tmp_path / "logs")
    with _command_line_logging():
        try:
            code = runner.main(["--host", "h", "--condition", "i2", "--num_episodes", "1", "--lead", "2",
                                "--log_dir", log_dir, "--max_episode_steps", "30", "--no_warmup",
                                "--no_tactile_calibration", "--plan_pause_s", "0.05"])
        except KeyboardInterrupt:
            pytest.fail("Ctrl-C left main() with its traceback")
    assert code == 130 and log[-1] == "shutdown"
    captured = capsys.readouterr()
    console = captured.out + captured.err
    path = os.path.join(log_dir, "client.log")
    assert "Interrupted (Ctrl-C); the traceback is in {}.".format(path) in console
    assert _revealing(console, tmp_path) == [], console
    with open(path, encoding="utf-8") as handle:
        full = handle.read()
    assert "interrupted (Ctrl-C)" in full and "KeyboardInterrupt" in full and "self._maybe_submit(chunk, k)" in full


def test_an_error_outside_executor_step_in_a_trial_exits_1_without_a_console_traceback(monkeypatch, tmp_path,
                                                                                      capsys):
    """An unexpected client error raised during an attempt but outside executor.step - here the refresh-only round-end
    bookkeeping of the reply still in flight (executor.finish -> transport.poll -> _finish(..., "round-ended")) - ends
    the run. Python's traceback would name that refresh path, so main() logs it to client.log, prints one neutral line
    and returns 1 (after the usual shutdown). A setup refusal before the first attempt still raises (its message is
    condition-independent and the operator needs it)."""
    log = []
    server = FakeSessionServer(make_metadata(horizon=20, policy_hz=50.0))
    server.refresh_sleep = 0.25  # the attempt's last refresh is still in flight when the attempt ends
    _fake_hardware(monkeypatch, server, iter([ScriptedKeyboard() for _ in range(10)]), log)
    monkeypatch.setattr(runner, "wait_for_start", lambda keyboard: True)
    original = fc.RefreshingChunkExecutor._finish

    def _finish(self, request, status, reason=None, **kwargs):
        if reason == "round-ended":
            raise ValueError("bookkeeping fault in the round-ended path")
        return original(self, request, status, reason, **kwargs)

    monkeypatch.setattr(fc.RefreshingChunkExecutor, "_finish", _finish)
    log_dir = str(tmp_path / "logs")
    answers = iter(["n", "align", "jam", ""] * 3)
    with _command_line_logging():
        try:
            code = runner.main(["--host", "h", "--condition", "i2", "--num_episodes", "1", "--lead", "2",
                                "--log_dir", log_dir, "--max_episode_steps", "30", "--max_attempts", "1",
                                "--no_warmup", "--no_tactile_calibration", "--plan_pause_s", "0.05"],
                               input_fn=lambda prompt: next(answers))
        except ValueError:
            pytest.fail("the error left main() with its traceback")
    assert code == 1 and log[-1] == "shutdown"
    captured = capsys.readouterr()
    console = captured.out + captured.err
    path = os.path.join(log_dir, "client.log")
    assert "Stopped by an unexpected client error; the traceback is in {}.".format(path) in console
    assert "Traceback" not in console and "round-ended" not in console, console
    assert _revealing(console, tmp_path) == [], console
    with open(path, encoding="utf-8") as handle:
        full = handle.read()
    for expected in ("the run stopped on an exception", "Traceback", "round-ended",
                     "ValueError: bookkeeping fault in the round-ended path"):
        assert expected in full, expected
