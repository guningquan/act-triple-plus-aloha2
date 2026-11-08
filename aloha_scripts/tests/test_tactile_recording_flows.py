"""The recording entry points end to end on fake hardware, with real tactile frames streaming in:
record_episodes_aloha2.capture_one_episode, record_episodes_aloha2_vtla.capture_one_episode,
hil_rollout.main (hdf5 + tactile), and the LiveStreams / RGBStreams snapshot logic."""

import csv
import json
import shutil
import sys
import threading
import time
import types

import h5py
import numpy as np
import pytest
from fakes import FakeBot

import tactile_stream
from tactile_stream import TactileStream
from test_tactile_pipeline import _force, _image, _payload

SENSORS = ["left_gripper_outer", "left_gripper_inner", "right_gripper_outer", "right_gripper_inner"]
CAMERAS = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
CONFIG = {"runtime_configs": {name: name.encode() for name in SENSORS}, "sdk_version": "1.7.0",
          "serials": {name: "OG" for name in SENSORS}, "calibrated": {name: True for name in SENSORS},
          "calibrated_at": 1.0, "exported_at": 1.0, "settings": {"image_codec": "raw"}}


@pytest.fixture
def feeder():
    """Offline TactileStreams for all four sensors (what acquire() returns), fed at 60 Hz by a thread
    until stop_after seconds (None = until the test ends)."""
    state = {"stop": threading.Event(), "threads": [], "topics": []}

    def start(codec="raw", stop_after=None, rate=60.0):
        streams = []
        for name in SENSORS:
            topic = tactile_stream.topic_for(name)
            stream = TactileStream._streams.get(topic) or TactileStream(topic, subscribe=False)
            TactileStream.register(stream)
            state["topics"].append(topic)
            streams.append(stream)

        def run():
            seq, begin = 0, time.time()
            while not state["stop"].is_set():
                if stop_after is not None and time.time() - begin > stop_after:
                    return
                seq += 1
                for stream in streams:
                    stream.feed(_payload(time.time(), seq, codec, force=_force(0.01)))
                time.sleep(1.0 / rate)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        state["threads"].append(thread)
        for stream in streams:
            assert stream.wait_for_frame(2.0) is not None
        return streams

    yield start
    state["stop"].set()
    for thread in state["threads"]:
        thread.join(1.0)
    with TactileStream._registry_lock:
        for topic in state["topics"]:
            TactileStream._streams.pop(topic, None)


class _Keyboard:
    """KeyboardControl stand-in: `events` maps poll count -> events."""

    def __init__(self, events=None):
        self.events = events or {}
        self.count = 0
        self.enabled = True

    def poll(self):
        self.count += 1
        return list(self.events.get(self.count, []))

    def drain(self):
        pass

    def close(self):
        pass


class _JointRecorder:
    qpos = [0.0] * 8
    qvel = [0.0] * 8
    effort = [0.0] * 8

    @property
    def secs(self):
        return int(time.time())

    @property
    def nsecs(self):
        return int((time.time() % 1) * 1e9)


class _ImageRecorder:
    camera_names = CAMERAS

    def get_images(self):
        now = time.time()
        for camera in CAMERAS:
            setattr(self, camera + "_secs", int(now))
            setattr(self, camera + "_nsecs", int((now % 1) * 1e9))
        return {camera: np.full((48, 64, 3), 100, np.uint8) for camera in CAMERAS}


def _real_env():
    """A real_env.RealEnv (its own get_observation/step/reset) over fake robots and recorders."""
    import real_env
    from interbotix_xs_msgs.msg import JointSingleCommand

    env = object.__new__(real_env.RealEnv)
    env.puppet_bot_left, env.puppet_bot_right = FakeBot(), FakeBot()
    env.recorder_left, env.recorder_right = _JointRecorder(), _JointRecorder()
    env.image_recorder = _ImageRecorder()
    env.gripper_command = JointSingleCommand(name="gripper")
    env.tactile = None
    return env


def _patch_recorder(monkeypatch, rec, env, keyboard, calls):
    monkeypatch.setattr(rec, "InterbotixManipulatorXS", lambda **kwargs: FakeBot())
    monkeypatch.setattr(rec, "make_real_env", lambda init_node, setup_robots: env)
    monkeypatch.setattr(rec, "opening_ceremony", lambda *args, **kwargs: None)
    monkeypatch.setattr(rec, "KeyboardControl", lambda: keyboard)
    monkeypatch.setattr(rec, "wait_for_start", lambda keyboard: True)
    for name in ("torque_on", "torque_off"):
        monkeypatch.setattr(rec, name, lambda bot: None)
    monkeypatch.setattr(rec, "move_grippers", lambda *args, **kwargs: None)
    monkeypatch.setattr(rec, "calibrate_sensors", lambda sensors: calls.append(sorted(sensors)) or CONFIG)


def test_record_episode_aligns_tactile_with_each_observation(tmp_path, monkeypatch, feeder):
    import check_episode_timing
    import record_episodes_aloha2 as rec

    feeder("raw")
    env, calls = _real_env(), []
    _patch_recorder(monkeypatch, rec, env, _Keyboard({41: ["right"]}), calls)
    status = rec.capture_one_episode(rec.DT, 200, CAMERAS, str(tmp_path), "episode_100", True,
                                     haptics_args={"no_force_feedback": True},
                                     tactile_sensors=SENSORS, tactile_outputs=["rectify", "force"])
    assert status == "save" and env.tactile is None
    assert calls == [sorted(tactile_stream.topic_for(name) for name in SENSORS)]   # zeroed once, first
    with h5py.File(str(tmp_path / "episode_100.hdf5"), "r") as f:
        steps = f["/action"].shape[0]
        assert steps == 40
        ts = f["/observations/timestamps"]
        local = ts["local_time"][()]
        for name in SENSORS:
            read_end = ts["tactile"][name]["read_end"][()]
            assert len(read_end) == steps
            # taken with the observation: never after it, and fresh
            assert np.all(read_end <= local + 1e-6) and np.all(local - read_end < 0.2)
            assert f["/observations/tactile/{}/force".format(name)].shape == (steps, 35, 20, 3)
            assert f["/observations/tactile/{}/rectify".format(name)].attrs["jpeg_quality"] == 90
        assert ts["control_start"].shape == (steps,) and ts["images"]["cam_high"].shape == (steps,)
        assert f["/metadata/tactile"].attrs["sdk_version"] == "1.7.0"
        assert f["/metadata/tactile/runtime_config/left_gripper_outer"][()].tobytes() == b"left_gripper_outer"
    assert check_episode_timing.main([str(tmp_path / "episode_100.hdf5")]) == 0


def test_record_episode_redoes_when_tactile_goes_stale(tmp_path, monkeypatch, feeder):
    import record_episodes_aloha2 as rec

    feeder("raw", stop_after=0.4)                    # the publisher dies 0.4 s in
    env, calls = _real_env(), []
    _patch_recorder(monkeypatch, rec, env, _Keyboard(), calls)
    status = rec.capture_one_episode(rec.DT, 500, CAMERAS, str(tmp_path), "episode_101", True,
                                     haptics_args={"no_force_feedback": True},
                                     tactile_sensors=SENSORS, tactile_outputs=["rectify", "force"])
    assert status == "redo" and env.tactile is None
    assert not (tmp_path / "episode_101.hdf5").exists()


def _vtla_args(**overrides):
    args = dict(overwrite=False, save_force=False, stream_timeout=1.0, max_stream_age=0.5, max_tactile_age=0.2,
                no_tactile_calibration=False, start_open_ratio=1.0, no_force_feedback=True, no_torque_gate=False,
                force_baseline_samples=0, max_timesteps=40, aloha_action_space="joint_position", task="t",
                task_name="aloha2_tactile_test")
    args.update(overrides)
    return types.SimpleNamespace(**args)


def _vtla_legacy(decisions):
    env = types.SimpleNamespace(puppet_bot_left=FakeBot(), puppet_bot_right=FakeBot(), step=lambda action: None)
    return types.SimpleNamespace(
        InterbotixManipulatorXS=lambda **kwargs: FakeBot(), make_real_env=lambda init_node, setup_robots: env,
        KeyboardControl=lambda: _Keyboard(), opening_ceremony=lambda *a, **k: None, wait_for_start=lambda kb: True,
        wait_for_decision=lambda kb, prompt: decisions.append(prompt) or "save", torque_on=lambda bot: None,
        torque_off=lambda bot: None, move_grippers=lambda *a, **k: None, get_action=lambda *a: np.full(14, 0.5),
        rospy=types.SimpleNamespace(is_shutdown=lambda: False))


def _fake_live_streams(stale_after=None):
    import record_episodes_aloha2_vtla as vtla

    class FakeLiveStreams:
        def __init__(self, sources, arm_names, calibration, save_force, max_tactile_age=None):
            self.sources, self.max_tactile_age, self.count = sources, max_tactile_age, 0
            self.tactile_names = list(vtla.TACTILE_SENSORS)

        def wait_ready(self, timeout, max_age):
            pass

        def snapshot(self, max_age):
            self.count += 1
            if stale_after is not None and self.count > stale_after:
                raise RuntimeError("VTLA streams unavailable: missing=[], stale=['left_gripper_outer']")
            streams = {}
            for i, name in enumerate(sorted(set(self.sources.values()))):
                entry = {"image": np.full((32, 48, 3), 10 * i, np.uint8), "received": time.monotonic(),
                         "stamp": time.time(), "sequence": self.count}
                if name in vtla.TACTILE_SENSORS:
                    entry["data_age"] = 0.005
                streams[name] = entry
            for side in ("left", "right"):
                streams["joint_" + side] = {"state": np.full(7, 0.5), "received": time.monotonic(), "stamp": 0.0,
                                            "sequence": self.count}
            return time.monotonic(), streams

        def close(self):
            pass

    return FakeLiveStreams


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_vtla_capture_saves_runtime_config_and_redoes_on_stale(tmp_path, monkeypatch):
    import record_episodes_aloha2_vtla as vtla

    zeroed = []
    monkeypatch.setattr(vtla, "calibrate_sensors", lambda sensors: zeroed.append(sorted(sensors)) or CONFIG)
    sources = vtla.video_sources("cam_high")

    monkeypatch.setattr(vtla, "LiveStreams", _fake_live_streams())
    decisions = []
    status = vtla.capture_one_episode(_vtla_args(), _vtla_legacy(decisions), {"episode_len": 40},
                                      tmp_path / "episode_0", sources)
    assert status == "save" and zeroed == [sorted(vtla.TACTILE_SENSORS)]
    episode = tmp_path / "episode_0"
    assert (episode / "tactile_runtime_config" / "left_gripper_outer.tar.gz").read_bytes() == b"left_gripper_outer"
    meta = json.loads((episode / "metadata.json").read_text())
    assert meta["tactile_runtime_config"]["sdk_version"] == "1.7.0"
    rows = list(csv.DictReader((episode / "timestamps.csv").open()))
    assert float(rows[3]["left_gripper_outer_data_age_s"]) == pytest.approx(0.005)

    monkeypatch.setattr(vtla, "LiveStreams", _fake_live_streams(stale_after=5))
    status = vtla.capture_one_episode(_vtla_args(), _vtla_legacy([]), {"episode_len": 40},
                                      tmp_path / "episode_1", sources)
    assert status == "redo" and not (tmp_path / "episode_1").exists()


def test_hil_hdf5_records_tactile_through_the_env(tmp_path, monkeypatch, feeder):
    import constants
    import hil_devices
    import hil_rollout
    import policy_client
    import record_episodes_aloha2 as real_recorder
    from test_hil_main import FakeArmHaptics, FakeClient, FakeEnv, ScriptedHilInput, make_legacy

    feeder("jpeg")

    class TactileEnv(FakeEnv):
        tactile = None

        def get_observation(self):
            obs = FakeEnv.get_observation(self)
            obs["timestamps"] = {"local_time": time.time()}
            if self.tactile is not None:
                obs["tactile"] = self.tactile.snapshot()
            return obs

    env, masters, log = TactileEnv(), (FakeBot(gripper=0.6), FakeBot(gripper=0.6)), []
    legacy = make_legacy(env, masters, log)
    legacy.TactileRecorder = real_recorder.TactileRecorder
    monkeypatch.setitem(sys.modules, "record_episodes_aloha2", legacy)
    monkeypatch.setitem(constants.TASK_CONFIGS, "tactile_hil_test", {
        "dataset_dir": str(tmp_path), "episode_len": 200, "camera_names": ["cam_high", "cam_left_wrist"],
        "tactile_sensors": SENSORS, "tactile_outputs": ["rectify", "force"]})
    zeroed = []
    monkeypatch.setattr(hil_rollout, "calibrate_sensors", lambda sensors: zeroed.append(sorted(sensors)) or CONFIG)
    monkeypatch.setattr(policy_client, "PolicyClient", FakeClient)
    ScriptedHilInput.script = {0: ["pause"], 40: ["end"], 41: ["pause"]}
    monkeypatch.setattr(hil_devices, "HilInput", ScriptedHilInput)
    FakeArmHaptics.instances = []
    hil_rollout.main(["--task_name", "tactile_hil_test", "--policy", "openpi", "--host", "server",
                      "--output_dir", str(tmp_path), "--max_timesteps", "100"])
    assert zeroed == [sorted(tactile_stream.topic_for(name) for name in SENSORS)] and env.tactile is None
    with h5py.File(str(tmp_path / "episode_0.hdf5"), "r") as f:
        steps = len(f["hil/phase"])
        assert steps > 10
        for name in SENSORS:
            assert f["/observations/timestamps/tactile/{}/read_end".format(name)].shape == (steps,)
            lens = f["/observations/tactile/compress_len"][()]
            assert lens.shape[1] == steps and lens.min() > 0              # published JPEGs, stored as is
        assert f["/metadata/tactile"].attrs["sdk_version"] == "1.7.0"
        assert f["/observations/timestamps/local_time"].shape == (steps,)


def test_live_streams_and_rgb_streams_snapshot_logic(feeder):
    import record_episodes_aloha2_vtla as vtla
    import run_vtla_policy

    streams = feeder("jpeg")
    by_name = dict(zip(SENSORS, streams))
    sources = vtla.video_sources("cam_high")

    live = object.__new__(vtla.LiveStreams)
    live.lock = threading.Lock()
    live.names = list(sources.values()) + ["joint_left", "joint_right"]
    live.save_force, live.max_tactile_age = False, 0.2
    live.tactile_names, live.tactile = list(vtla.TACTILE_SENSORS), {n: by_name[n] for n in vtla.TACTILE_SENSORS}
    live.latest = {name: {"image": np.zeros((48, 64, 3), np.uint8), "received": time.monotonic(), "stamp": 0.0,
                          "sequence": 1} for name in live.names if name.startswith("cam_")}
    live.latest.update({"joint_" + s: {"state": np.zeros(7), "received": time.monotonic(), "stamp": 0.0,
                                       "sequence": 1} for s in ("left", "right")})
    _, snapshot = live.snapshot(0.5)
    entry = snapshot["left_gripper_outer"]
    assert entry["image"].shape == (70, 40, 3) and entry["data_age"] < 0.2 and entry["stamp"] > 0
    assert np.abs(entry["image"].astype(int) - _image()[..., ::-1]).mean() < 3   # BGR for the video
    with pytest.raises(RuntimeError, match="stale"):
        live.snapshot(0.5, max_tactile_age=-1.0)

    rgb = object.__new__(run_vtla_policy.RGBStreams)
    rgb.camera_names, rgb.tactile_names = ["cam_high"], sorted(SENSORS)
    rgb.max_tactile_age, rgb._tactile = 0.2, by_name
    rgb._latest = {"cam_high": (np.zeros((48, 64, 3), np.uint8), time.monotonic())}
    frames = rgb.snapshot(0.5)
    assert np.abs(frames["left_gripper_outer"].astype(int) - _image()).mean() < 3   # true RGB for the model
    baseline = run_vtla_policy.capture_tactile_baseline(rgb, 0.5)                   # waits for fresh frames
    assert sorted(baseline) == sorted(SENSORS)


def test_record_episode_without_zeroing_keeps_the_latched_config(tmp_path, monkeypatch, feeder):
    import record_episodes_aloha2 as rec

    feeder("raw")
    env, calls = _real_env(), []
    _patch_recorder(monkeypatch, rec, env, _Keyboard({6: ["right"]}), calls)
    monkeypatch.setattr(rec, "current_runtime_config", lambda: dict(CONFIG, calibrated_at=None))
    status = rec.capture_one_episode(rec.DT, 50, CAMERAS, str(tmp_path), "episode_102", True,
                                     haptics_args={"no_force_feedback": True, "no_tactile_calibration": True},
                                     tactile_sensors=SENSORS, tactile_outputs=["force"])
    assert status == "save" and calls == []                       # --no_tactile_calibration: not zeroed
    with h5py.File(str(tmp_path / "episode_102.hdf5"), "r") as f:
        assert f["/action"].shape[0] == 5
        assert np.isnan(f["/metadata/tactile"].attrs["calibrated_at"])
        assert f["/metadata/tactile/runtime_config/right_gripper_inner"][()].tobytes() == b"right_gripper_inner"
