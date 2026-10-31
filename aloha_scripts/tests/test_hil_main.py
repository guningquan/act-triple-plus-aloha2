"""hil_rollout.main() end to end with every hardware/server dependency faked."""

import sys
import types

import h5py
import numpy as np
from fakes import FakeBot

import hil_rollout
import policy_client


class FakeImageRecorder:
    def __init__(self):
        for camera in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"):
            setattr(self, camera + "_secs", 0)
            setattr(self, camera + "_nsecs", 0)


class FakeEnv:
    def __init__(self):
        self.puppet_bot_left = FakeBot()
        self.puppet_bot_right = FakeBot()
        self.image_recorder = FakeImageRecorder()

    def get_observation(self):
        qpos = np.concatenate([self.puppet_bot_left.position[:7], self.puppet_bot_right.position[:7]])
        images = {c: np.full((48, 64, 3), 100, np.uint8)
                  for c in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")}
        return {"qpos": qpos, "qvel": np.zeros(14), "effort": np.zeros(14), "images": images}


class FakeArmHaptics:
    instances = []

    def __init__(self, bot, side, feedback, brake, torque_gate=True):
        self.calls = []
        FakeArmHaptics.instances.append(self)

    def wait_for_data(self):
        return True

    def calibrate(self, samples):
        self.calls.append("calibrate")

    def start(self):
        self.calls.append("start")

    def update(self):
        self.calls.append("update")

    def restore(self):
        self.calls.append("restore")

    def rezero(self, samples):
        self.calls.append("rezero")


def make_legacy(env, masters, log):
    legacy = types.ModuleType("record_episodes_aloha2")
    bots = iter(masters)
    legacy.InterbotixManipulatorXS = lambda **kwargs: next(bots)
    legacy.make_real_env = lambda init_node, setup_robots: env
    legacy.opening_ceremony = lambda *args, **kwargs: log.append("opening_ceremony")
    legacy.build_haptics_config = lambda args, side: ({}, {})
    legacy.ArmHaptics = FakeArmHaptics
    legacy.TactileRecorder = None
    legacy.wait_for_start = lambda keyboard: True
    legacy.wait_for_decision = lambda keyboard, prompt: log.append(prompt) or "save"
    legacy.move_grippers = lambda bots, targets, move_time: log.append("open_grippers")
    legacy.sleep_all_robots = lambda: log.append("sleep_all_robots")
    legacy.shut_down_all_robots = lambda: log.append("shut_down_all_robots")
    for name in hil_rollout.HAPTICS_TUNING + ("force_baseline_samples",):
        setattr(legacy, "DEFAULT_" + name.upper(), 1.0)
    return legacy


class FakeClient:
    def __init__(self, host, port, connect_timeout_s=30.0, infer_timeout_s=None):
        self.uri = "ws://{}:{}".format(host, port)
        self.metadata = {"reset_pose": [0.0] * 6}
        self.observations = []
        self.timeouts = []

    def infer(self, obs, timeout=None):
        self.observations.append(obs)
        self.timeouts.append(timeout)
        state = obs["state"]
        return {"actions": np.tile(state, (50, 1))}

    def close(self):
        pass


class ScriptedHilInput:
    script = {}

    def __init__(self, pedal_device=None, pedal_keys=None):
        self.keyboard = types.SimpleNamespace(enabled=True)
        self.count = 0

    def poll(self):
        events = self.script.get(self.count, [])
        self.count += 1
        return list(events)

    def drain(self):
        pass

    def close(self):
        pass


def test_main_records_a_takeover_episode(tmp_path, monkeypatch):
    env, masters, log = FakeEnv(), (FakeBot(gripper=0.6), FakeBot(gripper=0.6)), []
    monkeypatch.setitem(sys.modules, "record_episodes_aloha2", make_legacy(env, masters, log))
    clients = []
    monkeypatch.setattr(policy_client, "PolicyClient", lambda *a, **k: clients.append(FakeClient(*a, **k)) or clients[-1])
    import hil_devices

    # takeover lasts ~100 ticks: align + release + settle (~15) + unlabelled correction_skip_s (25) + margin
    ScriptedHilInput.script = {0: ["pause"], 9: ["pause"], 10: ["correct"], 111: ["correct"], 116: ["pause"],
                               131: ["end"], 132: ["pause"]}
    monkeypatch.setattr(hil_devices, "HilInput", ScriptedHilInput)
    FakeArmHaptics.instances = []

    hil_rollout.main(["--task_name", "packaging_simple_2", "--policy", "openpi", "--host", "server",
                      "--prompt", "packaging the items", "--output_dir", str(tmp_path), "--max_timesteps", "200",
                      "--align_s", "0.1", "--release_ramp_s", "0.08", "--settle_s", "0.06"])

    assert clients[0].timeouts[0] == 900.0  # warm-up call before the first episode
    assert all(t is None for t in clients[0].timeouts[1:])  # then the client's own --infer_timeout
    assert log[0] == "opening_ceremony" and log[-1] == "shut_down_all_robots"
    assert "sleep_all_robots" not in log  # only one episode
    with h5py.File(str(tmp_path / "episode_0.hdf5"), "r") as f:
        phases = f["hil/phase"][()]
        assert len(phases) == 130 and phases[0] == 0 and 2 in phases and phases[-1] == 0
        assert f["hil/intervention"][()].sum() > 0
        assert sorted(f["observations/images"]) == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        assert f.attrs["hil_policy"] == "openpi" and f.attrs["hil_prompt"] == "packaging the items"
        assert "reset_pose" in f.attrs["hil_server_metadata"]
    for haptics in FakeArmHaptics.instances:
        assert haptics.calls[0] == "calibrate" and "start" in haptics.calls and haptics.calls[-1] == "restore"
    assert all(m.torque for m in masters)  # leaders locked at the end


class FakeLiveStreams:
    def __init__(self, sources, arm_names, calibration, save_force):
        self.sources = sources
        self.count = 0

    def wait_ready(self, timeout, max_age):
        pass

    def snapshot(self, max_age):
        import time

        self.count += 1
        streams = {}
        for i, name in enumerate(sorted(set(self.sources.values()))):
            streams[name] = {"image": np.full((32, 48, 3), 10 * i, np.uint8), "received": 0.0, "stamp": 0.0,
                             "sequence": self.count}
        for side in ("left", "right"):
            streams["joint_" + side] = {"state": np.full(7, 0.5), "received": 0.0, "stamp": 0.0, "sequence": self.count}
        return time.monotonic(), streams

    def close(self):
        pass


def test_main_n0_joint_records_vtla_episode(tmp_path, monkeypatch):
    import shutil

    import pytest

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg needed")
    import record_episodes_aloha2_vtla as vtla

    env, masters, log = FakeEnv(), (FakeBot(gripper=0.6), FakeBot(gripper=0.6)), []
    monkeypatch.setitem(sys.modules, "record_episodes_aloha2", make_legacy(env, masters, log))
    monkeypatch.setattr(vtla, "LiveStreams", FakeLiveStreams)
    clients = []

    class N0Client(FakeClient):
        def __init__(self, *args, **kwargs):
            FakeClient.__init__(self, *args, **kwargs)
            clients.append(self)

        def infer(self, obs, timeout=None):
            self.observations.append(obs)
            return {"actions": np.tile(obs["state"][:14], (50, 1))}

    monkeypatch.setattr(policy_client, "PolicyClient", N0Client)
    import hil_devices

    ScriptedHilInput.script = {0: ["pause"], 6: ["pause"], 7: ["correct"], 61: ["correct"], 81: ["end"],
                               82: ["pause"]}
    monkeypatch.setattr(hil_devices, "HilInput", ScriptedHilInput)
    hil_rollout.main(["--task_name", "packaging_simple_2", "--policy", "n0_joint", "--host", "server",
                      "--output_dir", str(tmp_path), "--max_timesteps", "200", "--no_force_feedback",
                      "--align_s", "0.1", "--release_ramp_s", "0.04", "--settle_s", "0.06"])
    obs = clients[0].observations[0]
    assert sorted(obs) == ["images", "prompt", "state", "tactile"] and obs["prompt"] == "Perform the task."
    assert obs["tactile"]["left_wrist_left_tactile"].shape == (2, 32, 48, 3)
    np.testing.assert_allclose(obs["state"][:14], 0.5)
    episode = tmp_path / "episode_0"
    import csv
    import json

    meta = json.loads((episode / "metadata.json").read_text())
    rows = list(csv.DictReader((episode / "hil" / "data.csv").open()))
    assert meta["aloha_action_space"] == "joint_position" and meta["hil"]["policy"] == "n0_joint"
    assert len(rows) == meta["total_frames"] and any(r["phase"] == "2" for r in rows)
