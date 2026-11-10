import sys
import types

import numpy as np
import pytest
from fakes import FakeBot

import run_pi_policy
from chunk_executor import ChunkExecutor
from hil_devices import FollowerCommander

DT = 0.02


class ScriptedKeyboard:
    def __init__(self, script=None, enabled=True):
        self.script = dict(script or {})
        self.count = 0
        self.enabled = enabled

    def poll(self):
        events = self.script.get(self.count, [])
        self.count += 1
        return list(events)

    def drain(self):
        pass

    def close(self):
        pass


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def make(policy, script=None):
    puppets = (FakeBot(), FakeBot())
    follower = FollowerCommander(puppets, max_joint_step=None)
    executor = ChunkExecutor(policy, 50, DT, action_horizon=50)
    return puppets, follower, executor, ScriptedKeyboard(script)


def test_round_runs_the_chunks_on_the_followers():
    calls = []

    def policy(frame):
        calls.append(frame)
        chunk = np.tile(np.concatenate([[0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.5]] * 2), (50, 1))
        chunk[:, 0] = 0.001 * np.arange(50)
        return chunk

    puppets, follower, executor, keyboard = make(policy)
    clock = Clock()
    outcome = run_pi_policy.run_episode(executor, follower, keyboard, lambda: "frame", 120, DT, clock, clock.sleep)
    assert outcome == "done" and len(calls) == 3  # 120 ticks / 50-step chunks
    assert puppets[0].arm.published[49][0] == pytest.approx(0.049)
    assert puppets[0].arm.published[50][0] == pytest.approx(0.0)  # next chunk from its step 0


def test_left_aborts_and_esc_quits():
    policy = lambda frame: np.tile(FakeBot().position[:7] * 2, (50, 1))[:, :14]
    for key, expected in (("left", "reset"), ("reset", "reset"), ("esc", "quit")):
        puppets, follower, executor, keyboard = make(policy, {5: [key]})
        clock = Clock()
        assert run_pi_policy.run_episode(executor, follower, keyboard, lambda: "f", 100, DT, clock, clock.sleep) == expected
        assert len(puppets[0].arm.published) == 5


def test_inference_failure_or_stale_camera_stops_the_round():
    def broken(frame):
        raise TimeoutError("no reply")

    puppets, follower, executor, keyboard = make(broken)
    clock = Clock()
    assert run_pi_policy.run_episode(executor, follower, keyboard, lambda: "f", 100, DT, clock, clock.sleep) == "reset"
    assert puppets[0].arm.published == []

    def stale():
        raise run_pi_policy.StaleImage("cam_high image is 1.0s old")

    puppets, follower, executor, keyboard = make(lambda f: np.zeros((50, 14)))
    assert run_pi_policy.run_episode(executor, follower, keyboard, stale, 100, DT, clock, clock.sleep) == "reset"


def test_capture_checks_image_age():
    recorder = types.SimpleNamespace(cam_high_secs=10, cam_high_nsecs=0)
    env = types.SimpleNamespace(image_recorder=recorder, get_observation=lambda: {
        "qpos": np.zeros(14), "qvel": np.zeros(14), "effort": np.zeros(14), "images": {}})
    assert run_pi_policy.capture(env, ["cam_high"], 0.5, lambda: 10.2).qpos.shape == (14,)
    with pytest.raises(run_pi_policy.StaleImage):
        run_pi_policy.capture(env, ["cam_high"], 0.5, lambda: 11.0)
    with pytest.raises(run_pi_policy.StaleImage):
        run_pi_policy.capture(env, ["cam_low"], 0.5, lambda: 10.0)


def test_homing_restores_the_per_tick_trajectory_time():
    puppets = (FakeBot(), FakeBot())
    log = []
    for bot in puppets:
        bot.arm.set_trajectory_time = lambda moving_time=None, accel_time=None, bot=bot: log.append((bot, moving_time))
    run_pi_policy.home_followers(puppets, [0.0] * 6, lambda bots, poses, move_time: log.append("arms"),
                                 lambda bots, targets, move_time: log.append("grippers"), (0.07, 0.07), DT)
    assert log[:2] == ["arms", "grippers"] and [entry[1] for entry in log[2:]] == [DT, DT]


def _run_main(monkeypatch, argv, metadata=None):
    log = []

    class FakeImageRecorder:
        cam_high_secs = cam_left_wrist_secs = cam_right_wrist_secs = 0
        cam_high_nsecs = cam_left_wrist_nsecs = cam_right_wrist_nsecs = 0

    class FakeEnv:
        def __init__(self):
            self.puppet_bot_left, self.puppet_bot_right = FakeBot(), FakeBot()
            for bot in (self.puppet_bot_left, self.puppet_bot_right):
                bot.arm.set_trajectory_time = lambda moving_time=None, accel_time=None: None
            self.image_recorder = FakeImageRecorder()

        def get_observation(self):
            images = {c: np.full((48, 64, 3), 7, np.uint8) for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
            qpos = np.concatenate([self.puppet_bot_left.position[:7], self.puppet_bot_right.position[:7]])
            return {"qpos": qpos, "qvel": np.zeros(14), "effort": np.zeros(14), "images": images}

    class FakeClient:
        def __init__(self, host, port, infer_timeout_s=None):
            self.uri, self.metadata, self.timeouts, self.observations = "ws://x", metadata or {}, [], []

        def infer(self, obs, timeout=None):
            self.timeouts.append(timeout)
            self.observations.append(obs)
            return {"actions": np.tile(obs["state"], (50, 1))}

        def close(self):
            log.append("client_closed")

    clients = []
    monkeypatch.setattr(run_pi_policy, "PolicyClient", lambda *a, **k: clients.append(FakeClient(*a, **k)) or clients[-1])
    env = FakeEnv()
    monkeypatch.setitem(sys.modules, "real_env", types.SimpleNamespace(make_real_env=lambda init_node, setup_robots: env))
    monkeypatch.setitem(sys.modules, "robot_utils", types.SimpleNamespace(
        move_arms=lambda bots, poses, move_time: log.append("home"), move_grippers=lambda bots, t, move_time: None))
    monkeypatch.setitem(sys.modules, "sleep_plus", types.SimpleNamespace(
        shut_down_puppet_robots=lambda: log.append("shutdown")))
    keyboards = iter([ScriptedKeyboard({0: ["right"], 30: ["right"]})])
    monkeypatch.setitem(sys.modules, "one_side_teleop", types.SimpleNamespace(KeyboardControl=lambda: next(keyboards)))

    run_pi_policy.main(["--host", "server", "--num_episodes", "2", "--max_episode_steps", "20"] + argv)
    return clients[0], log


def test_main_with_faked_hardware(monkeypatch):
    client, log = _run_main(monkeypatch, ["--prompt", "packaging the items"])
    assert client.timeouts[0] == 900.0 and all(t is None for t in client.timeouts[1:])
    assert client.observations[1]["prompt"] == "packaging the items"
    assert client.observations[1]["images"]["cam_high"].shape == (3, 224, 224)
    assert log.count("home") == 2  # initial homing + between the two rounds
    assert log[-2:] == ["client_closed", "shutdown"]


def test_main_with_a_lerobot_server(monkeypatch):
    metadata = {"server": "lerobot", "policy_type": "molmoact2", "cameras": ["cam_high", "cam_right_wrist"],
                "default_task": "packaging the items", "n_action_steps": 30}
    client, log = _run_main(monkeypatch, ["--policy", "lerobot"], metadata)
    obs = client.observations[1]
    assert sorted(obs["images"]) == ["cam_high", "cam_right_wrist"]  # the checkpoint's cameras
    assert all(image.ndim == 1 and image.dtype == np.uint8 for image in obs["images"].values())  # JPEG
    assert {"state", "velocity", "effort"} <= set(obs) and "prompt" not in obs  # server falls back to --task
    assert log.count("home") == 2 and log[-2:] == ["client_closed", "shutdown"]
