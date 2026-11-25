"""Right-only recording must never connect to or command missing left hardware."""
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

from fakes import FakeBot
from test_tactile_recording_flows import _ImageRecorder, _JointRecorder, _Keyboard


@pytest.fixture
def hardware(monkeypatch):
    import real_env
    import record_episodes_aloha2 as rec
    import sleep_plus

    bots, subscriptions, moves = {}, [], []

    def connect(**kwargs):
        name = kwargs['robot_name']
        assert name.endswith('_right'), 'attempted to connect to ignored left arm'
        bot = FakeBot()
        bot.init_node = kwargs['init_node']
        bot.dxl.robot_reboot_motors = lambda *args: None
        bot.arm.set_trajectory_time = lambda **kw: None
        bots[name] = bot
        return bot

    def recorder(side, **kwargs):
        assert side == 'right', 'attempted to subscribe to ignored left arm'
        subscriptions.append(side)
        result = _JointRecorder()
        result.qpos = [0.2] * 8
        result.qvel = [0.3] * 8
        result.effort = [0.4] * 8
        return result

    def move(arms, targets, **kwargs):
        assert arms and all(bot in bots.values() for bot in arms)
        assert len(arms) == len(targets)
        moves.append(tuple(arms))

    for module in (real_env, rec, sleep_plus):
        monkeypatch.setattr(module, 'InterbotixManipulatorXS', connect)
        monkeypatch.setattr(module, 'move_arms', move)
        monkeypatch.setattr(module, 'move_grippers', move)
    monkeypatch.setattr(real_env, 'Recorder', recorder)
    monkeypatch.setattr(real_env, 'ImageRecorder', lambda **kw: _ImageRecorder())
    monkeypatch.setattr(rec, 'KeyboardControl', lambda: _Keyboard())
    monkeypatch.setattr(rec, 'wait_for_start', lambda keyboard: True)
    monkeypatch.setattr(rec, 'print_dt_diagnosis', lambda history: 50.0)
    monkeypatch.setattr(rec, 'calibrate_sensors', lambda *a: pytest.fail('unexpected tactile calibration'))
    return SimpleNamespace(bots=bots, subscriptions=subscriptions, moves=moves)


def test_right_env_initialization_reset_and_actions(hardware):
    import real_env

    env = real_env.make_real_env(init_node=True, ignore_left=True)
    assert env.puppet_bot_left is None and env.recorder_left is None
    assert hardware.bots['puppet_right'].init_node
    assert hardware.subscriptions == ['right']
    env.reset()  # Also exercise reboot, arm and gripper reset paths.
    leader = FakeBot()
    action = real_env.get_action(None, leader)
    np.testing.assert_array_equal(action[:7], 0)
    np.testing.assert_array_equal(action[7:13], leader.position[:6])
    obs = env.step(action).observation
    for key in ('qpos', 'qvel', 'effort'):
        assert obs[key].shape == (14,)
        np.testing.assert_array_equal(obs[key][:7], 0)
        assert np.any(obs[key][7:] != 0)
    assert np.isnan(obs['timestamps']['joints']['left'])
    np.testing.assert_array_equal(env.puppet_bot_right.arm.published[-1], action[7:13])
    assert env.puppet_bot_right.gripper.core.pub_single.sent


@pytest.mark.parametrize('decision', ['save', 'redo', 'quit'])
def test_right_episode_end_to_end(tmp_path, monkeypatch, hardware, decision):
    import record_episodes_aloha2 as rec

    monkeypatch.setattr(rec, 'wait_for_decision', lambda *args: decision)
    result = rec.capture_one_episode(rec.DT, 3, ['cam_high'], str(tmp_path), 'episode_0', False,
                                    haptics_args={'ignore_left': True, 'no_force_feedback': True})
    assert result == decision
    assert set(hardware.bots) == {'master_right', 'puppet_right'}
    assert hardware.bots['master_right'].init_node
    assert not hardware.bots['puppet_right'].init_node
    assert len(hardware.bots['puppet_right'].arm.published) == 3
    path = tmp_path / 'episode_0.hdf5'
    assert path.exists() == (decision == 'save')
    if decision == 'save':
        with h5py.File(path) as f:
            assert f.attrs['active_arms'] == 'right'
            assert 'not measurements' in f.attrs['inactive_arm_fill']
            for key in ('action', 'observations/qpos', 'observations/qvel', 'observations/effort'):
                assert f[key].shape == (3, 14)
                np.testing.assert_array_equal(f[key][:, :7], 0)
            assert np.isnan(f['observations/timestamps/joints/left'][:]).all()


@pytest.mark.parametrize('shutdown', [False, True])
def test_right_rest_never_constructs_left(monkeypatch, hardware, shutdown):
    import sleep_plus

    calls = []
    monkeypatch.setattr(sleep_plus, 'move_robot', lambda *args: calls.append(args))
    monkeypatch.setattr(sleep_plus, 'move_shut_robot', lambda *args: calls.append(args))
    operation = sleep_plus.shut_down_all_robots if shutdown else sleep_plus.sleep_all_robots
    operation(ignore_left=True)
    assert set(hardware.bots) == {'master_right', 'puppet_right'}
    assert len(calls) == 2
    scale = sleep_plus.SHUTDOWN_SPEED_SCALE if shutdown else 1.0
    assert calls[0][-1] == (scale, scale)
    assert calls[1][-1] == (scale, 1.5 * scale)


def test_main_preserves_ignore_left_on_redo(tmp_path, monkeypatch):
    import record_episodes_aloha2 as rec

    results = iter(['redo', 'save'])
    rests = []
    monkeypatch.setitem(rec.TASK_CONFIGS, 'right_test', {
        'dataset_dir': str(tmp_path), 'episode_len': 3, 'camera_names': ['cam_high']})
    def capture(*args, **kwargs):
        assert kwargs['haptics_args']['ignore_left']
        return next(results)
    monkeypatch.setattr(rec, 'capture_one_episode', capture)
    monkeypatch.setattr(rec, 'sleep_all_robots', lambda **kwargs: rests.append(kwargs))
    assert rec.main({'task_name': 'right_test', 'episode_idx': 0, 'ignore_left': True}) == 'save'
    assert rests == [{'ignore_left': True}]


def test_launch_only_starts_right_arms_and_recording_cameras():
    path = Path(__file__).parents[1] / 'launch' / 'right_arm_recording.launch'
    root = ET.parse(path).getroot()
    names = [arg.get('value') for inc in root.findall('include') for arg in inc.findall('arg')
             if arg.get('name') == 'robot_name']
    assert names == ['$(arg robot_name_master_right)', '$(arg robot_name_puppet_right)']
    nodes = [node.get('name') for node in root.findall('node')]
    assert not any('tactile' in name or 'xense' in name for name in nodes)
    assert {n for n in nodes if n.startswith('usb_cam')} == {
        'usb_cam_high', 'usb_cam_low', 'usb_cam_left_wrist', 'usb_cam_right_wrist'}
