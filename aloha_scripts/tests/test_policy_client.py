import importlib.util
import os
import threading
import types

import numpy as np
import pytest

import policy_client
from policy_client import Frame, OpenpiAlohaAdapter, PolicyClient, packb, resize_with_pad, unpackb

OPENPI_CLIENT = "/mnt/server_ssd2/Programs_server/openpi/packages/openpi-client/src/openpi_client"


def _load_openpi(module):
    path = os.path.join(OPENPI_CLIENT, module + ".py")
    if not os.path.exists(path):
        pytest.skip("openpi_client sources not found")
    spec = importlib.util.spec_from_file_location("openpi_" + module, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_msgpack_roundtrip_and_openpi_compatibility():
    openpi = _load_openpi("msgpack_numpy")
    obs = {"state": np.arange(14, dtype=np.float64), "images": {"cam_high": np.zeros((3, 224, 224), np.uint8)},
           "prompt": "packaging the items", "scalar": np.float32(1.5)}
    for decoded in (unpackb(packb(obs)), openpi.unpackb(packb(obs)), unpackb(openpi.packb(obs))):
        np.testing.assert_array_equal(decoded["state"], obs["state"])
        assert decoded["images"]["cam_high"].shape == (3, 224, 224)
        assert decoded["images"]["cam_high"].dtype == np.uint8
        assert decoded["prompt"] == "packaging the items"
        assert decoded["scalar"] == np.float32(1.5)


def test_resize_with_pad_matches_openpi():
    openpi = _load_openpi("image_tools")
    image = np.random.RandomState(0).randint(0, 256, (480, 640, 3), dtype=np.uint8)
    np.testing.assert_array_equal(resize_with_pad(image, 224, 224), openpi.resize_with_pad(image, 224, 224))
    same = np.zeros((224, 224, 3), np.uint8)
    assert resize_with_pad(same, 224, 224) is same


def test_openpi_adapter_matches_aloha_real_client():
    images = {name: np.random.RandomState(i).randint(0, 256, (480, 640, 3), dtype=np.uint8)
              for i, name in enumerate(("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"))}
    frame = Frame(0.0, np.linspace(-1, 1, 14), np.zeros(14), np.zeros(14), images)
    adapter = OpenpiAlohaAdapter("packaging the items")
    obs = adapter.prepare(frame)
    assert sorted(obs["images"]) == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    for name, image in obs["images"].items():
        assert image.shape == (3, 224, 224) and image.dtype == np.uint8
        np.testing.assert_array_equal(image, np.transpose(resize_with_pad(images[name], 224, 224), (2, 0, 1)))
    assert obs["state"].dtype == np.float64 and obs["prompt"] == "packaging the items"
    chunk = np.random.RandomState(1).randn(50, 14)
    np.testing.assert_array_equal(adapter.decode(chunk, frame), chunk)
    with pytest.raises(RuntimeError):
        adapter.decode(np.zeros((50, 7)), frame)


def _serve(handler):
    from websockets.sync.server import serve

    server = serve(handler, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.socket.getsockname()[1]


def test_policy_client_against_openpi_style_server():
    def handler(ws):
        ws.send(packb({"reset_pose": [0.0] * 6}))
        for message in ws:
            obs = unpackb(message)
            actions = np.tile(obs["state"], (50, 1)) + np.arange(50)[:, None]
            ws.send(packb({"actions": actions, "server_timing": {"infer_ms": 1.0}}))

    server, port = _serve(handler)
    try:
        client = PolicyClient("127.0.0.1", port, connect_timeout_s=5.0, infer_timeout_s=5.0)
        assert client.metadata == {"reset_pose": [0.0] * 6}
        result = client.infer({"state": np.ones(14)})
        assert result["actions"].shape == (50, 14)
        np.testing.assert_allclose(result["actions"][3], 4.0)
        client.close()
    finally:
        server.shutdown()


def test_policy_client_timeout_and_server_error():
    def handler(ws):
        ws.send(packb({}))
        for message in ws:
            if unpackb(message).get("fail"):
                ws.send("Traceback: boom")
            # otherwise never reply

    server, port = _serve(handler)
    try:
        client = PolicyClient("127.0.0.1", port, connect_timeout_s=5.0, infer_timeout_s=0.3)
        with pytest.raises(RuntimeError, match="boom"):
            client.infer({"fail": True})
        with pytest.raises(TimeoutError):
            client.infer({"fail": False})
        client.close()
    finally:
        server.shutdown()


class _FakeVtlaClient:
    EEF_LAYOUT = {"left_xyz": slice(0, 3), "left_rot6d": slice(3, 9), "left_gripper": 9,
                  "right_xyz": slice(10, 13), "right_rot6d": slice(13, 19), "right_gripper": 19}
    JOINT_LAYOUT = {"left_joints": slice(0, 6), "left_gripper": 6, "right_joints": slice(7, 13), "right_gripper": 13}

    def __init__(self):
        self.calls = 0

    def rot6d_to_matrix(self, d6):
        return np.eye(3), True


def _n0_adapter(action_space):
    adapter = policy_client.N0VtlaAdapter.__new__(policy_client.N0VtlaAdapter)
    adapter._client = _FakeVtlaClient()
    adapter.action_space = action_space
    adapter.puppet_bots = (object(), object())
    return adapter


def test_n0_joint_decode_clips_grippers():
    adapter = _n0_adapter("joint")
    chunk = np.zeros((10, 14))
    chunk[:, 6], chunk[:, 13] = 1.4, -0.2
    chunk[:, 0] = 0.3
    out = adapter.decode(chunk, Frame(0.0, np.zeros(14), None, None, {}))
    assert out[0, 0] == 0.3 and out[0, 6] == 1.0 and out[0, 13] == 0.0


def test_n0_eef_decode_holds_previous_target_on_ik_failure(monkeypatch):
    def fake_ik(arm, xyz, rotation, guess):
        return None if xyz[0] < 0 else list(np.full(6, xyz[0]))  # negative x: unreachable

    monkeypatch.setattr(policy_client, "solve_ik", fake_ik)
    adapter = _n0_adapter("eef")
    adapter.puppet_bots = (types.SimpleNamespace(arm=None), types.SimpleNamespace(arm=None))
    chunk = np.zeros((3, 32))
    chunk[:, 9] = chunk[:, 19] = 0.5
    chunk[0, 0], chunk[1, 0], chunk[2, 0] = 0.1, -1.0, 0.2  # step 1 left arm unreachable
    chunk[:, 10] = 0.4
    qpos = np.arange(14, dtype=np.float64)
    out = adapter.decode(chunk, Frame(0.0, qpos, None, None, {}))
    np.testing.assert_allclose(out[0, 0:6], 0.1)
    np.testing.assert_allclose(out[1, 0:6], 0.1)  # held
    np.testing.assert_allclose(out[2, 0:6], 0.2)
    np.testing.assert_allclose(out[:, 7:13], 0.4)
    np.testing.assert_allclose(out[:, [6, 13]], 0.5)


def test_run_vtla_policy_still_builds_both_wire_layouts():
    import run_vtla_policy
    from record_episodes_aloha2_vtla import TACTILE_SENSORS, video_sources

    sources = video_sources("cam_high")
    frames = {name: np.full((4, 6, 3), i, np.uint8) for i, name in enumerate(sorted(set(sources.values())))}
    baseline = {name: np.zeros((4, 6, 3), np.uint8) for name in TACTILE_SENSORS}
    state = np.arange(32, dtype=np.float64)
    joint = run_vtla_policy.observation_from_frames("joint", frames, TACTILE_SENSORS, sources, "p", state, baseline)
    assert sorted(joint) == ["images", "prompt", "state", "tactile"]
    assert sorted(joint["images"]) == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    assert joint["tactile"]["left_wrist_left_tactile"].shape == (2, 4, 6, 3)
    assert joint["state"].dtype == np.float32
    eef = run_vtla_policy.observation_from_frames("eef", frames, TACTILE_SENSORS, sources, "p", state, baseline)
    assert "observation.image.third_view" in eef and eef["observation.state"].shape == (32,)

    class Streams:
        tactile_names = TACTILE_SENSORS

        def snapshot(self, max_age):
            return frames

    via_streams = run_vtla_policy.build_observation("joint", Streams(), sources, "p", 0.5, state, baseline)
    np.testing.assert_array_equal(via_streams["images"]["cam_high"], joint["images"]["cam_high"])


def test_policy_client_reconnects_after_timeout_so_late_replies_are_dropped():
    import time

    def handler(ws):
        ws.send(packb({}))
        for message in ws:
            obs = unpackb(message)
            if obs["id"] == 1:
                time.sleep(0.6)  # e.g. the model compiling on its first call
            try:
                ws.send(packb({"id": obs["id"]}))
            except Exception:
                return  # client already hung up

    server, port = _serve(handler)
    try:
        client = PolicyClient("127.0.0.1", port, connect_timeout_s=5.0, infer_timeout_s=0.3)
        with pytest.raises(TimeoutError):
            client.infer({"id": 1})
        assert client.infer({"id": 2})["id"] == 2  # not the late answer to request 1
        assert client.infer({"id": 3}, timeout=2.0)["id"] == 3
        client.close()
    finally:
        server.shutdown()


MR_DESCRIPTIONS = ("/home/ubuntu20/interbotix_ws_noetic/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/"
                   "interbotix_xs_modules/src/interbotix_xs_modules/mr_descriptions.py")


def test_solve_ik_reaches_poses_far_from_the_current_command():
    """The whole chunk is solved at once, so steps far from the arm's current command must not be
    rejected (InterbotixArmXSInterface.set_ee_pose_matrix's velocity check would)."""
    mr = pytest.importorskip("modern_robotics")
    if not os.path.exists(MR_DESCRIPTIONS):
        pytest.skip("Interbotix mr_descriptions not found")
    spec = importlib.util.spec_from_file_location("mr_descriptions", MR_DESCRIPTIONS)
    descriptions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(descriptions)
    arm = types.SimpleNamespace(robot_des=descriptions.vx300s, rev=2 * np.pi, group_info=types.SimpleNamespace(
        joint_lower_limits=[-3.1, -1.85, -1.76, -3.1, -1.87, -3.1],
        joint_upper_limits=[3.1, 1.26, 1.6, 3.1, 2.23, 3.1]))
    seed = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
    target_q = np.array(seed) + np.array([0.4, 0.3, -0.3, 0.2, 0.3, -0.4])  # 0.4 rad away
    pose = mr.FKinSpace(arm.robot_des.M, arm.robot_des.Slist, target_q)
    theta = policy_client.solve_ik(arm, pose[:3, 3], pose[:3, :3], seed)
    assert theta is not None
    np.testing.assert_allclose(mr.FKinSpace(arm.robot_des.M, arm.robot_des.Slist, theta), pose, atol=1e-3)
    far = pose.copy()
    far[:3, 3] = [2.0, 0.0, 0.5]  # out of reach
    assert policy_client.solve_ik(arm, far[:3, 3], far[:3, :3], seed) is None
