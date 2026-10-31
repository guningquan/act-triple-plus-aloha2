import csv
import json
import shutil
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import pytest

import hil_split_episodes
from hil_recorder import Hdf5HilEpisode, VtlaHilEpisode
from policy_client import Frame

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def _image(seed, shape=(48, 64, 3)):
    return np.random.RandomState(seed).randint(0, 256, shape, dtype=np.uint8)


def record_hdf5(path, phases, interventions):
    episode = Hdf5HilEpisode(CAMERAS)
    for t, (phase, label) in enumerate(zip(phases, interventions)):
        frame = Frame(10.0 + 0.02 * t, np.full(14, t, float), np.zeros(14), np.ones(14),
                      {name: _image(t * 10 + i) for i, name in enumerate(CAMERAS + ["cam_low"])})
        episode.add(frame, np.full(14, t + 0.5), phase, label, np.full(14, t + 0.25))
    count = episode.save(str(path), {"policy": "openpi", "settings": {"correction_skip_s": 0.5},
                                     "server_metadata": {"reset_pose": np.zeros(3)}})
    episode.close()
    return count


def test_hdf5_matches_the_recorder_layout_and_adds_hil_labels(tmp_path):
    phases = [0] * 5 + [1] * 3 + [2] * 6 + [1] * 2
    labels = [0] * 8 + [0, 0, 1, 1, 1, 1] + [0, 0]
    assert record_hdf5(tmp_path / "episode_0", phases, labels) == 16
    with h5py.File(str(tmp_path / "episode_0.hdf5"), "r") as f:
        assert f.attrs["sim"] == False and f.attrs["compress"] == True  # noqa: E712
        for name in ("observations/qpos", "observations/qvel", "observations/effort", "action", "base_action"):
            assert f[name].dtype == np.float32
        assert f["action"].shape == (16, 14) and f["base_action"].shape == (16, 2)
        assert sorted(f["observations/images"]) == sorted(CAMERAS)  # only the task's cameras
        assert f["compress_len"].shape == (3, 16)
        # images decode exactly like record_episodes_aloha2.py's (cv2 JPEG quality 90)
        row = f["observations/images/cam_left_wrist"][7]
        decoded = cv2.imdecode(row[:int(f["compress_len"][1, 7])], 1)
        expected = cv2.imdecode(cv2.imencode(".jpg", _image(71), [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1], 1)
        np.testing.assert_array_equal(decoded, expected)
        np.testing.assert_array_equal(f["hil/phase"][()], phases)
        np.testing.assert_array_equal(f["hil/intervention"][()], labels)
        np.testing.assert_allclose(f["hil/time"][()], 0.02 * np.arange(16), atol=1e-9)
        np.testing.assert_allclose(f["hil/leader_action"][3], 3.25)
        assert json.loads(f.attrs["hil_settings"]) == {"correction_skip_s": 0.5}
        assert f.attrs["hil_policy"] == "openpi"


def test_runs_and_masks():
    phase = np.array([0, 0, 1, 2, 2, 2, 1, 0, 0, 0])
    label = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    assert hil_split_episodes.runs(hil_split_episodes.segment_mask(phase, label, "intervention"), 1) == [(4, 6)]
    assert hil_split_episodes.runs(hil_split_episodes.segment_mask(phase, label, "autonomous"), 1) == [(0, 2), (7, 10)]
    assert hil_split_episodes.runs(hil_split_episodes.segment_mask(phase, label, "active"), 3) == [(7, 10)]


def test_split_hdf5_into_correction_episodes(tmp_path):
    hil_dir, out_dir = tmp_path / "hil", tmp_path / "split"
    hil_dir.mkdir()
    record_hdf5(hil_dir / "episode_0", [0] * 4 + [2] * 6 + [0] * 4 + [2] * 5, [0] * 4 + [0] + [1] * 5 + [0] * 4 + [1] * 5)
    record_hdf5(hil_dir / "episode_1", [0] * 6, [0] * 6)  # no correction -> nothing
    hil_split_episodes.main([str(hil_dir), str(out_dir), "--min_frames", "3"])
    listing = json.loads((out_dir / "segments.json").read_text())["segments"]
    assert [(s["source"], s["start"], s["end"]) for s in listing] == [("episode_0.hdf5", 5, 10), ("episode_0.hdf5", 14, 19)]
    with h5py.File(str(out_dir / "episode_1.hdf5"), "r") as f, \
            h5py.File(str(hil_dir / "episode_0.hdf5"), "r") as src:
        np.testing.assert_array_equal(f["action"][()], src["action"][14:19])
        np.testing.assert_array_equal(f["observations/qpos"][()], src["observations/qpos"][14:19])
        np.testing.assert_array_equal(f["observations/images/cam_high"][()], src["observations/images/cam_high"][14:19])
        np.testing.assert_array_equal(f["compress_len"][()], src["compress_len"][:, 14:19])
        assert f["hil/intervention"][()].all() and f["hil/time"][0] == 0
        assert json.loads(f.attrs["hil_source"])["start"] == 14
        assert f.attrs["compress"] == True  # noqa: E712
    with pytest.raises(SystemExit):  # refuses to mix into a non-empty output
        hil_split_episodes.main([str(hil_dir), str(out_dir)])


class _Model:
    M = np.eye(4)
    Slist = np.eye(6)


def _vtla_sources():
    from record_episodes_aloha2_vtla import video_sources

    return video_sources("cam_high")


def _vtla_snapshot(t, sources):
    streams = {}
    for i, name in enumerate(sorted(set(sources.values()))):
        image = np.full((32, 48, 3), (t * 5 + i * 20) % 256, dtype=np.uint8)
        streams[name] = {"image": image, "received": 0.0, "stamp": 0.0, "sequence": t + 1}
    for side in ("left", "right"):
        streams["joint_" + side] = {"state": np.full(7, 0.01 * t), "received": 0.0, "stamp": 0.0, "sequence": t + 1}
    return streams


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_vtla_export_with_hil_csv_and_split(tmp_path):
    sources = _vtla_sources()
    episode = VtlaHilEpisode(sources, None, [{"closed": -1.5, "open": 0.08}] * 2, "joint_position", "test task")
    phases = [0] * 30 + [2] * 60 + [0] * 30  # 2.4 s at 50 Hz
    labels = [0] * 30 + [0] * 10 + [1] * 50 + [0] * 30
    for t, (phase, label) in enumerate(zip(phases, labels)):
        frame = Frame(5.0 + 0.02 * t, np.zeros(14), np.zeros(14), np.zeros(14), {}, _vtla_snapshot(t, sources))
        action = np.full(14, 0.1 * t)
        action[[6, 13]] = 1.5  # clipped to 1 like the VTLA recorder
        episode.add(frame, action, phase, label)
    hil_dir = tmp_path / "hil"
    frames = episode.save(hil_dir / "episode_0", {"policy": "n0_joint"})
    meta = json.loads((hil_dir / "episode_0" / "metadata.json").read_text())
    assert meta["total_frames"] == frames and meta["hil"] == {"policy": "n0_joint"}
    rows = list(csv.DictReader((hil_dir / "episode_0" / "hil" / "data.csv").open()))
    assert len(rows) == frames
    actions = list(csv.DictReader((hil_dir / "episode_0" / "actions.joint_position" / "data.csv").open()))
    assert float(actions[0]["left_gripper"]) == 1.0
    # same causal resampling as every other CSV: 30 Hz grid, labels follow their source samples
    flags = np.array([int(r["intervention"]) for r in rows])
    first = int(np.argmax(flags))
    assert abs(float(rows[first]["timestamp"]) - 40 * 0.02) < 1.0 / 30 + 1e-6

    out_dir = tmp_path / "split"
    hil_split_episodes.main([str(hil_dir), str(out_dir), "--min_frames", "10"])
    segment = out_dir / "episode_0"
    seg_meta = json.loads((segment / "metadata.json").read_text())
    seg_rows = list(csv.DictReader((segment / "hil" / "data.csv").open()))
    assert seg_meta["total_frames"] == len(seg_rows) == int(flags.sum())
    assert all(r["intervention"] == "1" for r in seg_rows) and float(seg_rows[0]["timestamp"]) == 0.0
    state_rows = list(csv.DictReader((segment / "observation.state.joint_position" / "data.csv").open()))
    assert len(state_rows) == len(seg_rows)
    capture = cv2.VideoCapture(str(segment / "observation.image.third_view" / "video.mp4"))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(seg_rows)
    capture.release()
