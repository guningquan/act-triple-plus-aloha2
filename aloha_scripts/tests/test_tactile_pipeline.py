"""The real-time tactile path without ROS: tactile_stream (shared, decode-once streams), the
time-based force-feedback filter, TactileRecorder's aligned snapshots and the HDF5 / VTLA output."""

import csv
import json
import shutil
import threading
import time
import zlib

import cv2
import h5py
import numpy as np
import pytest

import tactile_stream
from tactile_stream import TactileStale, TactileStream, decode_field

SENSORS = ["left_gripper_inner", "left_gripper_outer"]


def _image(h=70, w=40):
    """A smooth RGB image (JPEG round trips stay close; random noise would not)."""
    ys, xs = np.mgrid[0:h, 0:w]
    return np.stack([ys * 3 % 256, xs * 6 % 256, (ys + xs) * 2 % 256], axis=-1).astype(np.uint8)


def _force(value):
    """A force field whose normal channel sums to `value` (the feedback's abs_sum metric)."""
    field = np.zeros((35, 20, 3))
    field[..., 2] = value / 700.0
    return field


def _payload(read_end, seq, codec="raw", rgb=None, force=None, depth=None):
    """A /raw payload as xense_tactile_publisher builds it."""
    rgb = _image() if rgb is None else rgb
    frames, encoding = {}, {}
    if codec == "jpeg":
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        frames["rectify"] = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].reshape(-1)
        encoding["rectify"] = {"codec": "jpeg", "quality": 95, "shape": list(bgr.shape)}
    else:
        frames["rectify"] = rgb
    if force is not None:
        frames["force"] = force
    if depth is not None:
        depth = depth.astype(np.float32)
        if codec == "jpeg":
            frames["depth"] = np.frombuffer(zlib.compress(depth.tobytes(), 1), np.uint8)
            encoding["depth"] = {"codec": "zlib", "dtype": "float32", "shape": list(depth.shape)}
        else:
            frames["depth"] = depth
    meta = {"local_read_start": read_end - 0.01, "local_read_end": read_end, "timestamp": read_end,
            "timestamp_source": "xense", "seq": seq, "image_codec": codec, "encoding": encoding}
    return {"frames": frames, "meta": meta}


@pytest.fixture
def streams():
    """Make offline TactileStreams the process-wide ones (what acquire() returns); drop them after."""
    made = []

    def make(sensor):
        stream = TactileStream(tactile_stream.topic_for(sensor), subscribe=False)
        TactileStream.register(stream)
        made.append(stream.topic)
        return stream

    yield make
    with TactileStream._registry_lock:
        for topic in made:
            TactileStream._streams.pop(topic, None)


def test_decode_field_handles_raw_jpeg_and_zlib():
    rgb = _image()
    depth = np.random.RandomState(1).rand(70, 40).astype(np.float32)
    raw = _payload(1.0, 1, "raw", rgb, depth=depth)
    assert decode_field(raw["frames"], raw["meta"], "rectify", "rgb") is raw["frames"]["rectify"]
    assert np.array_equal(decode_field(raw["frames"], raw["meta"], "rectify", "bgr"), rgb[..., ::-1])
    assert np.array_equal(decode_field(raw["frames"], raw["meta"], "depth"), depth)
    assert decode_field(raw["frames"], raw["meta"], "force") is None

    jpeg = _payload(1.0, 1, "jpeg", rgb, depth=depth)
    bgr = decode_field(jpeg["frames"], jpeg["meta"], "rectify", "bgr")
    back = decode_field(jpeg["frames"], jpeg["meta"], "rectify", "rgb")
    assert bgr.shape == rgb.shape and np.array_equal(back, bgr[..., ::-1])
    assert np.abs(back.astype(int) - rgb).mean() < 3            # true-colour JPEG, not R/B swapped
    assert np.array_equal(decode_field(jpeg["frames"], jpeg["meta"], "depth"), depth)  # zlib: lossless


def test_stream_keeps_the_newest_frame_its_age_and_prefetches(streams):
    stream = streams("left_gripper_outer")
    assert stream.latest() is None and stream.wait_for_frame(0.01) is None
    stream._prefetch.add(("rectify", "bgr"))
    now = time.time()
    frame = stream.feed(_payload(now - 0.05, 7))
    assert stream.count == 1 and stream.latest() is frame and frame.seq == 7
    assert frame.age(now) == pytest.approx(0.05)
    assert ("rectify", "bgr") in frame._cache                   # decoded in the callback path
    assert frame.field("rectify", "bgr") is frame.field("rectify", "bgr")   # once, then cached
    assert stream.wait_newer_than(now, 0.02) is None             # nothing delivered after `now` yet
    threading.Timer(0.02, lambda: stream.feed(_payload(time.time(), 8))).start()
    assert stream.wait_newer_than(now, 1.0).seq == 8


def test_acquire_shares_one_stream_per_topic(streams):
    stream = streams("left_gripper_inner")
    a = TactileStream.acquire("left_gripper_inner")
    b = TactileStream.acquire("/tactile/left_gripper_inner/raw")
    assert a is stream and b is stream
    a.release()
    assert TactileStream._streams.get(stream.topic) is stream
    b.release()
    assert stream.topic not in TactileStream._streams


def _feedback(streams, **overrides):
    import one_side_teleop

    topic = "/tactile/left_gripper_outer/raw"
    stream = TactileStream._streams.get(topic) or streams(topic)
    config = dict(topic=topic, deadband=0.05, full_scale=0.6, curve=2.0, max_pwm=250, timeout_s=0.3,
                  filter_alpha=0.3, metric="abs_sum", sign=1, ramp_s=0.0, mode="wall")
    config.update(overrides)
    return one_side_teleop.GripperForceFeedback(**config), stream


def test_force_filter_runs_on_time_whatever_the_tick_rate(streams, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    results = {}
    for rate in (50, 200):
        clock[0] = 1000.0
        feedback, stream = _feedback(streams)
        stream.feed(_payload(clock[0], 0, force=_force(0.0)))
        feedback.command(0.5)                                     # filter starts at 0
        for i in range(int(0.2 * rate)):
            clock[0] += 1.0 / rate
            stream.feed(_payload(clock[0], i + 1, force=_force(1.0)))
            feedback.command(0.5)
        results[rate] = feedback._force
    # the old per-message filter at its steady 60 messages/s over the same 0.2 s: 12 steps
    expected = 1.0 - (1.0 - 0.3) ** 12
    assert results[50] == pytest.approx(expected) and results[200] == pytest.approx(expected)


def test_force_feedback_is_stale_by_the_frames_own_age(streams, monkeypatch):
    clock = [2000.0]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    feedback, stream = _feedback(streams)
    feedback.activate()
    stream.feed(_payload(clock[0] - 0.5, 1, force=_force(1.0)))   # just received, but 0.5 s old data
    pwm, force, age, depth = feedback.command(0.4)
    assert pwm == 0 and age == pytest.approx(0.5) and not feedback.contact
    stream.feed(_payload(clock[0], 2, force=_force(1.0)))
    feedback.command(0.4)                                         # contact: anchor at 0.4
    pwm, force, age, depth = feedback.command(0.3)                # squeezed 0.1 rad past it
    assert pwm > 0 and depth == pytest.approx(0.1) and age == pytest.approx(0.0)


def test_force_baseline_averages_distinct_frames(streams):
    feedback, stream = _feedback(streams)

    def feed():
        for i, value in enumerate((0.1, 0.2, 0.3)):
            time.sleep(0.02)
            stream.feed(_payload(time.time(), i, force=_force(value)))

    feeder = threading.Thread(target=feed)
    feeder.start()
    baseline = feedback.capture_baseline(3, timeout_s=2.0)
    feeder.join()
    assert baseline == pytest.approx(0.2)


def test_tactile_recorder_snapshots_pack_and_stale(streams):
    import record_episodes_aloha2 as recorder_module

    feeds = {name: streams(name) for name in SENSORS}
    recorder = recorder_module.TactileRecorder(SENSORS, ["rectify", "force"], max_age_s=0.2, jpeg_quality=90)
    with pytest.raises(TactileStale):
        recorder.snapshot()                                       # no frames yet
    rgb = _image()
    now = time.time()
    feeds[SENSORS[0]].feed(_payload(now, 1, "raw", rgb, force=_force(0.5)))
    feeds[SENSORS[1]].feed(_payload(now, 1, "jpeg", rgb, force=_force(0.5)))
    first, second = recorder.snapshot(), recorder.snapshot()
    assert first[SENSORS[0]]["new_frame"] and not second[SENSORS[0]]["new_frame"]
    assert first[SENSORS[1]]["encoded"] == {"rectify": 95} and first[SENSORS[0]]["encoded"] == {}
    assert np.array_equal(first[SENSORS[0]]["fields"]["rectify"], rgb[..., ::-1])   # kept as BGR

    packed, lengths = recorder.pack([first, second])
    assert lengths["entries"] == ["{}/rectify".format(SENSORS[0]), "{}/rectify".format(SENSORS[1])]
    published = feeds[SENSORS[1]].latest().frames["rectify"]
    stored = packed[SENSORS[1]]["rectify"]["data"][0][:lengths["lens"][1][0]]
    assert np.array_equal(stored, published)                     # the publisher's JPEG, as it was
    assert packed[SENSORS[1]]["rectify"]["jpeg_quality"] == 95   # ...labelled with its own quality
    assert packed[SENSORS[0]]["rectify"]["jpeg_quality"] == 90   # encoded here
    decoded = cv2.imdecode(packed[SENSORS[0]]["rectify"]["data"][0][:lengths["lens"][0][0]], cv2.IMREAD_COLOR)
    assert np.abs(decoded.astype(int) - rgb[..., ::-1]).mean() < 3
    assert packed[SENSORS[0]]["force"]["data"].shape == (2, 35, 20, 3)
    assert recorder.timing([first, second])[SENSORS[0]]["new_frame"].tolist() == [True, False]

    feeds[SENSORS[0]].feed(_payload(time.time() - 0.3, 2, "raw", rgb, force=_force(0.5)))
    with pytest.raises(TactileStale):
        recorder.snapshot()                                       # newest frame is 300 ms old
    recorder.close()


def _write_episode(path, streams, frame_age=0.003):
    import record_episodes_aloha2 as recorder_module
    import tactile_hdf5

    for name in SENSORS:
        if tactile_stream.topic_for(name) not in TactileStream._streams:
            streams(name)
    recorder = recorder_module.TactileRecorder(SENSORS, ["force"])
    times = 100.0 + 0.02 * np.arange(50)
    samples = [{name: {"fields": {"force": _force(0.1)}, "encoded": {}, "timestamp": t, "read_start": t - 0.01,
                       "read_end": t, "frame_age": frame_age, "seq": i, "new_frame": True} for name in SENSORS}
               for i, t in enumerate(times)]
    packed, lengths = recorder.pack(samples)
    observations = [{"timestamps": {"local_time": t + frame_age, "images": {"cam_high": t - 0.02},
                                    "joints": {"left": t - 0.005}}} for t in times]
    timestamps = tactile_hdf5.observation_timestamps(observations)
    timestamps.update(control_start=times, action_read=times + 0.001, step_done=times + 0.021)
    timestamps["tactile"] = recorder.timing(samples)
    config = {"runtime_configs": {SENSORS[0]: b"\x1f\x8b-not-really-a-tarball"}, "sdk_version": "1.7.0",
              "serials": {SENSORS[0]: "OG000707"}, "calibrated": {SENSORS[0]: True},
              "calibrated_at": 99.0, "exported_at": 99.0, "settings": {"image_codec": "raw"}}
    with h5py.File(str(path), "w") as root:
        obs = root.create_group("observations")
        tactile_hdf5.write_tactile(obs, packed, lengths)
        tactile_hdf5.write_timestamps(obs, timestamps)
        tactile_hdf5.write_tactile_metadata(root, config, recorder.settings())
    recorder.close()


def test_hdf5_timestamps_metadata_and_timing_check(tmp_path, streams):
    import check_episode_timing

    good = tmp_path / "episode_0.hdf5"
    _write_episode(good, streams)
    with h5py.File(str(good), "r") as f:
        assert f["/metadata/tactile/runtime_config/" + SENSORS[0]][()].tobytes() == b"\x1f\x8b-not-really-a-tarball"
        assert f["/metadata/tactile"].attrs["sdk_version"] == "1.7.0"
        assert f["/observations/timestamps/tactile/{}/new_frame".format(SENSORS[0])].dtype == bool
        assert f["/observations/timestamps/images/cam_high"].shape == (50,)
        assert f["/observations/tactile/{}/force".format(SENSORS[0])].shape == (50, 35, 20, 3)
    assert check_episode_timing.main([str(good)]) == 0

    stale = tmp_path / "episode_1.hdf5"
    _write_episode(stale, streams, frame_age=0.3)
    assert check_episode_timing.main([str(stale)]) == 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_vtla_export_keeps_tactile_data_age_and_runtime_config(tmp_path):
    import hil_split_episodes
    from hil_recorder import VtlaHilEpisode
    from policy_client import Frame
    from record_episodes_aloha2_vtla import TACTILE_SENSORS, video_sources

    sources = video_sources("cam_high")
    episode = VtlaHilEpisode(sources, None, [{"closed": -1.5, "open": 0.08}] * 2, "joint_position", "task")
    for t in range(60):                                  # a perfect 50 Hz loop
        now = 5.0 + 0.02 * t
        streams = {}
        for i, name in enumerate(sorted(set(sources.values()))):
            entry = {"image": np.full((32, 48, 3), (t * 5 + i * 20) % 256, dtype=np.uint8),
                     "received": now - 0.01, "stamp": 100.0 + 0.02 * t, "sequence": t + 1}
            if name in TACTILE_SENSORS:
                entry["data_age"] = 0.004
            streams[name] = entry
        for side in ("left", "right"):
            streams["joint_" + side] = {"state": np.zeros(7), "received": now - 0.005, "stamp": 0.0,
                                        "sequence": t + 1}
        episode.add(Frame(now, np.zeros(14), np.zeros(14), np.zeros(14), {}, streams), np.zeros(14), 2, 1)
    config = {"runtime_configs": {name: name.encode() for name in TACTILE_SENSORS}, "sdk_version": "1.7.0",
              "calibrated": {name: True for name in TACTILE_SENSORS}, "calibrated_at": 1.0, "exported_at": 1.0}
    root = tmp_path / "hil"
    episode.save(root / "episode_0", {"policy": "n0_joint"}, tactile_config=config)
    meta = json.loads((root / "episode_0" / "metadata.json").read_text())
    assert meta["tactile_runtime_config"]["sdk_version"] == "1.7.0"
    assert (root / "episode_0" / "tactile_runtime_config" / "left_gripper_outer.tar.gz").read_bytes() == b"left_gripper_outer"
    rows = list(csv.DictReader((root / "episode_0" / "timestamps.csv").open()))
    assert float(rows[0]["left_gripper_outer_data_age_s"]) == pytest.approx(0.004)
    assert float(rows[0]["left_gripper_outer_ros_stamp_s"]) > 0      # local_read_end, not 0
    assert meta["stream_quality"]["left_gripper_outer"]["max_data_age_s"] == pytest.approx(0.004)

    # The 30 Hz rows alternate 20/40 ms between held samples; the loop itself ran every 20 ms.
    import check_episode_timing
    steps, metrics = check_episode_timing.metrics_vtla(root / "episode_0")
    assert all(check_episode_timing.judge(name, stats) == "PASS" for name, stats in metrics.items()), metrics
    assert metrics["control period"]["max"] == pytest.approx(0.02)
    assert check_episode_timing.main([str(root / "episode_0")]) == 0

    hil_split_episodes.main([str(root), str(tmp_path / "split"), "--min_frames", "5"])
    assert (tmp_path / "split" / "episode_0" / "tactile_runtime_config" / "left_gripper_outer.tar.gz").exists()


def test_current_runtime_config_waits_then_warns(monkeypatch, capsys):
    config = tactile_stream.RuntimeConfig(subscribe=False)
    monkeypatch.setattr(tactile_stream.RuntimeConfig, "_instance", config)
    assert tactile_stream.current_runtime_config(timeout_s=0.05) is None
    assert "saved without the tactile runtime config" in capsys.readouterr().out
    threading.Timer(0.05, config.feed, args=({"exported_at": 5.0},)).start()
    assert tactile_stream.current_runtime_config(timeout_s=2.0) == {"exported_at": 5.0}   # returns on arrival
    assert config.wait(0.05, newer_than=5.0) is None              # zeroing's blob must be newer than the call
