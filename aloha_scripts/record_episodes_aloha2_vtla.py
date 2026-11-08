#!/usr/bin/env python3
"""Record ALOHA episodes for N0-VTLA/scripts/convert_canonical_data.py.

Raw output is episode_N/{metadata.json, observation.state.eef_pose/data.csv,
actions.eef_pose/data.csv, observation.image.*/video.mp4, ...}, not HDF5.
--aloha-action-space defaults to eef (also retains joint CSVs). Select
joint_position to write only joint state/action CSVs and skip EEF computation;
pass the same --aloha-action-space joint_position to the converter.
The seven videos retain their source dimensions (no resize, crop or padding),
encoded from BGR to H.264/yuv420p at 30 Hz. All modalities
use the SAME causal resampling indices from monotonic pre-action snapshots.
EEF poses use each follower arm's base frame and Interbotix model tool frame;
rot6d concatenates the first two rotation-matrix columns. Grippers use the
follower MOTOR angle, normalized with the same endpoints as action commands.

The current VTLA converter defaults to action = current EEF state for ALOHA,
EVEN when actions.eef_pose/data.csv exists. This recorder preserves commanded
EEF and joint actions, but cannot change that converter behavior. Its optional
--eef-action-from-state-shift-frames 1 instead uses the next observed state.

Example (in the ROS recording environment):
  python3 aloha_scripts/record_episodes_aloha2_vtla.py \
    --task_name aloha2_tactile_test --start_idx 0 --end_idx 9 \
    --output_dir /path/to/raw_episodes

Convert (in the N0-VTLA environment):
  python scripts/convert_canonical_data.py /path/to/raw_episodes \
    /path/to/datasets/canonical_tactile_task --robot aloha --task "Perform the task"

Hardware imports are deferred so --help and the export functions work without ROS.
"""

import argparse
import concurrent.futures
import csv
import json
from itertools import chain
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np
from tqdm import tqdm

from tactile_stream import DEFAULT_MAX_TACTILE_AGE_S, TactileStream, calibrate_sensors, current_runtime_config

FPS = 30.0
JOINT_COLS = tuple("{}_{}".format(side, joint) for side in ("left", "right")
                   for joint in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper"))
EEF_COLS = tuple("{}_{}".format(side, field) for side in ("left", "right")
                for field in ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "gripper"))
TACTILE_SENSORS = ("left_gripper_outer", "left_gripper_inner",
                   "right_gripper_outer", "right_gripper_inner")
# Physical left/right depends on installation; override these CLI slots if needed.
DEFAULT_TACTILE_MAP = {
    "left_wrist_left_tactile": "left_gripper_outer",
    "left_wrist_right_tactile": "left_gripper_inner",
    "right_wrist_left_tactile": "right_gripper_inner",
    "right_wrist_right_tactile": "right_gripper_outer",
}


def video_sources(third_camera="cam_high", tactile_map=None):
    mapping = dict(DEFAULT_TACTILE_MAP if tactile_map is None else tactile_map)
    if set(mapping) != set(DEFAULT_TACTILE_MAP) or set(mapping.values()) != set(TACTILE_SENSORS):
        raise ValueError("Tactile mapping must assign the four distinct sensors to the four VTLA slots")
    cameras = {"third_view": third_camera, "left_wrist_view": "cam_left_wrist",
               "right_wrist_view": "cam_right_wrist"}
    return {"observation.image." + key: source for key, source in dict(cameras, **mapping).items()}


def resample_indices(timestamps):
    """Zero-order hold onto 30 Hz; never choose a future observation or extrapolate."""
    stamps = np.asarray(timestamps, dtype=np.float64)
    if stamps.ndim != 1 or len(stamps) < 2 or not np.isfinite(stamps).all():
        raise ValueError("Need at least two finite sample timestamps")
    if np.any(np.diff(stamps) <= 0):
        raise ValueError("Sample timestamps must be strictly increasing")
    relative = stamps - stamps[0]
    grid = np.arange(int(np.floor(relative[-1] * FPS)) + 1, dtype=np.float64) / FPS
    indices = np.searchsorted(relative, grid, side="right") - 1
    return grid, indices


def joint_state_vector(message, arm_names, gripper_closed, gripper_open):
    """Read one immutable JointState message; do not use finger displacement index 7."""
    if len(arm_names) != 6 or gripper_open <= gripper_closed:
        raise ValueError("Expected six arm joints and increasing gripper calibration endpoints")
    positions = dict(zip(message.name, message.position))
    result = np.array([positions[name] for name in arm_names] + [positions["gripper"]], dtype=np.float64)
    result[-1] = (result[-1] - gripper_closed) / (gripper_open - gripper_closed)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite joint feedback")
    # Preserve small calibration overshoots instead of hiding them by clipping feedback.
    return result


def joints_to_eef(joints, models):
    import modern_robotics as mr
    if len(models) != 2:
        raise ValueError("Expected left and right follower kinematics models")
    for model in models:
        if (np.shape(model.M) != (4, 4) or np.shape(model.Slist) != (6, 6)
                or not np.isfinite(model.M).all() or not np.isfinite(model.Slist).all()):
            raise ValueError("Invalid six-joint follower kinematics model")
    values = np.asarray(joints, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 14 or not np.isfinite(values).all():
        raise ValueError("Expected finite (T, 14) joint values")
    out = np.empty((len(values), 20), dtype=np.float32)
    for row, q in enumerate(values):
        for arm, model in enumerate(models):
            offset, dest = arm * 7, arm * 10
            pose = mr.FKinSpace(model.M, model.Slist, q[offset:offset + 6])
            out[row, dest:dest + 10] = np.concatenate(
                [pose[:3, 3], pose[:3, 0], pose[:3, 1], [q[offset + 6]]])
    return out


def check_video_tools():
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            raise RuntimeError("{} is required to write and verify VTLA videos".format(name))
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                              check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if "libx264" not in encoders.stdout:
        raise RuntimeError("ffmpeg needs the libx264 encoder")


def write_video(path, frames, frame_count):
    frames = iter(frames)
    first = np.asarray(next(frames, None))
    if first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8:
        raise ValueError("Expected a nonempty sequence of uint8 BGR images")
    height, width = first.shape[:2]
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise ValueError("H.264/yuv420p needs positive even dimensions, got {}x{}; "
                         "source images will not be resized or padded".format(width, height))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "{}x{}".format(width, height),
               "-framerate", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-threads", "2", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            count = 0
            for frame in chain((first,), frames):
                frame = np.asarray(frame)
                if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
                    raise ValueError("Video frame size/type changed: expected uint8 {}, got {} {}".format(
                        (height, width, 3), frame.shape, frame.dtype))
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
                count += 1
            process.stdin.close()
            code = process.wait()
            if code or count != frame_count:
                errors.seek(0)
                raise RuntimeError("Video export failed: {}".format(errors.read().decode(errors="replace")))
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            if not process.stdin.closed:
                process.stdin.close()
            raise
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height,codec_name,pix_fmt,avg_frame_rate,nb_read_frames",
                            "-of", "json", str(path)], check=True, stdout=subprocess.PIPE, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    if (stream["width"], stream["height"], stream["codec_name"], stream["pix_fmt"],
        int(stream["nb_read_frames"]), numerator / denominator) != (width, height, "h264", "yuv420p", frame_count, FPS):
        raise RuntimeError("Encoded video does not match source dimensions/frame count: {}".format(stream))


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def export_episode(destination, samples, models, sources, task, *, overwrite=False, calibration=None,
                   action_space="eef", extra_csvs=None, extra_metadata=None, extra_files=None):
    """Publish a complete episode directory only after CSV/video validation succeeds.

    extra_csvs: {key: (columns, values)} with one row of `values` per source sample, resampled
    with the same indices as every other CSV and written to <key>/data.csv (hil_rollout.py's
    per-tick labels). extra_metadata is merged into metadata.json. extra_files: {relative path:
    bytes} written as they are (the tactile runtime configs, see tactile_config_extras()).
    """
    if action_space not in ("eef", "joint_position"):
        raise ValueError("Unsupported ALOHA action space: {}".format(action_space))
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError("Episode already exists: {}".format(destination))
    if len(sources) != 7 or len(set(sources.values())) != 7:
        raise ValueError("Expected seven distinct VTLA image sources")
    grid, indices = resample_indices([sample["time"] for sample in samples])
    selected = [samples[int(index)] for index in indices]
    state = np.asarray([sample["state"] for sample in selected], dtype=np.float32)
    action = np.asarray([sample["action"] for sample in selected], dtype=np.float32)
    count = len(selected)
    if (state.shape != (count, 14) or action.shape != (count, 14)
            or not np.isfinite(state).all() or not np.isfinite(action).all()):
        raise ValueError("Expected finite (T, 14) joint state and action")
    csv_specs = [("observation.state.joint_position", JOINT_COLS, state),
                 ("actions.joint_position", JOINT_COLS, action)]
    if action_space == "eef":
        csv_specs.extend([("observation.state.eef_pose", EEF_COLS, joints_to_eef(state, models)),
                          ("actions.eef_pose", EEF_COLS, joints_to_eef(action, models))])
    for key, (cols, values) in (extra_csvs or {}).items():
        values = np.asarray(values)
        if len(values) != len(samples):
            raise ValueError("extra CSV {} needs one row per sample".format(key))
        csv_specs.append((key, tuple(cols), values[indices]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="." + destination.name + "-", dir=str(destination.parent)) as temporary:
        root = Path(temporary) / "complete"
        root.mkdir()
        for key, cols, values in csv_specs:
            write_csv(root / key / "data.csv", ("timestamp",) + cols,
                      ((stamp,) + tuple(row) for stamp, row in zip(grid, values)))
        # The 7 video files are fully independent (their own ffmpeg subprocess, their own
        # frames), so write them concurrently instead of one after another - each write_video()
        # call is mostly waiting on ffmpeg/ffprobe subprocesses, not doing CPU work in this
        # thread, so plain threads (not multiprocessing) already parallelise the wall-clock time
        # well. On a 24-core bench this cut a 450-frame episode's video-writing step from ~45s
        # (7 x sequential encode+verify) to ~7s (all 7 running at once, bounded by the slowest
        # one) - 7 x "-threads 2" ffmpeg processes is still far under the core count.
        # ThreadPoolExecutor (not bare threading.Thread) specifically so a write_video() failure
        # - a corrupt frame, ffmpeg crashing - surfaces via future.result() instead of vanishing
        # into a background thread while the episode gets saved as if it had succeeded.
        video_sizes = {}
        for key, source in sources.items():
            height, width = selected[0]["streams"][source]["image"].shape[:2]
            video_sizes[key] = [width, height]

        def _write_one(key, source):
            write_video(root / key / "video.mp4",
                       (sample["streams"][source]["image"] for sample in selected), count)

        print("Writing {} videos ({} frames each) in parallel...".format(len(sources), count), flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sources)) as pool:
            futures = {pool.submit(_write_one, key, source): key for key, source in sources.items()}
            for future in concurrent.futures.as_completed(futures):
                future.result()  # re-raises here, with the traceback, if that video failed
                print("  wrote {}".format(futures[future]), flush=True)
        for sensor in TACTILE_SENSORS:
            fields = [sample["streams"][sensor].get("force") for sample in selected]
            if any(field is not None for field in fields):
                if any(field is None for field in fields):
                    raise ValueError("Incomplete force stream: " + sensor)
                force = np.asarray(fields, dtype=np.float32)
                if force.shape != (count, 35, 20, 3) or not np.isfinite(force).all():
                    raise ValueError("Invalid force field: " + sensor)
                target = root / "observation.tactile.force" / sensor
                target.mkdir(parents=True)
                np.save(str(target / "data.npy"), force, allow_pickle=False)
        stream_names = list(sources.values()) + ["joint_left", "joint_right"]
        tactile_names = [name for name in stream_names if name in TACTILE_SENSORS]
        # Tactile data age (sample time - the publisher's local_read_end) is appended after the
        # per-stream columns, so readers that look columns up by name are unaffected.
        timing_cols = ("timestamp", "source_index", "sample_elapsed_s", "resample_age_s") + tuple(
            name + suffix for name in stream_names for suffix in ("_sequence", "_receipt_age_s", "_ros_stamp_s")
        ) + tuple(name + "_data_age_s" for name in tactile_names)
        timing_rows = []
        for stamp, index, sample in zip(grid, indices, selected):
            elapsed = sample["time"] - samples[0]["time"]
            row = [stamp, int(index), elapsed, max(0.0, stamp - elapsed)]
            for name in stream_names:
                entry = sample["streams"][name]
                row.extend([entry["sequence"], sample["time"] - entry["received"], entry["stamp"]])
            row.extend(sample["streams"][name].get("data_age", float("nan")) for name in tactile_names)
            timing_rows.append(row)
        write_csv(root / "timestamps.csv", timing_cols, timing_rows)
        quality = {}
        for name in stream_names:
            sequence = [sample["streams"][name]["sequence"] for sample in selected]
            quality[name] = {
                "reused_adjacent_frames": int(np.count_nonzero(np.diff(sequence) == 0)),
                "max_receipt_age_s": max(sample["time"] - sample["streams"][name]["received"] for sample in selected),
            }
            if name in tactile_names:
                ages = [sample["streams"][name].get("data_age", float("nan")) for sample in selected]
                quality[name]["max_data_age_s"] = float(np.nanmax(ages)) if np.isfinite(ages).any() else None
        for relative, data in (extra_files or {}).items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        metadata = {
            "episode_id": destination.name, "total_frames": count, "fps": FPS,
            "robot": "aloha", "task": task, "robot_model": "vx300s", "video_sources": sources,
            "aloha_action_space": action_space,
            "state_csv": "observation.state.{}/data.csv".format("eef_pose" if action_space == "eef" else "joint_position"),
            "action_csv": "actions.{}/data.csv".format("eef_pose" if action_space == "eef" else "joint_position"),
            "video_sizes": video_sizes, "video_size_order": "width,height", "image_preprocessing": "none",
            "video_codec": "h264", "video_pix_fmt": "yuv420p",
            "joint_position_units": "radians",
            "gripper_representation": "follower motor angle normalized by calibrated close/open endpoints; feedback unclipped",
            "gripper_calibration": calibration,
            "sample_phase": "all observations sampled before the associated action is sent",
            "resampling": "30Hz zero-order hold using one common source index for all fields",
            "source_samples": len(samples), "source_duration_s": samples[-1]["time"] - samples[0]["time"],
            "max_resample_age_s": max(row[3] for row in timing_rows),
            "timing_note": ("receipt ages use host monotonic time; ROS header stamps are separate clocks; tactile "
                            "*_ros_stamp_s is the publisher's local_read_end (host time.time() when the XENSE SDK "
                            "delivered the frame; 0 in episodes recorded before 2026-11) and *_data_age_s the "
                            "frame's age when sampled"),
            "stream_quality": quality,
            "converter_action_note": (
                "Current convert_canonical_data.py --robot aloha defaults to action=state and ignores actions.eef_pose CSV"
                if action_space == "eef" else
                "Use --aloha-action-space joint_position to read actions.joint_position/data.csv as commanded targets"),
            "force_note": "Optional SDK force saved verbatim as float32 NPY; converter does not consume force arrays",
        }
        if action_space == "eef":
            metadata.update({
                "eef_position_units": "meters",
                "eef_reference_frames": ["puppet_left/base_link", "puppet_right/base_link"],
                "eef_tool_frame": "Interbotix robot_des.M tool frame (no extra TCP offset)",
                "rotation_representation": "rot6d: concatenate R[:,0], R[:,1]",
                "kinematics": [{"M": np.asarray(model.M).tolist(), "Slist": np.asarray(model.Slist).tolist()} for model in models],
            })
        metadata.update(extra_metadata or {})
        (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # If an explicit overwrite fails, restore the previous complete episode.
        backup = Path(temporary) / "previous"
        if destination.exists():
            if not overwrite:
                raise FileExistsError(str(destination))
            os.rename(str(destination), str(backup))
        try:
            os.rename(str(root), str(destination))
        except BaseException:
            if backup.exists():
                os.rename(str(backup), str(destination))
            raise
    print("Saved {}: {} frames at 30 fps, action space {}".format(destination, count, action_space))
    return metadata


class LiveStreams:
    """Immutable latest values with callback counters and host receipt timestamps.

    Camera callbacks explicitly convert to BGR. Tactile comes from each sensor's shared
    tactile_stream.TactileStream, whose callback thread already turns rectify into BGR (prefetch),
    whatever image codec the publisher uses, so snapshot() only copies it. Missing or stale streams
    stop recording; they are never replaced by black frames. Cameras and joints are stale by receipt
    age (max_age); tactile by the data's own age, now - the publisher's local_read_end
    (max_tactile_age), which also catches a publisher re-sending an old frame. Receipt counters
    measure message reuse; the tactile publisher sends each SDK frame once, so a reused tactile
    counter means no new SDK frame.
    """

    def __init__(self, sources, arm_names, calibration, save_force, max_tactile_age=DEFAULT_MAX_TACTILE_AGE_S):
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image, JointState
        self.rospy = rospy
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest = {}
        self.subs = []
        self.names = list(sources.values()) + ["joint_left", "joint_right"]
        self.save_force = save_force
        self.max_tactile_age = float(max_tactile_age)
        self.tactile_names = list(TACTILE_SENSORS)
        self.tactile = {sensor: TactileStream.acquire(sensor, prefetch=(("rectify", "bgr"),))
                        for sensor in TACTILE_SENSORS}
        for camera in dict.fromkeys(source for source in sources.values() if source.startswith("cam_")):
            topic = "/usb_{}/image_raw".format(camera)
            self.subs.append(rospy.Subscriber(topic, Image, self.camera_callback, callback_args=camera,
                                              queue_size=1, buff_size=2 ** 24, tcp_nodelay=True))
        for side, names, endpoints in zip(("left", "right"), arm_names, calibration):
            self.subs.append(rospy.Subscriber("/puppet_{}/joint_states".format(side), JointState,
                                              self.joint_callback, callback_args=(side, names, endpoints),
                                              queue_size=1, tcp_nodelay=True))

    def put(self, name, value, received, stamp=0.0):
        with self.lock:
            sequence = self.latest.get(name, {}).get("sequence", 0) + 1
            self.latest[name] = dict(value, received=received, stamp=stamp, sequence=sequence)

    def camera_callback(self, message, name):
        received = time.monotonic()
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            # Retain an owned snapshot at the exact incoming dimensions.
            image = np.array(frame, copy=True, order="C")
            self.put(name, {"image": image}, received, message.header.stamp.to_sec())
        except Exception as exc:
            self.rospy.logwarn_throttle(5, "VTLA camera {}: {}".format(name, exc))

    def tactile_value(self, frame, data_age):
        """One tactile frame as a snapshot entry: BGR rectify (+ force), receipt/stamp/sequence/data_age."""
        # The XENSE SDK's own type stub claims "rectify" is BGR, but the raw bytes off the publisher
        # are RGB (confirmed against XenseStudio's reference rendering); field("rectify", "bgr")
        # converts (or decodes a jpeg publisher's frame) so this matches camera_callback's true BGR
        # and write_video()'s "bgr24" input assumption.
        image = frame.field("rectify", "bgr")
        if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("rectify must be a uint8 image")
        value = {"image": np.array(image, copy=True, order="C"), "received": frame.recv_mono,
                 "stamp": frame.read_end, "sequence": frame.count, "data_age": data_age}
        if self.save_force:
            force = frame.field("force")
            force = None if force is None else np.array(force, dtype=np.float32, copy=True)
            if force is None or force.shape != (35, 20, 3) or not np.isfinite(force).all():
                raise ValueError("force must be finite (35, 20, 3)")
            value["force"] = force
        return value

    def joint_callback(self, message, config):
        received = time.monotonic()
        side, names, endpoints = config
        try:
            state = joint_state_vector(message, names, endpoints["closed"], endpoints["open"])
            self.put("joint_" + side, {"state": state}, received, message.header.stamp.to_sec())
        except Exception as exc:
            self.rospy.logwarn_throttle(5, "VTLA joint {}: {}".format(side, exc))

    def snapshot(self, max_age, max_tactile_age=None):
        tactile_limit = self.max_tactile_age if max_tactile_age is None else max_tactile_age
        with self.lock:
            now, now_wall = time.monotonic(), time.time()
            entries, missing, stale = {}, [], []
            for name in self.names:
                if name in self.tactile:
                    frame = self.tactile[name].latest()
                    if frame is None:
                        missing.append(name)
                        continue
                    data_age = frame.age(now_wall)
                    if data_age > tactile_limit:
                        stale.append(name)
                        continue
                    try:
                        entries[name] = self.tactile_value(frame, data_age)
                    except ValueError as exc:
                        raise RuntimeError("VTLA tactile {}: {}".format(name, exc))
                elif name not in self.latest:
                    missing.append(name)
                elif now - self.latest[name]["received"] > max_age:
                    stale.append(name)
                else:
                    entries[name] = self.latest[name]
            if missing or stale:
                raise RuntimeError("VTLA streams unavailable: missing={}, stale={}".format(missing, stale))
            return now, entries

    def wait_ready(self, timeout, max_age):
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.snapshot(max_age)
                return
            except RuntimeError:
                if time.monotonic() >= deadline or self.rospy.is_shutdown():
                    raise
                time.sleep(0.05)

    def close(self):
        for subscriber in self.subs:
            subscriber.unregister()
        self.subs.clear()
        for stream in self.tactile.values():
            stream.release()
        self.tactile = {}


def tactile_config_extras(config):
    """(extra_files, extra_metadata) for export_episode from a /tactile/runtime_config blob.

    The runtime configs (what the XENSE SDK's Sensor.createSolver needs) go to
    tactile_runtime_config/<sensor>.tar.gz; their provenance to metadata.json.
    """
    if not config:
        return {}, {}
    files = {"tactile_runtime_config/{}.tar.gz".format(sensor): bytes(blob)
             for sensor, blob in sorted((config.get("runtime_configs") or {}).items())}
    metadata = {"tactile_runtime_config": {
        "files": sorted(files), "sdk_version": config.get("sdk_version"), "serials": config.get("serials"),
        "calibrated": config.get("calibrated"), "calibrated_at": config.get("calibrated_at"),
        "exported_at": config.get("exported_at"), "publisher_settings": config.get("settings"),
    }}
    return files, metadata


def capture_one_episode(args, legacy, config, destination, sources):
    from constants import (DT, PUPPET_LEFT_GRIPPER_JOINT_CLOSE, PUPPET_LEFT_GRIPPER_JOINT_OPEN,
                           PUPPET_RIGHT_GRIPPER_JOINT_CLOSE, PUPPET_RIGHT_GRIPPER_JOINT_OPEN)
    calibration = [{"closed": PUPPET_LEFT_GRIPPER_JOINT_CLOSE, "open": PUPPET_LEFT_GRIPPER_JOINT_OPEN},
                   {"closed": PUPPET_RIGHT_GRIPPER_JOINT_CLOSE, "open": PUPPET_RIGHT_GRIPPER_JOINT_OPEN}]
    if destination.exists() and not args.overwrite:
        raise FileExistsError("{} exists; choose another index or use --overwrite".format(destination))
    master_left = legacy.InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                                 robot_name="master_left", init_node=True)
    master_right = legacy.InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                                  robot_name="master_right", init_node=False)
    env = legacy.make_real_env(init_node=False, setup_robots=False)
    models = ((env.puppet_bot_left.arm.robot_des, env.puppet_bot_right.arm.robot_des)
              if args.aloha_action_space == "eef" else None)
    arm_names = (env.puppet_bot_left.arm.group_info.joint_names, env.puppet_bot_right.arm.group_info.joint_names)
    keyboard = legacy.KeyboardControl()
    haptics, samples = [], []
    streams = None
    progress = None
    leaders_released = False
    tactile_config = None
    try:
        streams = LiveStreams(sources, arm_names, calibration, args.save_force,
                              max_tactile_age=args.max_tactile_age)
        streams.wait_ready(args.stream_timeout, args.max_stream_age)
        legacy.opening_ceremony(master_left, master_right, env.puppet_bot_left, env.puppet_bot_right,
                                destination.name, start_open_ratio=args.start_open_ratio)
        if not args.no_force_feedback:
            for side, bot in (("left", master_left), ("right", master_right)):
                feedback, brake = legacy.build_haptics_config(vars(args), side)
                haptic = legacy.ArmHaptics(bot, side, feedback, brake, torque_gate=not args.no_torque_gate)
                haptics.append(haptic)
                haptic.wait_for_data()
        # Re-zero every tactile sensor (followers at the start pose, grippers open, nothing touching)
        # before the force baseline; the runtime config that zeroing exports is saved with the episode.
        if args.no_tactile_calibration:
            tactile_config = current_runtime_config()
        else:
            print("Zeroing the tactile sensors - nothing may touch them...")
            tactile_config = calibrate_sensors(streams.tactile_names)
        for haptic in haptics:
            haptic.calibrate(args.force_baseline_samples)
        print("Ready for {}. RIGHT=start; while recording RIGHT=save, LEFT=redo, ESC=quit.".format(destination))
        if not legacy.wait_for_start(keyboard):
            return "quit"
        leaders_released = True
        legacy.torque_off(master_left)
        legacy.torque_off(master_right)
        for haptic in haptics:
            haptic.start()
        max_steps = args.max_timesteps or config["episode_len"]
        decision = None
        progress = tqdm(total=max_steps, desc=destination.name, unit="frame", dynamic_ncols=True)
        for _ in range(max_steps):
            if legacy.rospy.is_shutdown():
                decision = "quit"
                break
            for event in keyboard.poll():
                if event in ("right", "left", "esc"):
                    decision = {"right": "save", "left": "redo", "esc": "quit"}[event]
                    break
                if event == "rezero":
                    for haptic in haptics:
                        haptic.rezero(args.force_baseline_samples)
            if decision is not None:
                break
            # Snapshot ALL state/images/tactile BEFORE sending this sample's action. A missing or
            # stale stream stops the episode, which is then recorded again.
            try:
                stamp, snapshot = streams.snapshot(args.max_stream_age)
            except RuntimeError as exc:
                progress.write("Stopped: {}. This episode will be recorded again.".format(exc))
                decision = "redo"
                break
            state = np.concatenate([snapshot["joint_left"]["state"], snapshot["joint_right"]["state"]])
            action = np.array(legacy.get_action(master_left, master_right), dtype=np.float64, copy=True)
            if action.shape != (14,) or not np.isfinite(action).all():
                raise ValueError("Invalid leader action")
            action[[6, 13]] = np.clip(action[[6, 13]], 0, 1)
            env.step(action)
            samples.append({"time": stamp, "state": state, "action": action, "streams": snapshot})
            for haptic in haptics:
                haptic.update()
            time.sleep(max(0.0, DT - (time.monotonic() - stamp)))
            progress.update(1)
        progress.close()
        # Match the existing recorder: restore haptics, hold leaders and open followers.
        for haptic in haptics:
            haptic.restore()
        haptics.clear()
        legacy.torque_on(master_left)
        legacy.torque_on(master_right)
        leaders_released = False
        for bot in (env.puppet_bot_left, env.puppet_bot_right):
            bot.dxl.robot_set_operating_modes("single", "gripper", "position")
        legacy.move_grippers([env.puppet_bot_left, env.puppet_bot_right],
                              [PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN], move_time=0.5)
        if decision is None:
            decision = legacy.wait_for_decision(keyboard, "RIGHT=save, LEFT=redo, ESC=quit")
    finally:
        if progress is not None:
            progress.close()
        for haptic in haptics:
            haptic.restore()
        if leaders_released:
            legacy.torque_on(master_left)
            legacy.torque_on(master_right)
        keyboard.close()
        if streams is not None:
            streams.close()
    if decision != "save":
        return decision
    if len(samples) < 2:
        print("Episode too short; record it again.")
        return "redo"
    extra_files, extra_metadata = tactile_config_extras(tactile_config)
    export_episode(destination, samples, models, sources, args.task or args.task_name,
                   overwrite=args.overwrite, calibration=calibration, action_space=args.aloha_action_space,
                   extra_metadata=extra_metadata, extra_files=extra_files)
    return "save"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--output_dir", type=Path, help="Raw episode root; default: task dataset_dir")
    parser.add_argument("--task", help="Task description stored in raw metadata")
    parser.add_argument("--aloha-action-space", "--aloha_action_space", choices=("eef", "joint_position"),
                        default="eef", help="Default: eef plus joint CSVs; joint_position skips EEF export. "
                        "Use the same action-space option when converting.")
    parser.add_argument("--episode_idx", type=int, help="Record one episode (mutually exclusive with start/end)")
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--end_idx", type=int, help="Inclusive final index")
    parser.add_argument("--max_timesteps", type=int, help="Control samples, before 30 Hz resampling")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--third_camera", choices=("cam_high", "cam_low"), default="cam_high")
    for slot, sensor in DEFAULT_TACTILE_MAP.items():
        parser.add_argument("--" + slot, choices=TACTILE_SENSORS, default=sensor)
    parser.add_argument("--save_force", action="store_true", help="Also save force NPY; requires SDK extra_outputs:=force")
    parser.add_argument("--stream_timeout", type=float, default=5.0, help="Startup wait for all streams, seconds")
    parser.add_argument("--max_stream_age", type=float, default=0.5, help="Stop if any stream receipt is older than this, seconds")
    parser.add_argument("--max_tactile_age", type=float, default=DEFAULT_MAX_TACTILE_AGE_S,
                        help="Stop (and re-record) if a tactile frame is older than this, seconds, by the data's own age")
    parser.add_argument("--no_tactile_calibration", action="store_true",
                        help="Do not re-zero the tactile sensors (/tactile/calibrate) before each episode")
    parser.add_argument("--no_force_feedback", action="store_true")
    parser.add_argument("--no_torque_gate", action="store_true")
    for name in ("start_open_ratio", "force_deadband", "force_full_scale", "wall_stiffness_min",
                 "wall_stiffness_max", "max_feedback_pwm", "close_speed_limit", "close_brake_gain"):
        parser.add_argument("--" + name, type=float, default=argparse.SUPPRESS)
    parser.add_argument("--force_baseline_samples", type=int, default=argparse.SUPPRESS)
    return parser


def get_auto_index(root):
    root = Path(root)
    index = 0
    while (root / ("episode_" + str(index))).exists():
        index += 1
    return index


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.episode_idx is not None and (args.start_idx is not None or args.end_idx is not None):
        parser.error("--episode_idx cannot be combined with --start_idx/--end_idx")
    if args.end_idx is not None and args.start_idx is None:
        parser.error("--end_idx requires --start_idx")
    if any(value is not None and value < 0 for value in (args.episode_idx, args.start_idx, args.end_idx)):
        parser.error("episode indices must be non-negative")
    if args.start_idx is not None and args.end_idx is not None and args.end_idx < args.start_idx:
        parser.error("--end_idx must be >= --start_idx")
    if args.max_timesteps is not None and args.max_timesteps < 2:
        parser.error("--max_timesteps must be at least 2")
    if not all(np.isfinite(value) and value > 0 for value in (args.stream_timeout, args.max_stream_age)):
        parser.error("stream timeouts must be finite and positive")
    try:
        sources = video_sources(args.third_camera, {key: getattr(args, key) for key in DEFAULT_TACTILE_MAP})
    except ValueError as exc:
        parser.error(str(exc))
    check_video_tools()
    # Reuse the established hardware setup, key controls and feedback tuning.
    import record_episodes_aloha2 as legacy
    config = legacy.TASK_CONFIGS[args.task_name]
    for name in ("start_open_ratio", "force_deadband", "force_full_scale", "wall_stiffness_min",
                 "wall_stiffness_max", "max_feedback_pwm", "close_speed_limit", "close_brake_gain",
                 "force_baseline_samples"):
        if not hasattr(args, name):
            setattr(args, name, getattr(legacy, "DEFAULT_" + name.upper()))
    root = args.output_dir if args.output_dir is not None else Path(config["dataset_dir"])
    start = args.episode_idx if args.episode_idx is not None else args.start_idx
    start = get_auto_index(root) if start is None else start
    end = args.end_idx if args.end_idx is not None else start
    for index in range(start, end + 1):
        if (root / ("episode_" + str(index))).exists() and not args.overwrite:
            parser.error("Episode exists: {}; choose a new index or --overwrite".format(root / ("episode_" + str(index))))
    print("Raw output: {}\nVideo sources: {}".format(root, json.dumps(sources, indent=2)))
    print("ALOHA action space: {}".format(args.aloha_action_space))
    if args.aloha_action_space == "eef":
        print("Converter notice: --robot aloha defaults to action=state, ignoring commanded EEF CSV.")
    else:
        print("Convert with --robot aloha --aloha-action-space joint_position to use recorded joint targets.")
    try:
        for index in range(start, end + 1):
            destination = root / ("episode_" + str(index))
            while True:
                decision = capture_one_episode(args, legacy, config, destination, sources)
                if decision == "quit":
                    return
                if decision == "save":
                    break
                legacy.sleep_all_robots()
            if index < end:
                legacy.sleep_all_robots()
    finally:
        legacy.shut_down_all_robots()


if __name__ == "__main__":
    main()
