"""HDF5 layout shared by record_episodes_aloha2.py and hil_recorder.py: tactile frames, per-step
timestamps and the tactile runtime config. Pure h5py/numpy, no ROS.

    /observations/tactile/<sensor>/<field>     JPEG frames zero-padded to one shared width (image
                                               fields) or float32 (everything else), as before
    /observations/tactile/compress_len         true JPEG byte counts; row order in attr 'entries'
    /observations/timestamps/...               one value per stored step, all host/ROS seconds:
        local_time                             when the observation was assembled (time.time())
        control_start, action_read, step_done  the control loop's own clock for that step's action
        images/<camera>                        the camera message's header stamp (capture time)
        joints/<side>                          the puppet joint_states header stamp
        tactile/<sensor>/timestamp             the XENSE SDK's TimeStamp for that frame
        tactile/<sensor>/read_start, read_end  host time around the publisher's SDK call
        tactile/<sensor>/frame_age             local_time-ish sample time - read_end (the data's age)
        tactile/<sensor>/seq                   the publisher's per-sensor frame counter
        tactile/<sensor>/new_frame             False when the step reused the previous step's frame
    /metadata/tactile                          attrs: sdk_version, settings/serials/calibrated (JSON),
                                               calibrated_at, exported_at, recorder settings (JSON)
    /metadata/tactile/runtime_config/<sensor>  the SDK runtime config exported right after the
                                               zeroing that preceded this episode (tar.gz bytes);
                                               with lossless rectify, Sensor.createSolver() on it
                                               recomputes difference/depth offline
Every dataset under /observations whose first axis is time is sliced by hil_split_episodes.py.
"""

import json

import numpy as np


def write_tactile(obs_group, packed, lengths):
    """/observations/tactile from TactileRecorder.pack()'s (packed, lengths)."""
    if not packed:
        return
    tactile_group = obs_group.create_group("tactile")
    for sensor_name, fields in packed.items():
        sensor_group = tactile_group.create_group(sensor_name)
        for field_name, entry in fields.items():
            dataset = sensor_group.create_dataset(
                field_name, data=entry["data"],
                chunks=(1,) + entry["data"].shape[1:] if entry["data"].ndim > 1 else None)
            dataset.attrs["compressed"] = entry["compressed"]
            dataset.attrs["frame_shape"] = np.array(entry["shape"], dtype="int64")
            if entry.get("jpeg_quality") is not None:
                dataset.attrs["jpeg_quality"] = int(entry["jpeg_quality"])
    if lengths and lengths.get("entries"):
        lengths_dataset = tactile_group.create_dataset("compress_len", data=lengths["lens"])
        lengths_dataset.attrs["entries"] = lengths["entries"]


def write_timestamps(obs_group, timestamps):
    """/observations/timestamps from a nested {name: array or {name: ...}} dict."""
    if not timestamps:
        return
    group = obs_group.create_group("timestamps")

    def _write(parent, mapping):
        for key, value in mapping.items():
            if isinstance(value, dict):
                _write(parent.create_group(key), value)
            else:
                array = np.asarray(value)
                if array.dtype.kind == "f":
                    array = array.astype(np.float64)
                parent.create_dataset(key, data=array)

    _write(group, timestamps)


def observation_timestamps(observations):
    """Stack RealEnv.get_observation()'s per-step obs['timestamps'] into arrays (missing -> NaN)."""
    stamps = [obs.get("timestamps") or {} for obs in observations]
    out = {"local_time": np.array([s.get("local_time", np.nan) for s in stamps], dtype=np.float64)}
    for group in ("images", "joints"):
        names = sorted({name for s in stamps for name in (s.get(group) or {})})
        if names:
            out[group] = {name: np.array([(s.get(group) or {}).get(name, np.nan) for s in stamps], dtype=np.float64)
                          for name in names}
    return out


def write_tactile_metadata(root, config, recorder_settings=None):
    """/metadata/tactile from the /tactile/runtime_config blob (may be None) and recorder settings."""
    metadata = root.require_group("metadata")
    group = metadata.create_group("tactile")
    if recorder_settings:
        group.attrs["recorder_settings_json"] = json.dumps(recorder_settings, sort_keys=True, default=str)
    if not config:
        group.attrs["runtime_config_available"] = False
        return group
    group.attrs["runtime_config_available"] = bool(config.get("runtime_configs"))
    group.attrs["sdk_version"] = str(config.get("sdk_version") or "")
    for key in ("settings", "serials", "calibrated"):
        group.attrs[key + "_json"] = json.dumps(config.get(key) or {}, sort_keys=True, default=str)
    for key in ("calibrated_at", "exported_at"):
        value = config.get(key)
        group.attrs[key] = float(value) if value is not None else np.nan
    configs = group.create_group("runtime_config")
    for sensor, blob in sorted((config.get("runtime_configs") or {}).items()):
        configs.create_dataset(sensor, data=np.void(bytes(blob)))
    return group
