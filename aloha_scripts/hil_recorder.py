"""Episode writers for hil_rollout.py, in the raw formats the training repos already read.

Hdf5HilEpisode  record_episodes_aloha2.py's HDF5 layout, byte for byte (JPEG quality 90 images
                zero-padded to one width + /compress_len, float32 qpos/qvel/effort/action, zero
                base_action, optional /observations/tactile), so openpi's
                convert_aloha_data_to_lerobot_no_bgr2rgb*.py, lerobot's
                convert_aloha_hdf5_to_lerobot_v30.py and ACT training read it unchanged. Images are
                JPEG-encoded in a thread pool while recording (the plain recorder encodes after the
                episode), which keeps a long episode's RAM bounded.
VtlaHilEpisode  record_episodes_aloha2_vtla.py's raw episode directory (export_episode).

Both add the human-in-the-loop labels, one per recorded control tick:
    phase         0 = policy driving, 1 = paused (follower holding, leader aligning/releasing),
                  2 = human correcting
    intervention  1 on correcting ticks after the first correction_skip_s of each correction
HDF5: /hil/phase, /hil/intervention (uint8, (T,)), /hil/leader_action ((T, 14) absolute leader
reading, the clutch offset is leader_action - action on the arm joints), /hil/time (s since the
first tick) and root attrs hil_* (policy, prompt, settings, server metadata as JSON).
VTLA: hil/data.csv (timestamp, phase, intervention) resampled onto the same 30 Hz grid as every
other CSV, plus a "hil" entry in metadata.json.
hil_split_episodes.py cuts either format into per-segment episodes (e.g. corrections only).
"""

import concurrent.futures
import json

import cv2
import numpy as np

PHASE_AUTONOMOUS, PHASE_PAUSED, PHASE_CORRECTING = 0, 1, 2
JPEG_QUALITY = 90  # record_episodes_aloha2.py's value


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def to_json(value):
    return json.dumps(value, default=_json_default, ensure_ascii=False)


class Hdf5HilEpisode:
    def __init__(self, camera_names, tactile=None, encode_workers=4):
        self.camera_names = list(camera_names)
        # record_episodes_aloha2.TactileRecorder or None. hil_rollout attaches it to the env, so each
        # Frame carries its own tactile snapshot (frame.tactile), taken with that tick's images.
        self.tactile = tactile
        self.qpos, self.qvel, self.effort, self.action = [], [], [], []
        self.phase, self.intervention, self.leader_action, self.times = [], [], [], []
        self.tactile_samples, self.observation_stamps = [], []
        self.images = {camera: [] for camera in self.camera_names}
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=encode_workers,
                                                           thread_name_prefix="hil-jpeg")

    def __len__(self):
        return len(self.action)

    @staticmethod
    def _encode(image):
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return encoded.reshape(-1)

    def add(self, frame, action, phase, intervention, leader_action):
        for camera in self.camera_names:
            image = frame.images.get(camera)
            if image is None:
                raise RuntimeError("camera {} has not published yet".format(camera))
            self.images[camera].append(self._pool.submit(self._encode, image))
        self.qpos.append(np.asarray(frame.qpos, dtype=np.float64))
        self.qvel.append(np.asarray(frame.qvel, dtype=np.float64))
        self.effort.append(np.asarray(frame.effort, dtype=np.float64))
        self.action.append(np.asarray(action, dtype=np.float64))
        self.phase.append(int(phase))
        self.intervention.append(int(bool(intervention)))
        self.leader_action.append(np.asarray(leader_action, dtype=np.float64))
        self.times.append(float(frame.time))
        self.observation_stamps.append({"timestamps": getattr(frame, "timestamps", None) or {}})
        if self.tactile is not None:
            if getattr(frame, "tactile", None) is None:
                raise RuntimeError("frame carries no tactile snapshot (attach the TactileRecorder as env.tactile)")
            self.tactile_samples.append(frame.tactile)

    def close(self):
        """Release the encoder threads and the tactile subscriptions (after save() or a discard)."""
        self._pool.shutdown(wait=True)
        if self.tactile is not None:
            self.tactile.close()

    def save(self, path, attrs=None, tactile_config=None):
        """Write <path> (".hdf5" appended if missing) and return the number of timesteps.

        tactile_config: the /tactile/runtime_config blob of this episode's zeroing (or None).
        """
        import h5py

        from tactile_hdf5 import observation_timestamps, write_tactile, write_tactile_metadata, write_timestamps

        if not path.endswith(".hdf5"):
            path += ".hdf5"
        episode_len = len(self.action)
        if episode_len == 0:
            raise ValueError("nothing recorded")
        encoded = {camera: [future.result() for future in self.images[camera]] for camera in self.camera_names}
        compressed_len = np.array([[len(e) for e in encoded[camera]] for camera in self.camera_names])
        padded_size = int(compressed_len.max()) if self.camera_names else 0
        timestamps = observation_timestamps(self.observation_stamps)
        tactile_packed, tactile_lengths = {}, {"entries": [], "lens": None}
        if self.tactile is not None:
            tactile_packed, tactile_lengths = self.tactile.pack(self.tactile_samples, compress=True)
            timestamps["tactile"] = self.tactile.timing(self.tactile_samples)

        with h5py.File(path, "w", rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs["sim"] = False
            root.attrs["compress"] = True
            obs = root.create_group("observations")
            image = obs.create_group("images")
            for camera in self.camera_names:
                dataset = image.create_dataset(camera, (episode_len, padded_size), dtype="uint8",
                                               chunks=(1, padded_size))
                padded = np.zeros((episode_len, padded_size), dtype="uint8")
                for row, data in enumerate(encoded[camera]):
                    padded[row, :len(data)] = data
                dataset[...] = padded
            obs.create_dataset("qpos", (episode_len, 14))[...] = np.asarray(self.qpos)
            obs.create_dataset("qvel", (episode_len, 14))[...] = np.asarray(self.qvel)
            obs.create_dataset("effort", (episode_len, 14))[...] = np.asarray(self.effort)
            root.create_dataset("action", (episode_len, 14))[...] = np.asarray(self.action)
            root.create_dataset("base_action", (episode_len, 2))
            if self.camera_names:
                root.create_dataset("compress_len", (len(self.camera_names), episode_len))[...] = compressed_len

            # Same layout as record_episodes_aloha2.py: /observations/tactile, /observations/timestamps
            # and /metadata/tactile (tactile_hdf5.py)
            write_tactile(obs, tactile_packed, tactile_lengths)
            write_timestamps(obs, timestamps)
            if tactile_packed or tactile_config is not None:
                write_tactile_metadata(root, tactile_config,
                                       self.tactile.settings() if self.tactile is not None else None)

            hil = root.create_group("hil")
            hil.create_dataset("phase", data=np.asarray(self.phase, dtype=np.uint8))
            hil.create_dataset("intervention", data=np.asarray(self.intervention, dtype=np.uint8))
            hil.create_dataset("leader_action", data=np.asarray(self.leader_action, dtype=np.float32))
            hil.create_dataset("time", data=np.asarray(self.times) - self.times[0])
            root.attrs["hil"] = True
            for key, value in (attrs or {}).items():
                root.attrs["hil_" + key] = value if isinstance(value, (int, float, str)) else to_json(value)
        return episode_len


class VtlaHilEpisode:
    def __init__(self, sources, models, calibration, action_space, task):
        self.sources = dict(sources)
        self.models = models
        self.calibration = calibration
        self.action_space = action_space
        self.task = task
        self.samples = []
        self.phase, self.intervention = [], []

    def __len__(self):
        return len(self.samples)

    def add(self, frame, action, phase, intervention, leader_action=None):
        action = np.array(action, dtype=np.float64, copy=True)
        action[[6, 13]] = np.clip(action[[6, 13]], 0, 1)  # as record_episodes_aloha2_vtla.py
        state = np.concatenate([frame.vtla["joint_left"]["state"], frame.vtla["joint_right"]["state"]])
        self.samples.append({"time": frame.time, "state": state, "action": action, "streams": frame.vtla})
        self.phase.append(int(phase))
        self.intervention.append(int(bool(intervention)))

    def close(self):
        pass

    def save(self, destination, attrs=None, overwrite=False, tactile_config=None):
        from record_episodes_aloha2_vtla import export_episode, tactile_config_extras

        if len(self.samples) < 2:
            raise ValueError("episode too short")
        hil_values = np.stack([self.phase, self.intervention], axis=1)
        extra_metadata = {"hil": json.loads(to_json(attrs or {}))}
        extra_files, tactile_metadata = tactile_config_extras(tactile_config)
        extra_metadata.update(tactile_metadata)
        metadata = export_episode(destination, self.samples, self.models, self.sources, self.task,
                                  overwrite=overwrite, calibration=self.calibration,
                                  action_space=self.action_space,
                                  extra_csvs={"hil": (("phase", "intervention"), hil_values)},
                                  extra_metadata=extra_metadata, extra_files=extra_files)
        return metadata["total_frames"]
