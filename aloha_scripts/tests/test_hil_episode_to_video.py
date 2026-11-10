"""hil_episode_to_video.py: the video must show the colours of the scene, not the stored channel order."""

import sys

import cv2
import h5py
import numpy as np

import hil_episode_to_video

RGB_RED = (255, 0, 0)  # what the recorder stores for a red object: the camera's rgb8 bytes
H, W = 64, 96


def jpeg(rgb_image):
    """Exactly what hil_recorder / record_episodes_aloha2 do: cv2.imencode on the rgb8 array, no swap."""
    ok, encoded = cv2.imencode(".jpg", rgb_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return encoded.reshape(-1)


def write_episode(path, frames=3):
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[:, :] = RGB_RED
    encoded = [jpeg(image) for _ in range(frames)]
    width = max(len(e) for e in encoded)
    padded = np.zeros((frames, width), dtype=np.uint8)
    for i, e in enumerate(encoded):
        padded[i, :len(e)] = e
    with h5py.File(str(path), "w") as f:
        f.create_dataset("observations/images/cam_high", data=padded)
        f.create_dataset("hil/phase", data=np.zeros(frames, dtype=np.uint8))
        f.create_dataset("hil/intervention", data=np.zeros(frames, dtype=np.uint8))
        f.create_dataset("hil/time", data=np.arange(frames) / 50.0)


def test_decode_cam_returns_bgr_for_recorder_data(tmp_path):
    path = tmp_path / "episode_0.hdf5"
    write_episode(path)
    with h5py.File(str(path), "r") as f:
        frame = hil_episode_to_video.decode_cam(f["observations/images/cam_high"])[0]
    b, g, r = frame[H // 2, W // 2].astype(int)  # cv2's own channel order
    assert r > 200 and b < 60 and g < 60, "red object decoded as BGR=({}, {}, {})".format(b, g, r)


def test_rendered_video_shows_a_red_object_as_red(tmp_path, monkeypatch):
    path, out = tmp_path / "episode_0.hdf5", tmp_path / "out.mp4"
    write_episode(path)
    monkeypatch.setattr(sys, "argv", ["hil_episode_to_video", str(path), str(out), "--fps", "10"])
    hil_episode_to_video.main()

    capture = cv2.VideoCapture(str(out))
    ok, frame = capture.read()  # BGR, as any player would show it after decoding
    capture.release()
    assert ok
    b, g, r = frame[H // 2, W // 2].astype(int)  # the scene colour, away from the border and banner
    assert r > 180 and b < 80 and g < 80, "red object rendered as BGR=({}, {}, {})".format(b, g, r)
