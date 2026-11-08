#!/usr/bin/env python3
"""Cut hil_rollout.py episodes into per-segment episodes that the training converters read as is.

    python3 aloha_scripts/hil_split_episodes.py <hil_dir> <out_dir> [--keep intervention] [--min_frames 25]

--keep intervention  (default) each human correction (intervention == 1, i.e. after the first
                     correction_skip_s) becomes one episode - the DAgger data to add to the demos
--keep autonomous    each stretch the policy drove (phase == 0)
--keep active        policy and labelled-human stretches, cut wherever the robot was paused

Both raw formats are handled, detected per episode:
  episode_N.hdf5  every dataset whose first axis is time is sliced (compress_len tables along their
                  time axis), attrs are copied, hil_source records where the segment came from.
                  Output: <out_dir>/episode_K.hdf5 for openpi's convert_aloha_data_to_lerobot_*.py,
                  lerobot's convert_aloha_hdf5_to_lerobot_v30.py and ACT training.
  episode_N/      (VTLA) CSV rows are sliced (timestamps restarted at 0), each video is re-encoded
                  to exactly the segment's frames with record_episodes_aloha2_vtla.write_video,
                  metadata.json gets the new total_frames. Output: <out_dir>/episode_K/ for N0-VTLA's
                  convert_canonical_data.py. Segments shorter than --min_frames are dropped.
<out_dir>/segments.json lists every output episode's source file and frame range.
The unsplit episodes also convert as they are (the converters ignore /hil and hil/).
"""

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np


def segment_mask(phase, intervention, keep):
    phase = np.asarray(phase)
    intervention = np.asarray(intervention).astype(bool)
    if keep == "intervention":
        return intervention
    if keep == "autonomous":
        return phase == 0
    if keep == "active":
        return (phase == 0) | intervention
    raise ValueError("unknown --keep {!r}".format(keep))


def runs(mask, min_frames):
    """[start, end) of every run of True at least min_frames long."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_frames]


def _episode_number(path):
    match = re.search(r"episode_(\d+)", path.name)
    return int(match.group(1)) if match else -1


# -- HDF5 ------------------------------------------------------------------------------------------

def hdf5_labels(path):
    import h5py

    with h5py.File(str(path), "r") as f:
        if "hil" not in f:
            raise ValueError("{} has no /hil group (not a hil_rollout.py episode)".format(path))
        return f["hil/phase"][()], f["hil/intervention"][()]


def split_hdf5(src, segments, destinations):
    import h5py

    with h5py.File(str(src), "r") as f:
        length = f["action"].shape[0]
        for (start, end), destination in zip(segments, destinations):
            with h5py.File(str(destination), "w") as g:
                for key, value in f.attrs.items():
                    g.attrs[key] = value
                g.attrs["hil_source"] = json.dumps({"file": str(src), "start": start, "end": end})

                def copy(name, obj):
                    if isinstance(obj, h5py.Group):
                        group = g.require_group(name)
                        for key, value in obj.attrs.items():
                            group.attrs[key] = value
                        return
                    if name.split("/")[-1] == "compress_len" and obj.ndim == 2 and obj.shape[1] == length:
                        data = obj[:, start:end]
                    elif obj.ndim >= 1 and obj.shape[0] == length:
                        data = obj[start:end]
                    else:
                        data = obj[()]
                    if name == "hil/time":
                        data = data - data[0]
                    chunks = obj.chunks if obj.chunks and obj.ndim >= 1 and len(data) else None
                    if chunks is not None:
                        chunks = tuple(min(c, max(s, 1)) for c, s in zip(chunks, data.shape))
                    dataset = g.create_dataset(name, data=data, dtype=obj.dtype, chunks=chunks)
                    for key, value in obj.attrs.items():
                        dataset.attrs[key] = value

                f.visititems(copy)


# -- VTLA ------------------------------------------------------------------------------------------

def vtla_labels(path):
    rows = list(csv.DictReader((path / "hil" / "data.csv").open()))
    if not rows:
        raise ValueError("{} has an empty hil/data.csv".format(path))
    return (np.array([int(float(r["phase"])) for r in rows]),
            np.array([int(float(r["intervention"])) for r in rows]))


def _read_video_frames(path, start, end):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        for index in range(end):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("{} ended at frame {} (wanted {})".format(path, index, end))
            if index >= start:
                yield frame
    finally:
        capture.release()


def split_vtla(src, segments, destinations):
    from record_episodes_aloha2_vtla import write_video

    metadata = json.loads((src / "metadata.json").read_text(encoding="utf-8"))
    total = int(metadata["total_frames"])
    csv_files = [p for p in src.rglob("data.csv")] + [src / "timestamps.csv"]
    for (start, end), destination in zip(segments, destinations):
        if destination.exists():
            raise FileExistsError(destination)
        temporary = destination.parent / ("." + destination.name + ".partial")
        if temporary.exists():
            shutil.rmtree(str(temporary))
        temporary.mkdir(parents=True)
        for path in csv_files:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            header, body = rows[0], rows[1:]
            if len(body) != total:
                raise ValueError("{} has {} rows, metadata says {}".format(path, len(body), total))
            body = body[start:end]
            if header and header[0] == "timestamp":
                t0 = float(body[0][0])
                body = [["{:.10g}".format(float(row[0]) - t0)] + row[1:] for row in body]
            target = temporary / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(body)
        for video in src.rglob("video.mp4"):
            target = temporary / video.relative_to(src)
            write_video(target, _read_video_frames(video, start, end), end - start)
        for array in src.rglob("data.npy"):
            target = temporary / array.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(target), np.load(str(array))[start:end], allow_pickle=False)
        # Per-episode, not per-frame: every segment keeps the tactile runtime configs as they are.
        if (src / "tactile_runtime_config").is_dir():
            shutil.copytree(str(src / "tactile_runtime_config"), str(temporary / "tactile_runtime_config"))
        segment_metadata = dict(metadata, episode_id=destination.name, total_frames=end - start,
                                hil_source={"episode": str(src), "start": start, "end": end})
        (temporary / "metadata.json").write_text(
            json.dumps(segment_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.rename(destination)


# -- main ------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hil_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--keep", choices=("intervention", "autonomous", "active"), default="intervention")
    parser.add_argument("--min_frames", type=int, default=25,
                        help="Drop shorter segments (frames: 50 Hz for hdf5, 30 Hz for VTLA)")
    args = parser.parse_args(argv)

    sources = sorted((p for p in args.hil_dir.glob("episode_*")
                      if p.suffix == ".hdf5" or (p.is_dir() and (p / "metadata.json").exists())),
                     key=_episode_number)
    if not sources:
        parser.error("no episode_* in {}".format(args.hil_dir))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.glob("episode_*")):
        parser.error("{} already has episodes; use an empty output directory".format(args.out_dir))

    index = 0
    listing = []
    for src in sources:
        is_hdf5 = src.suffix == ".hdf5"
        phase, intervention = hdf5_labels(src) if is_hdf5 else vtla_labels(src)
        segments = runs(segment_mask(phase, intervention, args.keep), args.min_frames)
        destinations = [args.out_dir / ("episode_{}{}".format(index + k, ".hdf5" if is_hdf5 else ""))
                        for k in range(len(segments))]
        (split_hdf5 if is_hdf5 else split_vtla)(src, segments, destinations)
        for (start, end), destination in zip(segments, destinations):
            listing.append({"episode": destination.name, "source": src.name, "start": start, "end": end})
        print("{}: {} segment(s) {}".format(src.name, len(segments), segments))
        index += len(segments)
    (args.out_dir / "segments.json").write_text(json.dumps(
        {"keep": args.keep, "min_frames": args.min_frames, "segments": listing}, indent=2) + "\n")
    print("Wrote {} episode(s) to {}".format(index, args.out_dir))


if __name__ == "__main__":
    main()
