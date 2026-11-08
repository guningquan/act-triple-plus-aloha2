#!/usr/bin/env python3
"""Check the timing of recorded episodes: were the cameras, joints and tactile data stored for each
step really from about the same moment, and was the tactile data fresh?

    python3 aloha_scripts/check_episode_timing.py <episode.hdf5 | episode_dir | dataset_dir> [...]

HDF5 episodes (record_episodes_aloha2.py, hil_rollout.py --format hdf5) are read from
/observations/timestamps (written since 2026-11, see tactile_hdf5.py); VTLA episode directories
(record_episodes_aloha2_vtla.py, hil_rollout.py --format vtla) from timestamps.csv. Per episode it
prints p50 / p95 / max of

    control period     time between consecutive stored steps (VTLA: between the recorder's own
                       samples, not the 30 Hz rows resampled from them)
    camera age         observation time - the camera message's header stamp (capture)
    joint age          observation time - the joint_states header stamp
    tactile age        observation time - the publisher's local_read_end (the data's own age)
    tactile new        share of steps that got a new tactile frame (not the previous step's)
    tactile skew       spread of the sensors' read_end within one step

and judges each against the thresholds below (PASS / WARN / FAIL). Exit status 1 if anything FAILs.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# (warn above, fail above) in seconds; "tactile new" is a share (warn below, fail below).
THRESHOLDS = {
    "control period": (0.030, 0.050),
    "camera age": (0.100, 0.250),
    "joint age": (0.050, 0.200),
    "tactile age": (0.100, 0.200),
    "tactile skew": (0.050, 0.100),
    "tactile new": (0.90, 0.50),
}


def summarize(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {"n": int(values.size), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def judge(name, stats):
    if stats is None:
        return "n/a"
    warn, fail = THRESHOLDS[name]
    if name == "tactile new":
        value = stats["p50"]  # the share itself
        return "FAIL" if value < fail else "WARN" if value < warn else "PASS"
    return "FAIL" if stats["max"] > fail else "WARN" if stats["p95"] > warn else "PASS"


def metrics_hdf5(path):
    import h5py

    with h5py.File(str(path), "r") as f:
        if "/observations/timestamps" not in f:
            raise ValueError("no /observations/timestamps (recorded before timestamps were saved)")
        ts = f["/observations/timestamps"]
        local = ts["local_time"][()]
        out = {}
        control = ts["control_start"][()] if "control_start" in ts else local
        out["control period"] = summarize(np.diff(control))
        if "images" in ts:
            out["camera age"] = summarize(np.concatenate([local - ts["images"][c][()] for c in ts["images"]]))
        if "joints" in ts:
            out["joint age"] = summarize(np.concatenate([local - ts["joints"][s][()] for s in ts["joints"]]))
        if "tactile" in ts:
            sensors = list(ts["tactile"])
            out["tactile age"] = summarize(np.concatenate([ts["tactile"][s]["frame_age"][()] for s in sensors]))
            new = np.concatenate([ts["tactile"][s]["new_frame"][()][1:] for s in sensors])
            out["tactile new"] = summarize([new.mean()]) if new.size else None
            if len(sensors) > 1:
                ends = np.stack([ts["tactile"][s]["read_end"][()] for s in sensors])
                out["tactile skew"] = summarize(ends.max(axis=0) - ends.min(axis=0))
        steps = len(local)
    return steps, out


def metrics_vtla(directory):
    with (directory / "timestamps.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty timestamps.csv")
    columns = rows[0].keys()

    def column(name):
        return np.array([float(row[name]) if row[name] not in ("", "nan") else np.nan for row in rows])

    # Rows are a 30 Hz zero-order hold of the recorder's ~50 Hz samples, so consecutive rows are 20 or
    # 40 ms apart even when the loop is perfect: the loop's period is the time between the held
    # samples divided by how many samples apart they are (rows holding the same sample are skipped).
    apart, gap = np.diff(column("source_index")), np.diff(column("sample_elapsed_s"))
    moved = apart > 0
    out = {"control period": summarize(gap[moved] / apart[moved])}
    cams = [c[:-len("_receipt_age_s")] for c in columns if c.startswith("cam_") and c.endswith("_receipt_age_s")]
    joints = [c[:-len("_receipt_age_s")] for c in columns if c.startswith("joint_") and c.endswith("_receipt_age_s")]
    tactile = [c[:-len("_data_age_s")] for c in columns if c.endswith("_data_age_s")]
    if cams:  # receipt age: the VTLA recorder keeps no camera capture stamp relative to the sample
        out["camera age"] = summarize(np.concatenate([column(c + "_receipt_age_s") for c in cams]))
    if joints:
        out["joint age"] = summarize(np.concatenate([column(j + "_receipt_age_s") for j in joints]))
    if tactile:
        out["tactile age"] = summarize(np.concatenate([column(t + "_data_age_s") for t in tactile]))
        # 30 Hz resampled rows from a ~50 Hz source: a repeated counter means the same message twice.
        reused = np.concatenate([np.diff(column(t + "_sequence")) == 0 for t in tactile])
        out["tactile new"] = summarize([1.0 - reused.mean()]) if reused.size else None
        stamps = np.stack([column(t + "_ros_stamp_s") for t in tactile])
        if len(tactile) > 1 and np.all(stamps > 0):
            out["tactile skew"] = summarize(stamps.max(axis=0) - stamps.min(axis=0))
    return len(rows), out


def find_episodes(paths):
    found = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix == ".hdf5":
            found.append(path)
        elif path.is_dir() and (path / "timestamps.csv").exists():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(path.glob("episode_*.hdf5"), key=lambda p: (len(p.stem), p.stem)))
            found.extend(sorted((p for p in path.glob("episode_*") if (p / "timestamps.csv").exists()),
                                key=lambda p: (len(p.name), p.name)))
        else:
            print("skipping {}: not an episode file/directory".format(path))
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    episodes = find_episodes(args.paths)
    if not episodes:
        print("No episodes found.")
        return 1
    worst = "PASS"
    order = {"n/a": 0, "PASS": 0, "WARN": 1, "FAIL": 2}
    for episode in episodes:
        try:
            steps, metrics = metrics_hdf5(episode) if episode.suffix == ".hdf5" else metrics_vtla(episode)
        except Exception as exc:  # noqa: BLE001
            print("{}: cannot check ({})".format(episode, exc))
            worst = "FAIL"
            continue
        print("{} ({} steps)".format(episode, steps))
        for name in THRESHOLDS:
            if name not in metrics:
                continue
            stats = metrics[name]
            verdict = judge(name, stats)
            if order[verdict] > order[worst]:
                worst = verdict
            if stats is None:
                print("  {:15s} n/a".format(name))
            elif name == "tactile new":
                print("  {:15s} {:6.1%} of steps                         {}".format(name, stats["p50"], verdict))
            else:
                print("  {:15s} p50 {:7.1f}  p95 {:7.1f}  max {:7.1f} ms   {}".format(
                    name, stats["p50"] * 1000, stats["p95"] * 1000, stats["max"] * 1000, verdict))
    print("Overall: {}".format(worst))
    return 1 if worst == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
