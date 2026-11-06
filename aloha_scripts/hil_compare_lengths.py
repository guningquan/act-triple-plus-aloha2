#!/usr/bin/env python3
"""How long are the two datasets you are about to merge, and what share would the new part get?

    python3 aloha_scripts/hil_compare_lengths.py <base> <new> [--part intervention] [--min_frames 25]
                                                 [--ratio 0.1 0.2] [--per_episode]

<base>  the data you already train on: a directory of episode_*.hdf5, or a converted LeRobot dataset
        (a directory with meta/info.json - only its total_frames is read)
<new>   a hil_rollout.py hdf5 directory (episode_*.hdf5 with /hil). By default only its human-correction part is
        counted: exactly the frames hil_split_episodes.py --keep intervention --min_frames N would write.
        --part autonomous | active use the other hil_split_episodes.py cuts, --part all counts whole episodes.

Frames are counted on /action (what the converters check against), at 50 Hz for hdf5 (--fps).
openpi has no per-dataset weights, so for it the share of the new part in the merged data is simply its share of
frames, and the way to raise it is to convert k copies of the new episodes (see --ratio), or to record more.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from hil_split_episodes import _episode_number, hdf5_labels, runs, segment_mask

PARTS = ("intervention", "autonomous", "active", "all")
MANY_COPIES = 5  # beyond this the same few frames are repeated a lot per epoch


def hdf5_files(path):
    files = sorted(path.glob("episode_*.hdf5"), key=_episode_number)
    return files or sorted(path.rglob("*.hdf5"))


def episode_frames(path):
    import h5py

    with h5py.File(str(path), "r") as f:
        return int(f["action"].shape[0])


def measure_base(path, fps):
    info = path / "meta" / "info.json"
    if info.exists():  # converted LeRobot dataset
        data = json.loads(info.read_text())
        return {"kind": "lerobot", "episodes": int(data["total_episodes"]), "frames": int(data["total_frames"]),
                "fps": float(data.get("fps") or fps), "lengths": None}
    files = hdf5_files(path)
    if not files:
        raise SystemExit("no episode_*.hdf5 and no meta/info.json in {}".format(path))
    lengths = [episode_frames(f) for f in files]
    return {"kind": "hdf5", "episodes": len(files), "frames": sum(lengths), "fps": float(fps), "lengths": lengths}


def measure_new(path, part, min_frames, fps):
    files = hdf5_files(path)
    if not files:
        raise SystemExit("no episode_*.hdf5 in {}".format(path))
    rows = []
    for f in files:
        total = episode_frames(f)
        try:
            phase, intervention = hdf5_labels(f)
            phase, intervention = np.asarray(phase), np.asarray(intervention).astype(bool)
        except ValueError:
            if part != "all":
                raise SystemExit("{} has no /hil group (not a hil_rollout.py episode); use --part all".format(f))
            phase = intervention = None
        row = {"name": f.name, "frames": total, "phases": None}
        if phase is not None:
            row["phases"] = {"autonomous": int((phase == 0).sum()), "paused": int((phase == 1).sum()),
                             "correcting_skip": int(((phase == 2) & ~intervention).sum()),
                             "intervention": int(intervention.sum())}
        if part == "all":
            row.update(segments=1, kept=total, dropped=0)
        else:
            mask = segment_mask(phase, intervention, part)
            segments = runs(mask, min_frames)
            kept = sum(end - start for start, end in segments)
            row.update(segments=len(segments), kept=kept, dropped=int(mask.sum()) - kept)
        rows.append(row)
    return {"episodes": len(rows), "frames": sum(r["frames"] for r in rows), "kept": sum(r["kept"] for r in rows),
            "segments": sum(r["segments"] for r in rows), "dropped": sum(r["dropped"] for r in rows),
            "with_segments": sum(1 for r in rows if r["segments"]), "fps": float(fps), "rows": rows}


def copies_for(ratio, base_frames, new_frames):
    """k copies of the new part so that k * new / base is closest to ratio.

    Returns (k, k * new / base, k * new / (base + k * new))."""
    k = max(1, int(round(ratio * base_frames / float(new_frames))))
    return k, k * new_frames / float(base_frames), k * new_frames / float(base_frames + k * new_frames)


def duration(frames, fps):
    seconds = frames / fps
    return "{:.1f} min".format(seconds / 60) if seconds >= 120 else "{:.1f} s".format(seconds)


def report(base, new, part, min_frames, ratios, base_path, new_path, per_episode=False):
    lines = []
    average = " (avg {:,.0f}/episode)".format(base["frames"] / base["episodes"]) if base["episodes"] else ""
    lines.append("base : {}  [{}]  {} episodes, {:,} frames = {} @{:g} Hz{}".format(
        base_path, base["kind"], base["episodes"], base["frames"], duration(base["frames"], base["fps"]),
        base["fps"], average))
    phases = [r["phases"] for r in new["rows"] if r["phases"]]
    lines.append("new  : {}  [hdf5]  {} episodes, {:,} frames = {}".format(
        new_path, new["episodes"], new["frames"], duration(new["frames"], new["fps"])))
    if phases:
        totals = {k: sum(p[k] for p in phases) for k in phases[0]}
        lines.append("       autonomous {autonomous:,} | paused {paused:,} | correcting-skip {correcting_skip:,} | "
                     "intervention {intervention:,}".format(**totals))
    if part == "all":
        lines.append("       counted: whole episodes = {:,} frames".format(new["kept"]))
    else:
        lines.append("       counted: --part {} (segments >= {} frames) = {} segment(s) in {} of {} episodes = "
                     "{:,} frames = {}".format(part, min_frames, new["segments"], new["with_segments"],
                                               new["episodes"], new["kept"], duration(new["kept"], new["fps"])))
        if new["dropped"]:
            lines.append("       not counted: {:,} frames in segments shorter than {} frames".format(
                new["dropped"], min_frames))
    if per_episode:
        for r in new["rows"]:
            lines.append("         {:<18} {:>6} frames -> {} segment(s), {:>6} counted".format(
                r["name"], r["frames"], r["segments"], r["kept"]))
    if base["lengths"] and per_episode:
        lines.append("       base episode length: min {:,} / mean {:,.0f} / max {:,}".format(
            min(base["lengths"]), np.mean(base["lengths"]), max(base["lengths"])))

    kept = new["kept"]
    if kept == 0:
        lines.append("\nnothing counted in the new data, nothing to merge.")
        return "\n".join(lines)
    merged = base["frames"] + kept
    lines.append("\nnew / base = {:.1%}    share of new in base + new = {:.1%}".format(kept / base["frames"], kept / merged))
    per_segment = kept / float(new["segments"]) if new["segments"] else kept
    lines.append("\nopenpi / LeRobot (frames are drawn uniformly, no weights). To reach new / base = ratio, either convert "
                 "k copies of the new episodes\nor record more:")
    for ratio in ratios:
        k, got, share = copies_for(ratio, base["frames"], kept)
        missing = ratio * base["frames"] - kept
        more = ("already there" if missing <= 0 else "or record {:,.0f} more frames (~{:.0f} more corrections of "
                                                     "{:,.0f})".format(missing, np.ceil(missing / per_segment), per_segment))
        note = "  <- many repeats of few frames" if k > MANY_COPIES else ""
        lines.append("  new/base {:>4.0%}  ->  k = {:<2} copies (new/base {:.1%}, share of merged {:.1%}, {:,} frames); {}{}".format(
            ratio, k, got, share, k * kept, more, note))
    lines.append("\nACT (imitate_episodes_multi_gpu.py, dataset_dir = [base, new]): the share is set by sample_weights, "
                 "not by length;\n  no sample_weights = 50/50. Use [1 - p, p], p = share of merged (new/base 10% is "
                 "p = 9%, 20% is p = 17%), e.g. [90, 10].\n  Each new frame is then seen about "
                 "steps * batch_size * p / {:,} times in training.".format(kept))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--part", choices=PARTS, default="intervention")
    parser.add_argument("--min_frames", type=int, default=25, help="same default as hil_split_episodes.py")
    parser.add_argument("--ratio", type=float, nargs="+", default=[0.10, 0.20],
                        help="wanted new frames / base frames, e.g. 0.1 = the new part is 10%% of the base data")
    parser.add_argument("--fps", type=float, default=50.0, help="hdf5 frame rate (hil_rollout.py records at 50 Hz)")
    parser.add_argument("--per_episode", action="store_true")
    args = parser.parse_args(argv)
    if not all(r > 0.0 for r in args.ratio):
        parser.error("--ratio values must be positive")

    base = measure_base(args.base, args.fps)
    new = measure_new(args.new, args.part, args.min_frames, args.fps)
    print(report(base, new, args.part, args.min_frames, args.ratio, args.base, args.new, args.per_episode))


if __name__ == "__main__":
    sys.exit(main())
