#!/usr/bin/env python3
"""Render a hil_rollout.py episode_N.hdf5 as an annotated side-by-side video.

    python3 aloha_scripts/hil_episode_to_video.py <episode.hdf5> <out.mp4> [--fps FPS]

Every camera in observations/images is decoded and tiled side by side, with a colored border
and a text banner (frame index, timestamp, label) burned into each frame so the video can be
watched to see which stretches were autonomous, paused, or human-corrected:

  green  AUTONOMOUS  policy driving (hil/phase == 0)
  amber  PAUSED      policy stopped, waiting for a decision (hil/phase == 1)
  red    HUMAN        human actually in control (hil/intervention == 1 - a subset of
                      hil/phase == 2 CORRECTING, which also covers the first
                      --correction_skip_s of each takeover that is not counted as intervention)

Also prints the episode's attrs, a frame-count breakdown per phase/intervention, and the list of
contiguous label segments with their start/end time - the same numbers burned into the video, in
one place to read without scrubbing through it.

Episodes with no hil/ group (plain record_episodes_aloha2.py recordings) are rendered the same
way but with every frame labeled AUTONOMOUS, since there is nothing else to show.
"""

import argparse
import sys

import cv2
import h5py
import numpy as np

PHASE_NAMES = {0: "AUTONOMOUS", 1: "PAUSED", 2: "CORRECTING"}
PHASE_COLOR = {0: (80, 200, 80), 1: (0, 180, 230), 2: (40, 40, 230)}  # BGR: green / amber / red
HUMAN_COLOR = (0, 0, 220)  # BGR: red


def decode_cam(dataset):
    return [cv2.imdecode(row, cv2.IMREAD_COLOR) for row in dataset[()]]


def label_of(phase, intervention, i):
    if intervention[i]:
        return "HUMAN"
    return PHASE_NAMES[int(phase[i])]


def print_report(attrs, phase, intervention, t):
    n = len(phase)
    print("=== episode attrs ===")
    for k, v in attrs.items():
        print(f"  {k}: {v}")

    print("\n=== per-frame label (phase/intervention) summary ===")
    print(f"total frames: {n}, duration: {t[-1] - t[0]:.2f}s")
    for p in (0, 1, 2):
        cnt = int((phase == p).sum())
        print(f"  phase={p} ({PHASE_NAMES[p]:11s}): {cnt:5d} frames ({100 * cnt / n:5.1f}%)")
    hum = int(intervention.sum())
    print(f"  intervention=1 (human actually in control): {hum:5d} frames ({100 * hum / n:5.1f}%)")

    print("\n=== segments (contiguous label runs) ===")
    seg_start = 0
    cur = label_of(phase, intervention, 0)
    segments = []
    for i in range(1, n):
        lab = label_of(phase, intervention, i)
        if lab != cur:
            segments.append((seg_start, i - 1, cur))
            seg_start, cur = i, lab
    segments.append((seg_start, n - 1, cur))
    for s, e, lab in segments:
        print(f"  frames {s:5d}-{e:5d}  t={t[s]:6.2f}s-{t[e]:6.2f}s  dur={t[e] - t[s]:5.2f}s  {lab}")
    print(f"\ntotal segments: {len(segments)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", help="path to episode_N.hdf5")
    parser.add_argument("out_video", help="output .mp4 path")
    parser.add_argument("--fps", type=float, default=50.0,
                         help="output video frame rate (default: 50, the recorder's rate)")
    args = parser.parse_args()

    with h5py.File(args.episode, "r") as f:
        cams = sorted(f["observations/images"].keys())
        print(f"cameras: {cams}")
        frames = {cam: decode_cam(f[f"observations/images/{cam}"]) for cam in cams}
        n = len(frames[cams[0]])
        if "hil" in f:
            phase = f["hil/phase"][:]
            intervention = f["hil/intervention"][:]
            t = f["hil/time"][:]
        else:
            print("no hil/ group in this episode - labeling every frame AUTONOMOUS")
            phase = np.zeros(n, dtype=np.uint8)
            intervention = np.zeros(n, dtype=np.uint8)
            t = np.arange(n, dtype=np.float64) / args.fps
        attrs = dict(f.attrs)

    print_report(attrs, phase, intervention, t)

    h, w = frames[cams[0]][0].shape[:2]
    border = 10
    panel_h = h + 40
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_video, fourcc, args.fps, (w * len(cams), panel_h))
    for i in range(n):
        row = []
        for cam in cams:
            img = frames[cam][i].copy()
            color = HUMAN_COLOR if intervention[i] else PHASE_COLOR[int(phase[i])]
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, border)
            row.append(img)
        strip = np.hstack(row)
        banner = np.full((40, strip.shape[1], 3), 24, dtype=np.uint8)
        lab = "HUMAN" if intervention[i] else PHASE_NAMES[int(phase[i])]
        color = HUMAN_COLOR if intervention[i] else PHASE_COLOR[int(phase[i])]
        text = f"frame {i:5d}/{n}  t={t[i]:6.2f}s  {lab}"
        cv2.putText(banner, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        panel = np.vstack([strip, banner])
        writer.write(panel)
    writer.release()
    print(f"\nwrote video: {args.out_video}")


if __name__ == "__main__":
    sys.exit(main())
