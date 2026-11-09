#!/usr/bin/env python
"""Add HIL corrections recorded with `lerobot-rollout --strategy.type=dagger` to a LeRobot v3.0 dataset.

lerobot-rollout writes its own dataset (rollout_*, one episode per correction); it never touches the
dataset you train on. This script writes a NEW dataset = base episodes + the HIL episodes, ready for
lerobot-train. The inputs are left unchanged.

Runs in lerobot's Python 3.12 environment (it imports lerobot), not in the `aloha` conda env:

    source /mnt/server_ssd2/Programs_server/lerobot/.venv/bin/activate
    python aloha_scripts/merge_hil_into_lerobot_v30.py \
        --base /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_v30 \
        --hil /home/robot/Dataset_and_Checkpoint/lerobot-dataset/rollout_packaging_hil \
        --output-dir /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_v30_hil

--base  usually the output of convert_aloha_hdf5_to_lerobot_v30.py (or an earlier merge)
--hil   one or more dagger datasets; listing one twice adds its episodes twice (more weight)

The two kinds of dataset differ, so they are aligned before merging (LeRobot's merge needs identical
features and robot_type):
  - features only the HIL datasets have are dropped: `intervention` (all True in corrections-only
    mode, so it says nothing once the episodes are known to be HIL - see source_episodes.tsv) and
    cameras the base does not have (cam_low)
  - features only the base has are dropped too, as long as they are not cameras, state or action:
    observation.velocity / observation.effort (dagger does not record them; pi05, MolmoAct2 and ACT
    only read observation.state)
  - robot_type is set to the base's
fps, the state/action joint names and the image sizes must already match. The HIL data must have been
recorded with --robot.gripper_state_format=aloha2 when the base comes from ALOHA2 hdf5 (the gripper
state then means the same thing); this cannot be checked from the files.

Videos are concatenated, not re-encoded. meta/stats.json gets exact quantiles of the merged data (as
in convert_aloha_hdf5_to_lerobot_v30.py), and meta/source_episodes.tsv says where every episode
came from.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from convert_aloha_hdf5_to_lerobot_v30 import write_exact_quantiles
from lerobot.datasets.dataset_tools import merge_datasets, remove_feature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION, DEFAULT_FEATURES, OBS_STATE


def load_meta(root: Path) -> LeRobotDatasetMetadata:
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"{root} is not a LeRobot dataset (no meta/info.json)")
    return LeRobotDatasetMetadata(f"local/{root.name}", root=root)


def check_compatible(base: LeRobotDatasetMetadata, hil: LeRobotDatasetMetadata) -> tuple[list[str], list[str]]:
    """Raise if `hil` cannot join `base`; return (features to drop from base, features to drop from hil)."""
    if hil.fps != base.fps:
        raise ValueError(f"{hil.root}: fps {hil.fps}, base has {base.fps}")
    for key in (OBS_STATE, ACTION):
        if key not in hil.features:
            raise ValueError(f"{hil.root} has no {key}")
        if hil.features[key]["names"] != base.features[key]["names"]:
            raise ValueError(
                f"{hil.root}: {key} names {hil.features[key]['names']} differ from the base's "
                f"{base.features[key]['names']}"
            )
    for key in base.camera_keys:
        if key not in hil.features:
            raise ValueError(f"{hil.root} has no {key}, which the base dataset has")
        if list(hil.features[key]["shape"]) != list(base.features[key]["shape"]):
            raise ValueError(f"{hil.root}: {key} shape {hil.features[key]['shape']}, base has {base.features[key]['shape']}")

    drop_hil = sorted(set(hil.features) - set(base.features))
    drop_base = sorted(set(base.features) - set(hil.features))
    blocked = [key for key in drop_base if key in base.camera_keys or key in (OBS_STATE, ACTION)]
    if blocked:
        raise ValueError(f"{hil.root} lacks {blocked}, which cannot be dropped from the base")
    return drop_base, drop_hil


def aligned_copy(root: Path, drop: list[str], robot_type: str, target: Path) -> Path:
    """`root` without the `drop` features and with `robot_type`, as a new dataset at `target`
    (or `root` itself when nothing changes)."""
    meta = load_meta(root)
    if not drop and meta.robot_type == robot_type:
        return root
    if drop:
        dataset = LeRobotDataset(f"local/{root.name}", root=root)
        remove_feature(dataset, drop, output_dir=target, repo_id=f"local/{target.name}")
    else:
        shutil.copytree(root, target)
    info_path = target / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["robot_type"] = robot_type
    info_path.write_text(json.dumps(info, indent=4) + "\n")
    return target


def read_numeric(root: Path, keys: list[str]) -> dict[str, list[np.ndarray]]:
    """Every value of the given 1-D numeric features, read from the dataset's parquet files."""
    values: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for path in sorted((root / "data").rglob("*.parquet")):
        frame = pd.read_parquet(path, columns=keys)
        for key in keys:
            values[key].append(np.stack(frame[key].to_numpy()).astype(np.float64))
    return values


def merge(args: argparse.Namespace) -> None:
    base_root = Path(args.base)
    hil_roots = [Path(p) for p in args.hil]
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)

    base = load_meta(base_root)
    hils = [load_meta(root) for root in hil_roots]
    drop_base, drop_hil = set(), {}
    for root, hil in zip(hil_roots, hils, strict=True):
        from_base, from_hil = check_compatible(base, hil)
        drop_base |= set(from_base)
        drop_hil[root] = from_hil
    logging.info(f"Base {base_root}: {base.total_episodes} episodes, {base.total_frames} frames")
    for root, hil in zip(hil_roots, hils, strict=True):
        logging.info(f"HIL  {root}: {hil.total_episodes} episodes, {hil.total_frames} frames, dropping {drop_hil[root]}")
    if drop_base:
        logging.info(f"Dropping from the base (not recorded by dagger): {sorted(drop_base)}")

    work = output_dir.parent / f".{output_dir.name}.merge_tmp"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        roots = [aligned_copy(base_root, sorted(drop_base), base.robot_type, work / "base")]
        for i, root in enumerate(hil_roots):
            # drop_base features are absent from the HIL data already
            roots.append(aligned_copy(root, drop_hil[root], base.robot_type, work / f"hil_{i}"))
        datasets = [LeRobotDataset(f"local/{root.name}", root=root) for root in roots]
        merge_datasets(datasets, output_repo_id=args.repo_id or f"local/{output_dir.name}", output_dir=output_dir)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    merged = load_meta(output_dir)
    numeric_keys = [
        key
        for key, feature in merged.features.items()
        if key not in DEFAULT_FEATURES and feature["dtype"].startswith("float") and len(feature["shape"]) == 1
    ]
    write_exact_quantiles(output_dir, read_numeric(output_dir, numeric_keys))

    rows, index = [], 0
    for kind, root, meta in [("base", base_root, base)] + [("hil", r, m) for r, m in zip(hil_roots, hils, strict=True)]:
        for source_index, length in zip(meta.episodes["episode_index"], meta.episodes["length"], strict=True):
            rows.append(f"{index}\t{kind}\t{root}\t{source_index}\t{length}")
            index += 1
    if index != merged.total_episodes:
        raise RuntimeError(f"Merged dataset has {merged.total_episodes} episodes, expected {index}")
    (output_dir / "meta" / "source_episodes.tsv").write_text(
        "episode_index\tkind\tsource_dataset\tsource_episode_index\tnum_frames\n" + "\n".join(rows) + "\n"
    )
    hil_frames = sum(m.total_frames for m in hils)
    logging.info(
        f"Done: {merged.total_episodes} episodes, {merged.total_frames} frames "
        f"(HIL {hil_frames} frames = {100 * hil_frames / merged.total_frames:.1f}%) -> {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="LeRobot v3.0 dataset to add the HIL episodes to")
    parser.add_argument("--hil", nargs="+", required=True, help="Dataset(s) recorded by lerobot-rollout dagger")
    parser.add_argument("--output-dir", required=True, help="Merged dataset root to create (must not exist)")
    parser.add_argument("--repo-id", default=None, help="Dataset name in metadata (default: local/<output dir name>)")
    parser.add_argument("--overwrite", action="store_true", help="Delete --output-dir first if present")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    merge(args)


if __name__ == "__main__":
    main()
