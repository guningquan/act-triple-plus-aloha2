"""hil_compare_lengths.py: frame counts of the base data and of the part of the HIL data that would be merged."""

import json

import h5py
import numpy as np
import pytest

import hil_compare_lengths
import hil_split_episodes


def write_episode(path, phase=None, intervention=None, frames=None):
    """An episode with /action and, when phase is given, the /hil labels hil_rollout.py writes."""
    frames = len(phase) if phase is not None else frames
    with h5py.File(str(path), "w") as f:
        f.create_dataset("action", data=np.zeros((frames, 14), dtype=np.float32))
        if phase is not None:
            f.create_dataset("hil/phase", data=np.asarray(phase, dtype=np.int8))
            f.create_dataset("hil/intervention", data=np.asarray(intervention, dtype=np.int8))


def hil_labels(autonomous, paused, skipped, human):
    """phase / intervention for: policy, pause, the first frames of a correction, the human actually in control."""
    phase = [0] * autonomous + [1] * paused + [2] * (skipped + human)
    intervention = [0] * (autonomous + paused + skipped) + [1] * human
    return phase, intervention


@pytest.fixture
def datasets(tmp_path):
    base, new = tmp_path / "demos", tmp_path / "hil"
    base.mkdir()
    new.mkdir()
    for index, frames in enumerate((400, 600)):
        write_episode(base / "episode_{}.hdf5".format(index), frames=frames)
    write_episode(new / "episode_0.hdf5", *hil_labels(50, 10, 10, 60))   # 130 frames, 60 human
    write_episode(new / "episode_1.hdf5", *hil_labels(80, 20, 0, 0))     # never corrected
    write_episode(new / "episode_2.hdf5", *hil_labels(20, 5, 5, 8))      # a 8-frame correction: too short
    return base, new


def test_counts_the_base_frames_and_the_human_frames_of_the_new_data(datasets):
    base_dir, new_dir = datasets
    base = hil_compare_lengths.measure_base(base_dir, 50)
    new = hil_compare_lengths.measure_new(new_dir, "intervention", 25, 50)
    assert (base["kind"], base["episodes"], base["frames"], base["lengths"]) == ("hdf5", 2, 1000, [400, 600])
    assert (new["episodes"], new["frames"]) == (3, 130 + 100 + 38)
    assert (new["kept"], new["segments"], new["with_segments"], new["dropped"]) == (60, 1, 1, 8)
    assert new["rows"][0]["phases"] == {"autonomous": 50, "paused": 10, "correcting_skip": 10, "intervention": 60}


def test_the_count_equals_what_hil_split_episodes_writes(datasets, tmp_path):
    _, new_dir = datasets
    out_dir = tmp_path / "corrections"
    hil_split_episodes.main([str(new_dir), str(out_dir)])
    written = sum(hil_compare_lengths.episode_frames(p) for p in out_dir.glob("episode_*.hdf5"))
    assert written == hil_compare_lengths.measure_new(new_dir, "intervention", 25, 50)["kept"]
    for part in ("autonomous", "active"):
        out = tmp_path / part
        hil_split_episodes.main([str(new_dir), str(out), "--keep", part, "--min_frames", "5"])
        written = sum(hil_compare_lengths.episode_frames(p) for p in out.glob("episode_*.hdf5"))
        assert written == hil_compare_lengths.measure_new(new_dir, part, 5, 50)["kept"]


def test_min_frames_decides_which_corrections_count(datasets):
    _, new_dir = datasets
    assert hil_compare_lengths.measure_new(new_dir, "intervention", 8, 50)["kept"] == 68
    assert hil_compare_lengths.measure_new(new_dir, "intervention", 61, 50)["kept"] == 0


def test_part_all_counts_whole_episodes_and_needs_no_hil_labels(datasets, tmp_path):
    _, new_dir = datasets
    assert hil_compare_lengths.measure_new(new_dir, "all", 25, 50)["kept"] == 268
    plain = tmp_path / "plain"
    plain.mkdir()
    write_episode(plain / "episode_0.hdf5", frames=300)
    new = hil_compare_lengths.measure_new(plain, "all", 25, 50)
    assert (new["kept"], new["rows"][0]["phases"]) == (300, None)
    with pytest.raises(SystemExit, match="no /hil group"):
        hil_compare_lengths.measure_new(plain, "intervention", 25, 50)


def test_a_converted_lerobot_dataset_is_read_from_info_json(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"total_episodes": 50, "total_frames": 91616, "fps": 50}))
    assert hil_compare_lengths.measure_base(tmp_path, 25) == {
        "kind": "lerobot", "episodes": 50, "frames": 91616, "fps": 50.0, "lengths": None}


def test_copies_needed_for_a_ratio_of_new_to_base():
    # 1000 base frames, 50 new: 10 % of the base is 100 frames = 2 copies; 20 % = 4 copies
    k, ratio, share = hil_compare_lengths.copies_for(0.10, 1000, 50)
    assert k == 2 and ratio == pytest.approx(0.10) and share == pytest.approx(100 / 1100)
    k, ratio, share = hil_compare_lengths.copies_for(0.20, 1000, 70)  # 2.86 -> 3 copies = 21 %
    assert k == 3 and ratio == pytest.approx(0.21) and share == pytest.approx(210 / 1210)
    assert hil_compare_lengths.copies_for(0.01, 1000, 500)[0] == 1  # never fewer than one copy


def test_report_lists_the_ratios_the_frames_still_missing_and_the_act_rule(datasets, capsys):
    base_dir, new_dir = datasets
    hil_compare_lengths.main([str(base_dir), str(new_dir), "--ratio", "0.1", "0.5", "--per_episode"])
    out = capsys.readouterr().out
    assert "2 episodes, 1,000 frames" in out and "= 60 frames" in out
    assert "new / base = 6.0%" in out and "share of new in base + new = 5.7%" in out
    # 10 %: 100 frames = 2 copies of the 60, or 40 more frames (1 more correction of 60)
    assert "new/base  10%  ->  k = 2  copies (new/base 12.0%, share of merged 10.7%, 120 frames)" in out
    assert "or record 40 more frames (~1 more corrections of 60)" in out
    assert "new/base  50%  ->  k = 8  copies" in out and "many repeats of few frames" in out
    assert "episode_1.hdf5" in out and "0 segment(s)" in out
    assert "sample_weights" in out and "50/50" in out


def test_report_says_when_the_new_part_is_already_big_enough(datasets, capsys):
    base_dir, new_dir = datasets
    hil_compare_lengths.main([str(base_dir), str(new_dir), "--ratio", "0.05"])
    assert "already there" in capsys.readouterr().out


def test_report_says_so_when_nothing_would_be_merged(tmp_path, capsys):
    base, new = tmp_path / "base", tmp_path / "new"
    base.mkdir()
    new.mkdir()
    write_episode(base / "episode_0.hdf5", frames=100)
    write_episode(new / "episode_0.hdf5", *hil_labels(50, 10, 0, 0))
    hil_compare_lengths.main([str(base), str(new)])
    assert "nothing to merge" in capsys.readouterr().out


def test_non_positive_ratios_are_rejected(datasets):
    base_dir, new_dir = datasets
    with pytest.raises(SystemExit):
        hil_compare_lengths.main([str(base_dir), str(new_dir), "--ratio", "0"])
