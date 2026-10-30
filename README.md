# ALOHA / ALOHA2

This repository provides teleoperation, dataset collection, and imitation learning workflows
for ALOHA / ALOHA2 robot systems, with **optional tactile sensing and leader-gripper force feedback**.

- **Robot hardware setup**: install and configure the ROS drivers and launch files for your ALOHA hardware in your ROS workspace. Match the arm names, camera topics, and gripper calibration to your setup.
- **Optional tactile support**: compatible tactile sensors can provide leader-gripper force feedback and tactile data recording.

## 📂 Repo Structure

| Path                                          | What it's for                                                                                                                          |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `aloha_scripts/constants.py`                | Task config table (`TASK_CONFIGS`), gripper calibration constants, `START_ARM_POSE`, control rate (`DT`/`FPS`)                 |
| `aloha_scripts/test_arm_joints.py`          | Read-only: check `/joint_states` on all 4 arms before doing anything else                                                            |
| `aloha_scripts/test_gripper_range.py`       | Read-only: measure each gripper's open/close joint range, for calibrating `constants.py`                                             |
| `aloha_scripts/one_side_teleop.py`          | Single-arm or dual-arm teleoperation test (no recording); optional leader-gripper force feedback tuning via CLI flags                  |
| `aloha_scripts/check_tactile_feedback.py`   | Optional tactile support: live tactile viewer **and** interactive slider UI to tune force-feedback feel on one gripper at a time |
| `aloha_scripts/sleep_plus.py`               | Move the arms to a safe rest pose, with or without cutting torque                                                                      |
| `aloha_scripts/record_episodes_aloha2.py`   | Record a classic ACT-style HDF5 dataset (`episode_N.hdf5`)                                                                           |
| `aloha_scripts/replay_episodes.py`          | Replay a recorded HDF5 episode's `/action` on the real puppet arms (no confirmation prompt - real motion)                            |
| `aloha_scripts/visualize_episodes_multi.py` | Offline: plot qpos/qvel/effort/action curves and dump camera frames from a recorded episode                                            |
| `detr/`                                     | Model definitions                                                                                                                      |
| `imitate_episodes_multi_gpu.py`             | Train and evaluate ACT/TactileACT/Diffusion/CNNMLP policies, multi-GPU; real-robot eval has keyboard control (RIGHT/ESC/LEFT)          |
| `policy.py`                                 | Adaptor for ACT, DiffusionPolicy, CNNMLP, and TactileACT                                                                               |
| `aloha_scripts/robot_utils.py`              | Shared robot utilities used by other scripts; not a standalone command                                                                 |
| `utils.py`                                  | Data loading and training helper functions                                                                                             |

---

## 🏗️ Quick Start Guide

### 🖥️ Software Selection – OS Compatibility

This project has been tested and confirmed to work with the following configuration:

- ✅ **Ubuntu 20.04 + ROS 1 Noetic** (Fully tested and verified)

Other configurations may work as well, but they have not been tested yet. If you successfully run this project on a different setup, feel free to contribute by sharing your experience! 🚀

## 🛠️ Installation

```sh
    git clone https://github.com/guningquan/act-triple-plus-aloha2
    cd act-triple-plus-aloha2
    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install -r requirements.txt
    cd detr && pip install -e .
    cd .. && cd robomimic && pip install -e .
    cd ..
```

Before running the scripts, configure `DATA_DIR` near the top of `aloha_scripts/constants.py`.

## 1️⃣ Start ROS and check the hardware

Start the four arms and cameras using the ROS launch files for your ALOHA hardware.
Source your ROS workspace first (replace `~/interbotix_ws_noetic` if yours is elsewhere), then run
your hardware's launch command. Keep this terminal open.

```sh
# ROS terminal
source /opt/ros/noetic/setup.sh && source ~/interbotix_ws_noetic/devel/setup.sh
# Run the arm/camera launch command supplied by your hardware setup here.
```

**Optional — tactile sensing and force feedback:** if you use the XENSE integration, the following
launch command starts the arms, cameras, and tactile publisher together:

```sh
roslaunch aloha 4arms_tactile_teleop.launch extra_outputs:="force"
```

For this integration, `extra_outputs:="force"` enables the force field needed for
leader-gripper force feedback or raw force recording; `force_norm` does **not** need to be
added separately (it is just `force[..., 2]` published again). Without tactile hardware,
skip this tactile launch command and use `--no_force_feedback` for teleoperation and recording
as shown below.

In a second terminal, sanity-check the arms before touching anything:

```sh
python aloha_scripts/test_arm_joints.py --compact
python aloha_scripts/test_arm_joints.py --arms puppet_left,puppet_right --loop --rate 2
```

`--compact` prints one row of six arm joints plus the gripper for each arm. Use `--loop`
with `--rate N` for continuous updates at N Hz; press `Ctrl+C` to stop. These checks only
read joint states. Offline visualization does not require the robot to be running.

For the optional XENSE integration, `rectify` is always included in the tactile bundle.
Select additional fields with `extra_outputs`; an empty value disables extra fields,
but does not disable feedback in the Python scripts. Use `--no_force_feedback` explicitly
when force feedback is not needed.

## 2️⃣ Calibrate gripper range (first time, or after any hardware change)

`constants.py` hardcodes each gripper's measured open/close joint angles (`MASTER_LEFT_GRIPPER_JOINT_OPEN`,
`PUPPET_RIGHT_GRIPPER_JOINT_CLOSE`, etc.) — they drive the leader↔follower mapping, the force-feedback
soft limits, and the starting gripper opening, so they must match your actual hardware. Run once per gripper
(master grippers can be moved by hand directly; puppet grippers need `--release` to go torque-free first):

```sh
python aloha_scripts/test_gripper_range.py master_left
python aloha_scripts/test_gripper_range.py master_right
python aloha_scripts/test_gripper_range.py puppet_left --release
python aloha_scripts/test_gripper_range.py puppet_right --release
```

Move each gripper through its full range, `Ctrl+C`, and copy the printed `min/closed` and `max/open`
values into the matching 8 constants in `constants.py`. This is a manual edit — nothing writes it
back automatically. Set `*_CLOSE` to the measured minimum and `*_OPEN` to the maximum for
`MASTER_LEFT`, `MASTER_RIGHT`, `PUPPET_LEFT`, and `PUPPET_RIGHT` independently. Restart the
Python script after editing; ROS does not need to restart. Recalibrate after changing fingers,
linkages, or gripper motors, or whenever the leader/follower opening ranges no longer match.
The puppet `--release` option reboots its gripper motor, switches to PWM mode, and disables
torque so it can be moved by hand.

## 3️⃣ Teleoperation test and optional force-feedback tuning (no recording)

Try the leader/follower mirroring before recording anything. For a setup without tactile feedback:

```sh
python aloha_scripts/one_side_teleop.py left --no_force_feedback    # or: right / dual
```

For optional tactile force feedback, start the tactile publisher and omit `--no_force_feedback`
(the script enables feedback by default).

Use the keyboard in the terminal running the script:

| Key             | Action                                                                        |
| --------------- | ----------------------------------------------------------------------------- |
| `→`          | Start or resume teleoperation.                                                |
| `←`          | Pause follower motion, zero force feedback, and lock the leaders.             |
| `b`           | Re-zero the force-feedback baseline when enabled; keep the grippers empty.    |
| `ESC` / `q` | Stop and lock, then move to the sleep pose and power off after a short delay. |

`--start_open_ratio` sets the initial gripper opening (default `1.0`, fully open).
`--disable_gripper_contact_guard` disables the follower gripper contact/stall guard for
teleoperation debugging; leave the guard enabled during normal use.

### Optional tactile feedback controls

If you use tactile force feedback and it feels off (too strong, too weak, catches unexpectedly), tune it live with
sliders on a single gripper (arms must already be in a safe resting pose):

```sh
python aloha_scripts/check_tactile_feedback.py left   # or: right
```

The tuning tool controls the selected grippers only; it does not move or change torque on
the arm joints. Put the arms in a safe resting pose first, for example with
`python aloha_scripts/sleep_plus.py --sleep`.

Its ten sliders control force deadband, full scale, curve, minimum/maximum wall stiffness,
maximum feedback PWM, ramp time, closing speed limit, closing brake gain, and maximum brake PWM.
Press `1`–`9` or `0` to select a parameter, type a value, and press `Enter` to apply it;
`Esc` cancels text entry or exits when not editing. Press `b` to re-zero the baseline.
Press `p` or exit to print the tuned command-line arguments.

The following flags are shared by teleoperation and the HDF5 recording script:

| Flag                                               | Purpose                                                         |
| -------------------------------------------------- | --------------------------------------------------------------- |
| `--force_deadband`                               | Ignore small forces around the baseline.                        |
| `--force_full_scale`                             | Set the force scale used to normalize feedback.                 |
| `--wall_stiffness_min`, `--wall_stiffness_max` | Set the contact wall stiffness range.                           |
| `--max_feedback_pwm`                             | Limit the tactile feedback PWM command.                         |
| `--close_speed_limit`                            | Set the closing speed threshold for braking.                    |
| `--close_brake_gain`                             | Set the closing brake strength.                                 |
| `--force_baseline_samples`                       | Set the number of samples used to measure the resting baseline. |
| `--no_force_feedback`                            | Disable leader-gripper force feedback.                          |

Copy the supported flags into the recording command after tuning. The tuner also prints
`--force_curve`, `--force_ramp_s`, and `--close_brake_max_pwm`; these are accepted by
`one_side_teleop.py`, but are not currently exposed by the HDF5 recording CLI.
`one_side_teleop.py` supplies the shared defaults. Both sides use the same CLI parameter
values, with separate tactile topics and measured baselines; there are no separate
`--left_*` / `--right_*` feedback tuning flags.

## 4️⃣ Define your task

Add (or reuse) an entry in `TASK_CONFIGS` in `aloha_scripts/constants.py`:

```python
'packaging': {
    'dataset_dir': DATA_DIR + '/packaging',
    'episode_len': 1600,          # tune to how long your task actually takes, in control steps
    'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'],
    # Optional - only add these if you also want to record raw tactile arrays:
    # 'tactile_sensors': ['left_gripper_outer', 'left_gripper_inner',
    #                      'right_gripper_outer', 'right_gripper_inner'],
    # 'tactile_outputs': ['rectify', 'force'],
},
```

`--task_name` must match a key in `TASK_CONFIGS`. Use the current table in `constants.py`
as the source of truth for episode lengths and sensor lists. Existing examples include
`aloha2_test` for cameras and joints, and `aloha2_tactile_test` / `adapter_insertion` for
additional raw tactile arrays. The `packaging` block above is an example configuration;
adjust its episode length and cameras to your task.

Only list cameras that your hardware actually publishes. Some older `zip_tie` configurations
include a `gel` camera; selecting that task does not start the sensor. Tactile feedback and
saving tactile arrays are independent: enabling feedback alone does not add tactile data
to an HDF5 episode.

A training-only configuration such as `zip_tie_cotrain` may use a list of `dataset_dir` paths,
plus `stats_dir`, `sample_weights`, and `train_ratio`. The HDF5 recording script requires a
single dataset directory, so use a separate collection task for each source dataset.

## 5️⃣ Record a dataset

```sh
python aloha_scripts/record_episodes_aloha2.py \
    --task_name packaging \
    --start_idx 0 --end_idx 49 \
    --no_force_feedback
```

For optional tactile force feedback, omit `--no_force_feedback` and make sure the tactile
publisher provides the force field. Recording raw tactile arrays is configured separately:
add `tactile_sensors` and `tactile_outputs` to your task config only when needed, and leave
them out for a setup without tactile sensors.

This moves both arms to the start pose, then waits for `→` to begin recording. During recording:
`→` save and stop early, `←` discard and redo this episode, `b` re-zero the force-feedback
baseline (when enabled), `ESC` abort the whole run. Between episodes the arms automatically move to the sleep
pose; at the end (or on `ESC`) they power off.

`--start_idx` and `--end_idx` are required and inclusive. To record one HDF5 episode, set
both to the same index; the current batch loop assigns `--episode_idx` from this range.
Discarding an episode does not consume its index. `ESC` keeps previously saved episodes
and discards the current attempt. Recording starts with the right arrow key, not by
squeezing a gripper.

## 6️⃣ Check the data

```sh
python aloha_scripts/visualize_episodes_multi.py --dataset_dir <dataset_dir> --episode_idx 0
```

Plots qpos/qvel/effort/action curves and dumps camera frames; does not move the robot. Pass
multiple indices as `--episode_idx 3,5,8` or a range as `--episode_idx 5-19`.

To confirm the recorded `/action` actually drives the real arms correctly, you can replay it —
**this moves the real puppet arms immediately, with no confirmation prompt**:

```sh
python aloha_scripts/replay_episodes.py --dataset_dir <dataset_dir> --episode_idx 0
```

Replay has no `--dry_run` or step-limit option. Check the workspace and starting pose before
running it. It opens both puppet grippers at the end but does not move the arms to their
sleep pose; use `sleep_plus.py` afterward if needed.

## 7️⃣ Put the robot away

```sh
python aloha_scripts/sleep_plus.py --shut_down          # All robots move to rest pose, then torque off.
python aloha_scripts/sleep_plus.py --shut_down_puppet   # Only the puppet robots move to rest pose and torque off.
python aloha_scripts/sleep_plus.py --sleep              # All robots move to rest pose, torque stays on.
```

The rest motion uses a transition pose followed by the calibrated resting pose. Keep this
two-stage sequence when adapting the setup. Use `--sleep` between sessions when continuing
soon, `--shut_down` when finished, or `--shut_down_puppet` when only the followers should
rest and power off. The HDF5 recorder already moves all arms to sleep between episodes.

## 🧠 Policy Training

```sh
export CUDA_VISIBLE_DEVICES=0,1
python imitate_episodes_multi_gpu.py \
    --task_name packaging \
    --ckpt_dir <checkpoint_dir> \
    --policy_class ACT \
    --kl_weight 10 --chunk_size 100 \
    --hidden_dim 512 --batch_size 64 \
    --dim_feedforward 3200 --lr 1e-5 --seed 0 \
    --num_steps 100000 --eval_every 2000 \
    --validate_every 2000 --save_every 2000
```

`--policy_class` can be `ACT`, `TactileACT`, `ACTNormals`, `Diffusion`, or `CNNMLP` - use `TactileACT`
if your task's `TASK_CONFIGS` entry includes `tactile_sensors`/`tactile_outputs`. To continue
training from an existing checkpoint (e.g. after adding more HIL/corrected episodes), add
`--resume_ckpt_path <ckpt_dir>/policy_last.ckpt`.

## 📡 Policy Evaluation / Deployment

```sh
export CUDA_VISIBLE_DEVICES=0
python imitate_episodes_multi_gpu.py \
    --task_name packaging \
    --ckpt_dir <checkpoint_dir> \
    --policy_class ACT \
    --kl_weight 10 --chunk_size 100 \
    --hidden_dim 512 --batch_size 8 \
    --dim_feedforward 3200 --lr 1e-5 --seed 0 \
    --num_steps 100000 --eval_every 2000 \
    --validate_every 2000 --save_every 2000 \
    --temporal_ensemble --eval --num_rollouts 20
```

On real hardware (`real_robot=True` in the resolved task config), each rollout now has keyboard
control: `→` start the rollout, `ESC`/`q` stop and power off (`shut_down_puppet_robots()`), `←`
abort just this rollout and re-home for another attempt (does **not** count against
`--num_rollouts`).

## Troubleshooting

| Symptom                                          | Check                                                                                                                                                                   |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError: Unknown user` at startup          | Configure the username/path selection for `DATA_DIR` in `constants.py`.                                                                                             |
| Leader/follower gripper travel does not match    | Recalibrate all four grippers independently and update the eight open/close constants.                                                                                  |
| Optional force feedback stays at zero            | Check the tactile topic and freshness, and enable `force` in the publisher's `extra_outputs`. Missing/stale force data produces warnings and zero tactile feedback. |
| A requested HDF5 tactile field contains zeros    | Make sure the publisher supplies every configured `tactile_outputs` field; missing fields are warned about and zero-filled by the HDF5 recorder.                      |
| A tuner-generated flag is rejected by a recorder | Copy only the shared flags listed above; some tuner parameters are not exposed by the HDF5 recording CLI.                                                               |
| Left/right feedback feels different              | Re-zero each baseline with empty grippers and check sensor/calibration differences; both sides currently share tuning values.                                           |

## 🙏 Acknowledgements

This project codebase is built based on [ALOHA](https://github.com/tonyzhaozh/aloha) and [ACT](https://github.com/tonyzhaozh/act).
