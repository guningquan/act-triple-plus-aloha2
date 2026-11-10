#!/usr/bin/env python3
"""Run a trained openpi policy (pi0 / pi0.5 ALOHA config) or LeRobot checkpoint on the real robot, for evaluation only.

Nothing is recorded and the leader arms are not used - hil_rollout.py is the script for taking over
and collecting data. This is the act-triple counterpart of openpi's examples/aloha_real/main.py:
the same observation (qpos + three cameras resized with pad to 224x224, CHW uint8 + prompt), the
same blocking chunk execution (ActionChunkBroker: infer, play --action_horizon steps at 50 Hz,
infer again), but in this repo's ROS env with this repo's homing, safety limits and shutdown.

Server (in the openpi repo on the GPU box):
    uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
        --policy.config=pi05_packaging_simple_2_full \
        --policy.dir=<openpi_checkpoints>/pi05_packaging_simple_2_full/packaging_simple_2_pt/20000
Client (this machine, `conda activate aloha`, after roslaunch):
    python3 aloha_scripts/run_pi_policy.py --host 10.240.77.20 --port 8000 --prompt "packaging the items"

A LeRobot checkpoint trained on these hdf5 episodes (pi0.5, MolmoAct2, ACT, GR00T, ...) runs the same
way with --policy lerobot, served by lerobot's examples/aloha_ros/serve_policy_websocket.py:
    python examples/aloha_ros/serve_policy_websocket.py --policy_path <.../pretrained_model> \
        --task "packaging the items" --port 8000                       (server, lerobot repo)
    python3 aloha_scripts/run_pi_policy.py --policy lerobot --host 10.240.77.20 --port 8000
It sends qpos/qvel/effort and the checkpoint's cameras at full resolution (JPEG) instead of the
224x224 openpi observation; the chunk execution, keys and safety limits are the same.

Keys (terminal):
    RIGHT       start the round (the followers are homed first)
    LEFT or r   abort this round: the followers stop where they are, then re-home for the next one
    ESC or q    stop: grippers open, followers walk to the sleep pose, torque off
A failed or timed-out inference stops the round the same way as LEFT.

Before the first round one warm-up inference is sent (openpi compiles the model on its first call,
which can take minutes; --warmup_timeout). Followers are commanded through
hil_devices.FollowerCommander: targets clamped to the joint limits, at most --max_joint_step rad per
tick. The policy is paused (round aborted) if a camera image is older than --max_stream_age.
Step-by-step guide: README_run_robot.md.
"""

import argparse
import logging
import time

import numpy as np

from chunk_executor import ChunkExecutor, PolicyError
from policy_client import Frame, LerobotAlohaAdapter, OpenpiAlohaAdapter, PolicyClient

DEFAULT_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class StaleImage(RuntimeError):
    pass


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=("openpi", "lerobot"), default="openpi",
                        help="Server type: openpi serve_policy.py, or lerobot serve_policy_websocket.py")
    parser.add_argument("--host", required=True, help="Policy server host, e.g. 10.240.77.20")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default=None,
                        help="Task prompt; omitted -> the server config's default prompt (lerobot: its --task)")
    parser.add_argument("--num_episodes", type=int, default=1, help="Rounds to run (aborted rounds do not count)")
    parser.add_argument("--max_episode_steps", type=int, default=3000, help="Ticks per round (50 Hz)")
    parser.add_argument("--action_horizon", type=int, default=50, help="Chunk steps executed per inference")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Cameras sent to the server (openpi default: {}; lerobot default: the "
                             "checkpoint's cameras)".format(" ".join(DEFAULT_CAMERAS)))
    parser.add_argument("--resolution", type=int, default=224, help="openpi: client-side resize_with_pad size")
    parser.add_argument("--jpeg_quality", type=int, default=90,
                        help="lerobot: JPEG quality of the camera images sent (90 = the recorder's)")
    parser.add_argument("--infer_timeout", type=float, default=10.0, help="Seconds before a reply counts as lost")
    parser.add_argument("--warmup_timeout", type=float, default=900.0,
                        help="Timeout of the warm-up inference before the first round")
    parser.add_argument("--max_joint_step", type=float, default=0.1,
                        help="Per-tick follower arm joint change limit (rad); 0 disables")
    parser.add_argument("--max_stream_age", type=float, default=0.5, help="Abort the round on older camera images")
    return parser


def image_ages(image_recorder, cameras, now):
    """Seconds since each camera's last frame (ROS header stamps)."""
    ages = {}
    for camera in cameras:
        secs = getattr(image_recorder, camera + "_secs", None)
        if secs is None:
            raise StaleImage("no {} image yet".format(camera))
        ages[camera] = now - (secs + getattr(image_recorder, camera + "_nsecs") * 1e-9)
    return ages


def capture(env, cameras, max_age, ros_time):
    """Observation for the policy; raises StaleImage when a camera stopped publishing."""
    for camera, age in image_ages(env.image_recorder, cameras, ros_time()).items():
        if age > max_age:
            raise StaleImage("{} image is {:.2f}s old".format(camera, age))
    obs = env.get_observation()
    return Frame(time.monotonic(), obs["qpos"], obs["qvel"], obs["effort"], obs["images"])


def home_followers(puppets, reset_pose, move_arms, move_grippers, gripper_open, control_dt):
    """Followers to the reset pose with open grippers (where every training episode started), then
    back to the per-tick trajectory time: move_arms() leaves the slow homing time in the registers,
    which would make every later 50 Hz command crawl."""
    move_arms(list(puppets), [reset_pose, reset_pose], move_time=2.0)
    move_grippers(list(puppets), list(gripper_open), move_time=0.5)
    for bot in puppets:
        bot.arm.set_trajectory_time(moving_time=control_dt, accel_time=control_dt / 2)


def run_episode(executor, follower, keyboard, capture_frame, max_steps, control_dt, clock=time.monotonic,
                sleep=time.sleep, show_progress=True):
    """One round at 50 Hz. Returns "done", "reset" (LEFT / r / policy or camera failure) or "quit".
    Progress is one tqdm line (ticks, last inference time) instead of periodic prints."""
    from tqdm import tqdm

    executor.reset()
    keyboard.drain()
    bar = tqdm(total=max_steps, unit="tick", dynamic_ncols=True, disable=not show_progress)
    say = bar.write if show_progress else print
    next_tick = clock()
    try:
        for step in range(max_steps):
            for event in keyboard.poll():
                if event == "esc":
                    return "quit"
                if event in ("left", "reset"):
                    say("Round aborted by the operator.")
                    return "reset"
            try:
                action = executor.step(lambda: capture_frame())
            except (PolicyError, StaleImage) as exc:
                say("Round stopped: {}".format(exc))
                return "reset"
            follower.send(action)
            bar.update(1)
            if step % 10 == 0 and executor.latencies:
                bar.set_postfix_str("infer {:.0f} ms".format(1000 * executor.latencies[-1]), refresh=False)
            next_tick += control_dt
            delay = next_tick - clock()
            if delay > 0:
                sleep(delay)
            elif delay < -control_dt:
                next_tick = clock()  # blocking inference: no catch-up burst
        return "done"
    finally:
        bar.close()


def wait_for_start(keyboard, sleep=time.sleep):
    """True on RIGHT, False on ESC/q."""
    keyboard.drain()
    while True:
        for event in keyboard.poll():
            if event == "right":
                return True
            if event == "esc":
                return False
        sleep(0.02)


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Hardware imports here, so --help and the tests work without ROS.
    import rospy
    from constants import DT, PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN, START_ARM_POSE
    from hil_devices import FollowerCommander
    from one_side_teleop import KeyboardControl
    from real_env import make_real_env
    from robot_utils import move_arms, move_grippers
    from sleep_plus import shut_down_puppet_robots

    client = PolicyClient(args.host, args.port, infer_timeout_s=args.infer_timeout)
    logging.info("Connected to %s. Server metadata: %s", client.uri, client.metadata)
    reset_pose = list(client.metadata.get("reset_pose") or START_ARM_POSE)[:6]  # as openpi aloha_real

    env = make_real_env(init_node=True, setup_robots=True)
    puppets = (env.puppet_bot_left, env.puppet_bot_right)
    if args.policy == "lerobot":
        adapter = LerobotAlohaAdapter.from_metadata(client.metadata, args.prompt, cameras=args.cameras,
                                                    jpeg_quality=args.jpeg_quality)
    else:
        adapter = OpenpiAlohaAdapter(args.prompt, cameras=args.cameras or DEFAULT_CAMERAS,
                                     resolution=args.resolution)

    def infer(frame, timeout=None):
        obs = adapter.prepare(frame)
        if obs.get("prompt", "") is None:
            del obs["prompt"]  # the server falls back to the training config's default prompt
        return adapter.decode(client.infer(obs, timeout=timeout)["actions"], frame)

    def capture_frame():
        return capture(env, adapter.cameras, args.max_stream_age, rospy.get_time)

    keyboard = KeyboardControl()
    quit_requested = False
    try:
        print("Homing the followers...")
        home_followers(puppets, reset_pose, move_arms, move_grippers,
                       (PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN), DT)
        print("Warming up the policy server (its first inference compiles the model and can take minutes)...")
        start = time.monotonic()
        infer(capture_frame(), timeout=args.warmup_timeout)
        print("Policy server ready ({:.0f}s).".format(time.monotonic() - start))

        done = 0
        while done < args.num_episodes and not quit_requested:
            print("Round {}/{}: homed. RIGHT = start, ESC = quit.".format(done + 1, args.num_episodes))
            if keyboard.enabled and not wait_for_start(keyboard):
                quit_requested = True
                break
            print("Running. LEFT or r = abort this round, ESC/q = stop and power down.")
            follower = FollowerCommander(puppets, max_joint_step=args.max_joint_step or None)
            executor = ChunkExecutor(infer, adapter.policy_hz, DT, action_horizon=args.action_horizon)
            outcome = run_episode(executor, follower, keyboard, capture_frame, args.max_episode_steps, DT)
            print("Round {}/{}: {} ({}).".format(done + 1, args.num_episodes, outcome, executor.stats()))
            if outcome == "quit":
                quit_requested = True
                break
            if outcome == "done":
                done += 1
            if done < args.num_episodes:
                print("Re-homing the followers...")
                home_followers(puppets, reset_pose, move_arms, move_grippers,
                               (PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN), DT)
    finally:
        keyboard.close()
        client.close()
        print("Opening the grippers and moving the followers to the sleep pose...")
        shut_down_puppet_robots()
        print("Followers are at the sleep pose with torque off.")


if __name__ == "__main__":
    main()
