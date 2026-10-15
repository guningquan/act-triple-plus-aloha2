#!/usr/bin/env python3
"""Read gripper joint range from ROS joint_states, with optional torque release."""

from __future__ import annotations

import argparse
import signal
import sys
import time


ROBOT_MODELS = {
    "master_left": "wx250s",
    "master_right": "wx250s",
    "puppet_left": "vx300s",
    "puppet_right": "vx300s",
}


def release_gripper(robot_name: str, robot_model: str) -> None:
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    print(f"Releasing {robot_name} gripper torque...")
    bot = InterbotixManipulatorXS(
        robot_model=robot_model,
        robot_name=robot_name,
        moving_time=0.3,
        accel_time=0.3,
        init_node=False,
    )
    if robot_name.startswith("puppet"):
        bot.dxl.robot_reboot_motors("single", "gripper", True)
        time.sleep(0.3)
    bot.dxl.robot_set_operating_modes("single", "gripper", "pwm")
    bot.dxl.robot_torque_enable("single", "gripper", False)
    print("Gripper torque released. Move it by hand and watch the value.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", choices=sorted(ROBOT_MODELS))
    parser.add_argument("--release", action="store_true", help="Release only this gripper torque before reading.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        import rospy
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        print(f"Failed to import ROS Python packages: {exc}", file=sys.stderr)
        return 2

    robot_model = ROBOT_MODELS[args.robot]
    topic = f"/{args.robot}/joint_states"

    rospy.init_node(f"gripper_range_{args.robot}", anonymous=True)
    if args.release:
        release_gripper(args.robot, robot_model)

    latest = [None]
    history = []

    def cb(msg: JointState) -> None:
        for idx, name in enumerate(msg.name):
            if name == "gripper":
                latest[0] = float(msg.position[idx])
                history.append(float(msg.position[idx]))
                return

    rospy.Subscriber(topic, JointState, cb, queue_size=1)

    print(f"Waiting for {topic}...")
    for _ in range(50):
        if history:
            break
        time.sleep(0.1)
    if not history:
        print(f"No messages received from {topic}. Is roslaunch running?", file=sys.stderr)
        return 1

    print(f"\nGripper range test [{args.robot}]")
    print("Move the gripper to min/max positions. Ctrl+C exits and prints stats.\n")

    exit_flag = [False]
    signal.signal(signal.SIGINT, lambda *_: exit_flag.__setitem__(0, True))

    while not rospy.is_shutdown() and not exit_flag[0]:
        value = latest[0]
        if value is None:
            continue
        bar = "#" * min(int(abs(value) * 30), 50)
        print(f"\r  gripper: {value:+.4f}  {bar}", end="", flush=True)
        time.sleep(0.05)

    print("\n\nStats:")
    if len(history) > 10:
        print(f"  min/closed: {min(history):.4f}")
        print(f"  max/open:   {max(history):.4f}")
        print(f"  range:      {max(history) - min(history):.4f}")
    else:
        print("  Not enough samples. Move the gripper by hand and retry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
