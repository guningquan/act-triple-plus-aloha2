#!/usr/bin/env python3
"""Read ALOHA arm joint_states without sending any robot commands.

Usage:
  python aloha_scripts/test_arm_joints.py
  python aloha_scripts/test_arm_joints.py --arms puppet_left,puppet_right
  python aloha_scripts/test_arm_joints.py --loop --rate 2
  python aloha_scripts/test_arm_joints.py --compact --degrees
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass


ARM_TOPICS = {
    "master_left": "/master_left/joint_states",
    "master_right": "/master_right/joint_states",
    "puppet_left": "/puppet_left/joint_states",
    "puppet_right": "/puppet_right/joint_states",
}

DISPLAY_ORDER = ("master_left", "master_right", "puppet_left", "puppet_right")


@dataclass
class JointSnapshot:
    names: list[str]
    positions: list[float]
    velocities: list[float]
    efforts: list[float]
    stamp: float | None
    received_at: float


def parse_arms(value: str) -> list[str]:
    if value == "all":
        return list(DISPLAY_ORDER)

    arms = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [arm for arm in arms if arm not in ARM_TOPICS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown arm(s): {', '.join(unknown)}. Choices: all, {', '.join(DISPLAY_ORDER)}"
        )
    return arms


def fmt(value: float | None, width: int = 11, precision: int = 5) -> str:
    if value is None:
        return " " * (width - 3) + "n/a"
    return f"{value:{width}.{precision}f}"


def stamp_to_float(stamp) -> float | None:
    try:
        return float(stamp.secs) + float(stamp.nsecs) * 1e-9
    except AttributeError:
        return None


class JointStateMonitor:
    def __init__(self, rospy, joint_state_msg, arms: list[str]):
        self.rospy = rospy
        self.snapshots: dict[str, JointSnapshot] = {}
        self.subscribers = []

        for arm in arms:
            topic = ARM_TOPICS[arm]
            sub = rospy.Subscriber(topic, joint_state_msg, self._callback, callback_args=arm, queue_size=1)
            self.subscribers.append(sub)

    def _callback(self, msg, arm: str) -> None:
        names = list(msg.name) if msg.name else [f"joint_{idx}" for idx in range(len(msg.position))]
        self.snapshots[arm] = JointSnapshot(
            names=names,
            positions=list(msg.position),
            velocities=list(msg.velocity),
            efforts=list(msg.effort),
            stamp=stamp_to_float(msg.header.stamp),
            received_at=time.time(),
        )

    def wait_for_all(self, arms: list[str], timeout: float) -> list[str]:
        deadline = time.time() + timeout
        rate = self.rospy.Rate(50)
        while not self.rospy.is_shutdown() and time.time() < deadline:
            missing = [arm for arm in arms if arm not in self.snapshots]
            if not missing:
                return []
            rate.sleep()
        return [arm for arm in arms if arm not in self.snapshots]


def value_at(values: list[float], index: int) -> float | None:
    if index >= len(values):
        return None
    return values[index]


def print_compact(arm: str, snapshot: JointSnapshot, show_degrees: bool) -> None:
    positions = snapshot.positions
    if show_degrees:
        positions = [math.degrees(value) for value in positions]
        unit = "deg"
    else:
        unit = "rad"
    joined = ", ".join(f"{value:.5f}" for value in positions)
    age = time.time() - snapshot.received_at
    print(f"{arm}: position_{unit}=[{joined}] age={age:.3f}s")


def print_table(arm: str, topic: str, snapshot: JointSnapshot, show_degrees: bool) -> None:
    age = time.time() - snapshot.received_at
    stamp = "n/a" if snapshot.stamp is None else f"{snapshot.stamp:.6f}"
    print(f"\n{arm}  topic={topic}  stamp={stamp}  age={age:.3f}s")

    pos_label = "pos(deg)" if show_degrees else "pos(rad)"
    print(f"{'idx':>3s}  {'name':<24s}  {pos_label:>11s}  {'vel':>11s}  {'effort':>11s}")
    print("-" * 69)

    n_rows = max(len(snapshot.names), len(snapshot.positions), len(snapshot.velocities), len(snapshot.efforts))
    for idx in range(n_rows):
        name = snapshot.names[idx] if idx < len(snapshot.names) else f"joint_{idx}"
        pos = value_at(snapshot.positions, idx)
        if pos is not None and show_degrees:
            pos = math.degrees(pos)
        vel = value_at(snapshot.velocities, idx)
        effort = value_at(snapshot.efforts, idx)
        print(f"{idx:3d}  {name:<24s}  {fmt(pos)}  {fmt(vel)}  {fmt(effort)}")


def print_snapshots(monitor: JointStateMonitor, arms: list[str], compact: bool, show_degrees: bool) -> None:
    for arm in arms:
        snapshot = monitor.snapshots.get(arm)
        if snapshot is None:
            print(f"{arm}: no /joint_states message received from {ARM_TOPICS[arm]}")
            continue
        if compact:
            print_compact(arm, snapshot, show_degrees)
        else:
            print_table(arm, ARM_TOPICS[arm], snapshot, show_degrees)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read current ALOHA arm joint values from ROS joint_states.")
    parser.add_argument(
        "--arms",
        type=parse_arms,
        default=list(DISPLAY_ORDER),
        help="Comma-separated arms, or 'all'. Default: all.",
    )
    parser.add_argument("--timeout", type=float, default=3.0, help="Seconds to wait for first messages.")
    parser.add_argument("--loop", action="store_true", help="Continuously print joint values until Ctrl-C.")
    parser.add_argument("--rate", type=float, default=1.0, help="Print rate in Hz when --loop is set.")
    parser.add_argument("--compact", action="store_true", help="Print one compact position vector per arm.")
    parser.add_argument("--degrees", action="store_true", help="Display positions in degrees instead of radians.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    arms = args.arms

    try:
        import rospy
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        print(f"Failed to import ROS Python packages: {exc}", file=sys.stderr)
        print("Run this inside the ROS/ALOHA environment, for example after sourcing ROS setup.", file=sys.stderr)
        return 2

    rospy.init_node("test_arm_joints", anonymous=True)
    monitor = JointStateMonitor(rospy, JointState, arms)

    missing = monitor.wait_for_all(arms, timeout=args.timeout)
    if missing:
        print("No joint_states received for: " + ", ".join(missing), file=sys.stderr)
        for arm in missing:
            print(f"  expected topic: {ARM_TOPICS[arm]}", file=sys.stderr)
        if len(missing) == len(arms):
            return 1

    try:
        if args.loop:
            rate_hz = max(args.rate, 0.1)
            rate = rospy.Rate(rate_hz)
            while not rospy.is_shutdown():
                print_snapshots(monitor, arms, compact=args.compact, show_degrees=args.degrees)
                print()
                rate.sleep()
        else:
            print_snapshots(monitor, arms, compact=args.compact, show_degrees=args.degrees)
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
