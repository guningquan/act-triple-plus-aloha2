"""Fake Interbotix bots and a manual clock for the HIL tests."""

import concurrent.futures
from types import SimpleNamespace

import numpy as np


class FakeClock:
    def __init__(self, start=100.0):
        self.now = start

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(seconds, 0.0)


class FakeDxl:
    def __init__(self, bot, gains=800):
        self.bot = bot
        self.p_gains = [gains] * 6
        self.log = []

    @property
    def joint_states(self):
        return SimpleNamespace(position=list(self.bot.position), velocity=[0.0] * 9, effort=[0.0] * 9)

    def robot_get_motor_registers(self, cmd_type, name, reg):
        assert reg == "Position_P_Gain"
        return SimpleNamespace(values=list(self.p_gains))

    def robot_set_motor_registers(self, cmd_type, name, reg, value):
        self.log.append(("reg", cmd_type, name, reg, value))
        if reg == "Position_P_Gain":
            if cmd_type == "group":
                self.p_gains = [value] * 6
            else:
                self.p_gains[self.bot.joint_names.index(name)] = value

    def robot_set_operating_modes(self, cmd_type, name, mode, *args):
        self.log.append(("mode", cmd_type, name, mode))

    def robot_torque_enable(self, cmd_type, name, enable):
        self.log.append(("torque", cmd_type, name, enable))
        if cmd_type == "group":
            self.bot.torque = enable


class FakeArm:
    def __init__(self, bot):
        self.bot = bot
        self.group_info = SimpleNamespace(joint_names=bot.joint_names,
                                          joint_lower_limits=[-3.0, -1.8, -1.6, -3.0, -1.8, -3.0],
                                          joint_upper_limits=[3.0, 1.2, 1.6, 3.0, 2.2, 3.0])
        self.published = []
        self.moving_time = 2.0
        self.robot_des = SimpleNamespace(M=np.eye(4), Slist=np.eye(6))

    def publish_positions(self, positions, moving_time=None, accel_time=None, blocking=True):
        self.published.append(list(positions))
        self.bot.dxl.log.append(("arm_goal", list(positions)))
        if self.bot.torque and self.bot.follow_commands:
            self.bot.position[:6] = list(positions)


class FakePublisher:
    def __init__(self, bot):
        self.bot = bot
        self.sent = []

    def publish(self, command):
        self.sent.append(command.cmd)
        self.bot.dxl.log.append(("gripper_goal", command.cmd))
        if self.bot.follow_commands:
            self.bot.position[6] = command.cmd


class FakeBot:
    def __init__(self, arm=(0.0, -0.96, 1.16, 0.0, -0.3, 0.0), gripper=0.5, follow_commands=True):
        self.joint_names = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
        self.position = list(arm) + [gripper, 0.03, -0.03]
        self.torque = True
        self.follow_commands = follow_commands
        self.dxl = FakeDxl(self)
        self.arm = FakeArm(self)
        self.gripper = SimpleNamespace(core=SimpleNamespace(pub_single=FakePublisher(self)))

    def set_arm(self, arm):
        self.position[:6] = list(arm)

    def arm_position(self):
        return np.array(self.position[:6])


class FakeHaptics:
    def __init__(self):
        self.calls = []

    def start(self):
        self.calls.append("start")

    def update(self):
        self.calls.append("update")

    def restore(self):
        self.calls.append("restore")


class SyncExecutor:
    """Runs submitted work immediately (deterministic stand-in for a leader's worker thread)."""

    def submit(self, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True):
        pass
