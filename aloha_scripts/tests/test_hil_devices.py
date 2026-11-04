import types

import numpy as np
import pytest
from fakes import FakeBot, FakeClock, FakeHaptics, SyncExecutor

import hil_devices
from constants import (MASTER_LEFT_GRIPPER_JOINT_CLOSE, MASTER_LEFT_GRIPPER_JOINT_OPEN,
                       PUPPET_LEFT_GRIPPER_JOINT_CLOSE, PUPPET_LEFT_GRIPPER_JOINT_OPEN)
from hil_devices import ClutchLeader, FollowerCommander

DT = 0.02


def make_leader(*args, **kwargs):
    kwargs.setdefault("io_executor", SyncExecutor())
    return ClutchLeader(*args, **kwargs)


def run_ticks(leader, clock, seconds, on_tick=None):
    end = clock.now + seconds
    while clock.now < end - 1e-9:
        clock.now += DT
        if on_tick:
            on_tick()
        leader.tick(clock.now)


def test_move_drives_the_leader_onto_the_follower_and_sets_the_reference():
    bot, clock = FakeBot(), FakeClock()
    leader = make_leader(bot, "left", control_dt=DT)
    target = np.array([0.2, -0.5, 0.9, 0.1, -0.2, 0.3])
    leader.begin_move(target, 0.25, 1.0, clock.now)
    assert leader.state == ClutchLeader.MOVING
    run_ticks(leader, clock, 1.0)
    assert leader.state == ClutchLeader.LOCKED
    np.testing.assert_allclose(bot.arm_position(), target)
    expected_gripper = MASTER_LEFT_GRIPPER_JOINT_CLOSE + 0.25 * (MASTER_LEFT_GRIPPER_JOINT_OPEN - MASTER_LEFT_GRIPPER_JOINT_CLOSE)
    assert bot.position[6] == pytest.approx(expected_gripper)
    np.testing.assert_allclose(leader.reference, target)
    # the motion is interpolated, not a jump
    first = np.array(bot.arm.published[0])
    assert np.abs(first - np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])).max() < 0.01


def test_release_ramps_p_gains_then_cuts_torque_and_engages_clutch_after_settling():
    bot, clock, haptics = FakeBot(), FakeClock(), FakeHaptics()
    leader = make_leader(bot, "left", release_ramp_s=0.4, settle_s=0.2, settle_timeout_s=1.0,
                          haptics=haptics, control_dt=DT)
    leader.configure()
    reference = np.array([0.1, -0.9, 1.1, 0.0, -0.3, 0.05])
    leader.reference = reference.copy()
    bot.follow_commands = False
    start = clock.now
    leader.begin_release(clock.now)
    run_ticks(leader, clock, 0.37)
    gains = [entry[4] for entry in bot.dxl.log if entry[0] == "reg"]
    assert gains == sorted(gains, reverse=True) and gains[-1] == 0 and len(gains) == 10
    assert bot.torque  # still on until the ramp has run its full length
    run_ticks(leader, clock, 0.07)  # torque cut at 0.40, settling from the next tick
    assert not bot.torque and leader.state == ClutchLeader.SETTLING
    assert bot.dxl.p_gains == [800] * 6  # restored after the torque went off
    torque_off_index = next(i for i, e in enumerate(bot.dxl.log) if e[:2] == ("torque", "group"))
    restore_index = max(i for i, e in enumerate(bot.dxl.log) if e[0] == "reg")
    assert restore_index > torque_off_index
    assert haptics.calls[0] == "start"
    # the arm sags a little, then holds still -> offset taken after settle_s
    bot.set_arm(reference + 0.03)
    run_ticks(leader, clock, 0.24)
    assert leader.state == ClutchLeader.FREE and leader.engaged
    assert clock.now - start < 0.8
    np.testing.assert_allclose(leader.offset, 0.03, atol=1e-9)
    # follower command = leader - offset; gripper stays absolute
    bot.set_arm(reference + 0.03 + 0.1)
    action = leader.action()
    np.testing.assert_allclose(action[:6], reference + 0.1)
    assert action[6] == pytest.approx((0.5 - MASTER_LEFT_GRIPPER_JOINT_CLOSE)
                                      / (MASTER_LEFT_GRIPPER_JOINT_OPEN - MASTER_LEFT_GRIPPER_JOINT_CLOSE))


def test_settle_timeout_when_the_operator_keeps_moving():
    bot, clock = FakeBot(follow_commands=False), FakeClock()
    leader = make_leader(bot, "right", release_ramp_s=0.0, settle_s=0.2, settle_timeout_s=0.5, control_dt=DT)
    leader.reference = bot.arm_position()
    leader.begin_release(clock.now)
    wobble = iter(range(1000))
    run_ticks(leader, clock, 0.6, on_tick=lambda: bot.set_arm(np.full(6, 0.05 * (next(wobble) % 2))))
    assert leader.state == ClutchLeader.FREE and leader.engaged


def test_lock_clears_offset_aims_goal_at_present_pose_then_torques_on():
    bot, clock, haptics = FakeBot(follow_commands=False), FakeClock(), FakeHaptics()
    leader = make_leader(bot, "left", release_ramp_s=0.0, settle_s=0.2, haptics=haptics, control_dt=DT)
    leader.reference = bot.arm_position()
    leader.begin_release(clock.now)
    run_ticks(leader, clock, 0.3)
    assert leader.engaged
    bot.set_arm(bot.arm_position() + 0.2)
    command = leader.action()
    bot.dxl.log.clear()
    leader.lock()
    assert leader.state == ClutchLeader.LOCKED and not leader.engaged
    np.testing.assert_allclose(leader.offset, 0.0)
    np.testing.assert_allclose(leader.reference, command[:6])  # next release references the follower
    kinds = [entry[0] for entry in bot.dxl.log]
    assert kinds.index("arm_goal") < kinds.index("torque") and kinds.index("gripper_goal") < kinds.index("torque")
    np.testing.assert_allclose(bot.dxl.log[kinds.index("arm_goal")][1], bot.arm_position())
    assert haptics.calls[-1] == "restore"


def test_lock_during_ramp_restores_gains():
    bot, clock = FakeBot(), FakeClock()
    leader = make_leader(bot, "left", release_ramp_s=0.4, control_dt=DT)
    leader.configure()
    leader.begin_release(clock.now)
    run_ticks(leader, clock, 0.1)
    assert bot.dxl.p_gains[0] < 800
    leader.lock()
    assert bot.dxl.p_gains == [800] * 6 and bot.torque


def test_absolute_takeover_has_no_offset():
    bot, clock = FakeBot(follow_commands=False), FakeClock()
    leader = make_leader(bot, "left", release_ramp_s=0.0, relative=False, control_dt=DT)
    leader.reference = bot.arm_position() - 0.3
    leader.begin_release(clock.now)
    clock.now += DT
    leader.tick(clock.now)
    assert leader.state == ClutchLeader.FREE
    np.testing.assert_allclose(leader.action()[:6], bot.arm_position())


def _command(arm_left, grip_left, arm_right, grip_right):
    return np.concatenate([arm_left, [grip_left], arm_right, [grip_right]])


def test_follower_converts_grippers_and_returns_what_it_sent():
    bots = (FakeBot(), FakeBot())
    clock = FakeClock()
    follower = FollowerCommander(bots, max_joint_step=None, clock=clock)
    home = bots[0].arm_position()
    sent = follower.send(_command(home + 0.01, 0.25, home, 1.0))
    joint = PUPPET_LEFT_GRIPPER_JOINT_CLOSE + 0.25 * (PUPPET_LEFT_GRIPPER_JOINT_OPEN - PUPPET_LEFT_GRIPPER_JOINT_CLOSE)
    assert bots[0].gripper.core.pub_single.sent[-1] == pytest.approx(joint)
    np.testing.assert_allclose(sent, _command(home + 0.01, 0.25, home, 1.0))
    np.testing.assert_allclose(follower.current(), sent)


def test_follower_clamps_limits_and_limits_per_tick_jumps():
    bots = (FakeBot(), FakeBot())
    follower = FollowerCommander(bots, max_joint_step=0.1, clock=FakeClock())
    home = bots[0].arm_position()
    far = home.copy()
    far[1] = 5.0  # beyond the shoulder limit (1.2) and far away
    sent = follower.send(_command(far, 0.5, home, 0.5))
    assert sent[1] == pytest.approx(home[1] + 0.1)  # limited step towards the clamped target
    for _ in range(30):
        sent = follower.send(_command(far, 0.5, home, 0.5))
    assert sent[1] == pytest.approx(1.2)  # settles at the joint limit


def test_follower_blend_eases_in_from_the_measured_pose():
    bots = (FakeBot(follow_commands=False), FakeBot(follow_commands=False))
    clock = FakeClock()
    follower = FollowerCommander(bots, max_joint_step=None, clock=clock)
    home = bots[0].arm_position()
    target = _command(home + 0.2, 0.5, home + 0.2, 0.5)
    follower.begin_blend(0.1)
    first = follower.send(target)
    np.testing.assert_allclose(first[:6], home, atol=1e-9)  # w = 0 at the first command
    clock.now += 0.05
    np.testing.assert_allclose(follower.send(target)[:6], home + 0.1, atol=1e-9)
    clock.now += 0.06
    np.testing.assert_allclose(follower.send(target)[:6], home + 0.2, atol=1e-9)


def test_hil_keyboard_events():
    class FakeKeyboardControl:
        ESC_HOLDOFF_S = 0.05

        def __init__(self):
            self._buffer = b""
            self._buffer_time = 0.0

    keyboard = hil_devices._make_keyboard(FakeKeyboardControl)
    keyboard._buffer = b"\x1b[Ce\n\x1bODqbx\x1b"
    keyboard._buffer_time = 1e12  # the trailing ESC is still within its hold-off
    assert keyboard._parse() == ["right", "end", "end", "left", "esc", "rezero"]
    assert keyboard._buffer == b"\x1b"
    keyboard._buffer_time = 0.0
    assert keyboard._parse() == ["esc"]


def test_hil_input_maps_arrows_to_hil_events(monkeypatch):
    fake_teleop = types.ModuleType("one_side_teleop")

    class FakeKeyboardControl:
        ESC_HOLDOFF_S = 0.05

        def __init__(self):
            self.enabled = True
            self._buffer = b""
            self._buffer_time = 0.0

        def poll(self):
            self._buffer = b"\x1b[C\x1b[De"
            return self._parse()

        def close(self):
            pass

    fake_teleop.KeyboardControl = FakeKeyboardControl
    monkeypatch.setitem(__import__("sys").modules, "one_side_teleop", fake_teleop)
    inputs = hil_devices.HilInput()
    inputs._pedal_events.put("correct")
    assert inputs.poll() == ["pause", "correct", "end", "correct"]


def test_stalled_loop_slows_the_alignment_instead_of_jumping():
    bot, clock = FakeBot(), FakeClock()
    leader = make_leader(bot, "left", control_dt=DT)
    target = bot.arm_position() + np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    leader.begin_move(target, 0.5, 0.0, clock.now)  # --align_s 0 is stretched to >= 1 s per rad
    clock.now += 5.0  # e.g. a blocking inference timed out right after the pause
    leader.tick(clock.now)
    first = np.array(bot.arm.published[-1])
    assert abs(first[0]) < 0.05
    run_ticks(leader, clock, 1.2)
    assert leader.state == ClutchLeader.LOCKED
    steps = np.abs(np.diff(np.array(bot.arm.published)[:, 0]))
    assert steps.max() < 0.05


def test_stalled_loop_stretches_the_gain_ramp():
    bot, clock = FakeBot(), FakeClock()
    leader = make_leader(bot, "left", release_ramp_s=0.4, control_dt=DT)
    leader.configure()
    leader.begin_release(clock.now)
    clock.now += 3.0
    leader.tick(clock.now)
    gains = [entry[4] for entry in bot.dxl.log if entry[0] == "reg"]
    assert gains == [720, 640]  # one step per tick, not straight to 0
    assert bot.torque


def test_lock_mid_ramp_re_aims_the_goal_before_stiffening():
    bot, clock = FakeBot(), FakeClock()
    leader = make_leader(bot, "left", release_ramp_s=0.4, control_dt=DT)
    leader.configure()
    leader.begin_release(clock.now)
    run_ticks(leader, clock, 0.1)
    bot.follow_commands = False
    bot.set_arm(bot.arm_position() - 0.05)  # sagging while soft
    bot.dxl.log.clear()
    leader.lock()
    kinds = [entry[0] for entry in bot.dxl.log]
    assert kinds.index("arm_goal") < kinds.index("reg") < kinds.index("torque")
    np.testing.assert_allclose(bot.dxl.log[kinds.index("arm_goal")][1], bot.arm_position())
    assert bot.dxl.p_gains == [800] * 6


def test_configure_refuses_gains_left_low_by_a_killed_run():
    bot = FakeBot()
    bot.dxl.p_gains = [0] * 6
    with pytest.raises(RuntimeError, match="Power-cycle"):
        make_leader(bot, "left").configure()


def test_keyboard_does_not_stall_on_esc_followed_by_another_key():
    class FakeKeyboardControl:
        ESC_HOLDOFF_S = 0.05

        def __init__(self):
            self._buffer = b""
            self._buffer_time = 0.0

    keyboard = hil_devices._make_keyboard(FakeKeyboardControl)
    keyboard._buffer = b"\x1b\x1b[Cxe\x1bxe"
    keyboard._buffer_time = 1e12
    assert keyboard._parse() == ["esc", "right", "end", "esc", "end"]
    assert keyboard._buffer == b""


def test_one_side_teleop_keyboard_no_longer_stalls_on_esc_plus_key():
    import one_side_teleop

    keyboard = one_side_teleop.KeyboardControl.__new__(one_side_teleop.KeyboardControl)
    keyboard._buffer = b"\x1b\x1b[Cq\x1bxr"
    keyboard._buffer_time = 1e12
    assert keyboard._parse() == ["esc", "right", "esc", "esc", "reset"]
    assert keyboard._buffer == b""


def test_release_never_blocks_the_loop_and_both_arms_soften_in_parallel():
    import time as _time

    from hil_devices import ClutchLeaderPair

    bots = (FakeBot(follow_commands=False), FakeBot(follow_commands=False))
    for bot in bots:
        bot.dxl.p_gains = [800, 800, 800, 640, 640, 640]  # mixed gains: one service call per joint
        original = bot.dxl.robot_set_motor_registers

        def slow(cmd_type, name, reg, value, original=original):
            _time.sleep(0.02)  # a ROS service round trip
            original(cmd_type, name, reg, value)

        bot.dxl.robot_set_motor_registers = slow
    pair = ClutchLeaderPair(*(ClutchLeader(bot, side, release_ramp_s=0.12, relative=False, control_dt=DT)
                              for bot, side in zip(bots, ("left", "right"))))
    pair.configure()
    start = _time.monotonic()
    pair.begin_release(start)
    slowest = 0.0
    while not pair.released:
        assert _time.monotonic() - start < 3.0
        t0 = _time.monotonic()
        pair.tick(t0)
        slowest = max(slowest, _time.monotonic() - t0)
        _time.sleep(DT)
    total = _time.monotonic() - start
    # per arm: 3 steps x 6 calls + torque + 6 restore calls ~ 0.5 s; one after the other would be ~1 s
    assert slowest < 0.015 and total < 0.85
    assert all(not bot.torque and bot.dxl.p_gains == [800, 800, 800, 640, 640, 640] for bot in bots)
    pair.lock()
    pair.close()
    assert all(bot.torque for bot in bots)


def test_blend_starts_from_the_last_command_not_the_sagged_measurement():
    bots = (FakeBot(follow_commands=False), FakeBot(follow_commands=False))
    clock = FakeClock()
    follower = FollowerCommander(bots, max_joint_step=None, clock=clock)
    home = bots[0].arm_position()
    held = _command(home + 0.03, 0.5, home + 0.03, 0.5)
    follower.send(held)  # the servos hold below this goal: measured stays at `home`
    follower.begin_blend(0.1)
    first = follower.send(_command(home + 0.2, 0.5, home + 0.2, 0.5))
    np.testing.assert_allclose(first, held, atol=1e-9)  # continuous goal: no sag
    clock.now += 0.05
    np.testing.assert_allclose(follower.send(_command(home + 0.2, 0.5, home + 0.2, 0.5))[:6], home + 0.115, atol=1e-9)
