"""The take-over preparation display: release ramp / steady-time progress of the leaders, its text and colours."""

import re
import sys
import types

import pytest
from fakes import FakeBot, FakeClock, SyncExecutor
from test_hil_session import make_session

import hil_rollout
from hil_devices import ClutchLeader, ClutchLeaderPair
from hil_rollout import build_parser, format_takeover_progress

DT = 0.02
BLOCKS = ("█", "░")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# green = the policy drives, red = the operator is live: neither may colour the preparation
FORBIDDEN_CODES = ("\033[92m", "\033[91m", "\033[1;92m", "\033[1;91m", "\033[32m", "\033[31m")


def plain(text):
    return ANSI.sub("", text)


def kwargs(**overrides):
    base = dict(release_ramp_s=0.4, settle_s=0.2, settle_timeout_s=3.0, control_dt=DT, io_executor=SyncExecutor())
    base.update(overrides)
    return base


def step(target, clock):
    clock.now += DT
    target.tick(clock.now)


def test_takeover_timing_defaults():
    args = build_parser().parse_args(["--task_name", "t", "--policy", "openpi", "--host", "h"])
    assert (args.release_ramp_s, args.settle_s, args.settle_timeout_s, args.settle_tolerance) == (0.5, 0.5, 2.0, 0.01)
    leader = ClutchLeader(FakeBot(), "left", io_executor=SyncExecutor())  # the class agrees with the command line
    assert (leader.release_ramp_s, leader.settle_s, leader.settle_timeout_s, leader.settle_tolerance) == (
        0.5, 0.5, 2.0, 0.01)


def test_progress_follows_the_release_ramp_then_fills_while_still_and_empties_on_motion():
    bot, clock = FakeBot(follow_commands=False), FakeClock()
    leader = ClutchLeader(bot, "left", **kwargs())
    leader.configure()
    leader.reference = bot.arm_position()
    assert leader.takeover_progress() is None  # locked

    leader.begin_release(clock.now)
    first = leader.takeover_progress()
    assert first["stage"] == "releasing" and first["steps"] == 10 and first["share"] < 0.2
    shares = []
    while leader.state == ClutchLeader.RELEASING:
        step(leader, clock)
        if leader.state == ClutchLeader.RELEASING:
            shares.append(leader.takeover_progress()["share"])
    assert shares == sorted(shares) and shares[-1] == 1.0

    # settling starts empty; holding still fills it
    assert leader.state == ClutchLeader.SETTLING
    assert leader.takeover_progress()["share"] == 0.0
    for _ in range(4):
        step(leader, clock)
    still = leader.takeover_progress()
    assert still["stage"] == "settling" and 0.2 < still["share"] < 0.6 and still["span"] == 0.0

    # one joint moves: the bar empties and names the joint
    bot.position[2] += 0.05
    step(leader, clock)
    moved = leader.takeover_progress()
    assert moved["share"] == 0.0 and moved["span"] == pytest.approx(0.05)
    assert moved["joint_name"] == "elbow" and moved["tolerance"] == 0.01 and moved["timeout_s"] == 3.0

    # still again: it refills, then the offset is taken and there is nothing left to show
    shares = []
    while leader.state == ClutchLeader.SETTLING:
        step(leader, clock)
        if leader.state == ClutchLeader.SETTLING:
            shares.append(leader.takeover_progress()["share"])
    assert shares == sorted(shares) and shares[-1] > 0.8
    assert leader.engaged and leader.takeover_progress() is None


def test_pair_reports_the_arm_furthest_from_ready():
    bots, clock = (FakeBot(follow_commands=False), FakeBot(follow_commands=False)), FakeClock()
    pair = ClutchLeaderPair(ClutchLeader(bots[0], "left", **kwargs()), ClutchLeader(bots[1], "right", **kwargs()))
    pair.configure()
    for arm, bot in zip(pair.arms, bots):
        arm.reference = bot.arm_position()
    assert pair.takeover_progress() is None

    # the left leader is ahead (settling); the right one has only just started releasing
    pair.arms[0].begin_release(clock.now)
    for _ in range(28):
        step(pair, clock)
    assert pair.arms[0].state == ClutchLeader.SETTLING
    pair.arms[1].begin_release(clock.now)
    shot = pair.takeover_progress()
    assert (shot["side"], shot["stage"]) == ("right", "releasing")

    # the left leader is done first; the right one is what is left to wait for
    while pair.arms[1].state != ClutchLeader.SETTLING:
        step(pair, clock)
    assert pair.arms[0].state == ClutchLeader.FREE
    shot = pair.takeover_progress()
    assert (shot["side"], shot["stage"]) == ("right", "settling")
    while not pair.released:
        step(pair, clock)
    assert pair.takeover_progress() is None


def test_pair_reports_the_arm_with_less_steady_time_when_both_settle():
    bots, clock = (FakeBot(follow_commands=False), FakeBot(follow_commands=False)), FakeClock()
    pair = ClutchLeaderPair(ClutchLeader(bots[0], "left", **kwargs()), ClutchLeader(bots[1], "right", **kwargs()))
    pair.configure()
    for arm, bot in zip(pair.arms, bots):
        arm.reference = bot.arm_position()
    pair.begin_release(clock.now)
    while not all(arm.state == ClutchLeader.SETTLING for arm in pair.arms):
        step(pair, clock)
    for index in range(6):  # the left hand is still, the right one keeps shaking
        bots[1].position[0] += 0.05 if index % 2 == 0 else -0.05
        step(pair, clock)
    shot = pair.takeover_progress()
    assert shot["side"] == "right" and shot["share"] == 0.0
    assert pair.arms[0].takeover_progress()["share"] > 0.0


SETTLING = dict(stage="settling", side="right", share=0.4, steady_s=0.08, need_s=0.2, span=0.014, tolerance=0.01,
                joint_name="elbow", elapsed_s=0.4, timeout_s=1.0)


def test_settling_text_shows_the_numbers_and_names_the_joint_only_while_over_tolerance():
    assert plain(format_takeover_progress(SETTLING, BLOCKS)) == (
        "settling [████░░░░░░] steady 0.08/0.20s | move 0.014/0.010 rad (R elbow) | wait 0.4/1.0s")
    calm = dict(SETTLING, share=0.75, steady_s=0.15, span=0.004)
    assert plain(format_takeover_progress(calm, BLOCKS)) == (
        "settling [████████░░] steady 0.15/0.20s | move 0.004/0.010 rad | wait 0.4/1.0s")


def test_releasing_text_shows_the_ramp_steps():
    shot = dict(stage="releasing", side="left", share=0.5, done=6, steps=12)
    assert plain(format_takeover_progress(shot, BLOCKS)) == "releasing [█████░░░░░] 6/12"
    assert plain(format_takeover_progress(dict(shot, share=0.0, done=0, steps=0), BLOCKS)) == "releasing [░░░░░░░░░░]"


def test_preparation_text_never_uses_the_policy_or_takeover_colours():
    shots = [SETTLING, dict(SETTLING, span=0.002), dict(stage="releasing", side="left", share=0.5, done=6, steps=12)]
    for shot in shots:
        text = format_takeover_progress(shot, BLOCKS)
        assert not any(code in text for code in FORBIDDEN_CODES)


def test_bar_falls_back_to_ascii_when_stdout_cannot_print_blocks(monkeypatch):
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(encoding="ascii"))
    assert hil_rollout._bar_chars() == ("#", "-")
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(encoding="utf-8"))
    assert hil_rollout._bar_chars() == BLOCKS


class FakeBar:
    """Just enough of tqdm to see what the session draws and in which colour."""

    def __init__(self):
        self.colour = None
        self.drawn = []  # (plain text, refresh, colour)

    def set_postfix_str(self, text, refresh=True):
        self.drawn.append((plain(text), refresh, self.colour))

    def update(self, n=1):
        pass

    def write(self, message):
        pass

    def close(self):
        pass


def test_session_redraws_the_preparation_line_in_yellow_until_the_correction_starts():
    session, puppets, masters, recorder, executor = make_session({10: ["pause"], 12: ["correct"], 80: ["end"]}, 200)
    session._bar = bar = FakeBar()
    session.run()

    preparing = [entry for entry in bar.drawn if entry[0].startswith("paused | releasing")
                 or entry[0].startswith("paused | settling")]
    # both stages are redrawn immediately (a paused bar has no update() to do it)...
    redrawn = [text for text, refresh, _ in preparing if refresh]
    assert any("releasing" in text for text in redrawn) and any("settling" in text for text in redrawn)
    # ...and everything drawn during the preparation keeps the paused colour
    assert all(colour == "#FFFF00" for _, _, colour in preparing)
    assert all("infer" not in text for text, _, _ in preparing)
    # at most one immediate redraw per PREPARE_REFRESH_S
    assert len(redrawn) <= 1 + int(session.steps * DT / hil_rollout.PREPARE_REFRESH_S)
    # once the correction runs the line is the red correcting line again, without any preparation text
    first_red = next(i for i, (_, _, colour) in enumerate(bar.drawn) if colour == "#FF0000")
    assert all("releasing" not in text and "settling" not in text for text, _, _ in bar.drawn[first_red:])
    assert "correcting" in bar.drawn[first_red][0]
