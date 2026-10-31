import numpy as np
from fakes import FakeBot, FakeClock, SyncExecutor

from chunk_executor import PolicyError
from hil_devices import ClutchLeader, ClutchLeaderPair, FollowerCommander
from hil_recorder import PHASE_AUTONOMOUS, PHASE_CORRECTING, PHASE_PAUSED
from hil_rollout import HilSession, StaleStream

DT = 0.02


class ScriptedInputs:
    """Events keyed by tick index."""

    def __init__(self, script):
        self.script = dict(script)
        self.tick = 0

    def poll(self):
        events = self.script.get(self.tick, [])
        self.tick += 1
        return list(events)


class FakeExecutor:
    def __init__(self, base, fail_at=None):
        self.base = np.asarray(base, dtype=np.float64)
        self.calls = 0
        self.resets = 0
        self.fail_at = fail_at

    def step(self, frame_provider):
        frame_provider()
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise PolicyError("server crashed")
        action = self.base.copy()
        action[0] += 0.001 * self.calls  # policy slowly turns the left waist
        return action

    def reset(self):
        self.resets += 1

    def stats(self):
        return "fake"


class ListRecorder:
    def __init__(self):
        self.rows = []

    def __len__(self):
        return len(self.rows)

    def add(self, frame, action, phase, intervention, leader_action):
        self.rows.append((np.array(action), phase, bool(intervention), np.array(leader_action)))


def make_session(script, max_steps, executor=None, capture=None):
    clock = FakeClock()
    puppets = (FakeBot(), FakeBot())
    masters = (FakeBot(gripper=0.7), FakeBot(gripper=0.7))
    kwargs = dict(release_ramp_s=0.08, settle_s=0.06, settle_timeout_s=0.3, control_dt=DT, io_executor=SyncExecutor())
    leaders = ClutchLeaderPair(ClutchLeader(masters[0], "left", **kwargs), ClutchLeader(masters[1], "right", **kwargs))
    leaders.configure()
    follower = FollowerCommander(puppets, max_joint_step=0.1, clock=clock)
    executor = executor or FakeExecutor(follower.measured())
    recorder = ListRecorder()
    session = HilSession(follower, leaders, executor, capture or (lambda fresh: object()), recorder,
                         ScriptedInputs(script), DT, max_steps, align_s=0.1, correction_blend_s=0.04,
                         correction_skip_s=0.1, clock=clock, sleep=clock.sleep, show_progress=False)
    return session, puppets, masters, recorder, executor


def test_full_takeover_cycle():
    # tick 10: pause, 12: take over (while the leaders still align), 60: hand back, 70: resume, 90: end
    script = {10: ["pause"], 12: ["correct"], 60: ["correct"], 70: ["pause"], 90: ["end"]}
    session, puppets, masters, recorder, executor = make_session(script, 200)

    # operator: the released left leader sags 0.05 rad on every joint before settling, then the
    # operator moves its shoulder by +0.2 over ticks 40..50
    original_tick = session.tick
    sagged = []

    def tick_with_operator():
        if session.leaders.arms[0].state == ClutchLeader.SETTLING and not sagged:
            masters[0].position[:6] = list(np.array(masters[0].position[:6]) + 0.05)
            sagged.append(session.steps)
        if session.leaders.released and 40 <= session.steps < 50:
            masters[0].position[1] += 0.02
        return original_tick()

    session.tick = tick_with_operator
    assert session.run() == "end"
    phases = [row[1] for row in recorder.rows]
    assert phases[:10] == [PHASE_AUTONOMOUS] * 10
    assert phases[10] == PHASE_PAUSED
    first_correction = phases.index(PHASE_CORRECTING)
    # aligning (0.1 s) + release ramp (0.08 s) + settle (>= 0.06 s) happen while paused
    assert first_correction - 10 >= 12
    assert phases[60] == PHASE_PAUSED and phases[70] == PHASE_AUTONOMOUS and len(phases) == 90

    # the leaders were driven onto the followers' pose before the release
    hold = recorder.rows[10][0]
    assert any(np.allclose(goal, hold[7:13]) for goal in masters[1].arm.published)
    assert any(np.allclose(goal, hold[0:6]) for goal in masters[0].arm.published)
    assert sagged

    # clutch: the sag is absorbed by the offset - the follower starts from where it held and
    # follows only the operator's motion
    corrections = [row for row in recorder.rows if row[1] == PHASE_CORRECTING]
    np.testing.assert_allclose(corrections[0][0][:6], hold[:6], atol=1e-9)
    np.testing.assert_allclose(corrections[-1][0][1] - hold[1], 0.2, atol=1e-6)
    np.testing.assert_allclose(corrections[-1][0][7:13], hold[7:13], atol=1e-9)
    # leader_action is the absolute reading, so leader - action is the constant clutch offset
    offsets = np.array([row[3][:6] - row[0][:6] for row in corrections])
    np.testing.assert_allclose(offsets, 0.05, atol=1e-6)

    # the first correction_skip_s (5 ticks) is not labelled as intervention
    labels = [row[2] for row in corrections]
    assert labels[:5] == [False] * 5 and all(labels[5:])
    assert session.corrections == 1
    assert executor.resets >= 2  # pause and resume
    assert session.leaders.locked  # locked again on the way out


def test_policy_error_pauses_and_hands_over_to_the_operator():
    session, puppets, masters, recorder, executor = make_session({}, 40)
    session.executor = executor = FakeExecutor(session.follower.measured(), fail_at=5)
    session.run()
    phases = [row[1] for row in recorder.rows]
    assert phases[:4] == [PHASE_AUTONOMOUS] * 4 and set(phases[4:]) == {PHASE_PAUSED}
    assert executor.calls == 5


def test_stale_stream_pauses_autonomous_and_skips_recording_while_paused():
    calls = {"n": 0}

    def capture(fresh):
        calls["n"] += 1
        if 5 <= calls["n"] <= 8:
            raise StaleStream("cam_high image is 1.00s old")
        return object()

    session, puppets, masters, recorder, executor = make_session({}, 20, capture=capture)
    session.run()
    assert session.skipped_frames == 4
    # paused from tick 4 on: paused time does not count, so the 4x total cap (80 ticks) ends it
    assert session.steps == 80 and session.active_steps == 4 and len(recorder) == 76
    assert recorder.rows[4][1] == PHASE_PAUSED


def test_invalid_keys_are_ignored():
    # LEFT while autonomous, RIGHT while correcting are refused; ESC ends
    script = {2: ["correct"], 4: ["pause"], 5: ["correct"], 30: ["pause"], 35: ["esc"]}
    session, puppets, masters, recorder, executor = make_session(script, 100)
    assert session.run() == "esc"
    phases = [row[1] for row in recorder.rows]
    assert phases[2] == PHASE_AUTONOMOUS and phases[30] == PHASE_CORRECTING and phases[34] == PHASE_CORRECTING
    assert session.leaders.locked


def test_failed_blocking_inference_does_not_make_the_leader_jump():
    session, puppets, masters, recorder, executor = make_session({}, 80)
    clock = session.clock

    class BlockingFailure(FakeExecutor):
        def step(self, frame_provider):
            self.calls += 1
            if self.calls == 3:
                clock.now += 10.0  # sync inference blocked until its timeout
                raise PolicyError("no reply")
            return self.base.copy()

    hold = session.follower.measured()
    hold[0] += 1.0  # the followers hold 1 rad away from the leaders' current pose
    session.executor = BlockingFailure(hold)
    session.follower.max_joint_step = None
    session.run()
    goals = np.array(masters[0].arm.published)
    assert np.abs(np.diff(goals[:, 0])).max() < 0.05  # smooth alignment, no single-command jump
    np.testing.assert_allclose(goals[-1, 0], hold[0], atol=1e-6)


def test_paused_time_does_not_count_against_the_limit():
    # 10 policy ticks, then a long pause (60 ticks), resume: the episode ends after 20 active ticks
    session, puppets, masters, recorder, executor = make_session({10: ["pause"], 70: ["pause"]}, 20)
    assert session.run() == "max_steps"
    phases = [row[1] for row in recorder.rows]
    assert phases.count(PHASE_PAUSED) == 60 and phases.count(PHASE_AUTONOMOUS) == 20


def test_limit_reached_while_correcting_ends_on_hand_back():
    script = {3: ["pause"], 4: ["correct"], 120: ["correct"]}
    session, puppets, masters, recorder, executor = make_session(script, 40)
    assert session.run() == "max_steps"
    phases = [row[1] for row in recorder.rows]
    assert session.active_steps > 40  # the correction was not cut at the limit
    assert phases[-1] == PHASE_CORRECTING and len(phases) == 120  # ended right at the hand-back
    assert session.leaders.locked
