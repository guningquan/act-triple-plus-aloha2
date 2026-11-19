"""run_vtla_fused.py and vtla_fused_runtime.py without a robot or a policy server."""

import argparse
import logging
import sys
import threading
import types
from types import SimpleNamespace

import numpy as np
import pytest

from vtla_fused_runtime import AsyncProbe, TactileHistory, TakeoverMonitor


class SyncThread:
    """threading.Thread stand-in that runs the target inside start() (deterministic probes)."""

    def __init__(self, target, args=(), daemon=None):
        self._target, self._args = target, args

    def start(self):
        self._target(*self._args)

    def is_alive(self):
        return False

    def join(self):
        pass


# ----------------------------------------------------------------------------------------------- TactileHistory

def test_history_before_the_start_is_the_first_frame():
    h = TactileHistory(maxlen=8)
    assert h.frame(0) is None
    h.push(0, {"v": 0})
    h.push(1, {"v": 1})
    assert h.frame(-8) == {"v": 0}          # t - k < 0: the episode's first frame (the data loader's clamp)
    assert h.frame(1) == {"v": 1}


def test_history_missing_step_is_the_nearest_earlier_one():
    h = TactileHistory(maxlen=8)
    for s in (0, 1, 2, 4, 5):               # step 3: snapshot failed
        h.push(s, {"v": s})
    assert h.frame(3) == {"v": 2}
    assert h.frame(9) == {"v": 5}


def test_history_drops_old_steps_and_falls_back_to_the_oldest_stored():
    h = TactileHistory(maxlen=3)
    for s in range(10):
        h.push(s, {"v": s})
    assert h.frame(7) == {"v": 7}
    assert h.frame(2) == {"v": 7}           # dropped: the oldest stored frame, not the episode start
    assert h.frame(-1) == {"v": 0}          # before the episode start: still the first frame


# ----------------------------------------------------------------------------------------------- AsyncProbe

class ProbeClient:
    def __init__(self, fresh_at=(), replanned_at=(), fail=False):
        self.calls = []
        self.fresh_at, self.replanned_at, self.fail = set(fresh_at), set(replanned_at), fail

    def seg_step(self, k, p, tacs, hist, force=False, codec="jpg", tactile_size=None):
        self.calls.append((k, p, tacs, hist, codec))
        if self.fail:
            raise RuntimeError("server down")
        return {"status": "ok", "k": k, "p": p, "fresh": k in self.fresh_at, "replanned": k in self.replanned_at,
                "e": 0.1 * k, "delta": 0.5, "lag": k, "n_reread": 0, "ms": 3.0}


def test_probe_schedule_fresh_flag_and_chunk_ids():
    client = ProbeClient(fresh_at={10})
    probe = AsyncProbe(client, every=5, thread_factory=SyncThread)
    probe.start_chunk()
    for k in range(1, 13):
        probe.maybe_probe(k, k, {"t": k}, {8: {"t": k - 8}})
    assert [c[0] for c in client.calls] == [5, 10]
    assert client.calls[0][3] == {8: {"t": -3}} and client.calls[0][4] == "png"
    assert probe.fresh()
    replies = probe.drain()
    assert [(c, k, r["fresh"]) for c, k, r in replies] == [(0, 5, False), (0, 10, True)]
    assert probe.drain() == []
    probe.start_chunk()                     # a new chunk clears the flag
    assert not probe.fresh() and probe.chunk == 1


def test_probe_in_flight_is_not_doubled():
    gate = threading.Event()

    class SlowClient(ProbeClient):
        def seg_step(self, *a, **kw):
            gate.wait(5)
            return super().seg_step(*a, **kw)

    client = SlowClient()
    probe = AsyncProbe(client, every=5)
    probe.start_chunk()
    assert probe.maybe_probe(5, 5, {}, {})
    assert not probe.maybe_probe(10, 10, {}, {})    # the first one is still waiting for the server
    gate.set()
    probe.finish()
    assert probe.maybe_probe(15, 15, {}, {})
    probe.finish()
    assert [c[0] for c in client.calls] == [5, 15]


def test_probe_errors_never_raise():
    probe = AsyncProbe(ProbeClient(fail=True), every=1, thread_factory=SyncThread)
    probe.start_chunk()
    probe.maybe_probe(1, 1, {}, {})
    assert probe.errors == 1 and not probe.fresh()
    assert probe.drain()[0][2] == {"error": "server down"}


def test_probe_warns_once_when_the_server_regenerates(caplog):
    probe = AsyncProbe(ProbeClient(replanned_at={1, 2}), every=1, thread_factory=SyncThread)
    probe.start_chunk()
    with caplog.at_level(logging.WARNING):
        probe.maybe_probe(1, 1, {}, {})
        probe.maybe_probe(2, 2, {}, {})
    assert sum("IGNORED" in r.message for r in caplog.records) == 1
    assert not probe.fresh()


# ----------------------------------------------------------------------------------------------- TakeoverMonitor

def test_takeover_needs_limit_events_inside_the_window():
    m = TakeoverMonitor(limit=3, window=10)
    assert not m.event(0) and not m.event(5)
    assert m.event(9)
    m.clear()
    assert not m.event(20) and not m.event(25)
    assert not m.event(40)                  # 20 and 25 left the window
    assert not TakeoverMonitor(limit=0, window=10).event(0)


# ----------------------------------------------------------------------------------------------- run_vtla_fused

@pytest.fixture
def fused():
    pytest.importorskip("zmq")
    pytest.importorskip("msgpack")
    try:
        import dm_env  # noqa: F401  (real_env.py imports it; only its methods use it)
    except ImportError:
        sys.modules["dm_env"] = types.ModuleType("dm_env")
    import run_vtla_fused
    return run_vtla_fused


ARM = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]


def joint_states(arm, gripper_joint, finger):
    return SimpleNamespace(name=ARM + ["gripper", "left_finger", "right_finger"],
                           position=list(arm) + [gripper_joint, finger, -finger], velocity=[0.0] * 9)


class FakeArm:
    def __init__(self):
        self.group_info = SimpleNamespace(joint_names=list(ARM),
                                          joint_lower_limits=[-3.0, -1.8, -1.6, -3.0, -1.8, -3.0],
                                          joint_upper_limits=[3.0, 1.2, 1.6, 3.0, 2.2, 3.0])
        self.robot_des = SimpleNamespace(M=np.eye(4), Slist=np.eye(6))
        self.goals = []

    def set_trajectory_time(self, moving_time=None, accel_time=None):
        self.trajectory = (moving_time, accel_time)

    def set_joint_positions(self, positions, moving_time=None, accel_time=None, blocking=True):
        self.goals.append(list(positions))


class FakePuppet:
    def __init__(self):
        self.arm = FakeArm()
        self.dxl = SimpleNamespace(joint_states=joint_states([0.0, -0.96, 1.16, 0.0, -0.3, 0.0], 0.0767, 0.0298))


class FakeEnv:
    def __init__(self):
        self.puppet_bot_left, self.puppet_bot_right = FakePuppet(), FakePuppet()
        self.grippers = []

    def set_gripper_pose(self, left, right):
        self.grippers.append((left, right))


class FakeStreams:
    """snapshot() n -> every frame filled with n (so a frame tells which snapshot it came from)."""

    def __init__(self, sources):
        self.sources = sources
        self.tactile_names = sorted(s for k, s in sources.items() if k.endswith("_tactile"))
        self.n = 0

    def snapshot(self, max_age):
        self.n += 1
        return {s: np.full((4, 6, 3), self.n % 256, np.uint8) for s in self.sources.values()}


class FakeKeyboard:
    """Events become available after the n-th drain() (run_episode drains once at the start)."""

    def __init__(self, enabled=False, events=(), after_drain=2):
        self.enabled, self.events, self.after_drain, self.drains = enabled, list(events), after_drain, 0

    def drain(self):
        self.drains += 1

    def poll(self):
        if self.drains >= self.after_drain and self.events:
            return [self.events.pop(0)]
        return []


class FakeClient:
    def __init__(self, fresh_calls=(), horizon=50):
        self.predicts, self.probes, self.resets = [], [], 0
        self.fresh_calls, self.horizon = set(fresh_calls), horizon
        self.last_infer_time_ms = 100.0

    def reset(self):
        self.resets += 1

    def predict(self, state, prompt, cams, tacs, tacs_hist=None, codec="png"):
        self.predicts.append({"state": np.asarray(state), "cams": cams, "tacs": tacs, "hist": tacs_hist})
        actions = np.zeros((self.horizon, 14))
        actions[:, 0:6] = [0.1, -0.9, 1.1, 0.0, -0.3, 0.0]
        actions[:, 7:13] = [0.1, -0.9, 1.1, 0.0, -0.3, 0.0]
        actions[:, 6], actions[:, 13] = 0.5, 1.4    # 1.4: clipped to 1
        return actions

    def seg_step(self, k, p, tacs, hist, force=False, codec="jpg", tactile_size=None):
        self.probes.append((k, p, hist))
        return {"status": "ok", "fresh": len(self.probes) in self.fresh_calls, "replanned": False, "n_reread": 0}


def make_args(fused, **over):
    a = argparse.Namespace(data="hdf5", fps=None, state_gripper=None, hist=None, prompt="plug it in",
                           action_horizon=50, max_episode_steps=60, gripper_moving_time=None, max_stream_age=0.5,
                           codec="png", probe_every=0, probe_codec="png", takeover_after=0, takeover_window=None,
                           refine=False, refine_codec="png", refine_timeout=2.0, refine_every=1, refine_max_delta=0.0,
                           refine_max_grip_delta=0.0, refine_stop=None)
    vars(a).update(over)
    return fused.resolve_args(a)


def run(fused, client, args, keyboard=None, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr(fused, "AsyncProbe",
                            lambda c, every, codec, **kw: AsyncProbe(c, every, codec, thread_factory=SyncThread, **kw))
    sources = fused.video_sources("cam_high", fused.DEFAULT_TACTILE_MAP)
    env, streams, log = FakeEnv(), FakeStreams(sources), []
    clock = SimpleNamespace(t=0.0)
    outcome = fused.run_episode(client, env, streams, sources, args, keyboard or FakeKeyboard(), log.append,
                                clock=lambda: clock.t, sleep=lambda s: setattr(clock, "t", clock.t + s))
    return outcome, env, streams, log


def test_presets(fused):
    a = make_args(fused)
    assert (a.fps, a.state_gripper, a.hist) == (50.0, "finger", 8)
    b = make_args(fused, data="vtla")
    assert (b.fps, b.state_gripper, b.hist) == (30.0, "joint", 5)
    c = make_args(fused, data="vtla", fps=60.0, state_gripper="finger", hist=7)
    assert (c.fps, c.state_gripper, c.hist) == (60.0, "finger", 7)


def test_state_gripper_conventions(fused):
    from constants import (PUPPET_GRIPPER_POSITION_CLOSE, PUPPET_GRIPPER_POSITION_OPEN,
                           PUPPET_LEFT_GRIPPER_JOINT_CLOSE, PUPPET_LEFT_GRIPPER_JOINT_OPEN,
                           PUPPET_RIGHT_GRIPPER_JOINT_CLOSE, PUPPET_RIGHT_GRIPPER_JOINT_OPEN)
    cal = [(PUPPET_LEFT_GRIPPER_JOINT_CLOSE, PUPPET_LEFT_GRIPPER_JOINT_OPEN),
           (PUPPET_RIGHT_GRIPPER_JOINT_CLOSE, PUPPET_RIGHT_GRIPPER_JOINT_OPEN)]
    arm = [0.01, -1.0, 1.2, 0.0, -0.2, 0.01]
    left, right = joint_states(arm, 0.0767, 0.0298), joint_states(arm, -0.5, 0.0200)
    models = (None, None)
    joint = fused.observation_state("joint", left, right, ARM, ARM, cal, models)
    finger = fused.observation_state("finger", left, right, ARM, ARM, cal, models)
    assert joint.shape == (32,) and not joint[14:].any()
    np.testing.assert_allclose(joint[0:6], arm)
    np.testing.assert_allclose(finger[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]], joint[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]])
    assert joint[6] == pytest.approx(1.0)  # open gripper joint -> 1
    span = PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
    assert finger[6] == pytest.approx((0.0298 - PUPPET_GRIPPER_POSITION_CLOSE) / span)   # ~0.29, as aloha_insertion_v1
    assert finger[13] == pytest.approx((0.0200 - PUPPET_GRIPPER_POSITION_CLOSE) / span)
    assert joint[13] == pytest.approx((-0.5 - PUPPET_RIGHT_GRIPPER_JOINT_CLOSE)
                                      / (PUPPET_RIGHT_GRIPPER_JOINT_OPEN - PUPPET_RIGHT_GRIPPER_JOINT_CLOSE))


def test_episode_without_probes_executes_whole_chunks(fused):
    client = FakeClient()
    outcome, env, streams, log = run(fused, client, make_args(fused, max_episode_steps=120))
    assert outcome == "done" and client.resets == 1
    assert len(client.predicts) == 3                     # 50 + 50 + 20 steps
    assert len(env.puppet_bot_left.arm.goals) == 120 and len(env.grippers) == 120
    assert env.grippers[0] == (0.5, 1.0)                 # gripper clipped to [0, 1]
    first, second = client.predicts[0], client.predicts[1]
    assert set(first["tacs"]) == set(fused.DEFAULT_TACTILE_MAP)
    assert set(first["cams"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    assert first["hist"][8]["left_wrist_left_tactile"][0, 0, 0] == 1    # t - 8 < 0: the first frame (snapshot 1)
    # chunk 2 starts at step 50 (snapshot 52: 1 + 50 steps + its own); t - 8 = step 42 = snapshot 43
    assert second["tacs"]["left_wrist_left_tactile"][0, 0, 0] == 52
    assert second["hist"][8]["left_wrist_left_tactile"][0, 0, 0] == 43
    assert first["state"][6] == pytest.approx(fused.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(0.0298))   # finger state
    assert [r["event"] for r in log].count("chunk") == 3 and log[-1]["event"] == "end" and log[-1]["steps"] == 120


def test_fresh_reply_ends_the_chunk(fused, monkeypatch):
    client = FakeClient(fresh_calls={2})                 # the 2nd probe (k = 10 of chunk 1) answers "fresh"
    args = make_args(fused, probe_every=5, max_episode_steps=60)
    outcome, env, streams, log = run(fused, client, args, monkeypatch=monkeypatch)
    assert outcome == "done"
    assert [k for k, _, _ in client.probes][:2] == [5, 10]
    fresh = [r for r in log if r["event"] == "fresh"]
    assert len(fresh) == 1 and fresh[0]["step"] == 11    # seen at the next step, chunk 1 ended after 11 steps
    chunks = [r for r in log if r["event"] == "chunk"]
    assert [c["step"] for c in chunks] == [0, 11]
    assert client.predicts[1]["hist"][8]["left_wrist_left_tactile"][0, 0, 0] == 4    # step 3 = snapshot 4
    probes = [r for r in log if r["event"] == "probe"]
    assert probes[0]["chunk"] == 0 and probes[0]["k"] == 5 and probes[1]["fresh"] is True
    assert client.probes[0][2][8]["left_wrist_left_tactile"][0, 0, 0] == 1    # probe k=5 at step 5: t-8 < 0


def test_persistent_mismatch_asks_the_operator(fused, monkeypatch):
    client = FakeClient(fresh_calls={1})
    keyboard = FakeKeyboard(enabled=True, events=["right"])
    args = make_args(fused, probe_every=5, takeover_after=1, max_episode_steps=30)
    outcome, env, streams, log = run(fused, client, args, keyboard=keyboard, monkeypatch=monkeypatch)
    events = [r["event"] for r in log]
    assert outcome == "done" and "takeover_request" in events and "takeover_resume" in events
    assert len(client.predicts) == 2                     # resumed: a new chunk from the live observation

    client = FakeClient(fresh_calls={1})
    keyboard = FakeKeyboard(enabled=True, events=["left"])
    outcome, env, streams, log = run(fused, client, args, keyboard=keyboard, monkeypatch=monkeypatch)
    assert outcome == "reset" and len(client.predicts) == 1


# ----------------------------------------------------------------------------------------------- --refine

import json  # noqa: E402

from vtla_fused_runtime import AsyncRefine, ChunkPlan  # noqa: E402


def make_plan(h=50):
    rng = np.random.default_rng(1)
    plan = rng.normal(scale=0.3, size=(h, 14))
    plan[:, [6, 13]] = 0.5
    return plan


def test_chunkplan_never_rewrites_executed_rows():
    plan = make_plan()
    cp = ChunkPlan(3, plan, 50)
    for h in range(5):
        np.testing.assert_array_equal(cp.take(h), plan[h])
    info = cp.apply(3, 2, plan + 0.2)                   # sent after 2 steps, read after 5: d = 3
    assert (info["d"], info["start"], info["n_written"]) == (3, 5, 45)
    np.testing.assert_array_equal(cp.rows[:5], plan[:5])
    np.testing.assert_allclose(cp.rows[5:], plan[5:] + 0.2)
    np.testing.assert_array_equal(cp.executed(), plan[:5])
    assert info["delta_arm_max"] == pytest.approx(0.2) and info["delta_grip_max"] == pytest.approx(0.2)
    with pytest.raises(AssertionError):
        cp.take(7)                                      # rows are taken strictly in order
    with pytest.raises(ValueError):
        cp.plan[0, 0] = 1.0                             # the original plan is read-only


def test_chunkplan_corrections_are_relative_to_the_plan():
    plan = make_plan()
    cp = ChunkPlan(0, plan, 50)
    cp.take(0)
    cp.apply(0, 1, plan + 0.1)
    for h in (1, 2):
        cp.take(h)
    info = cp.apply(0, 3, plan + 0.05)
    assert info["start"] == 3 and info["d"] == 0
    np.testing.assert_allclose(cp.rows[3:], plan[3:] + 0.05)       # not + 0.15
    np.testing.assert_allclose(cp.executed()[1:3], plan[1:3] + 0.1)  # what was commanded stays
    np.testing.assert_array_equal(cp.executed()[0], plan[0])


def test_chunkplan_skips_stale_out_of_order_late_and_bad_replies():
    plan = make_plan()
    cp = ChunkPlan(5, plan, 20)                         # n_exec < H: only rows [0, 20) are ever commanded
    cp.take(0)
    assert cp.apply(4, 1, plan + 1)["skipped"] == "stale plan"
    assert cp.apply(5, 1, plan + 0.1)["start"] == 1
    assert cp.apply(5, 1, plan + 9)["skipped"] == "out of order"
    nan = plan + 9
    nan[10, 2] = np.nan                                  # NaN in an executable row: the whole reply is dropped
    assert cp.apply(5, 2, nan)["skipped"] == "bad actions"
    assert cp.apply(5, 3, plan[:10] + 9)["skipped"] == "bad actions"   # too few rows
    assert cp.apply(5, 4, (plan + 9)[:, :7])["skipped"] == "bad actions"
    assert cp.apply(5, 5, None)["skipped"] == "bad actions"
    np.testing.assert_allclose(cp.rows[1:20], plan[1:20] + 0.1)      # still the last good correction
    np.testing.assert_array_equal(cp.rows[20:], plan[20:])
    for h in range(1, 20):
        cp.take(h)
    late = cp.apply(5, 19, plan + 9)
    assert late["skipped"] == "no steps left" and late["d"] == 1
    assert ChunkPlan(1, plan, 80).n_exec == 50


def test_chunkplan_caps_arms_and_grippers_separately():
    plan = make_plan()
    cp = ChunkPlan(0, plan, 50, max_delta=0.05, max_grip_delta=0.2)
    cp.take(0)
    corrected = plan.copy()
    corrected[:, 0] += 0.3                               # arm: clipped to 0.05
    corrected[:, 8] -= 0.01                              # arm: within the cap
    corrected[:, 6] += 0.1                               # gripper: within its cap
    corrected[:, 13] -= 0.5                              # gripper: clipped to 0.2
    info = cp.apply(0, 1, corrected)
    np.testing.assert_allclose(cp.rows[1:, 0], plan[1:, 0] + 0.05)
    np.testing.assert_allclose(cp.rows[1:, 8], plan[1:, 8] - 0.01)
    np.testing.assert_allclose(cp.rows[1:, 6], plan[1:, 6] + 0.1)
    np.testing.assert_allclose(cp.rows[1:, 13], plan[1:, 13] - 0.2)
    assert info["clipped"] == 2 * 49 and info["delta_arm_max"] == pytest.approx(0.05)
    assert info["delta_grip_max"] == pytest.approx(0.2)
    assert info["jump"] == pytest.approx(np.abs(cp.rows[1, list(range(6)) + list(range(7, 13))]
                                                - plan[0, list(range(6)) + list(range(7, 13))]).max())
    json.dumps(info)
    for out in (cp.apply(9, 2, plan), cp.apply(0, 1, plan), cp.apply(0, 3, None)):
        json.dumps(out)


class RefineProbeClient:
    def __init__(self, fresh_at=(), stale_at=(), fail=False):
        self.calls = []
        self.fresh_at, self.stale_at, self.fail = set(fresh_at), set(stale_at), fail

    def refine(self, plan_id, m, executed, tacs, hist, state=None, force=False, codec="jpg", timeout_s=None,
               tactile_size=None):
        self.calls.append({"plan": plan_id, "m": m, "executed": executed, "state": state, "codec": codec,
                           "timeout_s": timeout_s})
        if self.fail:
            raise TimeoutError("no reply")
        fresh, stale = m in self.fresh_at, m in self.stale_at
        return {"status": "ok", "plan": plan_id, "m": m, "stale": stale, "fresh": fresh,
                "actions": None if fresh or stale else np.zeros((50, 14)), "pad": 1 if fresh else None}


def test_async_refine_drain_fresh_and_chunks():
    client = RefineProbeClient(fresh_at={4}, stale_at={5})
    ref = AsyncRefine(client, "jpg", 1.5, thread_factory=SyncThread)
    assert not ref.maybe_refine(1, np.zeros((1, 14)), {}, {})        # no plan yet
    ref.start_chunk(11)
    assert not ref.maybe_refine(0, np.zeros((0, 14)), {}, {})
    assert ref.maybe_refine(3, np.zeros((3, 14)), {}, {}, state=np.ones(32))
    assert not ref.fresh()
    ref.maybe_refine(5, np.zeros((5, 14)), {}, {})                    # stale replies never end the chunk
    assert not ref.fresh()
    ref.maybe_refine(4, np.zeros((4, 14)), {}, {})
    assert ref.fresh() and ref.sent == 3
    out = ref.drain()
    assert [(c, p, m) for c, p, m, _ in out] == [(0, 11, 3), (0, 11, 5), (0, 11, 4)]
    assert out[0][3]["actions"].shape == (50, 14) and "rtt_ms" in out[0][3]
    assert client.calls[0]["timeout_s"] == 1.5 and client.calls[0]["codec"] == "jpg"
    assert client.calls[0]["state"].shape == (32,)
    assert ref.drain() == []
    ref.start_chunk(12)
    assert not ref.fresh() and ref.chunk == 1 and ref.plan_id == 12


def test_async_refine_one_in_flight():
    gate = threading.Event()

    class SlowClient(RefineProbeClient):
        def refine(self, *a, **kw):
            gate.wait(5)
            return super().refine(*a, **kw)

    client = SlowClient()
    ref = AsyncRefine(client)
    ref.start_chunk(0)
    assert ref.maybe_refine(1, np.zeros((1, 14)), {}, {})
    assert ref.busy() and not ref.maybe_refine(2, np.zeros((2, 14)), {}, {})
    gate.set()
    ref.finish()
    assert not ref.busy() and ref.maybe_refine(3, np.zeros((3, 14)), {}, {})
    ref.finish()
    assert [c["m"] for c in client.calls] == [1, 3]


def test_async_refine_errors_never_raise():
    ref = AsyncRefine(RefineProbeClient(fail=True), thread_factory=SyncThread)
    ref.start_chunk(0)
    assert ref.maybe_refine(1, np.zeros((1, 14)), {}, {})
    assert ref.errors == 1 and not ref.fresh()
    (chunk, plan, m, reply), = ref.drain()
    assert (chunk, plan, m) == (0, 0, 1) and reply["error"] == "no reply"


class RefineClient(FakeClient):
    """FakeClient + prepare / refine: refine answers plan + 0.01 * m on joint 0 (fresh on the given call numbers)."""

    def __init__(self, fresh_calls=(), horizon=50):
        super().__init__(fresh_calls=(), horizon=horizon)
        self.refine_fresh, self.prepares, self.refines, self.plans = set(fresh_calls), [], [], {}

    def prepare(self, state, prompt, cams, tacs, tacs_hist=None, codec="png"):
        plan_id = 100 + len(self.prepares)
        self.prepares.append({"state": np.asarray(state), "tacs": tacs, "hist": tacs_hist, "codec": codec})
        self.plans[plan_id] = self.predict(state, prompt, cams, tacs, tacs_hist, codec)
        return self.plans[plan_id].copy(), plan_id

    def refine(self, plan_id, m, executed, tacs, hist, state=None, force=False, codec="jpg", timeout_s=None,
               tactile_size=None):
        self.refines.append({"plan": plan_id, "m": m, "executed": np.array(executed), "state": state,
                             "hist": hist, "codec": codec, "timeout_s": timeout_s})
        fresh = len(self.refines) in self.refine_fresh
        actions = None
        if not fresh:
            actions = self.plans[plan_id].copy()
            actions[:, 0] += 0.01 * m
        return {"status": "ok", "plan": plan_id, "m": m, "stale": False, "fresh": fresh, "actions": actions,
                "surprise": [0.1, 0.2, 0.3, 0.4], "cusum": [0.0, 0.0, 0.5, 0.0], "pad": 2 if fresh else None,
                "e": 0.5, "delta": 3.0, "delta_max": 0.01 * m, "saturation": 0.0, "n_reread": 0, "ms": 20.0,
                "infer_time_ms": 21.0}


class DeferredThread:
    """threading.Thread stand-in whose target runs at the (lag + 1)-th sleep() after start(): the reply of a refine
    sent after m commanded steps is read at the top of step m + lag, i.e. d = lag."""
    pending = []
    lag = 0

    def __init__(self, target, args=(), daemon=None):
        self._target, self._args, self.done, self.sleeps = target, args, False, 0

    def start(self):
        DeferredThread.pending.append(self)

    def is_alive(self):
        return not self.done

    def join(self):
        self.run()

    def run(self):
        if not self.done:
            self.done = True
            self._target(*self._args)

    @classmethod
    def on_sleep(cls):
        for t in list(cls.pending):
            t.sleeps += 1
            if t.sleeps > cls.lag:
                cls.pending.remove(t)
                t.run()


def run_refine(fused, client, args, monkeypatch, thread_factory=SyncThread, on_sleep=None, env=None, clock=None):
    monkeypatch.setattr(fused, "AsyncRefine",
                        lambda c, codec, timeout_s, **kw: AsyncRefine(c, codec, timeout_s, thread_factory=thread_factory,
                                                                       **kw))
    sources = fused.video_sources("cam_high", fused.DEFAULT_TACTILE_MAP)
    env = env or FakeEnv()
    streams, log = FakeStreams(sources), []
    clock = clock or SimpleNamespace(t=0.0)

    def sleep(s):
        clock.t += s
        if on_sleep is not None:
            on_sleep()

    outcome = fused.run_episode(client, env, streams, sources, args, FakeKeyboard(), log.append,
                                clock=lambda: clock.t, sleep=sleep)
    for record in log:
        json.dumps(record)                              # EpisodeLog writes every record as JSON
    return outcome, env, log


def test_refine_off_never_prepares(fused):
    client = RefineClient()
    outcome, env, streams, log = run(fused, client, make_args(fused, refine=False, max_episode_steps=60))
    assert outcome == "done" and client.prepares == [] and client.refines == [] and len(client.predicts) == 2
    assert set(log[-1]) == {"event", "outcome", "steps", "chunks", "fresh", "probe_errors"}


def test_refine_corrections_land_on_the_next_step(fused, monkeypatch):
    client = RefineClient()
    args = make_args(fused, refine=True, refine_stop=3, max_episode_steps=50)
    outcome, env, log = run_refine(fused, client, args, monkeypatch)
    assert outcome == "done" and len(client.prepares) == 1 and client.predicts[0]["state"].shape == (32,)
    goals = np.array(env.puppet_bot_left.arm.goals)
    assert goals[0, 0] == pytest.approx(0.1)
    for j in range(1, 47):                              # d = 0: the reply for m = j is commanded at step j
        assert goals[j, 0] == pytest.approx(0.1 + 0.01 * j)
    np.testing.assert_allclose(goals[47:, 0], 0.1 + 0.01 * 46)   # no request in the last 3 steps
    assert [r["m"] for r in client.refines] == list(range(1, 47))
    for r in client.refines:                            # executed = the rows actually commanded
        np.testing.assert_allclose(r["executed"][:, 0], goals[:r["m"], 0])
        assert r["executed"].shape == (r["m"], 14) and r["state"].shape == (32,)
        assert r["codec"] == "png" and r["timeout_s"] == 2.0
    assert client.refines[9]["hist"][8]["left_wrist_left_tactile"][0, 0, 0] == 3   # m = 10 at step 10: t-8 = snap 3
    refines = [r for r in log if r["event"] == "refine"]
    assert len(refines) == 46 and all(r["d"] == 0 and r["start"] == r["m"] for r in refines)
    assert refines[0]["n_written"] == 49 and refines[0]["delta_arm_max"] == pytest.approx(0.01)
    assert refines[0]["surprise"] == [0.1, 0.2, 0.3, 0.4] and refines[0]["delta"] == 3.0
    chunk = next(r for r in log if r["event"] == "chunk")
    assert chunk["plan"] == 100 and np.asarray(chunk["plan_rows"]).shape == (50, 14)
    end_chunk = next(r for r in log if r["event"] == "chunk_end")
    assert end_chunk["executed_steps"] == 50 and end_chunk["n_refine"] == 46 and end_chunk["n_applied"] == 46
    np.testing.assert_allclose(np.asarray(end_chunk["executed_rows"])[:, 0], goals[:, 0], atol=1e-4)
    end = log[-1]
    assert end["event"] == "end" and end["refines"] == 46 and end["refine_applied"] == 46
    assert end["refine_errors"] == 0 and end["d_mean"] == 0 and end["d_max"] == 0 and end["overruns"] == 0


def test_refine_adaptive_stop(fused, monkeypatch):
    client = RefineClient()
    args = make_args(fused, refine=True, max_episode_steps=50)       # stop = max recent d + 1 = 1 once measured
    run_refine(fused, client, args, monkeypatch)
    assert [r["m"] for r in client.refines] == list(range(1, 49))


def test_refine_latency_writes_back_from_m_plus_d(fused, monkeypatch):
    DeferredThread.pending, DeferredThread.lag = [], 2
    client = RefineClient()
    args = make_args(fused, refine=True, refine_stop=0, max_episode_steps=60)
    outcome, env, log = run_refine(fused, client, args, monkeypatch, thread_factory=DeferredThread,
                                   on_sleep=DeferredThread.on_sleep)
    goals = np.array(env.puppet_bot_left.arm.goals)
    sent = [r["m"] for r in client.refines if r["plan"] == 100]
    assert sent[:4] == [1, 4, 7, 10] and sent[-1] == 49               # one in flight: every lag + 1 steps
    applied = [r for r in log if r["event"] == "refine" and "start" in r]
    assert applied and all(r["d"] == 2 and r["start"] == r["m"] + 2 for r in applied)
    np.testing.assert_allclose(goals[:3, 0], 0.1)                      # m = 1 lands at step 3, not before
    np.testing.assert_allclose(goals[3:6, 0], 0.11)
    np.testing.assert_allclose(goals[6:9, 0], 0.14)
    for r in client.refines:
        if r["plan"] == 100:
            np.testing.assert_allclose(r["executed"][:, 0], goals[:r["m"], 0])
    late = [r for r in log if r["event"] == "refine" and r.get("skipped") == "chunk ended"]
    assert [(r["plan"], r["m"]) for r in late] == [(100, 49)]          # in flight at the chunk end: never applied
    np.testing.assert_allclose(goals[50, 0], 0.1)                      # chunk 2 starts from its own plan
    assert log[-1]["d_max"] == 2 and log[-1]["d_mean"] == 2.0


def test_refine_fresh_ends_the_chunk(fused, monkeypatch):
    client = RefineClient(fresh_calls={3})               # the 3rd refine (m = 3) answers fresh
    args = make_args(fused, refine=True, refine_stop=3, max_episode_steps=60)
    outcome, env, log = run_refine(fused, client, args, monkeypatch)
    assert outcome == "done"
    fresh = [r for r in log if r["event"] == "fresh"]
    # the fresh reply is read at the top of h = 3: no further row of the old plan is commanded
    assert len(fresh) == 1 and fresh[0]["step"] == 3 and fresh[0]["chunk_step"] == 3
    assert fresh[0]["source"] == "refine"
    assert [c["step"] for c in log if c["event"] == "chunk"] == [0, 3, 53] and len(client.prepares) == 3
    assert not any(e == "takeover_request" for e in (r["event"] for r in log))
    goals = np.array(env.puppet_bot_left.arm.goals)
    np.testing.assert_allclose(goals[2, 0], 0.12)       # the m = 2 correction; the fresh reply has no actions
    np.testing.assert_allclose(goals[3, 0], 0.1)        # new chunk, new plan
    assert client.refines[3]["plan"] == 101 and client.refines[3]["m"] == 1
    assert next(r for r in log if r["event"] == "refine" and r["m"] == 3)["skipped"] == "fresh"
    assert log[-1]["fresh"] == 1


def test_refine_flag_conflicts(fused):
    with pytest.raises(SystemExit):
        make_args(fused, refine=True, probe_every=5)
    with pytest.raises(SystemExit):
        make_args(fused, refine=True, takeover_after=3)
    with pytest.raises(SystemExit):
        make_args(fused, refine=True, refine_stop=-1)
    p = fused.build_arg_parser()
    a = p.parse_args(["--host", "h", "--data", "hdf5"])
    assert a.refine is False and a.refine_codec == "png" and a.probe_codec == "png" and a.refine_timeout == 2.0 \
        and a.refine_stop is None
    a = p.parse_args(["--host", "h", "--data", "hdf5", "--refine", "--refine-max-delta", "0.05", "--refine-every", "3"])
    assert a.refine and a.refine_max_delta == 0.05 and a.refine_every == 3 and a.refine_max_grip_delta == 0.0


def test_refine_logs_overruns_and_errors(fused, monkeypatch):
    clock = SimpleNamespace(t=0.0)

    class SlowEnv(FakeEnv):                              # step 10 takes 30 ms of a 20 ms period
        def set_gripper_pose(self, left, right):
            super().set_gripper_pose(left, right)
            if len(self.grippers) == 10:
                clock.t += 0.03

    class FlakyClient(RefineClient):
        def refine(self, plan_id, m, *a, **kw):
            if m == 6:
                self.refines.append({"plan": plan_id, "m": m})
                raise TimeoutError("no reply within 2.0 s")
            return super().refine(plan_id, m, *a, **kw)

    client = FlakyClient()
    args = make_args(fused, refine=True, refine_stop=3, refine_every=2, max_episode_steps=50)
    outcome, env, log = run_refine(fused, client, args, monkeypatch, env=SlowEnv(), clock=clock)
    assert [r["m"] for r in client.refines][:4] == [2, 4, 6, 8]       # every 2nd step
    err = next(r for r in log if r["event"] == "refine" and r["m"] == 6)
    assert err["skipped"] == "error" and "no reply" in err["error"]
    goals = np.array(env.puppet_bot_left.arm.goals)
    np.testing.assert_allclose(goals[4:8, 0], 0.14)     # the failed m = 6 changed nothing
    np.testing.assert_allclose(goals[8, 0], 0.18)
    end = log[-1]
    assert end["overruns"] == 1 and end["max_overrun_ms"] == pytest.approx(10.0)
    assert end["refine_errors"] == 1 and end["refines"] == 23


def test_probe_and_refine_send_model_size_frames_unless_full_res(monkeypatch):
    """The background probe / refine threads ask the client for model-size (224) tactile frames by default and for
    native frames with --probe-full-res; predict / prepare are untouched (native frames, as the original client)."""
    import run_vtla_fused as fused

    seen = []

    class SizeClient:
        def seg_step(self, k, p, tacs, hist, force=False, codec="jpg", tactile_size=None):
            seen.append(("seg_step", tactile_size))
            return {"fresh": False}

        def refine(self, plan_id, m, executed, tacs, hist, state=None, force=False, codec="jpg", timeout_s=None,
                   tactile_size=None):
            seen.append(("refine", tactile_size))
            return {"stale": True, "actions": None}

    for full_res, want in ((False, (224, 224)), (True, None)):
        seen.clear()
        size = None if full_res else fused.MODEL_TACTILE_SIZE
        probe = AsyncProbe(SizeClient(), 1, "jpg", thread_factory=SyncThread, tactile_size=size)
        probe.start_chunk()
        probe.maybe_probe(1, 1, {"v": np.zeros((2, 2, 3), np.uint8)}, {})
        ref = AsyncRefine(SizeClient(), "jpg", 1.0, thread_factory=SyncThread, tactile_size=size)
        ref.start_chunk(1)
        ref.maybe_refine(1, np.zeros((1, 14)), {"v": np.zeros((2, 2, 3), np.uint8)}, {})
        assert seen == [("seg_step", want), ("refine", want)], (full_res, seen)
