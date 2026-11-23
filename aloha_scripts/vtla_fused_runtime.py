"""Pure-Python pieces of run_vtla_fused.py (no ROS, no robot), unit-tested in tests/test_vtla_fused_runtime.py.

TactileHistory   ring buffer of the 4 tactile views per control step; frame(step - k) for the Direction-3 history frame
                 (before step k of the episode: the episode's first frame, as the training data loader clamps t - k)
AsyncProbe       Direction 2 C-fresh on the real robot: a background thread sends seg_step probes on the latest frames
                 every ``every`` control steps (at most one in flight) and raises a flag when the server answers
                 "fresh" (end the chunk; the next predict starts from the live observation). At 50 Hz a synchronous
                 probe (encode + network + ~20 ms GPU) would stall every step, so the robot never waits for it.
TakeoverMonitor  counts "fresh" events; ``limit`` of them within ``window`` control steps -> ask the operator to take
                 over (persistent tactile mismatch, architecture doc 3.4)
ChunkPlan        --refine: the executed chunk. Keeps the ORIGINAL plan read-only and the rows to command; a refine
                 reply for "m executed steps" is written back from the first row not yet commanded (m + d, d = steps
                 executed while the request was in flight) as original plan + correction, so corrections never
                 accumulate and rows already commanded never change.
AsyncRefine      --refine: the in-chunk tactile correction requests (server ``refine``) in a background thread, at most
                 one in flight; the replies (corrected chunk + the C-fresh decision) are applied by the main thread.
"""

import collections
import logging
import threading
import time

import numpy as np

# seg_step reply fields kept for the episode log
PROBE_LOG_KEYS = ("fresh", "replanned", "e", "delta", "lag", "n_reread", "ms")
# refine reply fields kept for the episode log (the corrected actions are applied, not logged); "delta" is the
# server's CUSUM threshold h, "delta_max" its largest |correction| on the arm dims
REFINE_LOG_KEYS = ("stale", "fresh", "surprise", "cusum", "pad", "e", "delta", "delta_max", "saturation", "n_reread",
                   "ms", "infer_time_ms")
# (14,) action row layout: [left j1..6, left gripper, right j1..6, right gripper]
ARM_DIMS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIP_DIMS = (6, 13)


class TactileHistory:
    def __init__(self, maxlen=256):
        self._frames = collections.OrderedDict()   # step -> {view: RGB}
        self._maxlen = max(1, int(maxlen))
        self._first = None

    def clear(self):
        self._frames.clear()
        self._first = None

    def push(self, step, frames):
        if self._first is None:
            self._first = dict(frames)
        self._frames[int(step)] = dict(frames)
        while len(self._frames) > self._maxlen:
            self._frames.popitem(last=False)

    def frame(self, step):
        """The frames of control step ``step``; the nearest earlier stored step if it is missing; the episode's
        first frames for step < 0 (t - k before the episode start); the oldest stored frames for a step that was
        already dropped from the buffer."""
        if self._first is None:
            return None
        step = int(step)
        if step in self._frames:
            return self._frames[step]
        earlier = [s for s in self._frames if s <= step]
        if earlier:
            return self._frames[max(earlier)]
        return self._first if step < 0 else next(iter(self._frames.values()))


class AsyncProbe:
    def __init__(self, client, every=5, codec="png", thread_factory=threading.Thread, tactile_size=(224, 224)):
        self.client = client
        self.every = max(1, int(every))
        self.codec = codec
        self.tactile_size = tactile_size      # frames resized to the model size before encoding (None = native)
        self._thread_factory = thread_factory   # tests pass a synchronous stand-in
        self.chunk = -1          # incremented by start_chunk(); each reply carries the chunk it was sent in
        self._thread = None
        self._fresh = threading.Event()
        self._replies = collections.deque()   # (chunk, k, reply) until drain(); deque append / popleft are atomic
        self.errors = 0
        self._warned_replan = False

    def start_chunk(self):
        """Call right after a chunk-start predict: a probe still in flight belongs to the previous chunk."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._fresh.clear()
        self.chunk += 1

    def maybe_probe(self, k, p, tacs, hist):
        """At control step k of the chunk (k >= 1): start a probe if it is due and none is in flight."""
        if k < 1 or k % self.every or (self._thread is not None and self._thread.is_alive()):
            return False
        self._thread = self._thread_factory(target=self._run, args=(self.chunk, k, p, tacs, hist), daemon=True)
        self._thread.start()
        return True

    def _run(self, chunk, k, p, tacs, hist):
        try:
            r = self.client.seg_step(k, p, tacs, hist, codec=self.codec, tactile_size=self.tactile_size)
        except Exception as exc:   # noqa: BLE001  - a failed probe must never stop the robot
            self.errors += 1
            self._replies.append((chunk, k, {"error": str(exc)}))
            return
        self._replies.append((chunk, k, {key: r.get(key) for key in PROBE_LOG_KEYS}))
        if r.get("replanned") and not self._warned_replan:
            # C without C-fresh re-generated the chunk on the server; this client only ends chunks (C-fresh)
            self._warned_replan = True
            logging.warning("the server re-generated a chunk (Direction 2 C without C-fresh): its actions are "
                            "IGNORED here; start the server with VTLA_DIR2_C_FRESH=1")
        if r.get("fresh"):
            self._fresh.set()

    def fresh(self):
        return self._fresh.is_set()

    def drain(self):
        """The replies received since the last drain(), oldest first: [(chunk, k, reply), ...]."""
        out = []
        while True:
            try:
                out.append(self._replies.popleft())
            except IndexError:
                return out

    def finish(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None


class TakeoverMonitor:
    def __init__(self, limit=3, window=250):
        self.limit = int(limit)
        self.window = int(window)
        self._events = collections.deque()

    def clear(self):
        self._events.clear()

    def event(self, step):
        """A "fresh" (tactile mismatch) at control step ``step`` -> True when the operator should take over."""
        self._events.append(int(step))
        while self._events and self._events[0] <= step - self.window:
            self._events.popleft()
        return self.limit > 0 and len(self._events) >= self.limit


class ChunkPlan:
    """The chunk being executed under --refine; main thread only (no lock).

    ``plan`` is the chunk as prepared (read-only, never changed); ``rows`` what is commanded. A reply to a refine
    sent after m commanded steps is applied when the main thread reads it: by then ``cursor`` (rows commanded so far)
    is m + d, and rows [cursor, n_exec) become plan + (corrected - plan), optionally clipped per element. Each reply
    REPLACES the previous correction (TacPAC: corrections are relative to the original plan, never cumulative), and
    rows before the cursor - what the arms were already told - are never rewritten.

    ``blend`` N > 1 (--refine-blend): the new correction does not replace the scheduled one at once but is reached
    over N rows, correction(start + i) = old + min(1, (i + 1) / N) * (new - old), where old is the correction
    currently scheduled on those rows (possibly mid-ramp). Successive replies that disagree then move each commanded
    row by at most 1/N of their difference instead of jumping (N = 1: the plain replace). Both old and new respect
    the caps, so every blended row does too, and once the ramp is done the rows are again plan + the new correction."""

    def __init__(self, plan_id, actions, n_exec, max_delta=0.0, max_grip_delta=0.0, blend=1):
        plan = np.array(actions, dtype=np.float64)
        if plan.ndim != 2 or plan.shape[1] < 14:
            raise ValueError("plan must be (H, >= 14), got {}".format(plan.shape))
        self.plan_id = int(plan_id)
        plan = plan[:, :14].copy()
        plan.setflags(write=False)
        self.plan = plan
        self.rows = plan.copy()
        self.n_exec = min(int(n_exec), plan.shape[0])
        self.max_delta = float(max_delta or 0.0)
        self.max_grip_delta = float(max_grip_delta or 0.0)
        self.blend = 1 if blend is None else int(blend)
        if self.blend < 1:
            raise ValueError("blend must be >= 1, got {}".format(blend))
        self.cursor = 0
        self.last_m = -1
        self.n_applied = 0
        self.ds = []          # measured d of every in-order reply of this plan

    def take(self, h):
        """The row to command at chunk step h (h must be the next one)."""
        if h != self.cursor:
            raise AssertionError("ChunkPlan.take({}) but the cursor is at {}".format(h, self.cursor))
        self.cursor = h + 1
        return self.rows[h]

    def executed(self):
        """The rows actually commanded so far in this chunk, (cursor, 14) copy."""
        return self.rows[:self.cursor].copy()

    def apply(self, plan_id, m, corrected):
        """Write a refine reply back -> a JSON-native dict for the episode log ("skipped" when nothing changed)."""
        if int(plan_id) != self.plan_id:
            return {"skipped": "stale plan"}
        m = int(m)
        if m <= self.last_m:
            return {"skipped": "out of order"}
        self.last_m = m
        start = max(m, self.cursor)
        d = self.cursor - m
        self.ds.append(int(d))
        if start >= self.n_exec:
            return {"skipped": "no steps left", "d": int(d)}
        arr = None if corrected is None else np.asarray(corrected, dtype=np.float64)
        if (arr is None or arr.ndim != 2 or arr.shape[0] < self.n_exec or arr.shape[1] < 14
                or not np.isfinite(arr[:self.n_exec, :14]).all()):
            return {"skipped": "bad actions", "d": int(d)}
        delta = arr[start:self.n_exec, :14] - self.plan[start:self.n_exec]
        clipped = 0
        arm, grip = list(ARM_DIMS), list(GRIP_DIMS)
        for dims, cap in ((arm, self.max_delta), (grip, self.max_grip_delta)):
            if cap > 0:
                part = delta[:, dims]
                clipped += int((np.abs(part) > cap).sum())
                delta[:, dims] = np.clip(part, -cap, cap)
        if self.blend > 1:                              # ramp from the scheduled correction to the new one
            old = self.rows[start:self.n_exec] - self.plan[start:self.n_exec]
            w = np.minimum(1.0, np.arange(1, len(delta) + 1, dtype=np.float64) / self.blend)[:, None]
            self.rows[start:self.n_exec] = self.plan[start:self.n_exec] + old + w * (delta - old)
        else:
            self.rows[start:self.n_exec] = self.plan[start:self.n_exec] + delta
        self.n_applied += 1
        jump = float(np.abs(self.rows[start, arm] - self.rows[start - 1, arm]).max()) if start > 0 else 0.0
        return {"d": int(d), "start": int(start), "n_written": int(self.n_exec - start),
                "delta_arm_max": float(np.abs(delta[:, arm]).max()),
                "delta_arm_mean": float(np.abs(delta[:, arm]).mean()),
                "delta_grip_max": float(np.abs(delta[:, grip]).max()),
                "jump": jump, "clipped": clipped}


class AsyncRefine:
    """In-chunk tactile corrections on the real robot: a background thread sends ``refine`` (the rows commanded so
    far + the live touch) and queues the reply; the main thread applies it (ChunkPlan.apply) at the next step.

    At most one request is in flight (one-shot threads, as AsyncProbe), so the request rate adapts to the round-trip
    time; JPEG encoding runs in the worker, outside the client lock. The reply also carries the C-fresh decision
    (per-pad surprise + CUSUM on the server): ``fresh()`` turns True when a reply of the current plan says so."""

    def __init__(self, client, codec="png", timeout_s=2.0, clock=time.monotonic, thread_factory=threading.Thread,
                 tactile_size=(224, 224)):
        self.client = client
        self.codec = codec
        self.tactile_size = tactile_size      # frames resized to the model size before encoding (None = native)
        self.timeout_s = float(timeout_s)
        self._clock = clock
        self._thread_factory = thread_factory   # tests pass a synchronous / deferred stand-in
        self.chunk = -1
        self.plan_id = None
        self._thread = None
        self._fresh = threading.Event()
        self._replies = collections.deque()      # (chunk, plan id, m, reply) until drain()
        self.errors = 0
        self.sent = 0

    def start_chunk(self, plan_id):
        """Call right after the chunk's prepare (which waited for any in-flight refine on the client lock)."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._fresh.clear()
        self.chunk += 1
        self.plan_id = int(plan_id)

    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    def maybe_refine(self, m, executed, tacs, hist, state=None):
        """After m commanded steps (m >= 1): send a refine unless one is still in flight."""
        if m < 1 or self.plan_id is None or self.busy():
            return False
        self.sent += 1
        self._thread = self._thread_factory(
            target=self._run, args=(self.chunk, self.plan_id, int(m), executed, tacs, hist, state, self._clock()),
            daemon=True)
        self._thread.start()
        return True

    def _run(self, chunk, plan_id, m, executed, tacs, hist, state, t_sent):
        try:
            r = self.client.refine(plan_id, m, executed, tacs, hist, state=state, codec=self.codec,
                                   timeout_s=self.timeout_s, tactile_size=self.tactile_size)
        except Exception as exc:   # noqa: BLE001  - a failed refine must never stop the robot
            self.errors += 1
            self._replies.append((chunk, plan_id, m, {"error": str(exc),
                                                      "rtt_ms": round((self._clock() - t_sent) * 1000, 1)}))
            return
        r = dict(r)
        r["rtt_ms"] = round((self._clock() - t_sent) * 1000, 1)
        self._replies.append((chunk, plan_id, m, r))
        if r.get("fresh") and not r.get("stale") and plan_id == self.plan_id:
            self._fresh.set()

    def fresh(self):
        return self._fresh.is_set()

    def drain(self):
        """The replies received since the last drain(), oldest first: [(chunk, plan id, m, reply), ...]."""
        out = []
        while True:
            try:
                out.append(self._replies.popleft())
            except IndexError:
                return out

    def finish(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
