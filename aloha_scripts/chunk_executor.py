"""Turn a remote policy's action chunks into one follower command per control tick.

Same behaviour as openpi's aloha_real client (ActionChunkBroker): when the current chunk has played
`action_horizon` steps, capture an observation, block on the server, and start the new chunk at its
step 0. The control loop stalls for one round trip per chunk; the servos hold meanwhile.

Playback runs on a policy-step clock that advances control_dt * policy_hz per tick (1.0 for a 50 Hz
openpi policy in this 50 Hz loop, 0.6 for 30 Hz N0-VTLA), not on wall time, so a 50 Hz chunk is
replayed one action per tick exactly like ActionChunkBroker; fractional positions are linearly
interpolated between neighbouring chunk steps.

step() raises PolicyError when inference failed or timed out, and reset() (pause, end of episode)
drops the current chunk so the next step() infers from a fresh observation.
"""

import time

import numpy as np


class PolicyError(RuntimeError):
    """Inference failed or timed out; the caller must stop autonomous motion."""


class _Chunk:
    def __init__(self, actions, start, length):
        self.actions = actions
        self.start = start  # policy-step clock value at which actions[0] plays
        self.length = length  # number of playable steps

    def sample(self, position):
        """Action at policy-step clock `position`, or None once the chunk has played out."""
        offset = position - self.start
        if offset > self.length - 1 + 1e-9:
            return None
        offset = max(offset, 0.0)
        low = int(np.floor(offset + 1e-9))
        frac = offset - low
        if frac < 1e-9 or low + 1 >= self.length:
            return self.actions[min(low, self.length - 1)].copy()
        return (1.0 - frac) * self.actions[low] + frac * self.actions[low + 1]


class ChunkExecutor:
    def __init__(self, infer_fn, policy_hz, control_dt, action_horizon=50, clock=time.monotonic):
        if action_horizon < 1 or policy_hz <= 0 or control_dt <= 0:
            raise ValueError("invalid executor settings")
        self.infer_fn = infer_fn
        self.policy_hz = float(policy_hz)
        self.action_horizon = int(action_horizon)
        self.clock = clock
        self.advance = control_dt * self.policy_hz
        self.position = 0.0
        self.latencies = []
        self._chunk = None

    def reset(self):
        """Forget the current chunk; the next step() infers again."""
        self._chunk = None

    def close(self):
        pass

    def step(self, frame_provider):
        """Command for this tick ((14,) array). `frame_provider()` captures an observation; it is
        only called when a new chunk is needed."""
        if self._chunk is None or self._chunk.sample(self.position) is None:
            start = self.clock()
            try:
                actions = self.infer_fn(frame_provider())
            except Exception as exc:
                raise PolicyError("inference failed: {}".format(exc)) from exc
            self.latencies.append(self.clock() - start)
            actions = np.asarray(actions, dtype=np.float64)
            if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
                raise PolicyError("policy returned an invalid action chunk {}".format(actions.shape))
            self._chunk = _Chunk(actions, self.position, min(self.action_horizon, len(actions)))
        action = self._chunk.sample(self.position)
        self.position += self.advance
        return action

    def stats(self):
        if not self.latencies:
            return "no inference yet"
        latencies = 1000 * np.asarray(self.latencies)
        return "inference {} calls, mean {:.0f} ms, max {:.0f} ms".format(
            len(latencies), latencies.mean(), latencies.max())
