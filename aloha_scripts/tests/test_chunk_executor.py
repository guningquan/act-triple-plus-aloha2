import numpy as np
import pytest

from chunk_executor import ChunkExecutor, PolicyError

DT = 0.02


class Obs:
    def __init__(self, position):
        self.position = position


def ramp_policy(calls):
    """Chunk step i = (position the observation was taken at) + i."""
    def infer(obs):
        calls.append(obs.position)
        return np.tile((obs.position + np.arange(50.0))[:, None], (1, 14))
    return infer


def test_replays_chunks_like_action_chunk_broker():
    calls = []
    executor = ChunkExecutor(ramp_policy(calls), policy_hz=50, control_dt=DT, action_horizon=25)
    outputs = [executor.step(lambda: Obs(executor.position))[0] for _ in range(60)]
    np.testing.assert_allclose(outputs, np.arange(60.0))
    assert calls == [0.0, 25.0, 50.0]  # observation captured only when a chunk is needed


def test_30hz_policy_is_interpolated_in_the_50hz_loop():
    executor = ChunkExecutor(lambda obs: np.tile(np.arange(50.0)[:, None], (1, 14)), policy_hz=30,
                             control_dt=DT, action_horizon=50)
    outputs = [executor.step(lambda: Obs(0))[0] for _ in range(5)]
    np.testing.assert_allclose(outputs, [0.0, 0.6, 1.2, 1.8, 2.4])


def test_reset_infers_again_from_a_fresh_observation():
    calls = []
    executor = ChunkExecutor(ramp_policy(calls), policy_hz=50, control_dt=DT, action_horizon=50)
    for _ in range(10):
        executor.step(lambda: Obs(executor.position))
    executor.reset()  # e.g. pause and resume
    assert executor.step(lambda: Obs(executor.position))[0] == 10.0
    assert calls == [0.0, 10.0]


def test_errors_become_policy_errors():
    def broken(obs):
        raise TimeoutError("no reply")

    executor = ChunkExecutor(broken, policy_hz=50, control_dt=DT)
    with pytest.raises(PolicyError, match="no reply"):
        executor.step(lambda: Obs(0))
    executor = ChunkExecutor(lambda obs: np.full((50, 14), np.nan), policy_hz=50, control_dt=DT)
    with pytest.raises(PolicyError):
        executor.step(lambda: Obs(0))
    executor = ChunkExecutor(lambda obs: np.zeros((50, 7)), policy_hz=50, control_dt=DT)
    with pytest.raises(PolicyError):
        executor.step(lambda: Obs(0))
