"""ITAndroidsPool with a fake agent (no vss_fast build needed)."""
from __future__ import annotations

import numpy as np

from vsss_sim import config
from vsss_sim.agents.itandroids import ITAndroidsPool


class _FakeAgent:
    """Echoes the ball x as left wheel and y as right wheel, in rad/s."""

    def __init__(self) -> None:
        self.resets = 0

    def robot_params(self):
        return {"wheel_radius": 0.02}

    def reset(self) -> None:
        self.resets += 1

    def step(self, allies, opponents, ball, ally_vel, opp_vel, ball_vel):
        assert allies.shape == (3, 3) and ally_vel.shape == (3, 3)
        assert opponents.shape == (3, 3) and opp_vel.shape == (3, 2)
        return np.tile(np.array([ball[0], ball[1]]) * 100.0, (3, 1))


def make_fake_agent() -> _FakeAgent:
    return _FakeAgent()


def test_pool_converts_wheel_speeds_and_clips() -> None:
    robots = np.zeros((2, config.N_TEAMS, config.N_ROBOTS, 6))
    ball = np.zeros((2, 4))
    ball[0, :2] = [0.5, -0.25]   # 50, -25 rad/s -> 1.0, -0.5 m/s at r=0.02
    ball[1, :2] = [10.0, 0.0]    # saturates
    with ITAndroidsPool(2, make_fake_agent) as pool:
        pool.reset(np.array([True, False]))
        actions = pool.step(robots, ball)
    assert actions.shape == (2, config.N_ROBOTS, 2)
    assert actions.dtype == np.float32
    expected = np.array([50.0, -25.0]) * 0.02 / config.ROBOT_MAX_WHEEL_SPEED
    np.testing.assert_allclose(actions[0, 0], expected, rtol=1e-5)
    assert actions[1, 0, 0] == 1.0
    assert np.all(np.abs(actions) <= 1.0)
