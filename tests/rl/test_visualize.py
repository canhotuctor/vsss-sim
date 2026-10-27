"""Checks for saved visualizer opponents and their observation histories."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim import physics
from vsss_sim.envs.match_referee import MatchPhase
from vsss_sim.perspective import switch_team_perspective

spec = importlib.util.spec_from_file_location(
    "visualize_script", Path(__file__).resolve().parents[2] / "scripts" / "visualize.py"
)
visualize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualize)


@pytest.mark.parametrize(
    ("blue_arch", "yellow_arch"),
    [("brandao_shared", "brandao_shared"), ("brandao_shared", "brandao_sets"),
     ("joint", "joint"), ("joint", "brandao_shared")],
)
def test_saved_opponent_actions_and_history(monkeypatch, blue_arch, yellow_arch):
    calls = []
    observations = {"blue": [], "yellow": []}
    raw_observation = np.arange(46, dtype=np.float32) / 100
    initial_state = physics.empty_state()

    class Model:
        def __init__(self, team, architecture):
            self.team = team
            self.architecture = architecture

        def apply(self, _params, observation):
            observations[self.team].append(np.asarray(observation))
            shape = (3, 2) if self.architecture.startswith("brandao_") else (6,)
            if self.architecture.startswith("brandao_"):
                first, second = (2.0, 1.0) if self.team == "blue" else (1.0, 3.0)
                return jnp.full(shape, first), jnp.full(shape, second), jnp.array(0.)
            mean = 0.4 if self.team == "blue" else -0.6
            return jnp.full(shape, mean), jnp.zeros(shape), jnp.array(0.)

    def load_snapshot(path):
        team = str(path)
        architecture = blue_arch if team == "blue" else yellow_arch
        return (Model(team, architecture), {},
                {"policy_architecture": architecture, "generation": 1, "lineage": 0},
                {"max_episode_steps": 3})

    def features(_state, previous, _remaining, _key, entity_ordered_actions=False):
        calls.append((entity_ordered_actions, np.asarray(previous).copy()))
        return jnp.zeros((2, 3, 53), dtype=jnp.float32)

    class Env:
        def __init__(self, opponent_policy, **_kwargs):
            self._opponent_policy = opponent_policy
            self._state = initial_state
            self._referee_state = SimpleNamespace(phase=MatchPhase.PLAYING)
            self.steps = 0
            self.yellow_actions = []

        def reset(self, seed):
            self.steps = 0
            self._referee_state.phase = MatchPhase.PLAYING
            return raw_observation.copy(), {}

        def step(self, _action):
            self.yellow_actions.append(np.asarray(self._opponent_policy(raw_observation)))
            self.steps += 1
            self._referee_state.phase = (
                MatchPhase.STOPPED if self.steps == 1 else MatchPhase.PLAYING
            )
            return raw_observation.copy(), 0., False, self.steps == 3, {
                "goal": 0, "score_blue": 0, "score_yellow": 0,
            }

        def close(self):
            pass

    envs = []
    def make_env(*args, **kwargs):
        env = Env(*args, **kwargs)
        envs.append(env)
        return env

    monkeypatch.setattr(visualize, "load_snapshot", load_snapshot)
    monkeypatch.setattr(visualize, "build_agent_features", features)
    monkeypatch.setattr(visualize, "VSSEnv", make_env)
    monkeypatch.setattr(jax, "jit", lambda fn: fn)
    monkeypatch.setattr(sys, "argv", ["visualize", "blue", "--opponent-snapshot", "yellow",
                                    "--episodes", "2"])
    visualize.main()

    blue_value = 1 / 3 if blue_arch.startswith("brandao_") else np.tanh(0.4)
    yellow_value = -0.5 if yellow_arch.startswith("brandao_") else np.tanh(-0.6)
    np.testing.assert_allclose(envs[0].yellow_actions, yellow_value, atol=1e-6)
    if yellow_arch == "joint":
        np.testing.assert_allclose(observations["yellow"][0],
                                   switch_team_perspective(raw_observation))
    if calls:
        orders = {blue_arch == "brandao_sets" for _ in [0] if blue_arch.startswith("brandao_")}
        if yellow_arch.startswith("brandao_"):
            orders.add(yellow_arch == "brandao_sets")
        for order in orders:
            values = [prev for used_order, prev in calls if used_order == order]
            assert len(values) == 6  # reset, action, stopped action, twice
            for offset in (0, 3):
                np.testing.assert_allclose(values[offset], 0)
                np.testing.assert_allclose(values[offset + 1][0], blue_value, atol=1e-6)
                np.testing.assert_allclose(values[offset + 1][1], yellow_value, atol=1e-6)
                np.testing.assert_allclose(values[offset + 2], 0)
