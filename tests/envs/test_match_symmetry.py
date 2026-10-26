"""Team rotation and accepted-event contracts for all match adapters."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vsss_sim.envs import VSSEnv, VSSVecEnv
from vsss_sim.envs import brandao_observation as observation
from vsss_sim.envs.jumanji import VSSJumanjiEnv
from vsss_sim.envs.match_referee import MatchPhase, RefereeEvent, RestartType, initial_referee_state
from vsss_sim.envs.match_restarts import _restart_positions
from vsss_sim.envs.match_rules import MatchRules, MatchStepInput
from vsss_sim.physics import jax_backend as physics


def rotate(simulation):
    robots = simulation.robots[::-1]
    robots = robots.at[..., :2].multiply(-1).at[..., 3:5].multiply(-1)
    robots = robots.at[..., 2].set(jnp.arctan2(
        jnp.sin(robots[..., 2] + jnp.pi), jnp.cos(robots[..., 2] + jnp.pi)
    ))
    return simulation._replace(
        ball=-simulation.ball, robots=robots, score=simulation.score[::-1],
        wheel_speeds=simulation.wheel_speeds[::-1],
    )


def assert_geometry_equal(actual, expected):
    np.testing.assert_allclose(actual.ball, expected.ball, atol=1e-5)
    np.testing.assert_allclose(actual.robots[..., :2], expected.robots[..., :2], atol=1e-5)
    np.testing.assert_allclose(actual.robots[..., 3:], expected.robots[..., 3:], atol=1e-5)
    np.testing.assert_allclose(jnp.sin(actual.robots[..., 2]), jnp.sin(expected.robots[..., 2]), atol=1e-5)
    np.testing.assert_allclose(jnp.cos(actual.robots[..., 2]), jnp.cos(expected.robots[..., 2]), atol=1e-5)
    np.testing.assert_array_equal(actual.score, expected.score)
    np.testing.assert_allclose(actual.wheel_speeds, expected.wheel_speeds, atol=1e-5)


@pytest.mark.parametrize("restart", [RestartType.PENALTY_KICK, RestartType.GOAL_KICK, RestartType.FREE_KICK])
def test_awarded_restarts_rotate_without_permuting_robot_roles(restart):
    before = physics.reset_kickoff(jax.random.PRNGKey(8), jnp.int32(0))
    place = jax.jit(_restart_positions)
    blue = place(before, jnp.int32(restart), jnp.int32(0), jnp.int32(-1))
    yellow = place(rotate(before), jnp.int32(restart), jnp.int32(1), jnp.int32(-1))
    assert_geometry_equal(yellow, rotate(blue))


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("quadrant", range(4))
def test_neutral_free_ball_placements_rotate_with_quadrant(stage, quadrant):
    before = physics.reset_kickoff(jax.random.PRNGKey(8), jnp.int32(0))
    place = jax.jit(_restart_positions)
    original = place(before, jnp.int32(4), jnp.int32(-1), jnp.int32(quadrant), jnp.int32(stage))
    other = place(rotate(before), jnp.int32(4), jnp.int32(-1), jnp.int32(3-quadrant), jnp.int32(stage))
    assert_geometry_equal(other, rotate(original))


def test_kickoff_roles_rotate_for_the_other_possession_team():
    keys = jax.random.split(jax.random.PRNGKey(8), 32)
    blue = jax.jit(jax.vmap(lambda k: physics.reset_kickoff(k, jnp.int32(0))))(keys)
    yellow = jax.jit(jax.vmap(lambda k: physics.reset_kickoff(k, jnp.int32(1))))(keys)
    assert_geometry_equal(yellow, jax.vmap(rotate)(blue))


@pytest.mark.parametrize("restart", [RestartType.PENALTY_KICK, RestartType.GOAL_KICK, RestartType.FREE_KICK])
def test_brandao_restart_features_are_equal_in_each_teams_canonical_view(restart, monkeypatch):
    # Couple camera noise by disabling it for this exact geometric invariant.
    monkeypatch.setattr(observation, "_POSITION_NOISE", jnp.zeros((2,), dtype=jnp.float32))
    monkeypatch.setattr(observation, "_ANGLE_NOISE", jnp.float32(0))
    before = physics.empty_state()
    blue = _restart_positions(before, jnp.int32(restart), jnp.int32(0), jnp.int32(-1))
    yellow = _restart_positions(rotate(before), jnp.int32(restart), jnp.int32(1), jnp.int32(-1))
    def features(simulation):
        return observation.build_agent_features(simulation, jnp.zeros((2, 3, 2)), jnp.float32(1), jax.random.PRNGKey(1))
    original = np.asarray(jax.jit(features)(blue))[::-1]
    other = np.asarray(jax.jit(features)(yellow))
    # Normalized angles represent a circle, so -1 and +1 are identical.
    angle_slots = [16, *range(19, 38, 3)]
    difference = other-original
    difference[..., angle_slots] = (difference[..., angle_slots]+1) % 2-1
    np.testing.assert_allclose(difference, 0, atol=2e-5)


def simultaneous_fouls():
    before = physics.empty_state()
    robots = before.robots.at[..., :2].set(jnp.asarray([
        [[-.20, -.25], [0, .25], [-.60, .50]],
        [[0, -.25], [.20, .25], [.60, -.50]],
    ], dtype=jnp.float32))
    before = before._replace(robots=robots)
    after = before._replace(robots=robots.at[0, 0, 0].set(-.075).at[1, 1, 0].set(.075))
    return before, after


@pytest.mark.parametrize("both", [True, False])
def test_simultaneous_call_ties_use_seeded_color_neutral_priority(both):
    before, after = simultaneous_fouls()
    if not both:
        after = after._replace(robots=after.robots.at[1, 1, 0].set(.20))
    rules = MatchRules()
    def call(key):
        return rules.step(MatchStepInput(before, after, jnp.int32(0), jnp.int32(1), key,
                                         initial_referee_state(), jnp.float32(.1)))
    result = jax.jit(jax.vmap(call))(jax.random.split(jax.random.PRNGKey(17), 512))
    assert np.all(np.asarray(result.event) == RefereeEvent.PERSONAL_FOUL)
    offenders = np.asarray(result.offending_team)
    if both:
        counts = np.bincount(offenders, minlength=2)
        assert np.all((counts > 180) & (counts < 332))
    else:
        np.testing.assert_array_equal(offenders, 0)


@pytest.mark.parametrize("adapter", ["gym", "vector", "jumanji"])
@pytest.mark.parametrize("direction", [-1, 1])
def test_stopped_frame_cannot_report_or_terminate_on_discarded_physics_goal(adapter, direction):
    sim = physics.empty_state()._replace(ball=jnp.asarray([direction * .80, 0, 0, 0], dtype=jnp.float32))
    stopped = initial_referee_state()._replace(phase=jnp.int32(MatchPhase.STOPPED))
    if adapter == "jumanji":
        env = VSSJumanjiEnv(terminate_on_goal=True, max_episode_steps=20)
        state, _ = env.reset(jax.random.PRNGKey(2))
        state = state._replace(simulation=sim, referee_state=stopped)
        next_state, timestep = jax.jit(env.step)(state, jnp.zeros((6,)))
        assert not bool(timestep.last())
        assert int(timestep.extras["goal"]) == 0
        np.testing.assert_array_equal(next_state.simulation.score, [0, 0])
    elif adapter == "gym":
        env = VSSEnv(opponent_policy="stationary")
        env.reset(seed=2)
        env._state, env._referee_state = sim, stopped
        _, reward, terminated, _, info = env.step(np.zeros((6,), np.float32))
        assert not terminated and info["goal"] == 0 and reward == 0
        np.testing.assert_array_equal(env._state.score, [0, 0])
        env.close()
    else:
        env = VSSVecEnv(num_envs=1, opponent_policy="stationary")
        env.reset(seed=2)
        env._state = jax.tree_util.tree_map(lambda x: x[None], sim)
        env._referee_state = jax.tree_util.tree_map(lambda x: x[None], stopped)
        _, reward, terminated, _, info = env.step(np.zeros((1, 6), np.float32))
        assert not terminated[0] and info["goal"][0] == 0 and reward[0] == 0
        np.testing.assert_array_equal(env._state.score, [[0, 0]])
        env.close()


def test_physics_one_step_rotates_when_teams_and_actions_are_swapped():
    keys = jax.random.split(jax.random.PRNGKey(19), 32)
    states = jax.jit(jax.vmap(physics.reset_random))(keys)
    actions = jax.random.uniform(jax.random.PRNGKey(20), (32, 2, 3, 2), minval=-1, maxval=1)
    step = jax.jit(jax.vmap(lambda s, a: physics.step(s, a, dt=.1, sub_steps=6)))
    result, info = step(states, actions)
    other, other_info = step(jax.vmap(rotate)(states), actions[:, ::-1])
    assert_geometry_equal(other, jax.vmap(rotate)(result))
    np.testing.assert_array_equal(other_info["goal"], -info["goal"])


@pytest.mark.parametrize("xy,allowed", [((0.,0.), [0,1,2,3]), ((0.,.2), [2,3]), ((.2,0.), [1,3])])
def test_free_ball_axis_ties_choose_neutral_quadrants(xy, allowed):
    from vsss_sim.envs.match_referee import _free_ball_quadrant
    ball = jnp.asarray([*xy, 0., 0.], dtype=jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(13), 1024)
    quadrants = np.asarray(jax.jit(jax.vmap(lambda key: _free_ball_quadrant(ball, key)))(keys))
    assert set(np.unique(quadrants)) == set(allowed)
    counts = np.bincount(quadrants, minlength=4)[allowed]
    expected = 1024 / len(allowed)
    assert np.all(np.abs(counts-expected) < expected*.3)


def test_free_ball_axis_choice_is_retained_until_stoppage_ends():
    from vsss_sim.envs.match_referee import _resolve_referee
    sim = physics.reset_kickoff(jax.random.PRNGKey(8), jnp.int32(0))
    decide = jax.jit(_resolve_referee)
    first = decide(sim, sim, initial_referee_state(), jnp.int32(0), jnp.float32(.1), jax.random.PRNGKey(5))
    referee = first.state._replace(field_impasse_seconds=jnp.float32(4.99))
    keys = jax.random.split(jax.random.PRNGKey(16), 32)
    calls = jax.jit(jax.vmap(lambda key: _resolve_referee(sim, sim, referee, jnp.int32(0), jnp.float32(.1), key)))(keys)
    assert np.all(np.asarray(calls.event) == RefereeEvent.FIELD_IMPASSE)
    np.testing.assert_array_equal(calls.quadrant, int(first.state.field_impasse_quadrant))
