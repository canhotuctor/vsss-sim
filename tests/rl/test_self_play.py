"""Behavioral tests for historical self-play and policy snapshot archives."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jumanji")
pytest.importorskip("flax")

from vsss_sim.envs.jumanji import VSSJumanjiEnv  # noqa: E402
from vsss_sim.envs.match_referee import RefereeEvent  # noqa: E402
from vsss_sim.perspective import switch_team_perspective  # noqa: E402
from vsss_sim.rl import PPO, PPOConfig  # noqa: E402
from vsss_sim.rl.evaluate import evaluate_experiment  # noqa: E402
from vsss_sim.rl.ppo import ActorCritic, create_policy_model  # noqa: E402
from vsss_sim.rl.self_play import (  # noqa: E402
    ExperimentConfig,
    OpponentSamplingWeights,
    SnapshotArchive,
    _make_env,
    _opponent_evaluation_summary,
    _require_gpu_backend,
    run_experiment,
)


def test_team_perspective_is_an_involution():
    observation = jnp.arange(46, dtype=jnp.float32)
    transformed = switch_team_perspective(observation)
    restored = switch_team_perspective(transformed)

    np.testing.assert_array_equal(restored, observation)
    np.testing.assert_array_equal(transformed[:4], -observation[:4])


def test_default_historical_selector_weights_are_normalized():
    probabilities = OpponentSamplingWeights().probabilities([0, 10, 20])

    assert probabilities.shape == (5,)
    np.testing.assert_allclose(probabilities.sum(), 1.0)
    np.testing.assert_allclose(probabilities[:3], [2 / 9, 1 / 3, 4 / 9], rtol=1e-6)
    assert probabilities[-2:].sum() == 0.0


def test_brandao_reward_profile_round_trips_and_reaches_environment():
    config = ExperimentConfig(
        experiment_name="brandao",
        num_generations=1,
        reward_profile="brandao_reward",
    )
    restored = ExperimentConfig.from_dict(config.to_dict())

    assert restored.reward_profile == "brandao_reward"
    assert _make_env(restored).reward_profile.value == "brandao_reward"


def test_brandao_shared_architecture_selects_paper_style_training_contract():
    config = ExperimentConfig(
        experiment_name="brandao-shared",
        num_generations=1,
        policy_architecture="brandao_shared",
    )
    restored = ExperimentConfig.from_dict(config.to_dict())
    env = _make_env(restored)

    assert restored.reward_profile == "brandao_reward"
    assert restored.hidden_sizes == (300, 200, 100)
    assert restored.max_episode_steps == 400
    assert restored.terminate_on_goal
    assert env.observation_spec.shape == (3, 424)
    assert env.action_spec.shape == (3, 2)


def test_brandao_set_architecture_enables_invariant_encoder_and_mirror_augmentation():
    config = ExperimentConfig(
        experiment_name="brandao-sets",
        num_generations=1,
        policy_architecture="brandao_sets",
    )
    env = _make_env(config)

    assert config.reward_profile == "brandao_reward"
    assert config.mirror_augmentation is True
    assert config.ppo_config().mirror_augmentation is True
    assert env.policy_architecture == "brandao_sets"
    assert env.observation_spec.shape == (3, 424)
    assert env.action_spec.shape == (3, 2)


def test_experiment_config_rejects_unknown_reward_profile():
    with pytest.raises(ValueError, match="reward_profile"):
        ExperimentConfig(
            experiment_name="bad-profile",
            num_generations=1,
            reward_profile="unknown",
        )


def test_selector_weights_allow_explicit_random_and_stationary_mix():
    weights = OpponentSamplingWeights(
        all_history=50,
        recent_half=0,
        latest=0,
        random=25,
        stationary=25,
    )
    probabilities = weights.probabilities([0, 10])

    np.testing.assert_allclose(probabilities, [0.25, 0.25, 0.25, 0.25])


def test_snapshot_archive_round_trips_and_is_lineage_scoped(tmp_path):
    params = {"dense": {"kernel": jnp.arange(6, dtype=jnp.float32).reshape(2, 3)}}
    archive = SnapshotArchive(tmp_path, "trial", lineage=2)
    archive.save(params, 0, seed=12, hidden_sizes=(8, 8))

    loaded = archive.load(params, 0, expected_hidden_sizes=(8, 8))
    pool, generations = archive.load_pool(params)
    other_lineage = SnapshotArchive(tmp_path, "trial", lineage=1)

    np.testing.assert_array_equal(loaded["dense"]["kernel"], params["dense"]["kernel"])
    np.testing.assert_array_equal(pool["dense"]["kernel"][0], params["dense"]["kernel"])
    assert generations == (0,)
    assert other_lineage.list_snapshots() == []


def test_active_pool_limit_does_not_prune_archived_snapshots(tmp_path):
    params = {"weights": jnp.ones((2, 3), dtype=jnp.float32)}
    archive = SnapshotArchive(tmp_path, "trial", lineage=0)
    for generation in (0, 10, 20):
        archive.save(params, generation, seed=0, hidden_sizes=(8,))

    _, active_generations = archive.load_pool(params, active_limit=2)

    assert active_generations == (10, 20)
    assert [item["generation"] for item in archive.list_snapshots()] == [0, 10, 20]


def test_snapshot_archive_rejects_architecture_mismatch(tmp_path):
    params = {"weights": jnp.ones((2, 3), dtype=jnp.float32)}
    archive = SnapshotArchive(tmp_path, "trial", lineage=0)
    archive.save(params, 0, seed=0, hidden_sizes=(8,))

    with pytest.raises(ValueError, match="architecture"):
        archive.load(params, 0, expected_hidden_sizes=(16,))


def test_ppo_update_runs_against_historical_policy_pool():
    config = PPOConfig(
        num_envs=2,
        rollout_length=2,
        update_epochs=1,
        num_minibatches=1,
        hidden_sizes=(8,),
    )
    trainer = PPO(VSSJumanjiEnv(max_episode_steps=1), config)
    initial = trainer.initialize(jax.random.PRNGKey(4))
    first = initial.train_state.params
    second = jax.tree_util.tree_map(lambda value: value + 0.01, first)
    pool = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), first, second)
    probabilities = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    runner = trainer.initialize(jax.random.PRNGKey(5), pool, probabilities)

    updated, metrics = trainer.update(runner, pool, probabilities)
    jax.block_until_ready((updated, metrics))

    changed = [
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree_util.tree_leaves(runner.train_state.params),
            jax.tree_util.tree_leaves(updated.train_state.params),
            strict=True,
        )
    ]
    assert any(changed)
    assert not hasattr(updated, "opponent_params")
    assert updated.opponent_indices.shape == (config.num_envs,)
    assert int(metrics["env_steps"]) == config.batch_size
    assert metrics["blue_goals_by_env"].shape == (config.num_envs,)
    assert metrics["yellow_goals_by_env"].shape == (config.num_envs,)
    assert metrics["referee_event_counts"].shape == (len(RefereeEvent),)
    assert int(metrics["referee_event_counts"].sum()) == (
        config.rollout_length * config.num_envs
    )
    assert all(np.isfinite(np.asarray(value)).all() for value in metrics.values())


def test_single_opponent_pool_keeps_policy_shared_across_environments():
    config = PPOConfig(
        num_envs=3,
        rollout_length=1,
        update_epochs=1,
        num_minibatches=1,
        hidden_sizes=(8,),
    )
    trainer = PPO(
        VSSJumanjiEnv(
            max_episode_steps=2,
            reward_profile="brandao_reward",
            policy_architecture="brandao_shared",
        ),
        config,
    )
    initial = trainer.initialize(jax.random.PRNGKey(44))
    params = initial.train_state.params
    pool = jax.tree_util.tree_map(lambda value: value[None, ...], params)
    probabilities = jnp.asarray([0.0, 0.0, 1.0, 0.0, 0.0], dtype=jnp.float32)

    runner = trainer.initialize(jax.random.PRNGKey(45), pool, probabilities)

    assert not hasattr(runner, "opponent_params")
    np.testing.assert_array_equal(runner.opponent_indices, np.zeros(3, dtype=np.int32))
    shared_outputs = trainer._apply_opponent_policy(
        pool, runner.opponent_indices, runner.observation
    )
    per_env_params = jax.tree_util.tree_map(
        lambda value: value[runner.opponent_indices], pool
    )
    per_env_outputs = jax.vmap(trainer.model.apply)(
        per_env_params, runner.observation
    )
    for shared_output, per_env_output in zip(
        shared_outputs, per_env_outputs, strict=True
    ):
        np.testing.assert_allclose(shared_output, per_env_output, rtol=1e-6, atol=1e-6)


def test_brandao_opponent_update_uses_mean_goal_delta_not_win_rate():
    scores = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.int32)

    summary = _opponent_evaluation_summary(scores, threshold=0.6)

    assert summary["evaluation_sample_count"] == 3
    assert summary["mean_goals_for"] == pytest.approx(2 / 3)
    assert summary["mean_goals_against"] == pytest.approx(1 / 3)
    assert summary["mean_goal_delta"] == pytest.approx(1 / 3)
    assert summary["opponent_replaced"] is False

    at_threshold = _opponent_evaluation_summary(
        np.asarray([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.int32),
        threshold=0.6,
    )
    assert at_threshold["mean_goal_delta"] == pytest.approx(0.6)
    assert at_threshold["opponent_replaced"] is True


def test_gpu_required_experiment_rejects_unavailable_gpu(monkeypatch):
    def unavailable_gpu(_backend):
        raise RuntimeError("no GPU backend")

    monkeypatch.setattr(jax, "devices", unavailable_gpu)

    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        _require_gpu_backend()


def test_tiny_brandao_run_logs_candidate_opponent_evaluation_and_env_metrics(tmp_path):
    config = ExperimentConfig(
        experiment_name="brandao-evaluation",
        num_generations=1,
        num_lineages=1,
        num_envs_per_lineage=1,
        rollout_length=2,
        update_epochs=1,
        num_minibatches=1,
        seed=11,
        snapshot_interval=1,
        max_episode_steps=2,
        init_mode="random",
        reward_profile="brandao_reward",
        policy_architecture="brandao_shared",
        evaluation_matches=2,
        output_root=str(tmp_path),
    )

    experiment_dir = run_experiment(config)
    row = json.loads((experiment_dir / "metrics.jsonl").read_text().splitlines()[0])

    assert row["generation"] == 1
    assert len(row["blue_goals_by_env"]) == 1
    assert len(row["yellow_goals_by_env"]) == 1
    assert row["mean_goal_delta_per_env"] == pytest.approx(
        row["blue_goals_by_env"][0] - row["yellow_goals_by_env"][0]
    )
    assert row["referee_event_counts"]["none"] + sum(
        count
        for event, count in row["referee_event_counts"].items()
        if event != "none"
    ) == 2
    assert row["opponent_evaluated"] is True
    assert row["opponent_generation"] == 0
    assert row["opponent_evaluation_sample_count"] == 2
    assert row["opponent_evaluation_mean_goal_delta"] == pytest.approx(
        row["opponent_evaluation_mean_goals_for"]
        - row["opponent_evaluation_mean_goals_against"]
    )


def test_brandao_run_warm_starts_policy_and_opponent_at_generation_offset(
    tmp_path, monkeypatch
):
    hidden_sizes = (300, 200, 100)
    model = create_policy_model("brandao_shared", hidden_sizes)
    source_policy = model.init(
        jax.random.PRNGKey(17), jnp.zeros((424,), dtype=jnp.float32)
    )
    source_opponent = jax.tree_util.tree_map(lambda value: value + 0.001, source_policy)
    source_archive = SnapshotArchive(tmp_path, "source", lineage=0)
    for params, generation in ((source_opponent, 200), (source_policy, 1000)):
        source_archive.save(
            params,
            generation,
            seed=17,
            hidden_sizes=hidden_sizes,
            observation_size=424,
            action_size=2,
            policy_architecture="brandao_shared",
        )

    config = ExperimentConfig(
        experiment_name="warm-started",
        num_generations=1,
        num_lineages=1,
        num_envs_per_lineage=1,
        rollout_length=2,
        update_epochs=1,
        num_minibatches=1,
        seed=19,
        snapshot_interval=1,
        generation_offset=1000,
        initial_policy_snapshot=str(
            source_archive.directory / "generation_00001000.msgpack"
        ),
        initial_opponent_snapshot=str(
            source_archive.directory / "generation_00000200.msgpack"
        ),
        max_episode_steps=2,
        init_mode="random",
        reward_profile="brandao_reward",
        policy_architecture="brandao_shared",
        evaluation_matches=1,
        output_root=str(tmp_path),
    )

    training_start_params = {}
    original_train_lineages = PPO.train_lineages

    def capture_training_start(trainer, runners, *args, **kwargs):
        training_start_params["params"] = jax.tree_util.tree_map(
            lambda value: np.asarray(value[0]), runners.train_state.params
        )
        return original_train_lineages(trainer, runners, *args, **kwargs)

    monkeypatch.setattr(PPO, "train_lineages", capture_training_start)

    experiment_dir = run_experiment(config)
    archive = SnapshotArchive(tmp_path, "warm-started", lineage=0)
    snapshots = archive.list_snapshots()
    row = json.loads((experiment_dir / "metrics.jsonl").read_text().splitlines()[0])
    warm_start = json.loads((experiment_dir / "warm_start.json").read_text())

    assert [item["generation"] for item in snapshots] == [200, 1000, 1001]
    assert row["generation"] == 1001
    assert row["opponent_generation"] == 200
    assert row["opponent_replaced"] is False
    assert row["selected_opponent_generation"] == 200
    for expected, actual in zip(
        jax.tree_util.tree_leaves(source_policy),
        jax.tree_util.tree_leaves(training_start_params["params"]),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert row["opponent_evaluated"] is True
    assert warm_start["generation_offset"] == 1000
    assert warm_start["optimizer_state_restored"] is False
    loaded_initial = archive.load(
        source_policy,
        1000,
        expected_hidden_sizes=hidden_sizes,
        expected_policy_architecture="brandao_shared",
    )
    for expected, actual in zip(
        jax.tree_util.tree_leaves(source_policy),
        jax.tree_util.tree_leaves(loaded_initial),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_tiny_experiment_keeps_lineage_archives_separate(tmp_path):
    config = ExperimentConfig(
        experiment_name="tiny",
        num_generations=1,
        num_lineages=2,
        num_envs_per_lineage=2,
        rollout_length=2,
        update_epochs=1,
        num_minibatches=1,
        hidden_sizes=(8,),
        max_episode_steps=1,
        output_root=str(tmp_path),
    )

    experiment_dir = run_experiment(config)

    assert (experiment_dir / "config.json").is_file()
    assert len(list((experiment_dir / "metrics.jsonl").read_text().splitlines())) == 2
    model = ActorCritic(action_size=6, hidden_sizes=config.hidden_sizes)
    template = model.init(jax.random.PRNGKey(0), jnp.zeros((46,), dtype=jnp.float32))
    lineage_params = []
    for lineage in range(2):
        archive = SnapshotArchive(tmp_path, "tiny", lineage)
        assert [item["generation"] for item in archive.list_snapshots()] == [0, 1]
        lineage_params.append(
            archive.load(template, 0, expected_hidden_sizes=config.hidden_sizes)
        )
    assert any(
        not np.array_equal(np.asarray(first), np.asarray(second))
        for first, second in zip(
            jax.tree_util.tree_leaves(lineage_params[0]),
            jax.tree_util.tree_leaves(lineage_params[1]),
            strict=True,
        )
    )


def test_short_evaluation_writes_long_match_metrics(tmp_path):
    config = ExperimentConfig(
        experiment_name="evaluation",
        num_generations=1,
        num_lineages=1,
        num_envs_per_lineage=1,
        rollout_length=1,
        update_epochs=1,
        num_minibatches=1,
        hidden_sizes=(8,),
        max_episode_steps=1,
        reward_profile="brandao_reward",
        evaluation_matches=1,
        evaluation_seconds=1,
        output_root=str(tmp_path),
    )
    model = ActorCritic(action_size=6, hidden_sizes=config.hidden_sizes)
    params = model.init(jax.random.PRNGKey(2), jnp.zeros((46,), dtype=jnp.float32))
    archive = SnapshotArchive(tmp_path, "evaluation", lineage=0)
    archive.save(params, 0, seed=0, hidden_sizes=config.hidden_sizes)
    archive.save(params, 1, seed=0, hidden_sizes=config.hidden_sizes)

    result_path = evaluate_experiment(
        config, lineage=0, generation=1, matches=1, opponents="all", batch_size=1
    )
    result = json.loads(result_path.read_text())

    assert result["control_steps"] == 15
    assert result["reward_profile"] == "brandao_reward"
    assert [item["opponent_type"] for item in result["results"]] == [
        "history",
        "random",
        "stationary",
    ]
    for item in result["results"]:
        summary = item["summary"]
        assert summary["matches"] == 1
        assert summary["wins"] + summary["draws"] + summary["losses"] == 1
