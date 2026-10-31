# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic learning, replay boundaries and deployment on a synthetic VecEnv."""

from __future__ import annotations

import copy
import math
import torch
from pathlib import Path
from tensordict import TensorDict

import onnx
import pytest

from rsl_rl.algorithms import SuperMimic
from rsl_rl.algorithms.supermimic import lambda_returns
from rsl_rl.algorithms.supermimic_task import DifferentiableTask, EuclideanStateRepresentation, State
from rsl_rl.env import VecEnv
from rsl_rl.modules import WorldModel
from rsl_rl.runners import OffPolicyRunner


class ScalarTask(DifferentiableTask):
    """A Euclidean system with privileged context and one successor event."""

    num_labels = 1

    def __init__(self, env: VecEnv, device: str) -> None:
        """Bind a one-dimensional Euclidean state."""
        self.representation = EuclideanStateRepresentation(1, error_floor=0.01)

    def actor_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> torch.Tensor:
        """Expose position and the most recent action."""
        return torch.cat((state[0], prev_actions[..., -1, :]), -1)

    def actor_goal(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        """Use the explicit deployable goal unchanged."""
        return goal

    def dynamics_context(self, state: State, context: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        """Expose privileged context and both action predecessors to dynamics."""
        return torch.cat((context, prev_actions.flatten(-2)), -1)

    def critic_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> torch.Tensor:
        """Build privileged scalar-value inputs."""
        return torch.cat((state[0], goal, context, prev_actions.flatten(-2)), -1)

    def advance(
        self, next_state: State, goal: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hold exogenous goals and context fixed."""
        return goal, context

    def reward(
        self,
        state: State,
        actions: torch.Tensor,
        next_state: State,
        goal: torch.Tensor,
        prev_actions: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reward goal tracking and penalize action magnitude."""
        target = goal[..., :1] if goal.shape[-1] else torch.zeros_like(next_state[0])
        reward = 1 - (next_state[0] - target).square().squeeze(-1) - 0.01 * actions.square().sum(-1)
        return reward if labels is None else reward + 0.01 * labels[..., 0]

    def termination_prob(
        self, next_state: State, goal: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Keep the synthetic physical system alive in imagination."""
        return torch.zeros_like(next_state[0][..., 0])


class ScalarEnv(VecEnv):
    """Owned, pre-reset observations with optional interval pushes and nonzero reset actions."""

    def __init__(
        self,
        device: str = "cpu",
        goal_dim: int = 1,
        context_dim: int = 1,
        reset_action: float = 0.0,
        pushes: bool = False,
    ) -> None:
        """Allocate a small vectorized system with explicit physical observation groups."""
        self.num_envs, self.num_actions = 4, 1
        self.device, self.cfg = device, {}
        self.max_episode_length = 5
        self.episode_length_buf = torch.zeros(4, dtype=torch.long, device=device)
        self.position = torch.zeros(4, 1, device=device)
        self.goal = torch.ones(4, goal_dim, device=device)
        self.context = torch.zeros(4, context_dim, device=device)
        self.previous = torch.full((4, 1), reset_action, device=device)
        self.previous_previous = self.previous.clone()
        self.reset_action, self.pushes = reset_action, pushes

    def get_observations(self) -> TensorDict:
        """Return owned physical observations and reported action predecessors."""
        return TensorDict(
            {
                "state": self.position.clone(),
                "goal": self.goal.clone(),
                "context": self.context.clone(),
                "labels": (self.position > 0).float(),
                "prev_action": self.previous.clone(),
                "prev_prev_action": self.previous_previous.clone(),
            },
            batch_size=[4],
            device=self.device,
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Advance, capture physical successors, then apply pushes and resets."""
        self.position.add_(0.2 * actions)
        self.previous_previous.copy_(self.previous)
        self.previous.copy_(actions)
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= self.max_episode_length
        final = self.get_observations()
        rewards = 1 - (self.position[:, 0] - 1).square()
        if self.pushes:
            self.position[~dones] += 10
        self.position[dones] = 0
        self.episode_length_buf[dones] = 0
        self.previous[dones] = self.reset_action
        self.previous_previous[dones] = self.reset_action
        return (
            self.get_observations(),
            rewards,
            dones,
            {"final_obs": final, "final_obs_every_step": True, "time_outs": dones.clone()},
        )


def _cfg(**options: object) -> dict:
    algorithm = {
        "class_name": "SuperMimic",
        "task_cfg": {"class_name": ScalarTask},
        "updates_per_step": 1.0,
        "replay_buffer_size": 64,
        "warmup_steps": 4,
        "world_model_window": 3,
        "num_world_model_updates": 1,
        "world_model_batch_size": 8,
        "world_model_warmup_iterations": 0,
        "horizon": 3,
        "policy_batch_size": 4,
        "critic_epochs": 1,
        "critic_minibatches": 2,
    }
    algorithm.update(options)
    return {
        "algorithm": algorithm,
        "num_steps_per_env": 3,
        "save_interval": 100,
        "multi_gpu": None,
        "obs_groups": {key: [key] for key in ("state", "goal", "context", "labels", "prev_action", "prev_prev_action")},
        "actor": {
            "class_name": "SuperMimicActor",
            "history_length": 2,
            "encoder_hidden_dims": (8,),
            "decoder_hidden_dims": (8,),
        },
        "world_model": {"class_name": "WorldModel", "hidden_dims": (16,), "label_hidden_dims": (8,)},
        "critic": {"class_name": "MLPModel", "hidden_dims": (8,), "obs_normalization": True},
    }


def _collect(learner: SuperMimic, env: ScalarEnv, count: int, inference: bool = True) -> None:
    with torch.inference_mode(inference), torch.no_grad():
        obs = env.get_observations()
        for _ in range(count):
            actions = learner.act(obs)
            obs, reward, done, extras = env.step(actions)
            learner.process_env_step(obs, reward, done, extras)


def _learner(env: ScalarEnv | None = None, **options: object) -> SuperMimic:
    env = ScalarEnv() if env is None else env
    return SuperMimic.construct_algorithm(env.get_observations(), env, _cfg(**options), env.device)


def test_world_model_preserves_state_and_action_gradients_when_weights_are_frozen() -> None:
    """Frozen dynamics weights preserve the Jacobians needed for imagined actor learning."""
    representation = EuclideanStateRepresentation(1)
    model = WorldModel(representation, 1, 1, 1, hidden_dims=(8,))
    with torch.no_grad():
        model.net.output.weight.normal_(0, 0.1)
    model.requires_grad_(False)
    state, action = torch.ones(4, 1, requires_grad=True), torch.ones(4, 1, requires_grad=True)
    model((state,), action, torch.zeros(4, 1))[0].square().sum().backward()
    assert state.grad.abs().sum() > 0 and action.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_direct_and_standard_runner_train_all_models() -> None:
    """The standard loop and direct interface produce finite updates of actor, model and critic."""
    torch.manual_seed(2)
    runner = OffPolicyRunner(ScalarEnv(), _cfg())
    modules = (runner.alg.actor, runner.alg.world_model, runner.alg.critic)
    before = [copy.deepcopy(module.state_dict()) for module in modules]
    runner.learn(3)
    assert runner.alg.iteration > 0
    for old, module in zip(before, modules):
        assert any(not torch.equal(old[name], value) for name, value in module.named_parameters())
    learner, env = _learner(), ScalarEnv()
    _collect(learner, env, 4)
    metrics = learner.update()
    assert {"world_model_loss", "policy_loss", "critic_loss", "label_loss"} <= metrics.keys()
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["policy_grad_norm"] > 0
    for module in (learner.actor, learner.world_model):
        assert all(not buffer.is_inference() for buffer in module.buffers())
    checkpoint = copy.deepcopy(learner.save())
    learner.load(checkpoint, None, strict=True)
    _collect(learner, env, 4, inference=False)
    assert torch.isfinite(torch.tensor(learner.update()["world_model_loss"]))


def test_default_experiment_flags_are_bit_exact() -> None:
    """Omitted flags retain the legacy imagination, clipping, critic, warmup and checkpoint behavior."""

    def update(**options: object) -> tuple[dict[str, float], dict, torch.Tensor]:
        torch.manual_seed(11)
        learner, env = _learner(world_model_warmup_iterations=1, **options), ScalarEnv()
        _collect(learner, env, 4)
        torch.manual_seed(12)
        metrics = learner.update()
        return metrics, copy.deepcopy(learner.save()), torch.get_rng_state()

    implicit = update()
    explicit = update(
        noise_faithful_imagination=False,
        actor_grad_clip_value=0.1,
        num_critics=1,
        actor_value_reduction="mean",
        critic_zero_init=False,
        critic_warmup=False,
        entropy_regularization=False,
        init_alpha=1.0,
        target_entropy_scale=0.5,
        alpha_learning_rate=5e-3,
    )
    assert implicit[0] == explicit[0] and "policy_loss" in implicit[0]
    assert (
        implicit[1].keys()
        == explicit[1].keys()
        == {
            "actor_state_dict",
            "critic_state_dict",
            "world_model_state_dict",
            "world_model_optimizer_state_dict",
            "actor_optimizer_state_dict",
            "critic_optimizer_state_dict",
            "iteration",
        }
    )
    assert "log_std" not in implicit[1]["actor_state_dict"]
    for key, value in implicit[1].items():
        if key.endswith("optimizer_state_dict"):
            torch.testing.assert_close(value["state"], explicit[1][key]["state"], rtol=0, atol=0)
            assert value["param_groups"] == explicit[1][key]["param_groups"]
        else:
            torch.testing.assert_close(value, explicit[1][key], rtol=0, atol=0)
    assert torch.equal(implicit[2], explicit[2])


def test_noise_faithful_imagination_uses_noisy_actions_for_dynamics_reward_and_history() -> None:
    """Imagined collection noise reaches each action consumer while retaining actor gradients."""
    learner, env = _learner(noise_faithful_imagination=True, action_noise=0.1, horizon=2), ScalarEnv()
    _collect(learner, env, 4)
    means, actions, reward_actions, reward_previous, histories = [], [], [], [], []
    actor_forward, predict, reward = learner._actor_forward, learner._predict, learner.task.reward

    def checked_actor(history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        histories.append(history.detach().clone())
        mean = actor_forward(history, goal)
        means.append(mean.detach().clone())
        return mean

    def checked_predict(
        state: State, action: torch.Tensor, context: torch.Tensor, previous: torch.Tensor
    ) -> tuple[State, torch.Tensor | None, torch.Tensor | None]:
        actions.append(action.detach().clone())
        return predict(state, action, context, previous)

    def checked_reward(
        state: State,
        action: torch.Tensor,
        next_state: State,
        goal: torch.Tensor,
        previous: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reward_actions.append(action.detach().clone())
        reward_previous.append(previous.detach().clone())
        return reward(state, action, next_state, goal, previous, labels)

    learner._actor_forward = checked_actor
    learner._predict = checked_predict
    learner.task.reward = checked_reward
    torch.manual_seed(13)
    metrics = learner._fit_policy()
    noise = torch.stack([action - mean for action, mean in zip(actions, means)])
    torch.testing.assert_close(metrics["imagined_action_noise_rms"], noise.square().mean().sqrt())
    assert noise.abs().sum() > 0 and metrics["policy_grad_norm"] > 0
    for dynamics_action, reward_action in zip(actions, reward_actions):
        torch.testing.assert_close(dynamics_action, reward_action, rtol=0, atol=0)
    torch.testing.assert_close(reward_previous[1][:, -1], actions[0], rtol=0, atol=0)
    torch.testing.assert_close(histories[1][:, -1, -1], actions[0].squeeze(-1), rtol=0, atol=0)


def test_actor_gradient_value_clip_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional actor clip controls the post-norm elementwise clipping call."""
    calls = []
    monkeypatch.setattr(torch.nn.utils, "clip_grad_value_", lambda parameters, value: calls.append(value))
    for value, expected in ((0.1, [0.1]), (None, [])):
        learner, env = _learner(actor_grad_clip_value=value), ScalarEnv()
        _collect(learner, env, 4)
        calls.clear()
        learner._fit_policy()
        assert calls == expected


def test_replay_start_padding_repeats_the_same_noisy_reset_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unavailable history repeats one corrupted frame, as during collection."""
    torch.manual_seed(0)
    env, cfg = ScalarEnv(), _cfg()
    cfg["actor"]["history_length"] = 5
    learner = SuperMimic.construct_algorithm(env.get_observations(), env, cfg, env.device)
    _collect(learner, env, 7)
    index = learner.storage.sample(64, history=4)
    assert (~index.valid).any() and index.valid.all(dim=1).any()
    monkeypatch.setattr(learner.storage, "sample", lambda *args, **kwargs: index)
    monkeypatch.setattr(learner.task, "add_noise", lambda frames: frames + torch.randn_like(frames))
    _, frames = learner._starts(64)
    first = frames[torch.arange(64), index.valid.long().argmax(dim=1)]
    torch.testing.assert_close(frames[~index.valid], first[:, None].expand_as(frames)[~index.valid], rtol=0, atol=0)
    assert (frames[index.valid.all(dim=1), 0] != frames[index.valid.all(dim=1), -1]).any()


def test_inference_device_moves_task_tensors() -> None:
    """Task observation constants move together with policy parameters and history."""
    learner = _learner()
    learner.task.offset = torch.ones(1)
    policy = learner.get_policy().to("meta")
    assert policy.history.device.type == "meta"
    assert learner.task.offset.device.type == "meta"


def test_learning_rate_tracks_learner_iterations_and_resume() -> None:
    """All optimizers share one clamped schedule despite unequal update counts."""
    learner, env = _learner(lr_decay_iterations=8, num_world_model_updates=2), ScalarEnv()
    _collect(learner, env, 4)
    learner.update()
    assert learner.iteration == 4
    expected = 2e-4 * (0.03 + 0.97 * 0.5)
    assert all(optimizer.param_groups[0]["lr"] == pytest.approx(expected) for optimizer in learner.optimizers.values())
    restored = _learner(lr_decay_iterations=8, num_world_model_updates=2)
    restored.load(copy.deepcopy(learner.save()), None, True)
    _collect(restored, ScalarEnv(), 8)
    restored.update()
    assert all(
        optimizer.param_groups[0]["lr"] == pytest.approx(2e-4 * 0.03) for optimizer in restored.optimizers.values()
    )


def test_missing_final_observations_fail_even_without_dones() -> None:
    """Done-only producers cannot silently train a model on interval pushes."""
    env, learner = ScalarEnv(), _learner()
    learner.act(env.get_observations())
    with pytest.raises(ValueError, match="every-step"):
        learner.process_env_step(env.get_observations(), torch.zeros(4), torch.zeros(4), {})
    with pytest.raises(ValueError, match="every-step"):
        learner.process_env_step(
            env.get_observations(), torch.zeros(4), torch.zeros(4), {"final_obs": env.get_observations()}
        )


def test_interval_edits_cut_only_future_world_model_targets() -> None:
    """Pre-push successors remain unchanged in replay; actor histories continue across pushes."""
    env = ScalarEnv(pushes=True)
    learner = _learner(env)
    _collect(learner, env, 4)
    assert learner.storage.data["window_break"][:4].all()
    window, mask = learner._windows(64)
    assert mask[:, 0].all() and not mask[:, 1:].any()
    assert (window["next_state"][:, 0] < window["state"][:, 0] + 1).all()
    metrics = learner.update()
    assert metrics["window_break_fraction"] == 1.0
    _, history = learner._starts(64)
    assert (history[:, 0, 0] != history[:, -1, 0]).any()


def test_nonzero_crouch_reset_actions_seed_both_predecessors() -> None:
    """Explicit action sets preserve held reset commands instead of assuming zero actions."""
    env = ScalarEnv(reset_action=0.4)
    learner = _learner(env)
    _collect(learner, env, 5)
    torch.testing.assert_close(learner.actor.previous_actions, torch.full((4, 2, 1), 0.4))
    learner.act(env.get_observations())
    torch.testing.assert_close(learner._last_previous_actions, torch.full((4, 2, 1), 0.4))
    assert learner.actor.history[:, -1, -1].eq(0.4).all()


def test_lambda_returns_terminal_reward_and_bootstrap() -> None:
    """A terminal transition retains its reward and excludes future values."""
    actual = lambda_returns(
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor([[0.0], [1.0], [0.0]]),
        torch.tensor([[7.0], [8.0], [9.0], [10.0]]),
        0.5,
        1.0,
    )
    torch.testing.assert_close(actual, torch.tensor([[2.0], [2.0], [8.0]]))


def test_value_cap_bounds_policy_bootstrap_and_critic_targets() -> None:
    """Limit optimistic bootstraps and targets while fitting the uncapped critic."""
    learner, env = _learner(max_step_reward=1.0, gamma=0.5, horizon=1, critic_minibatches=1), ScalarEnv()
    _collect(learner, env, 4)
    with torch.no_grad():
        for parameter in learner.critic.parameters():
            parameter.zero_()
        learner.critic.mlp.output.bias.fill_(10.0)
    metrics = learner._fit_policy()
    assert metrics["policy_loss"].item() == pytest.approx(-metrics["imagined_reward"].item() - 1.0)
    assert metrics["terminal_saturation"].item() == 1.0

    with torch.no_grad():
        learner.critic.mlp.output.bias.fill_(10.0)
    features = torch.zeros(2, 2, learner.critic.obs_dim)
    metrics = learner._fit_critic(features, torch.tensor([[0.0, 4.0]]), torch.zeros(1, 2))
    assert metrics["critic_loss"].item() == pytest.approx((9.0**2 + 8.0**2) / 2)
    assert 2.0 < learner.critic.mlp.output.bias.item() < 10.0
    values = torch.tensor([1.0, 2.2, 3.0])
    torch.testing.assert_close(learner._cap(values, torch.tensor(0.25)), torch.tensor([1.0, 2.2, 2.5]))
    torch.testing.assert_close(learner._cap(values, torch.tensor(-0.25)), torch.tensor([1.0, 2.0, 2.0]))


@pytest.mark.parametrize(("reduction", "bootstrap"), [("mean", 2.0), ("min", 1.0)])
def test_twin_critics_fit_min_targets_and_reduce_actor_bootstrap(reduction: str, bootstrap: float) -> None:
    """Independent twin critics fit targets from their minimum; the actor bootstraps from the configured reduction."""
    learner, env = (
        _learner(num_critics=2, actor_value_reduction=reduction, gamma=0.5, horizon=1, critic_minibatches=1),
        ScalarEnv(),
    )
    critic, twin = learner._raw_critic, learner._raw_twin_critic
    assert set(learner.optimizers) == {"world_model", "actor", "critic", "twin_critic"}
    assert critic.state_dict().keys() == twin.state_dict().keys()
    assert any(not torch.equal(first, second) for first, second in zip(critic.parameters(), twin.parameters()))
    _collect(learner, env, 4)

    def set_values() -> None:
        with torch.no_grad():
            for model, value in ((critic, 1.0), (twin, 3.0)):
                for parameter in model.parameters():
                    parameter.zero_()
                model.mlp.output.bias.fill_(value)

    set_values()
    metrics = learner._fit_policy()
    assert metrics["policy_loss"].item() == pytest.approx(-metrics["imagined_reward"].item() - 0.5 * bootstrap)
    set_values()
    metrics = learner._fit_critic(torch.zeros(2, 2, critic.obs_dim), torch.zeros(1, 2), torch.zeros(1, 2))
    assert metrics["critic_loss"].item() == pytest.approx(0.5**2)
    assert metrics["twin_critic_loss"].item() == pytest.approx(2.5**2)
    with pytest.raises(ValueError, match="twin critics"):
        _learner(num_critics=3)


def test_entropy_regularization_learns_noise_and_adds_dt_scaled_soft_rewards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Learned noise drives collection and reparameterized imagination; critic targets include the same bonus."""
    learner, env = _learner(entropy_regularization=True, init_alpha=0.5, target_entropy_scale=0.25), ScalarEnv()
    learner.task.dt = 0.5
    torch.testing.assert_close(learner.actor.output_std, torch.full((1,), 0.1))
    assert any(
        parameter is learner.actor.log_std for parameter in learner.optimizers["actor"].param_groups[0]["params"]
    )
    with monkeypatch.context() as patch, torch.no_grad():
        patch.setattr(torch, "randn_like", torch.ones_like)
        learner.actor.log_std.fill_(math.log(3.0))
        obs = env.get_observations()
        actions = learner.act(obs)
        mean = learner.actor.forward_history(learner.actor.history, obs["goal"])
        torch.testing.assert_close(actions - mean, torch.full_like(actions, 3.0))
        learner.actor.log_std.fill_(math.log(0.1))
    _collect(learner, env, 4)
    rewards, soft_rewards = [], []
    reward, fit_critic = learner.task.reward, learner._fit_critic

    def checked_reward(*args: object, **kwargs: object) -> torch.Tensor:
        value = reward(*args, **kwargs)
        rewards.append(value.detach())
        return value

    def checked_fit(
        features: torch.Tensor, soft: torch.Tensor, terminals: torch.Tensor, bonus: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        soft_rewards.append(soft)
        return fit_critic(features, soft, terminals, bonus)

    learner.task.reward, learner._fit_critic = checked_reward, checked_fit
    log_std = learner.actor.log_std.detach().clone()
    metrics = learner._fit_policy()
    entropy = 0.5 * math.log(2 * math.pi * math.e) + math.log(0.1)
    bonus = 0.5 * 0.5 * (entropy + 0.25) / (2 * 0.25)
    assert metrics["entropy"].item() == pytest.approx(entropy)
    assert metrics["alpha"].item() == pytest.approx(0.5)
    assert metrics["action_std"].item() == pytest.approx(0.1)
    assert metrics["entropy_bonus"].item() == pytest.approx(bonus)
    torch.testing.assert_close(soft_rewards[0], torch.stack(rewards) + bonus)
    assert metrics["imagined_action_noise_rms"] > 0 and learner.actor.log_std.grad.abs().sum() > 0
    assert not torch.equal(learner.actor.log_std.detach(), log_std)


@pytest.mark.parametrize(("target_entropy_scale", "alpha_increases"), [(0.5, True), (2.0, False)])
def test_entropy_bonus_widens_noise_and_temperature_tracks_target(
    target_entropy_scale: float, alpha_increases: bool
) -> None:
    """Without task gradients the bonus raises the log-std; the dual step moves the temperature toward the target."""
    learner = _learner(entropy_regularization=True, target_entropy_scale=target_entropy_scale, critic_zero_init=True)
    _collect(learner, ScalarEnv(), 4)
    learner.task.reward = lambda state, actions, *args, **kwargs: torch.zeros_like(actions[..., 0])
    log_std, log_alpha = learner.actor.log_std.item(), learner.log_alpha.item()
    learner._fit_policy()
    assert learner.actor.log_std.item() > log_std
    assert (learner.log_alpha.item() > log_alpha) is alpha_increases


def test_critic_zero_init_starts_every_critic_at_zero() -> None:
    """Zeroed output layers give exactly zero initial values; the default keeps random outputs."""
    features = torch.randn(5, _learner().critic.obs_dim)
    learner = _learner(critic_zero_init=True, num_critics=2)
    for critic in (learner.critic, learner.twin_critic):
        torch.testing.assert_close(learner._value(features, critic), torch.zeros(5), rtol=0, atol=0)
    assert _learner()._value(features).abs().sum() > 0


@pytest.mark.parametrize("critic_warmup", [False, True])
def test_critic_warmup_fits_critics_but_not_actor(critic_warmup: bool) -> None:
    """World-model warmup optionally fits the critics on imagined rollouts of the unchanged actor."""
    learner, env = _learner(world_model_warmup_iterations=100, critic_warmup=critic_warmup, num_critics=2), ScalarEnv()
    _collect(learner, env, 4)
    models = {"actor": learner.actor, "critic": learner._raw_critic, "twin_critic": learner._raw_twin_critic}
    before = {name: copy.deepcopy(model.state_dict()) for name, model in models.items()}
    metrics = learner.update()
    assert learner.iteration == 4 and "policy_loss" not in metrics and not learner.optimizers["actor"].state
    assert ({"critic_loss", "twin_critic_loss"} <= metrics.keys()) is critic_warmup
    for name, model in models.items():
        changed = any(not torch.equal(value, before[name][key]) for key, value in model.state_dict().items())
        assert changed is (critic_warmup and name != "actor")


def test_experiment_flags_combine_and_checkpoints_match_configuration(tmp_path: Path) -> None:
    """All flags train together; twin and temperature state restore under new keys and mismatches are rejected."""
    options = {
        "num_critics": 2,
        "actor_value_reduction": "min",
        "critic_zero_init": True,
        "critic_warmup": True,
        "entropy_regularization": True,
        "noise_faithful_imagination": True,
        "max_step_reward": 1.0,
        "world_model_warmup_iterations": 2,
    }
    runner = OffPolicyRunner(ScalarEnv(), _cfg(**options), str(tmp_path / "logs"))
    runner.learn(3)
    assert runner.alg.iteration > 2
    checkpoint = str(tmp_path / "learner.pt")
    runner.save(checkpoint)
    state = torch.load(checkpoint, weights_only=False)
    keys = {"twin_critic_state_dict", "twin_critic_optimizer_state_dict", "log_alpha", "alpha_optimizer_state_dict"}
    assert keys <= state.keys()
    assert "log_std" in state["actor_state_dict"]
    restored = OffPolicyRunner(ScalarEnv(), _cfg(**options))
    restored.load(checkpoint)
    torch.testing.assert_close(restored.alg.log_alpha, runner.alg.log_alpha)
    torch.testing.assert_close(restored.alg.actor.output_std, runner.alg.actor.output_std)
    torch.testing.assert_close(restored.alg._raw_twin_critic.state_dict(), runner.alg._raw_twin_critic.state_dict())
    assert restored.alg.alpha_optimizer.state and restored.alg.optimizers["twin_critic"].state
    _collect(restored.alg, ScalarEnv(), 4)
    metrics = restored.alg.update()
    assert {"alpha", "entropy", "action_std", "entropy_bonus", "twin_critic_loss"} <= metrics.keys()
    assert all(math.isfinite(value) for value in metrics.values())
    for mismatch in ({}, {"num_critics": 2}, {"entropy_regularization": True}):
        with pytest.raises(ValueError, match="must match"):
            _learner(**mismatch).load(state, None, True)
    with pytest.raises(ValueError, match="number of critics"):
        _learner(num_critics=2).load(copy.deepcopy(_learner().save()), None, True)
    _learner(entropy_regularization=True).load(state, {"actor": True}, True)


@pytest.mark.parametrize(("load_optimizer", "goal_dim"), [(False, 0), (False, 1), (True, 1)])
def test_checkpoint_normalized_jit_and_onnx(tmp_path: Path, load_optimizer: bool, goal_dim: int) -> None:
    """Full loads, fresh optimizers and stateless deployment preserve normalized policy outputs."""
    runner = OffPolicyRunner(ScalarEnv(goal_dim=goal_dim), _cfg(), str(tmp_path / "logs"))
    runner.learn(2)
    checkpoint = str(tmp_path / "learner.pt")
    runner.save(checkpoint)
    restored = OffPolicyRunner(ScalarEnv(goal_dim=goal_dim), _cfg())
    restored.load(checkpoint, load_optimizer=load_optimizer)
    assert restored.alg.iteration == runner.alg.iteration
    assert restored.alg.learning_rate == runner.alg.learning_rate
    assert restored.alg.storage.num_steps == 0 and not restored.alg.actor.history.any()
    for key in runner.alg.optimizers:
        assert bool(restored.alg.optimizers[key].state) is load_optimizer
    history, goal = torch.randn(4, 2, 2), torch.randn(4, goal_dim)
    runner.export_policy_to_jit(str(tmp_path))
    runner.export_policy_to_onnx(str(tmp_path))
    onnx_model = onnx.load(str(tmp_path / "policy.onnx"))
    onnx.checker.check_model(onnx_model)
    assert [value.name for value in onnx_model.graph.input] == (
        ["obs_history", "goal"] if goal_dim else ["obs_history"]
    )
    exported = torch.jit.load(str(tmp_path / "policy.pt"))
    inputs = (history, goal) if goal_dim else (history,)
    torch.testing.assert_close(exported(*inputs), runner.alg.actor.forward_history(history, goal))


def test_checkpoint_selection_dimensions_and_strictness() -> None:
    """Actor-only and full model selections validate tensor shapes through state-dict loading."""
    learner = _learner()
    state = copy.deepcopy(learner.save())
    with pytest.raises(ValueError, match="actor-only or full"):
        learner.load(state, {"actor": True, "critic": True}, True)
    bad = dict(state)
    del bad["world_model_state_dict"]
    with pytest.raises(ValueError, match="explicitly contain world_model"):
        learner.load(bad, None, True)
    changed_context = _learner(ScalarEnv(context_dim=2))
    with pytest.raises(RuntimeError, match="size mismatch"):
        changed_context.load(state, None, True)
    changed_context.load(state, {"actor": True}, True)
    changed_goal = _learner(ScalarEnv(goal_dim=2))
    with pytest.raises(RuntimeError, match="size mismatch"):
        changed_goal.load(state, {"actor": True}, True)
    state["actor_state_dict"]["unexpected"] = torch.ones(1)
    with pytest.raises(RuntimeError):
        learner.load(state, {"actor": True}, True)
    learner.load(state, {"actor": True}, False)


def test_failed_policy_update_restores_frozen_parameters() -> None:
    """Task failures cannot leave model/critic parameters frozen."""
    learner, env = _learner(), ScalarEnv()
    _collect(learner, env, 4)

    def fail(*args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError("task failure")

    learner.task.reward = fail
    with pytest.raises(RuntimeError, match="task failure"):
        learner._fit_policy()
    assert all(
        parameter.requires_grad for module in (learner.world_model, learner.critic) for parameter in module.parameters()
    )


def test_fractional_cadence_and_algorithm_level_multi_gpu_rejection() -> None:
    """Direct per-step updates retain the 2/3-step learner cadence without a custom runner hook."""
    learner, env = _learner(updates_per_step=0.4, warmup_steps=1, world_model_window=1), ScalarEnv()
    counts = []
    for _ in range(5):
        _collect(learner, env, 1)
        learner.update()
        counts.append(learner.iteration)
    assert counts == [0, 0, 1, 1, 2]
    cfg = _cfg()
    missing_state_cfg = _cfg()
    missing_state_cfg["obs_groups"] = {}
    with pytest.raises(ValueError, match="state"):
        SuperMimic.construct_algorithm(TensorDict({"policy": torch.zeros(4, 1)}, [4]), env, missing_state_cfg, "cpu")
    cfg["multi_gpu"] = {"world_size": 2}
    with pytest.raises(NotImplementedError, match="Multi-GPU"):
        SuperMimic.construct_algorithm(env.get_observations(), env, cfg, "cpu")


def test_task_analytic_successors_follow_labels_in_fitting_and_imagination() -> None:
    """Analytic post-dynamics runs after event prediction and retains the event-probability gradient."""
    learner, env = _learner(), ScalarEnv()
    post_calls = []
    reward = learner.task.reward

    def post(state: State, next_state: State, labels: torch.Tensor | None) -> State:
        successor = (next_state[0] + labels[..., :1],)
        post_calls.append(successor)
        return successor

    def checked_reward(
        state: State,
        actions: torch.Tensor,
        next_state: State,
        goal: torch.Tensor,
        prev_actions: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert next_state is post_calls[-1]
        return reward(state, actions, next_state, goal, prev_actions, labels)

    learner.task.post_dynamics = post
    learner.task.reward = checked_reward
    state = (torch.ones(4, 1),)
    successor, _, labels = learner._predict(state, torch.zeros(4, 1), torch.zeros(4, 1), torch.zeros(4, 2, 1))
    assert labels is not None
    torch.testing.assert_close(torch.autograd.grad(successor[0].sum(), labels)[0], torch.ones_like(labels))
    _collect(learner, env, 4)
    post_calls.clear()
    learner._fit_world_model()
    assert len(post_calls) == learner.world_model_window
    post_calls.clear()
    assert learner._fit_policy()["policy_grad_norm"] > 0
    assert len(post_calls) == learner.horizon


def test_collection_owns_start_observations() -> None:
    """Replay starts survive an environment reusing its observation tensors during stepping."""
    learner, env = _learner(), ScalarEnv()
    obs = env.get_observations()
    actions = learner.act(obs)
    obs["state"].fill_(100)
    obs["goal"].fill_(200)
    obs["context"].fill_(300)
    successor, rewards, dones, extras = env.step(actions)
    learner.process_env_step(successor, rewards, dones, extras)
    assert learner.storage.data["state"][0].eq(0).all()
    assert learner.storage.data["goal"][0].eq(1).all()
    assert learner.storage.data["context"][0].eq(0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")
@pytest.mark.parametrize(
    ("mixed_precision", "compile_mode", "options"),
    [
        (False, None, {}),
        (True, None, {}),
        (True, "default", {}),
        (
            True,
            "default",
            {
                "num_critics": 2,
                "entropy_regularization": True,
                "critic_warmup": True,
                "world_model_warmup_iterations": 1,
            },
        ),
    ],
)
def test_cuda_learning_and_checkpoint_roundtrip(mixed_precision: bool, compile_mode: str | None, options: dict) -> None:
    """CUDA FP32, BF16 and compiled learning preserve finite model updates and raw checkpoint keys."""
    cfg = _cfg(use_mixed_precision=mixed_precision, **options)
    cfg["torch_compile_mode"] = compile_mode
    torch.manual_seed(2)
    runner = OffPolicyRunner(ScalarEnv(device="cuda"), cfg, device="cuda")
    modules = {"actor": runner.alg.actor, "world_model": runner.alg.world_model, "critic": runner.alg._raw_critic}
    if runner.alg._raw_twin_critic is not None:
        modules["twin_critic"] = runner.alg._raw_twin_critic
    before = [copy.deepcopy(module.state_dict()) for module in modules.values()]
    runner.learn(2)
    checkpoint = copy.deepcopy(runner.alg.save())
    for old, module in zip(before, modules.values()):
        assert any(not torch.equal(old[name], value) for name, value in module.named_parameters())
        assert all(torch.isfinite(value).all() for value in module.state_dict().values())
    assert not any("_orig_mod" in key for name in modules for key in checkpoint[f"{name}_state_dict"])
    restored = _learner(ScalarEnv(device="cuda"), use_mixed_precision=mixed_precision, **options)
    restored.load(checkpoint, None, True)
    history, goal = torch.randn(4, 2, 2, device="cuda"), torch.randn(4, 1, device="cuda")
    torch.testing.assert_close(
        restored.actor.forward_history(history, goal), runner.alg.actor.forward_history(history, goal)
    )


def test_task_without_successor_event_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tasks without event heads train through the same physical replay and imagination paths."""
    env, cfg = ScalarEnv(), _cfg()
    monkeypatch.setattr(ScalarTask, "num_labels", 0)
    del cfg["obs_groups"]["labels"]
    learner = SuperMimic.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    _collect(learner, env, 4)
    metrics = learner.update()
    assert learner.world_model.label_head is None and "label_loss" not in metrics
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["policy_grad_norm"] > 0
