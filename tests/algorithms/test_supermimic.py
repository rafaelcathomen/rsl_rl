# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulator-independent learning checks on a differentiable scalar control system."""

from __future__ import annotations

import copy
import torch
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import pytest

from rsl_rl.algorithms.supermimic import SuperMimic, SuperMimicCfg, lambda_returns
from rsl_rl.env.model_based_env import DifferentiableTask, State
from rsl_rl.models.supermimic_model import WorldModel
from rsl_rl.runners import SuperMimicRunner, supermimic_runner
from rsl_rl.storage.sequence_replay_buffer import SequenceReplayBuffer


class ScalarRepresentation:
    """Euclidean state demonstrates that no floating-base or SO(3) layout is required."""

    feature_dim = increment_dim = 1
    error_floor = torch.tensor([0.01])

    def unflatten(self, state: torch.Tensor) -> State:
        """Decode a scalar replay state."""
        return (state,)

    def features(self, state: State) -> torch.Tensor:
        """Return Euclidean state features."""
        return state[0]

    def increments(self, state: State, successor: State) -> torch.Tensor:
        """Subtract physical states in Euclidean coordinates."""
        return successor[0] - state[0]

    def integrate(self, state: State, increment: torch.Tensor) -> State:
        """Apply a Euclidean increment."""
        return (state[0] + increment,)

    def error(self, prediction: State, target: State) -> torch.Tensor:
        """Return the signed scalar prediction error."""
        return prediction[0] - target[0]


class ScalarTask(DifferentiableTask):
    """Analytic scalar-control task with no robot-specific representation."""

    name = "scalar"
    representation = ScalarRepresentation()
    history_length = 2
    num_actions = obs_dim = goal_dim = xi_dim = critic_dim = 1
    max_step_reward = 1.0

    def obs(self, state: State, previous_action: torch.Tensor, xi: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Expose the measured scalar state."""
        return state[0]

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Keep this reference system deterministic."""
        return obs

    def policy_goal(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        """Return the exogenous target."""
        return goal

    def advance(self, goal: torch.Tensor) -> torch.Tensor:
        """Keep the target fixed over the imagined horizon."""
        return goal

    def valid_start(self, goal: torch.Tensor, horizon: int) -> torch.Tensor:
        """Allow imagination from every scalar state."""
        return torch.ones(goal.shape[0], dtype=torch.bool)

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        """Leave the nominal scalar context unchanged."""
        return xi

    def reward(
        self, state: State, goal: torch.Tensor, action: torch.Tensor, previous_action: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        """Reward approach to the target using the successor state."""
        return 1 - (state[0] - goal).square().squeeze(-1)

    def termination_prob(self, state: State, goal: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Keep every imagined transition alive."""
        return torch.zeros(state[0].shape[0])

    def critic_features(self, state: State, goal: torch.Tensor, xi: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Expose the physical scalar state to the critic."""
        return state[0]

    def checkpoint_metadata(self) -> dict:
        """Record the coordinate convention used by this test task."""
        return {"coordinates": "euclidean-v1"}


class ScalarEnv:
    """Deterministic transition source with periodic time-limit resets."""

    device = "cpu"
    num_envs = 4
    cfg: ClassVar[dict] = {}

    def __init__(self) -> None:
        """Initialize fresh physical state, context and episode counters."""
        self.task = ScalarTask()
        self.position = torch.zeros(self.num_envs, 1)
        self.prev_action = torch.zeros_like(self.position)
        self.prev_prev_action = torch.zeros_like(self.position)
        self.xi = torch.zeros_like(self.position)
        self.goal_state = torch.ones_like(self.position)
        self.episode_step = torch.zeros(self.num_envs, dtype=torch.long)

    def state(self) -> State:
        """Return physical state without aliasing mutable simulator buffers."""
        return (self.position.clone(),)

    def step(self, action: torch.Tensor) -> dict:
        """Advance known dynamics and return the true pre-reset successor."""
        before = self.position.clone()
        self.position = before + 0.2 * action
        done = self.episode_step == 6
        out = dict(
            state=before,
            next_state=self.position.clone(),
            action=action.clone(),
            prev_action=self.prev_action.clone(),
            prev_prev_action=self.prev_prev_action.clone(),
            xi=self.xi.clone(),
            goal_state=self.goal_state.clone(),
            next_goal_state=self.goal_state.clone(),
            episode_step=self.episode_step.clone(),
            done=done,
            terminated=torch.zeros_like(done),
            reward=self.task.reward(self.state(), self.goal_state, action, self.prev_action),
        )
        self.prev_prev_action, self.prev_action = (self.prev_action.clone(), action.clone())
        self.episode_step += 1
        self.position[done] = 0
        self.episode_step[done] = 0
        self.prev_action[done] = self.prev_prev_action[done] = 0
        return out


def small_cfg(**kwargs: object) -> SuperMimicCfg:
    """Build a small learner for observable learning and gradient checks."""
    args = dict(
        num_envs=4,
        buffer_steps=16,
        warmup_steps=4,
        world_model_window=3,
        world_model_batches=1,
        world_model_batch_size=8,
        horizon=3,
        policy_batch_size=4,
        world_model_warmup_iterations=0,
        critic_epochs=1,
        critic_minibatches=2,
        compile=False,
        world_model_hidden=(16,),
        encoder_hidden=(8,),
        decoder_hidden=(8,),
        critic_hidden=(8,),
    )
    return SuperMimicCfg(**args | kwargs)


def test_world_model_preserves_state_and_action_gradients_when_weights_are_frozen() -> None:
    """Frozen model weights retain the input Jacobians needed by policy optimization."""
    model = WorldModel(ScalarRepresentation(), 1, 1, hidden=(8,))
    with torch.no_grad():
        model.net[-1].weight.normal_(0, 0.1)
    model.requires_grad_(False)
    state, action = (torch.ones(4, 1, requires_grad=True), torch.ones(4, 1, requires_grad=True))
    prediction = model((state,), action, torch.zeros(4, 1))[0]
    prediction.square().sum().backward()
    assert state.grad.abs().sum() > 0 and action.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_learning_and_full_restart_with_fresh_replay() -> None:
    """Learn on Euclidean dynamics and resume learned state without stale collection history."""
    torch.manual_seed(2)
    env = ScalarEnv()
    learner = SuperMimic(small_cfg(), env.task, "cpu")
    initial = copy.deepcopy(learner.policy.state_dict())
    for _ in range(4):
        metrics = learner.iterate(env)
    assert {"world_model_loss", "policy_loss", "critic_loss"} <= metrics.keys()
    assert all(torch.isfinite(v).all() for v in metrics.values())
    assert metrics["policy_grad_norm"] > 0
    assert any((not torch.equal(value, initial[name]) for name, value in learner.policy.state_dict().items()))
    other = SuperMimic(small_cfg(), env.task, "cpu")
    other.load_state_dict(learner.state_dict())
    assert other.iteration == learner.iteration and other.buffer is None
    assert not other.obs_history.any()
    for name, value in learner.policy.state_dict().items():
        torch.testing.assert_close(other.policy.state_dict()[name], value)


def test_context_validation_and_policy_only_loading() -> None:
    """Full loads reject changed physical semantics while policy-only loads permit them."""
    env = ScalarEnv()
    learner = SuperMimic(small_cfg(), env.task, "cpu")
    state = learner.state_dict()
    state["task_metadata"] = {"coordinates": "different"}
    with pytest.raises(ValueError, match="task_metadata"):
        learner.load_state_dict(state)
    learner.load_state_dict(state, policy_only=True)
    state["goal_dim"] += 1
    with pytest.raises(ValueError, match="goal_dim"):
        learner.load_state_dict(state, policy_only=True)


def test_no_valid_imagination_starts_skip_without_nan() -> None:
    """A replay batch outside valid goal horizons does not create an empty loss."""
    env = ScalarEnv()
    learner = SuperMimic(small_cfg(), env.task, "cpu")
    learner.collect(env, 4)
    env.task.valid_start = lambda goal, horizon: torch.zeros(goal.shape[0], dtype=torch.bool)
    assert learner.update_policy()["policy_skipped_no_valid_starts"].item() == 1


def test_failed_policy_update_restores_requires_grad() -> None:
    """Task failures cannot leave model or critic weights frozen for later updates."""
    env = ScalarEnv()
    learner = SuperMimic(small_cfg(), env.task, "cpu")
    learner.collect(env, 4)

    def fail(*args: object, **kwargs: object) -> None:
        """Simulate an error in a differentiable task callback."""
        raise RuntimeError("task failure")

    env.task.reward = fail
    with pytest.raises(RuntimeError, match="task failure"):
        learner.update_policy()
    assert all(p.requires_grad for module in (learner.world_model, learner.critic) for p in module.parameters())


def test_lambda_returns_stop_at_terminal_but_include_terminal_reward() -> None:
    """Terminal reward remains in the target while all later rewards are excluded."""
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[7.0], [8.0], [9.0], [10.0]])
    actual = lambda_returns(rewards, torch.tensor([[0.0], [1.0], [0.0]]), values, 0.5, 1.0)
    torch.testing.assert_close(actual, torch.tensor([[2.0], [2.0], [8.0]]))


@pytest.mark.parametrize("load_optimizer", [True, False])
def test_runner_checkpoint_and_export(tmp_path: Path, load_optimizer: bool) -> None:
    """The public runner learns, restores and exports the same deterministic policy."""
    env = ScalarEnv()
    config = {"algorithm": vars(small_cfg()), "save_interval": 2}
    runner = SuperMimicRunner(env, config, str(tmp_path))
    runner.learn(2)
    assert runner.current_learning_iteration == 2
    restored_config = {**config, "algorithm": {**config["algorithm"], "learning_rate": 1e-5}}
    restored = SuperMimicRunner(ScalarEnv(), restored_config)
    restored.load(str(tmp_path / "model_final.pt"), load_optimizer=load_optimizer)
    assert restored.current_learning_iteration == 2
    for name in ("policy", "world_model", "critic"):
        torch.testing.assert_close(getattr(restored.alg, name).state_dict(), getattr(runner.alg, name).state_dict())
    for name, optimizer in restored.alg.optimizers.items():
        if load_optimizer:
            torch.testing.assert_close(optimizer.state_dict(), runner.alg.optimizers[name].state_dict())
        else:
            assert not optimizer.state
            assert optimizer.param_groups[0]["lr"] == 1e-5
    runner.export_policy_to_jit(str(tmp_path))
    exported = torch.jit.load(str(tmp_path / "policy.pt"))
    history, goal = (torch.randn(4, 2, 1), torch.randn(4, 1))
    torch.testing.assert_close(exported(history, goal), runner.get_inference_policy()(history, goal))


def test_replay_wraparound_masks_resets_and_copies_caller_storage() -> None:
    """Ring chronology and reset masking survive wraparound and caller buffer reuse."""
    example = {
        "state": torch.zeros(1, 1),
        "done": torch.zeros(1, dtype=torch.bool),
        "episode_step": torch.zeros(1, dtype=torch.long),
    }
    replay = SequenceReplayBuffer(4, 1, example)
    for i in range(7):
        example["state"].fill_(i)
        example["done"].fill_(i == 4)
        example["episode_step"].fill_(i if i <= 4 else i - 5)
        replay.add(example)
    example["state"].fill_(-999)
    torch.testing.assert_close(
        replay.data["state"][replay._rows(torch.arange(4))].flatten(), torch.tensor([3.0, 4.0, 5.0, 6.0])
    )
    windows = replay.windows(20, 4, ("state",))
    torch.testing.assert_close(windows["mask"], torch.tensor([[1.0, 1.0, 0.0, 0.0]]).expand(20, -1))
    starts, hist = replay.starts(100, 4, ("state", "episode_step"))
    assert ((hist["state"][:, -1] - hist["state"][:, 0]).flatten() <= starts["episode_step"]).all()


def test_invalid_config_and_replay_fail_before_training() -> None:
    """Invalid horizons, discounts and empty replay fail with actionable errors."""
    with pytest.raises(ValueError, match="warmup_steps"):
        SuperMimic(small_cfg(warmup_steps=1), ScalarTask(), "cpu")
    with pytest.raises(ValueError, match="gamma"):
        SuperMimic(small_cfg(gamma=1), ScalarTask(), "cpu")
    replay = SequenceReplayBuffer(3, 1, {"state": torch.zeros(1, 1)})
    with pytest.raises(ValueError, match="stored replay"):
        replay.windows(1, 2, ("state",))


def test_collection_snapshots_reused_reward_metrics_and_episode_buffers() -> None:
    """Reported values describe each step even when the simulator reuses its output."""

    class ReusingEnv(ScalarEnv):
        def __init__(self) -> None:
            super().__init__()
            self.output, self.count = {}, 0
            self.reward_buffer = torch.zeros(self.num_envs)
            self.metric_buffer = torch.zeros(())
            self.episode_buffer = torch.zeros(1)
            self.metrics = {"progress": self.metric_buffer}

        def step(self, action: torch.Tensor) -> dict:
            self.output.clear()
            self.output.update(super().step(action))
            self.count += 1
            self.reward_buffer.fill_(self.count)
            self.metric_buffer.fill_(10 * self.count)
            self.episode_buffer.fill_(self.count)
            self.output.update(reward=self.reward_buffer, metrics=self.metrics)
            if self.count % 2:
                self.output.update(
                    episode_length=self.episode_buffer,
                    episode_return=self.episode_buffer,
                    episode_terms={"tracking": self.episode_buffer},
                )
            return self.output

    env = ReusingEnv()
    learner = SuperMimic(small_cfg(world_model_warmup_iterations=10), env.task, "cpu")
    stats = learner.iterate(env)
    assert stats["sim_reward"].item() == 2.5
    assert stats["Metrics/progress"].item() == 25
    for key in ("episode_length", "episode_return", "Episode_Reward/tracking"):
        assert stats[key].item() == 2


@pytest.mark.parametrize("env_device,device", [("cuda:2", "cuda"), ("cuda", "cuda:2"), ("cuda", "cuda")])
def test_runner_resolves_implicit_cuda_indices(monkeypatch: pytest.MonkeyPatch, env_device: str, device: str) -> None:
    """Equivalent CUDA spellings select the same concrete device for every component."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    algorithm = Mock(side_effect=lambda cfg, task, device: SimpleNamespace(cfg=cfg))
    logger = Mock()
    monkeypatch.setattr(supermimic_runner, "SuperMimic", algorithm)
    monkeypatch.setattr(supermimic_runner, "Logger", logger)
    env = ScalarEnv()
    env.device = env_device
    runner = SuperMimicRunner(env, {"algorithm": vars(small_cfg())}, device=device)
    assert torch.device(runner.device) == torch.device("cuda:2")
    assert torch.device(algorithm.call_args.args[-1]) == torch.device("cuda:2")
    assert torch.device(logger.call_args.args[-1]) == torch.device("cuda:2")
    env.device = "cuda:1"
    with pytest.raises(ValueError, match="share a device"):
        SuperMimicRunner(env, {}, device="cuda:2")


@pytest.mark.parametrize("policy_only", [False, True])
def test_checkpoints_reject_changed_action_joint_order(policy_only: bool) -> None:
    """Equal decoder widths cannot silently redirect learned actions to different joints."""
    task = ScalarTask()
    task.num_actions = 2
    task.action_joints = torch.tensor([0, 1])
    source = SuperMimic(small_cfg(), task, "cpu")
    with torch.no_grad():
        source.policy.decoder[-1].bias.copy_(torch.tensor([0.25, -0.5]))
    state = copy.deepcopy(source.state_dict())
    target_task = copy.copy(task)
    target_task.action_joints = torch.tensor([1, 0])
    target = SuperMimic(small_cfg(), target_task, "cpu")
    with pytest.raises(ValueError, match="action_joints"):
        target.load_state_dict(state, policy_only=policy_only)
    target_task.action_joints = None
    compatible = SuperMimic(small_cfg(), target_task, "cpu")
    compatible.load_state_dict(state, policy_only=policy_only)
    history, goal = torch.ones(1, 2, 1), torch.ones(1, 1)
    torch.testing.assert_close(source.policy(history, goal), compatible.policy(history, goal))
