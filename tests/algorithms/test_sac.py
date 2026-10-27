# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the SAC algorithm and the off-policy runner."""

from __future__ import annotations

import copy
import math
import tempfile
import torch
from pathlib import Path
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms import SAC
from rsl_rl.env import VecEnv
from rsl_rl.modules import ResidualMLP, SquashedGaussianDistribution
from rsl_rl.runners import OffPolicyRunner

NUM_ENVS = 4
OBS_DIM = 6
NUM_ACTIONS = 3
MAX_EP_LEN = 10


class DummyEnv(VecEnv):
    """VecEnv with time-outs that reports the final observations of the episodes."""

    def __init__(self, report_final_obs: bool = True) -> None:  # noqa: D107
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}
        self.report_final_obs = report_final_obs
        self._obs = torch.randn(NUM_ENVS, OBS_DIM)

    def get_observations(self) -> TensorDict:  # noqa: D102
        return TensorDict({"policy": self._obs.clone()}, batch_size=[NUM_ENVS])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:  # noqa: D102
        assert actions.abs().max() <= 2.0 + 1e-6
        self.episode_length_buf += 1
        self._obs = self._obs + 0.1 * actions.sum(dim=1, keepdim=True)
        rewards = -self._obs.square().mean(dim=1)
        time_outs = self.episode_length_buf >= self.max_episode_length
        extras = {"time_outs": time_outs}
        if self.report_final_obs:
            extras["final_obs"] = {"policy": self._obs.clone()}
        self._obs[time_outs] = torch.randn(int(time_outs.sum()), OBS_DIM, device=self.device)
        self.episode_length_buf[time_outs] = 0
        return self.get_observations(), rewards, time_outs.long(), extras


class CudaEnv(DummyEnv):
    """DummyEnv whose buffers live on the selected CUDA device."""

    def __init__(self, device: str = "cuda") -> None:  # noqa: D107
        super().__init__()
        self.device = device
        self.episode_length_buf = self.episode_length_buf.to(device)
        self._obs = self._obs.to(device)


def _train_cfg(**alg_overrides: object) -> dict:
    """Return a small SAC training configuration."""
    algorithm = {
        "class_name": "SAC",
        "num_updates": 2,
        "batch_size": 16,
        "replay_buffer_size": 400,
        "n_steps": 3,
        "learning_starts": 8,
    }
    algorithm.update(alg_overrides)
    return {
        "num_steps_per_env": 4,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [16, 16], "activation": "relu"},
        "critic": {"class_name": "QMLPModel", "hidden_dims": [16, 16], "activation": "relu"},
        "algorithm": algorithm,
    }


VARIANTS = {
    "plain": {},
    "categorical_min": {"critic_head": "categorical", "num_bins": 11, "reward_normalization": True},
    "categorical_mean": {"critic_head": "categorical", "num_bins": 11, "q_aggregation": "mean"},
    "held_noise_bound_schedule": {
        "exploration": "held",
        "action_bound": 1.0,
        "final_action_bound": 2.0,
        "action_bound_hold_steps": 4,
        "action_bound_ramp_steps": 8,
    },
    "cosine_clip_adamw": {
        "learning_rate_schedule": "cosine",
        "schedule_updates": 10,
        "max_grad_norm": 1.0,
        "optimizer": "adamw",
        "weight_decay": 1e-3,
        "actor_update_interval": 2,
    },
}


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_runner_trains_and_restores(variant: str) -> None:
    """The runner trains all main variants with finite losses and restores them from a checkpoint."""
    torch.manual_seed(0)
    cfg = _train_cfg(**VARIANTS[variant])
    with tempfile.TemporaryDirectory() as log_dir:
        runner = OffPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=log_dir)
        runner.learn(num_learning_iterations=6)
        loss_dict = runner.alg.update()
        assert {"critic", "q_value", "actor", "alpha", "entropy"} <= set(loss_dict)
        assert all(math.isfinite(v) for v in loss_dict.values())

        path = f"{log_dir}/model.pt"
        runner.save(path)
        restored = OffPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=None)
        restored.load(path)
        obs = runner.env.get_observations()
        expected = runner.get_inference_policy()(obs)
        torch.testing.assert_close(restored.get_inference_policy()(obs), expected)


def test_residual_networks_with_batch_norm() -> None:
    """Batch-normalized residual networks train and keep their weights normalized."""
    torch.manual_seed(0)
    network_cfg = {"class_name": "ResidualMLP", "hidden_dim": 8, "num_blocks": 1}
    cfg = _train_cfg(critic_head="categorical", num_bins=11, exploration="held", reward_normalization=True)
    cfg["actor"] = {"class_name": "MLPModel", "network_cfg": network_cfg}
    cfg["critic"] = {"class_name": "QMLPModel", "network_cfg": network_cfg}
    runner = OffPolicyRunner(DummyEnv(), cfg, log_dir=None)
    runner.learn(num_learning_iterations=6)

    for model in [runner.alg.actor, *runner.alg.critics]:
        assert isinstance(model.mlp, ResidualMLP)
        norms = model.mlp.embed.weight.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms))


def test_time_outs_bootstrap_from_final_observations() -> None:
    """Transitions that time out store the final observation of the episode as the next observation."""
    env = DummyEnv()
    runner = OffPolicyRunner(env, _train_cfg(), log_dir=None)
    alg: SAC = runner.alg
    obs = env.get_observations()
    for _ in range(MAX_EP_LEN):
        actions = alg.act(obs)
        obs, rewards, dones, extras = env.step(actions)
        alg.process_env_step(obs, rewards, dones, extras)

    last = (alg.storage.step - 1) % alg.storage.num_slots
    assert alg.storage.dones[last].all()
    assert not alg.storage.terminated[last].any()
    torch.testing.assert_close(alg.storage.next_observations["policy"][last], extras["final_obs"]["policy"])
    assert not torch.allclose(alg.storage.next_observations["policy"][last], obs["policy"])


class ImageEnv(DummyEnv):
    """DummyEnv that also reports a (depth-like) camera image derived from the state."""

    def _image(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, :3, None, None].expand(-1, 3, 8, 8).clone()

    def get_observations(self) -> TensorDict:  # noqa: D102
        return TensorDict({"policy": self._obs.clone(), "image": self._image(self._obs)}, batch_size=[NUM_ENVS])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:  # noqa: D102
        obs, rewards, dones, extras = super().step(actions)
        extras["final_obs"]["image"] = self._image(extras["final_obs"]["policy"])
        return obs, rewards, dones, extras


@pytest.mark.parametrize("hidden_dims", [None, [16]])
def test_cnn_actor_with_images(hidden_dims: list[int] | None) -> None:
    """A CNN actor trains from 2D observations, and time-outs bootstrap from the final image."""
    torch.manual_seed(0)
    cfg = _train_cfg(replay_compact=True, replay_obs_dtype="float16")
    cfg["obs_groups"] = {"actor": ["policy", "image"], "critic": ["policy"]}
    cnn_cfg = {"output_channels": [4], "kernel_size": 3, "stride": 2}
    cfg["actor"] = {"class_name": "CNNModel", "cnn_cfg": cnn_cfg}
    if hidden_dims is not None:
        cfg["actor"]["hidden_dims"] = hidden_dims
    env = ImageEnv()
    runner = OffPolicyRunner(env, cfg, log_dir=None)
    alg: SAC = runner.alg
    obs = env.get_observations()
    for _ in range(MAX_EP_LEN):
        obs, rewards, dones, extras = env.step(alg.act(obs))
        alg.process_env_step(obs, rewards, dones, extras)
    last = (alg.storage.step - 1) % alg.storage.num_slots
    final = alg.storage.final_index[last]
    assert (final >= 0).all()
    stored = alg.storage.final_observations["image"][final].float()
    torch.testing.assert_close(stored, extras["final_obs"]["image"], rtol=1e-3, atol=1e-3)
    runner.learn(num_learning_iterations=4)
    assert all(math.isfinite(v) for v in alg.update().values())


def test_learning_starts_can_exceed_replay_capacity() -> None:
    """Warm-up ends after enough transitions are collected even when the replay ring is smaller."""
    runner = OffPolicyRunner(DummyEnv(), _train_cfg(replay_buffer_size=12, learning_starts=20), log_dir=None)
    runner.learn(num_learning_iterations=3)
    assert runner.alg.num_critic_updates > 0


@pytest.mark.parametrize(("compact", "randomize"), [(False, False), (True, True)])
def test_per_environment_episode_limits(compact: bool, randomize: bool) -> None:
    """Dense and compact replay support dynamic per-environment episode limits."""
    env = DummyEnv()
    env.max_episode_length = torch.tensor([10, 11, 12, 13])
    runner = OffPolicyRunner(env, _train_cfg(replay_compact=compact), log_dir=None)
    runner.learn(num_learning_iterations=3, init_at_random_ep_len=randomize)
    assert runner.alg.num_critic_updates > 0
    assert (env.episode_length_buf < env.max_episode_length).all()


def test_sgd_trains_without_cuda_graphs() -> None:
    """SGD works eagerly and rejects unsupported graph capture at construction."""
    runner = OffPolicyRunner(DummyEnv(), _train_cfg(optimizer="sgd"), log_dir=None)
    runner.learn(num_learning_iterations=3)
    assert all(math.isfinite(value) for value in runner.alg.update().values())
    with pytest.raises(ValueError, match="SGD does not support CUDA graph"):
        OffPolicyRunner(DummyEnv(), _train_cfg(optimizer="sgd", use_cuda_graph=True), log_dir=None)


def test_resume_without_optimizer_preserves_training_progress() -> None:
    """Fresh optimizers retain the saved action curriculum and update schedule progress."""
    cfg = _train_cfg(
        learning_starts=0,
        action_bound=1.0,
        final_action_bound=2.0,
        action_bound_hold_steps=4,
        action_bound_ramp_steps=8,
    )
    runner = OffPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=None)
    runner.learn(num_learning_iterations=3)
    restored = OffPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=None)
    restored.alg.load(
        runner.alg.save(), {"actor": True, "critic": True, "optimizer": False, "iteration": True}, strict=True
    )
    assert not restored.alg.actor_optimizer.state
    assert restored.alg.num_env_steps == runner.alg.num_env_steps
    assert restored.alg.num_critic_updates == runner.alg.num_critic_updates
    obs = restored.env.get_observations()
    obs, rewards, dones, extras = restored.env.step(restored.alg.act(obs))
    restored.alg.process_env_step(obs, rewards, dones, extras)
    assert restored.alg.actor.distribution.action_bound.item() == 2.0


def test_missing_final_observations_warn() -> None:
    """Time-outs without final observations fall back to the reset observations with a warning."""
    env = DummyEnv(report_final_obs=False)
    runner = OffPolicyRunner(env, _train_cfg(), log_dir=None)
    obs = env.get_observations()
    with pytest.warns(UserWarning, match="final observations"):
        for _ in range(MAX_EP_LEN):
            actions = runner.alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            runner.alg.process_env_step(obs, rewards, dones, extras)


@pytest.mark.parametrize("action_offset", [0.0, 0.5])
def test_squashed_gaussian_log_prob_and_bound(action_offset: float) -> None:
    """Sampled actions respect the bound, and their log probabilities match the analytic density."""
    torch.manual_seed(0)
    dist = SquashedGaussianDistribution(NUM_ACTIONS, action_bound=2.0, action_offset=action_offset)
    # Moderate standard deviations (about 0.3) avoid saturating the squashing, where atanh is ill-conditioned
    dist.update(torch.stack([0.5 * torch.randn(64, NUM_ACTIONS), 0.5 + 0.2 * torch.randn(64, NUM_ACTIONS)], dim=1))
    actions, log_prob = dist.rsample()
    assert (actions - action_offset).abs().max() <= 2.0
    torch.testing.assert_close(dist.log_prob(actions), log_prob, rtol=1e-3, atol=1e-3)

    # Doubling the bound halves the density in every dimension
    dist.action_bound.fill_(4.0)
    torch.testing.assert_close(
        dist.log_prob(2.0 * (actions - action_offset) + action_offset),
        log_prob - NUM_ACTIONS * math.log(2.0),
        rtol=1e-3,
        atol=1e-3,
    )
    zero_mean = torch.zeros(1, 2, NUM_ACTIONS)
    expected = torch.full((1, NUM_ACTIONS), action_offset)
    torch.testing.assert_close(dist.deterministic_output(zero_mean), expected)
    torch.testing.assert_close(dist.as_deterministic_output_module()(zero_mean), expected)


def test_jit_export_after_training_handles_single_observations(tmp_path: Path) -> None:
    """Direct export uses evaluation BatchNorm statistics without an inference-policy prerequisite."""
    cfg = _train_cfg()
    cfg["actor"] = {"class_name": "MLPModel"}
    runner = OffPolicyRunner(DummyEnv(), cfg, log_dir=None)
    runner.learn(num_learning_iterations=3)
    runner.export_policy_to_jit(str(tmp_path))
    exported = torch.jit.load(str(tmp_path / "policy.pt"))
    obs = runner.env.get_observations()[:1]
    torch.testing.assert_close(exported(obs["policy"]), runner.get_inference_policy()(obs))


def test_categorical_projection() -> None:
    """The projection preserves probability mass and is the identity for zero reward and unit discount."""
    runner = OffPolicyRunner(DummyEnv(), _train_cfg(critic_head="categorical", num_bins=11), log_dir=None)
    alg: SAC = runner.alg
    probs = torch.softmax(torch.randn(8, 11), dim=-1)
    zeros = torch.zeros(8)

    torch.testing.assert_close(alg._project(probs, zeros, torch.ones(8), zeros), probs)
    projected = alg._project(probs, torch.randn(8), torch.rand(8), torch.randn(8))
    torch.testing.assert_close(projected.sum(dim=-1), torch.ones(8))

    # Terminal transitions put all mass on the reward
    terminal = alg._project(probs, torch.full((8,), 1.0), zeros, zeros)
    torch.testing.assert_close(alg._expected_value(terminal.log()), torch.ones(8))


def test_target_entropy_from_std() -> None:
    """The default target entropy is the entropy of a Gaussian with the target standard deviation."""
    runner = OffPolicyRunner(DummyEnv(), _train_cfg(target_entropy_std=0.15), log_dir=None)
    expected = torch.distributions.Normal(0.0, 0.15).entropy().item() * NUM_ACTIONS
    assert runner.alg.target_entropy == pytest.approx(expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require a CUDA device.")
@pytest.mark.parametrize("variant", ["plain", "categorical_min", "obs_normalization"])
def test_cuda_graph_matches_eager(variant: str) -> None:
    """Gradient steps captured in a CUDA graph match eager gradient steps."""
    results = []
    for use_cuda_graph in (False, True):
        torch.manual_seed(0)
        alg_cfg = VARIANTS.get(variant, {})
        cfg = _train_cfg(**alg_cfg, use_cuda_graph=use_cuda_graph, learning_rate_schedule="cosine")
        cfg["algorithm"]["schedule_updates"] = 20
        if variant == "obs_normalization":
            # Normalization statistics change between graph replays and must be read from the same buffers
            cfg["actor"]["obs_normalization"] = True
            cfg["critic"]["obs_normalization"] = True
        runner = OffPolicyRunner(CudaEnv(), cfg, log_dir=None, device="cuda")
        alg: SAC = runner.alg
        obs = runner.env.get_observations()
        for _ in range(12):
            actions = alg.act(obs)
            obs, rewards, dones, extras = runner.env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
        torch.manual_seed(1)
        losses = []
        for _ in range(6):
            losses.append(alg.update())
            # Collect between updates so that the normalization statistics change between graph replays
            for _ in range(2):
                actions = alg.act(obs)
                obs, rewards, dones, extras = runner.env.step(actions)
                alg.process_env_step(obs, rewards, dones, extras)
        results.append(losses)
    for eager, graph in zip(*results):
        for key in eager:
            assert graph[key] == pytest.approx(eager[key], rel=1e-3, abs=1e-5), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require a CUDA device.")
@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_resume_switches_cuda_graph_mode(use_cuda_graph: bool) -> None:
    """Full checkpoints resume from CPU eager training to CUDA capture and back."""
    source_device = "cpu" if use_cuda_graph else "cuda"
    source_env = DummyEnv() if use_cuda_graph else CudaEnv()
    source = OffPolicyRunner(
        source_env, _train_cfg(use_cuda_graph=not use_cuda_graph), log_dir=None, device=source_device
    )
    source.learn(num_learning_iterations=4)
    device = "cuda" if use_cuda_graph else "cpu"
    env = CudaEnv() if use_cuda_graph else DummyEnv()
    restored = OffPolicyRunner(env, _train_cfg(use_cuda_graph=use_cuda_graph), log_dir=None, device=device)
    restored.alg.load(copy.deepcopy(source.alg.save()), None, strict=True)
    restored.learn(num_learning_iterations=8)
    assert restored.alg.num_critic_updates > source.alg.num_critic_updates
    assert all(math.isfinite(value) for value in restored.alg.update().values())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices.")
def test_cuda_graph_uses_noncurrent_device() -> None:
    """Capture on GPU 1 works while the process's current device remains GPU 0."""
    with torch.cuda.device(0):
        runner = OffPolicyRunner(CudaEnv("cuda:1"), _train_cfg(use_cuda_graph=True), log_dir=None, device="cuda:1")
        before = [parameter.detach().clone() for parameter in runner.alg.actor.parameters()]
        runner.learn(num_learning_iterations=8)
        assert torch.cuda.current_device() == 0
        assert any(not torch.equal(old, new) for old, new in zip(before, runner.alg.actor.parameters()))
        assert all(math.isfinite(value) for value in runner.alg.update().values())


def test_default_configuration() -> None:
    """Without network settings, SAC builds the default residual networks and a categorical critic, and trains."""
    torch.manual_seed(0)
    cfg = _train_cfg()
    cfg["actor"] = {"class_name": "MLPModel"}
    cfg["critic"] = {"class_name": "QMLPModel"}
    runner = OffPolicyRunner(DummyEnv(), cfg, log_dir=None)
    alg: SAC = runner.alg
    assert isinstance(alg.actor.mlp, ResidualMLP)
    assert all(isinstance(critic.mlp, ResidualMLP) for critic in alg.critics)
    assert alg.critic_head == "categorical"
    assert alg.critics[0].mlp.output.out_features == alg.num_bins
    runner.learn(num_learning_iterations=4)
    assert all(math.isfinite(v) for v in alg.update().values())
