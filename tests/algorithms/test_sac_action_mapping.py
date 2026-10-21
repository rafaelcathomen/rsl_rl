# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for per-dimension action mappings and the diagnostics of SAC."""

from __future__ import annotations

import copy
import math
import torch

import pytest

from rsl_rl.modules import SquashedGaussianDistribution
from rsl_rl.runners import OffPolicyRunner

from .test_sac import NUM_ACTIONS, DummyEnv, _train_cfg

BOUND = [0.5, 1.0, 2.0]
OFFSET = [0.3, -0.2, 0.0]


def test_mapping_log_prob_matches_change_of_variables() -> None:
    """Affine per-dimension mappings shift the log density by the log of the product of the bounds."""
    torch.manual_seed(0)
    dist = SquashedGaussianDistribution(NUM_ACTIONS)
    dist.update(torch.stack([0.5 * torch.randn(64, NUM_ACTIONS), 0.5 + 0.2 * torch.randn(64, NUM_ACTIONS)], dim=1))
    squashed, log_prob_unit = dist.rsample()
    dist.set_action_mapping(torch.tensor(BOUND), torch.tensor(OFFSET))
    actions = torch.tensor(OFFSET) + torch.tensor(BOUND) * squashed
    torch.testing.assert_close(
        dist.log_prob(actions), log_prob_unit - sum(math.log(b) for b in BOUND), rtol=1e-3, atol=1e-3
    )
    # Samples stay inside the mapped interval of every dimension
    samples, log_prob = dist.rsample()
    assert ((samples - torch.tensor(OFFSET)).abs() <= torch.tensor(BOUND) + 1e-6).all()
    torch.testing.assert_close(dist.log_prob(samples), log_prob, rtol=1e-3, atol=1e-3)
    # The exported deterministic output matches the in-model deterministic output
    mlp_output = torch.randn(5, 2, NUM_ACTIONS)
    torch.testing.assert_close(dist.as_deterministic_output_module()(mlp_output), dist.deterministic_output(mlp_output))


def test_mapping_in_sac_critic_sees_squashed_actions() -> None:
    """With a per-dimension mapping, actions lie in the mapped intervals and critics see values in [-1, 1]."""
    torch.manual_seed(0)
    cfg = _train_cfg(action_bound=BOUND, action_offset=OFFSET)
    runner = OffPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=None)
    runner.learn(num_learning_iterations=6)
    alg = runner.alg
    stored = alg.storage.actions[: alg.storage.size]
    assert ((stored - torch.tensor(OFFSET)).abs() <= torch.tensor(BOUND) + 1e-5).all()
    critic_actions = (stored - alg._critic_action_offset) * alg._critic_action_scale
    assert critic_actions.abs().max() <= 1.0 + 1e-5
    # Checkpoints of mapped policies restore the mapping
    state = alg.save()
    restored = OffPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=None)
    restored.alg.load(state, None, strict=True)
    torch.testing.assert_close(restored.alg.actor.distribution.action_offset, torch.tensor(OFFSET))


def test_scalar_bound_checkpoints_keep_their_keys() -> None:
    """Scalar bounds do not add the offset to the state dict, so existing checkpoints still load strictly."""
    dist = SquashedGaussianDistribution(NUM_ACTIONS, action_bound=2.0)
    assert set(dist.state_dict()) == {"action_bound"}


def test_mapping_rejects_schedules_and_wrong_sizes() -> None:
    """Per-dimension bounds cannot be combined with a schedule and need one entry per action."""
    with pytest.raises(ValueError):
        OffPolicyRunner(DummyEnv(), _train_cfg(action_bound=BOUND, final_action_bound=3.0), log_dir=None)
    with pytest.raises(ValueError):
        OffPolicyRunner(DummyEnv(), _train_cfg(action_bound=[1.0, 1.0]), log_dir=None)
    with pytest.raises(ValueError):
        OffPolicyRunner(DummyEnv(), _train_cfg(action_offset=OFFSET), log_dir=None)


def test_diagnostics_log_statistics_and_catch_nonfinite_observations() -> None:
    """Diagnostics log value statistics and raise on the first non-finite observation."""
    torch.manual_seed(0)
    cfg = _train_cfg(diagnostics=True, critic_head="categorical", num_bins=21, reward_normalization=True)
    runner = OffPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=None)
    runner.learn(num_learning_iterations=6)
    stats = runner.alg.update()
    assert "diag_target_mass_high" in stats and "diag_q_max" in stats

    class NanEnv(DummyEnv):
        def step(self, actions: torch.Tensor) -> tuple:
            obs, rewards, dones, extras = super().step(actions)
            if self.episode_length_buf[0] == 5:
                obs["policy"][1, 2] = float("nan")
            return obs, rewards, dones, extras

    cfg["check_for_nan"] = False
    runner = OffPolicyRunner(NanEnv(), copy.deepcopy(cfg), log_dir=None)
    with pytest.raises(
        FloatingPointError,
        match=r"Non-finite observation 'policy'.*envs \[1\], dims \[2\].*Previous actions finite: True",
    ):
        runner.learn(num_learning_iterations=6)


def test_diagnostics_do_not_change_training() -> None:
    """Training with diagnostics produces the same parameters as training without them."""
    params = []
    for diagnostics in (False, True):
        torch.manual_seed(0)
        cfg = _train_cfg(diagnostics=diagnostics, critic_head="categorical", num_bins=21, reward_normalization=True)
        runner = OffPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=None)
        runner.learn(num_learning_iterations=8)
        params.append(torch.cat([p.detach().flatten() for p in runner.alg.actor.parameters()]))
    torch.testing.assert_close(params[0], params[1], rtol=0, atol=0)


def test_gated_lifetime_max_ignores_outlier_returns() -> None:
    """With the gate, an outlier return does not enter the lifetime maximum of the reward scale; without it, it does.

    The outlier still enters the standard deviation, which in this short test also changes the scale.
    """

    class SpikeEnv(DummyEnv):
        calls = 0

        def step(self, actions: torch.Tensor) -> tuple:
            obs, rewards, dones, extras = super().step(actions)
            SpikeEnv.calls += 1
            if SpikeEnv.calls == 30:
                rewards[0] = -1e4
            return obs, rewards, dones, extras

    scales = {}
    for gate in (None, 100.0):
        SpikeEnv.calls = 0
        torch.manual_seed(0)
        cfg = _train_cfg(critic_head="categorical", num_bins=21, reward_normalization=True,
                         reward_normalization_max_gate=gate)  # fmt: skip
        runner = OffPolicyRunner(SpikeEnv(), copy.deepcopy(cfg), log_dir=None)
        runner.learn(num_learning_iterations=10)
        stats = runner.alg.update()
        scales[gate] = stats["max_abs_return"]
        if gate is not None:
            assert stats["gated_returns"] >= 1 and stats["raw_max_abs_return"] >= 1e4 * 0.9
    assert scales[None] >= 1e4 * 0.9 > 100 * scales[100.0]
