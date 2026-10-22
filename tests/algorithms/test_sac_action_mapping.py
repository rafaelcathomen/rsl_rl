# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for per-dimension action mappings and the diagnostics of SAC."""

from __future__ import annotations

import copy
import math
import torch
from tensordict import TensorDict

import pytest

from rsl_rl.modules import SquashedGaussianDistribution
from rsl_rl.runners import OffPolicyRunner

from .test_sac import NUM_ACTIONS, NUM_ENVS, DummyEnv, _train_cfg

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


def test_gate_excludes_the_discounted_trace_of_an_outlier() -> None:
    """The decaying discounted return after an outlier stays out of the lifetime maximum until the episode ends."""

    class SpikeEnv(DummyEnv):
        calls = 0

        def step(self, actions: torch.Tensor) -> tuple:
            obs, rewards, dones, extras = super().step(actions)
            SpikeEnv.calls += 1
            if SpikeEnv.calls == 21:  # Right after a reset, so the trace decays over a whole episode
                rewards[0] = -1e4
            return obs, rewards, dones, extras

    torch.manual_seed(0)
    cfg = _train_cfg(critic_head="categorical", num_bins=21, reward_normalization=True, gamma=0.99,
                     reward_normalization_max_gate=100.0)  # fmt: skip
    runner = OffPolicyRunner(SpikeEnv(), copy.deepcopy(cfg), log_dir=None)
    runner.learn(num_learning_iterations=10)
    assert runner.alg._max_abs_return < 100.0


def _gate_alg(gate: float = 100.0, gate_std: bool = False) -> object:
    """Return a SAC algorithm with reward normalization and the max gate, ready for direct process_env_step calls."""
    torch.manual_seed(0)
    cfg = _train_cfg(
        critic_head="categorical",
        num_bins=21,
        reward_normalization=True,
        reward_normalization_max_gate=gate,
        reward_normalization_gate_std=gate_std,
        learning_starts=0,
    )
    runner = OffPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=None)
    alg = runner.alg
    obs = runner.env.get_observations()
    alg.act(obs)
    return alg, obs


def _step(alg: object, obs: TensorDict, rewards: list[float], dones: list[bool] | None = None) -> None:
    """Process one environment step with the given rewards."""
    d = torch.tensor(dones or [False] * NUM_ENVS)
    alg.process_env_step(obs, torch.tensor(rewards), d.long(), {"time_outs": torch.zeros(NUM_ENVS, dtype=torch.bool)})
    alg.act(obs)


def test_gate_cold_start_with_zero_std_does_not_gate() -> None:
    """While the standard deviation is zero (all returns zero so far), the first nonzero returns set the maximum."""
    alg, obs = _gate_alg()
    for _ in range(5):
        _step(alg, obs, [0.0] * NUM_ENVS)
    assert float(alg.return_normalizer._std.squeeze()) == 0.0
    _step(alg, obs, [3.0, 0.0, 0.0, 0.0])
    assert alg._max_abs_return.item() == pytest.approx(3.0)
    assert alg._gated_episodes.item() == 0


def test_gate_handles_positive_and_negative_outliers_and_resets() -> None:
    """Outliers of either sign are gated until their episode ends; afterwards the environment counts again."""
    alg, obs = _gate_alg()
    for _ in range(50):
        _step(alg, obs, [1.0, -1.0, 0.5, -0.5], dones=[True] * NUM_ENVS)
    base = alg._max_abs_return.item()
    _step(alg, obs, [1e4, -1e4, 0.5, -0.5])
    assert alg._max_abs_return.item() == pytest.approx(base)
    assert alg._gated_episodes.item() == 2 and alg._gated_envs[:2].all()
    # Still gated while the episode continues (the discounted trace stays large)
    _step(alg, obs, [1.0, -1.0, 0.5, -0.5], dones=[True, True, False, False])
    assert alg._max_abs_return.item() == pytest.approx(base)
    assert not alg._gated_envs[:2].any()
    # After the reset, a new (legitimately larger) return counts
    _step(alg, obs, [2.0, -1.0, 0.5, -0.5])
    assert alg._max_abs_return.item() == pytest.approx(2.0)


def test_gate_survives_save_and_load() -> None:
    """Checkpoints keep the gated maximum; a restored run continues to train."""
    torch.manual_seed(0)
    cfg = _train_cfg(
        critic_head="categorical", num_bins=21, reward_normalization=True, reward_normalization_max_gate=100.0
    )
    runner = OffPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=None)
    runner.learn(num_learning_iterations=6)
    state = runner.alg.save()
    restored = OffPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=None)
    restored.alg.load(state, None, strict=True)
    assert restored.alg._max_abs_return.item() == pytest.approx(runner.alg._max_abs_return.item())
    restored.learn(num_learning_iterations=4)


@pytest.mark.parametrize("gate_std", [False, True])
def test_gate_std_keeps_outliers_out_of_the_standard_deviation(gate_std: bool) -> None:
    """With std gating, a gated outlier and its discounted trace change neither the standard deviation nor the maximum.

    Without it (the original gate), the outlier still inflates the standard deviation and with it the reward scale.
    """
    alg, obs = _gate_alg(gate_std=gate_std)
    for _ in range(200):
        _step(alg, obs, [1.0, -1.0, 0.5, -0.5], dones=[True] * NUM_ENVS)
    std, maximum = float(alg.return_normalizer._std.squeeze()), alg._max_abs_return.item()
    _step(alg, obs, [1e6, 0.5, 0.5, -0.5])
    for _ in range(5):  # the trace of the gated episode (the other environments keep resetting)
        _step(alg, obs, [1.0, -1.0, 0.5, -0.5], dones=[False, True, True, True])
    assert alg._max_abs_return.item() == pytest.approx(maximum)
    assert alg._gated_episodes.item() == 1
    if gate_std:
        assert float(alg.return_normalizer._std.squeeze()) < 2 * std
        assert alg._reward_scale().item() < 2 * max(std, maximum / 5)
    else:
        assert float(alg.return_normalizer._std.squeeze()) > 100 * std


def test_gate_std_still_learns_from_the_other_environments_and_after_reset() -> None:
    """Non-gated environments keep updating the standard deviation; the gated one counts again after its reset."""
    alg, obs = _gate_alg(gate_std=True)
    for _ in range(50):
        _step(alg, obs, [1.0, -1.0, 0.5, -0.5], dones=[True] * NUM_ENVS)
    count = alg.return_normalizer.count.item()
    _step(alg, obs, [1e6, 2.0, 0.5, -0.5])
    assert alg.return_normalizer.count.item() == count + NUM_ENVS - 1
    # The step that ends the gated episode still excludes it; the next one counts it again
    _step(alg, obs, [1.0, -1.0, 0.5, -0.5], dones=[True] * NUM_ENVS)
    _step(alg, obs, [1.0, -1.0, 0.5, -0.5])
    assert alg.return_normalizer.count.item() == count + 2 * (NUM_ENVS - 1) + NUM_ENVS


def test_gate_std_with_all_environments_gated_keeps_statistics() -> None:
    """If every environment is gated at once, the statistics stay unchanged (no update from an empty batch)."""
    alg, obs = _gate_alg(gate_std=True)
    for _ in range(50):
        _step(alg, obs, [1.0, -1.0, 0.5, -0.5], dones=[True] * NUM_ENVS)
    std, count = float(alg.return_normalizer._std.squeeze()), alg.return_normalizer.count.item()
    _step(alg, obs, [1e6, -1e6, 1e6, -1e6])
    assert alg.return_normalizer.count.item() == count
    assert float(alg.return_normalizer._std.squeeze()) == pytest.approx(std)
