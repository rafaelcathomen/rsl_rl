# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the replay buffer."""

from __future__ import annotations

import torch
from tensordict import TensorDict

import pytest

from rsl_rl.storage import ReplayBuffer

GAMMA = 0.9


def _obs(values: torch.Tensor) -> TensorDict:
    """Wrap per-env scalar values as observations."""
    return TensorDict({"policy": values.float().unsqueeze(-1)}, batch_size=[values.shape[0]])


def _fill(buffer: ReplayBuffer, rewards: list[float], dones: list[bool], terminated: list[bool]) -> None:
    """Add a single-env trajectory where the observation equals the step index."""
    for t, (r, d, term) in enumerate(zip(rewards, dones, terminated)):
        buffer.add(
            _obs(torch.tensor([t])),
            torch.zeros(1, 1),
            torch.tensor([r]),
            _obs(torch.tensor([100 + t])),  # next observation encodes the step it belongs to
            torch.tensor([d]),
            torch.tensor([term]),
        )


def _sample_all(buffer: ReplayBuffer) -> dict[int, tuple[float, float, float]]:
    """Sample until every start step was seen and return (reward, next_obs, discount) per start observation."""
    seen = {}
    for _ in range(50):
        batch = buffer.sample(256)
        for i in range(256):
            start = int(batch.observations["policy"][i, 0])
            seen[start] = (
                batch.rewards[i].item(),
                batch.next_observations["policy"][i, 0].item(),
                batch.bootstrap_discounts[i].item(),
            )
    return seen


def test_n_step_returns_stop_at_episode_end_and_newest_step() -> None:
    """n-step windows end at the first done (inclusive) or the newest step and use gamma**k."""
    buffer = ReplayBuffer(1, 10, _obs(torch.zeros(1)), 1, n_steps=3, gamma=GAMMA)
    # Steps: 0 1 2(time-out) 3 4(terminal) 5
    _fill(
        buffer,
        rewards=[1, 2, 3, 4, 5, 6],
        dones=[False, False, True, False, True, False],
        terminated=[False, False, False, False, True, False],
    )
    seen = _sample_all(buffer)

    # Full window without episode end
    assert seen[0] == pytest.approx((1 + GAMMA * 2 + GAMMA**2 * 3, 102, GAMMA**3))
    # Window cut by the time-out at step 2: bootstrap from its final observation with gamma**2
    assert seen[1] == pytest.approx((2 + GAMMA * 3, 102, GAMMA**2))
    assert seen[2] == pytest.approx((3, 102, GAMMA))
    # Window cut by the terminal step 4: no bootstrapping
    assert seen[3] == pytest.approx((4 + GAMMA * 5, 104, 0.0))
    assert seen[4] == pytest.approx((5, 104, 0.0))
    # Window cut by the newest stored step
    assert seen[5] == pytest.approx((6, 105, GAMMA))


def test_wrap_around_keeps_chronological_windows() -> None:
    """Windows follow the chronological order across the ring buffer boundary and skip overwritten steps."""
    buffer = ReplayBuffer(1, 4, _obs(torch.zeros(1)), 1, n_steps=2, gamma=GAMMA)
    _fill(buffer, rewards=[1, 2, 3, 4, 5, 6], dones=[False] * 6, terminated=[False] * 6)
    seen = _sample_all(buffer)

    assert set(seen) == {2, 3, 4, 5}
    assert seen[3] == pytest.approx((4 + GAMMA * 5, 104, GAMMA**2))
    assert seen[5] == pytest.approx((6, 105, GAMMA))


def test_multiple_envs_are_independent() -> None:
    """Episode ends of one environment do not affect the windows of another environment."""
    buffer = ReplayBuffer(2, 8, _obs(torch.zeros(2)), 1, n_steps=2, gamma=GAMMA)
    for t in range(3):
        buffer.add(
            _obs(torch.tensor([t, 10 + t])),
            torch.zeros(2, 1),
            torch.tensor([1.0, 1.0]),
            _obs(torch.tensor([100 + t, 110 + t])),
            torch.tensor([t == 0, False]),
            torch.tensor([t == 0, False]),
        )
    seen = _sample_all(buffer)

    assert seen[0] == pytest.approx((1.0, 100, 0.0))
    assert seen[10] == pytest.approx((1.0 + GAMMA, 111, GAMMA**2))
