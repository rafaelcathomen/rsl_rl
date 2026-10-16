# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import warnings
from dataclasses import dataclass
from tensordict import TensorDict


class ReplayBuffer:
    r"""Replay buffer for off-policy algorithms on vectorized environments.

    Transitions are stored per environment in chronological order, i.e., with shape ``[num_slots, num_envs, ...]``,
    where ``num_slots = capacity // num_envs``. This layout allows computing n-step returns at sampling time.

    An n-step window starting at step :math:`t` accumulates discounted rewards until the first step at which the episode
    ended (inclusive), the newest stored step, or :math:`n` steps, whichever comes first. If :math:`k` steps were
    accumulated, the window bootstraps from the next observation of its last step with discount :math:`\gamma^k`, or
    does not bootstrap if that step is a terminal state. Time-outs bootstrap, so the next observation stored for a
    time-out must be the final observation of the episode and not the first observation of the next episode.
    """

    @dataclass
    class Batch:
        """Batch of n-step transitions."""

        observations: TensorDict
        """Observations at the start of the window."""

        actions: torch.Tensor
        """Actions taken at the start of the window."""

        rewards: torch.Tensor
        """Discounted sum of the rewards in the window."""

        next_observations: TensorDict
        """Observations to bootstrap from at the end of the window."""

        bootstrap_discounts: torch.Tensor
        r"""Discount of the bootstrapped value, :math:`\gamma^k` or 0 if the window ended in a terminal state."""

    def __init__(
        self,
        num_envs: int,
        capacity: int,
        obs: TensorDict,
        action_dim: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        device: str = "cpu",
        max_memory_fraction: float | None = 0.5,
    ) -> None:
        """Initialize the replay buffer.

        Args:
            num_envs: Number of environments.
            capacity: Total number of transitions to store. It is rounded down to a multiple of ``num_envs``.
            obs: Example observations whose groups and shapes define the stored observations.
            action_dim: Dimension of the actions.
            n_steps: Maximum number of steps of the n-step returns.
            gamma: Discount factor.
            device: Device to store the transitions on.
            max_memory_fraction: On CUDA devices, the capacity is reduced (with a warning) so that the buffer uses at
                most this fraction of the free device memory. ``None`` disables the limit.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}.")
        if max_memory_fraction is not None and torch.device(device).type == "cuda":
            capacity = self._limit_capacity(capacity, obs, action_dim, num_envs, n_steps, device, max_memory_fraction)
        self.num_envs = num_envs
        self.num_slots = capacity // num_envs
        if self.num_slots < n_steps:
            raise ValueError(f"The replay buffer capacity ({capacity}) must hold at least {n_steps} steps per env.")
        self.n_steps = n_steps
        self.gamma = gamma
        self.device = device

        # Storage
        shape = (self.num_slots, num_envs)
        self.observations = {k: torch.zeros(*shape, *v.shape[1:], device=device) for k, v in obs.items()}
        self.next_observations = {k: torch.zeros_like(v) for k, v in self.observations.items()}
        self.actions = torch.zeros(*shape, action_dim, device=device)
        self.rewards = torch.zeros(*shape, device=device)
        self.dones = torch.zeros(*shape, dtype=torch.bool, device=device)
        self.terminated = torch.zeros(*shape, dtype=torch.bool, device=device)

        # Discount factors of the steps in an n-step window
        self._discounts = gamma ** torch.arange(n_steps + 1, device=device, dtype=torch.float)

        # Write position and number of stored steps per env
        self.step = 0
        self.num_stored = 0

    @staticmethod
    def _limit_capacity(
        capacity: int, obs: TensorDict, action_dim: int, num_envs: int, n_steps: int, device: str, fraction: float
    ) -> int:
        """Reduce the capacity so that the buffer fits into the given fraction of the free device memory."""
        # Observations and next observations, actions, rewards, and two flags per transition
        obs_bytes = sum(v[0].numel() * v.element_size() for v in obs.values())
        bytes_per_transition = 2 * obs_bytes + 4 * action_dim + 4 + 2
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        max_capacity = int(fraction * free / bytes_per_transition) // num_envs * num_envs
        if capacity <= max_capacity:
            return capacity
        reduced = max(max_capacity, num_envs * n_steps)
        warnings.warn(
            f"Reducing the replay buffer capacity from {capacity} to {reduced} transitions to fit into {fraction:.0%}"
            f" of the free GPU memory ({free / 2**30:.1f} GiB, {bytes_per_transition} bytes per transition).",
            stacklevel=3,
        )
        return reduced

    @property
    def size(self) -> int:
        """Return the number of stored transitions."""
        return self.num_stored * self.num_envs

    def add(
        self,
        observations: TensorDict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: TensorDict,
        dones: torch.Tensor,
        terminated: torch.Tensor,
    ) -> None:
        """Add one step of transitions of all environments.

        Args:
            observations: Observations before the step.
            actions: Actions taken.
            rewards: Rewards received.
            next_observations: Observations after the step. For time-outs, these must be the final observations.
            dones: Whether the episode ended at this step (terminated or timed out).
            terminated: Whether the episode ended in a terminal state (no bootstrapping).
        """
        for key, value in self.observations.items():
            value[self.step].copy_(observations[key])
            self.next_observations[key][self.step].copy_(next_observations[key])
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1))
        self.dones[self.step].copy_(dones.view(-1).bool())
        self.terminated[self.step].copy_(terminated.view(-1).bool())
        self.step = (self.step + 1) % self.num_slots
        self.num_stored = min(self.num_stored + 1, self.num_slots)

    def sample(self, batch_size: int) -> ReplayBuffer.Batch:
        """Sample a batch of n-step transitions uniformly."""
        if self.num_stored == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        oldest = (self.step - self.num_stored) % self.num_slots
        start_age = torch.randint(0, self.num_stored, (batch_size,), device=self.device)
        env_ids = torch.randint(0, self.num_envs, (batch_size,), device=self.device)

        # Steps of the n-step windows, limited to the stored steps
        ages = start_age.unsqueeze(1) + torch.arange(self.n_steps, device=self.device)
        stored = ages < self.num_stored
        slots = (oldest + ages) % self.num_slots
        flat_ids = slots * self.num_envs + env_ids.unsqueeze(1)

        # A step is included if it is stored and no earlier step of the window ended the episode
        dones = self.dones.view(-1)[flat_ids] & stored
        not_done_before = torch.cumprod((~dones).int(), dim=1).roll(1, dims=1)
        not_done_before[:, 0] = 1
        included = stored & not_done_before.bool()
        num_included = included.sum(dim=1)

        # Discounted reward sum and bootstrap discount
        rewards = (self.rewards.view(-1)[flat_ids] * self._discounts[: self.n_steps] * included).sum(dim=1)
        last_ids = flat_ids.gather(1, (num_included - 1).unsqueeze(1)).squeeze(1)
        bootstrap_discounts = self._discounts[num_included] * (~self.terminated.view(-1)[last_ids])

        first_ids = flat_ids[:, 0]
        return ReplayBuffer.Batch(
            observations=self._gather(self.observations, first_ids),
            actions=self.actions.view(-1, self.actions.shape[-1])[first_ids],
            rewards=rewards,
            next_observations=self._gather(self.next_observations, last_ids),
            bootstrap_discounts=bootstrap_discounts,
        )

    def _gather(self, storage: dict[str, torch.Tensor], flat_ids: torch.Tensor) -> TensorDict:
        """Gather observations at flat (slot, env) indices."""
        data = {k: v.flatten(0, 1)[flat_ids] for k, v in storage.items()}
        return TensorDict(data, batch_size=[flat_ids.shape[0]], device=self.device)
