# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
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

    With ``compact=True`` the next observations are not stored separately: within an episode the next observation of a
    step is the observation of the following step of the same environment. Only the final observations of time-outs
    (which bootstrap) are kept, in a ring buffer whose size bounds the number of time-outs the buffer can hold (each
    environment times out at most once per ``max_episode_length`` steps), plus the next observation of the newest step.
    ``obs_dtype`` sets the storage precision of the observations (e.g., ``torch.float16``); sampled observations are
    returned as float32.
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
        compact: bool = False,
        obs_dtype: torch.dtype = torch.float32,
        max_episode_length: int | None = None,
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
            compact: Whether to store only the final observations of time-outs instead of all next observations.
            obs_dtype: Storage precision of the observations.
            max_episode_length: Maximum episode length (time-out length) in steps. Required with ``compact``.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}.")
        self.num_envs = num_envs
        self.num_slots = capacity // num_envs
        if self.num_slots < n_steps:
            raise ValueError(f"The replay buffer capacity ({capacity}) must hold at least {n_steps} steps per env.")
        self.n_steps = n_steps
        self.gamma = gamma
        self.device = device

        # Storage
        shape = (self.num_slots, num_envs)
        self.compact = compact
        self.obs_dtype = obs_dtype
        self.observations = {
            k: torch.zeros(*shape, *v.shape[1:], dtype=obs_dtype, device=device) for k, v in obs.items()
        }
        if compact:
            if max_episode_length is None or max_episode_length < 1:
                raise ValueError("The compact replay buffer needs the maximum episode length.")
            # Each environment times out at most once per max_episode_length steps
            self.num_final = num_envs * (self.num_slots // int(max_episode_length) + 2)
            self.final_observations = {k: torch.zeros(self.num_final, *v.shape[2:], dtype=obs_dtype, device=device)
                                       for k, v in self.observations.items()}  # fmt: skip
            self.final_index = torch.full(shape, -1, dtype=torch.long, device=device)
            self._final_pos = 0
            self.newest_next_observations = {k: torch.zeros_like(v[0]) for k, v in self.observations.items()}
        else:
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
        dones_bool = dones.view(-1).bool()
        if self.compact:
            time_outs = dones_bool & ~terminated.view(-1).bool()
            env_ids = time_outs.nonzero().flatten()
            positions = (self._final_pos + torch.arange(env_ids.numel(), device=self.device)) % self.num_final
            self.final_index[self.step].fill_(-1)
            self.final_index[self.step, env_ids] = positions
            for key, value in self.final_observations.items():
                value[positions] = next_observations[key][env_ids].to(self.obs_dtype)
                self.newest_next_observations[key].copy_(next_observations[key])
            self._final_pos = (self._final_pos + env_ids.numel()) % self.num_final
        else:
            for key, value in self.next_observations.items():
                value[self.step].copy_(next_observations[key])
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1))
        self.dones[self.step].copy_(dones_bool)
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
            next_observations=self._next_observations(last_ids),
            bootstrap_discounts=bootstrap_discounts,
        )

    def _gather(self, storage: dict[str, torch.Tensor], flat_ids: torch.Tensor) -> TensorDict:
        """Gather observations at flat (slot, env) indices (as float32)."""
        data = {k: v.flatten(0, 1)[flat_ids].float() for k, v in storage.items()}
        return TensorDict(data, batch_size=[flat_ids.shape[0]], device=self.device)

    def _next_observations(self, last_ids: torch.Tensor) -> TensorDict:
        """Return the observations to bootstrap from after the steps at the flat (slot, env) indices."""
        if not self.compact:
            return self._gather(self.next_observations, last_ids)
        slots, env_ids = last_ids // self.num_envs, last_ids % self.num_envs
        newest = (self.step - 1) % self.num_slots
        final = self.final_index.view(-1)[last_ids]
        is_final = final >= 0
        is_newest = (slots == newest) & ~is_final
        # Within an episode the next observation is the observation of the following step (terminal steps do not
        # bootstrap, so any observation will do for them)
        following = ((slots + 1) % self.num_slots) * self.num_envs + env_ids
        data = {}
        for key, value in self.observations.items():
            out = value.flatten(0, 1)[following]
            out = torch.where(
                is_newest.view(-1, *[1] * (out.dim() - 1)), self.newest_next_observations[key][env_ids], out
            )
            out = torch.where(
                is_final.view(-1, *[1] * (out.dim() - 1)), self.final_observations[key][final.clamp(min=0)], out
            )
            data[key] = out.float()
        return TensorDict(data, batch_size=[last_ids.shape[0]], device=self.device)
