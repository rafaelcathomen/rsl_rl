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

    With ``compact=True`` the next observations are not stored separately: within an episode the next observation of a
    step is the observation of the following step of the same environment. Only the final observations of time-outs
    (which bootstrap) are kept in a recycled pool, plus the next observation of the newest step. The pool grows for
    early truncations; ``max_episode_length`` only estimates its initial size. GPU capacity budgeting reserves the
    worst-case pool and its temporary growth allocations, even when ordinary episodes use much less storage.
    ``obs_dtype`` sets the storage precision of the floating-point observations (e.g., ``torch.float16``); integer
    observations (e.g., uint8 camera images) keep their dtype. Sampled observations are returned as float32.
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
            max_memory_fraction: On CUDA devices, the capacity is reduced (with a warning) so that the buffer uses at
                most this fraction of the free device memory. ``None`` disables the limit.
            compact: Whether to store only the final observations of time-outs instead of all next observations.
            obs_dtype: Storage precision of the observations.
            max_episode_length: Maximum episode length (time-out length) in steps. Required with ``compact``.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}.")
        if max_memory_fraction is not None and not 0 < max_memory_fraction <= 1:
            raise ValueError("max_memory_fraction must lie in (0, 1].")
        if max_memory_fraction is not None and torch.device(device).type == "cuda":
            capacity = self._limit_capacity(
                capacity, obs, action_dim, num_envs, n_steps, device, max_memory_fraction, compact, obs_dtype
            )
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
            k: torch.zeros(
                *shape, *v.shape[1:], dtype=obs_dtype if v.dtype.is_floating_point else v.dtype, device=device
            )
            for k, v in obs.items()
        }
        if compact:
            if max_episode_length is None or max_episode_length < 1:
                raise ValueError("The compact replay buffer needs the maximum episode length.")
            self.num_final = min(num_envs * (self.num_slots // int(max_episode_length) + 2), self.num_slots * num_envs)
            self.final_observations = {k: torch.zeros(self.num_final, *v.shape[2:], dtype=v.dtype, device=device)
                                       for k, v in self.observations.items()}  # fmt: skip
            self.final_index = torch.full(shape, -1, dtype=torch.long, device=device)
            self._free_final = torch.arange(self.num_final, device=device)
            self._num_free_final = self.num_final
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

    @staticmethod
    def _limit_capacity(
        capacity: int,
        obs: TensorDict,
        action_dim: int,
        num_envs: int,
        n_steps: int,
        device: str,
        fraction: float,
        compact: bool,
        obs_dtype: torch.dtype,
    ) -> int:
        """Reduce the capacity so that the buffer fits into the given fraction of the free device memory."""
        float_size = torch.empty((), dtype=obs_dtype).element_size()
        obs_bytes = sum(
            v[0].numel() * (float_size if v.dtype.is_floating_point else v.element_size()) for v in obs.values()
        )
        # Compact storage can need a final frame for every transition. Growth temporarily retains both pools
        # and free-index stacks; budget that peak rather than assuming timeouts only occur at the episode limit.
        bytes_per_transition = (3 * obs_bytes + 24 if compact else 2 * obs_bytes) + 4 * action_dim + 4 + 2
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        fixed_bytes = num_envs * (obs_bytes + 24) + 4 * (n_steps + 1) if compact else 0
        max_capacity = int((fraction * free - fixed_bytes) / bytes_per_transition) // num_envs * num_envs
        if max_capacity < num_envs * n_steps:
            raise ValueError("Insufficient free GPU memory for the minimum replay window.")
        if capacity <= max_capacity:
            return capacity
        reduced = max_capacity
        warnings.warn(
            f"Reducing the replay buffer capacity from {capacity} to {reduced} transitions to fit into {fraction:.0%}"
            f" of the free GPU memory ({free / 2**30:.1f} GiB, {bytes_per_transition} bytes per transition).",
            stacklevel=3,
        )
        return reduced

    def _grow_final_storage(self, needed: int) -> None:
        """Extend the free stack and every observation group without moving existing final indices."""
        capacity = min(self.num_slots * self.num_envs, max(2 * self.num_final, self.num_final + needed))
        storage = {}
        for key, value in self.final_observations.items():
            expanded = value.new_zeros((capacity, *value.shape[1:]))
            expanded[: self.num_final].copy_(value)
            storage[key] = expanded
        free = self._free_final.new_empty(capacity)
        free[: self._num_free_final].copy_(self._free_final[: self._num_free_final])
        added = capacity - self.num_final
        free[self._num_free_final : self._num_free_final + added] = torch.arange(
            self.num_final, capacity, device=self.device
        )
        self.final_observations = storage
        self._free_final = free
        self._num_free_final += added
        self.num_final = capacity

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
        dones_bool = dones.view(-1).bool()
        if self.compact:
            time_outs = dones_bool & ~terminated.view(-1).bool()
            env_ids = time_outs.nonzero().flatten()
            outgoing = self.final_index[self.step]
            released = outgoing[outgoing >= 0]
            missing = env_ids.numel() - self._num_free_final - released.numel()
            if missing > 0:
                self._grow_final_storage(missing)
            self._free_final[self._num_free_final : self._num_free_final + released.numel()] = released
            self._num_free_final += released.numel()
            self._num_free_final -= env_ids.numel()
            positions = self._free_final[self._num_free_final : self._num_free_final + env_ids.numel()]
            self.final_index[self.step].fill_(-1)
            self.final_index[self.step, env_ids] = positions
            for key, value in self.final_observations.items():
                value[positions] = next_observations[key][env_ids].to(value.dtype)
                self.newest_next_observations[key].copy_(next_observations[key])
        else:
            for key, value in self.next_observations.items():
                value[self.step].copy_(next_observations[key])
        for key, value in self.observations.items():
            value[self.step].copy_(observations[key])
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
