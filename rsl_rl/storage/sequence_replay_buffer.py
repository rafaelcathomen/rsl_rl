# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-size GPU replay buffer of simulator transitions, stored as (time, env) rings."""

from __future__ import annotations

import torch


class SequenceReplayBuffer:
    """Chronological per-environment replay with reset-aware windows and history."""

    def __init__(
        self,
        steps: int,
        num_envs: int,
        example: dict[str, torch.Tensor],
        minimum_steps: int = 1,
        max_memory_fraction: float | None = None,
    ) -> None:
        """Allocate replay, optionally bounded by free GPU memory."""
        if num_envs < 1 or minimum_steps < 1 or steps < minimum_steps or not example:
            raise ValueError("Replay requires positive dimensions and enough steps for its windows")
        if any(v.shape[0] != num_envs for v in example.values()):
            raise ValueError("Replay tensors must have num_envs as their first dimension")
        device = next(iter(example.values())).device
        if max_memory_fraction is not None:
            if not 0 < max_memory_fraction <= 1:
                raise ValueError("max_memory_fraction must lie in (0, 1]")
            if device.type == "cuda":
                import warnings

                free, _ = torch.cuda.mem_get_info(device)
                bytes_per_step = sum(v.numel() * v.element_size() for v in example.values())
                affordable = int(free * max_memory_fraction) // bytes_per_step
                if affordable < minimum_steps:
                    raise ValueError("Insufficient free GPU memory for the minimum replay window")
                if affordable < steps:
                    warnings.warn(
                        f"Reducing SuperMimic replay from {steps} to {affordable} steps per env", stacklevel=2
                    )
                    steps = affordable
        self.steps, self.num_envs = steps, num_envs
        self.data = {k: torch.empty((steps, *v.shape), dtype=v.dtype, device=v.device) for k, v in example.items()}
        self.device = device
        self.ptr, self.size = 0, 0

    def add(self, transition: dict[str, torch.Tensor]) -> None:
        """Copy a physical transition before simulator buffers can be reused."""
        for k, buf in self.data.items():
            buf[self.ptr] = transition[k]
        self.ptr = (self.ptr + 1) % self.steps
        self.size = min(self.size + 1, self.steps)

    def _rows(self, age: torch.Tensor) -> torch.Tensor:
        """Storage row of an age-ordered time index (0 = oldest)."""
        return (self.ptr - self.size + age) % self.steps

    def _sample(self, n: int, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        if n < 1 or not 1 <= horizon <= self.size:
            raise ValueError("Sampling requires a positive batch and a horizon covered by stored replay")
        age = torch.randint(0, self.size - horizon + 1, (n,), device=self.device)
        return age, torch.randint(0, self.num_envs, (n,), device=self.device)

    def windows(self, n: int, k: int, keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
        """``n`` windows of ``k`` consecutive transitions of one env, as (n, k, ...) tensors.

        ``mask[:, j]`` is zero once an episode ended inside the window before transition ``j``.
        """
        age, env = self._sample(n, k)
        rows = self._rows(age.unsqueeze(-1) + torch.arange(k, device=self.device))
        out = {key: self.data[key][rows, env.unsqueeze(-1)] for key in {*keys, "done"}}
        alive = 1 - out["done"].float()
        out["mask"] = torch.cumprod(torch.cat((torch.ones_like(alive[:, :1]), alive[:, :-1]), -1), -1)
        return out

    def starts(
        self, n: int, history: int, keys: tuple[str, ...]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """``n`` start transitions and their last ``history`` transitions (oldest first, clipped to the episode)."""
        age, env = self._sample(n, 1)
        rows = self._rows(age)
        back = torch.arange(history - 1, -1, -1, device=self.device)
        back = torch.minimum(back, torch.minimum(self.data["episode_step"][rows, env], age).unsqueeze(-1))
        hist_rows = self._rows(age.unsqueeze(-1) - back)
        start = {key: self.data[key][rows, env] for key in keys}
        hist = {key: self.data[key][hist_rows, env.unsqueeze(-1)] for key in keys}
        return start, hist
