# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Running moments with SuperMimic's variance-floor convention."""

from __future__ import annotations

import torch
from torch import nn


class _RunningNorm(nn.Module):
    """Normalize with sqrt(variance + eps²), flattening all leading batch dimensions."""

    def __init__(self, dimension: int, eps: float = 1e-2) -> None:
        """Allocate moments before collection so inference-mode updates retain ordinary buffers."""
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("var", torch.ones(dimension))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        """Merge one observed batch using in-place writes, including the sample counter."""
        value = value.reshape(-1, value.shape[-1])
        if value.shape[0] == 0:
            return
        n = value.shape[0]
        mean, variance = value.mean(0), value.var(0, unbiased=False)
        total = self.count + n
        delta = mean - self.mean
        self.mean.add_(delta * n / total)
        self.var.copy_((self.var * self.count + variance * n + delta.square() * self.count * n / total) / total)
        self.count.copy_(total)

    @property
    def std(self) -> torch.Tensor:
        """Return the standard deviation with a variance floor."""
        return torch.sqrt(self.var + self.eps**2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize using stored moments."""
        return (value - self.mean) / self.std
