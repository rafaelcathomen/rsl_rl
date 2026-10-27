# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SuperMimic dynamics, goal-conditioned policy and value models."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.env.model_based_env import State, StateRepresentation
from rsl_rl.modules import MLP


class Mlp(MLP):
    """Shared RSL-RL MLP with optional bf16 evaluation and fp32 outputs.

    The name is retained for existing Agile model imports. Layer construction and
    parameter names follow :class:`~rsl_rl.modules.mlp.MLP`.
    """

    bf16 = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate with optional autocast while preserving fp32 outputs."""
        if torch.jit.is_scripting():
            for layer in self:
                x = layer(x)
            return x.float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            for layer in self:
                x = layer(x)
            return x.float()


def mlp(inp: int, hidden: list[int], out: int, act: type[nn.Module] = nn.ELU, layer_norm: bool = False) -> Mlp:
    """Build an MLP with a separate instance of the requested activation class per layer."""
    network = Mlp(inp, out, hidden, activation="identity", norm="layer" if layer_norm else None)
    for index, layer in enumerate(network):
        if isinstance(layer, nn.Identity):
            network[index] = act()
    return network


class RunningNorm(nn.Module):
    """Running moments with scale ``sqrt(variance + eps**2)``.

    SuperMimic checkpoints store these moments directly. Unlike
    :class:`~rsl_rl.modules.normalization.EmpiricalNormalization`, the stability
    term is inside the square root and updates flatten all leading dimensions.
    """

    def __init__(self, dim: int, eps: float = 1e-2) -> None:
        """Allocate stable running moments for the final feature dimension."""
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Merge batch moments into the running statistics outside autograd."""
        x = x.reshape(-1, x.shape[-1])
        if not len(x):
            return
        n, mean, var = len(x), x.mean(0), x.var(0, unbiased=False)
        total = self.count + n
        delta = mean - self.mean
        self.mean += delta * n / total
        self.var.copy_((self.var * self.count + var * n + delta**2 * self.count * n / total) / total)
        self.count = total

    @property
    def std(self) -> torch.Tensor:
        """Running standard deviation with a variance floor."""
        return torch.sqrt(self.var + self.eps**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input features using stored moments."""
        return (x - self.mean) / self.std


class WorldModel(nn.Module):
    """Predict normalized state increments and integrate them in task-owned coordinates."""

    def __init__(
        self,
        representation: StateRepresentation,
        num_actions: int,
        context_dim: int,
        hidden: tuple[int, ...] = (2048, 2048, 2048),
        num_labels: int = 0,
    ) -> None:
        """Build residual dynamics and optional successor-event heads.

        Args:
            representation: Task-owned features, increments and state integration.
            num_actions: Width of the physical action vector.
            context_dim: Width of normalized privileged dynamics inputs.
            hidden: Hidden layer dimensions for the increment network.
            num_labels: Number of learned successor-event probabilities, or zero.
        """
        super().__init__()
        self.representation = representation
        self.feature_dim, self.num_actions = representation.feature_dim, num_actions
        self.in_dim = self.feature_dim + num_actions + context_dim
        self.out_dim = representation.increment_dim
        self.net = Mlp(self.in_dim, self.out_dim, hidden, activation="swish", norm="layer")
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)
        self.label_head = (
            Mlp(self.feature_dim + context_dim, num_labels, [512, 512], norm="layer") if num_labels else None
        )

    def inputs(self, state: State, a: torch.Tensor, xi_n: torch.Tensor) -> torch.Tensor:
        """Concatenate state features, action and normalized privileged context."""
        return torch.cat((self.representation.features(state), a, xi_n), -1)

    def observe(self, state: State, a: torch.Tensor, xi_n: torch.Tensor, successor: State) -> None:
        """Update the input and target normalizers from a logged transition."""
        self.in_norm.update(self.inputs(state, a, xi_n))
        self.out_norm.update(self.representation.increments(state, successor))

    def label_logits(self, state: State, xi_n: torch.Tensor) -> torch.Tensor:
        """Event logits of state ``S`` (input normalized with the dynamics input statistics)."""
        feature_end = self.feature_dim
        context_start = feature_end + self.num_actions
        x = torch.cat((self.representation.features(state), xi_n), -1)
        mean = torch.cat((self.in_norm.mean[:feature_end], self.in_norm.mean[context_start:]))
        std = self.in_norm.std
        std = torch.cat((std[:feature_end], std[context_start:]))
        return self.label_head((x - mean) / std)

    def forward(self, state: State, a: torch.Tensor, xi_n: torch.Tensor) -> State:
        """Predict the physical successor without detaching the input state or action."""
        d = self.net(self.in_norm(self.inputs(state, a, xi_n))) * self.out_norm.std + self.out_norm.mean
        return self.representation.integrate(state, d)


class Policy(nn.Module):
    """``a = dec(tanh(enc(g)), o_{t-H+1:t})``: deterministic, with a 32-dimensional bounded latent."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        history: int,
        num_actions: int,
        latent_dim: int = 32,
        encoder_hidden: tuple[int, ...] | None = None,
        action_joints: torch.Tensor | None = None,
        decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256),
    ) -> None:
        """Build the goal encoder and observation-history decoder.

        Args:
            obs_dim: Width of one deployable observation.
            goal_dim: Width of the policy goal.
            history: Number of observations supplied to the decoder.
            num_actions: Width of the physical action vector.
            latent_dim: Width of the bounded goal encoding.
            encoder_hidden: Goal encoder hidden dimensions, or the default architecture.
            action_joints: Ordered controlled-action indices. Other outputs are zero.
            decoder_hidden: Action decoder hidden dimensions.
        """
        super().__init__()
        outputs = num_actions if action_joints is None else len(action_joints)
        # 0/1 map from controlled-joint to all-joint actions (a matmul: torch.compile-friendly)
        expand = None
        if action_joints is not None:
            expand = torch.zeros(outputs, num_actions, device=action_joints.device)
            expand[torch.arange(outputs), action_joints] = 1.0
        self.register_buffer("expand", expand, persistent=False)
        self.encoder = Mlp(goal_dim, latent_dim, (2048, 1024, 512, 256) if encoder_hidden is None else encoder_hidden)
        self.decoder = Mlp(latent_dim + obs_dim * history, outputs, decoder_hidden)
        with torch.no_grad():
            self.decoder[-1].weight.mul_(0.01)
            self.decoder[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, obs_history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Return actions from observation history and a goal.

        Args:
            obs_history: Noisy observations with shape ``(..., history, obs_dim)``, newest last.
            goal: Policy goals with shape ``(..., goal_dim)`` and the same leading dimensions.

        Returns:
            Actions with shape ``(..., num_actions)``.
        """
        z = torch.tanh(self.encoder(self.goal_norm(goal)))
        a = self.decoder(torch.cat((z, self.obs_norm(obs_history).flatten(-2)), -1))
        return a if self.expand is None else a @ self.expand


class Critic(nn.Module):
    """Privileged scalar value function for bootstrapping imagined returns."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (256, 128)) -> None:
        """Build a value MLP and its input normalizer."""
        super().__init__()
        self.net = Mlp(in_dim, 1, hidden)
        self.norm = RunningNorm(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a scalar value for each batch element."""
        return self.net(self.norm(x)).squeeze(-1)
