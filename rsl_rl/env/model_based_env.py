# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for model-based learning, independent of a simulator or robot geometry."""

from __future__ import annotations

import torch
from collections.abc import Callable
from typing import Protocol

State = tuple[torch.Tensor, ...]
"""Structured physical state; every component has the same leading batch dimensions."""


class StateRepresentation(Protocol):
    """Task-owned coordinates for learning and integrating physical state increments.

    A floating-base robot may use invariant features and SO(3) increments. A Euclidean
    system can use a one-element state tuple and ordinary subtraction/addition. All
    methods must preserve arbitrary leading batch dimensions and autograd.
    """

    feature_dim: int
    increment_dim: int
    error_floor: torch.Tensor

    def unflatten(self, state: torch.Tensor) -> State:
        """Decode a replay state without detaching it."""
        ...

    def features(self, state: State) -> torch.Tensor:
        """Encode dynamics inputs (without actions or privileged context)."""
        ...

    def increments(self, state: State, successor: State) -> torch.Tensor:
        """Encode the target increment from a real transition."""
        ...

    def integrate(self, state: State, increment: torch.Tensor) -> State:
        """Apply a predicted increment using differentiable manifold operations."""
        ...

    def error(self, prediction: State, target: State) -> torch.Tensor:
        """Return smooth prediction errors in increment coordinates."""
        ...


class DifferentiableTask:
    """Differentiable task heads used by SuperMimic.

    Implement ``obs(state, previous_action, xi, goal)``, ``add_noise(obs)``,
    ``policy_goal(state, goal)``, ``advance(goal)``, ``valid_start(goal, horizon)``,
    ``normalize_xi(xi)``, ``reward(...)``, ``termination_prob(...)`` and
    ``critic_features(...)``. Reward receives successor state/goal, action, previous
    action, and keyword arguments ``a_prev2``, ``labels``, ``S_prev``, ``g_prev``.
    Termination receives successor state/goal and ``labels``. Critic features receive
    state/goal, normalized context and ``a_prev``, ``a_prev2``. Soft termination lies
    in [0, 1]; it masks future rewards, never the reward of the current transition.

    Actor inputs must contain only deployable observations and goals. Physical
    context, disturbances and terrain may be exposed separately to the world model
    and critic. Task functions must not detach imagined states or surrogate gradients.
    """

    name: str
    representation: StateRepresentation
    history_length: int
    num_actions: int
    obs_dim: int
    goal_dim: int
    xi_dim: int
    critic_dim: int
    max_step_reward: float
    label_names: tuple[str, ...] = ()
    encoder_hidden: tuple[int, ...] | None = None
    action_joints: torch.Tensor | None = None
    disturbances = None
    wm_extra_dim: int = 0
    critic_extra_dim: int = 0
    obs: Callable[[State, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    add_noise: Callable[[torch.Tensor], torch.Tensor]
    policy_goal: Callable[[State, torch.Tensor], torch.Tensor]
    advance: Callable[[torch.Tensor], torch.Tensor]
    valid_start: Callable[[torch.Tensor, int], torch.Tensor]
    normalize_xi: Callable[[torch.Tensor], torch.Tensor]
    reward: Callable[..., torch.Tensor]
    termination_prob: Callable[..., torch.Tensor]
    critic_features: Callable[..., torch.Tensor]
    wm_extra: Callable[[State, torch.Tensor], torch.Tensor]
    critic_extra: Callable[[State], torch.Tensor]

    def begin_iteration(self) -> None:
        """Freeze/update task sampling state once before collection and imagination."""

    def model_loss(self, prediction: State, target: State, mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Return optional auxiliary model loss and detached validation metrics."""
        return prediction[0].new_zeros(()), {}

    def checkpoint_metadata(self) -> dict:
        """Describe task/context semantics; full loads require this metadata to match."""
        return {}

    def training_state_dict(self) -> dict:
        """Return optional task training state (for example, a sampling curriculum)."""
        return {}

    def load_training_state_dict(self, state: dict) -> None:
        """Restore task training state before collection resumes."""
        if state:
            raise ValueError("This task does not support checkpointed training state")


class ModelBasedEnv(Protocol):
    """Real transition source for SuperMimic, including privileged physical states.

    Unlike ``VecEnv``, ``step`` returns pre-reset physical transitions, not just
    post-reset observations. Required transition keys are ``state``, ``next_state``,
    ``action``, ``prev_action``, ``prev_prev_action``, ``xi``, ``goal_state``,
    ``next_goal_state``, ``episode_step``, ``done``, ``reward``. ``done`` includes
    truncations; it prevents replay windows/history from crossing any reset.
    ``next_state`` and ``next_goal_state`` must be the actual pre-reset successor.
    Add ``labels`` for learned event heads and ``disturbance`` when enabled.

    Environment buffers may be reused only after the learner has copied the
    transition into replay. Collection and simulation are never differentiated.
    """

    num_envs: int
    device: torch.device | str
    task: DifferentiableTask
    prev_action: torch.Tensor
    xi: torch.Tensor
    goal_state: torch.Tensor
    episode_step: torch.Tensor

    def state(self) -> State:
        """Return the current physical state in task coordinates."""
        ...

    def step(self, actions: torch.Tensor) -> dict:
        """Apply actions, record the physical transition, then reset completed environments."""
        ...
