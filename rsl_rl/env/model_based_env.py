# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for model-based learning, independent of a simulator or robot geometry."""

from __future__ import annotations

import torch
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


class DisturbanceModel(Protocol):
    """Optional task-owned disturbance state advanced alongside imagined dynamics."""

    feature_dim: int
    state_dim: int

    def features(self, disturbance: torch.Tensor, state: State) -> torch.Tensor:
        """Return privileged dynamics inputs for the current disturbance and state."""
        ...

    def apply_impulse(self, state: State, disturbance: torch.Tensor) -> State:
        """Apply the transition's impulse while preserving state derivatives."""
        ...

    def advance(self, disturbance: torch.Tensor) -> torch.Tensor:
        """Advance the disturbance schedule by one physical step."""
        ...

    def critic_features(self, disturbance: torch.Tensor) -> torch.Tensor:
        """Return privileged value-function inputs for the disturbance state."""
        ...


class DifferentiableTask:
    """Differentiable task heads used by SuperMimic.

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
    """Width of the privileged physical context, excluding disturbance and model extras."""
    critic_dim: int
    """Width of task critic features, excluding disturbance and critic extras."""
    max_step_reward: float
    label_names: tuple[str, ...] = ()
    encoder_hidden: tuple[int, ...] | None = None
    action_joints: torch.Tensor | None = None
    disturbances: DisturbanceModel | None = None
    wm_extra_dim: int = 0
    critic_extra_dim: int = 0

    def obs(self, state: State, previous_action: torch.Tensor, xi: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Return deployable observations before noise, preserving the leading batch dimensions."""
        raise NotImplementedError

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply the task's observation noise without detaching state derivatives."""
        raise NotImplementedError

    def policy_goal(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        """Encode the policy goal, retaining derivatives of state-dependent targets."""
        raise NotImplementedError

    def advance(self, goal: torch.Tensor) -> torch.Tensor:
        """Advance exogenous goal state by one physical step."""
        raise NotImplementedError

    def valid_start(self, goal: torch.Tensor, horizon: int) -> torch.Tensor:
        """Return a boolean mask of goals valid for the requested imagination horizon."""
        raise NotImplementedError

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        """Normalize privileged physical context using task-owned units and ranges."""
        raise NotImplementedError

    def reward(
        self,
        state: State,
        goal: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        *,
        a_prev2: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        S_prev: State | None = None,  # noqa: N803
        g_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate transition rewards on successor state and goal.

        ``a_prev2`` is the action preceding ``previous_action``. ``labels`` are
        successor event probabilities. ``S_prev`` and ``g_prev`` are the state
        and goal before the transition. Return one reward per batch element.
        """
        raise NotImplementedError

    def termination_prob(self, state: State, goal: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Return successor termination probabilities in [0, 1], masking only future rewards."""
        raise NotImplementedError

    def critic_features(
        self,
        state: State,
        goal: torch.Tensor,
        xi: torch.Tensor,
        a_prev: torch.Tensor | None = None,
        a_prev2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return privileged value inputs using normalized context and the two preceding actions."""
        raise NotImplementedError

    def wm_extra(self, state: State, previous_action: torch.Tensor | None) -> torch.Tensor:
        """Return extra dynamics context when ``wm_extra_dim`` is positive."""
        raise NotImplementedError

    def critic_extra(self, state: State) -> torch.Tensor:
        """Return extra value inputs when ``critic_extra_dim`` is positive."""
        raise NotImplementedError

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
