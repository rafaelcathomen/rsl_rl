# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-owned differentiable state coordinates and heads for SuperMimic."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn
from typing import Protocol

State = tuple[torch.Tensor, ...]


class StateRepresentation(Protocol):
    """Differentiable state coordinates supporting arbitrary leading batch dimensions."""

    increment_dim: int
    error_floor: torch.Tensor

    def unflatten(self, flat: torch.Tensor) -> State:
        """Decode a physical observation without detaching derivatives."""
        ...

    def features(self, state: State) -> torch.Tensor:
        """Encode physical dynamics inputs."""
        ...

    def increments(self, state: State, next_state: State) -> torch.Tensor:
        """Encode the per-step successor-minus-current increment in integration coordinates."""
        ...

    def integrate(self, state: State, increment: torch.Tensor) -> State:
        """Apply a per-step physical increment; the algorithm performs no time-step scaling."""
        ...

    def error(self, prediction: State, target: State) -> torch.Tensor:
        """Return prediction errors in increment coordinates."""
        ...


class EuclideanStateRepresentation:
    """Flat Euclidean coordinates for systems without manifold state components."""

    def __init__(self, dimension: int, error_floor: float = 1e-3) -> None:
        """Set state width and the minimum per-coordinate model-loss scale."""
        if dimension < 1 or error_floor <= 0:
            raise ValueError("Euclidean state dimension and error floor must be positive.")
        self.increment_dim = dimension
        self.error_floor = torch.full((dimension,), error_floor)

    def unflatten(self, flat: torch.Tensor) -> State:
        """Keep the flat vector as one structured component."""
        return (flat,)

    def features(self, state: State) -> torch.Tensor:
        """Return the physical vector."""
        return state[0]

    def increments(self, state: State, next_state: State) -> torch.Tensor:
        """Subtract successive physical vectors."""
        return next_state[0] - state[0]

    def integrate(self, state: State, increment: torch.Tensor) -> State:
        """Add the predicted physical increment."""
        return (state[0] + increment,)

    def error(self, prediction: State, target: State) -> torch.Tensor:
        """Subtract the target physical vector."""
        return prediction[0] - target[0]


class DifferentiableTask:
    """Differentiable task heads; simulator geometry, rewards and curricula stay task-owned.

    ``prev_actions`` has shape ``[..., action_history_length, num_actions]``, oldest first. Actor observations and
    goals must use deployable inputs. Privileged context is available separately to dynamics and critic heads.
    """

    representation: StateRepresentation
    action_history_length: int = 2
    num_labels: int = 0

    def to(self, device: torch.device | str) -> DifferentiableTask:
        """Move tensor attributes used by actor heads; override for nested task state."""
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self

    def actor_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> torch.Tensor:
        """Build one noise-free deployable observation frame."""
        raise NotImplementedError

    def actor_goal(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        """Encode a deployable policy goal, preserving state derivatives."""
        raise NotImplementedError

    def advance(
        self, next_state: State, goal: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance exogenous goals and privileged context after an imagined physical transition."""
        raise NotImplementedError

    def dynamics_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Map policy outputs to physical commands before world-model normalization, keeping the action width."""
        return actions

    def dynamics_context(self, state: State, context: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        """Return normalized privileged dynamics inputs."""
        raise NotImplementedError

    def critic_obs(
        self,
        state: State,
        goal: torch.Tensor,
        context: torch.Tensor,
        prev_actions: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """Build privileged value features for the synthetic critic observation set.

        ``history`` holds the actor observation frames ending at ``state``, with shape
        ``[..., history_length, obs_dim]``, oldest first.
        """
        raise NotImplementedError

    def reward(
        self,
        state: State,
        actions: torch.Tensor,
        next_state: State,
        goal: torch.Tensor,
        prev_actions: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return transition rewards from the :meth:`pre_dynamics` start, pre-transition goal and successor labels."""
        raise NotImplementedError

    def termination_prob(
        self, next_state: State, goal: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return successor termination probabilities in [0,1]."""
        raise NotImplementedError

    def pre_dynamics(self, state: State, context: torch.Tensor) -> State:
        """Apply exogenous edits to a transition start after the actor has acted, e.g. a scheduled velocity impulse.

        Replay keeps the state the actor observed. World-model inputs and statistics, :meth:`post_dynamics` and
        :meth:`reward` use the edited state.
        """
        return state

    def post_dynamics(
        self,
        state: State,
        next_state: State,
        labels: torch.Tensor | None,
        actions: torch.Tensor,
        context: torch.Tensor,
        imagined: bool,
    ) -> State:
        """Advance task-owned analytic state after integration and successor-label prediction.

        ``state`` is the transition start returned by :meth:`pre_dynamics`, ``actions`` are the policy outputs and
        ``labels`` the output of :meth:`label_activation`. ``imagined`` is ``False`` while the world model is fitted on
        replay windows, so state that the dynamics loss does not compare can pass through unchanged.
        """
        return next_state

    def label_activation(self, logits: torch.Tensor) -> torch.Tensor:
        """Map predicted successor-label logits to the labels read by rewards, terminations and post-dynamics."""
        return torch.sigmoid(logits)

    def label_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return one successor-label loss per replay position; the default treats labels as binary events."""
        return nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=-1)

    def reference_action(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        """Return the action that the policy output corrects when the actor uses ``use_reference_action``.

        Motion-tracking tasks can return the reference joint targets in action units, so the policy learns a
        correction to the reference pose. Use deployable inputs only: exported policies take this action as an
        additional ``reference_action`` input.
        """
        raise NotImplementedError

    def mirror(self, window: TensorDict) -> TensorDict:
        """Return the left-right mirror image of replay windows for world-model and policy-start augmentation.

        World-model windows hold flat ``state``, ``next_state``, ``actions``, ``context`` and ``prev_actions`` entries
        and ``labels`` when the task has labels. Policy-start histories hold ``state``, ``goal``, ``context`` and
        ``prev_actions``. All entries have batch and time dimensions first. The mirrored transitions must be physically
        valid for the robot, e.g. by swapping left and right joints and negating lateral motion.
        """
        raise NotImplementedError

    def world_model_loss(self, prediction: State, target: State, scale: torch.Tensor) -> torch.Tensor:
        """Return one dynamics loss per replay position, including any integrated state coordinates."""
        return (self.representation.error(prediction, target) / scale).square().mean(dim=-1)

    def add_noise(self, observation: torch.Tensor) -> torch.Tensor:
        """Add task observation noise while retaining imagined state derivatives."""
        return observation

    def process_env_step(self, goal: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Update task-owned curricula from a completed physical step and its pre-step goal."""
        pass
