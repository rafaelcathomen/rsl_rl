# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gradient sources for model-based SHAC: the dynamics contract and learned world-model dynamics."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn
from typing import Any, NamedTuple, Protocol

from rsl_rl.algorithms.supermimic_task import DifferentiableTask, State
from rsl_rl.modules.world_model import WorldModel
from rsl_rl.storage import SequenceReplayBuffer
from rsl_rl.utils import compile_model


class Imagined(NamedTuple):
    """Actor and critic inputs at one imagined state; the algorithm never reads ``state``."""

    state: Any
    history: torch.Tensor
    actor_goal: torch.Tensor
    critic_obs: torch.Tensor


class Dynamics(Protocol):
    """Differentiable transition source of model-based SHAC, e.g. a learned world model or a simulator.

    ``model`` holds trainable source parameters or is None. The algorithm optimizes, schedules, freezes, switches the
    mode of and checkpoints it as ``world_model``, and calls ``model_loss`` only when it is set. ``step`` preserves
    state and action derivatives, evaluates the reward with the pre-transition goal and returns the successor inputs,
    its reward and its conditional termination probability. Neither ``sample_starts`` nor lifecycle calls may alias
    live collection buffers. Starts have collection-distributed observation noise. ``observe`` runs under inference
    mode and returns detached scalars without drawing noise. Tensors entering the rollout graph must not be inference
    tensors. Sources must return finite inputs, rewards and termination probabilities for every row: divergent rows
    terminate with detached finite placeholders, sanitizing both branches before selecting with ``torch.where``.
    ``step`` uses ``task_transition`` so rewards, action history and noise retain shared task semantics. ``clear``
    discards replay and pending state, not model parameters.
    """

    model: nn.Module | None

    @property
    def ready(self) -> bool:
        """Whether collected starts permit learning."""
        ...

    def record_start(self, obs: TensorDict, previous_actions: torch.Tensor) -> None:
        """Own the pre-action state without drawing noise."""
        ...

    def observe(
        self, obs: TensorDict, actions: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> dict[str, torch.Tensor]:
        """Record an executed transition and return detached collection scalars."""
        ...

    def model_loss(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return one trainable source loss and detached metrics."""
        ...

    def sample_starts(self, batch_size: int, history_length: int) -> Imagined:
        """Sample owned starts with collection-distributed observation histories."""
        ...

    def step(self, current: Imagined, actions: torch.Tensor) -> tuple[Imagined, torch.Tensor, torch.Tensor]:
        """Advance differentiably and return successor inputs, reward and termination."""
        ...

    def clear(self) -> None:
        """Discard replay and pending collection state."""
        ...


class TaskState(NamedTuple):
    """Task-coordinate state shared by differentiable transition sources."""

    physical: State
    goal: torch.Tensor
    context: torch.Tensor
    previous_actions: torch.Tensor


def task_inputs(task: DifferentiableTask, state: TaskState, history: torch.Tensor, mixed_precision: bool) -> Imagined:
    """Build critic features, then the actor goal under the actor's autocast."""
    critic_obs = task.critic_obs(*state)
    with torch.autocast(history.device.type, dtype=torch.bfloat16, enabled=mixed_precision):
        actor_goal = task.actor_goal(state.physical, state.goal)
    return Imagined(state, history, actor_goal, critic_obs)


def task_transition(
    task: DifferentiableTask,
    current: Imagined,
    actions: torch.Tensor,
    successor: State,
    labels: torch.Tensor | None,
    mixed_precision: bool,
) -> tuple[Imagined, torch.Tensor, torch.Tensor]:
    """Apply task heads around a physical successor already advanced by ``post_dynamics``."""
    physical, goal, context, previous = current.state
    reward = task.reward(physical, actions, successor, goal, previous, labels=labels)
    termination = task.termination_prob(successor, goal, labels=labels)
    goal, context = task.advance(successor, goal, context)
    previous = torch.cat((previous[:, 1:], actions[:, None]), dim=1)
    frame = task.add_noise(task.actor_obs(successor, goal, context, previous))
    history = torch.cat((current.history[:, 1:], frame[:, None]), dim=1)
    state = TaskState(successor, goal, context, previous)
    return task_inputs(task, state, history, mixed_precision), reward, termination


class LearnedDynamics:
    """Learned physical increments and event probabilities, fitted on replayed real transitions.

    Args:
        world_model: Differentiable physical increment and optional event predictor.
        storage: Lazy replay holding complete physical transitions.
        task: Differentiable task heads and physical state coordinates, shared with the actor.
        obs_groups: Explicit physical observation-set mapping.
        world_model_window: Maximum autoregressive dynamics fitting horizon.
        world_model_batch_size: Replay anchors per dynamics optimizer step.
        warmup_steps: Stored steps per environment required before learning.
        use_mixed_precision: Enable BF16 matrix operations while integrating physical states in FP32.
        device: Device for the model and replay tensors.
    """

    def __init__(
        self,
        world_model: WorldModel,
        storage: SequenceReplayBuffer,
        task: DifferentiableTask,
        obs_groups: dict[str, list[str]],
        world_model_window: int = 8,
        world_model_batch_size: int = 2048,
        warmup_steps: int = 64,
        use_mixed_precision: bool = False,
        device: str = "cpu",
    ) -> None:
        """Bind the learned model, task and replay with unchanged transition coordinates."""
        if min(world_model_window, world_model_batch_size, warmup_steps) < 1:
            raise ValueError("Learning counts, cadence, learning rate and gradient limit must be positive.")
        if warmup_steps < world_model_window:
            raise ValueError("warmup_steps must cover the model window; world-model warmup must be nonnegative.")
        self.model = world_model.to(device)
        self.storage, self.task, self.obs_groups = storage, task, obs_groups
        self.window, self.batch_size, self.warmup_steps = world_model_window, world_model_batch_size, warmup_steps
        self.use_mixed_precision, self.device = use_mixed_precision, device
        self.error_floor = task.representation.error_floor.to(device)
        if self.error_floor.shape != (task.representation.increment_dim,) or (self.error_floor <= 0).any():
            raise ValueError("State error_floor must contain one positive value per increment coordinate.")
        self._forward = self.model
        self._last_obs: TensorDict | None = None
        self._last_previous_actions: torch.Tensor | None = None

    @property
    def ready(self) -> bool:
        """Whether replay covers the configured collection warmup."""
        return self.storage.num_steps >= self.warmup_steps

    def _set(self, obs: TensorDict, name: str) -> torch.Tensor:
        keys = self.obs_groups.get(name)
        return (
            torch.cat([obs[key] for key in keys], dim=-1)
            if keys
            else obs[self.obs_groups["state"][0]].new_empty((*obs.batch_size, 0))
        )

    def record_start(self, obs: TensorDict, previous_actions: torch.Tensor) -> None:
        """Copy the physical start and action predecessors before the actor advances them."""
        self._last_previous_actions = previous_actions.clone()
        keys = {key for name in ("state", "goal", "context") for key in self.obs_groups.get(name, ())}
        self._last_obs = obs.select(*keys).clone()

    @torch.no_grad()
    def observe(
        self, obs: TensorDict, actions: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> dict[str, torch.Tensor]:
        """Record every-step pre-event successors and update dynamics statistics in place."""
        if self._last_obs is None:
            raise RuntimeError("act() must precede process_env_step().")
        if extras.get("final_obs") is None or extras.get("final_obs_every_step") is not True:
            raise ValueError("SuperMimic requires every-step extras['final_obs'] with final_obs_every_step=True.")
        final = extras["final_obs"].to(self.device)
        state, successor = self._set(self._last_obs, "state"), self._set(final, "state")
        context, goal = self._set(self._last_obs, "context"), self._set(self._last_obs, "goal")
        labels = self._set(final, "labels") if self.task.num_labels else None
        valid = torch.isfinite(successor).all(dim=-1)
        if labels is not None:
            valid &= torch.isfinite(labels).all(dim=-1)
        if not valid.any():
            raise ValueError("Every physical successor is non-finite; refusing to hide a global failure.")
        if "invalid_transition" in extras:
            valid &= ~extras["invalid_transition"].reshape(-1).bool()
        has_invalid = not valid.all()
        if has_invalid:
            successor = torch.where(valid[:, None], successor, state)
            if labels is not None:
                labels = torch.where(valid[:, None], labels, 0)
        done = dones.reshape(-1).bool()
        window_break = ((self._set(obs, "state") != successor).any(dim=-1) & ~done) | ~valid
        transition = TensorDict(
            {
                "state": state,
                "next_state": successor,
                "goal": goal,
                "context": context,
                "actions": actions,
                "prev_actions": self._last_previous_actions,
                "dones": done,
                "window_break": window_break,
                "valid": valid,
            },
            batch_size=[self.storage.num_envs],
            device=self.device,
        )
        if labels is not None:
            transition["labels"] = labels
        self.storage.add(transition)
        observed = transition[valid] if has_invalid else transition
        if observed.batch_size[0]:
            start = self.task.representation.unflatten(observed["state"])
            self.model.observe(
                start,
                observed["actions"],
                self.task.dynamics_context(start, observed["context"], observed["prev_actions"]),
                self.task.representation.unflatten(observed["next_state"]),
            )
        self._last_obs = self._last_previous_actions = None
        return {"window_break_fraction": window_break.float().mean()}

    def clear(self) -> None:
        """Discard replay and pending collection state; model parameters are unchanged."""
        self.storage.clear()
        self._last_obs = self._last_previous_actions = None

    def compile(self, mode: str | None = None) -> None:
        """Compile the differentiable increment forward; event labels stay eager."""
        self._forward = compile_model(self.model, mode)

    def _windows(self, batch_size: int) -> tuple[TensorDict, torch.Tensor]:
        """Sample uniform anchors, additionally cutting world-model targets after external state edits."""
        index = self.storage.sample(batch_size, length=self.window)
        keys = ["state", "next_state", "actions", "context", "prev_actions", "window_break", "valid"]
        if self.task.num_labels:
            keys.append("labels")
        window = self.storage.gather(keys, index.ids)
        mask = index.valid.float() * window["valid"]
        if self.window > 1:
            mask[:, 1:] *= torch.cumprod((~window["window_break"][:, :-1]).long(), dim=1)
        return window, mask

    def _predict(
        self, state: State, actions: torch.Tensor, context: torch.Tensor, previous_actions: torch.Tensor
    ) -> tuple[State, torch.Tensor | None, torch.Tensor | None]:
        """Apply learned dynamics, event probabilities and task-owned analytic state."""
        with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16, enabled=self.use_mixed_precision):
            raw_next = self._forward(state, actions, self.task.dynamics_context(state, context, previous_actions))
            logits = (
                self.model.label_logits(raw_next, self.task.dynamics_context(raw_next, context, previous_actions))
                if self.task.num_labels
                else None
            )
        labels = torch.sigmoid(logits.float()) if logits is not None else None
        return self.task.post_dynamics(state, raw_next, labels), logits, labels

    def model_loss(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Autoregressive physical-successor and event-label loss on valid replay positions."""
        window, mask = self._windows(self.batch_size)
        state = self.task.representation.unflatten(window["state"][:, 0])
        predictions, logits = [], []
        for step in range(self.window):
            state, event_logits, _ = self._predict(
                state, window["actions"][:, step], window["context"][:, step], window["prev_actions"][:, step]
            )
            predictions.append(state)
            if event_logits is not None:
                logits.append(event_logits)
        prediction = tuple(torch.stack(values, dim=1) for values in zip(*predictions))
        target = self.task.representation.unflatten(window["next_state"])
        scale = torch.maximum(self.model.out_norm.std, self.error_floor)
        errors = (self.task.representation.error(prediction, target) / scale).square().mean(dim=-1)
        loss = (errors * mask).sum(dim=1).mean()
        metrics = {"world_model_error": (errors.detach() * mask).sum() / mask.sum().clamp(min=1)}
        if self.task.num_labels:
            bce = nn.functional.binary_cross_entropy_with_logits(
                torch.stack(logits, 1).float(), window["labels"].float(), reduction="none"
            )
            label_loss = (bce.mean(dim=-1) * mask).sum() / mask.sum().clamp(min=1)
            loss = loss + label_loss
            metrics["label_loss"] = label_loss.detach()
        metrics["world_model_loss"] = loss.detach() / self.window
        return loss, metrics

    def sample_starts(self, batch_size: int, history_length: int) -> Imagined:
        """Build reset-aware actor histories, repeating the earliest available frame when needed."""
        index = self.storage.sample(batch_size, history=history_length - 1)
        history = self.storage.gather(("state", "goal", "context", "prev_actions"), index.ids)
        state = self.task.representation.unflatten(history["state"])
        frames = self.task.add_noise(
            self.task.actor_obs(state, history["goal"], history["context"], history["prev_actions"])
        )
        first = index.valid.long().argmax(dim=1)
        first_frame = frames[torch.arange(batch_size, device=frames.device), first]
        frames = torch.where(index.valid[..., None], frames, first_frame[:, None])
        start = history[:, -1]
        physical = self.task.representation.unflatten(start["state"])
        return task_inputs(
            self.task,
            TaskState(physical, start["goal"], start["context"], start["prev_actions"]),
            frames,
            self.use_mixed_precision,
        )

    def step(self, current: Imagined, actions: torch.Tensor) -> tuple[Imagined, torch.Tensor, torch.Tensor]:
        """Predict one differentiable transition and its reward, termination and successor inputs."""
        physical, _, context, previous = current.state
        successor, _, labels = self._predict(physical, actions, context, previous)
        return task_transition(self.task, current, actions, successor, labels, self.use_mixed_precision)
