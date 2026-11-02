# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic world-model fitting and differentiable policy optimization."""

from __future__ import annotations

import math
import torch
from collections.abc import Generator
from contextlib import contextmanager
from tensordict import TensorDict
from torch import nn

from rsl_rl.algorithms.supermimic_task import DifferentiableTask, State
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.models.supermimic_actor import SuperMimicActor
from rsl_rl.modules.world_model import WorldModel
from rsl_rl.storage import SequenceReplayBuffer
from rsl_rl.utils import compile_model, resolve_class, resolve_obs_groups


def cosine_learning_rate(initial: float, iteration: int, decay_iterations: int | None, final_ratio: float) -> float:
    """Cosine schedule clamped at its floor; None keeps the rate constant."""
    if decay_iterations is None:
        return initial
    progress = min(iteration / decay_iterations, 1.0)
    return initial * (final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def lambda_returns(
    rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
) -> torch.Tensor:
    """Compute TD(lambda) targets, retaining terminal rewards and the final bootstrap state."""
    target, targets = values[-1], []
    for index in reversed(range(rewards.shape[0])):
        target = rewards[index] + gamma * (1 - dones[index]) * ((1 - lam) * values[index + 1] + lam * target)
        targets.append(target)
    return torch.stack(targets[::-1])


class SuperMimic:
    """Fit physical dynamics and optimize a goal-conditioned actor through imagined trajectories.

    The environment supplies noise-free state, goal, context and event-label observation groups. Every-step final
    observations preserve the physical successor before reset and interval events. The task owns geometry, rewards,
    analytic state updates and deployment features.

    Optional twin critics and entropy regularization follow SAPO's clipped double critics and normalized soft returns.

    References:
        SuperMimic. arXiv:2610.28480 (2026). https://arxiv.org/abs/2610.28480
        Xing et al. "Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation." arXiv:2412.12089
        (2025).
    """

    def __init__(
        self,
        actor: SuperMimicActor,
        world_model: WorldModel,
        critic: MLPModel,
        storage: SequenceReplayBuffer,
        task: DifferentiableTask,
        obs_groups: dict[str, list[str]],
        updates_per_step: float = 0.4,
        num_world_model_updates: int = 10,
        num_policy_updates: int = 1,
        world_model_window: int = 8,
        world_model_batch_size: int = 2048,
        horizon: int = 32,
        policy_batch_size: int = 4096,
        learning_rate: float = 2e-4,
        lr_decay_iterations: int = 250_000,
        lr_decay_iterations_by_optimizer: dict[str, int | None] | None = None,
        lr_final_ratio: float = 0.03,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        critic_epochs: int = 5,
        critic_minibatches: int = 4,
        twin_critic: MLPModel | None = None,
        actor_value_reduction: str = "mean",
        critic_zero_init: bool = False,
        warmup_steps: int = 64,
        world_model_warmup_iterations: int = 500,
        critic_warmup: bool = False,
        action_noise: float = 0.1,
        noise_faithful_imagination: bool = False,
        entropy_regularization: bool = False,
        init_alpha: float = 1.0,
        target_entropy_scale: float | None = None,
        max_step_reward: float | None = None,
        use_mixed_precision: bool = False,
        max_grad_norm: float = 1.0,
        actor_grad_clip_value: float | None = 0.1,
        actor_ema_halflife: int | None = None,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize explicit task, models, replay and gradient-update cadence.

        The actor Adam optimizer uses betas (0.9, 0.95).

        Args:
            actor: History-conditioned deployment actor.
            world_model: Differentiable physical increment and optional event predictor.
            critic: Scalar value model consuming synthetic task features.
            storage: Lazy replay holding complete physical transitions.
            task: Differentiable task heads and physical state coordinates.
            obs_groups: Explicit physical observation-set mapping.
            updates_per_step: Learner iterations accrued per collected environment step.
            num_world_model_updates: Dynamics optimizer steps per learner iteration.
            num_policy_updates: Actor updates and associated critic fits per learner iteration.
            world_model_window: Maximum autoregressive dynamics fitting horizon.
            world_model_batch_size: Replay anchors per dynamics optimizer step.
            horizon: Imagined policy horizon.
            policy_batch_size: Starts per actor update.
            learning_rate: Initial learning rate for all optimizers.
            lr_decay_iterations_by_optimizer: Optional decay lengths by optimizer name. Missing names use the shared
                schedule; None keeps that optimizer at its initial rate.
            lr_decay_iterations: Learner iterations in the shared cosine learning-rate schedule.
            lr_final_ratio: Final learning rate relative to its initial value.
            gamma: Reward discount.
            gae_lambda: TD(lambda) mixing coefficient for value targets.
            critic_epochs: Epochs per critic fit.
            critic_minibatches: Minibatches per critic epoch.
            twin_critic: Optional second critic with its own optimizer and normalization, fitted to the same targets.
                Targets bootstrap from the smaller capped value of both critics.
            actor_value_reduction: Combination of the capped twin-critic terminal values in the actor objective,
                ``"mean"`` or ``"min"``.
            critic_zero_init: Zero the output layer of each critic so initial values are zero.
            warmup_steps: Stored steps per environment required before learning.
            world_model_warmup_iterations: Learner iterations fitting only dynamics.
            critic_warmup: During world-model warmup, also fit the critics on imagined rollouts of the fixed actor.
                The rollouts draw from the global RNG, so later replay batches differ from a run without warmup.
            action_noise: Standard deviation of additive collection noise.
            noise_faithful_imagination: Apply collection action noise during policy imagination.
            entropy_regularization: Learn the state-independent action noise, sample reparameterized imagined actions
                and add the per-step entropy bonus ``alpha * dt * H_norm`` to imagined rewards and critic targets.
                ``H_norm = (H + |H_target|) / (2 |H_target|)`` offsets and scales the Gaussian entropy ``H`` in nats
                as in SAPO, so it is zero at the target entropy. ``dt`` is the task transition duration. The
                log-temperature takes dual steps at the shared learning rate. Below the target entropy the bonus is
                negative and, like any negative reward, favors imagined termination.
            init_alpha: Initial entropy temperature, in the units of task reward weights.
            target_entropy_scale: Positive scale ``s`` of the target entropy ``H_target = -s * num_actions``, with
                target noise std ``exp(-s) / sqrt(2 pi e)`` per action. SAPO's ``-dim(A) / 2`` is ``s = 0.5`` (std
                0.147). ``None`` targets ``action_noise``, which must then be below 0.242, so the bonus starts at zero.
            max_step_reward: Optional per-step reward bound used to cap values at the maximum discounted return.
                A positive entropy bonus raises the bound by its value.
            use_mixed_precision: Enable BF16 matrix operations while integrating physical states in FP32.
            max_grad_norm: Global gradient-norm limit for each model.
            actor_grad_clip_value: Optional elementwise actor-gradient limit applied after the norm limit.
            actor_ema_halflife: Optional half-life in actor optimizer steps for passive checkpoint averaging.
            device: Device for models and learning tensors.
            multi_gpu_cfg: Unsupported distributed configuration; supplying one raises.
        """
        if actor_ema_halflife is not None and actor_ema_halflife < 1:
            raise ValueError("actor_ema_halflife must be positive or None.")
        if multi_gpu_cfg is not None:
            raise NotImplementedError("Multi-GPU training is not supported for SuperMimic.")
        if critic.is_recurrent or (twin_critic is not None and twin_critic.is_recurrent):
            raise ValueError("SuperMimic requires a feed-forward critic.")
        if actor_value_reduction not in ("mean", "min"):
            raise ValueError("actor_value_reduction must be 'mean' or 'min'.")
        positive = (
            num_world_model_updates,
            num_policy_updates,
            world_model_window,
            world_model_batch_size,
            horizon,
            policy_batch_size,
            lr_decay_iterations,
            critic_epochs,
            critic_minibatches,
            warmup_steps,
        )
        if min(positive) < 1 or updates_per_step <= 0 or learning_rate <= 0 or max_grad_norm <= 0:
            raise ValueError("Learning counts, cadence, learning rate and gradient limit must be positive.")
        if warmup_steps < world_model_window or world_model_warmup_iterations < 0:
            raise ValueError("warmup_steps must cover the model window; world-model warmup must be nonnegative.")
        if not 0 <= gamma < 1 or not 0 <= gae_lambda <= 1 or not 0 <= lr_final_ratio <= 1:
            raise ValueError("Invalid discount, lambda or final learning-rate ratio.")
        if action_noise < 0:
            raise ValueError("Action noise must be nonnegative.")
        if entropy_regularization:
            if action_noise <= 0:
                raise ValueError("Entropy regularization requires positive action noise.")
            if target_entropy_scale is None:
                target_entropy_scale = -0.5 * math.log(2 * math.pi * math.e) - math.log(action_noise)
        if init_alpha <= 0 or (target_entropy_scale is not None and target_entropy_scale <= 0):
            raise ValueError("Entropy temperature and target scale must be positive.")
        if actor_grad_clip_value is not None and actor_grad_clip_value <= 0:
            raise ValueError("actor_grad_clip_value must be positive or None.")
        self.actor, self.world_model, self.critic = actor.to(device), world_model.to(device), critic.to(device)
        self.twin_critic = None if twin_critic is None else twin_critic.to(device)
        self._raw_critic, self._raw_twin_critic = self.critic, self.twin_critic
        self.storage, self.task, self.obs_groups = storage, task, obs_groups
        self.device = device
        self.updates_per_step = updates_per_step
        self.num_world_model_updates, self.num_policy_updates = num_world_model_updates, num_policy_updates
        self.world_model_window, self.world_model_batch_size = world_model_window, world_model_batch_size
        self.horizon, self.policy_batch_size = horizon, policy_batch_size
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.critic_epochs, self.critic_minibatches = critic_epochs, critic_minibatches
        self.warmup_steps, self.world_model_warmup_iterations = warmup_steps, world_model_warmup_iterations
        self.value_cap = None if max_step_reward is None else max_step_reward / (1 - gamma)
        self.use_mixed_precision, self.max_grad_norm = use_mixed_precision, max_grad_norm
        self.noise_faithful_imagination = noise_faithful_imagination
        self.actor_grad_clip_value = actor_grad_clip_value
        self.actor_value_reduction, self.critic_warmup = actor_value_reduction, critic_warmup
        self.entropy_regularization = entropy_regularization
        self.actor.noise_std.fill_(action_noise)
        if critic_zero_init:
            with torch.no_grad():
                for model in self._critics().values():
                    model.mlp.output.weight.zero_()
                    model.mlp.output.bias.zero_()
        if entropy_regularization:
            self.actor.log_std = nn.Parameter(self.actor.noise_std.log())
        self.error_floor = task.representation.error_floor.to(device)
        if self.error_floor.shape != (task.representation.increment_dim,) or (self.error_floor <= 0).any():
            raise ValueError("State error_floor must contain one positive value per increment coordinate.")
        optimizer_kwargs = {"lr": learning_rate, "fused": torch.device(device).type == "cuda"}
        self.optimizers = {
            "world_model": torch.optim.Adam(self.world_model.parameters(), **optimizer_kwargs),
            "actor": torch.optim.Adam(self.actor.parameters(), betas=(0.9, 0.95), **optimizer_kwargs),
            "critic": torch.optim.Adam(self.critic.parameters(), **optimizer_kwargs),
        }
        if self.twin_critic is not None:
            self.optimizers["twin_critic"] = torch.optim.Adam(self.twin_critic.parameters(), **optimizer_kwargs)
        self.log_alpha: torch.Tensor | None = None
        self.target_entropy: float | None = None
        if entropy_regularization:
            self.target_entropy = -target_entropy_scale * self.actor.num_actions
            self.log_alpha = torch.tensor(math.log(init_alpha), device=device, requires_grad=True)
            self.optimizers["alpha"] = torch.optim.Adam([self.log_alpha], **optimizer_kwargs)
        overrides = dict(lr_decay_iterations_by_optimizer or {})
        if unknown := overrides.keys() - self.optimizers.keys():
            raise ValueError(f"Unknown optimizers in lr_decay_iterations_by_optimizer: {sorted(unknown)}")
        if any(value is not None and value < 1 for value in overrides.values()):
            raise ValueError("Per-optimizer decay lengths must be positive or None.")
        self.lr_decay_by_optimizer = {name: overrides.get(name, lr_decay_iterations) for name in self.optimizers}
        self.learning_rates = dict.fromkeys(self.optimizers, learning_rate)
        self.actor_ema_halflife = actor_ema_halflife
        self.actor_ema = None if actor_ema_halflife is None else [p.detach().clone() for p in self.actor.parameters()]
        self.actor_ema_updates = 0
        self.initial_learning_rate = learning_rate
        self.lr_decay_iterations, self.lr_final_ratio = lr_decay_iterations, lr_final_ratio
        self.learning_rate = learning_rate
        self.iteration, self._update_debt = 0, 0.0
        self._last_obs: TensorDict | None = None
        self._last_actions: torch.Tensor | None = None
        self._last_previous_actions: torch.Tensor | None = None
        self._window_break_sum = torch.zeros((), device=device)
        self._collected_steps = 0
        self._world_model_forward = self.world_model
        self._actor_forward = self.actor.forward_history

    def _set(self, obs: TensorDict, name: str) -> torch.Tensor:
        """Read an explicit observation set with no policy-group fallback."""
        return self.actor.observation_set(obs, name)

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        """Record the physical start and choose a noisy actor action; update actor moments now."""
        self.actor.sync_actions(obs)
        self._last_previous_actions = self.actor.previous_actions.clone()
        keys = {key for name in ("state", "goal", "context") for key in self.obs_groups.get(name, ())}
        self._last_obs = obs.select(*keys).clone()
        action = self.actor(obs)
        self._last_actions = action + torch.randn_like(action) * self.actor.output_std
        self.actor.previous_actions[:, -1].copy_(self._last_actions)
        return self._last_actions

    @torch.no_grad()
    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Record every-step pre-event successors and update dynamics statistics in place."""
        if self._last_obs is None or self._last_actions is None:
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
                "actions": self._last_actions,
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
            self.world_model.observe(
                start,
                observed["actions"],
                self.task.dynamics_context(start, observed["context"], observed["prev_actions"]),
                self.task.representation.unflatten(observed["next_state"]),
            )
        reset_action = self._set(obs, "prev_action") if "prev_action" in self.obs_groups else None
        reset_previous = self._set(obs, "prev_prev_action") if "prev_prev_action" in self.obs_groups else None
        self.actor.reset(done, reset_action, reset_previous)
        self._window_break_sum.add_(window_break.float().mean())
        self._collected_steps += 1
        self._update_debt += self.updates_per_step
        self._last_obs = self._last_actions = self._last_previous_actions = None

    def compute_returns(self, obs: TensorDict) -> None:
        """Replay and imagination construct their own return targets during update."""
        pass

    def _windows(self, batch_size: int) -> tuple[TensorDict, torch.Tensor]:
        """Sample uniform anchors, additionally cutting world-model targets after external state edits."""
        index = self.storage.sample(batch_size, length=self.world_model_window)
        keys = ["state", "next_state", "actions", "context", "prev_actions", "window_break", "valid"]
        if self.task.num_labels:
            keys.append("labels")
        window = self.storage.gather(keys, index.ids)
        mask = index.valid.float() * window["valid"]
        if self.world_model_window > 1:
            mask[:, 1:] *= torch.cumprod((~window["window_break"][:, :-1]).long(), dim=1)
        return window, mask

    def _starts(self, batch_size: int) -> tuple[TensorDict, torch.Tensor]:
        """Build reset-aware actor histories, repeating the earliest available frame when needed."""
        index = self.storage.sample(batch_size, history=self.actor.history_length - 1)
        history = self.storage.gather(("state", "goal", "context", "prev_actions"), index.ids)
        state = self.task.representation.unflatten(history["state"])
        frames = self.task.add_noise(
            self.task.actor_obs(state, history["goal"], history["context"], history["prev_actions"])
        )
        first = index.valid.long().argmax(dim=1)
        first_frame = frames[torch.arange(batch_size, device=frames.device), first]
        frames = torch.where(index.valid[..., None], frames, first_frame[:, None])
        return history[:, -1], frames

    def _predict(
        self, state: State, actions: torch.Tensor, context: torch.Tensor, previous_actions: torch.Tensor
    ) -> tuple[State, torch.Tensor | None, torch.Tensor | None]:
        """Apply learned dynamics, event probabilities and task-owned analytic state."""
        with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16, enabled=self.use_mixed_precision):
            raw_next = self._world_model_forward(
                state, actions, self.task.dynamics_context(state, context, previous_actions)
            )
            logits = (
                self.world_model.label_logits(raw_next, self.task.dynamics_context(raw_next, context, previous_actions))
                if self.task.num_labels
                else None
            )
        labels = torch.sigmoid(logits.float()) if logits is not None else None
        return self.task.post_dynamics(state, raw_next, labels), logits, labels

    def _fit_world_model(self) -> dict[str, torch.Tensor]:
        """Fit autoregressive physical successors and event labels on valid replay positions."""
        window, mask = self._windows(self.world_model_batch_size)
        state = self.task.representation.unflatten(window["state"][:, 0])
        predictions, logits = [], []
        for step in range(self.world_model_window):
            state, event_logits, _ = self._predict(
                state, window["actions"][:, step], window["context"][:, step], window["prev_actions"][:, step]
            )
            predictions.append(state)
            if event_logits is not None:
                logits.append(event_logits)
        prediction = tuple(torch.stack(values, dim=1) for values in zip(*predictions))
        target = self.task.representation.unflatten(window["next_state"])
        scale = torch.maximum(self.world_model.out_norm.std, self.error_floor)
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
        self._apply(loss, "world_model", self.world_model)
        metrics["world_model_loss"] = loss.detach() / self.world_model_window
        return metrics

    def _critics(self) -> dict[str, nn.Module]:
        """Return the active critics keyed by optimizer name."""
        critics = {"critic": self.critic}
        if self.twin_critic is not None:
            critics["twin_critic"] = self.twin_critic
        return critics

    def _value(self, features: torch.Tensor, critic: nn.Module | None = None) -> torch.Tensor:
        """Read scalar values of a critic, by default the first, through the synthetic critic observation set."""
        critic = self.critic if critic is None else critic
        return critic(TensorDict({"critic": features}, batch_size=features.shape[:-1])).squeeze(-1)

    def _cap(self, values: torch.Tensor, bonus: torch.Tensor | None = None) -> torch.Tensor:
        """Clamp values above ``value_cap`` when set; there is no lower bound.

        A positive per-step entropy ``bonus`` raises the cap to the maximum discounted soft return. The bonus is equal
        on every imagined step because the action noise is state independent; a negative bonus keeps the reward cap.
        """
        if self.value_cap is None:
            return values
        if bonus is None:
            return values.clamp(max=self.value_cap)
        return values.clamp(max=self.value_cap + bonus.detach().clamp(min=0) / (1 - self.gamma))

    def _entropy_bonus(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the per-step bonus ``alpha * dt * H_norm``, differentiable in the log-std, and the entropy ``H``."""
        entropy = (0.5 * math.log(2 * math.pi * math.e) + self.actor.log_std).sum()
        offset = abs(self.target_entropy)
        return self.log_alpha.detach().exp() * self.task.dt * (entropy + offset) / (2 * offset), entropy

    def _imagine(self) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Roll the actor through learned dynamics from replay starts.

        Returns rewards and termination probabilities ``[horizon, batch]``, the ``horizon + 1`` critic feature frames
        and the mean power of sampled action noise per step, if any.
        """
        with torch.no_grad():
            start, history = self._starts(self.policy_batch_size)
        state = self.task.representation.unflatten(start["state"])
        goal, context, previous = start["goal"], start["context"], start["prev_actions"]
        rewards, terminals, features = [], [], []
        action_noise_power = []
        for _ in range(self.horizon):
            features.append(self.task.critic_obs(state, goal, context, previous))
            with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16, enabled=self.use_mixed_precision):
                action = self._actor_forward(history, self.task.actor_goal(state, goal))
            if self.noise_faithful_imagination or self.entropy_regularization:
                noise = torch.randn_like(action) * self.actor.output_std
                action = action + noise
                action_noise_power.append(noise.detach().square().mean())
            successor, _, labels = self._predict(state, action, context, previous)
            rewards.append(self.task.reward(state, action, successor, goal, previous, labels=labels))
            terminals.append(self.task.termination_prob(successor, goal, labels=labels))
            goal, context = self.task.advance(successor, goal, context)
            previous = torch.cat((previous[:, 1:], action[:, None]), dim=1)
            frame = self.task.add_noise(self.task.actor_obs(successor, goal, context, previous))
            history = torch.cat((history[:, 1:], frame[:, None]), dim=1)
            state = successor
        features.append(self.task.critic_obs(state, goal, context, previous))
        return torch.stack(rewards), torch.stack(terminals), features, action_noise_power

    def _fit_policy(self) -> dict[str, torch.Tensor]:
        """Backpropagate imagined returns through frozen dynamics and fit TD(lambda) critics on the soft rewards.

        The temperature step uses the detached rollout entropy, so it precedes the actor step without affecting it.
        """
        bonus, metrics = None, {}
        with _frozen(self.world_model, *self._critics().values()):
            reward, terminal, features, action_noise_power = self._imagine()
            if self.entropy_regularization:
                bonus, entropy = self._entropy_bonus()
                metrics = {"entropy_bonus": bonus.detach(), **self._fit_temperature(entropy.detach())}
            soft_reward = reward if bonus is None else reward + bonus
            alive = torch.cumprod(1 - terminal, dim=0)
            weights = (
                torch.cat((torch.ones_like(alive[:1]), alive[:-1]))
                * self.gamma ** torch.arange(self.horizon, device=self.device)[:, None]
            )
            value = self._value(features[-1])
            bootstrap = self._cap(value, bonus)
            if self.twin_critic is not None:
                twin = self._cap(self._value(features[-1], self.twin_critic), bonus)
                bootstrap = (
                    (bootstrap + twin) / 2 if self.actor_value_reduction == "mean" else torch.minimum(bootstrap, twin)
                )
            objective = (weights * soft_reward).sum(dim=0) + self.gamma**self.horizon * alive[-1] * bootstrap
            policy_loss = -objective.mean()
            gradient = self._apply(policy_loss, "actor", self.actor, clip_value=self.actor_grad_clip_value)
        features_tensor = torch.stack(features).detach()
        metrics.update({
            "policy_loss": policy_loss.detach(),
            "policy_grad_norm": gradient,
            "policy_grad_max_after_norm_clip": self._actor_grad_max,
            "imagined_survival": alive[-1].mean().detach(),
            "imagined_reward": reward.detach().mean(),
        })
        if action_noise_power:
            metrics["imagined_action_noise_rms"] = torch.stack(action_noise_power).mean().sqrt()
        if self.value_cap is not None:
            metrics["terminal_saturation"] = (value.detach() > self._cap(value.detach(), bonus)).float().mean()
        metrics.update(self._fit_critic(features_tensor, soft_reward.detach(), terminal.detach(), bonus))
        return metrics

    def _fit_temperature(self, entropy: torch.Tensor) -> dict[str, torch.Tensor]:
        """Take a dual step on the log-temperature toward the target entropy, as in SAC and SAPO.

        The loss is linear in the log-temperature, so its gradient does not grow with the temperature.
        """
        alpha = self.log_alpha.detach().exp()
        loss = self.log_alpha * (entropy - self.target_entropy)
        self.optimizers["alpha"].zero_grad(set_to_none=True)
        loss.backward()
        self.optimizers["alpha"].step()
        return {"alpha": alpha, "entropy": entropy, "action_std": self.actor.log_std.detach().exp().mean()}

    def _warm_up_critic(self) -> dict[str, torch.Tensor]:
        """Fit the critics on imagined rollouts of the current actor without updating it."""
        with torch.no_grad():
            reward, terminal, features, _ = self._imagine()
            bonus = self._entropy_bonus()[0] if self.entropy_regularization else None
        return self._fit_critic(torch.stack(features), reward if bonus is None else reward + bonus, terminal, bonus)

    def _fit_critic(
        self, features: torch.Tensor, rewards: torch.Tensor, terminals: torch.Tensor, bonus: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Fit scalar values on detached imagined features with TD(lambda) targets.

        ``rewards`` already include any per-step entropy ``bonus``, which only raises the value cap here.
        """
        critics = self._critics()
        with torch.no_grad():
            observations = TensorDict(
                {"critic": features.flatten(0, 1)}, batch_size=[features.shape[0] * features.shape[1]]
            )
            for model in critics.values():
                model.update_normalization(observations)
            values = self._cap(self._value(features), bonus)
            if self.twin_critic is not None:
                values = torch.minimum(values, self._cap(self._value(features, self.twin_critic), bonus))
            targets = lambda_returns(rewards, terminals, values, self.gamma, self.gae_lambda)
            targets = self._cap(targets, bonus).flatten()
        features = features[:-1].flatten(0, 1)
        losses = {name: [] for name in critics}
        for _ in range(self.critic_epochs):
            for indices in torch.randperm(features.shape[0], device=self.device).chunk(self.critic_minibatches):
                for name, model in critics.items():
                    with torch.autocast(
                        torch.device(self.device).type, dtype=torch.bfloat16, enabled=self.use_mixed_precision
                    ):
                        loss = (self._value(features[indices], model) - targets[indices]).square().mean()
                    self._apply(loss, name, model)
                    losses[name].append(loss.detach())
        return {f"{name}_loss": torch.stack(values).mean() for name, values in losses.items()}

    def _apply(self, loss: torch.Tensor, name: str, module: nn.Module, clip_value: float | None = None) -> torch.Tensor:
        """Apply finite gradients with optional value clipping."""
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(module.parameters(), self.max_grad_norm)
        if name == "actor":
            self._actor_grad_max = (
                torch.cat([p.grad.detach().flatten() for p in module.parameters() if p.grad is not None]).abs().amax()
            )
        if clip_value is not None:
            nn.utils.clip_grad_value_(module.parameters(), clip_value)
        if torch.isfinite(norm):
            optimizer.step()
            if name == "actor" and self.actor_ema is not None:
                self._update_actor_ema()
        return norm.detach()

    @torch.no_grad()
    def _update_actor_ema(self) -> None:
        """Read the live actor after a successful step, without changing its trajectory."""
        self.actor_ema_updates += 1
        decay = min(0.5 ** (1 / self.actor_ema_halflife), (1 + self.actor_ema_updates) / (10 + self.actor_ema_updates))
        torch._foreach_lerp_(self.actor_ema, list(self.actor.parameters()), 1 - decay)

    def update(self) -> dict[str, float]:
        """Run accumulated learner updates; the runner's collection cadence remains external."""
        metrics = {"learner_updates": float(self.iteration)}
        if self._collected_steps:
            metrics["window_break_fraction"] = (self._window_break_sum / self._collected_steps).item()
            self._window_break_sum.zero_()
            self._collected_steps = 0
        if self.storage.num_steps < self.warmup_steps:
            self._update_debt = 0.0
            return metrics
        count = int(self._update_debt + 1e-9)
        self._update_debt -= count
        totals: dict[str, list[torch.Tensor]] = {}
        for _ in range(count):
            for _ in range(self.num_world_model_updates):
                for key, value in self._fit_world_model().items():
                    totals.setdefault(key, []).append(value)
            if self.iteration >= self.world_model_warmup_iterations or self.critic_warmup:
                fit = self._fit_policy if self.iteration >= self.world_model_warmup_iterations else self._warm_up_critic
                for _ in range(self.num_policy_updates):
                    for key, value in fit().items():
                        totals.setdefault(key, []).append(value)
            self.iteration += 1
            self._update_learning_rate()
        metrics.update({key: torch.stack(values).mean().item() for key, values in totals.items()})
        metrics["learner_updates"] = float(self.iteration)
        if len(set(self.learning_rates.values())) > 1:
            metrics.update({f"learning_rate_{name}": rate for name, rate in self.learning_rates.items()})
        return metrics

    def _update_learning_rate(self) -> None:
        """Set all optimizer rates for the next learner iteration, clamping at the cosine floor."""
        self.learning_rates = {
            name: cosine_learning_rate(
                self.initial_learning_rate, self.iteration, self.lr_decay_by_optimizer[name], self.lr_final_ratio
            )
            for name in self.optimizers
        }
        for name, optimizer in self.optimizers.items():
            for group in optimizer.param_groups:
                group["lr"] = self.learning_rates[name]
        self.learning_rate = self.learning_rates["actor"]

    def train_mode(self) -> None:
        """Enable training models; real collection moments update in actor/process calls."""
        self.actor.train()
        self.world_model.train()
        for critic in self._critics().values():
            critic.train()

    def eval_mode(self) -> None:
        """Select deterministic deployment inference without updating actor moments."""
        self.actor.eval()
        self.world_model.eval()
        for critic in self._critics().values():
            critic.eval()

    def get_policy(self) -> SuperMimicActor:
        """Return the real-observation actor with stateless deployable export methods."""
        return self.actor

    def default_load_cfg(self, loaded_dict: dict) -> dict:
        """Select an explicit full learner checkpoint, including the world model."""
        return {"actor": True, "critic": True, "world_model": True, "optimizer": True, "iteration": True}

    def save(self) -> dict:
        """Save all trainable state; replay and simulator collection state are deliberately excluded."""
        saved = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "world_model_state_dict": self.world_model.state_dict(),
            **{f"{key}_optimizer_state_dict": optimizer.state_dict() for key, optimizer in self.optimizers.items()},
            "iteration": self.iteration,
        }
        if self.actor_ema is not None:
            saved["actor_ema_state_dict"] = {
                **saved["actor_state_dict"],
                **dict(zip(dict(self.actor.named_parameters()), self.actor_ema)),
            }
            saved["actor_ema_updates"] = self.actor_ema_updates
        if self._raw_twin_critic is not None:
            saved["twin_critic_state_dict"] = self._raw_twin_critic.state_dict()
        if self.entropy_regularization:
            saved["log_alpha"] = self.log_alpha.detach().clone()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load an actor alone or a complete model selection; partial dynamics/critic restores are rejected.

        Actor-only loads transfer the mean policy across entropy settings; without a saved ``log_std`` the learned
        noise keeps its initial value.
        """
        load_cfg = self.default_load_cfg(loaded_dict) if load_cfg is None else load_cfg
        core = {name for name in ("actor", "critic", "world_model") if load_cfg.get(name)}
        if core not in ({"actor"}, {"actor", "critic", "world_model"}):
            raise ValueError("SuperMimic supports actor-only or full actor/critic/world_model loading.")
        actor_only = core == {"actor"}
        if actor_only and any(load_cfg.get(key) for key in ("optimizer", "iteration")):
            raise ValueError("Actor-only loading cannot restore optimizer or iteration state.")
        if not actor_only and "world_model_state_dict" not in loaded_dict:
            raise ValueError("A full SuperMimic checkpoint must explicitly contain world_model.")
        if not actor_only and ("twin_critic_state_dict" in loaded_dict) != (self._raw_twin_critic is not None):
            raise ValueError("A full SuperMimic checkpoint must match the configured number of critics.")
        if not actor_only and ("log_alpha" in loaded_dict) != self.entropy_regularization:
            raise ValueError("A full SuperMimic checkpoint must match the configured entropy regularization.")
        actor_state = loaded_dict["actor_state_dict"]
        if actor_only and ("log_std" in actor_state) != (self.actor.log_std is not None):
            actor_state = {key: value for key, value in actor_state.items() if key != "log_std"}
            if self.actor.log_std is not None:
                actor_state["log_std"] = self.actor.log_std
        self.actor.load_state_dict(actor_state, strict=strict)
        if self.actor_ema is not None:
            ema_state = loaded_dict.get("actor_ema_state_dict", self.actor.state_dict())
            with torch.no_grad():
                for (name, _), average in zip(self.actor.named_parameters(), self.actor_ema):
                    average.copy_(ema_state[name])
            self.actor_ema_updates = (
                loaded_dict.get("actor_ema_updates", 0) if "actor_ema_state_dict" in loaded_dict else 0
            )
        if not actor_only:
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
            if self._raw_twin_critic is not None:
                self._raw_twin_critic.load_state_dict(loaded_dict["twin_critic_state_dict"], strict=strict)
            self.world_model.load_state_dict(loaded_dict["world_model_state_dict"], strict=strict)
            if self.entropy_regularization:
                with torch.no_grad():
                    self.log_alpha.copy_(loaded_dict["log_alpha"])
            if load_cfg.get("optimizer"):
                for key in self.optimizers:
                    self.optimizers[key].load_state_dict(loaded_dict[f"{key}_optimizer_state_dict"])
            if load_cfg.get("iteration"):
                self.iteration = loaded_dict["iteration"]
        self.storage.clear()
        self.actor.reset()
        self._last_obs = self._last_actions = self._last_previous_actions = None
        self._update_debt = 0.0
        self._window_break_sum.zero_()
        self._collected_steps = 0
        self._update_learning_rate()
        return bool(load_cfg.get("iteration"))

    def compile(self, mode: str | None = None) -> None:
        """Compile differentiable dynamics and stateless actor forwards using the standard mode restrictions."""
        self._world_model_forward = compile_model(self.world_model, mode)
        self.critic = compile_model(self._raw_critic, mode)
        if self._raw_twin_critic is not None:
            self.twin_critic = compile_model(self._raw_twin_critic, mode)
        self._actor_forward = (
            self.actor.forward_history if mode is None else torch.compile(self.actor.forward_history, mode=mode)
        )

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> SuperMimic:
        """Resolve the task and explicit network configs, inferring synthetic feature widths once."""
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        task_class, task_cfg = resolve_class(alg_cfg.pop("task_cfg"))
        num_critics = alg_cfg.pop("num_critics", 1)
        if num_critics not in (1, 2):
            raise ValueError("SuperMimic supports one critic or twin critics.")
        if "critic" in cfg["obs_groups"]:
            raise ValueError("SuperMimic critic observations are synthetic; omit 'critic' from obs_groups.")
        allowed = ("state", "goal", "context", "labels", "prev_action", "prev_prev_action")
        groups = resolve_obs_groups(obs, cfg["obs_groups"], ["state"], fallback_to_policy=False)
        if set(groups) - set(allowed):
            raise ValueError("Unknown SuperMimic observation set.")
        cfg["obs_groups"] = groups
        task = task_class(env=env, device=device, **task_cfg)
        example = obs.to(device)

        def observation_set(name: str) -> torch.Tensor:
            return (
                torch.cat([example[key] for key in groups[name]], dim=-1)
                if name in groups
                else example[groups["state"][0]].new_empty(env.num_envs, 0)
            )

        for name in groups:
            if any(example[key].ndim != 2 for key in groups[name]):
                raise ValueError("SuperMimic observation groups must be flat noise-free vectors.")
        previous = torch.zeros(env.num_envs, task.action_history_length, env.num_actions, device=device)
        for name, offset in (("prev_action", -1), ("prev_prev_action", -2)):
            if name in groups:
                if observation_set(name).shape[-1] != env.num_actions or task.action_history_length < -offset:
                    raise ValueError("Previous-action observation width/history is incompatible.")
                previous[:, offset] = observation_set(name)
        state, goal, context = (
            task.representation.unflatten(observation_set("state")),
            observation_set("goal"),
            observation_set("context"),
        )
        actor_obs, actor_goal = task.actor_obs(state, goal, context, previous), task.actor_goal(state, goal)
        critic_features = task.critic_obs(state, goal, context, previous)
        dynamics_context = task.dynamics_context(state, context, previous)
        if observation_set("labels").shape[-1] != task.num_labels:
            raise ValueError("Label observation width must equal task.num_labels.")
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        model_class, model_cfg = resolve_class(cfg["world_model"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])
        actor = actor_class(
            task, groups, actor_obs.shape[-1], actor_goal.shape[-1], env.num_envs, env.num_actions, **actor_cfg
        ).to(device)
        world_model = model_class(
            task.representation,
            task.representation.features(state).shape[-1],
            env.num_actions,
            dynamics_context.shape[-1],
            num_labels=task.num_labels,
            **model_cfg,
        ).to(device)
        critic_obs = TensorDict({"critic": critic_features}, batch_size=[env.num_envs])
        critics = [
            critic_class(critic_obs, {"critic": ["critic"]}, "critic", 1, **critic_cfg).to(device)
            for _ in range(num_critics)
        ]
        warmup = alg_cfg.get("warmup_steps", 64)
        storage = SequenceReplayBuffer(
            env.num_envs,
            alg_cfg.pop("replay_buffer_size", 1_000_000),
            device,
            min_steps=max(warmup, actor.history_length),
            max_memory_fraction=alg_cfg.pop("max_replay_memory_fraction", None),
        )
        learner = alg_class(
            actor,
            world_model,
            critics[0],
            storage,
            task,
            groups,
            twin_critic=critics[1] if num_critics == 2 else None,
            device=device,
            multi_gpu_cfg=cfg["multi_gpu"],
            **alg_cfg,
        )
        learner.compile(cfg.get("torch_compile_mode"))
        return learner


@contextmanager
def _frozen(*modules: nn.Module) -> Generator[None, None, None]:
    """Freeze parameters while preserving input derivatives and restoring flags after failures."""
    previous = [(parameter, parameter.requires_grad) for module in modules for parameter in module.parameters()]
    try:
        for parameter, _ in previous:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in previous:
            parameter.requires_grad_(enabled)
