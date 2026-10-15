# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import functools
import math
import torch
import torch.nn as nn
import warnings
from collections.abc import Callable
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel, QModel
from rsl_rl.modules import EmpiricalNormalization, SquashedGaussianDistribution
from rsl_rl.storage import ReplayBuffer
from rsl_rl.utils import resolve_class, resolve_obs_groups, resolve_optimizer


class SAC:
    """Soft Actor-Critic algorithm for massively parallel simulation.

    The implementation covers the design choices of the common SAC variants for GPU simulators as configuration
    options, including:

    - Scalar (mean-squared error) or categorical (cross-entropy) critics with min or mean aggregation over the ensemble.
    - n-step returns with exact per-sample discounts and bootstrapping from final observations on time-outs.
    - Plain, layer-normalized, or batch-normalized residual networks (see :class:`~rsl_rl.modules.ResidualMLP`).
      Batch-normalized critics evaluate current and next state-action pairs in a joint batch.
    - Reward scaling by the running standard deviation of the discounted return.
    - Gaussian exploration or exploration noise held for a random number of steps.
    - A configurable action bound with an optional linear schedule.

    References:
        - Haarnoja et al. "Soft Actor-Critic Algorithms and Applications." arXiv preprint arXiv:1812.05905 (2018).
        - Bellemare et al. "A Distributional Perspective on Reinforcement Learning." ICML (2017).
        - Bhatt et al. "CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and
          Simplicity." ICLR (2024).
        - Kim et al. "FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control."
          arXiv preprint arXiv:2604.04539 (2026).
    """

    actor: MLPModel
    """The actor model."""

    critics: nn.ModuleList
    """The ensemble of critic models."""

    def __init__(
        self,
        actor: MLPModel,
        critics: list[QModel],
        storage: ReplayBuffer,
        num_updates: int = 1,
        batch_size: int = 4096,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_update_interval: int = 1,
        learning_starts: int = 0,
        # Optimization
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        learning_rate_schedule: str = "constant",
        final_learning_rate_ratio: float = 0.5,
        schedule_updates: int | None = None,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        max_grad_norm: float | None = None,
        # Entropy
        init_alpha: float = 0.01,
        target_entropy: float | None = None,
        target_entropy_std: float = 0.15,
        # Critic
        critic_head: str = "scalar",
        num_bins: int = 101,
        value_range: tuple[float, float] = (-5.0, 5.0),
        q_aggregation: str = "min",
        # Rewards
        reward_normalization: bool = False,
        reward_normalization_max_return: float = 5.0,
        # Exploration
        exploration: str = "gaussian",
        noise_repeat_exponent: float = 2.0,
        noise_repeat_max: int = 16,
        # Action bound
        action_bound: float = 1.0,
        final_action_bound: float | None = None,
        action_bound_hold_steps: int = 0,
        action_bound_ramp_steps: int = 0,
        # Others
        use_mixed_precision: bool = False,
        use_cuda_graph: bool = False,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        r"""Initialize the algorithm.

        Args:
            actor: Actor model with a :class:`~rsl_rl.modules.SquashedGaussianDistribution`.
            critics: Critic models. Their output dimension is 1 for scalar critics and ``num_bins`` for categorical
                critics.
            storage: Replay buffer.
            num_updates: Number of gradient steps of the critics per call of :meth:`update`.
            batch_size: Number of transitions per gradient step.
            gamma: Discount factor.
            tau: Polyak averaging coefficient of the target critics.
            actor_update_interval: Number of critic updates per actor and temperature update.
            learning_starts: Number of transitions collected with uniformly random actions before training starts.
            actor_learning_rate: Learning rate of the actor.
            critic_learning_rate: Learning rate of the critics.
            alpha_learning_rate: Learning rate of the entropy temperature.
            learning_rate_schedule: ``"constant"`` or ``"cosine"`` decay to ``final_learning_rate_ratio`` times the
                initial learning rates over ``schedule_updates`` critic updates.
            final_learning_rate_ratio: Final learning rates relative to the initial ones for the cosine schedule.
            schedule_updates: Number of critic updates of the cosine schedule.
            optimizer: Name of the optimizer (see :func:`~rsl_rl.utils.resolve_optimizer`).
            weight_decay: Weight decay of the actor and critic optimizers.
            max_grad_norm: Maximum gradient norm of the actor and critic updates. ``None`` disables clipping.
            init_alpha: Initial entropy temperature.
            target_entropy: Target entropy. If ``None``, it is computed from ``target_entropy_std``.
            target_entropy_std: Standard deviation (in action units) of a Gaussian whose entropy is the target entropy,
                i.e., the target entropy is :math:`\frac{|A|}{2} \log(2 \pi e \sigma^2)`.
            critic_head: ``"scalar"`` (mean-squared error) or ``"categorical"`` (cross-entropy on a fixed support).
            num_bins: Number of bins of the categorical critics.
            value_range: Value range of the categorical critics.
            q_aggregation: ``"min"`` uses the critic with the lowest value for the target and the minimum for the actor.
                ``"mean"`` trains each critic on its own target and uses the mean for the actor.
            reward_normalization: Whether to scale rewards by the running standard deviation of the discounted return.
            reward_normalization_max_return: Lower bound of the reward scale such that the largest observed discounted
                return is mapped to at most this value.
            exploration: ``"gaussian"`` samples new noise every step, ``"held"`` holds the noise of each environment
                for a number of steps drawn from a truncated zeta distribution.
            noise_repeat_exponent: Exponent of the truncated zeta distribution of the noise repetitions.
            noise_repeat_max: Maximum number of noise repetitions.
            action_bound: Bound of the actions, i.e., actions are ``action_bound * tanh(u)``.
            final_action_bound: Final action bound of a linear schedule. ``None`` keeps ``action_bound`` constant.
            action_bound_hold_steps: Number of environment steps before the action bound schedule starts.
            action_bound_ramp_steps: Number of environment steps over which the action bound increases linearly.
            use_mixed_precision: Whether to use bfloat16 autocast for forward passes and losses.
            use_cuda_graph: Whether to capture the gradient step in a CUDA graph. This removes the kernel launch
                overhead, which dominates for small networks. Requires a CUDA device.
            device: Device to run the algorithm on.
            multi_gpu_cfg: Multi-GPU configuration. Multi-GPU training is not supported yet.
        """
        if multi_gpu_cfg is not None:
            raise NotImplementedError("Multi-GPU training is not supported for SAC yet.")
        if not isinstance(actor.distribution, SquashedGaussianDistribution):
            raise ValueError("The SAC actor requires a SquashedGaussianDistribution.")
        if critic_head not in ("scalar", "categorical"):
            raise ValueError(f"Invalid critic head '{critic_head}'. Valid values are: 'scalar', 'categorical'.")
        if q_aggregation not in ("min", "mean"):
            raise ValueError(f"Invalid Q aggregation '{q_aggregation}'. Valid values are: 'min', 'mean'.")
        if exploration not in ("gaussian", "held"):
            raise ValueError(f"Invalid exploration '{exploration}'. Valid values are: 'gaussian', 'held'.")
        if learning_rate_schedule not in ("constant", "cosine"):
            raise ValueError(f"Invalid schedule '{learning_rate_schedule}'. Valid values are: 'constant', 'cosine'.")
        if learning_rate_schedule == "cosine" and schedule_updates is None:
            raise ValueError("The cosine learning rate schedule requires 'schedule_updates'.")

        self.device = device
        self.is_multi_gpu = False

        # Models
        self.actor = actor.to(device)
        self.critics = nn.ModuleList(critics).to(device)
        self.target_critics = copy.deepcopy(self.critics)
        self.target_critics.requires_grad_(False)
        self.log_alpha = torch.tensor(math.log(init_alpha), device=device, requires_grad=True)

        # Handles to the uncompiled modules for state_dict operations and export
        self._raw_actor = self.actor
        self._raw_critics = self.critics
        self._raw_target_critics = self.target_critics

        # Batch normalization layers need different modes for the different forward passes
        self._actor_batch_norms = _batch_norm_layers(self.actor)
        self._critic_batch_norms = _batch_norm_layers(self.critics)
        self._target_batch_norms = _batch_norm_layers(self.target_critics)

        # Optimizers
        optimizer_class = resolve_optimizer(optimizer)
        self.actor_optimizer = optimizer_class(
            self.actor.parameters(), lr=actor_learning_rate, weight_decay=weight_decay
        )
        self.critic_optimizer = optimizer_class(
            self.critics.parameters(), lr=critic_learning_rate, weight_decay=weight_decay
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self._initial_learning_rates = (actor_learning_rate, critic_learning_rate, alpha_learning_rate)
        self.learning_rate = critic_learning_rate

        # CUDA graphs need optimizers whose state (including the learning rate) lives on the device
        self.use_cuda_graph = use_cuda_graph
        if use_cuda_graph:
            if torch.device(device).type != "cuda":
                raise ValueError("CUDA graphs require a CUDA device.")
            for optimizer in (self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer):
                for param_group in optimizer.param_groups:
                    param_group["capturable"] = True
                    param_group["lr"] = torch.tensor(param_group["lr"], device=device)
            self._graph_steps = {
                do_actor: _CudaGraphStep(functools.partial(self._gradient_step, do_actor=do_actor))
                for do_actor in (True, False)
            }

        # Storage
        self.storage = storage

        # Parameters
        self.num_updates = num_updates
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.actor_update_interval = actor_update_interval
        self.learning_starts = learning_starts
        self.learning_rate_schedule = learning_rate_schedule
        self.final_learning_rate_ratio = final_learning_rate_ratio
        self.schedule_updates = schedule_updates
        self.max_grad_norm = max_grad_norm
        self.critic_head = critic_head
        self.q_aggregation = q_aggregation
        self.exploration = exploration
        self.use_mixed_precision = use_mixed_precision

        # Entropy target
        num_actions = self.actor.distribution.output_dim
        if target_entropy is None:
            target_entropy = 0.5 * num_actions * math.log(2.0 * math.pi * math.e * target_entropy_std**2)
        self.target_entropy = target_entropy

        # Categorical critic support
        self.num_bins = num_bins
        self.value_range = value_range
        self._bin_values = torch.linspace(value_range[0], value_range[1], num_bins, device=device)

        # Reward scaling
        self.reward_normalization = reward_normalization
        self.reward_normalization_max_return = reward_normalization_max_return
        if reward_normalization:
            self.return_normalizer = EmpiricalNormalization(1, eps=0.0).to(device)
            self._returns = torch.zeros(storage.num_envs, device=device)
            self._max_abs_return = torch.zeros((), device=device)

        # Exploration noise repetition
        if exploration == "held":
            repeats = torch.arange(1, noise_repeat_max + 1, dtype=torch.float, device=device)
            self._noise_repeat_cdf = torch.cumsum(repeats**-noise_repeat_exponent, dim=0)
            self._noise_repeat_cdf /= self._noise_repeat_cdf[-1].clone()
            self._noise = torch.zeros(storage.num_envs, num_actions, device=device)
            self._noise_steps_left = torch.zeros(storage.num_envs, dtype=torch.long, device=device)

        # Action bound schedule (critics see actions normalized by the largest bound)
        self.action_bound = action_bound
        self.final_action_bound = action_bound if final_action_bound is None else final_action_bound
        self.action_bound_hold_steps = action_bound_hold_steps
        self.action_bound_ramp_steps = action_bound_ramp_steps
        self._critic_action_scale = 1.0 / max(action_bound, self.final_action_bound)

        # Counters
        self.num_env_steps = 0
        self.num_critic_updates = 0
        self._set_action_bound()

        # Transition data recorded between act() and process_env_step()
        self._observations: TensorDict | None = None
        self._actions: torch.Tensor | None = None
        self._warned_missing_final_obs = False

        # Fraction of exploration actions close to the action bound (summed over steps between updates)
        self._saturation_sum = torch.zeros((), device=device)
        self._saturation_steps = 0

    """
    Interaction
    """

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample exploration actions and record the observations."""
        if self.storage.size < self.learning_starts:
            bound = self.actor.distribution.action_bound
            actions = 2.0 * torch.rand(obs.batch_size[0], self.actor.distribution.output_dim, device=self.device) - 1.0
            actions = actions * bound
        else:
            with _batch_norm_mode(self._actor_batch_norms, training=False):
                self._update_actor_distribution(obs)
            distribution: SquashedGaussianDistribution = self.actor.distribution  # type: ignore
            if self.exploration == "held":
                resample = self._noise_steps_left <= 0
                self._noise = torch.where(resample.unsqueeze(1), torch.randn_like(self._noise), self._noise)
                repeats = torch.searchsorted(self._noise_repeat_cdf, torch.rand_like(self._noise_steps_left.float()))
                self._noise_steps_left = torch.where(resample, repeats + 1, self._noise_steps_left) - 1
                actions = distribution.sample_with_noise(self._noise)
            else:
                actions = distribution.sample()
        self._observations = obs
        self._actions = actions
        bound = self.actor.distribution.action_bound
        self._saturation_sum += (actions.abs() > 0.95 * bound).float().mean()
        self._saturation_steps += 1
        return actions

    @torch.no_grad()
    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Store the transition of the last step in the replay buffer."""
        dones = dones.view(-1).bool()
        time_outs = extras.get("time_outs")
        time_outs = torch.zeros_like(dones) if time_outs is None else time_outs.view(-1).to(self.device).bool()
        terminated = dones & ~time_outs

        # Bootstrap time-outs from the final observations of the episodes
        next_obs = obs
        if time_outs.any():
            final_obs = extras.get("final_obs")
            if final_obs is not None:
                next_obs = obs.clone()
                for key in self.storage.observations:
                    final = final_obs[key].to(self.device)
                    next_obs[key] = torch.where(time_outs.unsqueeze(-1), final, obs[key])
            elif not self._warned_missing_final_obs:
                warnings.warn(
                    "The environment reports time-outs but no final observations in extras['final_obs']. Time-outs "
                    "bootstrap from the first observation of the next episode, which biases the value targets.",
                    stacklevel=2,
                )
                self._warned_missing_final_obs = True

        self.storage.add(self._observations, self._actions, rewards, next_obs, dones, terminated)  # type: ignore

        # Update the normalization statistics
        self.actor.update_normalization(self._observations)  # type: ignore
        for critic in self.critics:
            critic.update_normalization(self._observations)  # type: ignore
        if self.reward_normalization:
            self._returns = self.gamma * self._returns + rewards.view(-1)
            self.return_normalizer.update(self._returns.unsqueeze(1))
            self._max_abs_return = torch.maximum(self._max_abs_return, self._returns.abs().max())
            self._returns = self._returns * ~dones

        # Resample the held exploration noise at the start of new episodes
        if self.exploration == "held":
            self._noise_steps_left = self._noise_steps_left * ~dones

        self.num_env_steps += 1
        self._set_action_bound()

    """
    Learning
    """

    def update(self) -> dict[str, float]:
        """Run the gradient steps and return the mean losses."""
        if self.storage.size < max(self.learning_starts, 1):
            return {}

        totals: dict[str, torch.Tensor] = {}
        num_actor_updates = 0
        for _ in range(self.num_updates):
            self._update_learning_rates()
            batch = self.storage.sample(self.batch_size)
            if self.reward_normalization:
                batch.rewards = batch.rewards / self._reward_scale()

            do_actor = self.num_critic_updates % self.actor_update_interval == 0
            inputs = (
                batch.observations,
                batch.actions,
                batch.rewards,
                batch.next_observations,
                batch.bootstrap_discounts,
            )
            if self.use_cuda_graph:
                stats = self._graph_steps[do_actor](*inputs)
            else:
                stats = self._gradient_step(*inputs, do_actor=do_actor)
            num_actor_updates += int(do_actor)
            self.num_critic_updates += 1

            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value  # type: ignore

        # Average over the number of updates (actor statistics only over actor updates)
        loss_dict = {}
        for key, value in totals.items():
            count = num_actor_updates if key in ("actor", "alpha", "entropy") else self.num_updates
            loss_dict[key] = (value / max(count, 1)).item()
        loss_dict["alpha_value"] = self.log_alpha.exp().item()
        if self._saturation_steps > 0:
            loss_dict["action_saturation"] = (self._saturation_sum / self._saturation_steps).item()
            self._saturation_sum.zero_()
            self._saturation_steps = 0
        return loss_dict

    def _gradient_step(
        self,
        observations: TensorDict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: TensorDict,
        bootstrap_discounts: torch.Tensor,
        do_actor: bool,
    ) -> TensorDict:
        """Run one gradient step of the critics, optionally the actor and temperature, and the target update."""
        batch = ReplayBuffer.Batch(observations, actions, rewards, next_observations, bootstrap_discounts)
        stats = self._update_critics(batch)
        if do_actor:
            stats.update(self._update_actor_and_alpha(batch))
        self._update_target_critics()
        return stats

    def _update_critics(self, batch: ReplayBuffer.Batch) -> dict[str, torch.Tensor]:
        """Run one gradient step of the critics."""
        alpha = self.log_alpha.detach().exp()
        actions = batch.actions * self._critic_action_scale
        joint_batch = len(self._critic_batch_norms) > 0

        with torch.autocast(
            device_type=torch.device(self.device).type, dtype=torch.bfloat16, enabled=self.use_mixed_precision
        ):
            # Compute the targets
            with torch.no_grad():
                with _batch_norm_mode(self._actor_batch_norms, training=False):
                    next_actions, next_log_probs = self._sample_actions(batch.next_observations)
                next_actions = next_actions * self._critic_action_scale
                if joint_batch:
                    # Evaluate current and next state-action pairs jointly for consistent batch statistics
                    target_obs = torch.cat([batch.observations, batch.next_observations])
                    target_actions = torch.cat([actions, next_actions])
                    with _batch_norm_mode(self._target_batch_norms, training=True):
                        next_outputs = [c(target_obs, target_actions)[self.batch_size :] for c in self.target_critics]
                else:
                    next_outputs = [c(batch.next_observations, next_actions) for c in self.target_critics]
                targets = self._compute_targets(next_outputs, batch, alpha * next_log_probs)

            # Compute the predictions
            if joint_batch:
                with _batch_norm_mode(self._critic_batch_norms, training=True):
                    outputs = [c(target_obs, target_actions)[: self.batch_size] for c in self.critics]
            else:
                outputs = [c(batch.observations, actions) for c in self.critics]

            if self.critic_head == "scalar":
                losses = [(output.squeeze(-1) - target).square().mean() for output, target in zip(outputs, targets)]
            else:
                losses = [
                    -(target * torch.log_softmax(output.float(), dim=-1)).sum(dim=-1).mean()
                    for output, target in zip(outputs, targets)
                ]
            critic_loss = torch.stack(losses).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critics.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        _normalize_parameters(self.critics)

        with torch.no_grad():
            q_values = torch.stack([self._expected_value(output) for output in outputs])
        return {"critic": critic_loss.detach(), "q_value": q_values.mean()}

    def _update_actor_and_alpha(self, batch: ReplayBuffer.Batch) -> dict[str, torch.Tensor]:
        """Run one gradient step of the actor and the entropy temperature."""
        alpha = self.log_alpha.detach().exp()
        joint_batch = len(self._actor_batch_norms) > 0

        with torch.autocast(
            device_type=torch.device(self.device).type, dtype=torch.bfloat16, enabled=self.use_mixed_precision
        ):
            if joint_batch:
                # Evaluate current and next observations jointly for consistent batch statistics
                with _batch_norm_mode(self._actor_batch_norms, training=True):
                    actions, log_probs = self._sample_actions(torch.cat([batch.observations, batch.next_observations]))
                actions, log_probs = actions[: self.batch_size], log_probs[: self.batch_size]
            else:
                actions, log_probs = self._sample_actions(batch.observations)

            self.critics.requires_grad_(False)
            with _batch_norm_mode(self._critic_batch_norms, training=False):
                q_values = torch.stack([
                    self._expected_value(c(batch.observations, actions * self._critic_action_scale))
                    for c in self.critics
                ])
            self.critics.requires_grad_(True)
            q_value = q_values.min(dim=0).values if self.q_aggregation == "min" else q_values.mean(dim=0)
            actor_loss = (alpha * log_probs - q_value).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        _normalize_parameters(self.actor)

        # Temperature update: increase alpha if the entropy is below the target and decrease it otherwise
        entropy = -log_probs.detach().float().mean()
        alpha_loss = self.log_alpha.exp() * (entropy - self.target_entropy)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return {"actor": actor_loss.detach(), "alpha": alpha_loss.detach(), "entropy": entropy}

    def _compute_targets(
        self, next_outputs: list[torch.Tensor], batch: ReplayBuffer.Batch, entropy_bonus: torch.Tensor
    ) -> list[torch.Tensor]:
        """Compute the critic targets (values for scalar critics, probabilities for categorical critics)."""
        rewards, discounts = batch.rewards, batch.bootstrap_discounts
        next_values = [self._expected_value(output) for output in next_outputs]
        if self.q_aggregation == "min":
            # Use the critic with the lowest value (per sample) for a shared target
            min_ids = torch.stack(next_values).argmin(dim=0)
            selected = torch.stack(next_outputs).gather(
                0, min_ids.view(1, -1, 1).expand(1, -1, next_outputs[0].shape[-1])
            )
            next_outputs = [selected[0]] * len(self.critics)
        if self.critic_head == "scalar":
            return [rewards + discounts * (output.squeeze(-1) - entropy_bonus) for output in next_outputs]
        return [
            self._project(torch.softmax(output.float(), dim=-1), rewards, discounts, entropy_bonus)
            for output in next_outputs
        ]

    def _project(
        self, probs: torch.Tensor, rewards: torch.Tensor, discounts: torch.Tensor, entropy_bonus: torch.Tensor
    ) -> torch.Tensor:
        """Project the shifted and scaled categorical distribution onto the fixed support."""
        v_min, v_max = self.value_range
        delta = (v_max - v_min) / (self.num_bins - 1)
        shifted = rewards.unsqueeze(1) + discounts.unsqueeze(1) * (self._bin_values - entropy_bonus.unsqueeze(1))
        positions = (shifted.clamp(v_min, v_max) - v_min) / delta
        lower = positions.floor().long().clamp(0, self.num_bins - 1)
        upper = (lower + 1).clamp(max=self.num_bins - 1)
        upper_weight = positions - lower.float()
        target = torch.zeros_like(probs)
        target.scatter_add_(1, lower, probs * (1.0 - upper_weight))
        target.scatter_add_(1, upper, probs * upper_weight)
        return target

    def _expected_value(self, output: torch.Tensor) -> torch.Tensor:
        """Return the Q-value of a critic output."""
        if self.critic_head == "scalar":
            return output.squeeze(-1).float()
        return (torch.softmax(output.float(), dim=-1) * self._bin_values).sum(dim=-1)

    @torch.no_grad()
    def _update_target_critics(self) -> None:
        """Polyak-average the target critic parameters and copy the buffers (e.g., normalization statistics)."""
        target_params = list(self.target_critics.parameters())
        torch._foreach_lerp_(target_params, list(self.critics.parameters()), self.tau)
        for target_buffer, buffer in zip(self.target_critics.buffers(), self.critics.buffers()):
            target_buffer.copy_(buffer)

    def _sample_actions(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions with the reparameterization trick and return them with their log probabilities."""
        self._update_actor_distribution(obs)
        return self.actor.distribution.rsample()  # type: ignore

    def _update_actor_distribution(self, obs: TensorDict) -> None:
        """Update the actor distribution for the given observations."""
        self.actor.distribution.update(self.actor.mlp(self.actor.get_latent(obs)))  # type: ignore

    def _reward_scale(self) -> torch.Tensor:
        """Return the reward scale from the running standard deviation and maximum of the discounted return."""
        return torch.maximum(
            self.return_normalizer._std.squeeze() + 1e-8,  # type: ignore
            self._max_abs_return / self.reward_normalization_max_return,
        )

    def _update_learning_rates(self) -> None:
        """Set the learning rates according to the schedule."""
        if self.learning_rate_schedule == "constant":
            return
        progress = min(self.num_critic_updates / self.schedule_updates, 1.0)  # type: ignore
        factor = self.final_learning_rate_ratio + (1.0 - self.final_learning_rate_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        optimizers = (self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer)
        for optimizer, initial in zip(optimizers, self._initial_learning_rates):
            for param_group in optimizer.param_groups:
                if isinstance(param_group["lr"], torch.Tensor):
                    param_group["lr"].fill_(initial * factor)
                else:
                    param_group["lr"] = initial * factor
        self.learning_rate = self._initial_learning_rates[1] * factor

    def _set_action_bound(self) -> None:
        """Set the action bound of the actor according to the schedule."""
        steps = self.num_env_steps - self.action_bound_hold_steps
        progress = min(max(steps / max(self.action_bound_ramp_steps, 1), 0.0), 1.0) if steps >= 0 else 0.0
        bound = self.action_bound + progress * (self.final_action_bound - self.action_bound)
        self.actor.distribution.action_bound.fill_(bound)  # type: ignore

    """
    Helpers for the runner
    """

    @property
    def action_std(self) -> torch.Tensor:
        """Return the standard deviation of the latest exploration distribution (before squashing)."""
        std = self.actor.distribution._std  # type: ignore
        return std if std is not None else torch.zeros(1)

    def train_mode(self) -> None:
        """Set train mode for learnable models (batch normalization modes are managed by the algorithm)."""
        self.actor.train()
        self.critics.train()
        self.target_critics.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critics.eval()
        self.target_critics.eval()

    def save(self) -> dict:
        """Return a dict of all models and training states for saving."""
        saved_dict = {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critics.state_dict(),
            "target_critic_state_dict": self._raw_target_critics.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "num_env_steps": self.num_env_steps,
            "num_critic_updates": self.num_critic_updates,
        }
        if self.reward_normalization:
            saved_dict["return_normalizer_state_dict"] = self.return_normalizer.state_dict()
            saved_dict["max_abs_return"] = self._max_abs_return
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models and training states from a saved dict."""
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}

        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critics.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
            self._raw_target_critics.load_state_dict(loaded_dict["target_critic_state_dict"], strict=strict)
            with torch.no_grad():
                self.log_alpha.copy_(loaded_dict["log_alpha"])
            if self.reward_normalization and "return_normalizer_state_dict" in loaded_dict:
                self.return_normalizer.load_state_dict(loaded_dict["return_normalizer_state_dict"])
                self._max_abs_return = loaded_dict["max_abs_return"].to(self.device)
        if load_cfg.get("optimizer"):
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            self.alpha_optimizer.load_state_dict(loaded_dict["alpha_optimizer_state_dict"])
            self.num_env_steps = loaded_dict["num_env_steps"]
            self.num_critic_updates = loaded_dict["num_critic_updates"]
            self._set_action_bound()
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile the models with ``torch.compile`` (not supported yet).

        Args:
            mode: ``torch.compile`` mode. Must be ``None``.
        """
        if mode is not None:
            raise NotImplementedError("torch.compile is not supported for SAC yet.")

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> SAC:
        """Construct the SAC algorithm."""
        # Resolve class callables and configs
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])

        # Resolve observation groups
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])

        # Initialize the actor
        actor_cfg.setdefault("distribution_cfg", {"class_name": "SquashedGaussianDistribution"})
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        print(f"Actor Model: {actor}")

        # Initialize the critics
        num_critics = alg_cfg.pop("num_critics", 2)
        critic_output_dim = alg_cfg.get("num_bins", 101) if alg_cfg.get("critic_head") == "categorical" else 1
        critics = [
            critic_class(obs, cfg["obs_groups"], "critic", critic_output_dim, env.num_actions, **critic_cfg).to(device)
            for _ in range(num_critics)
        ]
        print(f"Critic Model (x{num_critics}): {critics[0]}")

        # Initialize the replay buffer with the observation groups used by the models
        groups = list(dict.fromkeys(cfg["obs_groups"]["actor"] + cfg["obs_groups"]["critic"]))
        storage = ReplayBuffer(
            env.num_envs,
            alg_cfg.pop("replay_buffer_size", 1_000_000),
            obs.select(*groups),
            env.num_actions,
            n_steps=alg_cfg.pop("n_steps", 1),
            gamma=alg_cfg.get("gamma", 0.99),
            device=device,
        )

        # Initialize the algorithm
        alg: SAC = alg_class(actor, critics, storage, device=device, **alg_cfg, multi_gpu_cfg=cfg["multi_gpu"])

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg


class _CudaGraphStep:
    """Run a function in a CUDA graph after a few eager warm-up calls.

    The inputs of each call are copied into static buffers before the graph is replayed. The function must only use
    tensor operations that are valid during graph capture and return a dictionary of tensors.
    """

    def __init__(self, fn: Callable[..., dict[str, torch.Tensor]], num_warmup: int = 3) -> None:
        self.fn = fn
        self.num_warmup = num_warmup
        self.num_calls = 0
        self.graph: torch.cuda.CUDAGraph | None = None
        self._stream = torch.cuda.Stream()

    def __call__(self, *args: torch.Tensor | TensorDict) -> dict[str, torch.Tensor]:
        if self.graph is None and self.num_calls < self.num_warmup:
            # Warm up on a side stream as required before capturing
            self.num_calls += 1
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                outputs = self.fn(*args)
            torch.cuda.current_stream().wait_stream(self._stream)
            return outputs
        if self.graph is None:
            # Capturing records the kernels without running them, so the replay below performs this call's step
            self._static_args = tuple(arg.clone() for arg in args)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._static_outputs = self.fn(*self._static_args)
        else:
            for static, arg in zip(self._static_args, args):
                static.copy_(arg)
        self.graph.replay()
        return {key: value.clone() for key, value in self._static_outputs.items()}


def _batch_norm_layers(module: nn.Module) -> list[nn.Module]:
    """Return the batch normalization layers of a module."""
    return [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


class _batch_norm_mode:  # noqa: N801
    """Context manager that temporarily sets the mode of batch normalization layers."""

    def __init__(self, layers: list[nn.Module], training: bool) -> None:
        self.layers = layers
        self.training = training

    def __enter__(self) -> None:
        self.previous = [layer.training for layer in self.layers]
        for layer in self.layers:
            layer.training = self.training

    def __exit__(self, *args: object) -> None:
        for layer, training in zip(self.layers, self.previous):
            layer.training = training


def _normalize_parameters(module: nn.Module) -> None:
    """Apply weight normalization of networks that define it after an optimizer step."""
    for m in module.modules():
        if m is not module and hasattr(m, "normalize_parameters"):
            m.normalize_parameters()
