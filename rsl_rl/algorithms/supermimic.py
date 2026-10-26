# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SuperMimic training loop (Algorithm A.1): collect, fit the world model, improve the policy through it."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

from rsl_rl.env.model_based_env import DifferentiableTask, ModelBasedEnv, State
from rsl_rl.models.supermimic_model import Critic, Mlp, Policy, WorldModel
from rsl_rl.storage.sequence_replay_buffer import SequenceReplayBuffer


@dataclass
class SuperMimicCfg:
    """Hyperparameters of Table A.5 (one GPU)."""

    num_envs: int = 4096
    buffer_steps: int = 1000
    action_noise: float = 0.1
    updates_per_step: float = 0.4
    world_model_window: int = 8
    world_model_batches: int = 10
    world_model_batch_size: int = 2048
    horizon: int = 32
    policy_batch_size: int = 4096
    learning_rate: float = 2e-4
    lr_decay_iterations: int = 250_000
    lr_final_ratio: float = 0.03
    gamma: float = 0.99
    gae_lambda: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    # Not in the paper: initial random-policy steps, and world-model-only iterations before policy updates.
    warmup_steps: int = 64
    world_model_warmup_iterations: int = 500
    compile: bool = True
    # Not in the paper: bootstrap with min(V, U), U = max step reward / (1 - gamma) of the task's reward.
    cap_value: bool = False
    # Not in the paper (TF32): run the large MLP matmuls in bf16; kinematics and integration stay fp32.
    bf16: bool = False
    # Optional differentiable task-owned validation/auxiliary head.
    auxiliary_loss_weight: float = 0.0
    world_model_hidden: tuple[int, ...] = (2048, 2048, 2048)
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden: tuple[int, ...] = (256, 128)
    max_replay_memory_fraction: float | None = 0.5

    def validate(self, history: int) -> None:
        """Reject invalid horizons and schedules before allocating replay."""
        if history < 1 or self.world_model_warmup_iterations < 0:
            raise ValueError("history must be positive and world_model_warmup_iterations nonnegative")
        if not 0 <= self.lr_final_ratio <= 1:
            raise ValueError("lr_final_ratio must lie in [0, 1]")
        for name in (
            "num_envs",
            "buffer_steps",
            "world_model_window",
            "world_model_batches",
            "world_model_batch_size",
            "horizon",
            "policy_batch_size",
            "warmup_steps",
            "critic_epochs",
            "critic_minibatches",
            "lr_decay_iterations",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.buffer_steps < max(history, self.world_model_window):
            raise ValueError("buffer_steps must cover observation history and world_model_window")
        if self.warmup_steps < self.world_model_window:
            raise ValueError("warmup_steps must cover world_model_window")
        if not 0 < self.updates_per_step <= 1:
            raise ValueError("updates_per_step must lie in (0, 1]")
        if not 0 <= self.gamma < 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma must lie in [0, 1), gae_lambda in [0, 1]")
        if self.action_noise < 0 or self.auxiliary_loss_weight < 0 or self.learning_rate <= 0:
            raise ValueError("noise/auxiliary weight must be nonnegative and learning_rate positive")


def lambda_returns(
    rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
) -> torch.Tensor:
    """Compute lambda-return targets; values includes the final bootstrap state."""
    target, targets = values[-1], []
    for t in reversed(range(len(rewards))):
        target = rewards[t] + gamma * (1 - dones[t]) * ((1 - lam) * values[t + 1] + lam * target)
        targets.append(target)
    return torch.stack(targets[::-1])


BUFFER_KEYS = (
    "state",
    "next_state",
    "action",
    "prev_action",
    "prev_prev_action",
    "xi",
    "goal_state",
    "next_goal_state",
    "episode_step",
    "done",
)


class SuperMimic:
    """Learn physical dynamics and optimize a deterministic actor through imagined rollouts."""

    def __init__(self, cfg: SuperMimicCfg, mdp: DifferentiableTask, device: str | torch.device) -> None:
        """Build task-independent models, optimizers and replay sampling state."""
        self.cfg, self.task, self.device = cfg, mdp, torch.device(device)
        self.history = mdp.history_length
        num_actions = mdp.num_actions
        self.representation = mdp.representation
        cfg.validate(self.history)
        # world-model context: the normalized sim context xi, plus (optional) the exogenous disturbance features
        self.dist = getattr(mdp, "disturbances", None)
        # + task extras (full DR): the previous action with an actuator delay, terrain heights with rough terrain
        self.extra_dim = getattr(mdp, "wm_extra_dim", 0)
        self.context_dim = mdp.xi_dim + (self.dist.feature_dim if self.dist is not None else 0) + self.extra_dim
        self.world_model = WorldModel(
            self.representation,
            num_actions,
            self.context_dim,
            hidden=cfg.world_model_hidden,
            num_labels=len(mdp.label_names),
        ).to(device)
        self.policy = Policy(
            mdp.obs_dim,
            mdp.goal_dim,
            self.history,
            num_actions,
            encoder_hidden=mdp.encoder_hidden or cfg.encoder_hidden,
            decoder_hidden=cfg.decoder_hidden,
            action_joints=getattr(mdp, "action_joints", None),
        ).to(device)
        # privileged critic: the task's critic features (incl. xi) plus the disturbance schedule state when enabled
        # plus the terrain scan under the base when terrain is on (critic_extra)
        self.critic_extra_dim = getattr(mdp, "critic_extra_dim", 0)
        self.critic_dim = mdp.critic_dim + (self.dist.state_dim if self.dist is not None else 0) + self.critic_extra_dim
        self.critic = Critic(self.critic_dim, hidden=cfg.critic_hidden).to(device)
        self.value_cap = mdp.max_step_reward / (1 - cfg.gamma) if cfg.cap_value else None
        for module in (*self.world_model.modules(), *self.policy.modules()):
            if isinstance(module, Mlp):
                module.bf16 = cfg.bf16
        self.optimizers = {
            "world_model": torch.optim.Adam(
                self.world_model.parameters(), lr=cfg.learning_rate, fused=self.device.type == "cuda"
            ),
            "policy": torch.optim.Adam(
                self.policy.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95), fused=self.device.type == "cuda"
            ),
            "critic": torch.optim.Adam(
                self.critic.parameters(), lr=cfg.learning_rate, fused=self.device.type == "cuda"
            ),
        }
        self.error_floor = self.representation.error_floor.to(device)
        self.buffer: SequenceReplayBuffer | None = None
        self.obs_history = torch.zeros(cfg.num_envs, self.history, mdp.obs_dim, device=device)
        self.iteration, self._step_debt = 0, 0.0
        maybe_compile = torch.compile if cfg.compile else (lambda fn: fn)
        hot = (self._rollout_impl, self._imagine_impl, self._labels_impl, self._critic_input_impl)
        self._rollout, self._imagine, self._labels, self._critic_input = map(maybe_compile, hot)

    def _ctx(
        self, state: State, xi_n: torch.Tensor, d: torch.Tensor | None = None, a_prev: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Build privileged dynamics inputs for a real or imagined state.

        Context includes normalized physical parameters, optional disturbance features,
        and task extras such as action history or differentiable terrain samples.
        """
        parts = [xi_n]
        if self.dist is not None:
            parts.append(self.dist.features(d, state))
        if self.extra_dim:
            parts.append(self.task.wm_extra(state, a_prev))
        return parts[0] if len(parts) == 1 else torch.cat(parts, -1)

    # ------------------------------------------------------------------ collect
    @torch.no_grad()
    def collect(self, sim: ModelBasedEnv, steps: int) -> list[dict[str, torch.Tensor]]:
        """Roll out the policy with action noise and store the transitions."""
        if sim.num_envs != self.cfg.num_envs:
            raise ValueError("The environment count must match SuperMimicCfg.num_envs")
        mdp, outs = self.task, []
        for _ in range(steps):
            state = sim.state()
            o = mdp.add_noise(mdp.obs(state, sim.prev_action, sim.xi, sim.goal_state))
            shifted = torch.cat((self.obs_history[:, 1:], o.unsqueeze(1)), 1)
            fresh = (sim.episode_step == 0)[:, None, None]
            self.obs_history = torch.where(fresh, o.unsqueeze(1).expand_as(shifted), shifted)
            goal = mdp.policy_goal(state, sim.goal_state)
            self.policy.obs_norm.update(o)
            self.policy.goal_norm.update(goal)
            action = self.policy(self.obs_history, goal)
            out = sim.step(action + self.cfg.action_noise * torch.randn_like(action))
            if self.buffer is None:
                keys = (*BUFFER_KEYS, *(("labels",) if mdp.label_names else ()))
                keys = (*keys, *(("disturbance",) if self.dist is not None else ()))
                self.buffer = SequenceReplayBuffer(
                    self.cfg.buffer_steps,
                    sim.num_envs,
                    {k: out[k] for k in keys},
                    minimum_steps=max(self.history, self.cfg.world_model_window),
                    max_memory_fraction=self.cfg.max_replay_memory_fraction,
                )
            self.buffer.add(out)
            start_state = self.representation.unflatten(
                out["state"]
            )  # pre-impulse (what the actor saw); the transition starts post-impulse
            if self.dist is not None:
                start_state = self.dist.apply_impulse(start_state, out["disturbance"])
            self.world_model.observe(
                start_state,
                out["action"],
                self._ctx(start_state, mdp.normalize_xi(out["xi"]), out.get("disturbance"), out["prev_action"]),
                self.representation.unflatten(out["next_state"]),
            )
            outs.append(out)
        return outs

    # ------------------------------------------------------------------ world model
    def _rollout_impl(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        xi_n: torch.Tensor,
        d: torch.Tensor | None = None,
        prev_actions: torch.Tensor | None = None,
    ) -> State:
        """Roll out logged actions autoregressively, retaining physical context per transition."""
        state, preds = self.representation.unflatten(state[:, 0]), []
        for j in range(actions.shape[1]):
            if self.dist is not None:  # logged states are pre-impulse: the step's impulse edit, as in the simulator
                state = self.dist.apply_impulse(state, d[:, j])
            a_prev = None if prev_actions is None else prev_actions[:, j]
            state = self.world_model(
                state, actions[:, j], self._ctx(state, xi_n[:, j], None if d is None else d[:, j], a_prev)
            )
            preds.append(state)
        return tuple(torch.stack(x, 1) for x in zip(*preds))

    def update_world_model(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit k-step autoregressive replay windows (Eq. 1); returns the loss and the k-step error."""
        cfg = self.cfg
        keys = ("state", "next_state", "action", "xi", *(("labels",) if self.task.label_names else ()))
        keys = (
            *keys,
            *(("disturbance",) if self.dist is not None else ()),
            *(("prev_action",) if self.extra_dim else ()),
        )
        w = self.buffer.windows(cfg.world_model_batch_size, cfg.world_model_window, keys)
        xi_n = self.task.normalize_xi(w["xi"])
        d, pa = w.get("disturbance"), w.get("prev_action")
        prediction = self._rollout(w["state"], w["action"], xi_n, d, pa)
        scale = torch.maximum(self.world_model.out_norm.std, self.error_floor)
        error = (
            (self.representation.error(prediction, self.representation.unflatten(w["next_state"])) / scale) ** 2
        ).mean(-1)
        loss = (error * w["mask"]).sum(1).mean()
        if self.task.label_names:
            # learned event heads on predicted next states, trained on the simulator's event labels
            logits = self.world_model.label_logits(prediction, self._ctx(prediction, xi_n, d, pa))
            bce = nn.functional.binary_cross_entropy_with_logits(logits, w["labels"].float(), reduction="none")
            self.label_loss = (bce.mean(-1) * w["mask"]).sum() / w["mask"].sum().clamp(min=1)
            loss = loss + self.label_loss
        auxiliary, self.model_metrics = self.task.model_loss(
            prediction, self.representation.unflatten(w["next_state"]), w["mask"]
        )
        loss = loss + cfg.auxiliary_loss_weight * auxiliary
        self._apply(loss, "world_model", self.world_model)
        last = w["mask"][:, -1]
        return loss.detach() / cfg.world_model_window, (error[:, -1].detach() * last).sum() / last.sum().clamp(min=1)

    # ------------------------------------------------------------------ policy and critic
    def _imagine_impl(
        self,
        state: State,
        obs_history: torch.Tensor,
        prev_prev_action: torch.Tensor,
        prev_action: torch.Tensor,
        g: torch.Tensor,
        xi: torch.Tensor,
        xi_n: torch.Tensor,
        d: torch.Tensor | None = None,
    ) -> tuple:
        """Imagine H steps acting on noisy observation histories.

        Returns all states and goal states (H + 1 blocks) and actions with their two predecessors
        (H + 2 blocks); the task's goal state advances with the task's own (possibly exogenous) schedule.
        """
        mdp = self.task
        states, actions, goals = [state], [prev_prev_action, prev_action], [g]
        a_prev = prev_action
        pre, dists = [], []  # disturbances: post-impulse pre-step states and the step's disturbance state
        for _ in range(self.cfg.horizon):
            # goal features may depend on the (imagined) state: the root task's errors are differentiable in S
            action = self.policy(obs_history, mdp.policy_goal(state, g))
            if self.dist is not None:
                # the actor acted before the (unanticipated) impulse; the same state edit as the simulator's write
                state = self.dist.apply_impulse(state, d)
                pre.append(state), dists.append(d)
                ctx = self._ctx(state, xi_n, d, a_prev)
                state, g, d = self.world_model(state, action, ctx), mdp.advance(g), self.dist.advance(d)
            else:
                state, g = self.world_model(state, action, self._ctx(state, xi_n, None, a_prev)), mdp.advance(g)
            a_prev = action
            o = mdp.add_noise(mdp.obs(state, action, xi, g))
            obs_history = torch.cat((obs_history[:, 1:], o.unsqueeze(1)), 1)
            states.append(state), actions.append(action), goals.append(g)
        out = tuple(torch.cat(x) for x in zip(*states)), torch.cat(actions), torch.cat(goals)
        if self.dist is not None:
            # disturbance state of every imagined state (H + 1 blocks: the last one is the bootstrap state's)
            out = (*out, tuple(torch.cat(x) for x in zip(*pre)), torch.cat((*dists, d)))
        return out

    def _labels_impl(
        self,
        states: State,
        goals: torch.Tensor,
        actions: torch.Tensor,
        xi_n: torch.Tensor,
        pre: State | None = None,
        dists: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rewards and termination probabilities of every imagined transition, batched over the horizon.

        With disturbances, ``pre`` are the post-impulse pre-step states (the simulator's transition start) and
        ``dists`` the disturbance state of every step.
        """
        batch_size = len(xi_n)
        prev_states, next_states = tuple(x[:-batch_size] for x in states), tuple(x[batch_size:] for x in states)
        if pre is not None:
            prev_states = pre
        horizon = len(next_states[0]) // batch_size
        labels = None
        if self.task.label_names:
            ctx = self._ctx(next_states, xi_n.repeat(horizon, 1), dists, actions[batch_size:-batch_size])
            labels = torch.sigmoid(self.world_model.label_logits(next_states, ctx))
        reward = self.task.reward(
            next_states,
            goals[batch_size:],
            actions[2 * batch_size :],
            actions[batch_size:-batch_size],
            a_prev2=actions[: -2 * batch_size],
            labels=labels,
            S_prev=prev_states,
            g_prev=goals[:-batch_size],
        )
        return reward, self.task.termination_prob(next_states, goals[batch_size:], labels=labels)

    def _critic_input_impl(
        self,
        state: State,
        g: torch.Tensor,
        xi_n: torch.Tensor,
        a_prev: torch.Tensor,
        a_prev2: torch.Tensor,
        d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.task.critic_features(state, g, xi_n, a_prev=a_prev, a_prev2=a_prev2)
        if self.dist is not None:
            x = torch.cat((x, self.dist.critic_features(d)), -1)
        return x if not self.critic_extra_dim else torch.cat((x, self.task.critic_extra(state)), -1)

    def bootstrap(self, value: torch.Tensor) -> torch.Tensor:
        """Critic value used as a bootstrap; capped above at U when ``cap_value`` (no lower bound)."""
        return value if self.value_cap is None else value.clamp(max=self.value_cap)

    def update_policy(self) -> dict[str, torch.Tensor]:
        """BPTT through H imagined steps with a critic bootstrap (Eq. 2), then fit the critic (Eq. 3)."""
        cfg, mdp = self.cfg, self.task
        keys = ("state", "prev_action", "prev_prev_action", "xi", "goal_state")
        keys = (*keys, *(("disturbance",) if self.dist is not None else ()))
        start, hist = self.buffer.starts(2 * cfg.policy_batch_size, self.history, keys)
        valid = mdp.valid_start(start["goal_state"], cfg.horizon).nonzero().squeeze(-1)
        idx = valid[: cfg.policy_batch_size]
        if not len(idx):
            return {"policy_skipped_no_valid_starts": torch.ones((), device=self.device)}
        start, hist = ({k: v[idx] for k, v in d.items()} for d in (start, hist))
        state, g, xi = self.representation.unflatten(start["state"]), start["goal_state"], start["xi"]
        xi_n = mdp.normalize_xi(xi)
        with torch.no_grad():
            obs_history = mdp.add_noise(
                mdp.obs(
                    self.representation.unflatten(hist["state"]), hist["prev_action"], hist["xi"], hist["goal_state"]
                )
            )

        with _frozen(self.world_model, self.critic):
            if self.dist is not None:
                # logged states are pre-impulse (the actor's view); imagination applies each step's impulse edit once
                d0 = start["disturbance"]
                imagined = self._imagine(
                    state, obs_history, start["prev_prev_action"], start["prev_action"], g, xi, xi_n, d0
                )
                all_states, all_actions, all_goals, pre, d_all = imagined
            else:
                all_states, all_actions, all_goals = self._imagine(
                    state, obs_history, start["prev_prev_action"], start["prev_action"], g, xi, xi_n
                )
                pre = d_all = None
            # Rewards and terminations do not feed back into the rollout, so they are batched over all steps.
            horizon, batch_size = cfg.horizon, len(xi)
            dists = None if d_all is None else d_all[:-batch_size]  # the disturbance state of each transition
            rewards, dones = (
                x.view(horizon, batch_size) for x in self._labels(all_states, all_goals, all_actions, xi_n, pre, dists)
            )
            alive = torch.cumprod(1 - dones, 0)  # c_{t+1}
            weight = (
                torch.cat((torch.ones_like(alive[:1]), alive[:-1]))
                * cfg.gamma ** torch.arange(horizon, device=self.device)[:, None]
            )
            objective = (weight * rewards).sum(0)
            stats = {}
            if cfg.use_critic:
                final_state, final_goal = tuple(x[-batch_size:] for x in all_states), all_goals[-batch_size:]
                final_disturbance = None if d_all is None else d_all[-batch_size:]
                raw = self.critic(
                    self._critic_input(
                        final_state,
                        final_goal,
                        xi_n,
                        all_actions[-batch_size:],
                        all_actions[-2 * batch_size : -batch_size],
                        final_disturbance,
                    )
                )
                objective = objective + cfg.gamma**horizon * alive[-1] * self.bootstrap(raw)
                if self.value_cap is not None:
                    stats["terminal_saturation"] = (raw.detach() > self.value_cap).float().mean()
            loss = -objective.mean()
            grad_norm = self._apply(loss, "policy", self.policy, clip_value=0.1)

        stats |= {
            "policy_loss": loss.detach(),
            "policy_grad_norm": grad_norm,
            "imagined_survival": alive[-1].mean().detach(),
            "imagined_reward": rewards.detach().mean() / mdp.max_step_reward,
        }
        if cfg.use_critic:
            with torch.no_grad():
                states = tuple(x.detach() for x in all_states)
                features = self._critic_input(
                    states,
                    all_goals,
                    xi_n.repeat(horizon + 1, 1),
                    all_actions[batch_size:],
                    all_actions[:-batch_size],
                    d_all,
                ).view(horizon + 1, batch_size, -1)
            stats |= self._update_critic(features, rewards.detach(), dones.detach())
        return stats

    def _update_critic(
        self, features: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        with torch.no_grad():
            self.critic.norm.update(features)
            raw = self.critic(features)
            targets = lambda_returns(rewards, dones, self.bootstrap(raw), cfg.gamma, cfg.gae_lambda)
            if self.value_cap is not None:
                targets = targets.clamp(max=self.value_cap)
            value_stats = {"value_raw_max": raw.max(), "value_raw_mean": raw.mean(), "target_max": targets.max()}
            if self.value_cap is not None:
                value_stats["value_saturation"] = (raw > self.value_cap).float().mean()
        features, targets = features[:-1].flatten(0, 1), targets.flatten()
        for _ in range(cfg.critic_epochs):
            for idx in torch.randperm(len(features), device=self.device).chunk(cfg.critic_minibatches):
                # the raw critic is fitted, so values above the cap are pulled back down
                loss = ((self.critic(features[idx]) - targets[idx]) ** 2).mean()
                self._apply(loss, "critic", self.critic)
        return {"critic_loss": loss.detach(), **value_stats}

    # ------------------------------------------------------------------ schedule
    def _apply(self, loss: torch.Tensor, name: str, module: nn.Module, clip_value: float | None = None) -> torch.Tensor:
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        if clip_value is not None:
            nn.utils.clip_grad_value_(module.parameters(), clip_value)
        if torch.isfinite(grad_norm):
            optimizer.step()
        return grad_norm.detach()

    def learning_rate(self) -> float:
        """Cosine decay to ``lr_final_ratio`` over ``lr_decay_iterations``, then constant."""
        cfg = self.cfg
        progress = min(self.iteration / cfg.lr_decay_iterations, 1.0)
        return cfg.learning_rate * (
            cfg.lr_final_ratio + (1 - cfg.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        )

    def iterate(self, sim: ModelBasedEnv) -> dict[str, torch.Tensor]:
        """One iteration of Algorithm A.1."""
        cfg = self.cfg
        for optimizer in self.optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = self.learning_rate()
        self._step_debt += 1 / cfg.updates_per_step
        steps, self._step_debt = int(self._step_debt), self._step_debt % 1
        outs = self.collect(sim, cfg.warmup_steps if self.buffer is None else steps)
        for _ in range(cfg.world_model_batches):
            wm_loss, wm_error = self.update_world_model()
        stats = {"world_model_loss": wm_loss, "world_model_error": wm_error}
        if self.task.label_names:
            stats["label_loss"] = self.label_loss.detach()
        stats |= {k: v.detach() for k, v in self.model_metrics.items()}
        if self.iteration >= cfg.world_model_warmup_iterations:
            stats |= self.update_policy()
        stats["sim_reward"] = torch.stack([o["reward"].mean() for o in outs]).mean() / self.task.max_step_reward
        for key in {k for o in outs for k in o.get("metrics", {})}:
            stats[f"Metrics/{key}"] = torch.stack([
                o["metrics"][key] for o in outs if key in o.get("metrics", {})
            ]).mean()
        finished = [o for o in outs if "episode_length" in o]
        if finished:
            stats["episode_length"] = torch.cat([o["episode_length"] for o in finished]).mean()
            stats["episode_return"] = torch.cat([o["episode_return"] for o in finished]).mean()
            for key in finished[0].get("episode_terms", {}):
                stats[f"Episode_Reward/{key}"] = torch.cat([o["episode_terms"][key] for o in finished]).mean()
        self.iteration += 1
        return stats

    # ------------------------------------------------------------------ checkpoints
    def state_dict(self) -> dict:
        """Save trainable state and task semantics; replay/simulator state are deliberately omitted."""
        return {
            "world_model": self.world_model.state_dict(),
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
            "iteration": self.iteration,
            "task": self.task.name,
            "xi_dim": self.task.xi_dim,
            "goal_dim": self.task.goal_dim,
            "label_names": list(self.task.label_names),
            "context_dim": self.context_dim,
            "critic_dim": self.critic_dim,
            "task_metadata": self.task.checkpoint_metadata(),
            "task_training_state": self.task.training_state_dict(),
            "step_debt": self._step_debt,
        }

    def load_state_dict(self, state: dict, policy_only: bool = False) -> None:
        """Restore a compatible checkpoint; policy-only loads permit changed physical context.

        Full resume starts with empty replay and a fresh simulator. It is a training
        restart from learned parameters, not a bitwise continuation of collection.
        """
        for key, mine in (("task", self.task.name), ("goal_dim", self.task.goal_dim)):
            if state.get(key) != mine:
                raise ValueError(f"Checkpoint {key}={state.get(key)!r} does not match {mine!r}")
        if policy_only:
            self.loaded_policy_context = {k: state.get(k) for k in ("xi_dim", "task_metadata")}
            self.policy.load_state_dict(state["policy"])
            return
        for key, mine in (
            ("xi_dim", self.task.xi_dim),
            ("context_dim", self.context_dim),
            ("critic_dim", self.critic_dim),
            ("label_names", list(self.task.label_names)),
            ("task_metadata", self.task.checkpoint_metadata()),
        ):
            if state.get(key) != mine:
                raise ValueError(f"Checkpoint {key} does not match this task/context")
        self.task.load_training_state_dict(state.get("task_training_state", {}))
        self.policy.load_state_dict(state["policy"])
        self.world_model.load_state_dict(state["world_model"])
        self.critic.load_state_dict(state["critic"])
        for k, v in state["optimizers"].items():
            self.optimizers[k].load_state_dict(v)
        self.iteration = state["iteration"]
        self._step_debt = state.get("step_debt", 0.0)
        self.buffer = None
        self.obs_history.zero_()


@contextmanager
def _frozen(*modules: nn.Module) -> Generator[None, None, None]:
    """Freeze weights while preserving input derivatives and restore flags even on failure."""
    parameters = [(p, p.requires_grad) for module in modules for p in module.parameters()]
    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, required in parameters:
            parameter.requires_grad_(required)
