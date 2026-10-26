# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-device SuperMimic training using an environment's differentiable task."""

from __future__ import annotations

import os
import time
import torch
from dataclasses import asdict
from pathlib import Path

from rsl_rl.algorithms.supermimic import SuperMimic, SuperMimicCfg
from rsl_rl.env.model_based_env import DifferentiableTask, ModelBasedEnv
from rsl_rl.utils.logger import Logger


def _resolve_device(device: str | torch.device) -> torch.device:
    """Resolve an implicit CUDA index once so comparison and allocations agree."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


class SuperMimicRunner:
    """Native runner for physical transitions and differentiable task heads.

    ``env.task`` supplies a :class:`DifferentiableTask`. The simulator remains owned
    by the caller. This runner intentionally does not inherit the PPO runner: it
    collects physical state sequences and optimizes imagined returns, not PPO
    observation rollouts. Only single-device training is currently supported.
    """

    def __init__(
        self,
        env: ModelBasedEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        task: DifferentiableTask | None = None,
    ) -> None:
        """Build the learner and the standard RSL-RL logging stack."""
        if int(os.getenv("WORLD_SIZE", "1")) != 1:
            raise ValueError("SuperMimicRunner currently supports one training process")
        device = str(_resolve_device(device))
        if torch.device(device) != _resolve_device(env.device):
            raise ValueError("SuperMimic learning and physical transitions must share a device")
        self.env, self.device = env, device
        self.task = task if task is not None else env.task
        cfg = dict(train_cfg.get("algorithm", {}))
        cfg.pop("class_name", None)
        cfg["num_envs"] = env.num_envs
        self.alg = SuperMimic(SuperMimicCfg(**cfg), self.task, device)
        self.cfg = {**train_cfg, "algorithm": asdict(self.alg.cfg)}
        self.log_dir = log_dir
        self.logger = Logger(log_dir, self.cfg, getattr(env, "cfg", {}), env.num_envs, False, 1, 0, device)
        self.last_metrics: dict[str, torch.Tensor] = {}

    @property
    def current_learning_iteration(self) -> int:
        """Number of completed learner updates."""
        return self.alg.iteration

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Collect real states and update the model, actor and critic each iteration."""
        if init_at_random_ep_len:
            raise ValueError("Episode starts/history are owned by the model-based environment")
        if num_learning_iterations < 1:
            raise ValueError("num_learning_iterations must be positive")
        interval = self.cfg.get("save_interval", 2000)
        if interval < 1:
            raise ValueError("save_interval must be positive")
        self.alg.policy.train()
        self.alg.world_model.train()
        self.alg.critic.train()
        self.logger.init_logging_writer()
        try:
            for _ in range(num_learning_iterations):
                started = time.perf_counter()
                self.task.begin_iteration()
                metrics = self.alg.iterate(self.env)
                self.last_metrics = metrics
                writer = self.logger.writer
                if writer is not None:
                    iteration = self.current_learning_iteration
                    for name, value in metrics.items():
                        writer.add_scalar(name, value.item(), iteration)
                    writer.add_scalar("Perf/iteration_seconds", time.perf_counter() - started, iteration)
                    writer.add_scalar("Train/learning_rate", self.alg.learning_rate(), iteration)
                    if iteration % interval == 0:
                        self.save(str(Path(self.log_dir) / f"model_{iteration}.pt"))
            if self.log_dir is not None:
                self.save(str(Path(self.log_dir) / f"model_{self.current_learning_iteration}.pt"))
        finally:
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        """Atomically save models, optimizer state and task semantics (excluding replay)."""
        state = self.alg.state_dict()
        state["infos"], state["config"] = infos, self.cfg
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, target)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, policy_only: bool = False, *, load_optimizer: bool = True) -> dict | None:
        """Load a trusted checkpoint, optionally retaining fresh optimizer state.

        Full loads restore every model, task training state and iteration even when
        ``load_optimizer=False``. Collection restarts with empty replay.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.load_state_dict(state, policy_only=policy_only, load_optimizer=load_optimizer)
        return state.get("infos")

    def get_inference_policy(self, device: str | None = None) -> torch.nn.Module:
        """Return a separate deterministic policy accepting ``(observation_history, goal)``."""
        import copy

        return copy.deepcopy(self.alg.policy).to(device or self.device).eval()

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the deterministic policy, with normalization, to TorchScript."""
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        torch.jit.script(self.get_inference_policy("cpu")).save(str(target / filename))

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register a repository whose git state should be saved with training logs."""
        self.logger.git_status_repos.append(repo_file_path)
