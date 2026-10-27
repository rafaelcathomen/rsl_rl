# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import time
import torch

from rsl_rl.algorithms import SAC
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.runners.utils import export_jit, export_onnx, save_checkpoint
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger


class OffPolicyRunner:
    """Off-policy runner for reinforcement learning algorithms with a replay buffer.

    Each learning iteration collects ``num_steps_per_env`` steps from all environments, stores them in the replay
    buffer, and then runs the algorithm's update. The ratio of gradient steps to environment steps is set by the
    algorithm's ``num_updates``.
    """

    alg: SAC
    """The off-policy algorithm."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the off-policy algorithm and logging stack on one device."""
        if int(os.getenv("WORLD_SIZE", "1")) != 1:
            raise ValueError("OffPolicyRunner currently supports one training process")
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.gpu_world_size = 1
        self.is_distributed = False
        self.gpu_local_rank = 0
        self.gpu_global_rank = 0
        self.cfg["multi_gpu"] = None

        obs = self.env.get_observations()
        alg_class: type[SAC] = resolve_callable(self.cfg["algorithm"]["class_name"])  # type: ignore
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )
        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop for the specified number of iterations."""
        if init_at_random_ep_len:
            lengths = self.env.episode_length_buf
            limits = torch.as_tensor(self.env.max_episode_length, device=lengths.device)
            self.env.episode_length_buf = (torch.rand_like(lengths, dtype=torch.float) * limits).to(lengths.dtype)

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.no_grad():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras)

            stop = time.time()
            collect_time = stop - start
            start = stop

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.action_std,
                rnd_weight=None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the algorithm and runner state, and upload it through the logger."""
        save_checkpoint(path, self.alg.save(), self.current_learning_iteration, infos)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict | None:
        """Load a trusted checkpoint, honoring the algorithm's model and state selection.

        ``load_cfg`` selects models, optimizer state and iteration; ``None`` loads
        all state. ``map_location`` controls the device mapping for deserialization.
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        if self.alg.load(loaded_dict, load_cfg, strict):
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        """Return the deterministic policy on the requested device for inference."""
        self.alg.eval_mode()
        return self.alg.get_policy().to(device)  # type: ignore

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the deterministic policy, including normalization, to TorchScript."""
        export_jit(self.alg.get_policy().as_jit(), path, filename)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the deterministic policy, including normalization, to ONNX."""
        export_onnx(self.alg.get_policy().as_onnx(verbose=verbose), path, filename, verbose)

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register a repository path whose git status should be logged."""
        self.logger.git_status_repos.append(repo_file_path)
