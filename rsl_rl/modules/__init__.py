#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic, ActorCriticSeparate
from .actor_critic_recurrent import ActorCriticRecurrent
from .actor_critic_beta import ActorCriticBeta
from .normalizer import EmpiricalNormalization
from .local_nav_module import SimpleNavPolicy
from .ac_beta_compress import ActorCriticBetaCompress, ActorCriticBetaCompressTemporal, ActorCriticBetaLidarTemporal
from .ac_lidar import ActorCriticBetaRecurrentLidar
from .ac_lidar_extra import ActorCriticBetaRecurrentLidarCnn
from .ac_lidar_height import ActorCriticBetaRecurrentLidarHeightCnn

__all__ = [
    "ActorCritic",
    "ActorCriticRecurrent",
    "ActorCriticBeta",
    "EmpiricalNormalization",
    "SimpleNavPolicy",
    "ActorCriticSeparate",
    "ActorCriticBetaCompress",
    "ActorCriticBetaCompressTemporal",
    "ActorCriticBetaLidarTemporal",
    "ActorCriticBetaRecurrentLidar",
    "ActorCriticBetaRecurrentLidarCnn",
    "ActorCriticBetaRecurrentLidarHeightCnn",
]
