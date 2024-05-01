#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# from rsl_rl.modules.actor_critic import ActorCritic
# from rsl_rl.modules.actor_critic_beta import ActorCriticBeta
from rsl_rl.utils import unpad_trajectories

from torch.distributions import Beta
import math


class ParallelMLPs(nn.Module):
    def __init__(
        self, input_dims, hidden_dims, out_hidden_dims, output_dim, activation, module_types, conv_size: int = 7
    ):
        super().__init__()

        self.conv_size = conv_size
        self.module_types = module_types

        self.input_dims = input_dims
        activation = get_activation(activation)

        # Validate lengths
        assert (
            len(input_dims) == len(hidden_dims) == len(module_types)
        ), "Input dimensions, hidden dimensions, and module types must have the same number of elements."

        # Create a module list for the sub-MLPs or Conv layers
        self.sub_modules = nn.ModuleList()
        for idx, (mod_type, dims) in enumerate(zip(module_types, hidden_dims)):
            layers = []
            in_dim = input_dims[idx]

            if mod_type == "mlp":
                for hidden_dim in dims:
                    layers.append(nn.Linear(in_dim, hidden_dim))
                    layers.append(activation)
                    in_dim = hidden_dim
            elif mod_type == "conv":
                in_dim = 1  # Start with 1 channel
                for out_channels in dims[:-2]:
                    layers.append(
                        nn.Conv1d(
                            in_channels=in_dim,
                            out_channels=out_channels,
                            kernel_size=conv_size,
                            padding=conv_size // 2,
                            stride=2,
                        )
                    )
                    layers.append(activation)
                    layers.append(nn.MaxPool1d(kernel_size=5, stride=2))
                    in_dim = out_channels

                # Add the last convolution layer without pooling and add fully connected layer
                layers.append(
                    nn.Conv1d(
                        in_channels=in_dim,
                        out_channels=dims[-2],
                        kernel_size=conv_size,
                        padding=conv_size // 2,
                        stride=2,
                    )
                )
                layers.append(activation)
                layers.append(nn.AdaptiveMaxPool1d(1))
                layers.append(nn.Flatten())
                layers.append(nn.Linear(dims[-2], dims[-1]))

            else:
                raise ValueError("Unsupported module type")

            self.sub_modules.append(nn.Sequential(*layers))

        # Calculate the input dimension of the final MLP
        final_in_dim = sum([dims[-1] for dims in hidden_dims])

        # Create the final MLP
        final_layers = []
        for dim in out_hidden_dims:
            final_layers.append(nn.Linear(final_in_dim, dim))
            final_layers.append(activation)
            final_in_dim = dim
        final_layers.append(nn.Linear(final_in_dim, output_dim))

        self.final_mlp = nn.Sequential(*final_layers)

    def forward(self, x):
        sub_outputs = []
        for idx, (module, mod_type) in enumerate(zip(self.sub_modules, self.module_types)):
            if mod_type == "mlp":
                sub_input = x[:, sum(self.input_dims[:idx]) : sum(self.input_dims[: idx + 1])]
                sub_output = module(sub_input)
            elif mod_type == "conv":
                sub_input = x[:, sum(self.input_dims[:idx]) : sum(self.input_dims[: idx + 1])]
                sub_input = sub_input.unsqueeze(1)  # Add channel dimension
                sub_input = F.pad(sub_input, (self.conv_size // 2, self.conv_size // 2), mode="circular")
                sub_output = module(sub_input)
            sub_outputs.append(sub_output)

        # Concatenate all outputs from the sub-modules
        concat = torch.cat(sub_outputs, dim=1)

        # Pass the concatenated output through the final MLP
        out = self.final_mlp(concat)
        return out


class ActorCriticBetaCompress(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        input_dims=None,
        actor_hidden_dims_per_split=[256, 256, 256],
        critic_hidden_dims_per_split=[256, 256, 256],
        actor_out_hidden_dims=[256, 256, 256],
        critic_out_hidden_dims=[256, 256, 256],
        activation="elu",
        module_types=None,
        beta_initial_logit=0.5,  # centered mean intially
        beta_initial_scale=5.0,  # sharper distribution initially
        **kwargs,
    ):
        """
        create a neural network with len(input_dims) parallel input streams, which then get concatenated to the out hidden layers.
        """
        if kwargs:
            print(
                "ActorCriticBeta.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        if input_dims is None:
            # normal mlp
            input_dims = [num_actor_obs]
        if module_types is None:
            module_types = ["mlp"] * len(input_dims)

        if sum(input_dims) != num_actor_obs or sum(input_dims) != num_critic_obs:
            raise ValueError(
                f"sum of input dims must be equal to obs. num_actor_obs: {num_actor_obs}, num_critic_obs: {num_critic_obs}, sum(input_dims): {sum(input_dims)}"
            )

        if len(actor_hidden_dims_per_split) != len(input_dims):
            raise ValueError(
                f"input_dimes has to contain the same number of elements as actor_hidden_dims_per_split. len(input_dims): {len(input_dims)}, len(actor_hidden_dims_per_split): {len(actor_hidden_dims_per_split)}"
            )

        self.actor = ParallelMLPs(
            input_dims, actor_hidden_dims_per_split, actor_out_hidden_dims, num_actions * 2, activation, module_types
        )  # 2*num_actions for mean and entropy
        self.critic = ParallelMLPs(
            input_dims, critic_hidden_dims_per_split, critic_out_hidden_dims, 1, activation, module_types
        )

        print(f"Actor net: {self.actor}")

        print(f"Critic net: {self.critic}")

        print(f"num actor params: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"num critic params: {sum(p.numel() for p in self.critic.parameters())}")
        print(f"total num params: {sum(p.numel() for p in self.parameters())}")

        # Action noise
        self.distribution = Beta(1, 1)
        self.soft_plus = torch.nn.Softplus(beta=1)
        self.sigmoid = nn.Sigmoid()
        self.beta_initial_logit_shift = math.log(beta_initial_logit / (1.0 - beta_initial_logit))  # inverse sigmoid
        self.beta_initial_scale = beta_initial_scale
        self.output_dim = num_actions

        # disable args validation for speedup
        Beta.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def std(self):
        return self.distribution.stddev

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_beta_parameters(self, logits):
        """Get alpha and beta parameters from logits"""
        ratio = self.sigmoid(logits[..., : self.output_dim] + self.beta_initial_logit_shift)
        sum = (self.soft_plus(logits[..., self.output_dim :]) + 1) * self.beta_initial_scale

        # Compute alpha and beta
        alpha = ratio * sum
        beta = sum - alpha

        # Nummerical stability
        alpha += 1e-6
        beta += 1e-4
        return alpha, beta

    def update_distribution(self, observations):
        """Update the distribution of the policy"""
        logits = self.actor(observations)
        alpha, beta = self.get_beta_parameters(logits)

        # Update distribution
        self.distribution = Beta(alpha, beta, validate_args=False)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        logits = self.actor(observations)
        actions_mean = self.sigmoid(logits[:, : self.output_dim] + self.beta_initial_logit_shift)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
