#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal

# from rsl_rl.modules.actor_critic import ActorCritic
# from rsl_rl.modules.actor_critic_beta import ActorCriticBeta
from rsl_rl.utils import unpad_trajectories

from torch.distributions import Beta
import math


##
# sub modules
##
class ParallelNets(nn.Module):
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


class ParallelTemporalMLPs(nn.Module):
    def __init__(
        self,
        input_dims: list[int],
        parallel_hidden_dims: list[list[int]],
        parallel_hidden_out_dims: list[list[int]],
        out_hidden_dims: list[int],
        output_dim: int,
        activation_str: str,
        memory: list[int],
    ):
        super().__init__()

        self.input_dims = input_dims
        activation = get_activation(activation_str)

        # Create a module list for the sub-MLPs or Conv layers
        self.sub_modules = nn.ModuleList()
        for idx, (history, in_dims, out_dims) in enumerate(zip(memory, parallel_hidden_dims, parallel_hidden_out_dims)):
            input_dim = input_dims[idx]
            module = TemporalMLP(
                input_dim=input_dim,
                hidden_dims=in_dims,
                memory=history,
                output_dims=out_dims,
                activation_str=activation_str,
            )
            self.sub_modules.append(module)

        # Calculate the input dimension of the final MLP
        final_in_dim = sum([dims[-1] for dims in parallel_hidden_out_dims])

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
        for idx, module in enumerate(self.sub_modules):
            sub_input = x[:, sum(self.input_dims[:idx]) : sum(self.input_dims[: idx + 1])]
            sub_output = module(sub_input)
            sub_outputs.append(sub_output)

        # Concatenate all outputs from the sub-modules
        concat = torch.cat(sub_outputs, dim=1)

        # Pass the concatenated output through the final MLP
        out = self.final_mlp(concat)
        return out


class ParallelSiameseMLP(nn.Module):
    def __init__(
        self,
        input_dims: list[int],
        single_dim: list[int],
        n_parallels: list[int],
        parallel_siamese_hidden_dims: list[list[int]],
        parallel_hidden_out_dims: list[list[int]],
        out_hidden_dims: list[int],
        output_dim: int,
        activation_str: str,
    ):
        super().__init__()

        self.input_dims = input_dims
        activation = get_activation(activation_str)

        # Create a module list for the sub-MLPs or Conv layers
        self.sub_modules = nn.ModuleList()
        for idx, (siamese_dims, out_dims) in enumerate(zip(parallel_siamese_hidden_dims, parallel_hidden_out_dims)):

            module = SiameseMLP(
                full_input_dim=input_dims[idx],
                single_dim=single_dim[idx],
                n_parallels=n_parallels[idx],
                parallel_hidden_layers=siamese_dims,
                output_hidden_layers=out_dims,
                activation_str=activation_str,
            )
            self.sub_modules.append(module)

        # Calculate the input dimension of the final MLP
        final_in_dim = sum([dims[-1] for dims in parallel_hidden_out_dims])

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
        for idx, module in enumerate(self.sub_modules):
            sub_input = x[:, sum(self.input_dims[:idx]) : sum(self.input_dims[: idx + 1])]
            sub_output = module(sub_input)
            sub_outputs.append(sub_output)

        # Concatenate all outputs from the sub-modules
        concat = torch.cat(sub_outputs, dim=1)

        # Pass the concatenated output through the final MLP
        out = self.final_mlp(concat)
        return out


def create_mlp(input_dim: int, layer_dims: list[int], activation: nn.Module, remove_last_nonlinearity: bool = False):
    layers = []  # List to store layers

    # Get all layer dimensions pairs
    input_output_pairs = zip([input_dim] + layer_dims[:-1], layer_dims)

    # Create Linear and ReLU layers for each pair
    for input_size, output_size in input_output_pairs:
        layers.append(nn.Linear(input_size, output_size))
        layers.append(activation)

    if remove_last_nonlinearity:
        layers.pop()  # Remove the last ReLU added in the loop
    # Construct the sequential model
    return nn.Sequential(*layers)


class GRUHistoryProcessor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dims: list[int], activation: nn.Module) -> None:
        super(GRUHistoryProcessor, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        # Create additional MLP layers if more than one output dimension is specified
        layers = []
        input_size = hidden_dim
        for output_size in output_dims:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(activation)
            input_size = output_size
        if layers:
            layers.pop()  # Remove last activation if present
        self.mlp = nn.Sequential(*layers) if layers else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape [batch_size, sequence_length, input_dim]
        _, hn = self.gru(x)  # hn shape [1, batch_size, hidden_dim]
        x = hn.squeeze(0)  # reshape to [batch_size, hidden_dim]
        if self.mlp:
            x = self.mlp(x)
        return x


class NavigationLidar2dNetHistory(nn.Module):
    """Network specifically designed for navigation tasks with 2d LiDAR History observations.

    Parameters:
    ----------
    non_lidar_dim: int, The dimension of non-LiDAR observations.
    lidar_dim: int, The dimension of LiDAR observations.
    lidar_extra_dim: int, The dimension of extra LiDAR observations (also with history, ie to identify each lidar scan).
    history_length: int, The number of history steps.
    non_lidar_layer_dims: list[int], The hidden pre dimensions for the non-LiDAR observations.
    lidar_compress_layer_dims: list[int], The hidden dimensions for the compression mlp for the LiDAR observations.
    lidar_extra_in_dims: list[int], The hidden dimensions for the extra LiDAR observations pre processing.
    lidar_extra_merge_mlp_dims: list[int], The hidden dimensions for the mlp that combines lidar and lidar extra embeddings.
    history_processing_mlp_dims: list[int], The hidden dimensions for the history processing mlp.
    out_layer_dims: list[int], The hidden dimensions for the output layer.
    out_dim: int, The output dimension.

    input_dimension = non_lidar_dim + (lidar_dim + lidar_extra_dim) * history_length
    """

    def __init__(
        self,
        non_lidar_dim: int,
        lidar_dim: int,
        lidar_extra_dim: int,
        history_length: int,
        non_lidar_layer_dims: list[int],
        lidar_compress_layer_dims: list[int],
        lidar_extra_in_dims: list[int],
        lidar_extra_merge_mlp_dims: list[int],
        history_processing_mlp_dims: list[int],
        out_layer_dims: list[int],
        out_dim: int,
        activation_str: str,
        gru_dim: int | None = None,
    ) -> None:
        super().__init__()

        activation = get_activation(activation_str)
        self.history_length = history_length
        self.lidar_dim = lidar_dim
        self.non_lidar_dim = non_lidar_dim
        self.lidar_extra_dim = lidar_extra_dim
        self.using_gru = gru_dim is not None

        self.non_lidar_in_mlp = create_mlp(non_lidar_dim, non_lidar_layer_dims, activation)
        self.lidar_compression_mlp = create_mlp(lidar_dim, lidar_compress_layer_dims, activation)
        self.lidar_extra_mlp = create_mlp(lidar_extra_dim, lidar_extra_in_dims, activation)
        self.lidar_and_extra_mlp = create_mlp(
            lidar_compress_layer_dims[-1] + lidar_extra_in_dims[-1], lidar_extra_merge_mlp_dims, activation
        )
        if gru_dim is None:
            self.history_mlp = create_mlp(
                lidar_extra_merge_mlp_dims[-1] * history_length, history_processing_mlp_dims, activation
            )
        else:
            self.history_mlp = GRUHistoryProcessor(
                input_dim=lidar_extra_merge_mlp_dims[-1],
                hidden_dim=gru_dim,
                output_dims=history_processing_mlp_dims,
                activation=activation,
            )

        out_layer_dims.append(out_dim)
        self.out_mlp = create_mlp(
            history_processing_mlp_dims[-1] + non_lidar_layer_dims[-1],
            out_layer_dims,
            activation,
            remove_last_nonlinearity=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split the input into non-lidar, lidar and lidar extra observations
        non_lidar_obs = x[:, : self.non_lidar_dim]
        lidar_extra_history = x[:, self.non_lidar_dim :].reshape(
            -1, self.history_length, self.lidar_dim + self.lidar_extra_dim
        )

        # Process non-lidar observations
        non_lidar_out = self.non_lidar_in_mlp(non_lidar_obs)

        # Process lidar observations
        lidar_out = self.lidar_compression_mlp(lidar_extra_history[:, :, : self.lidar_dim])
        lidar_extra_out = self.lidar_extra_mlp(lidar_extra_history[:, :, self.lidar_dim :])
        history_merged = self.lidar_and_extra_mlp(torch.cat((lidar_out, lidar_extra_out), dim=-1))
        if self.using_gru:
            history_processed = self.history_mlp(history_merged)
        else:
            history_processed = self.history_mlp(history_merged.view(history_merged.size(0), -1))

        # Concatenate the processed non-lidar and lidar observations
        out = torch.cat((non_lidar_out, history_processed), dim=-1)
        out = self.out_mlp(out)
        return out


##
# sub sub modules
##
class TemporalMLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: list[int], memory: int, output_dims: list[int], activation_str: str
    ) -> None:
        super().__init__()

        self.use_memory = memory > 0

        activation = get_activation(activation_str)

        # First MLP processes the current input
        hidden_layers = []
        current_dim = input_dim
        for dim in hidden_dims:
            hidden_layers.append(nn.Linear(current_dim, dim))
            hidden_layers.append(activation)
            current_dim = dim
        self.hidden_mlp = nn.Sequential(*hidden_layers)

        # Memory to store previous outputs of hidden_mlp
        self.memory = memory
        if self.use_memory:
            self.outputs_memory = None  # will be initialized in forward

        # Second MLP processes concatenated outputs from memory
        total_input_dim = hidden_dims[-1] * (memory + 1)  # current output + memory outputs
        output_layers = []
        current_dim = total_input_dim
        for dim in output_dims:
            output_layers.append(nn.Linear(current_dim, dim))
            output_layers.append(activation)
            current_dim = dim
        self.output_mlp = nn.Sequential(*output_layers)

    def forward(self, x: torch.Tensor):
        # Pass current input through the hidden MLP
        current_output: torch.Tensor = self.hidden_mlp(x)

        if not self.use_memory:
            return self.output_mlp(current_output)

        if self.outputs_memory is None:
            # Initialize the memory tensor with zeros with the right dimensions
            self.outputs_memory = torch.zeros(x.size(0), self.memory, current_output.size(1), device=x.device)

        # Update memory: shift memory to make space for new output
        self.outputs_memory = torch.roll(self.outputs_memory, -1, dims=0)
        self.outputs_memory[:, -1, :] = current_output.detach()  # Detach to avoid backprop through time

        # Concatenate current output with outputs from memory
        concatenated_outputs = torch.cat((current_output, self.outputs_memory.reshape(x.size(0), -1)), dim=1)

        # Pass the concatenated vector through the output MLP
        output = self.output_mlp(concatenated_outputs)
        return output


class SiameseMLP(nn.Module):
    def __init__(
        self,
        full_input_dim: int,
        single_dim: int,
        n_parallels: int,
        parallel_hidden_layers: list[int],
        output_hidden_layers: list[int],
        activation_str: str,
    ) -> None:
        super().__init__()
        # Calculate number of parallel networks
        activation = get_activation(activation_str)
        self.n_parallels = n_parallels
        assert self.n_parallels * single_dim == full_input_dim, "full_input_dim must be divisible by single_dim"

        # Create the parallel MLPs
        parallel_layers = [nn.Linear(single_dim, parallel_hidden_layers[0]), activation]
        for idx in range(len(parallel_hidden_layers) - 1):
            parallel_layers.append(nn.Linear(parallel_hidden_layers[idx], parallel_hidden_layers[idx + 1]))
            parallel_layers.append(activation)
        self.parallel_mlp = nn.Sequential(*parallel_layers)

        # Output MLP
        output_layers = [nn.Linear(parallel_hidden_layers[-1] * self.n_parallels, output_hidden_layers[0]), activation]
        for idx in range(1, len(output_hidden_layers)):
            output_layers.append(nn.Linear(output_hidden_layers[idx - 1], output_hidden_layers[idx]))
            output_layers.append(nn.ReLU())
        self.output_mlp = nn.Sequential(*output_layers)

    def forward(self, x):
        # Split input into parallel chunks and process each through the parallel MLP
        x = x.view(-1, self.n_parallels, x.size(1) // self.n_parallels)
        x = [self.parallel_mlp(x[:, i]) for i in range(self.n_parallels)]
        x = torch.cat(x, dim=1)  # Concatenate outputs

        # Pass through the output MLP
        x = self.output_mlp(x)
        return x


class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, num_timestamps, conv_layers=None, output_dim=10):
        """
        Initializes the TemporalConvNet model.

        Parameters:
        input_dim (int): The number of features per timestamp.
        num_timestamps (int): The number of timestamps per input sample.
        conv_layers (list of tuples): A list where each tuple contains (filters, kernel_size, pooling_size).
                                      Default is [(64, 3, 2), (128, 3, 2)].
        output_dim (int): The dimension of the output layer (e.g., number of classes for classification).
        """
        super(TemporalConvNet, self).__init__()
        if conv_layers is None:
            conv_layers = [(64, 3, 2), (128, 3, 2)]  # Default convolutional layers configuration

        # Creating convolutional layers
        layers = []
        in_channels = input_dim
        for filters, kernel_size, pooling_size in conv_layers:
            layers.append(nn.Conv1d(in_channels, filters, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            if pooling_size > 0:
                layers.append(nn.MaxPool1d(pooling_size))
            in_channels = filters

        # Adding all layers to a Sequential module
        self.conv_layers = nn.Sequential(*layers)

        # Calculate the length of the output from the last Conv1d layer after pooling
        output_length = num_timestamps
        for _, _, pooling_size in conv_layers:
            if pooling_size > 0:
                output_length = (output_length + pooling_size - 1) // pooling_size

        # Fully connected layer
        self.fc = nn.Linear(in_channels * output_length, output_dim)

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters:
        x (Tensor): Input tensor of shape (batch_size, num_timestamps, input_dim).
        """
        # Permute x to match the (batch_size, channels, length) format expected by Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


##
# main modules
##


######################################################################
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

        self.actor = ParallelNets(
            input_dims, actor_hidden_dims_per_split, actor_out_hidden_dims, num_actions * 2, activation, module_types
        )  # 2*num_actions for mean and entropy
        self.critic = ParallelNets(
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


class ActorCriticBetaCompressTemporal(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        input_dims: list[int],
        single_dims: list[int],
        n_parallels: list[int],
        actor_siamese_hidden_dims_per_split: list[list[int]] = [[256, 256, 256]],
        actor_out_hidden_dims_per_split: list[list[int]] = [[256, 256, 256]],
        critic_siamese_hidden_dims_per_split: list[list[int]] = [[256, 256, 256]],
        critic_out_hidden_dims_per_split: list[list[int]] = [[256, 256, 256]],
        actor_out_hidden_dims: list[int] = [256, 256, 256],
        critic_out_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        beta_initial_logit: float = 0.5,  # centered mean intially
        beta_initial_scale: float = 5.0,  # sharper distribution initially
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

        if sum(input_dims) != num_actor_obs or sum(input_dims) != num_critic_obs:
            raise ValueError(
                f"sum of input dims must be equal to obs. num_actor_obs: {num_actor_obs}, num_critic_obs: {num_critic_obs}, sum(input_dims): {sum(input_dims)}"
            )

        if len(actor_siamese_hidden_dims_per_split) != len(input_dims) or len(actor_out_hidden_dims_per_split) != len(
            input_dims
        ):
            raise ValueError(
                f"list length miss match: \ninput dims: {len(input_dims)}\nlen(actor_in_hidden_dims_per_split): {len(actor_siamese_hidden_dims_per_split)} \nlen(actor_out_hidden_dims_per_split): {len(actor_out_hidden_dims_per_split)}"
            )

        self.actor = ParallelSiameseMLP(
            input_dims=input_dims,
            single_dim=single_dims,
            n_parallels=n_parallels,
            parallel_siamese_hidden_dims=actor_siamese_hidden_dims_per_split,
            parallel_hidden_out_dims=actor_out_hidden_dims_per_split,
            out_hidden_dims=actor_out_hidden_dims,
            output_dim=num_actions * 2,  # 2*num_actions for mean and entropy
            activation_str=activation,
        )

        self.critic = ParallelSiameseMLP(
            input_dims=input_dims,
            single_dim=single_dims,
            n_parallels=n_parallels,
            parallel_siamese_hidden_dims=critic_siamese_hidden_dims_per_split,
            parallel_hidden_out_dims=critic_out_hidden_dims_per_split,
            out_hidden_dims=critic_out_hidden_dims,
            output_dim=1,  # 1 for value
            activation_str=activation,
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


class ActorCriticBetaLidarTemporal(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        non_lidar_dim: int,
        lidar_dim: int,
        lidar_extra_dim: int,
        history_length: int,
        non_lidar_layer_dims: list[int],
        lidar_compress_layer_dims: list[int],
        lidar_extra_in_dims: list[int],
        lidar_extra_merge_mlp_dims: list[int],
        history_processing_mlp_dims: list[int],
        out_layer_dims: list[int],
        gru_dim: int | None = None,
        activation: str = "elu",
        beta_initial_logit: float = 0.5,  # centered mean intially
        beta_initial_scale: float = 5.0,  # sharper distribution initially
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

        if (
            num_actor_obs != non_lidar_dim + (lidar_dim + lidar_extra_dim) * history_length
            or num_actor_obs != num_critic_obs
        ):
            raise ValueError(
                f"num_actor_obs must be equal to non_lidar_dim + (lidar_dim + lidar_extra_dim) * history_length. num_actor_obs: {num_actor_obs}, non_lidar_dim: {non_lidar_dim}, lidar_dim: {lidar_dim}, lidar_extra_dim: {lidar_extra_dim}, history_length: {history_length}"
            )

        self.actor = NavigationLidar2dNetHistory(
            non_lidar_dim=non_lidar_dim,
            lidar_dim=lidar_dim,
            lidar_extra_dim=lidar_extra_dim,
            history_length=history_length,
            non_lidar_layer_dims=non_lidar_layer_dims,
            lidar_compress_layer_dims=lidar_compress_layer_dims,
            lidar_extra_in_dims=lidar_extra_in_dims,
            lidar_extra_merge_mlp_dims=lidar_extra_merge_mlp_dims,
            history_processing_mlp_dims=history_processing_mlp_dims,
            out_layer_dims=out_layer_dims,
            gru_dim=gru_dim,
            out_dim=num_actions * 2,
            activation_str=activation,
        )

        self.critic = NavigationLidar2dNetHistory(
            non_lidar_dim=non_lidar_dim,
            lidar_dim=lidar_dim,
            lidar_extra_dim=lidar_extra_dim,
            history_length=history_length,
            non_lidar_layer_dims=non_lidar_layer_dims,
            lidar_compress_layer_dims=lidar_compress_layer_dims,
            lidar_extra_in_dims=lidar_extra_in_dims,
            lidar_extra_merge_mlp_dims=lidar_extra_merge_mlp_dims,
            history_processing_mlp_dims=history_processing_mlp_dims,
            out_layer_dims=out_layer_dims,
            gru_dim=gru_dim,
            out_dim=1,
            activation_str=activation,
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
