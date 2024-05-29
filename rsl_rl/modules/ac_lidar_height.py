from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.modules.actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories
from torch.distributions import Beta
import math


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


class CircularPaddingConv1d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super().__init__()
        self.padding_size = kernel_size // 2
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, stride)

    def circular_pad_1d(self, x, pad):
        """Apply circular padding to the last dimension of a tensor."""
        if pad == 0:
            return x
        return torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)

    def forward(self, x):
        # Manually apply circular padding
        x_padded = self.circular_pad_1d(x, self.padding_size)
        return self.conv(x_padded)


class FlexibleBatchCircularlPadConv1d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super().__init__()
        self.padding_size = kernel_size // 2
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, stride)

    def circular_pad_1d(self, x, pad):
        """Apply circular padding to the last dimension of a tensor."""
        if pad == 0:
            return x
        return torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)

    def forward(self, x):
        original_shape = x.shape
        # Flatten all but the last two dimensions into one dimension for batch handling
        x = x.reshape(-1, *original_shape[-2:])
        # Apply circular padding
        x = self.circular_pad_1d(x, self.padding_size)
        # Apply convolution
        x = self.conv(x)
        # Reshape to the original batch structure with new channel dimension
        new_shape = original_shape[:-2] + (x.shape[-2], x.shape[-1])
        x = x.reshape(new_shape)
        return x


class FlexibleBatchCircularPadConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super().__init__()
        self.padding_size = kernel_size // 2
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride)

    def circular_pad_2d(self, x, pad):
        """Apply circular padding to the last two dimensions of a tensor."""
        if pad == 0:
            return x
        # Pad the last dimension (width)
        x = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)
        # Pad the second last dimension (height)
        x = torch.cat([x[..., -pad:, :], x, x[..., :pad, :]], dim=-2)
        return x

    def forward(self, x):
        original_shape = x.shape
        # Flatten all but the last three dimensions into one dimension for batch handling
        x = x.reshape(-1, *original_shape[-3:])
        # Apply circular padding
        x = self.circular_pad_2d(x, self.padding_size)
        # Apply convolution
        x = self.conv(x)
        # Reshape to the original batch structure with new channel dimension
        new_shape = original_shape[:-3] + (x.shape[-3], x.shape[-2], x.shape[-1])
        x = x.reshape(new_shape)
        return x


def create_cnn_1d(
    input_channels: int,
    layer_channels: list[int],
    kernel_sizes: list[int],
    strides: list[int],
    activation: nn.Module,
):
    layers = []  # List to store layers

    # Get all layer dimensions pairs
    input_output_pairs = zip([input_channels] + layer_channels[:-1], layer_channels, kernel_sizes, strides)

    # Create custom Conv1d with circular padding and activation layers for each pair
    for input_channels, output_channels, kernel_size, stride in input_output_pairs:
        layers.append(FlexibleBatchCircularlPadConv1d(input_channels, output_channels, kernel_size, stride))
        layers.append(activation)

    # Construct the sequential model
    return nn.Sequential(*layers)


def create_cnn_2d(
    input_channels: int,
    layer_channels: list[int],
    kernel_sizes: list[int],
    strides: list[int],
    activation: nn.Module,
    maxpool_kernels: list[int],
):
    layers = []  # List to store layers

    # Get all layer dimensions pairs
    input_output_pairs = zip(
        [input_channels] + layer_channels[:-1], layer_channels, kernel_sizes, strides, maxpool_kernels
    )

    # Create custom Conv2d with circular padding and activation layers for each pair
    for input_channels, output_channels, kernel_size, stride, maxpool_kernel in input_output_pairs:
        layers.append(FlexibleBatchCircularPadConv2d(input_channels, output_channels, kernel_size, stride))
        layers.append(activation)
        if maxpool_kernel > 1:
            layers.append(nn.MaxPool2d(kernel_size=maxpool_kernel))

    return nn.Sequential(*layers)


class ActorCriticBetaRecurrentLidarHeightCnn(nn.Module):
    # TODO: add last pos to gru history
    # add option to not use gru

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        non_lidar_dim: int,
        lidar_dim: int,
        lidar_extra_dim: int,
        num_lidar_channels: int,
        non_lidar_layer_dims: list[int],
        heigh_scan_dims: tuple[int, int, int],
        lidar_compress_conv_layer_dims: list[int],
        lidar_compress_conv_kernel_sizes: list[int],
        lidar_compress_conv_strides: list[int],
        lidar_compress_conv_to_mlp_dims: list[int],
        lidar_extra_in_dims: list[int],
        lidar_merge_mlp_dims: list[int],
        height_conv_channels: list[int],
        height_conv_kernel_sizes: list[int],
        height_conv_strides: list[int],
        heigh_conv_max_pool_kernel_sizes: list[int],
        height_conv_to_mlp_dims: list[int],
        history_processing_mlp_dims: list[int],
        out_layer_dims: list[int],
        gru_dim: int,
        gru_layers: int = 1,
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
            num_actor_obs
            != non_lidar_dim
            + lidar_dim * num_lidar_channels
            + lidar_extra_dim
            + heigh_scan_dims[0] * heigh_scan_dims[1] * heigh_scan_dims[2]
            or num_actor_obs != num_critic_obs
        ):
            raise ValueError(
                f"""num_actor_obs must be equal to non_lidar_dim + lidar_dim + lidar_extra_dim. num_actor_obs: 
                {num_actor_obs}, non_lidar_dim: {non_lidar_dim}, lidar_dim * num_channels: {lidar_dim * num_lidar_channels}, lidar_extra_dim: {lidar_extra_dim}"""
            )
        activation_module = get_activation(activation)

        self.lidar_dim = lidar_dim
        self.non_lidar_dim = non_lidar_dim
        self.lidar_channels = num_lidar_channels
        self.lidar_extra_dim = lidar_extra_dim
        self.heigh_scan_dims = heigh_scan_dims

        ##
        # define networks
        ##

        # -- recurrence
        # - lidar embedding
        self.actor_lidar_conv_embedder = create_cnn_1d(
            num_lidar_channels,
            lidar_compress_conv_layer_dims,
            kernel_sizes=lidar_compress_conv_kernel_sizes,
            strides=lidar_compress_conv_strides,
            activation=activation_module,
        )
        self.critic_lidar_conv_embedder = create_cnn_1d(
            num_lidar_channels,
            lidar_compress_conv_layer_dims,
            kernel_sizes=lidar_compress_conv_kernel_sizes,
            strides=lidar_compress_conv_strides,
            activation=activation_module,
        )
        # calculate cnn output size
        dummy_input = torch.zeros(1, num_lidar_channels, lidar_dim)
        dummy_out = self.actor_lidar_conv_embedder(dummy_input)
        flattened_out_dim = dummy_out.shape[1] * dummy_out.shape[2]

        # - conv to mlp
        self.actor_lidar_embedder_mlp = create_mlp(
            flattened_out_dim, lidar_compress_conv_to_mlp_dims, activation_module
        )
        self.critic_lidar_embedder_mlp = create_mlp(
            flattened_out_dim, lidar_compress_conv_to_mlp_dims, activation_module
        )

        # extra embedding
        self.actor_lidar_extra_mlp = create_mlp(lidar_extra_dim, lidar_extra_in_dims, activation_module)
        self.critic_lidar_extra_mlp = create_mlp(lidar_extra_dim, lidar_extra_in_dims, activation_module)

        # - height conv
        self.actor_height_conv = create_cnn_2d(
            heigh_scan_dims[2],
            height_conv_channels,
            height_conv_kernel_sizes,
            height_conv_strides,
            activation_module,
            heigh_conv_max_pool_kernel_sizes,
        )
        self.critic_height_conv = create_cnn_2d(
            heigh_scan_dims[2],
            height_conv_channels,
            height_conv_kernel_sizes,
            height_conv_strides,
            activation_module,
            heigh_conv_max_pool_kernel_sizes,
        )
        # calculate cnn output size
        dummy_input = torch.zeros(1, heigh_scan_dims[2], heigh_scan_dims[0], heigh_scan_dims[1])
        dummy_out = self.actor_height_conv(dummy_input)
        flattened_out_dim = dummy_out.shape[1] * dummy_out.shape[2] * dummy_out.shape[3]

        # height conv to mlp
        self.actor_height_embedder_mlp = create_mlp(flattened_out_dim, height_conv_to_mlp_dims, activation_module)
        self.critic_height_embedder_mlp = create_mlp(flattened_out_dim, height_conv_to_mlp_dims, activation_module)

        # merge
        self.actor_lidar_merger_mlp = create_mlp(
            lidar_compress_conv_to_mlp_dims[-1] + lidar_extra_in_dims[-1] + height_conv_to_mlp_dims[-1],
            lidar_merge_mlp_dims,
            activation_module,
        )
        self.critic_lidar_merger_mlp = create_mlp(
            lidar_compress_conv_to_mlp_dims[-1] + lidar_extra_in_dims[-1] + height_conv_to_mlp_dims[-1],
            lidar_merge_mlp_dims,
            activation_module,
        )

        # - memory
        if gru_layers > 0:
            self.actor_memory = Memory(
                input_size=lidar_merge_mlp_dims[-1], type="gru", num_layers=gru_layers, hidden_size=gru_dim
            )
            self.critic_memory = Memory(
                input_size=lidar_merge_mlp_dims[-1], type="gru", num_layers=gru_layers, hidden_size=gru_dim
            )
        else:  # replace with mlp
            self.is_recurrent = False
            self.actor_no_memory = create_mlp(lidar_merge_mlp_dims[-1], [gru_dim], activation_module)
            self.critic_no_memory = create_mlp(lidar_merge_mlp_dims[-1], [gru_dim], activation_module)

        # - history processing
        self.actor_memory_processor = create_mlp(
            input_dim=gru_dim, layer_dims=history_processing_mlp_dims, activation=activation_module
        )
        self.critic_memory_processor = create_mlp(
            input_dim=gru_dim, layer_dims=history_processing_mlp_dims, activation=activation_module
        )

        # -- non recurrence
        self.actor_non_lidar_mlp = create_mlp(non_lidar_dim, non_lidar_layer_dims, activation_module)
        self.critic_non_lidar_mlp = create_mlp(non_lidar_dim, non_lidar_layer_dims, activation_module)

        # -- output
        out_in_dim = history_processing_mlp_dims[-1] + non_lidar_layer_dims[-1]
        actor_out_layers = out_layer_dims + [num_actions * 2]
        critic_out_layers = out_layer_dims + [1]
        self.actor_out_mlp = create_mlp(out_in_dim, actor_out_layers, activation_module, remove_last_nonlinearity=True)
        self.critic_out_mlp = create_mlp(
            out_in_dim, critic_out_layers, activation_module, remove_last_nonlinearity=True
        )

        # print(f"Actor net: {self.actor}")

        # print(f"Critic net: {self.critic}")

        # print(f"num actor params: {sum(p.numel() for p in self.actor.parameters())}")
        # print(f"num critic params: {sum(p.numel() for p in self.critic.parameters())}")
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
        if self.is_recurrent:
            self.actor_memory.reset(dones)
            self.critic_memory.reset(dones)

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

    def actor_forward(self, x, masks=None, hidden_states=None):
        non_lidar_obs = x[..., : self.non_lidar_dim]
        lidar_obs = x[..., self.non_lidar_dim : self.non_lidar_dim + self.lidar_dim * self.lidar_channels]
        heigh_dim = self.heigh_scan_dims[0] * self.heigh_scan_dims[1] * self.heigh_scan_dims[2]
        lidar_extra_obs = x[..., -self.lidar_extra_dim - heigh_dim : -heigh_dim]
        heigh_scan = x[..., -heigh_dim:]
        # -- recurrent part

        # conv lidar
        batch_shape = lidar_obs.shape[:-1]
        lidar_obs = lidar_obs.reshape(*batch_shape, self.lidar_channels, self.lidar_dim)
        lidar_embedded = self.actor_lidar_conv_embedder(lidar_obs)
        # conv lidar to mlp
        lidar_embedded = lidar_embedded.view(*batch_shape, -1)
        lidar_embedded = self.actor_lidar_embedder_mlp(lidar_embedded)
        # extra
        lidar_extra_embedding = self.actor_lidar_extra_mlp(lidar_extra_obs)
        # conv height
        heigh_scan = heigh_scan.view(
            *batch_shape, self.heigh_scan_dims[2], self.heigh_scan_dims[0], self.heigh_scan_dims[1]
        )
        heigh_embedded = self.actor_height_conv(heigh_scan)
        # conv height to mlp
        heigh_embedded = heigh_embedded.view(*batch_shape, -1)
        heigh_embedded = self.actor_height_embedder_mlp(heigh_embedded)

        # merge
        lidar_merged = torch.cat((lidar_embedded, lidar_extra_embedding, heigh_embedded), dim=-1)
        lidar_merged = self.actor_lidar_merger_mlp(lidar_merged)

        # memory
        if self.is_recurrent:
            memory_out = self.actor_memory(lidar_merged, masks, hidden_states)
        else:
            memory_out = self.actor_no_memory(lidar_merged)
        memory_processed = self.actor_memory_processor(memory_out.squeeze(0))

        # -- non-recurrent part
        non_lidar_out = self.actor_non_lidar_mlp(non_lidar_obs)
        if masks is not None:
            non_lidar_out = unpad_trajectories(non_lidar_out, masks)

        # output
        out = torch.cat((non_lidar_out, memory_processed), dim=-1)
        out = self.actor_out_mlp(out)
        return out

    def critic_forward(self, x, masks=None, hidden_states=None):
        non_lidar_obs = x[..., : self.non_lidar_dim]
        lidar_obs = x[..., self.non_lidar_dim : self.non_lidar_dim + self.lidar_dim * self.lidar_channels]
        heigh_dim = self.heigh_scan_dims[0] * self.heigh_scan_dims[1] * self.heigh_scan_dims[2]
        lidar_extra_obs = x[..., -self.lidar_extra_dim - heigh_dim : -heigh_dim]
        heigh_scan = x[..., -heigh_dim:]
        # -- recurrent part

        # conv lidar
        batch_shape = lidar_obs.shape[:-1]
        lidar_obs = lidar_obs.reshape(*batch_shape, self.lidar_channels, self.lidar_dim)
        lidar_embedded = self.critic_lidar_conv_embedder(lidar_obs)
        # conv lidar to mlp
        lidar_embedded = lidar_embedded.view(*batch_shape, -1)
        lidar_embedded = self.critic_lidar_embedder_mlp(lidar_embedded)
        # extra
        lidar_extra_embedding = self.critic_lidar_extra_mlp(lidar_extra_obs)
        # conv height
        heigh_scan = heigh_scan.view(
            *batch_shape, self.heigh_scan_dims[2], self.heigh_scan_dims[0], self.heigh_scan_dims[1]
        )
        heigh_embedded = self.critic_height_conv(heigh_scan)
        # conv height to mlp
        heigh_embedded = heigh_embedded.view(*batch_shape, -1)
        heigh_embedded = self.critic_height_embedder_mlp(heigh_embedded)

        # merge
        lidar_merged = torch.cat((lidar_embedded, lidar_extra_embedding, heigh_embedded), dim=-1)
        lidar_merged = self.critic_lidar_merger_mlp(lidar_merged)

        # memory
        if self.is_recurrent:
            memory_out = self.critic_memory(lidar_merged, masks, hidden_states)
        else:
            memory_out = self.critic_no_memory(lidar_merged)
        memory_processed = self.critic_memory_processor(memory_out.squeeze(0))

        # -- non-recurrent part
        non_lidar_out = self.critic_non_lidar_mlp(non_lidar_obs)
        if masks is not None:
            non_lidar_out = unpad_trajectories(non_lidar_out, masks)

        # output
        out = torch.cat((non_lidar_out, memory_processed), dim=-1)
        out = self.critic_out_mlp(out)
        return out

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

    def update_distribution(self, observations, masks=None, hidden_states=None):
        """Update the distribution of the policy"""
        # forward pass

        # logits = self.actor(observations)
        logits = self.actor_forward(observations, masks, hidden_states)

        alpha, beta = self.get_beta_parameters(logits)

        # Update distribution
        self.distribution = Beta(alpha, beta, validate_args=False)

    def act(self, observations, masks=None, hidden_states=None):
        self.update_distribution(observations, masks, hidden_states)
        action_sample = self.distribution.sample()
        return action_sample

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        # forward pass
        logits = self.actor_forward(observations)
        actions_mean = self.sigmoid(logits[:, : self.output_dim] + self.beta_initial_logit_shift)
        return actions_mean

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        # forward pass critic
        value = self.critic_forward(critic_observations, masks, hidden_states)
        return value

    def get_hidden_states(self):
        return self.actor_memory.hidden_states, self.critic_memory.hidden_states


class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0


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
