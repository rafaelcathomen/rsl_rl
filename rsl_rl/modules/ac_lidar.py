from __future__ import annotations

import torch
import torch.nn as nn

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


class ActorCriticBetaRecurrentLidar(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        non_lidar_dim: int,
        lidar_dim: int,
        non_lidar_layer_dims: list[int],
        lidar_compress_layer_dims: list[int],
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

        if num_actor_obs != non_lidar_dim + lidar_dim or num_actor_obs != num_critic_obs:
            raise ValueError(
                f"num_actor_obs must be equal to non_lidar_dim + lidar_dim + lidar_extra_dim. num_actor_obs: {num_actor_obs}, non_lidar_dim: {non_lidar_dim}, lidar_dim: {lidar_dim}"
            )
        activation_module = get_activation(activation)

        self.lidar_dim = lidar_dim
        self.non_lidar_dim = non_lidar_dim
        ##
        # define networks
        ##

        # -- recurrence
        # - lidar embedding
        self.actor_lidar_embedder = create_mlp(lidar_dim, lidar_compress_layer_dims, activation_module)
        self.critic_lidar_embedder = create_mlp(lidar_dim, lidar_compress_layer_dims, activation_module)

        # - memory
        self.actor_memory = Memory(
            input_size=lidar_compress_layer_dims[-1], type="gru", num_layers=gru_layers, hidden_size=gru_dim
        )
        self.critic_memory = Memory(
            input_size=lidar_compress_layer_dims[-1], type="gru", num_layers=gru_layers, hidden_size=gru_dim
        )
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
        lidar_obs = x[..., self.non_lidar_dim :]

        # recurrent part
        lidar_embedded = self.actor_lidar_embedder(lidar_obs)
        memory_out = self.actor_memory(lidar_embedded, masks, hidden_states)
        memory_processed = self.actor_memory_processor(memory_out.squeeze(0))

        # non-recurrent part
        non_lidar_out = self.actor_non_lidar_mlp(non_lidar_obs)
        if masks is not None:
            non_lidar_out = unpad_trajectories(non_lidar_out, masks)

        # output
        out = torch.cat((non_lidar_out, memory_processed), dim=-1)
        out = self.actor_out_mlp(out)
        return out

    def critic_forward(self, x, masks=None, hidden_states=None):
        non_lidar_obs = x[..., : self.non_lidar_dim]
        lidar_obs = x[..., self.non_lidar_dim :]

        # recurrent part
        lidar_embedded = self.critic_lidar_embedder(lidar_obs)
        memory_out = self.critic_memory(lidar_embedded, masks, hidden_states)
        memory_processed = self.critic_memory_processor(memory_out.squeeze(0))

        # non-recurrent part
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
        return self.distribution.sample()

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
