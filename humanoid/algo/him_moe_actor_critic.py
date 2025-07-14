# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from humanoid.utils.utils import resolve_nn_activation
from .him_estimator import HIMEstimator

class SharedMoELoss:
    def __init__(self, load_balance_weight=0.01, diversity_weight=0.01):
        self.load_balance_weight = load_balance_weight
        self.diversity_weight = diversity_weight
    
    def compute_load_balance_loss(self, gate_weights):
        """负载平衡损失"""
        expert_usage = gate_weights.mean(dim=0)
        uniform_usage = torch.ones_like(expert_usage) / expert_usage.size(0)
        return torch.sum((expert_usage - uniform_usage) ** 2)
    
    def compute_diversity_loss(self, gate_weights):
        """专家多样性损失"""
        # 鼓励不同样本使用不同专家
        correlation_matrix = torch.corrcoef(gate_weights.T)
        # 减去对角线元素（自相关）
        mask = torch.eye(correlation_matrix.size(0), device=correlation_matrix.device)
        off_diagonal = correlation_matrix * (1 - mask)
        return torch.sum(off_diagonal ** 2)
    

    def compute_total_loss(self, actor_loss, critic_loss, gate_weights):
        load_balance_loss = self.compute_load_balance_loss(gate_weights)
        diversity_loss = self.compute_diversity_loss(gate_weights)
        
        total_loss = (actor_loss + critic_loss + 
                     self.load_balance_weight * load_balance_loss +
                     self.diversity_weight * diversity_loss)
        
        return total_loss, {
            'load_balance_loss': load_balance_loss.item(),
            'diversity_loss': diversity_loss.item()
        }

class TaskConditionedSharedMoE(nn.Module):
    def __init__(self, num_obs, num_privileged_obs, num_actions, num_experts=4, 
                 hidden_dims=[256, 256, 256], activation='elu', task_embedding_dim=64):
        super().__init__()
        
        self.num_experts = num_experts
        self.task_embedding_dim = task_embedding_dim

        task_input_dim = (num_obs-19) * 5  
        self.task_embedding = nn.Sequential( # 输入138*5 = 690
            nn.Linear(task_input_dim, task_embedding_dim), # num_obs 157 num_privileged_obs 234
            resolve_nn_activation(activation)
        )
        
        self.shared_gate_network = nn.Sequential(
            nn.Linear(task_embedding_dim, 256),
            resolve_nn_activation(activation),
            nn.Linear(256, num_experts),
            nn.Softmax(dim=-1)
        )
    
        self.actor_experts = nn.ModuleList([
            self._create_actor_expert(num_obs, num_actions, hidden_dims, activation)
            for _ in range(num_experts)
        ])
        
        self.critic_experts = nn.ModuleList([
            self._create_critic_expert(num_privileged_obs, hidden_dims, activation)
            for _ in range(num_experts)
        ])

        self.register_buffer('expert_usage_count', torch.zeros(num_experts))
        
    def _create_actor_expert(self, num_obs, num_actions, hidden_dims, activation):
        return nn.Sequential(
            nn.Linear(num_obs, hidden_dims[0]),
            resolve_nn_activation(activation),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            resolve_nn_activation(activation),
            nn.Linear(hidden_dims[1], num_actions)
        )
    
    def _create_critic_expert(self, num_privileged_obs, hidden_dims, activation):
        return nn.Sequential(
            nn.Linear(num_privileged_obs, hidden_dims[0]),
            resolve_nn_activation(activation),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            resolve_nn_activation(activation),
            nn.Linear(hidden_dims[1], 1)
        )
    
    def forward(self, observations, privileged_obs=None):
        batch_size = observations.shape[0]

        task_embed = self.task_embedding(observations)

        gate_weights = self.shared_gate_network(task_embed)
        
        with torch.no_grad():
            expert_selection = torch.argmax(gate_weights, dim=1)
            for i in range(self.num_experts):
                self.expert_usage_count[i] += (expert_selection == i).float().sum()
        
        actor_outputs = self._forward_actor(observations, gate_weights)
    
        critic_input = privileged_obs if privileged_obs is not None else observations
        critic_outputs = self._forward_critic(critic_input, gate_weights)
        
        return actor_outputs, critic_outputs, gate_weights
    
    def _forward_actor(self, observations, gate_weights):
        expert_outputs = torch.stack([expert(observations) for expert in self.actor_experts], dim=1)
        return torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)
    
    def _forward_critic(self, critic_input, gate_weights):
        expert_outputs = torch.stack([expert(critic_input) for expert in self.critic_experts], dim=1)
        return torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)
    
class HIMMOEActorCritic(nn.Module):
    is_recurrent = False
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_one_step_obs,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.is_moed = True
        self.is_distillation = False
        self.history_size = int(num_actor_obs/num_one_step_obs)
        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.num_one_step_obs = num_one_step_obs

        # mlp_input_dim_a = num_actor_obs + 3 + 16
        mlp_input_dim_a = num_one_step_obs + 3 + 16 # 138 + 3 + 16 = 157
        mlp_input_dim_c = num_critic_obs


        # Estimator
        self.estimator = HIMEstimator(temporal_steps=self.history_size, num_one_step_obs=num_one_step_obs, num_one_step_priveleged_obs=num_critic_obs)

        if actor_hidden_dims != critic_hidden_dims:
            raise ValueError(
                "Actor and critic hidden dimensions must be the same for MoE actor-critic."
            )

        self.moe_loss = SharedMoELoss(load_balance_weight=0.01, diversity_weight=0.01)

        self.moe = TaskConditionedSharedMoE(
            num_obs=mlp_input_dim_a, # num_obs 157 num_privileged_obs 234
            num_privileged_obs=mlp_input_dim_c, # 150
            num_actions=num_actions,
            num_experts=5,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            task_embedding_dim=64
        )

        self.actor = self.moe.actor_experts
        self.actor.forward = self.moe._forward_actor
        # self.gate = self.moe.shared_gate_network
        self.critic = self.moe.critic_experts
        self.critic.forward = self.moe._forward_critic
        self.task_embedding = self.moe.task_embedding
        self.shared_gate_network = self.moe.shared_gate_network
        

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f'Estimator: {self.estimator.encoder}')
        print(f'MoE Gate: {self.shared_gate_network}')

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

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
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        with torch.no_grad():
            vel, latent = self.estimator(observations)
        # actor_input = torch.cat((observations, vel, latent), dim=-1)
        actor_input = torch.cat((observations[:, -self.num_one_step_obs:], vel, latent), dim=-1)
        # compute mean
        task_embed = self.task_embedding(observations) #observations 138*5 = 690
        gate_weights = self.shared_gate_network(task_embed)
        mean = self.actor(actor_input,gate_weights)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        vel, latent = self.estimator(observations)
        # actions_mean = self.actor(torch.cat((observations, vel, latent), dim=-1))
        task_embed = self.task_embedding(observations)
        gate_weights = self.shared_gate_network(task_embed)
        actions_mean = self.actor(torch.cat((observations[:, -self.num_one_step_obs:], vel, latent), dim=-1),gate_weights)
        return actions_mean

    def evaluate(self, critic_observations,observations = None ,**kwargs):
        task_embed = self.task_embedding(observations)
        gate_weights = self.shared_gate_network(task_embed)
        value = self.critic(critic_observations,gate_weights)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        gate_state_dict = {k.replace("shared_gate_network.", ""): v for k, v in state_dict.items() 
            if k.startswith("shared_gate_network.")
        }
        if gate_state_dict:
            self.shared_gate_network.load_state_dict(gate_state_dict)
        else:
            raise ValueError("No shared_gate_network parameters found in checkpoint")
        task_embedding_state_dict = {k.replace("task_embedding.", ""): v for k, v in state_dict.items() 
            if k.startswith("task_embedding.")
        }
        if task_embedding_state_dict:
            self.task_embedding.load_state_dict(task_embedding_state_dict)
        else:
            raise ValueError("No task_embedding parameters found in checkpoint")
        return True