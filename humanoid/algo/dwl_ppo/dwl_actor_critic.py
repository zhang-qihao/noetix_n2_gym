# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from humanoid.algo.ppo.actor_critic import ActorCritic
from humanoid.algo.dwl_ppo.dwl_module import DWLModule
from humanoid.utils.utils import resolve_nn_activation

class DWLActorCritic(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_embedding,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        enc_hidden_dims=[256],
        dec_hidden_dims=[64],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "DWLActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=num_embedding,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = resolve_nn_activation(activation)
        self.dwl_model = DWLModule(num_actor_obs, num_critic_obs, num_embedding, dec_hidden_dims=dec_hidden_dims, rnn_type=rnn_type, rnn_num_layers=rnn_num_layers)

        emb_layers = []
        emb_layers.append(nn.Linear(rnn_hidden_dim, enc_hidden_dims[0]))
        emb_layers.append(activation)
        for layer_index in range(len(enc_hidden_dims)):
            if layer_index == len(enc_hidden_dims) - 1:
                emb_layers.append(nn.Linear(enc_hidden_dims[layer_index], num_embedding))
            else:
                emb_layers.append(nn.Linear(enc_hidden_dims[layer_index], enc_hidden_dims[layer_index + 1]))
                emb_layers.append(activation)
        self.embodied = nn.Sequential(*emb_layers)

        print(f"Embedding MLP: {self.embodied}")


    def reset(self, dones=None):
        self.dwl_model.encoder.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        emb_input = self.dwl_model.encoder(observations, masks, hidden_states)
        actor_input = self.embodied(emb_input.squeeze(0))
        return super().act(actor_input)

    def act_inference(self, observations):
        emb_input = self.dwl_model.encoder(observations)
        actor_input = self.embodied(emb_input.squeeze(0))
        return super().act_inference(actor_input)

    def evaluate(self, critic_observations):
        return super().evaluate(critic_observations)

    def get_hidden_states(self):
        return self.dwl_model.encoder.hidden_states, None
    
    def compute_loss(self, observations, critic_obs, masks, hidden_states, lambda_reg=2e-3):
        emb_input = self.dwl_model.encoder(observations, masks, hidden_states)
        latent = self.embodied(emb_input.squeeze(0))
        pred_state = self.dwl_model.decoder(latent)

        estimation_loss = F.mse_loss(pred_state, critic_obs.detach())
        l1_reg_loss = torch.mean(torch.sum(torch.abs(latent), dim=-1))
        losses = estimation_loss + lambda_reg * l1_reg_loss

        return losses