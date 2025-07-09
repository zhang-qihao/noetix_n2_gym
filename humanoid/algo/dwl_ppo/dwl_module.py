# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import torch.nn as nn
from humanoid.algo.ppo.memory import Memory
from humanoid.utils.utils import resolve_nn_activation


class DWLModule(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_embedding,
        dec_hidden_dims=[64],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        **kwargs,
    ):
        super().__init__()  

        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "DWLModule.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        activation = resolve_nn_activation(activation)
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        self.encoder = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        print(f"DWL_Encoder RNN: {self.encoder}")

        # Decoder
        dec_input_dim = num_embedding
        dec_layers = []
        for l in range(len(dec_hidden_dims)):
            dec_layers += [nn.Linear(dec_input_dim, dec_hidden_dims[l]), activation]
            dec_input_dim = dec_hidden_dims[l]
        dec_layers += [nn.Linear(dec_input_dim, num_critic_obs)]
        self.decoder = nn.Sequential(*dec_layers)

        print(f"DWL_Decoder MLP: {self.decoder}")




        
