# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain


from humanoid.algo.ppo.rnd import RandomNetworkDistillation
from humanoid.algo.amp_him_ppo.him_rollout_storage import HIMRolloutStorage
from humanoid.algo.amp_him_ppo.him_actor_critic import HIMActorCritic
from humanoid.utils.utils import string_to_callable
from humanoid.amp_utils.discriminator import Discriminator
from humanoid.amp_utils.replay_buffer import ReplayBuffer


class AMP_HIM_PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: HIMActorCritic
    discriminator: Discriminator
    """The actor critic module."""

    def __init__(
        self,
        policy,
        discriminator,
        amp_expert_data,
        amp_state_normalizer,
        amp_style_reward_normalizer,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        discriminator_learning_rate = 1e-3,
        discriminator_momentum=0.9,
        discriminator_weight_decay=0.0005,
        discriminator_gradient_penalty_coef=5,
        discriminator_logit_reg_coef = 0.01,
        discriminator_weight_decay_coef = 0.0001,
        discriminator_num_mini_batches=10,
        amp_replay_buffer_size=100000,
        discriminator_loss_function="MSELoss",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                self.data_augmentation_func = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(self.data_augmentation_func):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: HIMRolloutStorage = None  # type: ignore
        self.transition = HIMRolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # Discriminator components
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_policy_data = ReplayBuffer(discriminator.observation_dim, discriminator.observation_horizon, amp_replay_buffer_size, device)
        self.amp_expert_data = amp_expert_data
        self.amp_state_normalizer = amp_state_normalizer
        self.amp_style_reward_normalizer = amp_style_reward_normalizer

        # Discriminator parameters
        self.discriminator_learning_rate = discriminator_learning_rate
        self.discriminator_momentum = discriminator_momentum
        self.discriminator_weight_decay = discriminator_weight_decay
        self.discriminator_gradient_penalty_coef = discriminator_gradient_penalty_coef
        self.discriminator_logit_reg_coef = discriminator_logit_reg_coef
        self.discriminator_weight_decay_coef = discriminator_weight_decay_coef
        self.discriminator_loss_function = discriminator_loss_function
        self.discriminator_num_mini_batches = discriminator_num_mini_batches

        discriminator_optimizer = optim.Adam
        self.discriminator_optimizer = discriminator_optimizer(self.discriminator.parameters(), lr=self.discriminator_learning_rate)

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = HIMRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs):
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_critic_obs, amp_observation_buf):
        # Record the next critic obs
        self.transition.next_privileged_observations = next_critic_obs.clone()
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.amp_policy_data.insert(amp_observation_buf)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_estimate_loss = 0
        mean_swap_loss = 0

        mean_discriminator_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            next_critic_obs_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = self.data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = self.data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch)
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            
            # Estimator Update
            estimation_loss, swap_loss = self.policy.estimator.update(obs_batch, critic_obs_batch, next_critic_obs_batch, lr=self.learning_rate)

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # obtain the origin actions
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    symm_obs_batch, _ = self.data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    symm_critic_obs_batch, _ = self.data_augmentation_func(
                        obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                actions_mean_symm_batch = self.policy.act_inference(symm_obs_batch.detach().clone())

                # values predicted by the critic for symmetrically-augmented privileged observations
                values_symm_batch = self.policy.evaluate(symm_critic_obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                _, actions_mean_symm_batch = self.data_augmentation_func(
                    obs=None, actions=actions_mean_symm_batch, env=self.symmetry["_env"], obs_type="policy"
                )

                # compute the loss 
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(mean_actions_batch, actions_mean_symm_batch.detach()) 
                symmetry_loss += mse_loss(value_batch, values_symm_batch.detach()) 
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_estimate_loss += estimation_loss
            mean_swap_loss += swap_loss
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
        
        # Discriminator update
        amp_policy_generator = self.amp_policy_data.feed_forward_generator(
            self.discriminator_num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.discriminator_num_mini_batches)
        amp_expert_generator = self.amp_expert_data.feed_forward_generator(
            self.discriminator_num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.discriminator_num_mini_batches)

        # Disceiminator update
        for sample_amp_policy, sample_expert_data in zip(amp_policy_generator, amp_expert_generator):
            sample_amp_expert, _ = sample_expert_data[0], sample_expert_data[1]
            # Discriminator loss
            policy_state_buf = torch.zeros_like(sample_amp_policy)
            expert_state_buf = torch.zeros_like(sample_amp_expert)
            if self.amp_state_normalizer is not None:
                for i in range(self.discriminator.observation_horizon):
                    with torch.no_grad():
                        policy_state_buf[:, i] = self.amp_state_normalizer.normalize(sample_amp_policy[:, i])
                        expert_state_buf[:, i] = self.amp_state_normalizer.normalize(sample_amp_expert[:, i])

            policy_data = policy_state_buf.flatten(1, 2)
            expert_data = expert_state_buf.flatten(1, 2)
            policy_data.requires_grad = True
            expert_data.requires_grad = True

            policy_d = self.discriminator(policy_state_buf.flatten(1, 2))
            expert_d = self.discriminator(expert_state_buf.flatten(1, 2))

            if self.discriminator_loss_function == "BCEWithLogitsLoss":
                expert_loss = torch.nn.BCEWithLogitsLoss()(expert_d, torch.ones_like(expert_d))
                policy_loss = torch.nn.BCEWithLogitsLoss()(policy_d, torch.zeros_like(policy_d))
                grad_pen_loss = self.discriminator.compute_grad_pen(expert_data,
                                                                lambda_=self.discriminator_gradient_penalty_coef)
            elif self.discriminator_loss_function == "MSELoss":
                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                grad_pen_loss = self.discriminator.compute_grad_pen(expert_data,
                                                                lambda_=self.discriminator_gradient_penalty_coef)
            elif self.discriminator_loss_function == "WassersteinLoss":
                expert_loss = -expert_d.mean()
                policy_loss = policy_d.mean()
                
                grad_pen_loss = self.discriminator.compute_wgan_div_grad_pen(expert_data, policy_data)
            else:
                raise ValueError("Unexpected loss function specified")
            
            amp_loss = 0.5 * (expert_loss + policy_loss)

            # logit reg
            disc_logit_loss = self.discriminator.compute_logit_reg(lambda_=self.discriminator_logit_reg_coef)

            # Weight Decay Loss
            weight_decay_loss = self.discriminator.compute_weight_decay(lambda_ = self.discriminator_weight_decay_coef)
            
            # Gradient step
            discriminator_loss = amp_loss + grad_pen_loss + disc_logit_loss + weight_decay_loss

            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            if self.amp_state_normalizer is not None:
                self.amp_state_normalizer.update(sample_amp_policy[:, 0])
                self.amp_state_normalizer.update(sample_amp_expert[:, 0])

            mean_discriminator_loss += discriminator_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_estimate_loss /= num_updates
        mean_swap_loss /= num_updates
        # -- For AMP
        discriminator_num_updates = self.discriminator_num_mini_batches
        mean_discriminator_loss /= discriminator_num_updates
        mean_policy_pred /= discriminator_num_updates
        mean_expert_pred /= discriminator_num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimate": mean_estimate_loss,
            "swap": mean_swap_loss,
            "amp_loss": mean_discriminator_loss,
            "expert_pred": mean_expert_pred,
            "policy_pred": mean_policy_pred
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel