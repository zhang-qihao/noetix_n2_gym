# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from humanoid.utils.utils import split_and_pad_trajectories


class RolloutStorage:
    """
    回放缓冲区类，用于存储和管理PPO算法训练过程中的轨迹数据
    支持强化学习和蒸馏训练两种模式
    """

    class Transition:
        """
        转换类，用于存储单个时间步的环境转换数据
        包含观测、动作、奖励、价值等信息
        """
        def __init__(self):
            # 观测值
            self.observations = None
            # 特权观测值（包含更多信息的观测）
            self.privileged_observations = None
            # 动作
            self.actions = None
            # 特权动作（用于蒸馏训练）
            self.privileged_actions = None
            # 奖励
            self.rewards = None
            # 是否结束标志
            self.dones = None
            # 状态价值
            self.values = None
            # 动作的对数概率
            self.actions_log_prob = None
            # 动作分布的均值
            self.action_mean = None
            # 动作分布的标准差
            self.action_sigma = None
            # 随机状态
            self.rnd_state = None

        def clear(self):
            """
            清空转换数据，重新初始化
            """
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        device="cpu",
    ):
        """
        初始化回FFER冲区
        
        Args:
            training_type: 训练类型 ("rl" 表示强化学习, "distillation" 表示蒸馏)
            num_envs: 并行环境数量
            num_transitions_per_env: 每个环境的转换数量
            obs_shape: 观测空间形状
            privileged_obs_shape: 特权观测空间形状
            actions_shape: 动作空间形状
            rnd_state_shape: 随机状态形状
            device: 计算设备
        """
        # 存储输入参数
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.rnd_state_shape = rnd_state_shape
        self.actions_shape = actions_shape

        # 核心数据存储
        # 观测值缓冲区 [时间步, 环境数, 观测维度]
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        # 特权观测值缓冲区（如果存在）
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        # 奖励缓冲区 [时间步, 环境数, 1]
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # 动作缓冲区 [时间步, 环境数, 动作维度]
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # 结束标志缓冲区 [时间步, 环境数, 1]
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # 蒸馏训练相关数据
        # if training_type == "distillation":
        #     self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # 强化学习相关数据
        if training_type == "rl":
            # 状态价值缓冲区
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            # 动作对数概率缓冲区
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            # 动作分布均值缓冲区
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            # 动作分布标准差缓冲区
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            # 回报缓冲区
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            # 优势函数缓冲区
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # 存储转换计数器
        self.step = 0

    def add_transitions(self, transition: Transition):
        """
        向缓冲区添加转换数据
        
        Args:
            transition: Transition对象，包含要添加的转换数据
        """
        # 检查缓冲区是否已满
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # 存储核心数据
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # 蒸馏训练数据存储
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # 强化学习数据存储
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # 增加计数器
        self.step += 1


    def clear(self):
        """
        清空缓冲区，重置计数器
        """
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        """
        使用广义优势估计(GAE)计算回报和优势函数
        
        Args:
            last_values: 最后状态的价值
            gamma: 折扣因子
            lam: GAE参数
            normalize_advantage: 是否标准化优势函数
        """
        advantage = 0
        # 反向遍历所有时间步计算回报和优势
        for step in reversed(range(self.num_transitions_per_env)):
            # 如果是最后一个时间步，使用bootstrap值
            if step == self.num_transitions_per_env - 1:
                next_values = last_values.to(self.device)
            else:
                next_values = self.values[step + 1]
            # 计算下一个状态是否为非终止状态 (1表示非终止，0表示终止)
            next_is_not_terminal = 1.0 - self.dones[step].float().to(self.device)
            next_is_not_terminal = next_is_not_terminal.to(self.device)
            # 计算TD误差: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # 计算优势函数: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # 计算回报: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # 计算优势函数
        self.advantages = self.returns - self.values
        # 如果需要则标准化优势函数，防止双重标准化
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    # 蒸馏训练的数据生成器
    def generator(self):
        """
        为蒸馏训练生成数据批次
        
        Yields:
            tuple: (观测, 特权观测, 动作, 特权动作, 结束标志)
        """
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            if self.privileged_observations is not None:
                privileged_observations = self.privileged_observations[i]
            else:
                privileged_observations = self.observations[i]
            yield self.observations[i], privileged_observations, self.actions[i], self.privileged_actions[
                i
            ], self.dones[i]

    # 前馈网络强化学习的小批次生成器
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """
        为前馈网络强化学习生成小批次数据
        
        Args:
            num_mini_batches: 小批次数量
            num_epochs: 训练轮数
            
        Yields:
            tuple: 包含观测、动作、价值、优势、回报等的小批次数据
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        # 生成随机索引
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # 核心数据展平处理
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # PPO相关数据展平处理
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # 选择小批次的索引
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # 创建小批次数据
                # -- 核心数据
                obs_batch = observations[batch_idx]
                privileged_observations_batch = privileged_observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- PPO相关数据
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # 产出小批次数据
                yield obs_batch, privileged_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None

    # 循环网络强化学习的小批次生成器
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """
        为循环网络强化学习生成小批次数据
        
        Args:
            num_mini_batches: 小批次数量
            num_epochs: 训练轮数
            
        Yields:
            tuple: 包含观测轨迹、动作、价值、优势、回报、隐藏状态等的小批次数据
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        # 对轨迹进行填充和分割处理
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_privileged_obs_trajectories = padded_obs_trajectories

        if self.rnd_state_shape is not None:
            padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.rnd_state, self.dones)
        else:
            padded_rnd_state_trajectories = None

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                # 计算轨迹批次大小
                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                # 创建批次数据
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # 重塑隐藏状态为[num_envs, time, num layers, hidden dim]格式
                # 然后只取结束标志后的时间步，取轨迹批次并重塑回[num_layers, batch, hidden_dim]格式
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # 移除GRU的元组格式
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                # 产出小批次数据
                yield obs_batch, privileged_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch

                first_traj = last_traj