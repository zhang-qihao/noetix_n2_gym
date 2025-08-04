from humanoid.envs.n2.n2_env import N2Env

import time
from isaacgym.torch_utils import *
from isaacgym import gymtorch
import torch

from humanoid.utils.helpers import class_to_dict

from humanoid.envs.n2.n2_mimic_config import N2MimicCfg
from humanoid.utils.isaacgym_utils import get_euler_xyz_tensor
from humanoid.amp_utils.motion_loader import *


class N2MimicEnv(N2Env):
    def _create_envs(self):
        super()._create_envs()
        key_names = []
        for name in self.cfg.asset.key_name:
            key_names.extend([s for s in self.body_names if name in s])
        self.key_indices = torch.zeros(len(key_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(key_names)):
            self.key_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], key_names[i])
        
        upper_body_names = []
        for name in self.cfg.asset.upper_body_name:
            upper_body_names.extend([s for s in self.body_names if name in s])
        self.upper_body_indices = torch.zeros(len(upper_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(upper_body_names)):
            self.upper_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], upper_body_names[i])
    
        lower_body_names = []
        for name in self.cfg.asset.lower_body_name:
            lower_body_names.extend([s for s in self.body_names if name in s])
        self.lower_body_indices = torch.zeros(len(lower_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(lower_body_names)):
            self.lower_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], lower_body_names[i])

    def _get_noise_scale_vec(self, cfg):
        if self.cfg.env.frame_stack is not None:
            noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        else:
            noise_vec = torch.zeros_like(self.obs_buf[0])

        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:2] = 0.  # commands
        noise_vec[2:5] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[5:8] = noise_scales.gravity * noise_level
        noise_vec[8:8+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[8+self.num_actions:8+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[8+2*self.num_actions:8+3*self.num_actions] = 0. # previous actions

        return noise_vec

    def update_key_pos_state(self):
        self.key_pos_state = self.rigid_body_states_view[:, self.key_indices, :]
        self.key_pos = self.key_pos_state[:, :, :3]

    def update_body_pos_state(self):
        self.upper_body_pos_state = self.rigid_body_states_view[:, self.upper_body_indices, :]
        self.upper_body_pos = self.upper_body_pos_state[:, :, :3]

        self.lower_body_pos_state = self.rigid_body_states_view[:, self.lower_body_indices, :]
        self.lower_body_pos = self.lower_body_pos_state[:, :, :3]

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.update_key_pos_state()
        self.update_body_pos_state()

    def _init_buffers(self):
        super()._init_buffers()
        self.cfg: N2MimicCfg
        self.estimator = None # assigned in runner
        self._init_adaptive_sigma()
        self.contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        self.contacts_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False) 
        # termination
        if self.cfg.termination.terminate_when_motion_far and self.cfg.termination.termination_curriculum.terminate_when_motion_far_curriculum:
            self.terminate_when_motion_far_threshold = self.cfg.termination.termination_curriculum.terminate_when_motion_far_initial_threshold
        else:
            self.terminate_when_motion_far_threshold = self.cfg.termination.scales.termination_motion_far_threshold
        # get key pos state
        self.update_key_pos_state()
        # for reward penalty curriculum
        self.average_episode_length = 0. # num_compute_average_epl last termination episode length
        self.last_episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.num_compute_average_epl = self.cfg.rewards.num_compute_average_epl
        # load motion components
        self.reference_motion_file = self.cfg.motion_loader.reference_motion_file
        self.reference_observation_horizon = self.cfg.motion_loader.reference_observation_horizon
        self.num_preload_transitions = self.cfg.motion_loader.num_preload_transitions
        self.motion_loader_class = eval(self.cfg.motion_loader.motion_loader_name)
        self.motion_loader: MotionLoaderNing = self.motion_loader_class(
            device=self.device,
            time_between_frames=self.dt,
            reference_observation_horizon=self.reference_observation_horizon,
            num_preload_transitions=self.num_preload_transitions,
            motion_files=self.reference_motion_file
        )
        self.motion_lenth = self.motion_loader.trajectory_lens[0]
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)

        self.compute_ref_state()
    
    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        if self.cfg.rewards.use_vec_reward:
            num_rew_fn = len(self.reward_functions)
            self.rew_buf = torch.zeros(self.num_envs, num_rew_fn, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.use_reward_penalty_curriculum = self.cfg.rewards.reward_penalty_curriculum
        if self.use_reward_penalty_curriculum:
            self.reward_penalty_scale = self.cfg.rewards.reward_initial_penalty_scale
    
    @property
    def num_rew_fn(self):
        return len(self.reward_functions) if self.cfg.rewards.use_vec_reward else 1
    
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # dynamic randomization
        # if not self.cfg.env.test:
        # delay = torch.rand((self.num_envs, 1), device=self.device) * 0.5
        # # delay = 0.2
        # actions = (1 - delay) * actions + delay * self.actions
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        termination_ids, termination_priveleged_obs, _ = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, \
            termination_priveleged_obs, None
    
    def  _get_phase(self):
        current_time = self.episode_length_buf * self.dt + self.motion_start_times
        phase = (current_time / self.motion_lenth) % 1.
        return phase
    
    def compute_ref_state(self):
        phase = self._get_phase()
        motion_times = (phase * self.motion_lenth).clamp(min=0.0, max=self.motion_lenth-self.dt).cpu().numpy()
        frames = self.motion_loader.get_full_frame_at_time_batch(np.array([0] * self.num_envs), motion_times)

        ref_lin_vel = self.motion_loader_class.get_linear_vel_batch(frames)
        self.ref_lin_vel = quat_rotate_inverse(self.base_quat, ref_lin_vel)

        ref_ang_vel = self.motion_loader_class.get_angular_vel_batch(frames)
        self.ref_ang_vel = quat_rotate_inverse(self.base_quat, ref_ang_vel)

        self.ref_dof_pos = self.motion_loader_class.get_joint_pose_batch(frames)
        self.ref_dof_vel = self.motion_loader_class.get_joint_vel_batch(frames)
        self.ref_key_pos = self.motion_loader_class.get_tar_toe_pos_local_batch(frames)
        self.ref_contact_mask = self.motion_loader_class.get_contact_mask_batch(frames)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.last_episode_length_buf = self.episode_length_buf.clone()

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.contacts = self.contact_forces[:, self.feet_indices, 2] > 5.
        self.contacts_filt = torch.logical_or(self.contacts, self.last_contacts).float()

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.privileged_obs_buf[env_ids].clone()
        self.reset_idx(env_ids)
        
        if self.cfg.domain_rand.push_robots:
            self._push_robots()
        
        if self.cfg.domain_rand.disturbance:
            self._disturbance_robots()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_contacts[:] = self.contacts[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        return env_ids, termination_privileged_obs, None
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 5., dim=1)
        time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= time_out_buf
        if self.cfg.termination.terminate_by_gravity:
            reset_terminate_by_gravity = torch.norm(self.projected_gravity[:, 0:2], dim=-1) > self.cfg.termination.scales.termination_gravity
            self.reset_buf |= reset_terminate_by_gravity
        if self.cfg.termination.terminate_by_fallen:
            fallen_buf = (self.root_states[:, 2] - self.terrain_h) < self.cfg.termination.scales.termination_height
            self.reset_buf |= fallen_buf
        if self.cfg.termination.terminate_when_motion_far:
            root_pos = self.root_states[:, :3].clone() 
            key_pos = (self.key_pos - root_pos.unsqueeze(1))
            ref_key_pos = self.ref_key_pos.reshape(self.num_envs, -1, 3)
            reset_buf_motion_far = torch.any(torch.norm(ref_key_pos - key_pos, dim=-1) > self.terminate_when_motion_far_threshold, dim=-1)
            self.reset_buf_terminate_by_motion_far = reset_buf_motion_far
            self.reset_buf |= reset_buf_motion_far
        if self.cfg.termination.terminate_when_motion_end:
            current_time = (self.episode_length_buf) * self.dt + self.motion_start_times
            self.reset_buf_terminate_by_motion_end = current_time > self.motion_lenth
            self.time_out_buf |= self.reset_buf_terminate_by_motion_end
    
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        if self.cfg.env.frame_stack is not None:
            # clear observation history
            for i in range(self.obs_history.maxlen):
                self.obs_history[i][env_ids] *= 0
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.use_reward_penalty_curriculum:
            self._update_reward_penalty_curriculum()
        if self.cfg.termination.terminate_when_motion_far and self.cfg.termination.termination_curriculum.terminate_when_motion_far_curriculum:
            self._update_terminate_when_motion_far_curriculum()
        
        self._resample_motion_times(env_ids)
        # reset robot states
        self._reset_dofs_motion(env_ids, self.motion_start_times[env_ids])
        self._reset_root_states_motion(env_ids, self.motion_start_times[env_ids])

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.feet_contact_time[env_ids] = 0.
        self.feet_both_contact_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.episode_length_buf[env_ids] = 0.
        self._update_average_episode_length(env_ids)

        # update height measurements
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        self.terrain_h = self._get_ground_heights()

        # reset randomized prop
        if self.cfg.domain_rand.randomize_gains:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.p_gain_range[0], self.cfg.domain_rand.p_gain_range[1], (len(env_ids), self.num_actions), device=self.device)
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.d_gain_range[0], self.cfg.domain_rand.d_gain_range[1], (len(env_ids), self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), self.num_actions), device=self.device)

        self._refresh_actor_rigid_shape_props(env_ids)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["min_command_x"] = self.command_ranges["lin_vel_x"][0]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        self.extras["episode"]["end_epis_length"] = self.last_episode_length_buf[env_ids]
        
        # fix reset gravity bug
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
    
    def _resample_motion_times(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.cfg.env.test:
            self.motion_start_times[env_ids] = 0.0
        else:
            self.motion_start_times[env_ids] = torch_rand_float(0.0, self.motion_lenth - self.dt,(len(env_ids), 1), device=self.device).squeeze(-1)
    
    def _update_average_episode_length(self, env_ids):
        num = len(env_ids)
        current_average_episode_length = torch.mean(self.last_episode_length_buf[env_ids], dtype=torch.float)
        self.average_episode_length = self.average_episode_length * (1 - num / self.num_compute_average_epl) + current_average_episode_length * (num / self.num_compute_average_epl)

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if name in self.cfg.rewards.reward_penalty_reward_names:
                if self.cfg.rewards.reward_penalty_curriculum:
                    rew *= self.reward_penalty_scale
            if self.cfg.rewards.use_vec_reward:
                self.rew_buf[:,i] += rew
            else:
                self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def _update_reward_penalty_curriculum(self):
        """
        Update the penalty curriculum based on the average episode length.

        If the average episode length is below the penalty level down threshold,
        decrease the penalty scale by a certain level degree.
        If the average episode length is above the penalty level up threshold,
        increase the penalty scale by a certain level degree.
        Clip the penalty scale within the specified range.

        Returns:
            None
        """
        if self.average_episode_length < self.cfg.rewards.reward_penalty_level_down_threshold:
            self.reward_penalty_scale *= (1 - self.cfg.rewards.reward_penalty_degree)
        elif self.average_episode_length > self.cfg.rewards.reward_penalty_level_up_threshold:
            self.reward_penalty_scale *= (1 + self.cfg.rewards.reward_penalty_degree)

        self.reward_penalty_scale = np.clip(self.reward_penalty_scale, self.cfg.rewards.reward_min_penalty_scale, self.cfg.rewards.reward_max_penalty_scale)
    
    def _update_terminate_when_motion_far_curriculum(self):
        assert self.cfg.termination.terminate_when_motion_far and self.cfg.termination.termination_curriculum.terminate_when_motion_far_curriculum
        if self.average_episode_length < self.cfg.termination.termination_curriculum.terminate_when_motion_far_curriculum_level_down_threshold:
            self.terminate_when_motion_far_threshold *= (1 + self.cfg.termination.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        elif self.average_episode_length > self.cfg.termination.termination_curriculum.terminate_when_motion_far_curriculum_level_up_threshold:
            self.terminate_when_motion_far_threshold *= (1 - self.cfg.termination.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        self.terminate_when_motion_far_threshold = np.clip(self.terminate_when_motion_far_threshold, 
                                                         self.cfg.termination.termination_curriculum.terminate_when_motion_far_threshold_min, 
                                                         self.cfg.termination.termination_curriculum.terminate_when_motion_far_threshold_max)
    
    def compute_observations(self):
        """ Computes observations
        """
        phase = self._get_phase().unsqueeze(1)
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase)
        cos_pos = torch.cos(2 * torch.pi * phase)

        obs_buf = torch.cat((   sin_pos,                                        
                                cos_pos,    
                                self.base_ang_vel  * self.obs_scales.ang_vel,
                                self.projected_gravity,  
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                self.dof_vel * self.obs_scales.dof_vel,
                                self.actions
                            ),dim=-1)
        
        root_pos = self.root_states[:, :3].clone() 
        key_pos = (self.key_pos - root_pos.unsqueeze(1)).view(self.num_envs, -1) 
        ref_key_pos = self.ref_key_pos.clone()
        self.privileged_obs_buf = torch.cat((  
                                    sin_pos,                                        
                                    cos_pos, 
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,  
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.payload * 0.5,
                                    self.friction_coeffs,
                                    self.restitution_coeffs,
                                    self.Kp_factors,
                                    self.Kd_factors,
                                    self.motor_strength,
                                    key_pos,
                                    ref_key_pos,
                                    self.contacts_filt
                                ),dim=-1)
        
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        
        if self.cfg.env.frame_stack is not None:
            self.obs_history.append(obs_now)
            obs_buf_all = torch.stack([self.obs_history[i]
                                    for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K
            self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        else:
            self.obs_buf = obs_now

# ================================================ Sigma ================================================== #
    def _init_adaptive_sigma(self):
        if not self.cfg.rewards.enable_adaptive_tracking_sigma:
            self._update_adaptive_sigma = lambda *args, **kwargs: None
            return
        self._reward_error_ema = dict()
        for key, value in self.cfg.rewards.reward_tracking_sigma.items():
            self._reward_error_ema[key] = value
    
    def _update_adaptive_sigma(self, error:torch.Tensor, term:str):
        alpha = self.cfg.rewards.tracking_sigma_alpha
        scale = self.cfg.rewards.tracking_sigma_scale 
        adptype = self.cfg.rewards.tracking_sigma_type
        
        self._reward_error_ema[term] = self._reward_error_ema[term] * (1-alpha) + error.mean().item() * alpha
        
        if adptype == "scale":
            self.cfg.rewards.reward_tracking_sigma[term] = min(self._reward_error_ema[term] * scale, 
                                                                self.cfg.rewards.reward_tracking_sigma[term])
        elif adptype == "mean":
            self.cfg.rewards.reward_tracking_sigma[term] = (min(self._reward_error_ema[term], 
                                                                self.cfg.rewards.reward_tracking_sigma[term]) +
                                                                self._reward_error_ema[term]) / 2
        elif adptype == "origin":
            self.cfg.rewards.reward_tracking_sigma[term] = min(self._reward_error_ema[term], 
                                                                self.cfg.rewards.reward_tracking_sigma[term])
    
    def _reset_dofs_motion(self, env_ids, motion_times):
        traj_idxs = np.array([0] * motion_times.shape[0])
        times = motion_times.clone().cpu().numpy()
        frames = self.motion_loader.get_full_frame_at_time_batch(traj_idxs, times)

        self.dof_pos[env_ids] = self.motion_loader_class.get_joint_pose_batch(frames)
        self.dof_vel[env_ids] = self.motion_loader_class.get_joint_vel_batch(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_motion(self, env_ids, motion_times):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        traj_idxs = np.array([0] * motion_times.shape[0])
        times = motion_times.clone().cpu().numpy()
        frames = self.motion_loader.get_full_frame_at_time_batch(traj_idxs, times)

        # base position
        root_pos = self.motion_loader_class.get_root_pos_batch(frames)
        root_pos[:, :3] = root_pos[:, :3] + self.env_origins[env_ids, :3] + 0.02
        self.root_states[env_ids, :3] = root_pos
        root_orn = self.motion_loader_class.get_root_rot_batch(frames)
        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = self.motion_loader_class.get_linear_vel_batch(frames)
        self.root_states[env_ids, 10:13] = self.motion_loader_class.get_angular_vel_batch(frames)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
# ================================================ Rewards ================================================== #
    def _reward_tracking_body_pos(self):
        root_pos = self.root_states[:, :3].clone() 
        key_pos = (self.key_pos - root_pos.unsqueeze(1))
        ref_key_pos = self.ref_key_pos.reshape(self.num_envs, -1, 3)
        
        upper_body_diff = ref_key_pos[:, self.upper_body_indices, :] - key_pos[:, self.upper_body_indices, :]
        lower_body_diff = ref_key_pos[:, self.lower_body_indices, :] - key_pos[:, self.lower_body_indices, :]

        diff_body_pos_dist_upper = (upper_body_diff**2).mean(dim=-1).mean(dim=-1)
        diff_body_pos_dist_lower = (lower_body_diff**2).mean(dim=-1).mean(dim=-1)

        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.cfg.rewards.reward_tracking_sigma["tracking_upper_body_pos"])
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self.cfg.rewards.reward_tracking_sigma["tracking_lower_body_pos"])
        rew = r_body_pos_lower + r_body_pos_upper 
    
        self._update_adaptive_sigma(diff_body_pos_dist_upper, 'tracking_upper_body_pos')
        self._update_adaptive_sigma(diff_body_pos_dist_lower, 'tracking_lower_body_pos')
        return rew
    
    def _reward_tracking_feet_pos(self):
        root_pos = self.root_states[:, :3].clone() 
        key_pos = (self.key_pos - root_pos.unsqueeze(1))
        ref_key_pos = self.ref_key_pos.reshape(self.num_envs, -1, 3)
        
        feet_diff = ref_key_pos[:, self.feet_indices, :] - key_pos[:, self.feet_indices, :]
        feet_dist = (feet_diff**2).mean(dim=-1).mean(dim=-1)
        rew = torch.exp(-feet_dist / self.cfg.rewards.reward_tracking_sigma["tracking_feet_pos"])
        
        self._update_adaptive_sigma(feet_dist, 'tracking_feet_pos')
        return rew
    
    def _reward_tracking_body_vel(self):
        body_vel = self.base_lin_vel.clone()
        body_vel_target = self.ref_lin_vel.clone()
        diff = ((body_vel - body_vel_target) ** 2).mean(dim=-1)
        rew = torch.exp(-diff / self.cfg.rewards.reward_tracking_sigma["tracking_body_vel"])
        self._update_adaptive_sigma(diff, 'tracking_body_vel')
        return rew
    
    def _reward_tracking_body_ang_vel(self):
        body_ang_vel = self.base_ang_vel.clone()
        body_ang_vel_target = self.ref_ang_vel.clone()
        diff = ((body_ang_vel - body_ang_vel_target) ** 2).mean(dim=-1)
        rew = torch.exp(-diff / self.cfg.rewards.reward_tracking_sigma["tracking_body_ang_vel"]) 
        self._update_adaptive_sigma(diff, 'tracking_body_ang_vel')
        return rew
    
    def _reward_tracking_joint_pos(self):
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        joint_pos[:, self.ankle_dof_idxs] = 0.0
        pos_target[:, self.ankle_dof_idxs] = 0.0
        diff = ((joint_pos - pos_target) ** 2).mean(dim=-1)
        rew = torch.exp(-diff / self.cfg.rewards.reward_tracking_sigma["tracking_joint_pos"])
        self._update_adaptive_sigma(diff, 'tracking_joint_pos')
        return rew
    
    def _reward_tracking_joint_vel(self):
        joint_vel = self.dof_vel.clone()
        vel_target = self.ref_dof_vel.clone()
        joint_vel[:, self.ankle_dof_idxs] = 0.0
        vel_target[:, self.ankle_dof_idxs] = 0.0
        diff = ((joint_vel - vel_target) ** 2).mean(dim=-1)
        rew = torch.exp(-diff / self.cfg.rewards.reward_tracking_sigma["tracking_joint_vel"])
        self._update_adaptive_sigma(diff, 'tracking_joint_vel')
        return rew
    
    def _reward_tracking_max_joint_pos(self):
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        joint_pos[:, self.ankle_dof_idxs] = 0.0
        pos_target[:, self.ankle_dof_idxs] = 0.0
        max_diff_joint_pos = ((joint_pos - pos_target).abs()).max(dim=-1)[0]
        r_max_joint_pos = torch.exp(-max_diff_joint_pos / self.cfg.rewards.reward_tracking_sigma["tracking_max_joint_pos"])
        
        self._update_adaptive_sigma(max_diff_joint_pos, 'tracking_max_joint_pos')
        return r_max_joint_pos
    
    def _reward_tracking_contact_mask(self):
        cur_contact_mask = self.contacts_filt
        ref_contact_mask = self.ref_contact_mask
        
        error_contact_mask = (cur_contact_mask - ref_contact_mask).abs()

        rew = 1 - error_contact_mask.mean(dim=-1)
        return rew
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact_filt = torch.logical_or(self.contacts, self.last_contacts) 
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.3) * first_contact, dim=1) # reward only on first contact with the ground
        self.feet_air_time *= ~contact_filt
        return rew_airTime