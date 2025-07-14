from humanoid.envs.n2.n2_env import N2Env

import time
from isaacgym.torch_utils import *
from isaacgym import gymtorch
import torch

from humanoid.utils.isaacgym_utils import get_euler_xyz_tensor
from humanoid.amp_utils.motion_loader import *
from humanoid.amp_utils.discriminator import Discriminator


class N2AMPEnv(N2Env):
    def _create_envs(self):
        super()._create_envs()
        key_names = []
        for name in self.cfg.asset.key_name:
            key_names.extend([s for s in self.body_names if name in s])
        self.key_indices = torch.zeros(len(key_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(key_names)):
            self.key_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], key_names[i])

    def update_key_pos_state(self):
        self.key_pos_state = self.rigid_body_states_view[:, self.key_indices, :]
        self.key_pos = self.key_pos_state[:, :, :3]

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.update_key_pos_state()

    def _init_buffers(self):
        super()._init_buffers()
        self.estimator = None # assigned in runner
        # get key pos state
        self.update_key_pos_state()
        # load AMP components
        self.reference_motion_file = self.cfg.motion_loader.reference_motion_file
        self.reference_observation_horizon = self.cfg.motion_loader.reference_observation_horizon
        self.num_preload_transitions = self.cfg.motion_loader.num_preload_transitions
        self.motion_loader_class = eval(self.cfg.motion_loader.motion_loader_name)
        self.motion_loader: MotionLoaderNing10DOF = self.motion_loader_class(
            device=self.device,
            time_between_frames=self.dt,
            reference_observation_horizon=self.reference_observation_horizon,
            num_preload_transitions=self.num_preload_transitions,
            motion_files=self.reference_motion_file
        )
        self.discriminator: Discriminator = None # assigned in runner
        self.amp_state_normalizer = None # assigned in runner
        self.amp_style_reward_normalizer = None # assigned in runner
        self.amp_observation_buf = torch.zeros(
            self.num_envs, self.reference_observation_horizon, self.motion_loader.observation_dim, 
            dtype=torch.float, device=self.device, requires_grad=False
        )
    
    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        self.episode_sums["task"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums["style"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
    
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # dynamic randomization
        # if not self.cfg.env.test:
        delay = torch.rand((self.num_envs, 1), device=self.device)
        actions = (1 - delay) * actions + delay * self.actions
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
        termination_ids, termination_priveleged_obs, terminal_amp_states = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, \
            termination_priveleged_obs, terminal_amp_states
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.contacts = self.contact_forces[:, self.feet_indices, 2] > 5.

        self._post_physics_step_callback()
        self.record_amp_state = ~self.disturbance_state

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        self.update_amp_observation_buf()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.privileged_obs_buf[env_ids].clone()
        terminal_amp_states = self.get_amp_observation_buf()[env_ids]
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
        return env_ids, termination_privileged_obs, terminal_amp_states
    
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
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        if self.cfg.env.reference_state_initialization and self.cfg.env.prob_rsi > torch.rand(1, dtype=torch.float32, device=self.device):
            frames = self.motion_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        # update reset amp observation
        self.reset_amp_observation_buf(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.feet_contact_time[env_ids] = 0.
        self.feet_both_contact_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

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
        self._refresh_cmd_resample_time(env_ids)

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
        
        # fix reset gravity bug
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.discriminator is not None and self.amp_state_normalizer is not None:
            self.next_amp_observations = self.get_amp_observations()
            amp_observation_buf = torch.cat((self.amp_observation_buf[:, 1:], self.next_amp_observations.unsqueeze(1)), dim=1)
            task_rew = self.rew_buf.clone()
            tot_rew, style_rew = self.discriminator.predict_amp_reward(amp_observation_buf, task_rew, self.dt, self.amp_state_normalizer, self.amp_style_reward_normalizer)
            if self.cfg.env.test: tot_rew = tot_rew.unsqueeze(0)
            self.episode_sums["task"] += task_rew
            self.episode_sums["style"] += style_rew
            self.rew_buf = tot_rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        if len(env_ids) > 0:
            random_tensor = torch.rand_like(self.commands[env_ids, 0])
            self.commands[env_ids[random_tensor < 0.1], 0] = 0
            self.commands[env_ids[torch.logical_and(random_tensor >= 0.1, random_tensor < 0.2)], 2] = 0

        # set small commands to zero
        self.commands[env_ids, :3] *= (torch.norm(self.commands[env_ids, :3], dim=1) > self.min_cmd_vel).unsqueeze(1)
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 70% of the maximum, increase the range of commands
        cmd_update_ids = env_ids[~self.standing_cmd[env_ids]]
        if torch.mean(self.episode_sums["tracking_lin_vel"][cmd_update_ids]) / self.max_episode_length > 0.7 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.1, 0., self.cfg.commands.max_curriculum)
    
    def compute_observations(self):
        """ Computes observations
        """
        obs_buf = torch.cat((  
                                self.commands[:, :3] * self.commands_scale,
                                self.base_ang_vel  * self.obs_scales.ang_vel,
                                self.projected_gravity,  
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                self.dof_vel * self.obs_scales.dof_vel,
                                self.actions
                            ),dim=-1)
        
        self.privileged_obs_buf = torch.cat((  
                                    self.commands[:, :3] * self.commands_scale,
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
                                    self.contacts
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


# ================================================ AMP ======================================================== #
    def get_amp_observations(self):
        root_pos = self.root_states[:, :3].clone() 
        z_pos = root_pos[:, 2:3] - self.terrain_h.unsqueeze(1)
        root_vel = self.root_states[:, 7:10].clone()
        root_ang_vel = self.root_states[:, 10:13].clone()
        dof_pos = self.dof_pos.clone()
        dof_vel = self.dof_vel.clone()
        key_pos = (self.key_pos - root_pos.unsqueeze(1)).view(self.num_envs, -1) 

        # mask feet: we mask the expert's ankels and force robot to learn how to use its feet on its own. 
        dof_pos[:, self.ankle_dof_idxs] = 0.0
        dof_vel[:, self.ankle_dof_idxs] = 0.0

        return torch.cat((dof_pos, key_pos, root_vel, root_ang_vel, dof_vel, z_pos), dim=-1)
    
    def update_amp_observation_buf(self):
        self.amp_observation_buf[:, :-1] = self.amp_observation_buf[:, 1:].clone()
        self.amp_observation_buf[:, -1] = self.next_amp_observations.clone()
    
    def get_amp_observation_buf(self):
        return self.amp_observation_buf.clone()
    
    def reset_amp_observation_buf(self, env_ids):
        self.amp_observation_buf[env_ids, :] = 0.0
        reset_observation = self.get_amp_observations()[env_ids, :]
        self.amp_observation_buf[env_ids, -1] = reset_observation
    
    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = self.motion_loader_class.get_joint_pose_batch(frames)
        self.dof_vel[env_ids] = self.motion_loader_class.get_joint_vel_batch(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        root_pos = self.motion_loader_class.get_root_pos_batch(frames)
        root_pos[:, :3] = root_pos[:, :3] + self.env_origins[env_ids, :3]
        root_pos[:, 2] += self.terrain_h[env_ids]
        self.root_states[env_ids, :3] = root_pos
        root_orn = self.motion_loader_class.get_root_rot_batch(frames)
        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = self.motion_loader_class.get_linear_vel_batch(frames)
        self.root_states[env_ids, 10:13] = self.motion_loader_class.get_angular_vel_batch(frames)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))