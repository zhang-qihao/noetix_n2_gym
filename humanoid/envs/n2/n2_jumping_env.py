from isaacgym.torch_utils import *

import torch
import torch.nn.functional as F
from humanoid.envs.n2.n2_amp_env import N2AMPEnv

from humanoid.utils import torch_utils
from humanoid.utils.keyboard_controller import KeyBoardController, KeyboardAction, Delta, Switch

from humanoid.utils.isaacgym_utils import get_euler_xyz_tensor

class N2JumpingEnv(N2AMPEnv):
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
    
    def _post_physics_step_callback(self):
        self.update_feet_state()
        self.update_key_pos_state()

    def _init_buffers(self):
        super()._init_buffers()

        command_motions = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.num_motions = 2
        self.enable_termination_orientation = False
        self.motions = F.one_hot(command_motions, num_classes=self.num_motions).float()
        self.phase_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.random_toggle_steps = torch.randint(low=int(self.cfg.commands.toggle_time[0] / self.dt), 
                                                 high=int(self.cfg.commands.toggle_time[1] / self.dt) + 1, size=(self.num_envs,), device=self.device)
        self.feet_rejust_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.landing_velocity = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.max_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.min_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.jump_toggle = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.has_jumped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.mid_air = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.was_in_flight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.settled_after_init = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.settled_after_init_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)

        self.landing_poses = torch.zeros(self.num_envs, 7, dtype=torch.float, device=self.device, requires_grad=False)
        self.reset_idx_landing_error = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.command_vels = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, z vel

        self.flight_time = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        self._has_jumped_rand_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.has_jumped_reset_flag = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.05],device=self.device))
        self._reset_randomised_has_jumped_timer = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self._has_jumped_switched_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)

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

        # Set jump toggle to True at random steps
        mask = self.episode_length_buf == self.random_toggle_steps
        # mask = self.episode_length_buf % self.random_toggle_steps == 0
        self.jump_toggle[mask] = True

        phase = self._get_phase()
        self.jump_mask = torch.logical_and(phase > 0.1, phase < 0.9)
        self.stand_mask = ~self.jump_mask
        command_motions = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        command_motions[self.jump_mask] = 1
        command_motions[self.stand_mask] = 0

        self.motions = F.one_hot(command_motions, num_classes=self.num_motions).float()
        # print(self.has_jumped, phase, self.base_lin_vel[:, 2].clone(), self.max_height)
        # if not self.record_amp_state.any():
        #     self.record_amp_state[:] = True

        # compute observations, rewards, resets, ...
        self.check_jump()
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

        # # For some of those envs that have started with has_jumped flag, change the flag 
        # # back to 0 to make them jump.
        # if self.cfg.domain_rand.randomize_has_jumped and self.cfg.domain_rand.reset_has_jumped:
        #     rand_envs = self._has_jumped_rand_envs
        #     idx = torch.nonzero(torch.logical_and(rand_envs, self._reset_randomised_has_jumped_timer == self.episode_length_buf),as_tuple=False).flatten()
        #     if not self.cfg.domain_rand.manual_has_jumped_reset_time == 0:
        #         idx = torch.nonzero(torch.logical_and(rand_envs, self.cfg.domain_rand.manual_has_jumped_reset_time == self.episode_length_buf),as_tuple=False).flatten()
        #     self.has_jumped[idx] = False
        #     self._has_jumped_rand_envs[idx] = False
        #     self._reset_randomised_has_jumped_timer[idx] = 0
        #     # Keep track of when the has_jumped flag was switched back to 0 (for certain terminations):
        #     self._has_jumped_switched_time[idx] = self.episode_length_buf[idx]

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_contacts[:] = self.contacts[:]

        self.phase_step += 1
        self.phase_step *= self.jump_toggle

        # jump again
        # rejump_mask = torch.logical_or(phase == 1., torch.logical_and(self.has_jumped, self.max_height < 0.72))
        # self.jump_toggle[rejump_mask] = False
        # self.settled_after_init[rejump_mask] = False
        # self.was_in_flight[rejump_mask] = False
        # self.has_jumped[rejump_mask] = False
        # self.max_height[rejump_mask] = self.base_init_state[2] - 0.1
        # self.min_height[rejump_mask] = self.base_init_state[2] - 0.1

        # Only update the max height achieved during the episode during the first jump while in mid-air:
        idx = self.mid_air * ~self.has_jumped * self.was_in_flight 
        self.max_height[idx] = torch.max(self.max_height[idx], self.root_states[idx, 2] - self.terrain_h[idx]) # update max height achieved
        self.min_height[~self.has_jumped] = torch.min(self.min_height[~self.has_jumped], self.root_states[~self.has_jumped, 2] - self.terrain_h[~self.has_jumped]) # update min height achieved

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        return env_ids, termination_privileged_obs, terminal_amp_states

    def check_jump(self):
        """ Check if the robot has jumped
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) # Contact is true only if either current or previous contact was true
        self.contact_filt = contact_filt.clone() # Store it for the rewards that use it

        # Handle starting in mid-air (initialise in air):
        settled_after_init = torch.logical_and(torch.all(contact_filt, dim=1), self.root_states[:, 2] <= 0.7)

        jump_filter = torch.all(~contact_filt, dim=1) # If no contact for all 2 feet, jump is true
        self.mid_air = jump_filter.clone()

        idx_record_pose = torch.logical_and(settled_after_init, ~self.settled_after_init)
        self.settled_after_init_timer[idx_record_pose] = self.episode_length_buf[idx_record_pose].clone()   # Record the time at which the robot settled after initialisation:
        self.settled_after_init[settled_after_init] = True

        # Only consider in flight if robot has settled after initialisation and is in the air:
        # (only switched to true once for each robot per episode)
        self.was_in_flight[torch.logical_and(jump_filter, self.settled_after_init)] = True # If no contact for all 2 feet, robot is in flight

        # The robot has already jumped if it was previously in flight and has now landed:
        has_jumped = torch.logical_and(torch.any(contact_filt, dim=1), self.was_in_flight) 
       
        # Record landing pose after first jump (before self.has_jumped is updated):
        self.landing_poses[torch.logical_and(has_jumped, ~self.has_jumped)] = self.root_states[torch.logical_and(has_jumped, ~self.has_jumped), :7]

        # Only count the first time flight is achieved:
        self.has_jumped[has_jumped] = True 
        # self.was_in_flight[has_jumped] = False
        # self.landing_velocity[has_jumped] = self.base_lin_vel[has_jumped, 2].clone()
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        if self.cfg.env.enable_early_termination:
            measured_heights = self.terrain_h.clone()
            self.reset_buf |= (self.root_states[:, 2] - measured_heights) < self.cfg.env.termination_height
            # if self.enable_termination_orientation:
            self.reset_buf |= torch.norm(self.base_euler_xyz[:, :2], dim=1) > self.cfg.env.termination_orientation

        # self.reset_buf[self.reset_idx_landing_error] = True # Reset agent if landing error is big
    
    def  _get_phase(self):
        phase = self.phase_step * self.dt / self.cfg.rewards.phase_time
        phase = phase.clamp(min=0, max=1)
        # mask = torch.logical_and(self.has_jumped, self.max_height < 0.72)
        # phase[mask] = 0
        # self.has_jumped[mask] = False
        # self.jump_toggle[mask] = False
        return phase

    def compute_observations(self):
        root_pos = self.root_states[:, :3].clone() 
        phase = self._get_phase().unsqueeze(1)
        sin_pos = torch.sin(2 * torch.pi * phase)
        cos_pos = torch.cos(2 * torch.pi * phase)

        # swing_mask = 1 - self._get_jump_phase()
        # jump_mask = torch.all(swing_mask == 1., dim=1).unsqueeze(1)

        has_jumped = self.has_jumped.unsqueeze(1)
        jump_toggle = torch.where(self.jump_toggle, 1.0, 0.0).unsqueeze(1)

        obs_buf = torch.cat((  
                                sin_pos,                                        
                                cos_pos,    
                                # jump_toggle,                                    
                                self.base_ang_vel  * self.obs_scales.ang_vel,
                                self.projected_gravity,  
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                self.dof_vel * self.obs_scales.dof_vel,
                                self.actions
                            ),dim=-1)

        self.privileged_obs_buf = torch.cat((  
                                    sin_pos,                                        
                                    cos_pos,                                       
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,  
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    has_jumped,        
                                    jump_toggle,                               
                                    # root_pos[:, 2:3],                                    
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
        
        # clear observation history
        if self.cfg.env.frame_stack is not None:
            # clear observation history
            for i in range(self.obs_history.maxlen):
                self.obs_history[i][env_ids] *= 0
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # if torch.mean(self.episode_sums["orientation"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["orientation"]:
        #     self.enable_termination_orientation = True
        
        # reset robot states
        if self.cfg.env.reference_state_initialization and self.cfg.env.prob_rsi > torch.rand(1, dtype=torch.float32, device=self.device):
            frames = self.motion_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        # Recompute commands
        self.random_toggle_steps[env_ids] = torch.randint(low=int(self.cfg.commands.toggle_time[0] / self.dt), 
                                                          high=int(self.cfg.commands.toggle_time[1] / self.dt) + 1, size=(len(env_ids),), device=self.device)
        # self._recompute_commands(env_ids)

        self.jump_toggle[env_ids] = False 
        self.was_in_flight[env_ids] = False
        self.mid_air[env_ids] = False
        self.has_jumped[env_ids] = False
        self.settled_after_init[env_ids] = False
        # self.reset_idx_landing_error[env_ids] = False
        # self._has_jumped_rand_envs[env_ids] = False
        # self.landing_poses[env_ids, :] = float('nan')

        # if self.cfg.domain_rand.randomize_has_jumped:
        #     self.has_jumped[env_ids] = self.has_jumped_randomisation_prob.sample((len(env_ids),1)).bool().flatten()
        #     self._has_jumped_rand_envs[env_ids] = self.has_jumped[env_ids] == True
        #     # Idx of environments that have has_jumped as true now:
        #     idx = env_ids[self._has_jumped_rand_envs[env_ids] == 1]
        #     self._reset_randomised_has_jumped_timer[idx] = torch_rand_float(1.0, 0.3 * self.max_episode_length, (len(idx), 1),device=self.device).int().flatten()
        #     # If allowing them to jump in the final part of the episode - just don't allow them as there isnt enough time.
        #     self._has_jumped_switched_time[idx] = self.max_episode_length

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.feet_rejust_time[env_ids] = 0.
        self.flight_time[env_ids] = 0.
        self.max_height[env_ids] = self.base_init_state[2] - 0.3 - self.terrain_h[env_ids]
        self.min_height[env_ids] = self.base_init_state[2] - 0.3 - self.terrain_h[env_ids]
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

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
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
            skill_labels = self.motions.clone()
            tot_rew, style_rew = self.discriminator.predict_amp_reward(amp_observation_buf, skill_labels, task_rew, self.dt, self.amp_state_normalizer, self.amp_style_reward_normalizer)
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

    def _get_keyboard_events(self):
        """Simple keyboard controller for linear and angular velocity."""

        def print_command():
            print("[LeggedRobot]: Environment 0 command: ", self.commands_des[0])
            self.commands[:, 0] = self.commands_des[:, 0] 
            self.commands[:, 1] = self.commands_des[:, 1] 
            self.commands[:, 2] = self.commands_des[:, 2] 
        
        def print_jump():
            self.was_in_flight[:] = False
            self.has_jumped[:] = False
            self.settled_after_init[:] = False
            self.reset_idx_landing_error[:] = False
            self._has_jumped_rand_envs[:] = False
            self.landing_poses[:, :] = float('nan')

            self.initial_root_states = self.root_states
            print("[LeggedRobot]: Environment 0 start jump:", self.jump_toggle)
        
        def print_reset():
            print("[LeggedRobot]: Environment reset")
            self.reset()

        key_board_events = {
            # "w": Delta("pos_dx", amount=0.05, variable_reference=self.commands_des[:, 0], callback=print_command),
            # "s": Delta("pos_dx", amount=-0.05, variable_reference=self.commands_des[:, 0], callback=print_command),
            # "a": Delta("pos_dy", amount=0.05, variable_reference=self.commands_des[:, 1], callback=print_command),
            # "d": Delta("pos_dy", amount=-0.05, variable_reference=self.commands_des[:, 1], callback=print_command),
            # "q": Delta("pos_dw", amount=0.05, variable_reference=self.commands_des[:, 2], callback=print_command),
            # "e": Delta("pos_dw", amount=-0.05, variable_reference=self.commands_des[:, 2], callback=print_command),
            "j": Switch("Start_jump", True, False, variable_reference=self.jump_toggle, callback=print_jump),
            "r": Switch("Reset", True, False, variable_reference=self.reset_toggle, callback=print_reset)
        }
        return key_board_events

# ================================================ Rewards ================================================== #
    # def _reward_task_pos(self):
    #     # Reward for completing the task
        
    #     env_ids = self.episode_length_buf == self.max_episode_length
    #     rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

    #     # Base position relative to initial states:
    #     rel_root_states = self.landing_poses[:,:2] - self.initial_root_states[:,:2]

    #     tracking_error = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
    #     tracking_error = torch.linalg.norm(rel_root_states[:] - self.commands[:, :2],dim=1)
    #     # Check which envs have actually jumped (and not just been initialised at an already "jumped" state)
    #     has_jumped_idx = torch.logical_and(self.has_jumped,~self._has_jumped_rand_envs)

    #     max_tracking_error = self.cfg.env.reset_landing_error 

    #     self.reset_idx_landing_error[torch.logical_and(has_jumped_idx,tracking_error > max_tracking_error)] = True

    #     if torch.all(env_ids == False): # if no env is done return 0 reward for all
    #         pass
    #     else:
            
    #         # Only give a reward for robots that have landed and are at the end of the episode:
    #         idx = torch.logical_and(env_ids, has_jumped_idx)

    #         rew[idx] = torch.exp(-torch.square(tracking_error[idx]) * 20)


    #     return rew
    
    # def _reward_task_ori(self):
    #     # Reward for completing the task
    #     env_ids = self.episode_length_buf == self.max_episode_length

    #     rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

    #     _, _, yaw_landing = get_euler_xyz(self.landing_poses[:, 3:7])
    #     _, _, yaw_initial = get_euler_xyz(self.initial_root_states[:, 3:7])

    #     ori_tracking_error_yaw = torch.abs(wrap_to_pi(self.commands[:, 2] - wrap_to_pi((yaw_landing - yaw_initial))))

    #     # Check which envs have actually jumped (and not just been initialised at an already "jumped" state)
    #     has_jumped_idx = torch.logical_and(self.has_jumped, ~self._has_jumped_rand_envs)
    #     self.reset_idx_landing_error[torch.logical_and(has_jumped_idx,ori_tracking_error_yaw > 0.1)] = True

    #     if torch.all(env_ids == False): # if no env is done return 0 reward for all
    #         pass
    #     else:
    #         idx = env_ids * has_jumped_idx
            
    #         rew[idx] = torch.exp(-torch.square(ori_tracking_error_yaw[idx]) * 20)

    #     return rew
    
    # def _reward_post_landing_pos(self):
    #     # Reward for remaining at the same position after landing:
    #     env_ids = self.has_jumped
    #     rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

    #     if torch.all(env_ids == False): # if no env is done return 0 reward for all
    #         pass
    #     else:
    #         tracking_error  = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
    #         # Track landing position deviation:
    #         tracking_error[env_ids] = torch.linalg.norm(self.root_states[env_ids, :2] - self.landing_poses[env_ids, :2],dim=1)

    #         # For those that started as has_jumped, track initial position deviation:
    #         idx = torch.logical_and(env_ids, self._has_jumped_rand_envs)
    #         tracking_error[idx] = torch.linalg.norm(self.root_states[idx,:2] - self.initial_root_states[idx,:2],dim=1)

    #         rew[env_ids] = torch.exp(-torch.square(tracking_error[env_ids]) * 1000)
    #     return rew
    
    # def _reward_post_landing_ori(self):
    #     # Reward for remaining at the same orientation after landing:
    #     env_ids = self.has_jumped
    #     rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

    #     if torch.all(env_ids == False): # if no env is done return 0 reward for all
    #         pass
    #     else:
    #         quat_ini = self.root_states[:, 3:7]
    #         quat_landing = self.landing_poses[:, 3:7]
    #         ori_tracking_error = quat_distance(quat_ini, quat_landing)
    #         rew[env_ids] = torch.exp(-torch.square(ori_tracking_error[env_ids]) * 20)
    #     return rew
    
    
    # def _reward_base_height_stance(self):
    #     # Reward feet height
    #     base_height = self.root_states[:, 2]
    #     rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

    #     base_height_stance = (base_height - 0.68)[self.has_jumped]

    #     rew[self.has_jumped] =  torch.exp(-torch.square(base_height_stance) * 200)
        
    #     return rew
    
    # def _reward_action_smoothness(self):
    #     """
    #     Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
    #     This is important for achieving fluid motion and reducing mechanical stress.
    #     """
    #     return torch.sum(torch.square(
    #         self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
    
    # def _reward_jumping(self):
    #     # Reward if the robot has jumped in the episode:
    #     env_ids = torch.logical_or(self.episode_length_buf == self.max_episode_length,
    #               torch.logical_and(self.reset_buf, self.episode_length_buf < self.max_episode_length))

    #     rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        
    #     rew[env_ids * self.has_jumped * self.max_height > 0.65] = 1        
        
    #     return rew
    
    # def _reward_tracking_lin_vel(self):
    #     # Tracking of linear velocity commands (xy axes)

    #     rew = torch.zeros(self.num_envs, device=self.device)
    #     lin_vel_error = torch.zeros(self.num_envs, device=self.device)
    #     # Linear velocity commands for flight phase:
    #     flight_idx = self.mid_air * ~self.has_jumped
    #     lin_vel_error[flight_idx] = torch.sum(torch.square(self.root_states[flight_idx, 7:9] - self.command_vels[flight_idx, :2]), dim=-1)
    #     # If told to stand in place, penalise the velocity:
    #     stance_idx = self.has_jumped * self._has_jumped_rand_envs
    #     lin_vel_error[stance_idx] = torch.sum(torch.square(self.root_states[stance_idx, 7:9]), dim=-1)
        
    #     rew = torch.exp(-lin_vel_error * 10)
    #     rew[~self.has_jumped * ~self.mid_air] = 0
    #     rew[self.has_jumped * ~self._has_jumped_rand_envs] = 0

    #     return rew
    
    # def _reward_tracking_ang_vel(self):
    #     # Tracking of angular velocity commands (yaw only)
    #     rew = torch.zeros(self.num_envs, device=self.device)
    #     ang_vel_error = torch.zeros(self.num_envs, device=self.device) 

    #     flight_idx = self.mid_air * ~self.has_jumped
    #     ang_vel_error[flight_idx] = torch.square(self.root_states[flight_idx, 12] - self.command_vels[flight_idx, 2])

    #     stance_idx = self.has_jumped *  self._has_jumped_rand_envs
    #     ang_vel_error[stance_idx] = torch.sum(torch.square(self.root_states[stance_idx, 10:13]),dim=-1)
        
        
    #     rew = torch.exp(-ang_vel_error * 10)

    #     rew[~self.has_jumped * ~self.mid_air] = 0
    #     rew[self.has_jumped * ~self._has_jumped_rand_envs] = 0
        
    #     return rew
    
    # def _reward_early_contact(self):
    #     # Reward maintaining contact at the very beginning of the episode:
        
    #     env_ids = torch.logical_or((self.episode_length_buf - self.settled_after_init_timer <= 10) * \
    #                                (self.episode_length_buf - self.settled_after_init_timer >= 0) * self.settled_after_init,
    #                                 (self.episode_length_buf - self._has_jumped_switched_time <= 10) *\
    #                                 (self.episode_length_buf - self._has_jumped_switched_time >= 0) * self.settled_after_init)


    #     rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
    #     # Give a reward of 1 if all feet are in contact:
    #     idx = torch.all(self.contacts,dim=1)
    #     rew[torch.logical_and(env_ids,idx)] = 1.
    #     # Give a smaller reward if all feet are in contact when landed:
    #     rew[self.has_jumped * self.was_in_flight * idx] = 0.2

    #     return rew
    
    # def _reward_flight_time(self):
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts) # Contact is true only if either current or previous contact was true

    #     jump_filter = torch.all(~contact_filt, dim=1) # If no contact for all 2 feet, jump is true
        
    #     first_jump = (self.flight_time > 0.) * ~jump_filter
    #     self.flight_time += self.dt
    #     rew_airTime = torch.sum((self.flight_time - self.target_flight_time) * first_jump, dim=1) # reward only on first contact with the ground
    #     rew_airTime *= ~self.has_jumped
    #     self.flight_time *= jump_filter.unsqueeze(1)

    #     return rew_airTime
    
    # def _reward_base_acc(self):
    #     # Penalize base accelerations
    #     rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

    #     base_acc_approx = (self.root_states[:,7:10] - self.last_root_vel[:,:3]) / self.dt

    #     base_acc = base_acc_approx.clone()

    #     idx_time = torch.logical_or((self.episode_length_buf - self.settled_after_init_timer <= 10) * \
    #                                 (self.episode_length_buf - self.settled_after_init_timer >= 0) * \
    #                                 self.settled_after_init,
    #                                 (self.episode_length_buf - self._has_jumped_switched_time <= 10) * \
    #                                 (self.episode_length_buf - self._has_jumped_switched_time >= 0) * \
    #                                 self.settled_after_init)

    #     rew = torch.square(base_acc)
    #     rew[idx_time] *= 1e3
        
    #     return torch.sum(rew, dim=1)

    def _reward_task_max_height(self):
        # Reward for max height achieved during the episode:
        env_ids = torch.logical_and(self.episode_length_buf == self.max_episode_length, self.has_jumped)
        rew  = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        if torch.all(env_ids == False): # if no env is done return 0 reward for all
            return rew
        max_height_reward = (self.max_height[env_ids] - 0.85)
        rew[env_ids] = torch.exp(-torch.square(max_height_reward) * 50)
        return rew
    
    # def _reward_base_height_flight(self):
    #     # Reward flight height
    #     rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
    #     base_height_flight = (self.root_states[self.mid_air, 2] - 1.0)
    #     rew[self.mid_air] = torch.exp(-torch.square(base_height_flight) * 10)
    #     rew[self.has_jumped + ~self.mid_air] = 0.       
    #     return rew

    def _reward_jumping(self):
        # Reward if the robot has jumped in the episode:
        env_ids = torch.logical_or(self.episode_length_buf == self.max_episode_length,
                  torch.logical_and(self.reset_buf, self.episode_length_buf < self.max_episode_length))


        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        rew[env_ids * self.has_jumped * self.max_height > 0.72] = 1        
        return rew
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        joint_diff = torch.square(self.dof_pos - self.default_dof_pos)
        joint_diff[:, self.ankle_dof_idxs] = 0
        rew = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        # rew[self.stand_mask] = 0.5 * torch.exp(-torch.sum(torch.square(-self.dof_vel[self.stand_mask]), dim=1) * 20) + 0.5 * torch.exp(-torch.sum(torch.abs(joint_diff[self.stand_mask]), dim=1) * 5)
        rew[self.stand_mask] = torch.sum(torch.abs(joint_diff[self.stand_mask]), dim=1)
        return rew 
    
    def _reward_default_joint_pos(self):
        joint_diff = self.dof_pos - self.default_dof_pos

        left_yaw_roll = joint_diff[:, self.left_yaw_roll]
        right_yaw_roll = joint_diff[:, self.right_yaw_roll]

        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)

        rew = torch.exp(-yaw_roll * 100) - 0.05 * torch.norm(joint_diff[:, self.left_yaw_roll + self.right_yaw_roll], dim=1)
        return rew
    
    # def _reward_stand_still(self):
    #     phase = self._get_phase()
    #     joint_diff = torch.square(self.dof_pos - self.default_joint_pd_target)
    #     joint_diff[:, ANKLE_DOF_IDXS] = 0
    #     # rew_mask = torch.logical_and(torch.norm(self.projected_gravity[:, :2], dim=1) < 0.2, torch.logical_or(phase < 0.3, phase > 0.6))
    #     rew_mask = torch.logical_and(torch.norm(self.projected_gravity[:, :2], dim=1) < 0.3, torch.logical_or(phase < 0.2, phase > 0.7))
    #     rew = 0.5 * torch.exp(-torch.sum(torch.square(-self.dof_vel), dim=1) * 20) + 0.5 * torch.exp(-torch.sum(torch.abs(joint_diff), dim=1) * 5)
    #     rew[~rew_mask] = 1.
    #     return rew
    
    def _reward_joint_pos_symmetry(self):
        diff = torch.square(self.actions[:, 0] - self.actions[:, 9])
        diff += torch.square(self.actions[:, 1:3] + self.actions[:, 10:12]).sum(dim=-1)
        diff += torch.square(self.actions[:, 3] - self.actions[:, 12])
        diff += torch.square(self.actions[:, 4:6] + self.actions[:, 13:15]).sum(dim=-1)
        diff += torch.square(self.actions[:, 6:9] - self.actions[:, 15:18]).sum(dim=-1)
        return diff

    def _reward_lin_vel_z(self):
        phase = self._get_phase()
        mask = torch.logical_and(torch.logical_and(phase > 0.45, phase < 0.6), ~self.has_jumped)
        return self.base_lin_vel[:, 2].clone().clamp(min=0, max=3) * mask
    
    def _reward_feet_height(self):
        phase = self._get_phase()
        mask = torch.logical_and(torch.logical_and(phase > 0.55, phase < 0.65), ~self.has_jumped)
        feet_height = torch.mean(self.feet_pos[:, :, 2], dim=-1) - 0.04
        return feet_height * mask
    
    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(-self.base_lin_vel[:, :2]), dim=1)
        rew = torch.exp(-lin_vel_error * 5)
        return rew

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        
        ang_vel_error = torch.square(self.base_ang_vel[:, 2])
        rew = torch.exp(-ang_vel_error * 5)
        return rew
    
    def _reward_change_of_contact(self):
        # Penalty for changing contact state:
        rew = torch.sum(torch.abs(self.contacts.int() - self.last_contacts.int()),dim=1)
        rew = torch.exp(-torch.square(rew) * 5)
        return rew
    
    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        env_ids = torch.logical_or(self.stand_mask, self.has_jumped)
        if torch.all(env_ids == False): # if no env is done return 0 reward for all
            return rew
        rew[env_ids] = torch.exp(-torch.norm(root_acc[env_ids], dim=1) * 3)
        return rew
    
    def _reward_landing_buffer(self):
        phase = self._get_phase()
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        # env_ids = torch.logical_and(self.has_jumped, torch.logical_and(phase > 0.75, phase < 0.9))
        env_ids = torch.logical_and(self.has_jumped, phase < 0.8)
        # print(self.root_states[:, 2], torch.norm(self.base_euler_xyz[:, :2], dim=1), phase)
        if torch.all(env_ids == False): # if no env is done return 0 reward for all
            return rew
        # target_height = 0.65 # + (self.landing_velocity[env_ids] / 10) 
        # height_err = (self.root_states[env_ids, 2] - self.terrain_h - target_height)
        # rew[env_ids] = torch.exp(-torch.square(height_err) * 50)
        rew[env_ids] = -self.base_lin_vel[env_ids, 2].clone().clamp(min=-0.5, max=0.0)
        return rew
    
    def _reward_jump_up(self):
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        root_h = self.root_states[:, 2]
        root_h_error_jump = torch.sqrt(torch.square(0.85 - root_h))
        root_h_error_loc = torch.sqrt(torch.square(0.70 - root_h))
        jump_goal = (root_h_error_jump < 0.1) & self.jump_toggle
        rew[jump_goal] += self.cfg.rewards.jump_goal
        self.jump_toggle[jump_goal] = 0

        root_h_error_rwd_jump = torch.exp(-torch.square(root_h_error_jump) * 50)
        root_h_error_rwd_loc = torch.exp(-torch.square(root_h_error_loc) * 50)

        rew[self.jump_toggle] += root_h_error_rwd_jump[self.jump_toggle]
        rew[~self.jump_toggle] += root_h_error_rwd_loc[~self.jump_toggle]
        return rew