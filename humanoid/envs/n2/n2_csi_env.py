from isaacgym.torch_utils import *

import torch
import torch.nn.functional as F
from humanoid.envs.n2.n2_amp_env import N2AMPEnv

from humanoid.utils.terrain import  HumanoidTerrain
from humanoid.utils.keyboard_controller import KeyBoardController, KeyboardAction, Delta, Switch

from humanoid.utils.isaacgym_utils import get_euler_xyz_tensor
from humanoid.csi_utils.csi_discriminator import CSIDiscriminator

class N2CSIEnv(N2AMPEnv):
    def _get_noise_scale_vec(self, cfg):
        if self.cfg.env.frame_stack is not None:
            noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        else:
            noise_vec = torch.zeros_like(self.obs_buf[0])

        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:6+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[6+self.num_actions:6+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[6+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions

        return noise_vec
    
    def _post_physics_step_callback(self):
        env_ids = (self.episode_length_buf % (self.resampling_time / self.dt).int()==0).nonzero(as_tuple=False).flatten()
        self._resample_motion_commands(env_ids)

        self.update_feet_state()
        self.update_key_pos_state()

    def _init_buffers(self):
        super()._init_buffers()
        self.num_skill_labels = self.motion_loader.num_motions
        self.motion_commands = torch.zeros((self.num_envs, self.num_skill_labels), dtype=torch.float, device=self.device)
    
    def _resample_motion_commands(self, env_ids):
        random_motions = torch.randint(low=0, high=self.num_skill_labels, size=(len(env_ids),), dtype=torch.long, device=self.device)
        self.motion_commands[env_ids] = F.one_hot(random_motions, num_classes=self.num_skill_labels).float()

    def compute_observations(self):
        """ Computes observations
        """
        obs_buf = torch.cat((  
                                self.base_ang_vel  * self.obs_scales.ang_vel,
                                self.projected_gravity,  
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                self.dof_vel * self.obs_scales.dof_vel,
                                self.actions
                            ),dim=-1)

        self.privileged_obs_buf = torch.cat((  
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
    
    
    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.discriminator is not None and self.amp_state_normalizer is not None:
            self.next_amp_observations = self.get_amp_observations()
            amp_observation_buf = torch.cat((self.amp_observation_buf[:, 1:], self.next_amp_observations.unsqueeze(1)), dim=1)
            skill_labels = self.motion_commands.clone()
            task_rew = self.rew_buf
            tot_rew, style_rew = self.discriminator.predict_csi_reward(amp_observation_buf, skill_labels, task_rew, self.dt, self.amp_state_normalizer, self.amp_style_reward_normalizer)
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
        
        # reset robot states
        if self.cfg.env.reference_state_initialization and self.cfg.env.prob_rsi > torch.rand(1, dtype=torch.float32, device=self.device):
            frames = self.motion_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        self._resample_motion_commands(env_ids)
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

    def _resample_motion_commands(self, env_ids):
        random_motions = torch.randint(low=0, high=self.num_skill_labels, size=(len(env_ids),), dtype=torch.long, device=self.device)
        self.motion_commands[env_ids] = F.one_hot(random_motions, num_classes=self.num_skill_labels).float()
    
    def _get_keyboard_events(self):
        """Simple keyboard controller for linear and angular velocity."""

        def print_command():
            # motions = 0 * torch.ones(size=(1,), dtype=torch.long, device=self.device)
            # self.motion_commands[0] = F.one_hot(motions, num_classes=self.num_skill_labels).float()
            self._resample_motion_commands([0])
            print("[LeggedRobot]: Environment 0 motion command: ", self.motion_commands[0])

        def print_command_2():
            motions = 2 * torch.ones(size=(1,), dtype=torch.long, device=self.device)
            self.motion_commands[0] = F.one_hot(motions, num_classes=self.num_skill_labels).float()
        
        def print_reset():
            print("[LeggedRobot]: Environment reset")
            self.reset()

        key_board_events = {
            "w": Delta("lin_vel_x", amount=0.05, variable_reference=self.commands_des[:, 0], callback=print_command),
            "s": Delta("lin_vel_x", amount=0.05, variable_reference=self.commands_des[:, 0], callback=print_command_2),
            "r": Delta("lin_vel_x", amount=-0.05, variable_reference=self.commands_des[:, 0], callback=print_reset),
        }

        return key_board_events
