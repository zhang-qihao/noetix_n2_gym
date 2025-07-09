from humanoid.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class N2Env(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        if self.num_dofs == 10:
            self.left_yaw_roll  = [0, 1]        
            self.right_yaw_roll  = [5, 6]   
            self.ankle_dof_idxs = [4, 9] 
        elif self.num_dofs == 18:
            self.left_yaw_roll  = [4, 5]        
            self.right_yaw_roll  = [13, 14]   
            self.up_left_yaw_roll  = [1, 2]        
            self.up_right_yaw_roll  = [10, 11]   
            self.ankle_dof_idxs = [8, 17] 
        elif self.num_dofs == 20:
            self.left_yaw_roll  = [4, 5]        
            self.right_yaw_roll  = [14, 15]   
            self.up_left_yaw_roll  = [1, 2]        
            self.up_right_yaw_roll  = [11, 12]   
            self.ankle_dof_idxs = [8, 9, 18, 19] 
        else:
            assert False, "Unsupported num joints"

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        if self.cfg.env.frame_stack is not None:
            noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        else:
            noise_vec = torch.zeros_like(self.obs_buf[0])
            
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = 0. # commands
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions

        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        self.last_feet_vel = self.feet_vel.clone()
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.last_feet_vel = self.feet_vel.clone()
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()
        return super()._post_physics_step_callback()
    
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
            self.commands[env_ids[random_tensor < 0.20], :] = 0
            self.commands[env_ids[torch.logical_and(random_tensor >= 0.20, random_tensor < 0.30)], 0] = 0
            self.commands[env_ids[torch.logical_and(random_tensor >= 0.30, random_tensor < 0.40)], 2] = 0

        # set small commands to zero
        self.commands[env_ids, :3] *= (torch.norm(self.commands[env_ids, :3], dim=1) > self.min_cmd_vel).unsqueeze(1)
    
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
    

# ================================================ Rewards ================================================== #
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        stand_lin_vel_error = torch.norm(-self.base_lin_vel[self.standing_cmd, :2], dim=1)
        rew = torch.exp(-lin_vel_error * 5)
        rew[self.standing_cmd] = torch.exp(-stand_lin_vel_error * 5)
        return rew
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * 5)
    
    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        return reward * (self.commands[:, 0].abs() > self.min_cmd_vel)
    
    def _reward_feet_contact(self):
        rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        contact_filt = torch.logical_or(self.contacts, self.last_contacts)
        single_feet_contact = torch.logical_xor(contact_filt[:, 0], contact_filt[:, 1]) 
        both_feet_contact = torch.logical_and(contact_filt[:, 0], contact_filt[:, 1])

        self.feet_both_contact_time[both_feet_contact] += self.dt
        self.feet_both_contact_time *= both_feet_contact
        
        rew_filter = torch.logical_or(single_feet_contact, self.feet_both_contact_time < 0.2)
        rew_filter = torch.logical_or(rew_filter, self.standing_cmd)
        rew[rew_filter] = 1.0
        return rew
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact_feet_vel = self.feet_vel * self.contacts.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_default_joint_pos(self):
        joint_diff = self.dof_pos - self.default_dof_pos

        left_yaw_roll = joint_diff[:, self.left_yaw_roll]
        right_yaw_roll = joint_diff[:, self.right_yaw_roll]

        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)

        rew = torch.exp(-yaw_roll * 100) - 0.05 * torch.norm(joint_diff[:, self.left_yaw_roll + self.right_yaw_roll], dim=1)
        rew_filter = torch.logical_or(torch.any(torch.abs(self.commands[:, [1, 2]]) > self.min_cmd_vel, dim=1), self.standing_cmd) 
        
        rew[rew_filter] = 1.0
        return rew
    
    def _reward_default_up_joint_pos(self):
        joint_diff = self.dof_pos - self.default_dof_pos

        left_yaw_roll = joint_diff[:, self.up_left_yaw_roll]
        right_yaw_roll = joint_diff[:, self.up_right_yaw_roll]

        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)

        rew = torch.exp(-yaw_roll * 100) - 0.05 * torch.norm(joint_diff[:, self.up_left_yaw_roll + self.up_right_yaw_roll], dim=1)
        return rew
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        rew[self.standing_cmd] = torch.sum(torch.abs(self.dof_pos[self.standing_cmd] - self.default_dof_pos), dim=1) 
        rew[self.standing_cmd] += torch.sum(torch.square(self.dof_vel[self.standing_cmd]), dim=1)
        return rew 
    
    def _reward_center_of_mass(self):
        left_contact, right_contact = self.contacts[:, 0], self.contacts[:, 1]
        total_contact = (torch.sum(self.contacts, dim=-1)).clamp(min=1e-5)

        p_csp = (left_contact.unsqueeze(-1) * self.feet_pos[:, 0] +
                 right_contact.unsqueeze(-1) * self.feet_pos[:, 1]) / total_contact.unsqueeze(-1)
        diff = p_csp[:, :2] - self.root_states[:, :2].clone()

        rew = torch.exp(-torch.norm(diff, dim=1).pow(2) * 50)
        rew[total_contact == 0] = 0.0
        return rew

    def _reward_feet_orientation(self):        
        feet_quat = self.feet_state[:, :, 3:7]
        foot_quat_1 = feet_quat[:, 0, :]
        foot_quat_2 = feet_quat[:, 1, :]
        lfoot_pitch = get_euler_xyz(foot_quat_1)[1]
        rfoot_pitch = get_euler_xyz(foot_quat_2)[1]
        lfoot_pitch[lfoot_pitch > torch.pi] -= 2*torch.pi
        lfoot_pitch[lfoot_pitch < -torch.pi] += 2*torch.pi
        rfoot_pitch[rfoot_pitch > torch.pi] -= 2*torch.pi
        rfoot_pitch[rfoot_pitch < -torch.pi] += 2*torch.pi
        feet_orient = torch.cat((lfoot_pitch.unsqueeze(1), rfoot_pitch.unsqueeze(1)), dim=-1)
        return torch.exp(-torch.norm(feet_orient * self.contacts, dim=1) * 20)