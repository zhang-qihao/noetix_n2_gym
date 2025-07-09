import torch

def data_augmentation_func_n2(obs, actions, env, obs_type):
    if obs is None:
        output_obs = None
    else:
        if obs_type == "policy":
            if hasattr(env, "num_single_obs"): single_obs_dim = env.num_single_obs 
            else: single_obs_dim = env.num_obs
            total_dims = obs.shape[-1]
            num_frames = total_dims // single_obs_dim
            # If it's a single frame, use original logic
            if num_frames == 1:
                output_obs = flip_n2_actor_obs(obs, env)
            else:
                # Split into frames and process each frame
                frames = torch.split(obs, single_obs_dim, dim=-1)
                flipped_frames = [flip_n2_actor_obs(frame, env) for frame in frames]
                output_obs = torch.cat(flipped_frames, dim=-1)
        elif obs_type == "critic":
            output_obs = flip_n2_critic_obs(obs, env)

    if actions is None:
        output_actions = None
    else:
        output_actions = flip_n2_actions(actions)
    return output_obs, output_actions

def flip_n2_actor_obs(obs, env):
    """
        commands: lin_x lin_y ang_yaw       3
        base_ang_vel: roll pitch yaw        3
        projected_gravity                   3   
        dof_pos                             num_dofs
        dof_vel                             num_dofs
        actions                             num_dofs
    """
    if obs is None:
        return obs
    if hasattr(env, "num_single_obs"): single_obs_dim = env.num_single_obs 
    else: single_obs_dim = env.num_obs
    one_step_obs = obs[..., :single_obs_dim]

    flipped_one_step_obs = torch.zeros_like(one_step_obs)
    flipped_one_step_obs[..., :3] = one_step_obs[..., 0:3] 
    flipped_one_step_obs[..., 1] = -flipped_one_step_obs[..., 1]    # cmd lin_vel_y
    flipped_one_step_obs[..., 2] = -flipped_one_step_obs[..., 2]    # cmd ang_vel_z

    flipped_one_step_obs[..., 3:6] = one_step_obs[..., 3:6]         # ang_vel
    flipped_one_step_obs[..., 3] = -flipped_one_step_obs[..., 3]    # ang_vel_x
    flipped_one_step_obs[..., 5] = -flipped_one_step_obs[..., 5]    # ang_vel_z
    # projected_gravity
    flipped_one_step_obs[..., 6:9] = one_step_obs[..., 6:9] * torch.tensor(
        [1.0, -1.0, 1.0], 
        dtype=one_step_obs.dtype, 
        device=one_step_obs.device
    ) 
    flipped_one_step_obs[..., 9:9+env.num_actions] = flip_n2_dof(one_step_obs[..., 9:9+env.num_actions])                                            # dof_pos
    flipped_one_step_obs[..., 9+env.num_actions:9+2*env.num_actions] = flip_n2_dof(one_step_obs[..., 9+env.num_actions:9+2*env.num_actions])        # dof_vel
    flipped_one_step_obs[..., 9+2*env.num_actions:9+3*env.num_actions] = flip_n2_dof(one_step_obs[..., 9+2*env.num_actions:9+3*env.num_actions])    # last_actions

    return flipped_one_step_obs

def flip_n2_critic_obs(obs, env):
    """
        actor_obs                           num_single_obs
        base_lin_vel                        3 
        payload                             1
        friction_coeffs                     1
        restitution_coeffs                  1
        Kp_factors                          num_dofs
        Kd_factors                          num_dofs
        motor_strength                      num_dofs
        contacts                            2
        measure_heights                     len(measured_points_x) * len(measured_points_y)
    """
    if obs is None:
        return obs
    
    if hasattr(env, "num_single_obs"): single_obs_dim = env.num_single_obs 
    else: single_obs_dim = env.num_obs
    actor_obs = obs[..., :single_obs_dim]
    num_actor_obs = single_obs_dim

    flipped_critic_obs = torch.zeros_like(obs)
    flipped_critic_obs[..., :num_actor_obs] = flip_n2_actor_obs(actor_obs, env)

    flipped_critic_obs[..., num_actor_obs:num_actor_obs+3] = obs[..., num_actor_obs:num_actor_obs+3] # base_lin_vel
    flipped_critic_obs[..., num_actor_obs+1] = -flipped_critic_obs[..., num_actor_obs+1] # base_lin_vel_y

    flipped_critic_obs[..., num_actor_obs+3:num_actor_obs+6] = obs[..., num_actor_obs+3:num_actor_obs+6] # payload friction restitution_coeffs

    flipped_critic_obs[..., num_actor_obs+6:num_actor_obs+6+1*env.num_actions] = flip_n2_dof(obs[..., num_actor_obs+6:num_actor_obs+6+1*env.num_actions], is_rd=True) # Kp 
    flipped_critic_obs[..., num_actor_obs+6+1*env.num_actions:num_actor_obs+6+2*env.num_actions] = flip_n2_dof(obs[..., num_actor_obs+6+1*env.num_actions:num_actor_obs+6+2*env.num_actions], is_rd=True) # Kd 
    flipped_critic_obs[..., num_actor_obs+6+2*env.num_actions:num_actor_obs+6+3*env.num_actions] = flip_n2_dof(obs[..., num_actor_obs+6+2*env.num_actions:num_actor_obs+6+3*env.num_actions], is_rd=True) # motor_strength

    flipped_critic_obs[..., num_actor_obs+6+3*env.num_actions:num_actor_obs+6+3*env.num_actions+1] = obs[..., num_actor_obs+6+3*env.num_actions+1:num_actor_obs+6+3*env.num_actions+2]  # contact mask
    flipped_critic_obs[..., num_actor_obs+6+3*env.num_actions+1:num_actor_obs+6+3*env.num_actions+2] = obs[..., num_actor_obs+6+3*env.num_actions:num_actor_obs+6+3*env.num_actions+1]  

    # measured_heights
    if env.cfg.terrain.measure_heights:
        num_x = len(env.cfg.terrain.measured_points_x)  
        num_y = len(env.cfg.terrain.measured_points_y)  
        base_idx = torch.arange(num_y * num_x, device=flipped_critic_obs.device).view(num_y, num_x)
        flip_indices = base_idx.flip(dims=[1]).reshape(-1)
        flipped_critic_obs[..., num_actor_obs+6+3*env.num_actions+2:] = obs[..., num_actor_obs+6+3*env.num_actions+2:][..., flip_indices]
    return flipped_critic_obs

def flip_n2_actions(actions):
    if actions is None:
        return None
    fliped_actions = flip_n2_dof(actions)
    return fliped_actions


def flip_n2_dof(dof, is_rd=False):
    num_actions = dof.shape[-1]
    flipped_dof = torch.zeros_like(dof)
    if num_actions == 10:
        if is_rd:
            flipped_dof[..., 0] = dof[..., 5] # hip yaw
            flipped_dof[..., 5] = dof[..., 0] # hip yaw

            flipped_dof[..., 1] = dof[..., 6] # hip roll
            flipped_dof[..., 6] = dof[..., 1] # hip roll

            flipped_dof[..., 2] = dof[..., 7] # hip pitch
            flipped_dof[..., 7] = dof[..., 2] # hip pitch

            flipped_dof[..., 3] = dof[..., 8] # knee
            flipped_dof[..., 8] = dof[..., 3] # knee

            flipped_dof[..., 4] = dof[..., 9] # ankle
            flipped_dof[..., 9] = dof[..., 4] # ankle
        else:
            flipped_dof[..., 0] = -dof[..., 5] # hip yaw
            flipped_dof[..., 5] = -dof[..., 0] # hip yaw

            flipped_dof[..., 1] = -dof[..., 6] # hip roll
            flipped_dof[..., 6] = -dof[..., 1] # hip roll

            flipped_dof[..., 2] = dof[..., 7] # hip pitch
            flipped_dof[..., 7] = dof[..., 2] # hip pitch

            flipped_dof[..., 3] = dof[..., 8] # knee
            flipped_dof[..., 8] = dof[..., 3] # knee

            flipped_dof[..., 4] = dof[..., 9] # ankle
            flipped_dof[..., 9] = dof[..., 4] # ankle
    elif num_actions == 18:
        if is_rd:
            flipped_dof[..., 0] = dof[..., 9]  # shoulder pitch
            flipped_dof[..., 9] = dof[..., 0]  # shoulder pitch
            
            flipped_dof[..., 1] = dof[..., 10] # shoulder roll
            flipped_dof[..., 10] = dof[..., 1] # shoulder roll

            flipped_dof[..., 2] = dof[..., 11] # shoulder yaw
            flipped_dof[..., 11] = dof[..., 2] # shoulder yaw

            flipped_dof[..., 3] = dof[..., 12]  # elbow
            flipped_dof[..., 12] = dof[..., 3]  # elbow

            flipped_dof[..., 4] = dof[..., 13] # hip yaw
            flipped_dof[..., 13] = dof[..., 4] # hip yaw

            flipped_dof[..., 5] = dof[..., 14] # hip roll
            flipped_dof[..., 14] = dof[..., 5] # hip roll

            flipped_dof[..., 6] = dof[..., 15] # hip pitch
            flipped_dof[..., 15] = dof[..., 6] # hip pitch

            flipped_dof[..., 7] = dof[..., 16] # knee
            flipped_dof[..., 16] = dof[..., 7] # knee

            flipped_dof[..., 8] = dof[..., 17] # ankle
            flipped_dof[..., 17] = dof[..., 8] # ankle
        else: 
            flipped_dof[..., 0] = dof[..., 9]  # shoulder pitch
            flipped_dof[..., 9] = dof[..., 0]  # shoulder pitch
            
            flipped_dof[..., 1] = -dof[..., 10] # shoulder roll
            flipped_dof[..., 10] = -dof[..., 1] # shoulder roll

            flipped_dof[..., 2] = -dof[..., 11] # shoulder yaw
            flipped_dof[..., 11] = -dof[..., 2] # shoulder yaw

            flipped_dof[..., 3] = dof[..., 12]  # elbow
            flipped_dof[..., 12] = dof[..., 3]  # elbow

            flipped_dof[..., 4] = -dof[..., 13] # hip yaw
            flipped_dof[..., 13] = -dof[..., 4] # hip yaw

            flipped_dof[..., 5] = -dof[..., 14] # hip roll
            flipped_dof[..., 14] = -dof[..., 5] # hip roll

            flipped_dof[..., 6] = dof[..., 15] # hip pitch
            flipped_dof[..., 15] = dof[..., 6] # hip pitch

            flipped_dof[..., 7] = dof[..., 16] # knee
            flipped_dof[..., 16] = dof[..., 7] # knee

            flipped_dof[..., 8] = dof[..., 17] # ankle
            flipped_dof[..., 17] = dof[..., 8] # ankle
    elif num_actions == 20:
        if is_rd:
            flipped_dof[..., 0] = dof[..., 10]  # shoulder pitch
            flipped_dof[..., 10] = dof[..., 0]  # shoulder pitch
            
            flipped_dof[..., 1] = dof[..., 11] # shoulder roll
            flipped_dof[..., 11] = dof[..., 1] # shoulder roll

            flipped_dof[..., 2] = dof[..., 12] # shoulder yaw
            flipped_dof[..., 12] = dof[..., 2] # shoulder yaw

            flipped_dof[..., 3] = dof[..., 13]  # elbow
            flipped_dof[..., 13] = dof[..., 3]  # elbow

            flipped_dof[..., 4] = dof[..., 14] # hip yaw
            flipped_dof[..., 14] = dof[..., 4] # hip yaw

            flipped_dof[..., 5] = dof[..., 15] # hip roll
            flipped_dof[..., 15] = dof[..., 5] # hip roll

            flipped_dof[..., 6] = dof[..., 16] # hip pitch
            flipped_dof[..., 16] = dof[..., 6] # hip pitch

            flipped_dof[..., 7] = dof[..., 17] # knee
            flipped_dof[..., 17] = dof[..., 7] # knee

            flipped_dof[..., 8] = dof[..., 18] # ankle pitch
            flipped_dof[..., 18] = dof[..., 8] # ankle pitch

            flipped_dof[..., 9] = dof[..., 19] # ankle roll
            flipped_dof[..., 19] = dof[..., 9] # ankle roll
        else: 
            flipped_dof[..., 0] = dof[..., 10]  # shoulder pitch
            flipped_dof[..., 10] = dof[..., 0]  # shoulder pitch
            
            flipped_dof[..., 1] = -dof[..., 11] # shoulder roll
            flipped_dof[..., 11] = -dof[..., 1] # shoulder roll

            flipped_dof[..., 2] = -dof[..., 12] # shoulder yaw
            flipped_dof[..., 12] = -dof[..., 2] # shoulder yaw

            flipped_dof[..., 3] = dof[..., 13]  # elbow
            flipped_dof[..., 13] = dof[..., 3]  # elbow

            flipped_dof[..., 4] = -dof[..., 14] # hip yaw
            flipped_dof[..., 14] = -dof[..., 4] # hip yaw

            flipped_dof[..., 5] = -dof[..., 15] # hip roll
            flipped_dof[..., 15] = -dof[..., 5] # hip roll

            flipped_dof[..., 6] = dof[..., 16] # hip pitch
            flipped_dof[..., 16] = dof[..., 6] # hip pitch

            flipped_dof[..., 7] = dof[..., 17] # knee
            flipped_dof[..., 17] = dof[..., 7] # knee

            flipped_dof[..., 8] = dof[..., 18] # ankle pitch
            flipped_dof[..., 18] = dof[..., 8] # ankle pitch

            flipped_dof[..., 9] = -dof[..., 19] # ankle roll
            flipped_dof[..., 19] = -dof[..., 9] # ankle roll
    else:
        assert False, "Unsupported num joints"

    return flipped_dof
