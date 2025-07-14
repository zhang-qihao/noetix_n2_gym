import glob
from humanoid.envs.n2.n2_18dof_amp_config import N2_18DofAMPCfg, N2_18DofAMPCfgPPO

class N2JumpingCfg(N2_18DofAMPCfg):
    """
    Configuration class for the ning humanoid robot.
    """
    class env(N2_18DofAMPCfg.env):
        # change the observation dim
        frame_stack = 5
        num_single_obs = 62
        num_privileged_obs = 126 + 96
        num_observations = int(frame_stack * num_single_obs)
        num_actions = 18

        episode_length_s = 8  # episode length in seconds

        enable_early_termination = True
        termination_height = 0.4
        termination_orientation = 0.4

        reset_landing_error = 0.1

        # reference_setting
        reference_state_initialization = False
        prob_rsi = 0.7

    class domain_rand(N2_18DofAMPCfg.domain_rand):
        randomize_gains = True
        p_gain_range = [0.8, 1.2]
        d_gain_range = [0.8, 1.2]

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_friction = True
        friction_range = [0.1, 2.]

        randomize_restitution = True
        restitution_range = [0., 1.]

        randomize_base_mass = True
        added_mass_range = [-5., 5.]

        disturbance = False
        push_force_range = [50.0, 300.0]
        push_torque_range = [25.0, 100.0]
        disturbance_probabilities = 0.002
        disturbance_interval = [10, 25] # * dt * decimation ms

    class terrain(N2_18DofAMPCfg.terrain):
        mesh_type = 'plane' # plane trimesh
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        restitution = 0.

    class commands:
        curriculum = False
        min_cmd_vel = 0
        num_commands = 3
        toggle_time = [2, 5]
        resampling_time = [3, 6]
        class ranges(): 
            # The command distances are relative to the initial agent position and are sampled from
            # the ranges below:
            # This is the min/maximum ranges in the jump's distance curriculum (x_des = dx~pos_dx + x)
            pos_dx = [0.0, 0.0]
            pos_dy = [-0.0, 0.0]
            pos_dw = [-0.0, 0.0]
        
    class rewards:           
        phase_time = 2.0
        only_positive_rewards = True
        max_contact_force = 800  # forces above this value are penalized
        target_flight_time = 0.5
        target_height = 0.7
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.8
        jump_goal = 100.

        class scales:
            # #---------- Task rewards (once per episode): ----------- #
            # task_max_height = 1000.0 # Reward for maximum height (minus minimum height) achieved
            # task_pos = 400.0 # Final reward is scale * (dt*decimation)
            # task_ori = 400.0

            # termination = -40.
            # jumping = 100.

            # #---------- Continuous rewards (at every time step): ----------- #
            # ang_vel_z = -1.
           
            # post_landing_pos = 6. # Reward for maintaining landing position
            # post_landing_ori = 6. # Reward for returning to desired orientation after landing

            # base_height_flight = 80
            # base_height_stance = 10

            # stage 1 reward_coef = 20 reward_lerp = 0.8
            task_max_height = 0.0 # Reward for maximum height (minus minimum height) achieved
            lin_vel_z = 15.0
            stand_still = -0.5
            orientation = 0.5
            feet_height = 50.0
            # jump_up = 1.2

            # stage 2 reward_coef = 50 reward_lerp = 0.5
            # task_max_height = 2000.0 # Reward for maximum height (minus minimum height) achieved
            # lin_vel_z = 100.0
            change_of_contact = 0.5
            base_acc = 0.0
            # landing_buffer = 50.0

            jumping = 0.0
            tracking_lin_vel = 1.2
            tracking_ang_vel = 0.8

            joint_pos_symmetry = -0.02
            default_joint_pos = 1.0

            # ---------- Regularisation rewards ----------- #
            # energy
            dof_acc = -2.5e-7
            energy_cost = -1e-3
            action_rate = -0.01
            action_smoothness = -0.01

            # other
            collision = -10.0
            dof_pos_limits = -10.0
            dof_vel_limits = -5.0
            feet_contact_forces = -0.01
    
    class noise:
        add_noise = True
        noise_level = 1.0    # 1 scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.2
            lin_vel = 0.05
            gravity = 0.05
            quat = 0.05
            height_measurements = 0.1


    class motion_loader:
        motion_loader_name = "MotionLoaderNing"
        reference_motion_file  = sorted(glob.glob("datasets/mocap_motions/ning/jumping/*"))
        num_preload_transitions = 500000
        reference_observation_horizon = 4


class NingAMPJumpingCfgPPO(N2_18DofAMPCfgPPO):
    runner_class_name = "CAMPHIMOnPolicyRunner"
    class discriminator:
        reward_coef = 50
        reward_lerp = 0.8   # reward = (1 - reward_lerp) * style_reward + reward_lerp * task_reward
        style_reward_function = "quad_mapping" # log_mapping, quad_mapping, wasserstein_mapping
        normalize_style_reward = False
        shape = [1024, 512]

    class algorithm(N2_18DofAMPCfgPPO.algorithm):
        class_name = "CAMP_HIM_PPO" 
        entropy_coef = 0.01

        # AMP setting 
        discriminator_learning_rate = 1e-5
        discriminator_momentum = 0.5
        discriminator_weight_decay = 1e-3
        discriminator_gradient_penalty_coef = 1
        discriminator_logit_reg_coef = 0.0
        discriminator_weight_decay_coef = 0.0
        discriminator_num_mini_batches = 80
        amp_replay_buffer_size = 1000000
        discriminator_loss_function = "MSELoss" # BCEWithLogitsLoss, MSELoss, WassersteinLoss

        # symmetry setting
        symmetry_cfg = {
            "use_data_augmentation": False,
            "use_mirror_loss": False,
            "data_augmentation_func": "humanoid.envs.n2.n2_sym_utils:data_augmentation_func_n2",
            "mirror_loss_coeff": 1.0
        }
    
    class runner( N2_18DofAMPCfgPPO.runner ):
        max_iterations = 20000
        run_name = ''
        experiment_name = 'n2_jumping'
