from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class N2_10DofCfg(LeggedRobotCfg):
    """
    Configuration class for the N2 humanoid robot.
    """
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.75]
        default_joint_angles = {
            "L_leg_hip_yaw_joint": 0.,
            "L_leg_hip_roll_joint": 0.,
            "L_leg_hip_pitch_joint": -0.1495,
            "L_leg_knee_joint": 0.3215,
            "L_leg_ankle_joint": -0.1720,
            "R_leg_hip_yaw_joint": 0.,
            "R_leg_hip_roll_joint": 0.,
            "R_leg_hip_pitch_joint": -0.1495,
            "R_leg_knee_joint": 0.3215,
            "R_leg_ankle_joint": -0.1720,
        }

    class env(LeggedRobotCfg.env):
        # frame_stack = 5
        num_single_obs = 39 
        num_privileged_obs = 77 + 96 
        # num_observations = int(frame_stack * num_single_obs)
        num_observations = num_single_obs
        num_actions = 10

        enable_early_termination = True
        termination_height = 0.4
    
    class domain_rand(LeggedRobotCfg.domain_rand):
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

        disturbance = True
        push_force_range = [50.0, 300.0]
        push_torque_range = [25.0, 100.0]
        disturbance_probabilities = 0.002
        disturbance_interval = [10, 25] # * dt * decimation ms
    
    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {
            'leg_hip_yaw_joint': 80.0, 'leg_hip_roll_joint': 80.0, 'leg_hip_pitch_joint': 120.0,
            'leg_knee_joint': 120.0, 'leg_ankle_joint': 20.0
        }
        damping = {
            'leg_hip_yaw_joint': 5.0, 'leg_hip_roll_joint': 5.0, 'leg_hip_pitch_joint': 5.0,
            'leg_knee_joint': 5.0, 'leg_ankle_joint': 1.0
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10 
    
    class sim(LeggedRobotCfg.sim):
        dt =  0.002

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/N2/urdf/N2_10dof.urdf'
        name = "Ning"
        foot_name = "ankle"
        
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False
    
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # plane trimesh
        curriculum = False
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4]
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 4.
        terrain_width = 4.

        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        max_init_terrain_level = 0 #10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        restitution = 0.
    
    class rewards:
        soft_dof_pos_limit = 0.9
        base_height_target = 0.698
        max_contact_force = 300. # forces above this value are penalized
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        
        class scales:
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.0
            # base pos
            orientation = 1.0
            base_height = -30.0
            # style
            stand_still = -0.15
            feet_air_time = 2.
            default_joint_pos = 1.0
            # contact 
            feet_contact = 1.0
            contact_no_vel = -2
            feet_contact_forces = -0.05
            # energy
            dof_acc = -2.5e-7
            energy_cost = -1e-3
            action_smoothness = -0.01
            # other
            collision = 0.0
            dof_pos_limits = -5.0

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

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = [5, 15] # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        min_cmd_vel = 0.2
        class ranges:
            lin_vel_x = [-0.8, 0.8] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.

class N2_10DofCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'OnPolicyRunner'
    class policy:
        # class_name = 'ActorCritic' 

        # actor_hidden_dims = [1024, 256, 128]
        # critic_hidden_dims = [768, 256, 128]
        
        # init_noise_std = 1.0
        # activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        class_name = "ActorCriticRecurrent"
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128]
        critic_hidden_dims = [256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 256
        rnn_num_layers = 1

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        class_name = "PPO"
        entropy_coef = 0.01

        # symmetry setting
        # symmetry_cfg = {
        #     "use_data_augmentation": False,
        #     "use_mirror_loss": True,
        #     "data_augmentation_func": "humanoid.envs.n2.n2_sym_utils:data_augmentation_func_n2",
        #     "mirror_loss_coeff": 1.0
        # }

    class runner( LeggedRobotCfgPPO.runner ):
        max_iterations = 10000
        run_name = ''
        experiment_name = 'n2'
