import glob
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class N2CSICfg(LeggedRobotCfg):
    """
    Configuration class for the N2 humanoid robot.
    """
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.75]
        default_joint_angles = {
            "L_arm_shoulder_pitch_joint": 0.,
            "L_arm_shoulder_roll_joint": 0.2,
            "L_arm_shoulder_yaw_joint": 0.,
            "L_arm_elbow_joint": 0.,
            "L_leg_hip_yaw_joint": 0.,
            "L_leg_hip_roll_joint": 0.,
            "L_leg_hip_pitch_joint": -0.1495,
            "L_leg_knee_joint": 0.3215,
            "L_leg_ankle_pitch_joint": -0.1720,
            "L_leg_ankle_roll_joint": 0.,
            "R_arm_shoulder_pitch_joint": 0.,
            "R_arm_shoulder_roll_joint": -0.2,
            "R_arm_shoulder_yaw_joint": 0.,
            "R_arm_elbow_joint": 0.,
            "R_leg_hip_yaw_joint": 0.,
            "R_leg_hip_roll_joint": 0.,
            "R_leg_hip_pitch_joint": -0.1495,
            "R_leg_knee_joint": 0.3215,
            "R_leg_ankle_pitch_joint": -0.1720,
            "R_leg_ankle_roll_joint": 0.
        }

    class env(LeggedRobotCfg.env):
        frame_stack = 5
        num_single_obs = 69 
        num_privileged_obs = 137 + 96 
        num_observations = int(frame_stack * num_single_obs)
        num_actions = 20

        enable_early_termination = True
        termination_height = 0.4

        # reference_setting
        reference_state_initialization = True
        prob_rsi = 1.0
    
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

        disturbance = False
        push_force_range = [50.0, 300.0]
        push_torque_range = [25.0, 100.0]
        disturbance_probabilities = 0.002
        disturbance_interval = [10, 25] # * dt * decimation ms
    
    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {
            "arm_shoulder_pitch_joint": 30.0, "arm_shoulder_roll_joint": 30.0, "arm_shoulder_yaw_joint": 30.0, "arm_elbow_joint": 30.0,
            'leg_hip_yaw_joint': 80.0, 'leg_hip_roll_joint': 80.0, 'leg_hip_pitch_joint': 120.0,
            'leg_knee_joint': 120.0, 'leg_ankle_pitch_joint': 20.0, 'leg_ankle_roll_joint': 20.0
        }
        damping = {
            "arm_shoulder_pitch_joint": 1.0, "arm_shoulder_roll_joint": 1.0, "arm_shoulder_yaw_joint": 1.0, "arm_elbow_joint": 1.0,
            'leg_hip_yaw_joint': 5.0, 'leg_hip_roll_joint': 5.0, 'leg_hip_pitch_joint': 5.0,
            'leg_knee_joint': 5.0, 'leg_ankle_pitch_joint': 2.0, 'leg_ankle_roll_joint': 2.0
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10 
    
    class sim(LeggedRobotCfg.sim):
        dt =  0.002

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/N2_20dof/urdf/N2_20dof.urdf'
        name = "Ning"
        foot_name = "ankle_roll"
        key_name = ["hand", "ankle_roll"]
        penalize_contacts_on = ["hip", "hand", "arm"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False
 
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane' # plane trimesh
        curriculum = False
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4]
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 4.
        terrain_width = 4.

        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 0 #10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        # terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        terrain_proportions = [0.4, 0.0, 0.0, 0.2, 0.2, 0.1, 0.1]
        # terrain_proportions = [0.3, 0.1, 0.0, 0.2, 0.2, 0.1, 0.1]
        restitution = 0.
    
    class rewards:
        soft_dof_pos_limit = 0.90
        soft_dof_vel_limit = 0.80
        base_height_target = 0.70
        max_contact_force = 400. # forces above this value are penalized
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        
        class scales:
            # contact 
            feet_contact_forces = -0.2
            # energy
            action_smoothness = -0.5 
            energy_cost = -5e-2
            collision = -100
            termination = -100

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
        num_commands = 1
        resampling_time = [2, 8]  # time before command are changed[s]
        class ranges(): 
            None

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 0  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2





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

    class motion_loader:
        motion_loader_name = "MotionLoaderNing20DOF"
        reference_motion_file  = sorted(glob.glob("datasets/mocap_motions/ning_20dof/csi/*"))
        num_preload_transitions = 500000
        reference_observation_horizon = 4

class N2CSICfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "CSIOnPolicyRunner"
    class policy:
        class_name = 'HIMActorCritic' 

        actor_hidden_dims = [1024, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        
        init_noise_std = 1.0
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class discriminator:
        reward_coef = 4.0
        reward_lerp = 0.8  # reward = reward_coef * (1 - reward_lerp) * style_reward + reward_lerp * task_reward
        style_reward_function = "wasserstein_mapping" # log_mapping, quad_mapping, wasserstein_mapping
        normalize_style_reward = True
        shape = [1024, 512]

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        class_name = "AMP_HIM_PPO" 
        entropy_coef = 0.01

        # AMP setting 
        discriminator_learning_rate = 5e-6
        discriminator_momentum = 0.5
        discriminator_weight_decay = 1e-3
        discriminator_gradient_penalty_coef = 5
        discriminator_logit_reg_coef = 0.0
        discriminator_weight_decay_coef = 0
        discriminator_num_mini_batches = 80
        amp_replay_buffer_size = 1000000
        discriminator_loss_function = "WassersteinLoss" # BCEWithLogitsLoss, MSELoss, WassersteinLoss

        # symmetry setting
        symmetry_cfg = {
            "use_data_augmentation": False,
            "use_mirror_loss": False,
            "data_augmentation_func": "humanoid.envs.n2.n2_sym_utils:data_augmentation_func_n2",
            "mirror_loss_coeff": 2.0
        }

    class runner( LeggedRobotCfgPPO.runner ):
        max_iterations = 20000
        run_name = ''
        experiment_name = 'n2_csi_20dof'
