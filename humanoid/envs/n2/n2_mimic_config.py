import glob
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class N2MimicCfg(LeggedRobotCfg):
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
            "L_leg_ankle_joint": -0.1720,

            "R_arm_shoulder_pitch_joint": 0.,
            "R_arm_shoulder_roll_joint": -0.2,
            "R_arm_shoulder_yaw_joint": 0.,
            "R_arm_elbow_joint": 0.,
            "R_leg_hip_yaw_joint": 0.,
            "R_leg_hip_roll_joint": 0.,
            "R_leg_hip_pitch_joint": -0.1495,
            "R_leg_knee_joint": 0.3215,
            "R_leg_ankle_joint": -0.1720,
        }

    class env(LeggedRobotCfg.env):
        frame_stack = 5
        num_single_obs = 62 
        num_privileged_obs = 250 + 96
        num_observations = int(frame_stack * num_single_obs)
        num_actions = 18
        episode_length_s = 45
    
    class termination:
        terminate_by_gravity = False
        terminate_by_fallen = True
        terminate_when_motion_end = False
        terminate_when_motion_far = True

        class scales:
            termination_gravity = 0.9
            termination_motion_far_threshold = 1.5
            termination_height = 0.4

        class termination_curriculum:
            terminate_when_motion_far_curriculum = True
            terminate_when_motion_far_initial_threshold = 1.5
            terminate_when_motion_far_threshold_max = 2.0
            terminate_when_motion_far_threshold_min = 0.3
            terminate_when_motion_far_curriculum_degree = 2.5e-05
            terminate_when_motion_far_curriculum_level_down_threshold = 40
            terminate_when_motion_far_curriculum_level_up_threshold = 42
    
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
            "arm_shoulder_pitch_joint": 30.0, "arm_shoulder_roll_joint": 30.0, "arm_shoulder_yaw_joint": 30.0, "arm_elbow_joint": 30.0,
            'leg_hip_yaw_joint': 80.0, 'leg_hip_roll_joint': 80.0, 'leg_hip_pitch_joint': 120.0,
            'leg_knee_joint': 120.0, 'leg_ankle_joint': 20.0
        }
        damping = {
            "arm_shoulder_pitch_joint": 1.0, "arm_shoulder_roll_joint": 1.0, "arm_shoulder_yaw_joint": 1.0, "arm_elbow_joint": 1.0,
            'leg_hip_yaw_joint': 5.0, 'leg_hip_roll_joint': 5.0, 'leg_hip_pitch_joint': 5.0,
            'leg_knee_joint': 5.0, 'leg_ankle_joint': 2.0
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10 
    
    class sim(LeggedRobotCfg.sim):
        dt =  0.002

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/N2/urdf/N2.urdf'
        name = "Ning"
        foot_name = "ankle"
        key_name = [""]
        upper_body_name = ["arm"]
        lower_body_name = ["leg"]
        penalize_contacts_on = ["base", "knee", "hip", "hand", "arm"]
        terminate_after_contacts_on = []
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

        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        max_init_terrain_level = 0 #10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # terrain_proportions = [0.6, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0]
        # terrain_proportions = [0.3, 0.1, 0.0, 0.2, 0.2, 0.1, 0.1]
        restitution = 0.
    
    class rewards:
        soft_dof_pos_limit = 0.90
        soft_dof_vel_limit = 0.80
        base_height_target = 0.70
        max_contact_force = 400. # forces above this value are penalized
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        use_vec_reward = True

        enable_adaptive_tracking_sigma = True
        tracking_sigma_alpha = 0.001
        tracking_sigma_scale = 1.0
        tracking_sigma_type = "origin"
        reward_tracking_sigma = {
            "tracking_upper_body_pos": 0.015,
            "tracking_lower_body_pos": 0.015,
            "tracking_body_vel": 1.0,
            "tracking_body_ang_vel": 15.0,
            "tracking_feet_pos": 0.01,
            "tracking_max_joint_pos": 1.0,
            "tracking_joint_pos": 0.3,
            "tracking_joint_vel": 30.0,
        }
        
        reward_penalty_curriculum = True
        reward_initial_penalty_scale = 0.1
        reward_min_penalty_scale = 0.0
        reward_max_penalty_scale = 1.0
        reward_penalty_level_down_threshold = 40
        reward_penalty_level_up_threshold = 42
        reward_penalty_degree = 1.0e-05
        reward_penalty_reward_names = [
            "contact_no_vel",
            "feet_contact_forces",
            "dof_acc",
            "energy_cost",
            "action_smoothness",
            "action_rate",
            "dof_pos_limits",
            "dof_vel_limits"
        ]

        num_compute_average_epl = 10000

        class scales:
            ################ reward ################
            # motion tracking
            tracking_body_pos = 1.0
            tracking_body_vel = 0.5
            tracking_body_ang_vel = 0.5
            tracking_feet_pos = 1.5
            tracking_max_joint_pos = 1.0
            tracking_joint_pos = 1.0
            tracking_joint_vel = 1.0
            tracking_contact_mask = 0.5
            # other
            feet_air_time = 1.0
            ################ penalty ################
            # contact 
            contact_no_vel = -5.0
            feet_contact_forces = -0.01
            # energy
            dof_acc = -2.5e-7
            energy_cost = -1e-3
            action_smoothness = -0.05
            action_rate = -0.1
            # other
            collision = -10.0
            dof_pos_limits = -5.0
            dof_vel_limits = -5.0
        

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
        motion_loader_name = "MotionLoaderNingTracking"
        reference_motion_file  = glob.glob("datasets/mocap_motions/ning/dancing/*")
        num_preload_transitions = 10
        reference_observation_horizon = 2

class N2MimicCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'MHOnPolicyRunner' 
    class policy:
        class_name = 'HIMActorCritic' 

        actor_hidden_dims = [1024, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        
        init_noise_std = 1.0
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        class_name = "MH_PPO" 
        entropy_coef = 0.005

        # symmetry setting
        symmetry_cfg = {
            "use_data_augmentation": False,
            "use_mirror_loss": False,
            "data_augmentation_func": "humanoid.envs.n2.n2_sym_utils:data_augmentation_func_n2",
            "mirror_loss_coeff": 2.0
        }

    class runner( LeggedRobotCfgPPO.runner ):
        max_iterations = 200000
        save_interval = 500 
        run_name = ''
        experiment_name = 'n2_mimic'
        init_at_random_ep_len = True
