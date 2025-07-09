import json
from humanoid.envs import *
from humanoid.utils import  get_args, task_registry
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.sim.physx.max_gpu_contact_pairs = 2**10
    env_cfg.terrain.mesh_type = 'plane' # plane trimesh
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.disturbance_probabilities = 0.005
    env_cfg.domain_rand.push_force_range = [50.0, 500.0]
    env_cfg.domain_rand.push_torque_range = [0.0, 0.0]

    env_cfg.env.test = True
    env_cfg.env.episode_length_s = 100


    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    record_duration = 9 # s
    stop_state_log = round(record_duration / env.dt)
    motion_frames = []
    for i in tqdm(range(stop_state_log)):
        actions = policy(obs.detach()) # * 0.

        if i < 250: 
            env.commands[:, 0] = 0.4
        else:
            env.commands[:, 0] = 0.002 * i - 0.1
        # if i < 350: 
        #     env.commands[:, 0] = 1.2
        # elif i < 450:
        #     env.commands[:, 0] = 1.6
        # elif i < 550:
        #     env.commands[:, 0] = 2.0
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        env.commands[:, 3] = 0.0

        obs = env.step(actions.detach())[0]

        root_pos = env.root_states[0, :3].clone()
        root_rot= env.root_states[0, 3:7].clone()
        root_vel = env.root_states[0, 7:10].clone()
        root_ang_vel = env.root_states[0, 10:13].clone()
        dof_pos = env.dof_pos[0]
        dof_pos[[4, 5, 14, 15]] = 0
        dof_vel = env.dof_vel[0]
        dof_vel[[4, 5, 14, 15]] = 0
        root_ang_vel[2] = 0
        key_pos = (env.key_pos.squeeze(0) - root_pos).flatten()
        base_height = root_pos[2].clone().unsqueeze(0)

        cur_frames_tensor = torch.cat((root_pos, root_rot, dof_pos, key_pos, root_vel, root_ang_vel, dof_vel, base_height), dim=0)
        cur_frames = cur_frames_tensor.tolist()
        motion_frames.append(cur_frames)

    motion_data = {}
    motion_data["LoopMode"] = "Wrap"
    motion_data["FrameDuration"] = env.dt
    motion_data["EnableCycleOffsetPosition"] = "true"
    motion_data["EnableCycleOffsetRotation"] = "true"
    motion_data["MotionWeight"] = 0.5
    motion_data["Frames"] = motion_frames[250:]

    # export motion capture to json format   
    with open("datasets/mocap_motions/" + ROBOT + "/" + MOTION_FILE, "w") as f:
        json.dump(motion_data, f, indent=1, separators=(",", ":"))
        print("Generating " + MOTION_FILE)

if __name__ == '__main__':
    ROBOT = "ning"
    MOTION_FILE = ROBOT + "_walk_20dof.txt" 
    args = get_args()
    play(args)
