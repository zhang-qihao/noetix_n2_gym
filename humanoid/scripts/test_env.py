from multiprocessing import Process

from humanoid.envs import *
from humanoid.utils import  get_args, task_registry
import matplotlib.pyplot as plt

import torch
import numpy as np

def generate_sine_wave(duration, sample_rate=1000, frequency=30, amplitude=1):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    y = amplitude * np.sin(2 * np.pi * frequency * t)
    return t, y

def plot_states(_plot, env, desired_times, desired_angle, actual_times, actual_angle, actual_vel, actual_torque):
    plot_process = Process(target=_plot(env, desired_times, desired_angle, actual_times, actual_angle, actual_vel, actual_torque))
    plot_process.start()

def _plot(env, desired_times, desired_angle, actual_times, actual_angle, actual_vel, actual_torque):
    if env.num_dofs == 10:
        nb_rows = 2
        nb_cols = 5
    elif env.num_dofs == 18:
        nb_rows = 3
        nb_cols = 6
    elif env.num_dofs == 20:
        nb_rows = 4
        nb_cols = 5
    else:
        assert False, "Unsupported num joints"
    
    # Plotting the desired vs actual angles
    plt.figure(figsize=(18, 10))
    for i in range(nb_rows):
        for j in range(nb_cols):
            if i * nb_cols + j < desired_angle.shape[1]:
                plt.subplot(nb_rows, nb_cols, i * nb_cols + j + 1)
                plt.plot(desired_times[:-1], desired_angle[:, i*nb_cols+j], label="Desired angle", color="blue")
                plt.plot(actual_times, actual_angle[:, i*nb_cols+j], label="Actual angle", color="red", linestyle="--")
                plt.xlabel("Time (s)")
                plt.ylabel("Actual angle (rad)")
                plt.title(env.dof_names[i*nb_cols+j])
                plt.grid(True)
    plt.tight_layout()
    plt.savefig('pd_dof_pos.png')

    # Plotting the joint velocities
    plt.figure(figsize=(18, 10))
    for i in range(nb_rows):
        for j in range(nb_cols):
            if i * nb_cols + j < desired_angle.shape[1]:
                plt.subplot(nb_rows, nb_cols, i * nb_cols + j + 1)
                plt.plot(actual_times, actual_vel[:, i*nb_cols+j], label="Dof vel", color="black")
                plt.xlabel("Time (s)")
                plt.ylabel("Actual vel (rad/s)")
                plt.title(env.dof_names[i*nb_cols+j])
                plt.grid(True)
    plt.tight_layout()
    plt.savefig('pd_dof_vel.png')

    # Plotting the torques
    plt.figure(figsize=(18, 10))
    for i in range(nb_rows):
        for j in range(nb_cols):
            if i * nb_cols + j < desired_angle.shape[1]:
                plt.subplot(nb_rows, nb_cols, i * nb_cols + j + 1)
                plt.plot(actual_times, actual_torque[:, i*nb_cols+j], label="Torque", color="green")
                plt.xlabel("Time (s)")
                plt.ylabel("Actual torque (rad)")
                plt.title(env.dof_names[i*nb_cols+j])
                plt.grid(True)
    plt.tight_layout()
    plt.savefig('pd_torques.png')

def make_env(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs =  min(env_cfg.env.num_envs, 1)
    env_cfg.sim.physx.max_gpu_contact_pairs = 2**10
    env_cfg.asset.fix_base_link = True
    env_cfg.asset.self_collisions = 1
    env_cfg.env.reference_state_initialization = False
    env_cfg.env.enable_early_termination = False
    env_cfg.terrain.mesh_type = 'plane' # plane trimesh
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    return env

def test_env(args):
    env = make_env(args)
    # pd sin wave setting
    action_scale = torch.tensor([1] * env.num_actions, device=env.device) # ning
    joint_index = torch.arange(0, env.num_actions, 1, device=env.device)
    # generate_sine_wave
    T = 4
    frequency = 1
    dt = env.dt
    times = torch.arange(0, T, dt / frequency)

    desired_times = times.clone()
    actual_times = desired_times.clone()
    actual_times = actual_times[::frequency].cpu().numpy()
    _ , generated_sin = generate_sine_wave(duration=T, frequency=frequency, amplitude=1, sample_rate=frequency/dt)
    generated_sin = torch.tensor(generated_sin, device=env.device) 
    desired_angle = torch.zeros((generated_sin.size()[0], env.num_dofs), device=env.device)
    for idx in joint_index:
        desired_angle[:, idx] = generated_sin
    desired_angle *= action_scale 
    
    # step and plot
    actions = torch.zeros(env.num_envs, env.num_dofs, dtype=torch.float, device=env.device, requires_grad=False)
    actual_angle = []
    actual_vel = []
    actual_torque = []
    for i in range(len(times)-1):
        actions[0, :] = desired_angle[i] 
        if i % frequency == 0:
            env.step(actions)
            dof_pos, dof_vel, torques = env.dof_pos.clone(), env.dof_vel.clone(), env.torques.clone()
            actual_angle.append(dof_pos[0].cpu().numpy())
            actual_vel.append(dof_vel[0].cpu().numpy() )
            actual_torque.append(torques[0].cpu().numpy())
    actual_angle = np.asarray(actual_angle)
    actual_vel = np.asarray(actual_vel)
    actual_torque = np.asarray(actual_torque)
    desired_angle = (desired_angle * env.cfg.control.action_scale + env.default_dof_pos).cpu().numpy()

    if actual_times.size != len(actual_angle):
        actual_times = actual_times[:-1]
    plot_states(_plot, env, desired_times, desired_angle, actual_times, actual_angle, actual_vel, actual_torque)
    print("Done")

if __name__ == '__main__':
    args = get_args()
    test_env(args)
