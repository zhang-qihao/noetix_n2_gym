# 导入所有环境相关模块
from humanoid.envs import *
# 导入参数解析和任务注册工具
from humanoid.utils import get_args, task_registry

def train(args):
    """
    训练函数：根据提供的参数执行强化学习训练
    
    参数:
        args: 命令行参数对象，包含训练所需的各种配置
    """
    # 根据任务名称和参数创建环境实例
    # env: 环境对象，用于模拟和交互
    # env_cfg: 环境配置对象，包含环境的具体配置参数
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # 创建算法运行器实例
    # ppo_runner: PPO算法运行器对象，负责执行训练过程
    # train_cfg: 训练配置对象，包含训练算法的具体参数
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # 开始训练过程
    # num_learning_iterations: 最大训练迭代次数
    # init_at_random_ep_len: 是否在随机episode长度初始化
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=train_cfg.runner.init_at_random_ep_len)

# 程序入口点
if __name__ == '__main__':
    # 解析命令行参数
    args = get_args()
    # 调用训练函数开始训练
    train(args)