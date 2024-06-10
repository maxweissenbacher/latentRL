import sys
import os
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    InitTracker,
    FiniteTensorDictCheck,
    TransformedEnv,
    VecNorm,
    ParallelEnv,
    Compose,
    ObservationNorm,
    EnvCreator,
)

cwd = os.getcwd()
sys.path.append(cwd + "/../")
from Cylinder_Env.simulation_base.env import resume_env
from torchrl.envs.utils import check_env_specs


def make_cylinder_env(device, cfg, n_env=1, sim_log_name = "Sim"):
    from torchrl.envs.libs.gym import GymWrapper
    # Create the 2D cylinder Gym environment here
    env = resume_env(plot=False,
                     single_run=False,
                     horizon=cfg.cyl.horizon,
                     dump_vtu=cfg.cyl.dump_vtu, 
                     random_start= cfg.cyl.random_start,
                     n_env=n_env,
                     simulation_duration=cfg.cyl.simulation_duration,
                     sim_log_name = sim_log_name
                     )
    env = GymWrapper(env, device=device)
    return env


def make_parallel_cylinder_env(cfg, exp_name):
    # make_env_fn = EnvCreator(lambda: make_cylinder_env(cfg.collector.device, cfg))
    # env = ParallelEnv(cfg.cyl.num_envs, make_env_fn)
    
    # env = ParallelEnv(cfg.cyl.num_envs, make_env_fn)
    env = ParallelEnv(cfg.cyl.num_envs, 
                      [lambda: make_cylinder_env(cfg.collector.device, cfg, n_env, sim_log_name = exp_name + "/train") for n_env in range(cfg.cyl.num_envs)])  
    # env = ParallelEnv(cfg.cyl.num_envs, [lambda: make_ks for i in range(num_envs)])
    return env


def make_cylinder_eval_env(cfg, exp_name):
    test_env = make_cylinder_env(cfg.collector.device, cfg, sim_log_name = exp_name + "/eval")
    test_env.eval()
    return test_env


if __name__ == '__main__':
    # Create torchrl env
    env = make_cylinder_env(device='cpu')

    # Test reset
    print(env.reset())

    # Check env specs
    check_env_specs(env)

    # Do a rollout
    print(env.rollout(max_steps=10))

