import sys
import os
cwd = os.getcwd()
sys.path.append(cwd + "/../")

from Cylinder_Env.simulation_base.env_raw_bp import resume_env
from torchrl.envs.utils import check_env_specs


def make_cylinder_torchrl_env(device):
    from torchrl.envs.libs.gym import GymWrapper
    # Create the 2D cylinder Gym environment here
    env = resume_env(plot=False, single_run=True, horizon=400, dump_vtu=100, n_env=99)
    env = GymWrapper(env, device=device)
    return env


if __name__ == '__main__':
    # Create torchrl env
    env = make_cylinder_torchrl_env(device='cpu')

    # Test reset
    print(env.reset())

    # Check env specs
    check_env_specs(env)

    # Do a rollout
    print(env.rollout(max_steps=10))

