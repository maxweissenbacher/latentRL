from env.cylinder_flow_env import make_cylinder_torchrl_env
from torchrl.envs.utils import check_env_specs


if __name__ == '__main__':
    # Create torchrl env
    env = make_cylinder_torchrl_env(device='cpu')

    # Test reset
    print(env.reset())

    # Check env specs
    check_env_specs(env)

    # Do a rollout
    print(env.rollout(max_steps=10))

