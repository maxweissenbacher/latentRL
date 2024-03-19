import torch
import torch.nn as nn
import numpy as np
from solver.ks_solver import KS


def rollout(u0, policy, num_steps, **kwargs):
    """
    Compute a rollout with any initial condition and policy
    :param u0: Initial condition. Torch tensor
    :param policy: Function mapping a torch tensor of same shape as u0 to actions
    :param num_steps: Number of rollout steps
    :param kwargs: Arguments to specify the KS class
    :return: a Numpy array of shape [num_steps, N] where N is the spatial resolution of the solver
    """
    solver = KS(**kwargs)
    if not u0.shape[0] == solver.n:
        raise ValueError(f"Solution u0 must have shape [{solver.n}]. Got {u0.shape}")
    uu = [u0]
    u = u0
    for _ in range(num_steps-1):
        action = policy(u)
        u = solver.advance(u, action)
        uu.append(u)
    uu = np.asarray(uu)
    return uu


class StaticPolicy(nn.Module):
    def __init__(self, values):
        super().__init__()
        assert isinstance(values, torch.Tensor)
        self.action = values

    def forward(self, u):
        return self.action


class RandomPolicy(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

    def forward(self, u):
        return torch.randn(self.num_actions)


if __name__ == '__main__':
    # Specify the arguments for the KS solver
    ks_args = {
        "actuator_locs": torch.tensor([0., torch.pi]),
        "actuator_scale": 0.1,
        "nu": 0.08,
        "N": 256,
        "dt": 0.5,
    }

    # Make a list of policies
    # A policy is a function which maps a full state u to an action a
    random_policy = RandomPolicy(num_actions=2)
    static_policy = StaticPolicy(torch.tensor([2.0, 2.0]))

    # Define the initial condition
    u0 = torch.zeros(ks_args["N"])

    # Compute a rollout
    out = rollout(u0, random_policy, 100, **ks_args)

    print(out.shape)

