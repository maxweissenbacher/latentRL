import torch
import torch.nn as nn
import numpy as np
from solver.ks_solver import KS
from utils.plotting import contourplot_KS
import matplotlib.pyplot as plt


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

    uu = []
    actions = []
    forcings = []

    # Initialisation
    u = u0
    action = policy(u0)
    forcing = solver.compute_forcing(action)

    # Loop and append results to lists
    for _ in range(num_steps):
        uu.append(u)
        actions.append(action)
        forcings.append(forcing)
        u = solver.advance(u, action)
        action = policy(u)
        forcing = solver.compute_forcing(action)

    uu = np.asarray(uu)
    actions = np.asarray(actions)
    forcings = np.asarray(forcings)
    return uu, actions, forcings


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
        "actuator_locs": torch.tensor(np.linspace(0.0, 2*np.pi, 5, endpoint=False)),
        "actuator_scale": 0.1,
        "nu": 0.08156697852139966,
        "N": 256,
        "dt": 0.05,
    }
    T = 1000  # total timesteps per rollout

    # Make a list of policies
    random_policy = RandomPolicy(num_actions=len(ks_args["actuator_locs"]))
    zero_policy = StaticPolicy(torch.zeros(len(ks_args["actuator_locs"])))

    policies = {
        'zero': (zero_policy, 10),
        'random': (random_policy, 10)
    }  # contains pairs of policies and number of episodes for that policy

    outputs_u = []
    outputs_action = []
    outputs_forcing = []
    for (policy, num_episodes) in policies.values():
        for i in range(num_episodes):
            # Define the initial condition
            u0 = 1e-2 * np.random.normal(size=ks_args["N"])  # noisy intial data
            u0 = u0 - u0.mean()
            u0 = torch.tensor(u0)

            # Compute a rollout
            uu, actions, forcings = rollout(u0, policy=policy, num_steps=T, **ks_args)
            outputs_u.append(uu)
            outputs_action.append(actions)
            outputs_forcing.append(forcings)

    # Concatenate and save outputs
    outputs = {'u': outputs_u, 'action': outputs_action, 'forcing': outputs_forcing}
    for key, out in outputs.items():
        out = np.concatenate(out, axis=0)
        with open(f'datasets/test_{key}.dat', 'wb') as file:
            np.save(file, out)

    # Print information about the run
    total_rollouts = sum([n for (_, n) in policies.values()])
    total_timesteps = T * total_rollouts
    print("Completed the following rollouts:")
    for name, (_, num) in policies.items():
        print(f"\t Policy = {name} with {num} episodes a {T} timesteps each.")
    print(f"Summary: {total_rollouts} rollouts with {T} timesteps each, so {total_timesteps} time steps in total.")

