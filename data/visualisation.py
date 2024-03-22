import torch
import torch.nn as nn
import numpy as np
from solver.ks_solver import KS
from utils.plotting import contourplot_KS
import matplotlib.pyplot as plt
from data.data_generation import rollout, StaticPolicy, RandomPolicy


if __name__ == '__main__':
    # Specify the arguments for the KS solver
    ks_args = {
        "actuator_locs": torch.tensor([0., torch.pi]),
        "actuator_scale": 0.1,
        "nu": 0.005,  # 0.08156697852139966,
        "N": 256,
        "dt": 0.05,
    }

    # Make a list of policies
    # A policy is a function which maps a full state u to an action a
    random_policy = RandomPolicy(num_actions=2)
    static_policy = StaticPolicy(torch.tensor([2.0, 2.0]))
    zero_policy = StaticPolicy(torch.zeros(2))

    # Define the initial condition
    u0 = 1e-2 * np.random.normal(size=ks_args["N"])  # noisy intial data
    u0 = u0 - u0.mean()
    u0 = torch.tensor(u0)

    # Compute a rollout
    out = rollout(u0, zero_policy, 1000, **ks_args)

    contourplot_KS(out, dt=ks_args["dt"])

    # Plot the std as a function of timesteps to see onset of attractor
    # This is a bit hacky but might be a good way to see it
    # As first approximation, when std > 1.0 we are probably on the attractor
    plt.plot(out.std(axis=1))
    plt.show()

