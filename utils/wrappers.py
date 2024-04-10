import torch
import numpy as np


class CAEWrapper(torch.nn.Module):
    def __init__(self, model, normalisation=1.):
        super().__init__()
        self.cae = model
        self.normalisation_constant = normalisation

    def forward(self, x):
        batch_size = x.shape[:-1]
        observation_size = x.shape[-1]
        x = x.view(int(np.prod(batch_size)), 1, observation_size)
        x = x/self.normalisation_constant
        x = self.cae(x)
        if isinstance(x, tuple):
            x = x[1]
        x = x.view(*batch_size, x.shape[-1])
        return x

