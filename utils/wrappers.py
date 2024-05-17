import torch
import numpy as np
from autoencoder.utils.loss_function import symexp


class CAEWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.cae = model

    def forward(self, x):
        batch_size = x.shape[:-1]
        observation_size = x.shape[-1]
        # Resize to correct shape for CAE
        x = x.view(int(np.prod(batch_size)), 1, observation_size)
        # Map through model
        x = self.cae(x)
        # x = self.cae.symp_forward(x)
        if isinstance(x, tuple):
            # If we get both encoded and decoded, the model is the full CAE
            # In this case, we want to keep only decoded and map through symexp
            x = x[1]
            x = symexp(x)
        # Reshape to correct shape for RL training
        x = x.view(*batch_size, x.shape[-1])
        return x

