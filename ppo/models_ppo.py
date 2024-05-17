# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn
import numpy as np
import torch.optim
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.data import CompositeSpec
from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from autoencoder.visualize_cae_model import load_cae_model
from pathlib import Path
from utils.wrappers import CAEWrapper


# ====================================================================
# Model utils
# --------------------------------------------------------------------
def make_ppo_models(cfg, observation_spec, action_spec, path_to_model=None):
    # Define input shape
    observation_size = observation_spec["observation"].shape[-1]
    mlp_input_size = observation_size

    if path_to_model:
        # Load trained CAE model
        modelpath = Path(path_to_model)
        cae = load_cae_model(modelpath)
        encoder = cae.encoder
        cae = CAEWrapper(model=cae)
        encoder = CAEWrapper(model=encoder)
        # Freeze parameters
        for param in encoder.parameters():
            param.requires_grad = False
        for param in cae.parameters():
            param.requires_grad = False
        # Set correct input size for MLP
        mlp_input_size = encoder.cae.enc_linear_dense.out_features
        print("Using CAE")

    # Define policy output distribution class
    num_outputs = action_spec.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "min": action_spec.space.low,
        "max": action_spec.space.high,
        "tanh_loc": False,
    }

    # Define policy architecture
    policy_mlp = MLP(
        in_features=mlp_input_size,
        activation_class=torch.nn.Tanh,
        out_features=num_outputs,  # predict only loc
        num_cells=[64, 64],
    )

    # Initialize policy weights
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    policy_list = []
    if path_to_model:
        policy_list.append(encoder)
    policy_list.append(policy_mlp)
    policy_list.append(
        AddStateIndependentNormalScale(action_spec.shape[-1], scale_lb=1e-8)
    )

    # Add state-independent normal scale
    policy_mlp = torch.nn.Sequential(*policy_list)

    policy_module = TensorDictModule(
        module=policy_mlp,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )

    if path_to_model:
        cae_module = TensorDictModule(
            module=cae,
            in_keys=["observation"],
            out_keys=["cae_output"]
        )
        policy_module = TensorDictSequential(policy_module, cae_module)

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["loc", "scale"],
        spec=CompositeSpec(action=action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define value architecture
    value_mlp = MLP(
        in_features=mlp_input_size,
        activation_class=torch.nn.Tanh,
        out_features=1,
        num_cells=[64, 64],
    )

    # Initialize value weights
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    value_list = []
    if path_to_model:
        value_list.append(encoder)
    value_list.append(value_mlp)

    value_mlp = torch.nn.Sequential(*value_list)

    # Define value module
    value_module = ValueOperator(
        value_mlp,
        in_keys=["observation"],
    )

    return policy_module, value_module

