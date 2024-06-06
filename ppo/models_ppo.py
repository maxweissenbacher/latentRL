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
from autoencoder.convolutional_autoencoder import CAE


# ====================================================================
# Model utils
# --------------------------------------------------------------------
def make_ppo_models(cfg, observation_spec, action_spec, use_cae=False):
    # Define input shape
    observation_size = observation_spec["observation"].shape[-1]
    mlp_input_size = observation_size

    if use_cae:
        # Load trained CAE model
        # modelpath = Path(use_cae)
        # cae = load_cae_model(modelpath)  # we're not loading a trained model now
        cae = CAE(latent_size=cfg.cae.latent_size, weight_init_name=cfg.cae.weight_init_name)
        encoder = cae.encoder
        decoder = cae.decoder
        encoder = CAEWrapper(model=encoder, mode='encode')
        decoder = CAEWrapper(model=decoder, mode='decode')
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

    if use_cae:
        encoder_module = TensorDictModule(
            module=encoder,
            in_keys=["observation"],
            out_keys=["latent_state"],
        )
        decoder_module = TensorDictModule(
            module=decoder,
            in_keys=["latent_state"],
            out_keys=["cae_output"],
        )

    policy_list = []
    policy_list.append(policy_mlp)
    policy_list.append(
        AddStateIndependentNormalScale(action_spec.shape[-1], scale_lb=1e-8)
    )
    policy_mlp = torch.nn.Sequential(*policy_list)

    policy_module = TensorDictModule(
        module=policy_mlp,
        in_keys=["latent_state"] if use_cae else ["observation"],
        out_keys=["loc", "scale"],
    )

    if use_cae:
        policy_module = TensorDictSequential(encoder_module, decoder_module, policy_module)

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
    if use_cae:
        value_list.append(encoder)
    value_list.append(value_mlp)

    value_mlp = torch.nn.Sequential(*value_list)

    # Define value module
    value_module = ValueOperator(
        value_mlp,
        in_keys=["observation"],
    )

    return policy_module, value_module

