import torchinfo
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
import h5py
import einops
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import time
import warnings
from utils.config_tools import load_config
from utils.losses import LossTracker
from utils.earlystopping import EarlyStopper
from utils.preprocessing import load_U_from_dat, train_valid_test_split
from convolutional_autoencoder import CAE
# ignore a matlab warning - to be removed when we update the solution import
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
folderpath = Path("../models/cae")


def load_ks_data(modelpath):
    ks_data = load_config(modelpath/"ks.json")

    U = load_U_from_dat(folderpath / "u_SAC_NU0.05_A20_NUMENVS5_BURNIN5000.dat")
    print(np.max(U))
    # U_valid = load_U_from_dat("../data/datasets/validation_u.dat")
    # U_test =  load_U_from_dat("../data/datasets/test_u.dat")
    # U_train = U_train/ks_data['maxnorm']
    # U_valid = U_valid/ks_data['maxnorm']
    # U_test = U_test/ks_data['maxnorm']
    U_train, U_valid, U_test = train_valid_test_split(U/ks_data['maxnorm'], ks_data)
    return U_train, U_valid, U_test
 

def load_cae_model(modelpath):

    cae_config = load_config(modelpath/"wandb_config.json")
    model = CAE(cae_config["latent_size"],
                weight_init_name=cae_config["weight_init_name"]).to(device)
    model.load_state_dict(torch.load(modelpath/"best_model.pth", map_location=device))
    return model.to(device)


if __name__ == '__main__':

    modelpath=Path("../models/cae/dainty-sweep-2")
    U_train, U_valid, U_test = load_ks_data(modelpath)
    cae_model=load_cae_model(modelpath)

    test_snapshot = torch.from_numpy(U_test).float()
    encoded, test_snapshot_reconstructed = cae_model(test_snapshot.to(device)) ## this is how we get the reconstruction
    error = test_snapshot- test_snapshot_reconstructed.numpy(force=True) 
    print(f"Relative L2 error {np.linalg.norm(error)/ np.linalg.norm(test_snapshot)} ")


    fig, axs = plt.subplots(1, 3)
    vmin = -1
    vmax = 1
    N_plot = 3000
    axs[0].imshow(test_snapshot[:N_plot, 0, :], vmin=vmin, vmax=vmax)
    axs[1].imshow(test_snapshot_reconstructed[:N_plot, 0, :].numpy(force=True), vmin=vmin, vmax=vmax)
    im2 = axs[2].imshow(error[:N_plot, 0, :], vmin=vmin, vmax=vmax)

    axs[0].set_title("Test data")
    axs[1].set_title("CAE")
    axs[2].set_title("Difference")
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')
    fig.show()