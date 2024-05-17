import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import warnings
from utils.config_tools import load_config
from utils.preprocessing import load_U_from_dat, train_valid_test_split
from convolutional_autoencoder import CAE

# ignore a matlab warning - to be removed when we update the solution import
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folderpath = Path("../models/cae")


def load_ks_data(modelpath):
    
    ks_data = load_config(modelpath / "ks.json")
    U = load_U_from_dat(
        "/home/eo821/Documents/latentRL/data/data_SAC_NU0.05/u_SAC_NU0.05_A20_NUMENVS5_BURNIN5000.dat"
    )
    print(np.max(U))
    U_train, U_valid, U_test = train_valid_test_split(U, ks_data)
    return U_train, U_valid, U_test


def load_cae_model(modelpath):
    ks_data = load_config(modelpath / "ks.json")
    cae_config = load_config(modelpath / "wandb_config.json")
    input_shape = (cae_config['batch_size'], 1, ks_data["N_x"]) #hardcoded output shape in the autoencoder...
    model = CAE(cae_config["latent_size"], weight_init_name=cae_config["weight_init_name"]).to(device)
    model.load_state_dict(torch.load(modelpath / "best_model.pth", map_location=device))
    return model.to(device)


if __name__ == "__main__":
    pathdir = Path(
        "/home/eo821/Documents/latentRL/data/data_SAC_NU0.05/symlog/"
    )
    for modelname in os.listdir(pathdir):
        print(modelname)
        modelpath = pathdir / modelname
        U_train, U_valid, U_test = load_ks_data(modelpath)
        cae_model = load_cae_model(modelpath)

    test_snapshot = torch.from_numpy(U_test).float()
    encoded, test_snapshot_reconstructed = cae_model.symp_forward(
        test_snapshot.to(device)
    )  ## this is how we get the reconstruction
    error = test_snapshot - test_snapshot_reconstructed.numpy(force=True)
    print(
        f"Relative L2 error {np.linalg.norm(error)/ np.linalg.norm(test_snapshot)} "
    )
    print(
        f"MSE {torch.nn.MSELoss()(test_snapshot_reconstructed[:, 0, :], test_snapshot[:, 0, :])}"
    )

    fig, axs = plt.subplots(1, 3)
    N_plot = 800
    vmax = torch.max(
        torch.max(
            torch.abs(test_snapshot_reconstructed[:N_plot]),
            torch.max(torch.abs(test_snapshot[:N_plot])),
        )
    )
    vmin = -vmax

    colomrmap = "RdYlBu_r"
    axs[0].imshow(
        test_snapshot[:N_plot, 0, :], vmin=vmin, vmax=vmax, cmap=colomrmap
    )
    axs[1].imshow(
        test_snapshot_reconstructed[:N_plot, 0, :].numpy(force=True),
        vmin=vmin,
        vmax=vmax,
        cmap=colomrmap,
    )
    im2 = axs[2].imshow(error[:N_plot, 0, :], vmin=vmin, vmax=vmax, cmap=colomrmap)

    axs[0].set_title("Test data")
    axs[1].set_title("CAE")
    axs[2].set_title("Difference")
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax, orientation="vertical")
    fig.show()
    fig.savefig(modelpath / "test.png")
