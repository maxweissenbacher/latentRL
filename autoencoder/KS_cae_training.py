import torchinfo
import numpy as np
import torch
import torch.nn as nn
import wandb
from pathlib import Path
import time
import warnings
from utils.config_tools import save_config
from utils.losses import LossTracker
from utils.earlystopping import EarlyStopper
from utils.preprocessing import load_U_from_dat, train_valid_test_split
from convolutional_autoencoder import CAE
from utils.loss_function import symlog

# ignore a matlab warning - to be removed when we update the solution import
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
folderpath = Path("../data_SAC_NU0.05/")


def train():
    global wand

    config_defaults = {
        "latent_size": 128,
        "batch_size": 128 * 20,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "epochs": 0,
        "patience": 200,
        "weight_init_name": "kaiming_uniform",
        "nu": 0.0,
        "input_normalization": None,
    }
    # Initialize wandb with a sample project name
    wand = wandb.init(config=config_defaults)
    print(f"WANDB sweep name {wand.name}")
    modelpath = folderpath / wand.name
    modelpath.mkdir(parents=True, exist_ok=True)

    ks_data = {
        "L": 22,  # TO-DO: this is not correct
        "dt": 0.05,
        "N_x": 256,
        "batchsize": 128,
        "normtype": "max",
        "maxnorm": 18,
        "train_ratio": 0.5,
        "valid_ratio": 0.1,
        "N_data": 500000,
    }
    U = load_U_from_dat(folderpath / "u_SAC_NU0.05_A20_NUMENVS5_BURNIN5000.dat")
    print(np.max(U))
    U_train, U_valid, U_test = train_valid_test_split(U, ks_data)

    print(
        f"Points in Training/Validation/Test: {len(U_train), len(U_valid), len(U_test)}"
    )

    save_config(modelpath / "ks.json", ks_data)
    save_config(modelpath / "wandb_config.json", dict(wand.config))
    input_shape = (wandb.config.batch_size, 1, ks_data["N_x"]) #hardcoded output shape in the autoencoder...
    print(f"Latent size {wandb.config.latent_size}")
    batchsize = wandb.config.batch_size 
    model = CAE(wandb.config.latent_size, weight_init_name=wandb.config.weight_init_name
    ).to(device)
    # Set best model state to current model
    best_model_state_dict = model.state_dict()
    torchinfo.summary(model, input_size=(1, 1, ks_data["N_x"]))
    # train with noisy input

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    train_loader = torch.utils.data.DataLoader(
        torch.from_numpy(U_train).float(), batch_size=batchsize
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.from_numpy(U_valid).float(), batch_size=batchsize
    )

    # Define the number of epochs and the gamma parameter for the scheduler
    epochs = wandb.config.epochs
    gamma = 0.999

    # Create an instance of ExponentialLR and associate it with your optimizer
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    loss_tracker = LossTracker(len(train_loader), len(valid_loader))

    early_stopper = EarlyStopper(patience=wandb.config.patience, min_delta=1e-6)

    wandb.watch(model)
    for epoch in range(epochs):
        loss_tracker.set_start_time(time.time())
        loss_tracker.reset_current_loss()

        # Training loop
        model.train()
        for step, x_batch_train in enumerate(train_loader):
            x_batch_train = x_batch_train.to(device)
            optimizer.zero_grad()
            encoded, output = model(x_batch_train)
            loss_dec = criterion(output, symlog(x_batch_train))
            loss = loss_dec
            loss.backward()
            optimizer.step()
            loss_tracker.update_current_loss(
                "training",
                loss,
                loss_dec,
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
            )
        loss_tracker.print_current_loss(epoch, "training")

        loss_tracker.set_start_time(time.time())
        scheduler.step()
        # Validation loop
        model.eval()
        with torch.no_grad():
            for valid_step, x_batch_valid in enumerate(valid_loader):
                x_batch_valid = x_batch_valid.to(device)
                encoded, output = model(x_batch_valid)
                loss_dec = criterion(output, symlog(x_batch_valid))
                loss = loss_dec
                loss_tracker.update_current_loss(
                    "validation",
                    loss,
                    loss_dec,
                    torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device),
                )
        loss_tracker.print_current_loss(epoch, "validation")
        loss_tracker.calculate_and_store_average_losses()
        wandb.log(
            {
                f"{loss_type}": loss_tracker.losses_dict[loss_type][-1]
                for loss_type in loss_tracker.loss_types
            }
        )

        if loss_tracker.check_best_validation_loss():
            early_stopper.reset_counter()
            best_model_state_dict = model.state_dict()  # Save the best model state_dict
            torch.save(
                best_model_state_dict, modelpath / "best_model.pth"
            )  # Save the best model
        if early_stopper.track(loss_tracker.get_current_validation_loss()):
            break

    # Save losses to a single file
    loss_tracker.save_losses(path=modelpath)


if __name__ == "__main__":
    sweep_config = {
        "method": "grid",
        "metric": {"name": "validation_loss", "goal": "minimize"},
        "parameters": {
            "latent_size": {"values": [16, 24, 48]},  # 8, 12,
            "batch_size": {"values": [512]},
            "learning_rate": {"values": [0.001]},
            "epochs": {"values": [1000]},
            "patience": {"values": [100]},
            "weight_init_name": {"values": ["xavier_normal"]},
            "nu": {"values": [0.05]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="lantentRL-cae-ks")
    wandb.agent(sweep_id, function=train, count=3)
