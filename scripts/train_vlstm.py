import pickle
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from information_hydrology.modelzoo.vlstm import VLSTM, ErrorMode, SamplingMode
from information_hydrology.utils.logging import get_logger
from information_hydrology.utils.loss_fn import (
        loss_kld,
        loss_nll,
        loss_nll_kde,
        loss_nll_knn,
        loss_nll_norm,
)
from information_hydrology.utils.miscellaneous import seconds_to_time, set_seed
from information_hydrology.utils.training import Period, get_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# # # # # # # # # # # # # # # PART 00 # # # # # # # # # # # # # # # # #

# General config
experiment_name = "VLSTM_NLL-NORM-1-KLD_064_LEARN_PRO_60"
seed = set_seed(42)
path_save_folder = Path("experiments") / (experiment_name + time.strftime(r"_%Y-%m-%d_%H-%M-%S"))

# Read config
path_config = Path("scripts/config_data.yml")
config_data = yaml.safe_load(path_config.read_text())

# Logger
path_logger = path_save_folder / "run.log"
path_logger.parent.mkdir(parents=True, exist_ok=True)
logger = get_logger(path_logger)

# Set CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Start log
logger.info(f"Experiment: {experiment_name}")
logger.info(f"Seed: {seed}")
logger.info(f"Using device: {device}")

# # # # # # # # # # # # # # # PART 01 # # # # # # # # # # # # # # # # #

# Model
num_inputs = len(config_data["dynamic_inputs"]) + len(config_data["static_attributes"])
num_hidden = 64
output_dropout = 0.4
model = VLSTM(num_inputs, num_hidden, output_dropout, ErrorMode.PROPORTIONAL).to(device)
config_model = {
        "name": "vLSTM",
        "num_inputs": num_inputs,
        "num_hidden": num_hidden,
        "percent_dropout": output_dropout,
        "error": "proportional",
    }

logger.info(f"Model: {config_model['name']}")

# Dump config
config = {
    "experiment": experiment_name,
    "seed": seed,
    "data": config_data,
    "model": config_model,
}
with Path.open(path_save_folder / "config.yml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

# # # # # # # # # # # # # # # PART 02 # # # # # # # # # # # # # # # # #

# Training
ds_train = get_dataset(config_data, Period.TRAINING)
ds_train.calculate_basin_std()
ds_train.calculate_global_statistics(path_save_scaler=str(path_save_folder.resolve()))
ds_train.standardize_data(standardize_output=False)

dl_train = DataLoader(
    ds_train,
    batch_size=config_data["batch_size"],
    shuffle=True,
    drop_last=True,
    collate_fn=ds_train.collate_fn,
)

# Validation
ds_val = get_dataset(config_data, Period.VALIDATION)

# Standardize data using the saved training scaler
ds_val.calculate_basin_std()
with Path.open(path_save_folder / "scaler.pickle", "rb") as f:
    ds_val.scaler = pickle.load(f)
ds_val.standardize_data(standardize_output=False)

dl_val = DataLoader(
    ds_val,
    batch_size=config_data["batch_size"],
    shuffle=True,
    drop_last=True,
    collate_fn=ds_val.collate_fn,
)

# # # # # # # # # # # # # # # PART 03 # # # # # # # # # # # # # # # # #

# Custom training/validation loop

def training_loop(epoch: int, period: str):
    if period == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=lrs[epoch])
        loader = dl_train
        model.train()
        misc = {"desc": "Training", "track_grad": True}
    else:
         loader = dl_val
         model.eval()
         misc = {"desc": "Validation", "track_grad": False}
    
    time_epoch = time.time()
    epoch_loss_1, epoch_loss_2, epoch_total_loss = [], [], []
    for sample in tqdm(loader, desc=misc["desc"], ncols=79, ascii=True, unit="batch", position=1):
        # Fix inputs
        x_d, x_s, y, _, _, _ = sample.values()
        x_s = x_s.unsqueeze(1).repeat(1, x_d.shape[1], 1)
        x = torch.cat([x_d, x_s], dim=-1).to(device)
        y = y[:, -1, :].to(device)

        # Forward pass
        if period == "train":
            optimizer.zero_grad()

        _, _, logvar = model(x)
        samples = model.sample(x, 1000, SamplingMode.LEARNED, track_grad=misc["track_grad"])

        loss_1 = loss_nll_norm(samples, y, p_norm=1)
        loss_2 = loss_kld(logvar)
        loss = loss_1 + betas[epoch] * loss_2

        if period == "train":
            loss.backward()
            optimizer.step()

        epoch_loss_1.append(loss_1.item())
        epoch_loss_2.append(loss_2.item())
        epoch_total_loss.append(loss.item())

        del x_d, y, x_s, x, logvar, loss_1, loss_2, loss
    
    # Average loss epoch
    loss_1 = sum(epoch_loss_1) / len(epoch_loss_1)
    loss_2 = sum(epoch_loss_2) / len(epoch_loss_2)
    loss_total = sum(epoch_total_loss) / len(epoch_total_loss)

    if period == "train":
        path_save_model = path_save_folder / f"model_epoch_{(epoch + 1):02d}.pt"
        torch.save(model.state_dict(), path_save_model)

    time_epoch = seconds_to_time(time.time() - time_epoch)

    return loss_1, loss_2, loss_total, time_epoch

# # # # # # # # # # # # # # # PART 04 # # # # # # # # # # # # # # # # #

num_epochs = 40
num_validate_every = 2

lrs = [1e-3] * 30 + [1e-4] * 10
betas = [1e-3] * 30 + [1e-2] * 10

logger.info("Training loop")
time_training = time.time()
logger.info(f"{'':^5} | {'':^8} | {'':^8} | {'Train Loss':^32} | {'':^8} | {'Val. Loss':^32} | {'':^8}")
logger.info(f"{'Epoch':^5} | {'LR':^8} | {'Beta':^8} | {'Loss 1':^8} | {'Loss 2':^10} | {'Total':^8} | {'Time':^8} | {'Loss 1':^8} | {'Loss 2':^10} | {'Total':^8} | {'Time':^8}")

time_training = time.time()
for epoch in trange(num_epochs, desc="Epochs", ncols=78, ascii=True, unit="epoch"):
    # Train
    loss_train_1, loss_train_2, loss_train_total, time_train = training_loop(epoch, "train")
    if (epoch + 1) % num_validate_every != 0:
        logger.info(f"{epoch + 1:^5} | {lrs[epoch]:^8.1e} | {betas[epoch]:^8.1e} | {loss_train_1:^8.4f} | {loss_train_2:^10.3e} | {loss_train_total:^8.4f} | {time_train:^8} | {'':^8} | {'':^10} | {'':^8} | {'':^8}")
        continue
    # Validate
    loss_val_1, loss_val_2, loss_val_total, time_val = training_loop(epoch, "validate")
    logger.info(f"{epoch + 1:^5} | {lrs[epoch]:^8.1e} | {betas[epoch]:^8.1e} | {loss_train_1:^8.4f} | {loss_train_2:^10.3e} | {loss_train_total:^8.4f} | {time_train:^8} | {loss_val_1:^8.4f} | {loss_val_2:^10.3e} | {loss_val_total:^8.4f} | {time_val:^8}")

time_training = time.time() - time_training
logger.info("Run completed successfully")
logger.info(f"Total run time: {seconds_to_time(time_training)}")
