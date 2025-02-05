import pickle
import time
from pathlib import Path

import torch
import yaml
from hy2dl.datasetzoo.camelsus import CAMELS_US
from information_hydrology.modelzoo.lstmgmm import LSTMGMM
from information_hydrology.utils.logging import get_logger
from information_hydrology.utils.loss_fn import loss_nll
from information_hydrology.utils.miscellaneous import seconds_to_time, set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# # # # # # # # # # # # # # # PART 00 # # # # # # # # # # # # # ## # #

# General config
experiment_name = "LSTMGMM_NLL_064_10_60"
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
num_gaussians = 10
output_dropout = 0.4
model = LSTMGMM(num_inputs, num_hidden, num_gaussians, output_dropout).to(device)
config_model = {
        "model": "LSTM-GMM",
        "num_inputs": num_inputs,
        "num_hidden": num_hidden,
        "percent_dropout": output_dropout,
        "num_gaussians": num_gaussians,
}

logger.info(f"Model: {config_model['model']}")

# Dump config
config = {
    "experiment": experiment_name,
    "seed": seed,
    "data": config_data,
    "model": config_model,
}
with Path.open(path_save_folder / "config.yml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

# # # # # # # # # # # # # # # PART 02 # # # # # # # # # # # # # ## # #

# Training
ds_train = CAMELS_US(
    dynamic_input=config_data["dynamic_inputs"],
    forcing=config_data["forcings"],
    target=config_data["target_variables"],
    sequence_length=config_data["sequence_length"],
    time_period=config_data["train_period"],
    path_data=config_data["data_dir"],
    path_entities=config_data["train_basin_file"],
    check_NaN=True,
    static_input=config_data["static_attributes"],
)

# Standardize data and save scaler
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
ds_val = CAMELS_US(
    dynamic_input=config_data["dynamic_inputs"],
    forcing=config_data["forcings"],
    target=config_data["target_variables"],
    sequence_length=config_data["sequence_length"],
    time_period=config_data["train_period"],
    path_data=config_data["data_dir"],
    path_entities=config_data["train_basin_file"],
    check_NaN=True,
    static_input=config_data["static_attributes"],
)

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

num_epochs = 2
num_validate_every = 2

lrs = [1e-3] * 10 + [1e-4] * 20

logger.info("Training loop")
time_training = time.time()
logger.info(f"{'Epoch':<5} | {'Train Loss':<10} | {'Time':<8} | {'Val. Loss':<10} | {'Time':<8}")

for epoch in trange(num_epochs, desc="Epochs", ncols=78, ascii=True, unit="epoch"):
    # Change LR at every epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=lrs[epoch])

    # Training
    time_epoch = time.time()
    epoch_loss = []

    model.train()
    for sample in tqdm(dl_train, desc="Training", ncols=79, ascii=True, unit="batch", position=1):
        # Fix inputs
        x_d, x_s, y, _, _, _ = sample.values()
        x_s = x_s.unsqueeze(1).repeat(1, x_d.shape[1], 1)
        x = torch.cat([x_d, x_s], dim=-1).to(device)
        y = y[:, -1, :].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_nll(y_hat, y)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

        # Delete
        del x_d, y, x_s, x, y_hat, loss

    # Average loss epoch
    epoch_average_loss = sum(epoch_loss) / len(epoch_loss)

    # Save model
    path_save_model = path_save_folder / f"model_epoch_{(epoch + 1):02d}.pt"
    torch.save(model.state_dict(), path_save_model)

    # Save time
    time_epoch = time.time() - time_epoch

    if (epoch + 1) % num_validate_every != 0:
        logger.info(f"{epoch + 1:<5} | {epoch_average_loss:<10.5f} | {seconds_to_time(time_epoch)} | {'':<10} | {'':<10}")
        continue

    # Save from training
    train_loss = epoch_average_loss
    train_time = time_epoch

    # Start validation
    time_epoch = time.time()
    epoch_loss = []

    model.eval()
    for sample in tqdm(dl_val, desc="Validation", ncols=79, ascii=True, unit="batch", position=1):
        # Fix inputs
        x_d, x_s, y, _, _, _ = sample.values()
        x_s = x_s.unsqueeze(1).repeat(1, x_d.shape[1], 1)
        x = torch.cat([x_d, x_s], dim=-1).to(device)
        y = y[:, -1, :].to(device)

        # Forward pass
        y_hat = model(x)
        loss = loss_nll(y_hat, y)
        if loss.isnan():
            continue
        epoch_loss.append(loss.item())

        # Delete
        del x_d, y, x_s, x, y_hat, loss

    # Average loss epoch
    epoch_average_loss = sum(epoch_loss) / len(epoch_loss)

    # Print report
    time_epoch = time.time() - time_epoch
    logger.info(f"{epoch + 1:<5} | {train_loss:<10.5f} | {seconds_to_time(train_time)} | {epoch_average_loss:<10.5f} | {seconds_to_time(time_epoch)}")

# Print final report
time_training = time.time() - time_training
logger.info("Run completed successfully")
logger.info(f"Total run time: {seconds_to_time(time_training)}")
