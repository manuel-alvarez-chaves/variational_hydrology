import time
from pathlib import Path

import torch
import yaml
from information_hydrology.modelzoo.noisylstm import NoisyLSTM
from information_hydrology.utils.logging import get_logger
from information_hydrology.utils.loss_fn import loss_kld, loss_nll
from information_hydrology.utils.miscellaneous import (
    dump_config,
    seconds_to_time,
    set_seed,
)
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.utils.config import Config
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# # # # # # # # # # # # # # # PART 00 # # # # # # # # # # # # # ## # #

# General config
experiment_name = "NoisyLSTM_NLL_064_60"
seed = set_seed(42)
path_save_folder = Path("experiments") / (experiment_name + time.strftime(r"_%Y-%m-%d_%H-%M-%S"))

# NeuralHydrology onfig file for data
path_config = Path("scripts/config_data.yml")
config = yaml.safe_load(Path.open(path_config, "r"))
config.update({"train_dir": path_save_folder})

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
num_inputs = len(config["dynamic_inputs"]) + len(config["static_attributes"])
num_hidden = 64
output_dropout = 0.4
model = NoisyLSTM(num_inputs, num_hidden, output_dropout).to(device)

config.update(
    {
        "model": "NoisyLSTM",
        "num_inputs": num_inputs,
        "num_hidden": num_hidden,
        "percent_dropout": output_dropout,
    }
)

# Dump config as YAML
logger.info(f"Model: {config['model']}")
dump_config(config, path_save_folder / "config.yml")

# # # # # # # # # # # # # # # PART 02 # # # # # # # # # # # # # ## # #

# Dataset and Loader
config = Config(config, dev_mode=True)

# Training
ds_train = get_dataset(cfg=config, is_train=True, period="train")
dl_train = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True, collate_fn=ds_train.collate_fn)
logger.info(f"Batches in training: {len(dl_train)}")

# Validation
ds_val = get_dataset(cfg=config, is_train=False, period="validation")
dl_val = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False, collate_fn=ds_val.collate_fn)
logger.info(f"Batches in validation: {len(dl_val)}")

# Items
sample = next(iter(dl_train))

logger.info("Input data keys:")
for k, v in sample.items():
    logger.info(f"{k}: {v.shape}")

x_d, y, date, x_s = sample.values()

# # # # # # # # # # # # # # # PART 03 # # # # # # # # # # # # # # # # #

num_epochs = 40
num_validate_every = 2

lrs = [1e-3] * 40

logger.info("Training loop")
time_training = time.time()
logger.info(f"{'Epoch':<5} | {'Train Loss':<10} | {'Time':<8} | {'Val. Loss':<10} | {'Time':<8}")

optimizer = torch.optim.Adam(model.parameters())
for epoch in trange(num_epochs, desc="Epochs", ncols=78, ascii=True, unit="epoch"):
    # Training
    time_epoch = time.time()
    epoch_loss = []

    model.train()
    for sample in tqdm(dl_train, desc="Training", ncols=79, ascii=True, unit="batch", position=1):
        # Fix inputs
        x_d, y, _, x_s = sample.values()
        x_s = x_s.unsqueeze(1).repeat(1, x_d.shape[1], 1)
        x = torch.cat([x_d, x_s], dim=-1).to(device)
        y = y[:, -1, :].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        y_hat = model(x, 1_000)
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
        x_d, y, _, x_s = sample.values()
        x_s = x_s.unsqueeze(1).repeat(1, x_d.shape[1], 1)
        x = torch.cat([x_d, x_s], dim=-1).to(device)
        y = y[:, -1, :].to(device)

        # Forward pass
        y_hat = model(x, 1000)
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

