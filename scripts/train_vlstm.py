import time
from pathlib import Path

import torch
import yaml
from information_hydrology.modelzoo.vlstm import VLSTM, ErrorMode
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
experiment_name = "VLSTM_NLLKLD_064_PRO_60"
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
model = VLSTM(num_inputs, num_hidden, output_dropout, ErrorMode.PROPORTIONAL).to(device)

config.update(
    {
        "model": "vLSTM",
        "num_inputs": num_inputs,
        "num_hidden": num_hidden,
        "percent_dropout": output_dropout,
        "error": "proportional",
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

lrs = [1e-3] * 30 + [1e-4] * 10  # NLL KLD
betas = [1e-5] * 40 # NLL KLD


logger.info("Training loop")
time_training = time.time()
logger.info(f"{'':^5} | {'':^8} | {'':^8} | {'Train Loss':^32} | {'':^8} | {'Val. Loss':^32} | {'':^8}")
logger.info(f"{'Epoch':^5} | {'LR':^8} | {'Beta':^8} | {'Loss 1':^8} | {'Loss 2':^10} | {'Total':^8} | {'Time':^8} | {'Loss 1':^8} | {'Loss 2':^10} | {'Total':^8} | {'Time':^8}")

for epoch in trange(num_epochs, desc="Epochs", ncols=78, ascii=True, unit="epoch"):
    # Change LR at every epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=lrs[epoch])
    
    # Training
    time_epoch = time.time()
    epoch_loss_1, epoch_loss_2, epoch_total_loss = [], [], []

    model.train()
    for sample in tqdm(dl_train, desc="Training", ncols=79, ascii=True, unit="batch", position=1):
        # Fix inputs
        x_d, y, _, x_s = sample.values()
        x_s = x_s.unsqueeze(1).repeat(1, x_d.shape[1], 1)
        x = torch.cat([x_d, x_s], dim=-1).to(device)
        y = y[:, -1, :].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        hs, _, mu, log_var = model(x)
        
        # Model
        w = model.state_dict()["decoder.weight"]
        b = model.state_dict()["decoder.bias"]
        y_mu = hs @ w.T + b
        y_sigma = torch.sqrt(hs ** 2 @ (w ** 2).T)

        loss_1 = loss_nll((y_mu, y_sigma, torch.ones_like(y_mu)), y)
        loss_2 = loss_kld(mu, log_var)
        if loss_1.isnan() or loss_2.isnan():
            continue
        loss = loss_1 + betas[epoch] * loss_2
        loss.backward()
        optimizer.step()

        epoch_loss_1.append(loss_1.item())
        epoch_loss_2.append(loss_2.item())
        epoch_total_loss.append(loss.item())

        # Delete
        del x_d, y, x_s, x, hs, mu, log_var, w, b, y_mu, y_sigma, loss_1, loss_2, loss

    # Average loss epoch
    loss_train_1 = sum(epoch_loss_1) / len(epoch_loss_1)
    loss_train_2 = sum(epoch_loss_2) / len(epoch_loss_2)
    loss_train_total = sum(epoch_total_loss) / len(epoch_total_loss)

    # Save model
    path_save_model = path_save_folder / f"model_epoch_{(epoch + 1):02d}.pt"
    torch.save(model.state_dict(), path_save_model)

    # Save time
    time_train = seconds_to_time(time.time() - time_epoch)

    if (epoch + 1) % num_validate_every != 0:
        logger.info(f"{epoch + 1:^5} | {lrs[epoch]:^8.1e} | {betas[epoch]:^8.1e} | {loss_train_1:^8.4f} | {loss_train_2:^10.3e} | {loss_train_total:^8.4f} | {time_train:^8} | {'':^8} | {'':^10} | {'':^8} | {'':^8}")
        continue

    # Start validation
    time_epoch = time.time()
    epoch_loss_1, epoch_loss_2, epoch_total_loss = [], [], []

    model.eval()
    for sample in tqdm(dl_val, desc="Validation", ncols=79, ascii=True, unit="batch", position=1):
        # Fix inputs
        x_d, y, _, x_s = sample.values()
        x_s = x_s.unsqueeze(1).repeat(1, x_d.shape[1], 1)
        x = torch.cat([x_d, x_s], dim=-1).to(device)
        y = y[:, -1, :].to(device)

        # Forward pass
        hs, _, mu, log_var = model(x)
        w = model.state_dict()["decoder.weight"]
        b = model.state_dict()["decoder.bias"]
        y_mu = hs @ w.T + b
        y_sigma = torch.sqrt(hs ** 2 @ (w ** 2).T)
        loss_1 = loss_nll((y_mu, y_sigma, torch.ones_like(y_mu)), y)
        loss_2 = loss_kld(mu, log_var)
        if loss_1.isnan() or loss_2.isnan():
            continue
        loss = loss_1 + betas[epoch] * loss_2
        
        epoch_loss_1.append(loss_1.item())
        epoch_loss_2.append(loss_2.item())
        epoch_total_loss.append(loss.item())

        # Delete
        del x_d, y, x_s, x, hs, mu, log_var, w, b, y_mu, y_sigma, loss_1, loss_2, loss

    # Average loss epoch
    loss_val_1 = sum(epoch_loss_1) / len(epoch_loss_1)
    loss_val_2 = sum(epoch_loss_2) / len(epoch_loss_2)
    loss_val_total = sum(epoch_total_loss) / len(epoch_total_loss)

    # Print report
    time_val = seconds_to_time(time.time() - time_epoch)
    logger.info(f"{epoch + 1:^5} | {lrs[epoch]:^8.1e} | {betas[epoch]:^8.1e} | {loss_train_1:^8.4f} | {loss_train_2:^10.3e} | {loss_train_total:^8.4f} | {time_train:^8} | {loss_val_1:^8.4f} | {loss_val_2:^10.3e} | {loss_val_total:^8.4f} | {time_val:^8}")

# Print final report
time_training = time.time() - time_training
logger.info("Run completed successfully")
logger.info(f"Total run time: {seconds_to_time(time_training)}")
