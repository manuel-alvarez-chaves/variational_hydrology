import pickle
import time
from pathlib import Path

import torch
import yaml
from information_hydrology.modelzoo.mdn import LSTMMDN
from information_hydrology.utils.distributions import Distribution
from information_hydrology.utils.logging import get_logger
from information_hydrology.utils.loss_fn import loss_nll
from information_hydrology.utils.metrics import calc_nse
from information_hydrology.utils.miscellaneous import seconds_to_time, set_seed
from information_hydrology.utils.training import Period, get_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# # # # # # # # # # # # # # # PART 00 # # # # # # # # # # # # # # # # #

# General config
experiment_name = "LSTMCMAL-010-03_NLL-LAPLACE_S01_005"
seed = set_seed(1)
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
num_hidden = 10
distribution = Distribution.LAPLACE
num_components = 3
output_dropout = 0.4
model = LSTMMDN(num_inputs, num_hidden, distribution, num_components, output_dropout).to(device)
config_model = {
        "name": "LSTMCMAL",
        "num_inputs": num_inputs,
        "num_hidden": num_hidden,
        "head": distribution,
        "num_components": num_components,
        "percent_dropout": output_dropout,
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
    epoch_loss, epoch_nse = [], []
    for sample in tqdm(loader, desc=misc["desc"], ncols=79, ascii=True, unit="batch", position=1):
        # Fix inputs
        x_d, x_s, y, _, _, _ = sample.values()
        x_s = x_s.unsqueeze(1).repeat(1, x_d.shape[1], 1)
        x = torch.cat([x_d, x_s], dim=-1).to(device)
        y = y[:, -1, :].to(device)

        # Forward pass
        if period == "train":
            optimizer.zero_grad()

        y_hat = model(x)
        loss = loss_nll(y_hat, y, Distribution.LAPLACE)

        mean = model.calc_mean(x)
        nse = calc_nse(y.flatten().detach().cpu().numpy(), mean)

        if period == "train":
            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], max_norm=1, error_if_nonfinite=True)
            except RuntimeError as e:
                tqdm.write(f"Error: {e}")
                continue
            optimizer.step()

        epoch_loss.append(loss.item())
        epoch_nse.append(nse)

        del x_d, y, x_s, x, loss
    
    # Average loss epoch
    loss = sum(epoch_loss) / len(epoch_loss)
    nse = sum(epoch_nse) / len(epoch_nse)

    if period == "train":
        path_save_model = path_save_folder / f"model_epoch_{(epoch + 1):02d}.pt"
        torch.save(model.state_dict(), path_save_model)

    time_epoch = seconds_to_time(time.time() - time_epoch)

    return loss, nse, time_epoch

# # # # # # # # # # # # # # # PART 04 # # # # # # # # # # # # # # # # #

num_epochs = 20
num_validate_every = 1

lrs = [1e-2] * 20

logger.info("Training loop")
time_training = time.time()
logger.info(f"{'':^5} | {'':^8} | {'Trainining':^30} | {'Validation':^30} |")
logger.info(f"{'Epoch':^5} | {'LR':^8} | {'Loss':^8} | {'NSE':^8} | {'Time':^8} | {'Loss':^8} | {'NSE':^8} | {'Time':^8} |")

time_training = time.time()
for epoch in trange(num_epochs, desc="Epochs", ncols=78, ascii=True, unit="epoch"):
    # Train
    loss_train, nse_train, time_train = training_loop(epoch, "train")
    if (epoch + 1) % num_validate_every != 0:
        logger.info(f"{epoch + 1:^5} | {lrs[epoch]:^8.1e} | {loss_train:^8.4f} | {nse_train:^8.4f} | {time_train:^8} | {'':^8} | {'':^8} | {'':^8} |")
        continue
    # Validate
    loss_val, nse_val, time_val = training_loop(epoch, "validate")
    logger.info(f"{epoch + 1:^5} | {lrs[epoch]:^8.1e} | {loss_train:^8.4f} | {nse_train:^8.4f} | {time_train:^8} | {loss_val:^8.4f} | {nse_val:^8.4f} | {time_val:^8} |")

time_training = time.time() - time_training
logger.info("Run completed successfully")
logger.info(f"Total run time: {seconds_to_time(time_training)}")
