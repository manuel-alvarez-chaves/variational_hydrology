import pickle
import time
from pathlib import Path

import torch
import yaml
from hy2dl.aux_functions.functions_training import nse_basin_averaged
from hy2dl.modelzoo.cudalstm import CudaLSTM
from information_hydrology.utils.logging import get_logger
from information_hydrology.utils.metrics import calc_nse
from information_hydrology.utils.miscellaneous import seconds_to_time, set_seed
from information_hydrology.utils.training import Period, get_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# # # # # # # # # # # # # # # PART 00 # # # # # # # # # # # # # # # # #

# General config
experiment_name = "LSTM-064_NSE_060"
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

config_model = {
        "name": "LSTM",
        "input_size_lstm": num_inputs,
        "hidden_size": num_hidden,
        "dropout_rate": output_dropout,
        "no_of_layers": 1,
        "predict_last_n": 1,
}

model = CudaLSTM(config_model).to(device)

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
    shuffle=False,
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
        y = sample["y_obs"]
        y = y[:, -1, :].to(device)

        # Forward pass
        if period == "train":
            optimizer.zero_grad()

        y_hat = model(sample)["y_hat"][:, 0]
        loss = nse_basin_averaged(y_hat, y, sample["basin_std"])
        nse = calc_nse(y.flatten().detach().numpy(), y_hat.flatten().detach().numpy())

        if period == "train":
            loss.backward()
            optimizer.step()

        epoch_loss.append(loss.item())
        epoch_nse.append(nse)

        del sample, y, y_hat, loss, nse
    
    # Average loss epoch
    loss = sum(epoch_loss) / len(epoch_loss)
    nse = sum(epoch_nse) / len(epoch_nse)

    if period == "train":
        path_save_model = path_save_folder / f"model_epoch_{(epoch + 1):02d}.pt"
        torch.save(model.state_dict(), path_save_model)

    time_epoch = seconds_to_time(time.time() - time_epoch)

    return loss, nse, time_epoch

# # # # # # # # # # # # # # # PART 04 # # # # # # # # # # # # # # # # #

num_epochs = 40
num_validate_every = 2

lrs = [1e-3] * 30 + [1e-4] * 10

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
