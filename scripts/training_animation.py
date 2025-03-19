import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from hy2dl.datasetzoo.camelsus import CAMELS_US
from information_hydrology.modelzoo.lstmgmm import LSTMGMM
from information_hydrology.modelzoo.vlstm import VLSTM, ErrorMode, SamplingMode
from information_hydrology.utils.metrics import calc_kde_loglik, calc_nse
from matplotlib.animation import FuncAnimation
from torch.utils.data import DataLoader
from tqdm import tqdm

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

path_training_run = Path(sys.argv[1])

# Load config
with Path.open(path_training_run / "config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)

# Experiment name
experiment_name = config["experiment"]

# Load model
model_name = config["model"]["name"]
num_inputs = config["model"]["num_inputs"]
num_hidden = config["model"]["num_hidden"]
percent_dropout = config["model"]["percent_dropout"]
num_samples = 1_000
match model_name:
    case "vLSTM":
        match config["model"]["error"]:
            case "proportional":
                error_mode = ErrorMode.PROPORTIONAL
            case "exponential":
                error_mode = ErrorMode.EXPONENTIAL
            case "dense":
                error_mode = ErrorMode.DENSE
        model = VLSTM(num_inputs, num_hidden, percent_dropout, error_mode)
    case "LSTM_GMM":
        num_gaussians = config["model"]["num_gaussians"]
        model = LSTMGMM(num_inputs, num_hidden, num_gaussians, percent_dropout)
    case _:    
        raise ValueError(f"Model {model_name} not recognized")

# Load model weights and biases
path_models = list(path_training_run.glob("*.pt"))
path_models.sort()

# Scaler
with Path.open(path_training_run / "scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

# One dataset per basin
ds = CAMELS_US(
    dynamic_input=config["data"]["dynamic_inputs"],
    forcing=config["data"]["forcings"],
    target=config["data"]["target_variables"],
    sequence_length=config["data"]["sequence_length"],
    time_period=config["data"]["test_period"],
    path_data=config["data"]["data_dir"],
    entity="13011900",
    check_NaN=False,
    static_input=config["data"]["static_attributes"],
)
ds.scaler = scaler
ds.standardize_data(standardize_output=False)

loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=ds.collate_fn)

# Fix inputs
for idx, sample in enumerate(loader):
    if idx == 2:
        break

x_d, x_s, y, basin, dates = sample.values()
dates = dates.flatten()
x_s = x_s.unsqueeze(1).repeat(1, x_d.shape[1], 1)
x = torch.cat([x_d, x_s], dim=-1)
y = y[:, -1, :].flatten().detach().numpy()

tqdm.write(f"Basin: {basin[0]}")
tqdm.write(f"Dates: {dates[0]} - {dates[-1]}")
# Iterate over models
medians, lowers, uppers, nses, logliks = [], [], [], [], []
for path_model in tqdm(path_models, ascii=True):
    model.load_state_dict(torch.load(path_model, map_location="cpu", weights_only=False))
    model.eval()
    match model_name:
            case "vLSTM":
                pred = model.sample(x, num_samples, mode=SamplingMode.LEARNED, track_grad=False)
                y_hat_sample = pred.detach().cpu().clone().numpy()[:, :, 0] # [batch_size, num_samples, num_targets]
            case "LSTM_GMM":
                pred = model.sample(x, num_samples)
                y_hat_sample = pred.detach().cpu().clone().numpy() # [batch_size, num_samples]

    # Compute median and quantiles
    medians.append(np.median(y_hat_sample, axis=1))
    lowers.append(np.quantile(y_hat_sample, 0.05, axis=1))
    uppers.append(np.quantile(y_hat_sample, 0.95, axis=1))
    nses.append(calc_nse(y, np.mean(y_hat_sample, axis=1)))
    logliks.append(calc_kde_loglik(y, y_hat_sample))
    

epochs = [i + 1 for i in range(len(medians))]
for _ in range(20):
    epochs.append(epochs[-1])
    medians.append(medians[-1])
    lowers.append(lowers[-1])
    uppers.append(uppers[-1])
    nses.append(nses[-1])
    logliks.append(logliks[-1])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Make animation
fig, ax = plt.subplots(figsize=(10, 5))

observed_line, = ax.plot(dates, y, label="Observed")
median_line, = ax.plot([], [], label=f"{experiment_name}", color="tab:green")
fill_between = ax.fill_between(dates, lowers[0], uppers[0], alpha=0.3, label="Prediction CI [5%, 95%]", color="tab:green")
title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center")

ax.set_xlim(min(dates), max(dates))
ax.set_ylim(-0.35, 14.35)
ax.set_xlabel("Date")
ax.set_ylabel(r"Streamflow [mm/day]")
ax.grid(ls="--")
ax.legend()

def update(frame):
    median_line.set_data(dates, medians[frame])
    if len(ax.collections) > 0:
        for collection in ax.collections:
            collection.remove()

    ax.fill_between(dates, lowers[frame], uppers[frame], alpha=0.3, label="Prediction CI [5%, 95%]", color="tab:green")

    title.set_text(f"Epoch: {epochs[frame]}, NSE: {nses[frame]:.3f}, Loglik: {logliks[frame]:.4f}")
    return median_line, ax.collections[0], title

ani = FuncAnimation(fig, update, frames=len(medians), interval=100, blit=True)
ani.save(path_training_run / "training.gif", writer="pillow", fps=5)
