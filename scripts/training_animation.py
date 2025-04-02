import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch
import yaml
from hy2dl.datasetzoo.camelsus import CAMELS_US
from information_hydrology.modelzoo.lstmgmm import LSTMGMM
from information_hydrology.modelzoo.vlstm import VLSTM, ErrorMode
from information_hydrology.utils.metrics import (
    calc_crps,
    calc_kde_loglik,
    calc_nse,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "savefig.bbox": "standard",
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "black",
    "legend.framealpha": 1.0,
    "legend.borderpad": 0.4,
    "patch.linewidth": 0.3,
})

phi = (1 + np.sqrt(5)) / 2
columns = {
    "single": (3.33 * phi, 3.33),
    "double": (6.96 * phi, 6.96 / 2),
}

colors = {
    "LSTM-256": '#0C5DA5',
    "LSTMGMM-250-10": '#FF9500',
    "VLSTM-250-PRO": '#FF2C00',
    "VLSTM-250-DENSE": '#00B945',
}

labels = {
    "LSTM-256": "LSTM",
    "LSTMGMM-250-10": r"LSTM $\to$ GMM",
    "VLSTM-250-PRO": "vLSTM (proportional)",
    "VLSTM-250-DENSE": "vLSTM (dense)",
}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

path_experiment = Path(sys.argv[1])

path_animation = path_experiment / "animation"
path_animation.mkdir(exist_ok=True)

# Load config
with Path.open(path_experiment / "config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)

# Experiment name
experiment_name = config["experiment"]
name = experiment_name.split("_")[0]

# Load model
model_name = config["model"]["name"]

num_inputs = config["model"]["num_inputs"]
num_hidden = config["model"]["num_hidden"]
percent_dropout = config["model"]["percent_dropout"]
num_dnn_layers = 1
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
                num_dnn_layers = config["model"]["num_layers"]
        model = VLSTM(num_inputs, num_hidden, percent_dropout, error_mode, num_layers=num_dnn_layers)
    case "LSTMGMM":
        num_gaussians = config["model"]["num_gaussians"]
        model = LSTMGMM(num_inputs, num_hidden, num_gaussians, percent_dropout)
    case _:    
        raise ValueError(f"Model {model_name} not recognized")

# Load model weights and biases
path_models = sorted(path_experiment.glob("*.pt"))

# Scaler
with Path.open(path_experiment / "scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

basin = "13011900"

ds_train = CAMELS_US(
    dynamic_input=config["data"]["dynamic_inputs"],
    forcing=config["data"]["forcings"],
    target=config["data"]["target_variables"],
    sequence_length=config["data"]["sequence_length"],
    time_period=config["data"]["train_period"],
    path_data=config["data"]["data_dir"],
    entity=basin,
    check_NaN=False,
    static_input=config["data"]["static_attributes"],
)
ds_train.scaler = scaler
ds_train.standardize_data(standardize_output=False)

indices = list(range(2336, 2592))
subset = Subset(ds_train, indices)
subset_train = DataLoader(subset, batch_size=len(subset), collate_fn=ds_train.collate_fn)
batch_train = next(iter(subset_train))

ds_test = CAMELS_US(
    dynamic_input=config["data"]["dynamic_inputs"],
    forcing=config["data"]["forcings"],
    target=config["data"]["target_variables"],
    sequence_length=config["data"]["sequence_length"],
    time_period=config["data"]["test_period"],
    path_data=config["data"]["data_dir"],
    entity=basin,
    check_NaN=False,
    static_input=config["data"]["static_attributes"],
)
ds_test.scaler = scaler
ds_test.standardize_data(standardize_output=False)

indices = list(range(512, 768))
subset = Subset(ds_test, indices)
subset_test = DataLoader(subset, batch_size=len(subset), collate_fn=ds_test.collate_fn)
batch_test = next(iter(subset_test))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

for epoch, params in tqdm(enumerate(path_models), total=len(path_models), ascii=True):

    fig = plt.figure(figsize=(6.96 * phi, 6.96 / 1.5))
    axes = []
    axes.append(fig.add_axes([0.04, 0.09, 0.47, 0.78]))
    axes.append(fig.add_axes([0.52, 0.09, 0.47, 0.78]))

    model.load_state_dict(torch.load(params, map_location=torch.device("cpu")))
    for ax, batch in zip(axes, [batch_train, batch_test]):
        x_d, x_s, y_obs, basin, dates = batch.values()
        dates = dates.flatten()
        y_obs = y_obs.flatten().numpy()
        x_s = x_s.unsqueeze(1).repeat(1, x_d.shape[1], 1)
        x = torch.cat([x_d, x_s], dim=-1)
        if name.split("-")[0] == "VLSTM":
            y_hat = model.sample(x, 1000).numpy()[:, :, 0]
        else:
            y_hat = model.sample(x).numpy()
        median = np.median(y_hat, axis=1).flatten()
        lower = np.quantile(y_hat, 0.025, axis=1).flatten() - y_obs
        upper = np.quantile(y_hat, 0.975, axis=1).flatten() - y_obs
        nse = calc_nse(y_obs.reshape(-1, 1), median.reshape(-1, 1))
        loglik = calc_kde_loglik(y_obs, y_hat)
        crps = calc_crps(y_obs, y_hat)
        ax.plot(dates, np.zeros_like(y_obs))
        ax.plot(dates, median - y_obs, color=colors[name])
        ax.fill_between(dates, lower, upper, color=colors[name], alpha=0.20)
        ax.set_xlabel("Date")
        ax.set_title(f"NSE = {nse:.3f} | Log-likelihood = {loglik:>6.3f} | CRPS = {crps:.3f}")
        ax.grid(ls="--", alpha=0.6)
        ax.set_ylim(-5.5, 5.5)

    axes[0].set_ylabel("Streamflow [mm/day]")
    axes[1].set_yticklabels([])
    fig.suptitle(f"Model: {labels[name]} | Epoch: {(epoch + 1):02d}")
    fig.text(0.44, 0.81, "Training", fontsize=12)
    fig.text(0.93, 0.81, "Testing", fontsize=12)
    plt.savefig(path_animation / f"epoch_{(epoch + 1):02d}.png", dpi=300)
    plt.close()
