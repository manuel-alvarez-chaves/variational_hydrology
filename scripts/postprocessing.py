import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from hy2dl.datasetzoo.camelsus import CAMELS_US
from information_hydrology.modelzoo.lstmgmm import LSTMGMM
from information_hydrology.modelzoo.vlstm import VLSTM, ErrorMode, SamplingMode
from torch.utils.data import DataLoader
from tqdm import tqdm

path_model_dict = Path(sys.argv[1])
path_run = path_model_dict.parent

# Set CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
with Path.open(path_run / "config.yml", "r") as f:
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
    case "LSTM-GMM":
        num_gaussians = config["model"]["num_gaussians"]
        model = LSTMGMM(num_inputs, num_hidden, num_gaussians, percent_dropout)
    case _:    
        raise ValueError(f"Model {model_name} not recognized")

model.load_state_dict(torch.load(path_model_dict, map_location=device, weights_only=False))
model.to(device)
model.eval()

# Load data

# Dates
date_initial, date_final = config["data"]["test_period"]
dates = pd.date_range(date_initial, date_final, freq="D")

# Basins
path_test_basins = Path(config["data"]["test_basin_file"])
with path_test_basins.open("r") as f:
    basins = [basin.rstrip() for basin in f]

# Scaler
with Path.open(path_run / "scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

out = {}
for basin in tqdm(basins, ascii=True):
    # One dataset per basin
    ds = CAMELS_US(
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
    ds.scaler = scaler
    ds.standardize_data(standardize_output=False)

    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=ds.collate_fn)
    
    dates, y_obs, y_hat = [], [], []
    for sample in loader:
        # Fix inputs
        x_d, x_s, y, _, date = sample.values()
        x_s = x_s.unsqueeze(1).repeat(1, x_d.shape[1], 1)
        x = torch.cat([x_d, x_s], dim=-1).to(device)
        y = y[:, -1, :].to(device)

        match model_name:
            case "vLSTM":
                pred = model.sample(x, num_samples, mode=SamplingMode.LEARNED, track_grad=False)
                y_hat_sample = pred.detach().cpu().clone().numpy()[:, :, 0] # [batch_size, num_samples, num_targets]
            case "LSTM-GMM":
                mu, _, w = model.forward(x)
                pred = model.sample(x, num_samples)
                pred[:, 0] = (mu * w).sum(dim=1)
                y_hat_sample = pred.detach().cpu().clone().numpy() # [batch_size, num_samples]
        
        dates.append(date) # list of num_batches
        y_obs.append(y.detach().cpu().clone().numpy()[:, 0]) # [batch_size, num_targets]
        y_hat.append(y_hat_sample)
        del x_d, x_s, y, date, x, pred
    
      
    out[basin] = {
        "dates": np.concatenate(dates),
        "y_obs": np.concatenate(y_obs),
        "y_hat": np.concatenate(y_hat),
    }
    del dates, y_obs, y_hat

# Reshape results
dates = out[basins[0]]["dates"].flatten()
y_obs = np.stack(([out[basin]["y_obs"] for basin in basins]), axis=0)
y_hat = np.stack(([out[basin]["y_hat"] for basin in basins]), axis=0)

ds = xr.Dataset(
    data_vars={
        "y_obs": (("basin", "date"), y_obs),
        "y_hat": (("basin", "date", "samples"), y_hat),
    },
    coords={"date": dates, "basin": basins},
)

ds.to_netcdf(path_run.parent / f"res_{experiment_name}.nc")