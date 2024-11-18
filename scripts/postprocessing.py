import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from information_hydrology.modelzoo.cudalstm import CudaLSTM
from information_hydrology.modelzoo.lstmgmm import LSTMGMM
from information_hydrology.modelzoo.vlstm import VLSTM, ErrorMode, SamplingMode
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.utils.config import Config
from torch.utils.data import DataLoader
from tqdm import tqdm

path_run = Path(sys.argv[1])

# Set CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
with Path.open(path_run / "config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)

# Experiment name
idx_name = path_run.name.find("2025") if path_run.name.find("2024") == -1 else path_run.name.find("2024")
experiment_name = path_run.name[:idx_name - 1]


# Load model
model_name = config["model"]
match model_name:
    case "CudaLSTM":
        num_inputs = config["num_inputs"]
        num_hidden = config["num_hidden"]
        model = CudaLSTM(num_inputs, num_hidden)
        num_samples = 1
    case "vLSTM":
        num_inputs = config["num_inputs"]
        num_hidden = config["num_hidden"]
        num_samples = 1_000
        if config["error"] == "additive":
            model = VLSTM(num_inputs, num_hidden, error=ErrorMode.ADDITIVE)
        elif config["error"] == "proportional":
            model = VLSTM(num_inputs, num_hidden, error=ErrorMode.PROPORTIONAL)
    case "LSTM-GMM":
        num_inputs = config["num_inputs"]
        num_hidden = config["num_hidden"]
        num_gaussians = config["num_gaussians"]
        num_samples = 1_000
        model = LSTMGMM(num_inputs, num_hidden, num_gaussians)
    case _:    
        raise ValueError(f"Model {model_name} not recognized")

model.load_state_dict(torch.load(list(path_run.glob("*.pt"))[-1], map_location=device, weights_only=False))
model.to(device)
model.eval()

# Load data

# Dates
date_initial = config["test_start_date"]
date_final = config["test_end_date"]
dates = pd.date_range(date_initial, date_final, freq="D")

# Basins
path_test_basins = Path(config["test_basin_file"])
with path_test_basins.open("r") as f:
    basins = [basin.rstrip() for basin in f]

with Path.open(path_run / "train_data_scaler.yml", "r") as f:
    scaler_train = yaml.safe_load(f)

# Dataset
# The train dataset has to be loaded... NH bug?
config = Config(config, dev_mode=True)
ds_train = get_dataset(cfg=config, is_train=True, period="train")

out = {}
for basin in tqdm(basins, ascii=True):
    # Test dataset and laoder
    ds_test = get_dataset(cfg=config, is_train=False, period="test", basin=basin)
    loader = DataLoader(ds_test, batch_size=256, shuffle=False, collate_fn=ds_test.collate_fn)
    
    dates, y_obs, y_hat = [], [], []
    for sample in loader:
        x_d, y, date, x_s = sample.values()
        x_s = x_s.unsqueeze(1).repeat(1, x_d.shape[1], 1)
        x = torch.cat([x_d, x_s], dim=-1).to(device)

        match model_name:
            case "CudaLSTM":
                pred = model(x)
            case "vLSTM":
                pred = model.sample(x, num_samples, mode=SamplingMode.STANDARD, track_grad=False)
            case "LSTM-GMM":
                pred = model.sample(x, num_samples)
        
        dates.append(date[:, -1])
        y_obs.append(y[:, -1, 0].detach().clone().numpy())
        y_hat.append(pred.detach().cpu().clone().numpy())
        del x_d, y, date, x_s, x, pred
    
    if model_name == "vLSTM":
        y_hat = [array[:, :, 0] for array in y_hat]
        
    out[basin] = {
        "dates": np.concatenate(dates),
        "y_obs": np.concatenate(y_obs),
        "y_hat": np.concatenate(y_hat).T,
    }
    del dates, y_obs, y_hat

# Reshape results
dates = out[basins[0]]["dates"]
y_obs = np.stack(([out[basin]["y_obs"] for basin in basins]), axis=0)
y_hat = np.stack(([out[basin]["y_hat"] for basin in basins]), axis=0)

ds = xr.Dataset(
    data_vars={
        "y_obs": (("basin", "date"), y_obs),
        "y_hat": (("basin", "sample", "date"), y_hat),
    },
    coords={"date": dates, "basin": basins},
)

ds.to_netcdf(path_run.parent / f"res_{experiment_name}.nc")

