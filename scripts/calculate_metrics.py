import argparse
import json
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from hy2dl.datasetzoo.camelsus import CAMELS_US
from hy2dl.modelzoo.cudalstm import CudaLSTM
from information_hydrology.modelzoo.lstmgmm import LSTMGMM
from information_hydrology.modelzoo.vlstm import VLSTM, ErrorMode, SamplingMode
from information_hydrology.utils.metrics import calc_crps, calc_kde_loglik
from neuralhydrology.evaluation import metrics as calc_metrics
from torch.utils.data import DataLoader
from tqdm import tqdm


def generate_netcdf(path_to_model_dict: Path) -> Tuple[str, xr.Dataset]:
    # Set CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    path_run = path_to_model_dict.parent
    with Path.open(path_run / "config.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Experiment name
    experiment_name = config["experiment"]

    # Load model
    model_name = config["model"]["name"]

    if model_name not in ["LSTM"]:
        num_inputs = config["model"]["num_inputs"]
        num_hidden = config["model"]["num_hidden"]
        percent_dropout = config["model"]["percent_dropout"]
        num_dnn_layers = 1
    num_samples = 1_000
    match model_name:
        case "LSTM":
            model = CudaLSTM(config["model"])
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

    model.load_state_dict(torch.load(path_to_model_dict, map_location=device, weights_only=False))
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
        ds.calculate_basin_std()
        ds.standardize_data(standardize_output=False)

        loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=ds.collate_fn)
        
        dates, y_obs, y_hat = [], [], []
        for sample in loader:
            # Fix inputs
            x_d, x_s, y, _, _, date = sample.values()
            x_s = x_s.unsqueeze(1).repeat(1, x_d.shape[1], 1)
            x = torch.cat([x_d, x_s], dim=-1).to(device)
            y = y[:, -1, :].to(device)
            sample = {"x_d": x, "basin_std": sample["basin_std"].to(device)}

            match model_name:
                case "LSTM":
                    pred = model(sample)["y_hat"][:, 0] # [batch_size, num_targets]
                    pred = pred.unsqueeze(1) # [batch_size, num_samples, num_targets]
                    y_hat_sample = pred.detach().cpu().clone().numpy()[:, :, 0] # [batch_size, num_samples] 
                case "vLSTM":
                    pred = model.sample(x, num_samples, mode=SamplingMode.LEARNED, track_grad=False) # [batch_size, num_samples, num_targets]
                    y_hat_sample = pred.detach().cpu().clone().numpy()[:, :, 0] # [batch_size, num_samples]
                case "LSTMGMM":
                    mu, _, w = model.forward(x)
                    pred = model.sample(x, num_samples)
                    pred[:, 0] = (mu * w).sum(dim=1)
                    y_hat_sample = pred.detach().cpu().clone().numpy() # [batch_size, num_samples]
            
            dates.append(date) # list of num_batches
            y_obs.append(y.detach().cpu().clone().numpy()[:, 0]) # [batch_size, num_targets]
            y_hat.append(y_hat_sample)
            del x_d, x_s, y, date, x, sample, pred, y_hat_sample
        
        
        out[basin] = {
            "dates": np.concatenate(dates),
            "y_obs": np.concatenate(y_obs),
            "y_hat": np.concatenate(y_hat),
        }
        del dates, y_obs, y_hat
        if basin == "01047000":
            break

    # Reshape results
    dates = out[basins[0]]["dates"].flatten()
    y_obs = np.stack(([out[basin]["y_obs"] for basin in out.keys()]), axis=0)
    y_hat = np.stack(([out[basin]["y_hat"] for basin in out.keys()]), axis=0)

    ds = xr.Dataset(
        data_vars={
            "y_obs": (("basin", "date"), y_obs),
            "y_hat": (("basin", "date", "samples"), y_hat),
        },
        coords={"date": dates, "basin": list(out.keys())},
    )

    return experiment_name, ds


def generate_metrics(ds: xr.Dataset, name: str, path_metrics: Path) -> None:
    experiment_name = name
    model = experiment_name.split("-")[0]

    if path_metrics.exists():
        with Path.open(path_metrics, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    tqdm.write(f"Calculating metrics for {experiment_name}")
    metrics[experiment_name] = {}
    basin_ids = [str(basin) for basin in ds.coords["basin"].values]
    for basin in tqdm(basin_ids, ascii=True):
        # Read basin data
        data = ds.sel(basin=basin)

        # Metrics
        metrics[experiment_name][basin] = {}
        metrics[experiment_name][basin]["NSE"] = calc_metrics.nse(data.y_obs, data.y_hat.mean(dim="samples"))
        metrics[experiment_name][basin]["KGE"] = calc_metrics.kge(data.y_obs, data.y_hat.mean(dim="samples"))
        metrics[experiment_name][basin]["CORR"] = calc_metrics.pearsonr(data.y_obs, data.y_hat.mean(dim="samples"))
        metrics[experiment_name][basin]["a_NSE"] = calc_metrics.alpha_nse(data.y_obs, data.y_hat.mean(dim="samples"))
        metrics[experiment_name][basin]["b_NSE"] = calc_metrics.beta_nse(data.y_obs, data.y_hat.mean(dim="samples"))
        metrics[experiment_name][basin]["FHV"] = float(calc_metrics.fdc_fhv(data.y_obs, data.y_hat.mean(dim="samples")))
        metrics[experiment_name][basin]["FLV"] = float(calc_metrics.fdc_flv(data.y_obs, data.y_hat.mean(dim="samples")))
        metrics[experiment_name][basin]["FMS"] = float(calc_metrics.fdc_fms(data.y_obs, data.y_hat.mean(dim="samples")))
        if model not in ["LSTM"]:
            metrics[experiment_name][basin]["LOGLIK"] = calc_kde_loglik(data.y_obs.values, data.y_hat.values)
            metrics[experiment_name][basin]["CRPS"] = calc_crps(data.y_obs.values, data.y_hat.values)
        else:
            metrics[experiment_name][basin]["LOGLIK"] = float("nan")
            metrics[experiment_name][basin]["CRPS"] = float("nan")
        
    with Path.open(path_metrics, "w") as f:
        json.dump(metrics, f, indent=4)

    return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    parser.add_argument("-s", "--save", action="store_true")
    args = parser.parse_args()

    path_experiment = Path(args.filepath)
    save_netcdf = args.save

    # Postprocess experiment
    match path_experiment.suffix:
        case ".pt":
            name, results = generate_netcdf(path_experiment)
            path_metrics = path_experiment.parent.parent / "metrics.json"

            if save_netcdf:
                path_experiment = path_experiment.parent.parent / f"res_{name}.nc"
                results.to_netcdf(path_experiment)
        case ".nc":
            name = path_experiment.name.split(".")[0][4:]
            results = xr.open_dataset(path_experiment)
            path_metrics = path_experiment.parent / "metrics.json"
        case _:
            raise ValueError(f"File must be .pt or .nc, not {path_experiment.suffix}")

    # Calculate metrics
    generate_metrics(results, name, path_metrics)

if __name__ == "__main__":
    main()