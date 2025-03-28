import json
import sys
from pathlib import Path

import xarray as xr
from information_hydrology.utils.metrics import calc_kde_loglik
from neuralhydrology.evaluation import metrics as calc_metrics
from tqdm import tqdm

path_metrics = Path("experiments/metrics.json")

# Path to .nc file
path_experiment = Path(sys.argv[1])
experiment_name = path_experiment.name.split(".")[0][4:]
model = experiment_name.split("-")[0]

if path_metrics.exists():
    with Path.open(path_metrics, "r") as f:
        metrics = json.load(f)
else:
    metrics = {}

tqdm.write(f"Calculating metrics for {experiment_name}")
metrics[experiment_name] = {}
ds = xr.open_dataset(path_experiment)
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
    else:
        metrics[experiment_name][basin]["LOGLIK"] = float("nan")

with Path.open(path_metrics, "w") as f:
    json.dump(metrics, f, indent=4)