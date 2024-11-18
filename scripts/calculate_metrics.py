import json
from pathlib import Path

import xarray as xr
from neuralhydrology.evaluation import metrics as calc_metrics
from tqdm import tqdm

path_experiments = Path("experiments")
path_metrics = Path("experiments/metrics.json")

experiments = list(path_experiments.glob("*.nc"))

recalculate = True
if path_metrics.exists() and not recalculate:
    with Path.open(path_metrics, "r") as f:
        metrics = json.load(f)
else:
    metrics = {}
    for experiment in experiments:
        experiment_name = experiment.name.split(".")[0][4:]
        metrics[experiment_name] = {}

for experiment in experiments:
    experiment_name = experiment.name.split(".")[0][4:]
    tqdm.write(f"Calculating metrics for {experiment_name}")
    ds = xr.open_dataset(experiment)
    basin_ids = [str(basin) for basin in ds.coords['basin'].values]
    if recalculate or metrics[experiment_name] == {}:
        for basin in tqdm(basin_ids, ascii=True):
            data = ds.sel(basin=basin)

            metrics[experiment_name][basin] = {}
            metrics[experiment_name][basin]["NSE"] = calc_metrics.nse(data.y_obs, data.y_hat.mean(dim='sample'))
            metrics[experiment_name][basin]["KGE"] = calc_metrics.kge(data.y_obs, data.y_hat.mean(dim='sample'))
            metrics[experiment_name][basin]["KGE"] = calc_metrics.kge(data.y_obs, data.y_hat.mean(dim='sample'))
            metrics[experiment_name][basin]["CORR"] = calc_metrics.pearsonr(data.y_obs, data.y_hat.mean(dim='sample'))
            metrics[experiment_name][basin]["a_NSE"] = calc_metrics.alpha_nse(data.y_obs, data.y_hat.mean(dim='sample'))
            metrics[experiment_name][basin]["b_NSE"] = calc_metrics.beta_nse(data.y_obs, data.y_hat.mean(dim='sample'))
            # metrics[experiment_name][basin]["FHV"] = calc_metrics.fdc_fhv(data.y_obs, data.y_hat.mean(dim='sample'))
            # metrics[experiment_name][basin]["FLV"] = calc_metrics.fdc_flv(data.y_obs, data.y_hat.mean(dim='sample'))
            # metrics[experiment_name][basin]["FMS"] = calc_metrics.fdc_fms(data.y_obs, data.y_hat.mean(dim='sample'))

with Path.open(path_metrics, "w") as f:
    json.dump(metrics, f, indent=4)