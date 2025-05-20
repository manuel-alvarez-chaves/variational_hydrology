import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm


def calc_reliability_benchmark(data):
    basin_ids = [str(basin) for basin in data.coords["basin"].values]
    alphas = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    
    alpha_values = 1 - alphas
    lower_quantiles = alpha_values / 2
    upper_quantiles = 1 - alpha_values / 2

    results = {"alphas": alphas.tolist(), "basins": {}}
    
    for basin in tqdm(basin_ids, total=len(basin_ids), ascii=True):
        # Select data
        basin_data = data.sel(basin=basin)
        
        # Extract data (to numpy arrays)
        y_obs = basin_data["y_obs"].values
        y_hat = basin_data["y_hat"].values
        
        # Calculate all quantiles at once
        model_lowers = np.quantile(y_hat, lower_quantiles, axis=1)
        model_uppers = np.quantile(y_hat, upper_quantiles, axis=1)
        
        # For each alpha, determine how many observations fall within the interval
        basin_cdf = np.empty(len(alphas))
        for idx, _ in enumerate(alphas):
            model_is_between = (model_lowers[idx] <= y_obs) & (y_obs <= model_uppers[idx])
            basin_cdf[idx] = np.mean(model_is_between)

        results["basins"][basin] = basin_cdf.tolist()

    return results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")

    args = parser.parse_args()

    path_netcdf = Path(args.filepath)
    name = path_netcdf.name.split("_")[1]

    data = xr.open_dataset(path_netcdf)
    res = calc_reliability_benchmark(data)
    
    # Dump to json
    with open(path_netcdf.parent / f"reliability-{name}.json", "w") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    main()