import json
from pathlib import Path

import numpy as np
import xarray as xr
from information_hydrology.utils.metrics import calc_nse
from tqdm import tqdm

paths_to_results = [
    Path("experiments/531_Final-Runs-Epoch-10/res_LSTMCMAL-250-03_NLL-LAPLACE_S01_531.nc"),
    Path("experiments/531_Final-Runs-Epoch-10/res_LSTMCMAL-250-03_NLL-LAPLACE_S02_531.nc"),
    Path("experiments/531_Final-Runs-Epoch-10/res_LSTMCMAL-250-03_NLL-LAPLACE_S03_531.nc"),
    Path("experiments/531_Final-Runs-Epoch-10/res_LSTMCMAL-250-03_NLL-LAPLACE_S04_531.nc"),
    Path("experiments/531_Final-Runs-Epoch-10/res_LSTMCMAL-250-03_NLL-LAPLACE_S05_531.nc"),
]

def main():
    # Read basin_ids from the first file
    basin_ids = xr.open_dataset(paths_to_results[0])["basin"].values.tolist()
    num_time_steps = len(xr.open_dataset(paths_to_results[0])["date"])
    # Loop through each basin_id
    results = {}
    for basin in tqdm(basin_ids, ascii=True):
        # Get the predictions from each .nc file
        y_hat = np.empty((len(paths_to_results), num_time_steps))
        for idx, path in enumerate(paths_to_results):
            ds = xr.open_dataset(path)
            y_hat[idx, :] = ds["y_hat"].sel(basin=basin).values[:, 0]

        y_obs = ds["y_obs"].sel(basin=basin).values
        y_hat_median = np.median(y_hat, axis=0)

        nse = calc_nse(y_obs, y_hat_median)
        results[basin] = float(nse)
    
    # Save the results to a JSON file
    output_path = path.parent / "ensemble-nse-cmal.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()