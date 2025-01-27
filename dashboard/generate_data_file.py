from pathlib import Path

import numpy as np
import pandas as pd
import pyeto

# PyETo is not available in PyPI and must be installed from source

# Read CAMELS-US file
path_daymet = Path("../CAMELS-US/basin_mean_forcing/daymet")
path_data = path_daymet / "14" / "09081600_lump_cida_forcing_leap.txt"

# Choose period of analysis
selected_period = pd.date_range(start="1998-04-01", end="1999-10-01", freq="D")

# Process file
with path_data.open("r") as f:
    lat_rad = float(f.readline()) * (np.pi / 180)  # First line is latitude [in degrees]
data = pd.read_csv(path_data, skiprows=3, delimiter="\s+")

# Set datetime as index
datetimes = pd.to_datetime(
    {
        "year": data["Year"],
        "month": data["Mnth"],
        "day": data["Day"],
    }
)
data.set_index(datetimes, inplace=True)

# Extract selected period
data = data.loc[selected_period]

# Hargreaves-Simoni
doy = data.index.dayofyear


def calc_hargreaves(
    day: int,
    lat: float,
    t_min: float,
    t_max: float,
    t_mean: float,
) -> float:
    sol_dec = pyeto.sol_dec(day)
    sha = pyeto.sunset_hour_angle(lat, sol_dec)
    ird = pyeto.inv_rel_dist_earth_sun(day)
    et_rad = pyeto.et_rad(lat, sol_dec, sha, ird)
    eto = pyeto.hargreaves(t_min, t_max, t_mean, et_rad)
    return eto


eto = []
for idx, d in enumerate(doy):
    t_min = data.iloc[idx]["tmin(C)"]
    t_max = data.iloc[idx]["tmax(C)"]
    t_mean = (t_min + t_max) / 2
    eto.append(calc_hargreaves(d, lat_rad, t_min, t_max, t_mean))

data["eto(mm/day)"] = eto

# Clean up
columns_to_keep = ["prcp(mm/day)", "eto(mm/day)"]
data = data[columns_to_keep]

# Save to CSV
data.to_csv("dashboard/data.csv", index_label="date")
