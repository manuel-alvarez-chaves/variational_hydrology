from pathlib import Path

import numpy as np
import pandas as pd
import pyeto

# PyETo is not available in PyPI and must be installed from source

# Read CAMELS-US file
path_daymet = Path("../CAMELS-US/basin_mean_forcing/daymet")
path_streamflow = Path("../CAMELS-US/usgs_streamflow")

path_forcing_data = path_daymet / "14/09081600_lump_cida_forcing_leap.txt"
path_qobs = path_streamflow / "14/09081600_streamflow_qc.txt"

# Choose period of analysis
selected_period = pd.date_range(start="1998-04-01", end="1999-10-01", freq="D")

# Process forcing
with path_forcing_data.open("r") as f:
    lat_rad = float(f.readline()) * (
        np.pi / 180
    )  # First line is latitude [in degrees]
    f.readline()  # Skip second line
    area = float(f.readline())  # Second line is area [in square meters]
data = pd.read_csv(path_forcing_data, skiprows=3, delimiter="\s+")

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

# Process streamflow
col_names = ["basin", "Year", "Mnth", "Day", "QObs", "flag"]
data_qobs = pd.read_csv(path_qobs, sep=r"\s+", header=None, names=col_names)
datetimes = pd.to_datetime(
    {
        "year": data_qobs["Year"],
        "month": data_qobs["Mnth"],
        "day": data_qobs["Day"],
    }
)
data_qobs = data_qobs.set_index(datetimes)
data_qobs = data_qobs.loc[selected_period]
qobs = 28316846.592 * data_qobs["QObs"].values * 86400 / (area * 10**6)

# Extract selected period
data["qobs(mm/day)"] = qobs

# Clean up
columns_to_keep = ["prcp(mm/day)", "eto(mm/day)", "qobs(mm/day)"]
data = data[columns_to_keep]

# Save to CSV
data.to_csv("dashboard/data.csv", index_label="date")
