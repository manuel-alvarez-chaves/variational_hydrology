from pathlib import Path

import numpy as np
import pandas as pd
import pyeto

# Read CAMELS-US file
path_forcing = Path("../CAMELS-US/basin_mean_forcing/daymet")
path_streamflow = Path("../CAMELS-US/usgs_streamflow")

path_forcing_data = path_forcing / "14/09081600_lump_cida_forcing_leap.txt"
path_qobs = path_streamflow / "14/09081600_streamflow_qc.txt"

# Choose period of analysis
selected_period = pd.date_range(start="1998-04-01", end="1999-10-01", freq="D")

# Process forcing
with path_forcing_data.open("r") as f:
    lat_rad = float(f.readline()) * (np.pi / 180)  # First line is latitude [in radians]
    f.readline() # Skip second line
    area = float(f.readline())  # Third line is area [in square meters]
col_names = ["Year", "Month", "Day", "Hr", "dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]
data = pd.read_csv(path_forcing_data, skiprows=4, delimiter="\s+", names=col_names)

# Set datetime as index
datetimes = pd.to_datetime(
    {
        "year": data["Year"],
        "month": data["Month"],
        "day": data["Day"],
    }
)
data.set_index(datetimes, inplace=True)

# ETo calculation
num_days = len(data)
eto = np.zeros(num_days)
doy = data.index.dayofyear
for idx in range(num_days):
    sol_dec = pyeto.sol_dec(doy[idx])
    sha = pyeto.sunset_hour_angle(lat_rad, sol_dec)
    ird = pyeto.inv_rel_dist_earth_sun(doy[idx])
    et_rad = pyeto.et_rad(lat_rad, sol_dec, sha, ird)
    tmin, tmax = data.iloc[idx]["tmin"], data.iloc[idx]["tmax"]
    tmean = (tmax + tmin) / 2
    eto[idx] = pyeto.hargreaves(tmin, tmax, tmean, et_rad)

data["eto"] = eto

# Shift data to better match streamflow 
data = data.shift(periods=45)

# Process streamflow
col_names = ["basin", "Year", "Month", "Day", "QObs", "flag"]
data_qobs = pd.read_csv(path_qobs, sep=r"\s+", header=None, names=col_names)
datetimes = pd.to_datetime(
    {
        "year": data_qobs["Year"],
        "month": data_qobs["Month"],
        "day": data_qobs["Day"],
    }
)
data_qobs = data_qobs.set_index(datetimes)
qobs = 28316846.592 * data_qobs["QObs"].values * 86400 / (area * 10**6)

# Extract selected period
data["qobs"] = qobs

# Clean up
columns_to_keep = ["prcp", "eto", "qobs"]
data = data[columns_to_keep]

# Select period
data = data.loc[selected_period]

# Save to CSV
data.to_csv("dashboard/data.csv", index_label="date")
