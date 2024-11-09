import time
from pathlib import Path

import numpy as np
import torch
import yaml
from information_hydrology.utils.miscellaneous import set_seed
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.utils.config import Config
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# # # # # # # # # # # # # # # PART 00 # # # # # # # # # # # # # ## # #

# General config
experiment_name = "LSTM_531"
path_save_folder = Path("experiments") / (experiment_name + time.strftime(r"_%Y-%m-%d_%H-%M-%S"))

# NeuralHydrology onfig file for data
path_config = Path("scripts/config_data.yml")
config = yaml.safe_load(Path.open(path_config, "r"))
config.update({"train_dir": path_save_folder})
config = Config(config)

# Dataset and Loader

# Training
ds_train = get_dataset(cfg=config, is_train=True, period="train")
dl_train = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True, collate_fn=ds_train.collate_fn)
print("Batches in training:", len(dl_train))

# Validation
ds_val = get_dataset(cfg=config, is_train=False, period="validation")
dl_val = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False, collate_fn=ds_val.collate_fn)
print("Batches in validation:", len(dl_val))

# Items
sample = next(iter(dl_train))

print("\nSample keys:")
for k, v in sample.items():
    print(f"{k}: {v.shape}")

x_d, y, date, x_s = sample.values()