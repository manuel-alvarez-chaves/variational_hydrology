import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seconds_to_time(elapsed_time: float) -> str:
    """Format seconds to time as %H:%M:%S.

    Parameters
    ----------
    elapsed_time : float
        Time in seconds.

    """
    m, s = divmod(elapsed_time, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def set_seed(seed: int | None) -> int:
    """Set random seed.

    Parameters
    ----------
    seed : int or None
        Random seed.

    Returns
    -------
    int
        Random seed.

    """
    if seed is None:
        seed = np.random.randint(1, 10_000)

    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    return seed

def dump_config(config: dict, path: str) -> None:
    config["train_dir"] = str(config["train_dir"].as_posix())
    with Path.open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)