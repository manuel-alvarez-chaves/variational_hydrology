import random

import numpy as np
import torch


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
