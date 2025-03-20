import numpy as np
from unite_toolbox.kde_estimators import calc_kde_density


def _mask(*arrays: np.array) -> np.array:
    masks = []
    for array in arrays:
        num_dim = array.ndim
        for _ in range(num_dim - 1):
            array = array.sum(axis=1)
        mask = ~np.isnan(array)
        masks.append(mask)
    mask = np.stack(masks, axis=1).all(axis=1)
    return tuple(array[mask] for array in arrays)

def calc_cdf(metric: list):
    metric = np.array(metric)
    metric = metric[~np.isnan(metric)]

    x = np.sort(metric)
    y = np.arange(1, len(metric) + 1) / len(metric)
    median = np.median(x)
    idx_nonzero = sum(x < 0)
    auc = np.trapezoid(y[idx_nonzero:], x[idx_nonzero:])
    # auc = np.trapezoid(y, x)

    return {"x": x, "y": y, "median": median, "auc": auc}

def calc_nse(obs: np.array, sim: np.array):
    obs, sim = _mask(obs, sim)
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

def calc_kde_loglik(obs, sim):
    obs, sim = _mask(obs, sim)
    n = len(obs)
    loglik = np.empty(n)
    for idx in range(n):
        p = calc_kde_density(obs[idx].reshape(-1, 1), sim[idx, :].reshape(-1, 1))
        loglik[idx] = float(np.log(p + 1e-10)[0, 0])
    return np.mean(loglik)
