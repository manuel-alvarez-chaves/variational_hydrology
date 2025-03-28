from typing import Union

import numpy as np
from scipy import stats
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
    auc = np.trapezoid(y, x)

    return {"x": x, "y": y, "median": median, "auc": auc}

def calc_nse(obs: np.array, sim: np.array):
    obs, sim = _mask(obs, sim)
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

def calc_kde_loglik(obs: np.array, sim: np.array) -> np.float64:
    obs, sim = _mask(obs, sim)
    n = len(obs)
    loglik = np.empty(n)
    for idx in range(n):
        p = calc_kde_density(obs[idx].reshape(-1, 1), sim[idx, :].reshape(-1, 1))
        loglik[idx] = float(np.log(p + 1e-10)[0, 0])
    if n == 1:
        return loglik[0]
    else:
        return np.mean(loglik)

def calc_winkler(obs: np.array, sim: np.array, level=0.90) -> Union[np.float64, np.array]:
    sim = sim.flatten()
    alpha = 1 - level
    l = np.quantile(sim, alpha/2)
    u = np.quantile(sim, 1-alpha/2)

    if obs < l:
        winkler = (u - l) + 2/alpha * (l - obs)
    if l <= obs <= u:
        winkler = u - l
    if obs > u:
        winkler = (u - l) + 2/alpha * (obs - u)
    return float(winkler)
    
def heaviside(x: float) -> float:
    return 1 * (x >= 0)     

def calc_crps(obs: np.array, sim: np.array) -> np.float64:
    obs, sim = _mask(obs, sim)
    n = len(obs)
    crps = np.empty(n)
    for idx in range(n):
        ecdf = stats.ecdf(np.hstack([sim[idx, :], obs[idx]]))
        score = (ecdf.cdf.probabilities - heaviside(ecdf.cdf.quantiles - obs[idx]))**2
        crps[idx] = np.trapezoid(score, ecdf.cdf.quantiles)
    if n == 1:
        return crps[0]
    else:
        return np.mean(crps)