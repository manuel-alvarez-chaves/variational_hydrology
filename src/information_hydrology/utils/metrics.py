import numpy as np


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
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)