from typing import Tuple

import numpy as np
import torch
from torch import nn


def loss_mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = ~(y.isnan() | y_hat.isnan())
    y_hat = y_hat[mask]
    y = y[mask]
    return nn.functional.mse_loss(y_hat, y)

def loss_kld(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def loss_nll(y_hat: torch.Tensor | Tuple[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    if type(y_hat) is tuple:
        mu, sigma, w = y_hat
        mask = ~(y.isnan() | mu.sum(-1).reshape(-1, 1).isnan()).flatten()
        mu, sigma, w = mu[mask], sigma[mask], w[mask]
    else:
        mask = ~(y.isnan() | y_hat.sum(dim=1).isnan()).flatten()
        y_hat = y_hat[mask, ...]
        mu, sigma = y_hat.mean(dim=1), y_hat.std(dim=1)
        w = torch.ones_like(mu)

    y = y[mask, :]
    p = - 0.5 * ((y - mu) / sigma) ** 2
    p = torch.exp(p) / (sigma * np.sqrt(2 * np.pi))
    p = (p * w).sum(dim=-1)
    
    loss = - torch.log(p + 1e-10)
    loss = torch.mean(loss)
    return loss

def calc_kernel(x, y, sigma=1.0):
    # Gaussian kernel
    dist = torch.cdist(x, y, p=2)
    kernel = torch.exp((-1 * (dist**2)) / (2 * sigma**2))
    return kernel

def loss_mmd(x, y):
    x_kernel = calc_kernel(x, x)
    y_kernel = calc_kernel(y, y)
    xy_kernel = calc_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)