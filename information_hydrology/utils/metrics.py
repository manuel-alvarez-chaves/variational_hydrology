from typing import Tuple

import numpy as np
import torch
from torch import nn


def loss_mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(y) & ~torch.isnan(y_hat)
    y_hat = y_hat[mask]
    y = y[mask]
    return nn.functional.mse_loss(y_hat, y)

def loss_kld(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def loss_nll(y_hat: torch.Tensor | Tuple[torch.Tensor], y: torch.Tensor, samples=True) -> torch.Tensor:
    if samples:
        mu = torch.mean(y_hat, dim=1)
        sigma = torch.std(y_hat, dim=1)
    else:
        mu, sigma = y_hat

    p = - 0.5 * ((y - mu) / sigma) ** 2
    p = torch.exp(p) / (sigma * np.sqrt(2 * np.pi))
    loss = - torch.log(torch.sum(p, dim=-1) + 1e-10)
    loss = torch.mean(loss)
    # loss = nn.GaussianNLLLoss()(mu, y, sigma)
    return loss