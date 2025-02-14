from enum import Enum
from typing import Tuple

import numpy as np
import torch
from scipy.special import gamma
from torch import nn
from torchkde import KernelDensity


def _calc_moments(y_hat: torch.Tensor | Tuple[torch.Tensor]):
    """Calculates the moments from an array of samples.

    Models can predict either the moments of a distribution directly or
    generate samples from the distribution. The moments are typically passed as
    a tuple. This function calculates the moments if the input is a tensor of
    samples. If a mixture density distribution is used, calculating the moments
    from the samples assumes that the weights are uniform.

    Parameters
    ----------
    y_hat : torch.Tensor | Tuple[torch.Tensor]
        Tensor of samples or tuple of moments.

    Returns
    -------
    Tuple[torch.Tensor]
        Tuple of moments (and weights).
    """
    if type(y_hat) is not tuple:
        mu, sigma = y_hat.mean(dim=1), y_hat.std(dim=1)
        w = torch.ones_like(mu)
        y_hat = (mu, sigma, w)
    return y_hat


def _mask(*tensors: torch.Tensor) -> Tuple[torch.Tensor]:
    """Masks not-a-number values in the tensors.

    Masks NaNs in the input tensors by finding indices where a specific tensor
    has a NaN value and then masking all tensors at the same index. The index
    masked is the first dimension of the tensor, typically the "batch"
    dimension.

    Parameters
    ----------
    tensors : torch.Tensor
        Tensors to be masked, allows for multiple inputs.

    Returns
    -------
    Tuple[torch.Tensor]
        Tensors with NaN values removed.
    """
    masks = []
    for tensor in tensors:
        num_dim = tensor.dim()
        for _ in range(num_dim - 1):
            tensor = tensor.sum(dim=1)
        mask = ~tensor.isnan()
        masks.append(mask)
    mask = torch.stack(masks, dim=1).all(dim=1)

    return tuple(tensor[mask] for tensor in tensors)

class Distribution(Enum):
    "Enum for the distribution to use in the NLL loss function."
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"

def loss_nll(
    y_hat: torch.Tensor | Tuple[torch.Tensor],
    y: torch.Tensor,
    dist: Distribution = Distribution.GAUSSIAN,
) -> torch.Tensor:
    """
    Computes the negative log-likelihood loss for the given predictions and
    targets.

    Parameters
    ----------
    y_hat : torch.Tensor or Tuple[torch.Tensor]
        Predicted values. If a tuple, it should contain moments and weights.
    y : torch.Tensor
        Ground truth values.
    dist : Distribution, optional
        The distribution to use for the loss calculation. Defaults to
        a Distribution.GAUSSIAN.

    Returns
    -------
    torch.Tensor
        The computed negative log-likelihood loss.
    """
    y_hat = _calc_moments(y_hat)
    mu, sigma, w = y_hat
    mu, sigma, w, y = _mask(mu, sigma, w, y)

    match dist:
        case Distribution.GAUSSIAN:
            p = (y - mu) / sigma
            p = -0.5 * p.pow(2)
            p = p.exp() / (sigma * np.sqrt(2 * np.pi))
        case Distribution.EXPONENTIAL:
            lmbda = 1 / sigma
            p = lmbda * (y - mu).abs()
            p = lmbda * p.exp()
            # p = torch.where(p >= lmbda, torch.zeros_like(p), p)
    
    p = (p * w).sum(dim=-1)
    loss = - torch.log(p + 1e-10)
    loss = torch.mean(loss)
    return loss

def loss_kld(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Calculates de Kullback-Leibler divergence (KLD) loss.

    Calculates the KLD loss between a predicted distribution with mean mu and
    covariance logvar and a standard multivariate normal distribution.

    Parameters
    ----------
    mu : torch.Tensor
        Predicted mean.
    logvar : torch.Tensor
        Predicted covariance.

    Returns
    -------
    torch.Tensor
        The computed KLD loss.
    """
    mu, logvar = _mask(mu, logvar)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def loss_mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean squared error (MSE) loss.

    Parameters
    ----------
    y_hat : torch.Tensor
        Predicted values.
    y : torch.Tensor
        Ground truth values.
    
    Returns
    -------
    torch.Tensor
        The computed MSE loss.
    """
    y_hat, y = _mask(y_hat, y)
    return nn.functional.mse_loss(y_hat, y)

def vol_lp_unit_ball(d: int, p: float):
    """Calculate the volume of a d-dimensional unit ball.
    
    Parameters
    ----------
    d : int
        Dimension of the unit ball
    p : float
        Norm of the unit ball
    """
    a = (2 * gamma(1 / p + 1)) ** d
    b = gamma(d / p + 1)
    return a / b

def loss_nll_knn(y_hat: torch.Tensor, y: torch.Tensor, k: int = 5, p_norm: float = 2):
    """Calculates the negative log-likelihood loss using k-nearest neighbors
    (*k*-NN).

    Calculates the negative log-likelihood loss using k-NN to approximate the
    likelihood of the query point *y* given the data points *y_hat*. This
    approximation is based on distance and was proposed by Wang et al. (2019).
    10.1109/TIT.2009.2016060

    Parameters
    ----------
    y_hat : torch.Tensor (num_batches x num_samples x num_dim)
        The data points.
    y : torch.Tensor (num_batches x num_dim)
        The query point.
    k : int
        Number of neighbors to calculate.
    p_norm : float
        Norm to calculate distance (p_norm = 1, 2, ..., np.inf)

    Returns
    -------
    torch.Tensor
        The computed negative log-likelihood loss.
    """
    y_hat, y = _mask(y_hat, y)
    _, num_samples, num_dim = y_hat.shape
    
    vol = vol_lp_unit_ball(num_dim, p_norm)

    # Calculate distances between x and all y_hat points, pick the k smallest
    y = y.unsqueeze(1).expand(-1, num_samples, -1)
    dist = (y_hat - y).norm(p=p_norm, dim=2)
    radius = dist.topk(k, dim=1, largest=False).values[:, -1]

    # Calculate density
    p = (k / (num_samples - 1)) * (1 / (vol * radius))
    loss = -torch.log(p + 1e-10).mean()
    return loss

def loss_nll_kde(y_hat, y):
    # y_hat [num_batches, num_samples, num_dim]
    # y [num_batches, num_dim]
    y_hat, y = _mask(y_hat, y)
    num_batches, num_samples, num_dim = y_hat.shape

    logprob = []
    for idx in range(num_batches):
        kde = KernelDensity(bandwidth="silverman", kernel="gaussian")
        _ = kde.fit(y_hat[idx])
        logprob.append(kde.score_samples(y[idx]))
    logprob = torch.stack(logprob)
    loss = -logprob.mean()
    return loss