from typing import Tuple

import numpy as np
import torch
from torch import nn
from torchkde import KernelDensity

from information_hydrology.utils.distributions import Distribution


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
        y_hat = (mu, sigma, None), w
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
    y = _mask(y)[0]

    y_hat = _calc_moments(y_hat)
    moments, w = y_hat
    
    match dist:
        case Distribution.GAUSSIAN:
            loc, scale, _ = moments
            scale = torch.clamp(scale, min=1e-6)
            p = (y - loc) / scale
            log_p = -0.5 * p.pow(2) - torch.log(scale) - 0.5 * np.log(2 * np.pi)
            
        case Distribution.LAPLACE:
            loc, scale, kappa = moments
            scale = torch.clamp(scale, min=1e-6)
            kappa = torch.clamp(kappa, min=1e-6)
            
            p = (y - loc) / scale
            
            mask = (p >= 0)
            
            log_p = torch.zeros_like(p)
            
            log_p[mask] = -1 * p[mask] * kappa[mask]
            log_p[~mask] = p[~mask] / kappa[~mask]
            
            log_p = log_p - torch.log(kappa + 1 / kappa) - torch.log(scale)
    
    log_w = torch.log(torch.clamp(w, min=1e-10))
    loss = -torch.logsumexp(log_p + log_w, dim=1)
    return loss.mean()

def loss_kld(logvar: torch.Tensor) -> torch.Tensor:
    """
    Calculates de Kullback-Leibler divergence (KLD) loss.

    Calculates the KLD loss between a predicted distribution with mean mu and
    covariance logvar and a standard multivariate normal distribution.

    Parameters
    ----------
    logvar : torch.Tensor
        Predicted covariance.

    Returns
    -------
    torch.Tensor
        The computed KLD loss.
    """
    logvar = _mask(logvar)[0]
    return -0.5 * torch.sum(1 + logvar - logvar.exp())

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

def loss_nll_kde(y_hat, y):
    """Calculates the negative log-likelihood loss using kernel density
    estimation (KDE).

    For each batch in the sample, creates a KDE distribution using the
    predicted points *y_hat* to evaluate the density of *y*. This density is
    used to approximate the negative log-likelihood loss.

    Parameters
    ----------
    y_hat : torch.Tensor (num_batches x num_samples x num_dim)
        The predicted data points.
    y : torch.Tensor (num_batches x num_dim)
        The observed point.

    Returns
    -------
    torch.Tensor
        The computed negative log-likelihood loss.
    """
    y_hat, y = _mask(y_hat, y)
    num_batches, _, _ = y_hat.shape

    logprob = []
    for idx in range(num_batches):
        kde = KernelDensity(bandwidth="silverman", kernel="gaussian")
        _ = kde.fit(y_hat[idx])
        logprob.append(kde.score_samples(y[idx]))
    logprob = torch.stack(logprob)
    loss = -logprob.mean()
    return loss
