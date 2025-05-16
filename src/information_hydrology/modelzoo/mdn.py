import torch
import torch.nn.functional as F
from torch import nn

from information_hydrology.utils.distributions import Distribution


class LSTMMDN(nn.Module):
    def __init__(self, input_size, hidden_size, distribution, num_components, output_dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.distribution = distribution
        self.num_components = num_components
        self.output_dropout = output_dropout

        match distribution:
            case Distribution.GAUSSIAN:
                self.num_moments = 2
            case Distribution.LAPLACE:
                self.num_moments = 3

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_moments = nn.Linear(hidden_size, self.num_moments * num_components)
        self.fc_weights = nn.Sequential(
            nn.Linear(hidden_size, num_components),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(output_dropout)

        self._reset_parameters()
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1] # many-to-one
        out = self.dropout(out)
        w = self.fc_weights(out)
        out = self.fc_moments(out)
        match self.distribution:
            case Distribution.GAUSSIAN:
                loc, scale = out.chunk(2, dim=-1)
                scale = F.softplus(scale)
                moments = (loc, scale, None)
            case Distribution.LAPLACE:
                loc, scale, kappa = out.chunk(3, dim=-1)
                scale = F.softplus(scale) 
                kappa = F.softplus(kappa)
                moments = (loc, scale, kappa)
        return moments, w
    
    def mean(self, x):
        with torch.no_grad():
            moments, w = self(x)
            match self.distribution:
                case Distribution.GAUSSIAN:
                    loc, scale, _ = moments
                    mean = loc
                case Distribution.LAPLACE:
                    loc, scale, kappa = moments
                    mean = loc + scale * (1 - kappa.pow(2)) / kappa
            mean = (mean * w).sum(axis=1)
        return mean
    
    def sample(self, x, num_samples):
        with torch.no_grad():
            moments, w = self(x)
            num_batches, num_components = moments[0].shape
            match self.distribution:
                case Distribution.GAUSSIAN:
                    loc, scale, _ = moments
                    samples = torch.randn(num_batches, num_components, num_samples).to(x.device)
                case Distribution.LAPLACE:
                    loc, scale, kappa = moments
                    u = torch.rand(num_batches, num_components, num_samples).to(x.device)
                    # Sampling left or right of the mode?
                    kappa = kappa.unsqueeze(-1).repeat((1, 1, num_samples))  # [num_batches, num_components, num_samples]
                    p_at_mode = kappa**2 / (1 + kappa**2) # [num_batches, num_components]

                    mask = u < p_at_mode  # [num_batches, num_components, num_samples]

                    samples = torch.zeros_like(u)  # [num_batches, num_components, num_samples]

                    samples[mask] = kappa[mask] * torch.log(u[mask] * (1 + kappa[mask].pow(2)) / kappa[mask].pow(2)) # Left side
                    samples[~mask] = -1 * torch.log((1 - u[~mask]) * (1 + kappa[~mask].pow(2))) / kappa[~mask] # Right side

            # Rescale the samples
            samples = samples * scale.unsqueeze(-1) + loc.unsqueeze(-1)  # [num_batches, num_components, num_samples]

            # Select samples according to weights
            indices = torch.multinomial(w, num_samples, replacement=True) # [num_batches, num_samples]
            indices = indices.unsqueeze(1)  # [num_batches, 1, num_samples]
                
            # Now gather
            result = torch.gather(samples, dim=1, index=indices)  # Shape: [num_batches, 1, num_samples]
            result = result.squeeze(1)  # [num_batches, num_samples]
            
        return result
    
    def _reset_parameters(self):
        self.lstm.bias_hh_l0.data[self.hidden_size:2 * self.hidden_size] = 3.0